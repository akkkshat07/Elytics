import logging
from typing import Dict, Any
from ..utils.bedrock import BedrockClient
from ..state import QueryState
logger = logging.getLogger(__name__)
PYTHON_AGENT_SYSTEM_PROMPT = 'You are an expert Python Data Analyst. Your task is to write\nPython code using Pandas, NumPy, and Plotly that analyzes data and answers a business question.\n\nThe code will be executed in a sandbox environment with these pre-loaded variables:\n- `query_results`: List[Dict] — raw data rows from Amazon Redshift (already loaded)\n- `pd`: pandas module\n- `np`: numpy module\n- `px`: plotly.express module\n- `go`: plotly.graph_objects module\n- `charts`: List — append Plotly chart dicts here: charts.append(fig.to_dict())\n- `text_outputs`: List — append string findings here: text_outputs.append("...")\n- `statistics`: Dict — store computed metrics here: statistics["key"] = value\n\nRULES:\n1. Start with: df = pd.DataFrame(query_results)\n2. Handle edge cases: check if df is empty before analysis\n3. Every Plotly figure MUST be appended to `charts`: charts.append(fig.to_dict())\n4. Store all key numeric findings in `statistics` dict for the Insights agent\n5. Append text summaries or findings to `text_outputs`\n6. Use plotly.express (px) for most charts — it is simpler and cleaner\n7. NEVER use: open(), import os, import sys, import subprocess, exec(), eval()\n8. Apply professional chart styling: template="plotly_white", descriptive titles\n\nRespond with ONLY valid JSON:\n{\n  "python_code": "your complete Python code as a single string",\n  "code_explanation": "Plain English explanation of what the code does step by step",\n  "expected_outputs": {\n    "charts": ["description of chart 1", "description of chart 2"],\n    "statistics": ["stat1", "stat2"],\n    "text_outputs": ["finding 1"]\n  }\n}'

class PythonAgent:

    def __init__(self, llm_client: BedrockClient):
        self.llm_client = llm_client
        logger.info('PythonAgent initialized')

    def _describe_data(self, query_results: list) -> str:
        if not query_results:
            return 'Dataset: EMPTY — no rows returned'
        columns = list(query_results[0].keys())
        total_rows = len(query_results)
        sample_rows = query_results[:5]
        sample_lines = []
        for i, row in enumerate(sample_rows, 1):
            row_str = ', '.join((f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}' for k, v in row.items()))
            sample_lines.append(f'  Row {i}: {row_str}')
        return f"Dataset shape: {total_rows} rows × {len(columns)} columns\nColumns: {', '.join(columns)}\nSample rows (first 5 of {total_rows}):\n" + '\n'.join(sample_lines)

    def process(self, state: QueryState) -> Dict[str, Any]:
        query_results = state.get('query_results', [])
        plan = state.get('plan', {})
        intent = state.get('intent', plan.get('intent', 'aggregation'))
        user_query = state['user_query']
        if not query_results:
            logger.warning('PythonAgent: No query results to analyze')
            return {'generated_python': "# No data returned from Redshift\ntext_outputs.append('No data was found matching your query. Please check your filters or try a different time range.')\nstatistics['total_rows'] = 0", 'step_log': ['⚠️ Python Agent: No data — generating empty-result code']}
        data_description = self._describe_data(query_results)
        logger.info(f"PythonAgent generating code for intent='{intent}' | rows={len(query_results)}")
        analysis_requirements = []
        intent_analysis_map = {'trend_analysis': ['Create a time-series line chart showing the trend', 'Calculate period-over-period change rates', 'Identify the peak and lowest points'], 'aggregation': ['Compute sum, mean, and count for all numeric columns', 'Create a bar chart of the aggregated results', 'Store total values in statistics dict'], 'comparison': ['Compare values across categories', 'Create a grouped bar chart or side-by-side comparison', 'Compute percentage differences'], 'ranking': ['Sort data by the primary numeric column descending', 'Show top 10 and bottom 10 items', 'Create a horizontal bar chart (better for ranking readability)'], 'distribution': ['Create a histogram to show the distribution', 'Calculate percentiles (25th, 50th, 75th, 95th)', 'Compute mean and standard deviation', 'Identify outliers using the IQR method'], 'correlation': ['Calculate correlation matrix between numeric columns', 'Create a scatter plot for the two most correlated columns', 'Identify the strongest positive and negative correlations']}
        specific_requirements = intent_analysis_map.get(intent, intent_analysis_map['aggregation'])
        analysis_requirements.extend(specific_requirements)
        grouping_dims = plan.get('grouping_dimensions', [])
        if grouping_dims:
            analysis_requirements.append(f"Group analysis by: {', '.join(grouping_dims)}")
        time_period = plan.get('time_period')
        if time_period:
            analysis_requirements.append(f'Filter or highlight: {time_period}')
        user_message = f"""Write Python code to analyze this business data:\n\nORIGINAL USER QUESTION: {user_query}\n\nANALYTICAL INTENT: {intent}\nANALYTICAL OBJECTIVE: {plan.get('analytical_objective', 'Perform data analysis')}\n\nDATA AVAILABLE:\n{data_description}\n\nREQUIRED ANALYSIS STEPS:\n{chr(10).join((f'  {i + 1}. {req}' for i, req in enumerate(analysis_requirements)))}\n\nIMPORTANT REMINDERS:\n- Start with: df = pd.DataFrame(query_results)\n- Append ALL charts: charts.append(fig.to_dict())\n- Store ALL metrics: statistics["key"] = value\n- Append text findings: text_outputs.append("...")\n- Apply chart titles that answer the business question\n\nRespond with ONLY valid JSON."""
        try:
            response_text = self.llm_client.generate(system_prompt=PYTHON_AGENT_SYSTEM_PROMPT, user_message=user_message, model_id=BedrockClient.SONNET_MODEL_ID, temperature=0.1, max_tokens=4096)
            result = self.llm_client.parse_json_response(response_text)
            python_code = result.get('python_code', '').strip()
            if not python_code:
                raise ValueError('LLM returned empty Python code')
            code_explanation = result.get('code_explanation', '')
            expected_outputs = result.get('expected_outputs', {})
            logger.info(f"PythonAgent generated {len(python_code)} chars of code | expected_charts={len(expected_outputs.get('charts', []))}")
            return {'generated_python': python_code, 'step_log': [f'✅ Python Agent: Generated {len(python_code)} chars of code | intent={intent} | {code_explanation[:80]}']}
        except Exception as e:
            error_msg = f'PythonAgent failed: {e}'
            logger.error(error_msg)
            fallback_code = "import pandas as pd\nimport numpy as np\nimport plotly.express as px\n\ndf = pd.DataFrame(query_results)\n\nif df.empty:\n    text_outputs.append('No data was returned.')\nelse:\n    statistics['total_rows'] = len(df)\n    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n    for col in numeric_cols:\n        statistics[f'{col}_sum'] = float(df[col].sum())\n        statistics[f'{col}_mean'] = float(df[col].mean())\n    # Basic bar chart with first two columns\n    if len(df.columns) >= 2:\n        fig = px.bar(df.head(20), x=df.columns[0], y=df.columns[1],\n                     title='Query Results', template='plotly_white')\n        charts.append(fig.to_dict())\n    text_outputs.append(f'Analysis complete. {len(df)} rows analyzed.')\n"
            return {'generated_python': fallback_code, 'error': error_msg, 'step_log': [f'⚠️ Python Agent: LLM failed, using fallback code | {e}']}