"""
agents/python_agent.py — Python Code Generation Agent (Architecture v2)

ROLE IN THE PIPELINE:
    Step 6. Runs AFTER the Redshift query execution node.
    New in Architecture v2 — replaces the old hard-coded AnalystAgent.

WHAT IT DOES:
    Takes the raw Redshift query results (list of dicts) + the analytical plan
    and asks Claude to write executable Python code (Pandas, NumPy, Plotly) that
    will perform the appropriate analysis for this specific question.

WHY LLM-GENERATED CODE INSTEAD OF HARD-CODED ANALYSIS?
    The old AnalystAgent had pre-written, hard-coded logic for each analysis type
    (trend → line chart, ranking → bar chart, etc.). This was limited:
    - You can't pre-code every possible analysis a business user might need
    - Correlation analysis, outlier detection, clustering all need different logic
    - The LLM can write exactly the right code for the exact question asked

    Architecture v2 makes the system truly adaptive: the LLM reads the data shape
    and the question intent, then writes custom Pandas/Plotly code on the fly.
    The ExecutorAgent then runs it safely.

CODE GENERATION CONTRACT:
    The generated code MUST follow these conventions so the ExecutorAgent can
    capture outputs correctly:
    1. `query_results` is available as a pre-loaded variable (list of dicts)
    2. `df = pd.DataFrame(query_results)` is the primary DataFrame
    3. Plotly charts MUST be appended to a pre-existing `charts` list variable:
           charts.append(fig.to_dict())
    4. Text outputs MUST be appended to `text_outputs` list:
           text_outputs.append("some finding")
    5. Summary statistics MUST be stored in `statistics` dict:
           statistics["total_revenue"] = float(df["revenue"].sum())

ADAPTED FROM:
    data_analyst_agent.py — the pattern of asking an LLM to generate Python code,
    then capturing the code string in state for another agent to execute.
    excecuter_agent.py — establishes the variable injection contract for the sandbox.
"""

import logging
from typing import Dict, Any

from ..utils.bedrock import BedrockClient
from ..state import QueryState

logger = logging.getLogger(__name__)

# The system prompt instructs Claude on exactly how to write the analysis code.
# The contract (charts list, statistics dict, etc.) is explicitly defined here
# so that the ExecutorAgent knows exactly how to capture outputs.
PYTHON_AGENT_SYSTEM_PROMPT = """You are an expert Python Data Analyst. Your task is to write
Python code using Pandas, NumPy, and Plotly that analyzes data and answers a business question.

The code will be executed in a sandbox environment with these pre-loaded variables:
- `query_results`: List[Dict] — raw data rows from Amazon Redshift (already loaded)
- `pd`: pandas module
- `np`: numpy module
- `px`: plotly.express module
- `go`: plotly.graph_objects module
- `charts`: List — append Plotly chart dicts here: charts.append(fig.to_dict())
- `text_outputs`: List — append string findings here: text_outputs.append("...")
- `statistics`: Dict — store computed metrics here: statistics["key"] = value

RULES:
1. Start with: df = pd.DataFrame(query_results)
2. Handle edge cases: check if df is empty before analysis
3. Every Plotly figure MUST be appended to `charts`: charts.append(fig.to_dict())
4. Store all key numeric findings in `statistics` dict for the Insights agent
5. Append text summaries or findings to `text_outputs`
6. Use plotly.express (px) for most charts — it is simpler and cleaner
7. NEVER use: open(), import os, import sys, import subprocess, exec(), eval()
8. Apply professional chart styling: template="plotly_white", descriptive titles

Respond with ONLY valid JSON:
{
  "python_code": "your complete Python code as a single string",
  "code_explanation": "Plain English explanation of what the code does step by step",
  "expected_outputs": {
    "charts": ["description of chart 1", "description of chart 2"],
    "statistics": ["stat1", "stat2"],
    "text_outputs": ["finding 1"]
  }
}"""


class PythonAgent:
    """
    LLM-powered Python code generation agent.

    Takes the query results and analytical plan, asks Claude to write
    the correct Pandas/NumPy/Plotly code for this specific question.

    The generated code is stored in state["generated_python"] and then
    passed to the ExecutorAgent for safe execution.
    """

    def __init__(self, llm_client: BedrockClient):
        """
        Args:
            llm_client: Shared BedrockClient instance.
        """
        self.llm_client = llm_client
        logger.info("PythonAgent initialized")

    def _describe_data(self, query_results: list) -> str:
        """
        Create a compact description of the data shape and sample values
        to include in the prompt so Claude knows what columns/types to work with.

        WHY NOT pass the full data to the LLM?
            If the query returns 10,000 rows, passing all of it would exceed token limits.
            Instead, we pass the schema + a few sample rows so Claude understands
            the data structure without seeing every row.

        Args:
            query_results: Raw rows from Redshift

        Returns:
            Compact string description of the dataset
        """
        if not query_results:
            return "Dataset: EMPTY — no rows returned"

        # Extract column names from first row
        columns = list(query_results[0].keys())
        total_rows = len(query_results)

        # Show first 5 rows as a sample
        sample_rows = query_results[:5]
        sample_lines = []
        for i, row in enumerate(sample_rows, 1):
            row_str = ", ".join(
                f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                for k, v in row.items()
            )
            sample_lines.append(f"  Row {i}: {row_str}")

        return (
            f"Dataset shape: {total_rows} rows × {len(columns)} columns\n"
            f"Columns: {', '.join(columns)}\n"
            f"Sample rows (first 5 of {total_rows}):\n"
            + "\n".join(sample_lines)
        )

    def process(self, state: QueryState) -> Dict[str, Any]:
        """
        LangGraph node function.

        Reads:
            state["query_results"]  — raw Redshift data rows
            state["plan"]           — structured plan from PlannerAgent
            state["intent"]         — top-level intent string

        Writes:
            state["generated_python"] — the Python code string to be executed
        """
        query_results = state.get("query_results", [])
        plan = state.get("plan", {})
        intent = state.get("intent", plan.get("intent", "aggregation"))
        user_query = state["user_query"]

        if not query_results:
            logger.warning("PythonAgent: No query results to analyze")
            return {
                "generated_python": (
                    "# No data returned from Redshift\n"
                    "text_outputs.append('No data was found matching your query. "
                    "Please check your filters or try a different time range.')\n"
                    "statistics['total_rows'] = 0"
                ),
                "step_log": ["⚠️ Python Agent: No data — generating empty-result code"],
            }

        data_description = self._describe_data(query_results)
        logger.info(f"PythonAgent generating code for intent='{intent}' | rows={len(query_results)}")

        # Build the analysis requirements from the plan
        analysis_requirements = []

        # Map intent to specific analysis types requested by the design doc
        intent_analysis_map = {
            "trend_analysis": [
                "Create a time-series line chart showing the trend",
                "Calculate period-over-period change rates",
                "Identify the peak and lowest points",
            ],
            "aggregation": [
                "Compute sum, mean, and count for all numeric columns",
                "Create a bar chart of the aggregated results",
                "Store total values in statistics dict",
            ],
            "comparison": [
                "Compare values across categories",
                "Create a grouped bar chart or side-by-side comparison",
                "Compute percentage differences",
            ],
            "ranking": [
                "Sort data by the primary numeric column descending",
                "Show top 10 and bottom 10 items",
                "Create a horizontal bar chart (better for ranking readability)",
            ],
            "distribution": [
                "Create a histogram to show the distribution",
                "Calculate percentiles (25th, 50th, 75th, 95th)",
                "Compute mean and standard deviation",
                "Identify outliers using the IQR method",
            ],
            "correlation": [
                "Calculate correlation matrix between numeric columns",
                "Create a scatter plot for the two most correlated columns",
                "Identify the strongest positive and negative correlations",
            ],
        }

        specific_requirements = intent_analysis_map.get(intent, intent_analysis_map["aggregation"])
        analysis_requirements.extend(specific_requirements)

        # Add any specific requirements from the plan
        grouping_dims = plan.get("grouping_dimensions", [])
        if grouping_dims:
            analysis_requirements.append(f"Group analysis by: {', '.join(grouping_dims)}")

        time_period = plan.get("time_period")
        if time_period:
            analysis_requirements.append(f"Filter or highlight: {time_period}")

        user_message = f"""Write Python code to analyze this business data:

ORIGINAL USER QUESTION: {user_query}

ANALYTICAL INTENT: {intent}
ANALYTICAL OBJECTIVE: {plan.get('analytical_objective', 'Perform data analysis')}

DATA AVAILABLE:
{data_description}

REQUIRED ANALYSIS STEPS:
{chr(10).join(f'  {i+1}. {req}' for i, req in enumerate(analysis_requirements))}

IMPORTANT REMINDERS:
- Start with: df = pd.DataFrame(query_results)
- Append ALL charts: charts.append(fig.to_dict())
- Store ALL metrics: statistics["key"] = value
- Append text findings: text_outputs.append("...")
- Apply chart titles that answer the business question

Respond with ONLY valid JSON."""

        try:
            response_text = self.llm_client.generate(
                system_prompt=PYTHON_AGENT_SYSTEM_PROMPT,
                user_message=user_message,
                model_id=BedrockClient.SONNET_MODEL_ID,
                temperature=0.1,  # Slightly above 0 for code creativity, but mostly deterministic
                max_tokens=4096,  # Code can be long — allow more tokens
            )
            result = self.llm_client.parse_json_response(response_text)
            python_code = result.get("python_code", "").strip()

            if not python_code:
                raise ValueError("LLM returned empty Python code")

            code_explanation = result.get("code_explanation", "")
            expected_outputs = result.get("expected_outputs", {})

            logger.info(
                f"PythonAgent generated {len(python_code)} chars of code | "
                f"expected_charts={len(expected_outputs.get('charts', []))}"
            )

            return {
                "generated_python": python_code,
                "step_log": [
                    f"✅ Python Agent: Generated {len(python_code)} chars of code | "
                    f"intent={intent} | {code_explanation[:80]}"
                ],
            }

        except Exception as e:
            error_msg = f"PythonAgent failed: {e}"
            logger.error(error_msg)

            # Fallback: generate minimal safe code that returns basic stats
            fallback_code = (
                "import pandas as pd\nimport numpy as np\nimport plotly.express as px\n\n"
                "df = pd.DataFrame(query_results)\n\n"
                "if df.empty:\n"
                "    text_outputs.append('No data was returned.')\n"
                "else:\n"
                "    statistics['total_rows'] = len(df)\n"
                "    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n"
                "    for col in numeric_cols:\n"
                "        statistics[f'{col}_sum'] = float(df[col].sum())\n"
                "        statistics[f'{col}_mean'] = float(df[col].mean())\n"
                "    # Basic bar chart with first two columns\n"
                "    if len(df.columns) >= 2:\n"
                "        fig = px.bar(df.head(20), x=df.columns[0], y=df.columns[1],\n"
                "                     title='Query Results', template='plotly_white')\n"
                "        charts.append(fig.to_dict())\n"
                "    text_outputs.append(f'Analysis complete. {len(df)} rows analyzed.')\n"
            )
            return {
                "generated_python": fallback_code,
                "error": error_msg,
                "step_log": [f"⚠️ Python Agent: LLM failed, using fallback code | {e}"],
            }
