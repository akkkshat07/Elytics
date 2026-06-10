import logging
from typing import Dict, Any, List
from ..utils.bedrock import BedrockClient
from ..state import QueryState
logger = logging.getLogger(__name__)
INSIGHTS_SYSTEM_PROMPT = 'You are a Business Intelligence Analyst providing insights to executive-level business users.\nYour job is to translate data analysis results into clear, actionable business insights.\n\nCRITICAL RULES:\n1. ONLY cite numbers that are explicitly provided in the statistics or text_outputs — NEVER invent or compute new numbers\n2. Use plain business language — avoid technical jargon like "DataFrame" or "Pandas"\n3. Be specific and concrete — "Revenue totaled $4.2M" not "Revenue was high"\n4. Directly answer the user\'s original question in the first insight\n5. Each insight string should be a complete, standalone sentence or short paragraph\n\nRespond with ONLY valid JSON:\n{\n  "insights": [\n    "Headline insight that directly answers the user\'s question (1-2 sentences)",\n    "Key finding #1 with specific numbers from the data",\n    "Key finding #2 with specific numbers from the data",\n    "Key finding #3 or trend observation",\n    "Optional: Business recommendation or action (only if clearly supported by data)"\n  ],\n  "data_notes": "Any important caveats (e.g., date range, row count, missing data)"\n}'

class InsightsAgent:

    def __init__(self, llm_client: BedrockClient):
        self.llm_client = llm_client
        logger.info('InsightsAgent initialized (v2)')

    def _format_execution_results_for_prompt(self, execution_results: Dict, text_outputs: List[str]) -> str:
        if not execution_results or execution_results.get('error'):
            error = execution_results.get('error', 'Unknown error') if execution_results else 'No results'
            return f'Execution error: {error}'
        lines = []
        statistics = execution_results.get('statistics', {})
        if statistics:
            lines.append('COMPUTED STATISTICS (use ONLY these numbers in insights):')
            for key, value in statistics.items():
                if isinstance(value, float):
                    lines.append(f"  {key.replace('_', ' ')}: {value:,.2f}")
                elif isinstance(value, int):
                    lines.append(f"  {key.replace('_', ' ')}: {value:,}")
                else:
                    lines.append(f"  {key.replace('_', ' ')}: {value}")
        if text_outputs:
            lines.append('\nANALYSIS FINDINGS (from Python analysis):')
            for finding in text_outputs:
                lines.append(f'  • {finding}')
        return '\n'.join(lines) if lines else 'No statistical results available'

    def _describe_charts(self, charts: List[Dict]) -> str:
        if not charts:
            return 'No charts generated'
        descriptions = []
        for i, chart in enumerate(charts, 1):
            chart_data = chart.get('data', [])
            layout = chart.get('layout', {})
            chart_type = chart_data[0].get('type', 'chart') if chart_data else 'chart'
            title_obj = layout.get('title', {})
            if isinstance(title_obj, dict):
                title = title_obj.get('text', 'Untitled')
            else:
                title = str(title_obj) if title_obj else 'Untitled'
            descriptions.append(f"  Chart {i}: {chart_type.capitalize()} — '{title}'")
        return 'CHARTS GENERATED:\n' + '\n'.join(descriptions)

    def process(self, state: QueryState) -> Dict[str, Any]:
        user_query = state['user_query']
        plan = state.get('plan', {})
        intent = state.get('intent', plan.get('intent', 'unknown'))
        execution_results = state.get('execution_results', {})
        charts = state.get('charts', [])
        sql = state.get('generated_sql', '')
        query_results = state.get('query_results', [])
        text_outputs = execution_results.get('text_outputs', []) if execution_results else []
        logger.info(f"InsightsAgent v2 generating insights | intent={intent} | charts={len(charts)} | stats={len(execution_results.get('statistics', {}))}")
        stats_text = self._format_execution_results_for_prompt(execution_results, text_outputs)
        charts_text = self._describe_charts(charts)
        total_rows = len(query_results)
        user_message = f"Generate business insights for this analysis:\n\nORIGINAL USER QUESTION: {user_query}\n\nANALYTICAL INTENT: {intent}\nOBJECTIVE: {plan.get('analytical_objective', 'Data analysis')}\n\n{stats_text}\n\n{charts_text}\n\nCONTEXT:\n- Total rows of data analyzed: {total_rows}\n- SQL was executed on Amazon Redshift\n- Python analysis was performed using Pandas and Plotly\n\nIMPORTANT: Only cite numbers from the COMPUTED STATISTICS section above.\nDo NOT invent numbers or percentages that are not shown in the data.\n\nRespond with ONLY valid JSON."
        try:
            response_text = self.llm_client.generate(system_prompt=INSIGHTS_SYSTEM_PROMPT, user_message=user_message, model_id=BedrockClient.SONNET_MODEL_ID, temperature=0.3, max_tokens=2048)
            result = self.llm_client.parse_json_response(response_text)
            insights_list = result.get('insights', [])
            data_notes = result.get('data_notes', '')
            if data_notes:
                insights_list.append(f'📋 Data note: {data_notes}')
            if not insights_list:
                insights_list = [f'Analysis of {total_rows} records complete. Please review the charts for visual findings.']
            logger.info(f'InsightsAgent generated {len(insights_list)} insight strings')
            return {'insights': insights_list, 'step_log': [f'✅ Insights: Generated {len(insights_list)} insights | intent={intent} | rows={total_rows}']}
        except Exception as e:
            error_msg = f'InsightsAgent failed: {e}'
            logger.error(error_msg)
            fallback_insights = [f'Analysis complete. {total_rows} records were retrieved from the database.']
            for finding in text_outputs[:3]:
                fallback_insights.append(str(finding))
            return {'insights': fallback_insights, 'error': error_msg, 'step_log': [f'⚠️ Insights: LLM failed, using fallback | {e}']}