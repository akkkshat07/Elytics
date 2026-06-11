import logging
from typing import Dict, Any, List
from ..utils.bedrock import BedrockClient
from ..state import QueryState
logger = logging.getLogger(__name__)
INSIGHTS_SYSTEM_PROMPT = 'You are an Executive Business Analyst. Your job is to translate data analysis results into CXO-grade strategic insights.\n\nCRITICAL RULES:\n1. You write for C-suite executives and directors. Every sentence must earn its place — no filler.\n2. ONLY cite numbers that are explicitly provided in the statistics or text_outputs. NEVER invent numbers.\n3. Use plain business language — avoid technical jargon like "DataFrame", "Pandas", or "Query".\n4. Output must be structured strictly as JSON.\n5. Format large numbers for readability (e.g., $4.2M, 500K).\n6. Provide actionable recommendations based ONLY on the data shown.\n\nRespond with ONLY valid JSON matching this structure exactly:\n{\n "summary": "Direct answer to the user\'s question with the key number. One sentence.",\n "metrics": [\n  "Metric #1: specific number with context (vs average, % of total, rank)",\n  "Metric #2: specific number with context"\n ],\n "insights": [\n  "Insight #1: [What] + [So What] + [Now What] - Explain the business implication of a trend or outlier",\n  "Insight #2: Another key finding"\n ],\n "recommendations": [\n  "Actionable step 1 based on the data",\n  "Actionable step 2 based on the data"\n ],\n "data_notes": "Any important caveats (e.g., date range, missing data). Leave empty if clean."\n}'

class InsightsAgent:

  def __init__(self, llm_client: BedrockClient):
    self.llm_client = llm_client
    logger.info('InsightsAgent initialized (CoreSight Upgrade)')

  def _format_execution_results_for_prompt(self, execution_results: Dict, text_outputs: List[str]) -> str:
    if not execution_results or execution_results.get('error'):
      error = execution_results.get('error', 'Unknown error') if execution_results else 'No results'
      return f'Execution error: {error}'
    lines = []
    statistics = execution_results.get('statistics', {})
    if statistics:
      lines.append('COMPUTED STATISTICS (use ONLY these numbers):')
      for key, value in statistics.items():
        if isinstance(value, float):
          lines.append(f" {key.replace('_', ' ')}: {value:,.2f}")
        elif isinstance(value, int):
          lines.append(f" {key.replace('_', ' ')}: {value:,}")
        else:
          lines.append(f" {key.replace('_', ' ')}: {value}")
    if text_outputs:
      lines.append('\nANALYSIS FINDINGS:')
      for finding in text_outputs:
        lines.append(f' • {finding}')
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
      descriptions.append(f" Chart {i}: {chart_type.capitalize()} — '{title}'")
    return 'CHARTS GENERATED:\n' + '\n'.join(descriptions)

  def process(self, state: QueryState) -> Dict[str, Any]:
    user_query = state.get('user_query', '')
    plan = state.get('plan', {})
    intent = state.get('intent', plan.get('intent', 'unknown'))
    execution_results = state.get('execution_results', {})
    charts = state.get('charts', [])
    query_results = state.get('query_results', [])
    text_outputs = execution_results.get('text_outputs', []) if execution_results else []
    logger.info(f'InsightsAgent generating CXO insights | intent={intent} | charts={len(charts)}')
    stats_text = self._format_execution_results_for_prompt(execution_results, text_outputs)
    charts_text = self._describe_charts(charts)
    total_rows = len(query_results)
    user_message = f"Generate executive business insights for this analysis:\n\nORIGINAL USER QUESTION: {user_query}\n\nANALYTICAL INTENT: {intent}\nOBJECTIVE: {plan.get('analytical_objective', 'Data analysis')}\n\n{stats_text}\n\n{charts_text}\n\nCONTEXT:\n- Total rows analyzed: {total_rows}\n- Domain: Edelweiss Life Insurance (Focus on policies, premiums, agents, etc. if applicable)\n\nIMPORTANT: Only cite numbers from the COMPUTED STATISTICS section above.\nRespond with ONLY valid JSON."
    try:
      response_text = self.llm_client.generate(system_prompt=INSIGHTS_SYSTEM_PROMPT, user_message=user_message, model_id=BedrockClient.SONNET_MODEL_ID, temperature=0.1, max_tokens=2048)
      result = self.llm_client.parse_json_response(response_text)
      formatted_insights = []
      if result.get('summary'):
        formatted_insights.append(f" **Summary:** {result['summary']}")
      if result.get('metrics'):
        metrics_str = ' \n'.join([f'• {m}' for m in result['metrics']])
        formatted_insights.append(f' **Key Metrics:**\n{metrics_str}')
      if result.get('insights'):
        for i, ins in enumerate(result['insights'], 1):
          formatted_insights.append(f' **Insight {i}:** {ins}')
      if result.get('recommendations'):
        rec_str = ' \n'.join([f'• {r}' for r in result['recommendations']])
        formatted_insights.append(f' **Recommendations:**\n{rec_str}')
      if result.get('data_notes'):
        formatted_insights.append(f" **Note:** {result['data_notes']}")
      if not formatted_insights:
        formatted_insights = [f'Analysis of {total_rows} records complete. Please review the charts.']
      logger.info(f'InsightsAgent generated {len(formatted_insights)} CXO insight blocks')
      return {'insights': formatted_insights, 'step_log': [f' Insights: Generated CXO-grade analysis | rows={total_rows}']}
    except Exception as e:
      error_msg = f'InsightsAgent failed: {e}'
      logger.error(error_msg)
      fallback = [f'Analysis complete. {total_rows} records were retrieved.']
      for finding in text_outputs[:3]:
        fallback.append(str(finding))
      return {'insights': fallback, 'error': error_msg, 'step_log': [f' Insights: LLM failed, using fallback | {e}']}