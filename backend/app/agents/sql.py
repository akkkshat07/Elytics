import logging
from typing import Dict, Any
from ..utils.bedrock import BedrockClient
from ..state import QueryState
logger = logging.getLogger(__name__)
SQL_SYSTEM_PROMPT = 'You are an expert Amazon Redshift SQL developer. Your task is to write\na single, valid SELECT SQL query to answer an analytical question.\n\nREDSHIFT-SPECIFIC RULES:\n- Use DATE_TRUNC(\'month\', date_column) for month-level grouping\n- Use DATEADD(day, -30, GETDATE()) for relative date filters\n- Use DATEDIFF(day, start_date, end_date) for date differences\n- Always alias columns with descriptive names (e.g., SUM(amount) AS total_sales)\n- Use LIMIT 10000 unless a specific limit is requested\n- Write only SELECT statements — never INSERT, UPDATE, DELETE, or DROP\n- Use double quotes for column names that are reserved words\n\nRespond with ONLY valid JSON:\n{\n "sql": "Your complete SQL query here",\n "explanation": "Plain English explanation of what the SQL does",\n "estimated_rows": "rough estimate like \'hundreds\' or \'thousands\'"\n}'

class SQLAgent:

  def __init__(self, llm_client: BedrockClient):
    self.llm_client = llm_client
    logger.info('SQLAgent initialized')

  def _format_schema_context_for_prompt(self, schema_context: Dict) -> str:
    if not schema_context or not schema_context.get('relevant_tables'):
      return 'No schema context available.'
    lines = ['RELEVANT TABLES AND COLUMNS:']
    for table_info in schema_context.get('relevant_tables', []):
      table_name = table_info.get('table_name', 'unknown')
      columns = ', '.join(table_info.get('relevant_columns', []))
      reason = table_info.get('reason', '')
      lines.append(f'\nTABLE: {table_name}')
      lines.append(f' COLUMNS: {columns}')
      lines.append(f' PURPOSE: {reason}')
    join_hints = schema_context.get('join_hints', '')
    if join_hints:
      lines.append(f'\nJOIN HINTS: {join_hints}')
    schema_notes = schema_context.get('schema_notes', '')
    if schema_notes:
      lines.append(f'\nSCHEMA NOTES: {schema_notes}')
    return '\n'.join(lines)

  def process(self, state: QueryState) -> Dict[str, Any]:
    user_query = state['user_query']
    plan = state.get('plan', {})
    schema_context = state.get('schema_context', {})
    logger.info(f"SQLAgent generating SQL for: '{user_query[:100]}'")
    schema_str = self._format_schema_context_for_prompt(schema_context)
    user_message = f"Generate a Redshift SQL query to answer this question.\n\nORIGINAL USER QUESTION: {user_query}\n\nANALYTICAL PLAN:\n- Intent: {plan.get('intent', 'unknown')}\n- Objective: {plan.get('analytical_objective', '')}\n- Key Filters: {plan.get('key_filters', [])}\n- Grouping Dimensions: {plan.get('grouping_dimensions', [])}\n- Time Period: {plan.get('time_period', 'not specified')}\n\n{schema_str}\n\nWrite a single SELECT SQL query. Respond with ONLY valid JSON."
    try:
      response_text = self.llm_client.generate(system_prompt=SQL_SYSTEM_PROMPT, user_message=user_message, model_id=BedrockClient.SONNET_MODEL_ID, temperature=0.0)
      result = self.llm_client.parse_json_response(response_text)
      sql = result.get('sql', '').strip()
      if not sql:
        raise ValueError('LLM returned empty SQL')
      logger.info(f'SQLAgent generated SQL: {sql[:200]}...')
      return {'generated_sql': sql, 'step_log': [f" SQL Agent: Generated SQL ({len(sql)} chars) | {result.get('explanation', '')[:80]}"]}
    except Exception as e:
      error_msg = f'SQLAgent failed: {e}'
      logger.error(error_msg)
      return {'generated_sql': '', 'error': error_msg, 'step_log': [f' SQL Agent failed: {e}']}