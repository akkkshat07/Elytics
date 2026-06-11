import logging
from typing import Dict, Any
from ..utils.bedrock import BedrockClient
from ..utils.redshift import RedshiftClient
from ..state import QueryState

logger = logging.getLogger(__name__)

SQL_SYSTEM_PROMPT = """You are an expert Amazon Redshift SQL developer. Your task is to write
a single, valid SELECT SQL query to answer an analytical question.

REDSHIFT-SPECIFIC RULES:
- Use DATE_TRUNC('month', date_column) for month-level grouping
- Use DATEADD(day, -30, GETDATE()) for relative date filters
- Always alias columns with descriptive names (e.g., SUM(amount) AS total_sales)
- Use LIMIT 10000 unless a specific limit is requested
- Write only SELECT statements — never INSERT, UPDATE, DELETE, or DROP
- Use double quotes for column names that are reserved words

Respond with ONLY valid JSON:
{
  "sql": "Your complete SQL query here",
  "explanation": "Plain English explanation of what the SQL does"
}"""

SQL_REPAIR_PROMPT = """Fix the Amazon Redshift SQL query. It failed with an error.
Return ONLY corrected SELECT SQL using the provided schema.

Respond with ONLY valid JSON:
{
  "sql": "Your complete corrected SQL query here",
  "explanation": "Explanation of what you fixed"
}"""

class SQLAgent:
    def __init__(self, llm_client: BedrockClient, redshift_client: RedshiftClient = None):
        self.llm_client = llm_client
        self.redshift_client = redshift_client
        self.max_retries = 3
        logger.info("SQLAgent initialized with Live SQL Retry Handler (CoreSight Architecture)")

    def _format_schema_context_for_prompt(self, schema_context: Dict) -> str:
        if not schema_context or not schema_context.get('relevant_tables'):
            return "No schema context available."
        lines = ["RELEVANT TABLES AND COLUMNS:"]
        for table_info in schema_context.get('relevant_tables', []):
            table_name = table_info.get('table_name', 'unknown')
            columns = ', '.join(table_info.get('relevant_columns', []))
            lines.append(f"TABLE: {table_name}\nCOLUMNS: {columns}")
        return "\n".join(lines)

    def process(self, state: QueryState) -> Dict[str, Any]:
        user_query = state['user_query']
        plan = state.get('plan', {})
        schema_context = state.get('schema_context', {})
        
        logger.info(f"SQLAgent generating SQL for: '{user_query[:100]}'")
        schema_str = self._format_schema_context_for_prompt(schema_context)
        
        # Initial Generation Prompt
        user_message = f"""Generate a Redshift SQL query to answer this question.

USER QUESTION: {user_query}

ANALYTICAL PLAN:
- Intent: {plan.get('intent', 'unknown')}
- Objective: {plan.get('analytical_objective', '')}

{schema_str}

Write a single SELECT SQL query. Respond with ONLY valid JSON."""

        step_log = []
        sql = ""
        last_error = ""
        query_results = []
        
        # Live SQL Retry Loop
        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt == 1:
                    prompt_to_use = SQL_SYSTEM_PROMPT
                    msg_to_use = user_message
                else:
                    logger.warning(f"SQLAgent Attempt {attempt}: Retrying SQL generation after error")
                    step_log.append(f"🔄 SQL Agent: Retrying query execution (Attempt {attempt})")
                    prompt_to_use = SQL_REPAIR_PROMPT
                    msg_to_use = f"""{schema_str}\n\nUSER_QUERY:\n{user_query}\n\nFAILED_QUERY:\n{sql}\n\nERROR:\n{last_error}"""

                response_text = self.llm_client.generate(
                    system_prompt=prompt_to_use, 
                    user_message=msg_to_use, 
                    model_id=BedrockClient.SONNET_MODEL_ID, 
                    temperature=0.0
                )
                
                result = self.llm_client.parse_json_response(response_text)
                sql = result.get('sql', '').strip()
                
                if not sql:
                    raise ValueError("LLM returned empty SQL")
                    
                step_log.append(f"✅ SQL Agent: Generated SQL (Attempt {attempt}) | {result.get('explanation', '')[:60]}...")
                
                # Execute immediately if redshift client is available
                if self.redshift_client:
                    try:
                        logger.info(f"Executing SQL (Attempt {attempt}): {sql[:100]}...")
                        query_results = self.redshift_client.execute_query(sql)
                        step_log.append(f"✅ Redshift Query: {len(query_results)} rows returned successfully")
                        
                        return {
                            'generated_sql': sql,
                            'query_results': query_results,
                            'error': "",
                            'step_log': step_log
                        }
                    except Exception as db_err:
                        last_error = str(db_err)
                        logger.error(f"Redshift execution failed on attempt {attempt}: {last_error}")
                        step_log.append(f"❌ Redshift Query Error: {last_error}")
                        # Loop will continue and feed this back to the LLM
                else:
                    # If no redshift client injected (testing mode), just return the SQL
                    return {'generated_sql': sql, 'step_log': step_log}
                    
            except Exception as e:
                last_error = f"Generation failed: {e}"
                logger.error(f"SQLAgent generation failed on attempt {attempt}: {last_error}")
                step_log.append(f"⚠️ SQL Agent generation error: {e}")

        # If we exhausted all retries
        error_msg = f"SQLAgent exhausted {self.max_retries} attempts. Last error: {last_error}"
        logger.error(error_msg)
        step_log.append(f"❌ SQL Agent Failed: {error_msg}")
        
        return {
            'generated_sql': sql, 
            'query_results': [],
            'error': error_msg, 
            'step_log': step_log
        }