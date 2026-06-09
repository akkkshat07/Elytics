"""
agents/sql.py — SQL Agent

ROLE IN THE PIPELINE:
    Step 3. Runs after SchemaAgent.

WHAT IT DOES:
    Takes the structured plan (from PlannerAgent) and the relevant schema context
    (from SchemaAgent) and generates a valid Amazon Redshift SQL SELECT query
    that will answer the user's question.

WHY A DEDICATED SQL AGENT?
    SQL generation is one of the most error-prone steps. Using a focused agent
    with a highly specific system prompt — that knows it's writing for Redshift
    specifically — produces much better SQL than a general-purpose agent.

REDSHIFT vs STANDARD SQL:
    Redshift is PostgreSQL-compatible but has some differences:
    - Uses DATEADD(), DATEDIFF(), DATE_TRUNC() instead of standard SQL date functions
    - Has its own WINDOW functions optimized for columnar storage
    - The LIMIT clause is standard PostgreSQL-style
    This system prompt specifically guides Claude for Redshift dialect.

PATTERN ADAPTED FROM:
    business_agent.py's approach of building a structured prompt from state data
    and making a single targeted LLM call to produce code/SQL.
"""

import logging
from typing import Dict, Any

from ..utils.bedrock import BedrockClient
from ..state import QueryState

logger = logging.getLogger(__name__)

SQL_SYSTEM_PROMPT = """You are an expert Amazon Redshift SQL developer. Your task is to write
a single, valid SELECT SQL query to answer an analytical question.

REDSHIFT-SPECIFIC RULES:
- Use DATE_TRUNC('month', date_column) for month-level grouping
- Use DATEADD(day, -30, GETDATE()) for relative date filters
- Use DATEDIFF(day, start_date, end_date) for date differences
- Always alias columns with descriptive names (e.g., SUM(amount) AS total_sales)
- Use LIMIT 10000 unless a specific limit is requested
- Write only SELECT statements — never INSERT, UPDATE, DELETE, or DROP
- Use double quotes for column names that are reserved words

Respond with ONLY valid JSON:
{
  "sql": "Your complete SQL query here",
  "explanation": "Plain English explanation of what the SQL does",
  "estimated_rows": "rough estimate like 'hundreds' or 'thousands'"
}"""


class SQLAgent:
    """
    Generates a Redshift SQL query from the plan and schema context.
    
    WHY one LLM call?
        The schema context already contains all the relevant tables and columns,
        so one well-prompted call is sufficient. The Validation Agent will
        catch any errors before execution.
    """

    def __init__(self, llm_client: BedrockClient):
        self.llm_client = llm_client
        logger.info("SQLAgent initialized")

    def _format_schema_context_for_prompt(self, schema_context: Dict) -> str:
        """
        Convert the schema_context dict from SchemaAgent into a readable string
        for the SQL generation prompt.
        """
        if not schema_context or not schema_context.get("relevant_tables"):
            return "No schema context available."

        lines = ["RELEVANT TABLES AND COLUMNS:"]
        for table_info in schema_context.get("relevant_tables", []):
            table_name = table_info.get("table_name", "unknown")
            columns = ", ".join(table_info.get("relevant_columns", []))
            reason = table_info.get("reason", "")
            lines.append(f"\nTABLE: {table_name}")
            lines.append(f"  COLUMNS: {columns}")
            lines.append(f"  PURPOSE: {reason}")

        join_hints = schema_context.get("join_hints", "")
        if join_hints:
            lines.append(f"\nJOIN HINTS: {join_hints}")

        schema_notes = schema_context.get("schema_notes", "")
        if schema_notes:
            lines.append(f"\nSCHEMA NOTES: {schema_notes}")

        return "\n".join(lines)

    def process(self, state: QueryState) -> Dict[str, Any]:
        """
        LangGraph node function.

        Reads:
            state["user_query"]     — original question (for context)
            state["plan"]           — structured plan from PlannerAgent
            state["schema_context"] — relevant schema from SchemaAgent

        Writes:
            state["generated_sql"]  — the SQL string ready for validation
        """
        user_query = state["user_query"]
        plan = state.get("plan", {})
        schema_context = state.get("schema_context", {})

        logger.info(f"SQLAgent generating SQL for: '{user_query[:100]}'")

        schema_str = self._format_schema_context_for_prompt(schema_context)

        user_message = f"""Generate a Redshift SQL query to answer this question.

ORIGINAL USER QUESTION: {user_query}

ANALYTICAL PLAN:
- Intent: {plan.get('intent', 'unknown')}
- Objective: {plan.get('analytical_objective', '')}
- Key Filters: {plan.get('key_filters', [])}
- Grouping Dimensions: {plan.get('grouping_dimensions', [])}
- Time Period: {plan.get('time_period', 'not specified')}

{schema_str}

Write a single SELECT SQL query. Respond with ONLY valid JSON."""

        try:
            response_text = self.llm_client.generate(
                system_prompt=SQL_SYSTEM_PROMPT,
                user_message=user_message,
                model_id=BedrockClient.SONNET_MODEL_ID,
                temperature=0.0,  # SQL must be deterministic
            )
            result = self.llm_client.parse_json_response(response_text)
            sql = result.get("sql", "").strip()

            if not sql:
                raise ValueError("LLM returned empty SQL")

            logger.info(f"SQLAgent generated SQL: {sql[:200]}...")

            return {
                "generated_sql": sql,
                "step_log": [f"✅ SQL Agent: Generated SQL ({len(sql)} chars) | {result.get('explanation', '')[:80]}"],
            }

        except Exception as e:
            error_msg = f"SQLAgent failed: {e}"
            logger.error(error_msg)
            return {
                "generated_sql": "",
                "error": error_msg,
                "step_log": [f"❌ SQL Agent failed: {e}"],
            }
