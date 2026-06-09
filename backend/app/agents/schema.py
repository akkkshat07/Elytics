"""
agents/schema.py — Schema Agent

ROLE IN THE PIPELINE:
    Step 2. Runs after PlannerAgent.

WHAT IT DOES:
    1. Fetches the full database schema from Redshift (all tables and columns).
    2. Uses the Planner's analytical objective to intelligently filter down to
       ONLY the relevant tables and columns needed for this specific question.
    3. Formats this schema context for the SQL Agent.

WHY THIS IS CRITICAL:
    If you give the SQL Agent the entire database schema (which could be hundreds
    of tables with thousands of columns), it either ignores the irrelevant parts
    or worse, gets confused and references wrong tables. By pre-selecting only
    relevant schema, we dramatically improve SQL quality.

PATTERN ADAPTED FROM:
    - explorer/schema_limits.py for schema normalization and filtering
    - router_agent.py's data context loading (_load_data_context method) for
      the pattern of caching schema and producing a compact context string.
"""

import json
import logging
from collections import defaultdict
from typing import Dict, Any, List

from ..utils.bedrock import BedrockClient
from ..utils.redshift import RedshiftClient
from ..state import QueryState

logger = logging.getLogger(__name__)

SCHEMA_SYSTEM_PROMPT = """You are a Database Schema Expert. Given an analytical objective
and a full database schema, identify the MINIMUM set of tables and columns needed to answer
the question.

Respond with ONLY valid JSON:
{
  "relevant_tables": [
    {
      "table_name": "table_name_here",
      "reason": "Why this table is needed",
      "relevant_columns": ["col1", "col2", "col3"]
    }
  ],
  "join_hints": "Optional: describe how tables should be joined if multiple tables are needed",
  "schema_notes": "Any important schema observations (e.g., date format, naming conventions)"
}"""


class SchemaAgent:
    """
    Discovers and filters the Redshift schema to find only what's needed
    for the current analytical query.

    WHY cache the full schema?
        Fetching the full schema via information_schema is a lightweight metadata
        query. We fetch it once and let the LLM filter it — this is much faster
        than making multiple targeted schema queries.
    """

    def __init__(self, llm_client: BedrockClient, redshift_client: RedshiftClient):
        """
        Args:
            llm_client:      Shared Bedrock client for the LLM filtering call.
            redshift_client: Redshift connection to fetch schema metadata.
        """
        self.llm_client = llm_client
        self.redshift_client = redshift_client
        # Cache the full schema in memory so it's only fetched once per server lifetime
        self._cached_schema: Dict[str, List[dict]] = {}
        logger.info("SchemaAgent initialized")

    def _fetch_and_group_schema(self) -> Dict[str, List[dict]]:
        """
        Fetch the full Redshift schema and group columns by table name.

        WHY group by table?
            The raw schema query returns one row per column. We group them by
            table_name to create a structured dict like:
            {
                "sales": [{"column_name": "date", "data_type": "date"}, ...],
                "customers": [...]
            }
            This is much easier to pass to the LLM.
        """
        if self._cached_schema:
            logger.info("Using cached schema")
            return self._cached_schema

        raw_schema = self.redshift_client.get_schema()
        schema_by_table = defaultdict(list)

        for row in raw_schema:
            schema_by_table[row["table_name"]].append({
                "column_name": row["column_name"],
                "data_type": row["data_type"],
            })

        # Convert defaultdict to regular dict and cache it
        self._cached_schema = dict(schema_by_table)
        logger.info(f"Fetched schema: {len(self._cached_schema)} tables")
        return self._cached_schema

    def _format_schema_for_llm(self, schema: Dict[str, List[dict]]) -> str:
        """
        Format the schema dict as a compact string for the LLM prompt.

        WHY compact format?
            LLMs have token limits. A compact format like:
            TABLE: sales | COLUMNS: date(date), amount(numeric), region(varchar)
            is much shorter than a JSON blob.
        """
        lines = []
        for table_name, columns in schema.items():
            col_strs = [f"{c['column_name']}({c['data_type']})" for c in columns]
            lines.append(f"TABLE: {table_name} | COLUMNS: {', '.join(col_strs)}")
        return "\n".join(lines)

    def process(self, state: QueryState) -> Dict[str, Any]:
        """
        LangGraph node function.

        Reads:
            state["plan"] — the structured plan from PlannerAgent

        Writes:
            state["schema_context"] — dict with relevant_tables, join_hints, etc.
        """
        plan = state.get("plan", {})
        if not plan:
            return {
                "schema_context": {},
                "error": "SchemaAgent: No plan found in state. Did PlannerAgent succeed?",
                "step_log": ["❌ Schema Agent: No plan available"],
            }

        analytical_objective = plan.get("analytical_objective", state["user_query"])
        logger.info(f"SchemaAgent processing for objective: '{analytical_objective[:100]}'")

        try:
            # Step 1: Get the full schema (cached after first call)
            full_schema = self._fetch_and_group_schema()
            schema_text = self._format_schema_for_llm(full_schema)

            user_message = f"""ANALYTICAL OBJECTIVE:
{analytical_objective}

INTENT: {plan.get('intent', 'unknown')}
KEY FILTERS: {plan.get('key_filters', [])}
GROUPING DIMENSIONS: {plan.get('grouping_dimensions', [])}

FULL DATABASE SCHEMA:
{schema_text}

Based on the above objective and schema, identify ONLY the relevant tables and columns.
Respond with ONLY valid JSON."""

            response_text = self.llm_client.generate(
                system_prompt=SCHEMA_SYSTEM_PROMPT,
                user_message=user_message,
                model_id=BedrockClient.SONNET_MODEL_ID,
                temperature=0.0,
            )
            schema_context = self.llm_client.parse_json_response(response_text)

            # Attach the full list of available table names for reference
            schema_context["all_available_tables"] = list(full_schema.keys())

            relevant_tables = [t["table_name"] for t in schema_context.get("relevant_tables", [])]
            logger.info(f"SchemaAgent identified tables: {relevant_tables}")

            return {
                "schema_context": schema_context,
                "step_log": [f"✅ Schema Agent: Identified tables {relevant_tables}"],
            }

        except Exception as e:
            error_msg = f"SchemaAgent failed: {e}"
            logger.error(error_msg)
            return {
                "schema_context": {},
                "error": error_msg,
                "step_log": [f"❌ Schema Agent failed: {e}"],
            }
