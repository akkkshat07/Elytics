import json
import logging
from collections import defaultdict
from typing import Dict, Any, List
from ..utils.bedrock import BedrockClient
from ..utils.redshift import RedshiftClient
from ..state import QueryState
logger = logging.getLogger(__name__)
SCHEMA_SYSTEM_PROMPT = 'You are a Database Schema Expert. Given an analytical objective\nand a full database schema, identify the MINIMUM set of tables and columns needed to answer\nthe question.\n\nRespond with ONLY valid JSON:\n{\n  "relevant_tables": [\n    {\n      "table_name": "table_name_here",\n      "reason": "Why this table is needed",\n      "relevant_columns": ["col1", "col2", "col3"]\n    }\n  ],\n  "join_hints": "Optional: describe how tables should be joined if multiple tables are needed",\n  "schema_notes": "Any important schema observations (e.g., date format, naming conventions)"\n}'

class SchemaAgent:

    def __init__(self, llm_client: BedrockClient, redshift_client: RedshiftClient):
        self.llm_client = llm_client
        self.redshift_client = redshift_client
        self._cached_schema: Dict[str, List[dict]] = {}
        logger.info('SchemaAgent initialized')

    def _fetch_and_group_schema(self) -> Dict[str, List[dict]]:
        if self._cached_schema:
            logger.info('Using cached schema')
            return self._cached_schema
        raw_schema = self.redshift_client.get_schema()
        schema_by_table = defaultdict(list)
        for row in raw_schema:
            schema_by_table[row['table_name']].append({'column_name': row['column_name'], 'data_type': row['data_type']})
        self._cached_schema = dict(schema_by_table)
        logger.info(f'Fetched schema: {len(self._cached_schema)} tables')
        return self._cached_schema

    def _format_schema_for_llm(self, schema: Dict[str, List[dict]]) -> str:
        lines = []
        for table_name, columns in schema.items():
            col_strs = [f"{c['column_name']}({c['data_type']})" for c in columns]
            lines.append(f"TABLE: {table_name} | COLUMNS: {', '.join(col_strs)}")
        return '\n'.join(lines)

    def process(self, state: QueryState) -> Dict[str, Any]:
        plan = state.get('plan', {})
        if not plan:
            return {'schema_context': {}, 'error': 'SchemaAgent: No plan found in state. Did PlannerAgent succeed?', 'step_log': ['❌ Schema Agent: No plan available']}
        analytical_objective = plan.get('analytical_objective', state['user_query'])
        logger.info(f"SchemaAgent processing for objective: '{analytical_objective[:100]}'")
        try:
            full_schema = self._fetch_and_group_schema()
            schema_text = self._format_schema_for_llm(full_schema)
            user_message = f"ANALYTICAL OBJECTIVE:\n{analytical_objective}\n\nINTENT: {plan.get('intent', 'unknown')}\nKEY FILTERS: {plan.get('key_filters', [])}\nGROUPING DIMENSIONS: {plan.get('grouping_dimensions', [])}\n\nFULL DATABASE SCHEMA:\n{schema_text}\n\nBased on the above objective and schema, identify ONLY the relevant tables and columns.\nRespond with ONLY valid JSON."
            response_text = self.llm_client.generate(system_prompt=SCHEMA_SYSTEM_PROMPT, user_message=user_message, model_id=BedrockClient.SONNET_MODEL_ID, temperature=0.0)
            schema_context = self.llm_client.parse_json_response(response_text)
            schema_context['all_available_tables'] = list(full_schema.keys())
            relevant_tables = [t['table_name'] for t in schema_context.get('relevant_tables', [])]
            logger.info(f'SchemaAgent identified tables: {relevant_tables}')
            return {'schema_context': schema_context, 'step_log': [f'✅ Schema Agent: Identified tables {relevant_tables}']}
        except Exception as e:
            error_msg = f'SchemaAgent failed: {e}'
            logger.error(error_msg)
            return {'schema_context': {}, 'error': error_msg, 'step_log': [f'❌ Schema Agent failed: {e}']}