import re
import logging
from typing import Dict, Any, Tuple
from ..utils.bedrock import BedrockClient
from ..state import QueryState
logger = logging.getLogger(__name__)
FORBIDDEN_SQL_KEYWORDS = frozenset({'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE'})
VALIDATOR_SYSTEM_PROMPT = 'You are a SQL Security and Quality Reviewer specializing in Amazon Redshift.\nReview the provided SQL query for:\n1. Logical correctness (does it answer the stated objective?)\n2. Potential issues (cartesian joins, missing WHERE clauses on large tables, etc.)\n3. Redshift-specific syntax correctness\n\nRespond with ONLY valid JSON:\n{\n  "is_valid": true or false,\n  "issues": ["list of issues found, or empty list if none"],\n  "corrected_sql": "If is_valid is false and you can fix it, provide corrected SQL. Otherwise null.",\n  "confidence": "high | medium | low",\n  "review_notes": "Brief notes on query quality"\n}'

class ValidationAgent:

    def __init__(self, llm_client: BedrockClient):
        self.llm_client = llm_client
        logger.info('ValidationAgent initialized')

    def _rule_based_checks(self, sql: str, schema_context: Dict) -> Tuple[bool, list]:
        issues = []
        if not sql or not sql.strip():
            return (False, ['SQL is empty'])
        sql_upper = sql.strip().upper()
        first_keyword = sql_upper.split()[0] if sql_upper.split() else ''
        if first_keyword in FORBIDDEN_SQL_KEYWORDS:
            issues.append(f"SAFETY VIOLATION: SQL starts with '{first_keyword}' — only SELECT is allowed")
            return (False, issues)
        if not sql_upper.startswith('SELECT') and 'WITH' not in sql_upper[:10]:
            issues.append('SQL does not start with SELECT or a CTE (WITH clause)')
        table_pattern = re.compile('\\b(?:FROM|JOIN)\\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.IGNORECASE)
        referenced_tables = {m.group(1).lower() for m in table_pattern.finditer(sql)}
        all_available = {t.lower() for t in schema_context.get('all_available_tables', [])}
        if all_available:
            unknown_tables = referenced_tables - all_available
            if unknown_tables:
                issues.append(f'Unknown tables referenced: {unknown_tables}')
        if 'FROM' not in sql_upper:
            issues.append('SQL missing FROM clause')
        is_safe = len(issues) == 0
        return (is_safe, issues)

    def process(self, state: QueryState) -> Dict[str, Any]:
        sql = state.get('generated_sql', '')
        schema_context = state.get('schema_context', {})
        plan = state.get('plan', {})
        if not sql:
            return {'sql_validation': {'is_valid': False, 'issues': ['No SQL was generated']}, 'error': 'No SQL to validate', 'step_log': ['❌ Validator: No SQL to validate']}
        logger.info(f'ValidationAgent checking SQL: {sql[:200]}...')
        rule_pass, rule_issues = self._rule_based_checks(sql, schema_context)
        if not rule_pass and any(('SAFETY' in i for i in rule_issues)):
            logger.warning(f'SECURITY BLOCK: {rule_issues}')
            return {'sql_validation': {'is_valid': False, 'issues': rule_issues, 'corrected_sql': None, 'confidence': 'high', 'review_notes': 'Blocked by safety rules'}, 'error': f'SQL validation failed: {rule_issues}', 'step_log': [f'🚫 Validator: BLOCKED dangerous SQL — {rule_issues}']}
        try:
            user_message = f"Review this Amazon Redshift SQL query:\n\nANALYTICAL OBJECTIVE: {plan.get('analytical_objective', 'Not specified')}\n\nSQL QUERY:\n{sql}\n\nRULE-BASED PRE-CHECK ISSUES (if any): {(rule_issues if rule_issues else 'None')}\n\nRespond with ONLY valid JSON."
            response_text = self.llm_client.generate(system_prompt=VALIDATOR_SYSTEM_PROMPT, user_message=user_message, model_id=BedrockClient.HAIKU_MODEL_ID, temperature=0.0)
            validation = self.llm_client.parse_json_response(response_text)
            all_issues = rule_issues + validation.get('issues', [])
            validation['issues'] = all_issues
            corrected_sql = validation.get('corrected_sql')
            updated_sql = corrected_sql if corrected_sql else sql
            is_valid = validation.get('is_valid', False) and (not rule_issues)
            logger.info(f"ValidationAgent result: valid={is_valid} | confidence={validation.get('confidence')} | issues={all_issues}")
            return {'sql_validation': validation, 'generated_sql': updated_sql, 'step_log': [f"{('✅' if is_valid else '⚠️')} Validator: valid={is_valid} | issues={len(all_issues)} | confidence={validation.get('confidence')}"]}
        except Exception as e:
            error_msg = f'ValidationAgent LLM review failed: {e}'
            logger.error(error_msg)
            is_valid = rule_pass
            return {'sql_validation': {'is_valid': is_valid, 'issues': rule_issues, 'corrected_sql': None, 'confidence': 'low', 'review_notes': f'LLM review failed, using rule-based result: {e}'}, 'step_log': [f'⚠️ Validator: LLM review failed, rule-based={is_valid}']}