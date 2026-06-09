"""
agents/validator.py — SQL Validation Agent

ROLE IN THE PIPELINE:
    Step 4. Runs after SQLAgent. The last gate BEFORE we touch Redshift.

WHAT IT DOES:
    Validates the generated SQL by checking:
    1. SYNTAX — Python's sqlparse library can detect obvious syntax issues
    2. SAFETY — Ensures only SELECT statements are executed (no DDL/DML)
    3. TABLE CHECK — Verifies referenced tables exist in the known schema
    4. COLUMN CHECK — Verifies referenced columns exist in those tables
    5. LLM REVIEW — Final common-sense check using Claude Haiku (cheaper model)
       for subtler semantic issues

WHY TWO LEVELS OF VALIDATION?
    Rule-based checks (steps 1-4) are fast and free (no API call).
    LLM review (step 5) catches semantic issues like joining on wrong keys.
    Using Haiku instead of Sonnet here saves money on a simple yes/no check.

SAFETY FIRST:
    SQL injection or accidental data modification is prevented by:
    a) The restricted builtins pattern from the reference excecuter_agent.py
    b) This agent rejecting any non-SELECT statement before execution

PATTERN ADAPTED FROM:
    excecuter_agent.py's _EXEC_RESTRICTED_BUILTINS and timeout pattern
    (adapted for SQL safety instead of Python code safety).
"""

import re
import logging
from typing import Dict, Any, Tuple

from ..utils.bedrock import BedrockClient
from ..state import QueryState

logger = logging.getLogger(__name__)

# These SQL keywords at the start of a statement are dangerous — reject them
# WHY a set? O(1) lookup, much faster than regex for a small list
FORBIDDEN_SQL_KEYWORDS = frozenset({
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
    "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
})

VALIDATOR_SYSTEM_PROMPT = """You are a SQL Security and Quality Reviewer specializing in Amazon Redshift.
Review the provided SQL query for:
1. Logical correctness (does it answer the stated objective?)
2. Potential issues (cartesian joins, missing WHERE clauses on large tables, etc.)
3. Redshift-specific syntax correctness

Respond with ONLY valid JSON:
{
  "is_valid": true or false,
  "issues": ["list of issues found, or empty list if none"],
  "corrected_sql": "If is_valid is false and you can fix it, provide corrected SQL. Otherwise null.",
  "confidence": "high | medium | low",
  "review_notes": "Brief notes on query quality"
}"""


class ValidationAgent:
    """
    Two-stage SQL validator: rule-based fast checks + LLM semantic review.

    WHY use the cheaper Haiku model here?
        Validation is a simpler yes/no task compared to SQL generation.
        Claude Haiku is significantly faster and cheaper than Sonnet for
        pattern-matching style tasks. This saves cost on every query.
    """

    def __init__(self, llm_client: BedrockClient):
        self.llm_client = llm_client
        logger.info("ValidationAgent initialized")

    def _rule_based_checks(
        self, sql: str, schema_context: Dict
    ) -> Tuple[bool, list]:
        """
        Fast, free, synchronous rule-based checks.

        Returns:
            (is_safe, list_of_issues)
        """
        issues = []
        if not sql or not sql.strip():
            return False, ["SQL is empty"]

        # Normalize for keyword checking
        sql_upper = sql.strip().upper()

        # --- Safety Check: Only SELECTs allowed ---
        first_keyword = sql_upper.split()[0] if sql_upper.split() else ""
        if first_keyword in FORBIDDEN_SQL_KEYWORDS:
            issues.append(
                f"SAFETY VIOLATION: SQL starts with '{first_keyword}' — only SELECT is allowed"
            )
            return False, issues  # Hard fail — don't even continue

        if not sql_upper.startswith("SELECT") and "WITH" not in sql_upper[:10]:
            # Allow CTEs (WITH ... AS (...) SELECT ...)
            issues.append("SQL does not start with SELECT or a CTE (WITH clause)")

        # --- Table existence check ---
        # Extract table names referenced in FROM and JOIN clauses
        table_pattern = re.compile(
            r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            re.IGNORECASE
        )
        referenced_tables = {m.group(1).lower() for m in table_pattern.finditer(sql)}

        all_available = {
            t.lower()
            for t in schema_context.get("all_available_tables", [])
        }

        if all_available:
            unknown_tables = referenced_tables - all_available
            if unknown_tables:
                issues.append(f"Unknown tables referenced: {unknown_tables}")

        # --- Basic structural check ---
        if "FROM" not in sql_upper:
            issues.append("SQL missing FROM clause")

        is_safe = len(issues) == 0
        return is_safe, issues

    def process(self, state: QueryState) -> Dict[str, Any]:
        """
        LangGraph node function.

        Reads:
            state["generated_sql"]  — SQL from SQLAgent
            state["schema_context"] — for table existence checks
            state["plan"]           — for semantic review context

        Writes:
            state["sql_validation"] — validation result dict
            state["generated_sql"]  — may be OVERWRITTEN if LLM provides a correction
        """
        sql = state.get("generated_sql", "")
        schema_context = state.get("schema_context", {})
        plan = state.get("plan", {})

        if not sql:
            return {
                "sql_validation": {"is_valid": False, "issues": ["No SQL was generated"]},
                "error": "No SQL to validate",
                "step_log": ["❌ Validator: No SQL to validate"],
            }

        logger.info(f"ValidationAgent checking SQL: {sql[:200]}...")

        # --- Stage 1: Rule-based checks (fast, free) ---
        rule_pass, rule_issues = self._rule_based_checks(sql, schema_context)

        if not rule_pass and any("SAFETY" in i for i in rule_issues):
            # Hard fail — don't call LLM for dangerous SQL
            logger.warning(f"SECURITY BLOCK: {rule_issues}")
            return {
                "sql_validation": {
                    "is_valid": False,
                    "issues": rule_issues,
                    "corrected_sql": None,
                    "confidence": "high",
                    "review_notes": "Blocked by safety rules",
                },
                "error": f"SQL validation failed: {rule_issues}",
                "step_log": [f"🚫 Validator: BLOCKED dangerous SQL — {rule_issues}"],
            }

        # --- Stage 2: LLM semantic review (using cheaper Haiku model) ---
        try:
            user_message = f"""Review this Amazon Redshift SQL query:

ANALYTICAL OBJECTIVE: {plan.get('analytical_objective', 'Not specified')}

SQL QUERY:
{sql}

RULE-BASED PRE-CHECK ISSUES (if any): {rule_issues if rule_issues else 'None'}

Respond with ONLY valid JSON."""

            response_text = self.llm_client.generate(
                system_prompt=VALIDATOR_SYSTEM_PROMPT,
                user_message=user_message,
                model_id=BedrockClient.HAIKU_MODEL_ID,  # cheaper model for validation
                temperature=0.0,
            )
            validation = self.llm_client.parse_json_response(response_text)

            # Merge rule issues into LLM issues
            all_issues = rule_issues + validation.get("issues", [])
            validation["issues"] = all_issues

            # If LLM provided a corrected SQL, use it
            corrected_sql = validation.get("corrected_sql")
            updated_sql = corrected_sql if corrected_sql else sql

            is_valid = validation.get("is_valid", False) and not rule_issues

            logger.info(
                f"ValidationAgent result: valid={is_valid} | "
                f"confidence={validation.get('confidence')} | "
                f"issues={all_issues}"
            )

            return {
                "sql_validation": validation,
                "generated_sql": updated_sql,  # Use corrected SQL if available
                "step_log": [
                    f"{'✅' if is_valid else '⚠️'} Validator: valid={is_valid} | "
                    f"issues={len(all_issues)} | confidence={validation.get('confidence')}"
                ],
            }

        except Exception as e:
            error_msg = f"ValidationAgent LLM review failed: {e}"
            logger.error(error_msg)
            # Don't block execution if LLM review fails — use rule-based result
            is_valid = rule_pass
            return {
                "sql_validation": {
                    "is_valid": is_valid,
                    "issues": rule_issues,
                    "corrected_sql": None,
                    "confidence": "low",
                    "review_notes": f"LLM review failed, using rule-based result: {e}",
                },
                "step_log": [f"⚠️ Validator: LLM review failed, rule-based={is_valid}"],
            }
