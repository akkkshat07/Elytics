from __future__ import annotations
from dataclasses import dataclass
from sqlalchemy import text

@dataclass
class SQLValidationResult:
    valid: bool
    error: str | None = None

async def validate_sql_with_explain(query: str, session) -> SQLValidationResult:
    q = (query or '').strip().rstrip(';')
    if not q:
        return SQLValidationResult(valid=False, error='Empty SQL query')
    if not q.lower().startswith('select'):
        return SQLValidationResult(valid=False, error='Only SELECT queries are allowed')
    try:
        await session.execute(text('EXPLAIN ' + q))
        return SQLValidationResult(valid=True, error=None)
    except Exception as exc:
        return SQLValidationResult(valid=False, error=str(exc))