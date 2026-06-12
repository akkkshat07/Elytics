from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sqlalchemy import text
from agents.live_sql.sql_validator import validate_sql_with_explain

@dataclass
class SQLExecutionResult:
    status: str
    dataframe: pd.DataFrame | None
    error: str | None
    query: str

async def safe_sql_execute(query: str, session) -> dict:
    q = (query or '').strip().rstrip(';')
    validation = await validate_sql_with_explain(q, session)
    if not validation.valid:
        return SQLExecutionResult(status='error', dataframe=None, error=validation.error, query=q).__dict__
    try:
        result = await session.execute(text(q))
        rows = result.mappings().all()
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        df.columns = [str(c).strip().lower() for c in df.columns]
        return SQLExecutionResult(status='success', dataframe=df, error=None, query=q).__dict__
    except Exception as exc:
        return SQLExecutionResult(status='error', dataframe=None, error=str(exc), query=q).__dict__