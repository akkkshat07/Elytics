from __future__ import annotations
import json
from sqlalchemy import text
from agents.live_sql.sql_executor import safe_sql_execute
from agents.live_sql.sql_generator import generate_sql

async def _load_schema(session, db_type: str='postgres', schema_name: str='public') -> dict:
    if db_type.lower() == 'mysql':
        schema_sql = text('\n            SELECT table_name, column_name\n            FROM information_schema.columns\n            WHERE table_schema = DATABASE()\n            ORDER BY table_name, ordinal_position\n            ')
        result = await session.execute(schema_sql)
        rows = result.fetchall()
    else:
        schema_sql = text('\n            SELECT table_name, column_name\n            FROM information_schema.columns\n            WHERE table_schema = :schema_name\n            ORDER BY table_name, ordinal_position\n            ')
        result = await session.execute(schema_sql, {'schema_name': schema_name})
        rows = result.fetchall()
    tables: dict[str, list[str]] = {}
    for table_name, column_name in rows:
        tables.setdefault(str(table_name), []).append(str(column_name))
    return {'tables': tables}

async def execute_with_retries(user_query: str, session_factory, llm_complete, db_type: str='postgres', schema_name: str='public', max_retries: int=2, initial_query: str | None=None) -> dict:
    async with session_factory() as session:
        schema = await _load_schema(session, db_type=db_type, schema_name=schema_name)
        query = (initial_query or '').strip()
        if not query:
            query = await generate_sql(user_query=user_query, schema=schema, llm_complete=llm_complete)
        result = await safe_sql_execute(query=query, session=session)
    attempt = 0
    while result.get('status') != 'success' and attempt < max_retries:
        attempt += 1
        repair_prompt = f"Fix the SQL. Return ONLY corrected SELECT SQL using provided schema.\n\nSCHEMA_JSON:\n{json.dumps(schema, ensure_ascii=False)}\n\nUSER_QUERY:\n{user_query}\n\nFAILED_QUERY:\n{result.get('query')}\n\nERROR:\n{result.get('error')}"
        query = str(await llm_complete(repair_prompt) or '').replace('```sql', '').replace('```', '').strip().strip('`')
        async with session_factory() as session:
            result = await safe_sql_execute(query=query, session=session)
    result['attempts'] = attempt
    return result