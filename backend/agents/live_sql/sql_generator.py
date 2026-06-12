from __future__ import annotations
import json
from dataclasses import dataclass

@dataclass
class SQLGeneratorInput:
    user_query: str
    schema: dict

async def generate_sql(user_query: str, schema: dict, llm_complete) -> str:
    payload = SQLGeneratorInput(user_query=user_query, schema=schema)
    prompt = f'Return ONLY one valid SELECT SQL statement. Use ONLY tables/columns present in schema JSON. No markdown, no explanation.\n\nSCHEMA_JSON:\n{json.dumps(payload.schema, ensure_ascii=False)}\n\nUSER_QUERY:\n{payload.user_query}'
    sql = str(await llm_complete(prompt) or '').strip()
    sql = sql.replace('```sql', '').replace('```', '').strip().strip('`')
    if sql.lower().startswith('sql '):
        sql = sql[4:].strip()
    return sql