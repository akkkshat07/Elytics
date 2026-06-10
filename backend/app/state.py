from typing import Dict, List, Any, TypedDict, Annotated
import operator

class QueryState(TypedDict):
    user_query: str
    intent: str
    plan: Dict[str, Any]
    schema_context: Dict[str, Any]
    generated_sql: str
    sql_validation: Dict[str, Any]
    query_results: List[Dict[str, Any]]
    generated_python: str
    execution_results: Dict[str, Any]
    charts: List[Dict[str, Any]]
    insights: List[str]
    error: str
    step_log: Annotated[List[str], operator.add]