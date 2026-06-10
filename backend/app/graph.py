import logging
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from .state import QueryState
from .utils.bedrock import BedrockClient
from .utils.redshift import RedshiftClient
from .agents.planner import PlannerAgent
from .agents.schema import SchemaAgent
from .agents.sql import SQLAgent
from .agents.validator import ValidationAgent
from .agents.python_agent import PythonAgent
from .agents.executor import ExecutorAgent
from .agents.insights import InsightsAgent
logger = logging.getLogger(__name__)

def build_graph():
    llm_client = BedrockClient()
    redshift_client = RedshiftClient()
    planner_agent = PlannerAgent(llm_client=llm_client)
    schema_agent = SchemaAgent(llm_client=llm_client, redshift_client=redshift_client)
    sql_agent = SQLAgent(llm_client=llm_client)
    validator_agent = ValidationAgent(llm_client=llm_client)
    python_agent = PythonAgent(llm_client=llm_client)
    executor_agent = ExecutorAgent()
    insights_agent = InsightsAgent(llm_client=llm_client)

    def redshift_query_node(state: QueryState) -> Dict[str, Any]:
        sql = state.get('generated_sql', '')
        logger.info(f'redshift_query_node: Executing SQL | {sql[:150]}...')
        try:
            rows = redshift_client.execute_query(sql)
            logger.info(f'Redshift returned {len(rows)} rows')
            return {'query_results': rows, 'step_log': [f'✅ Redshift Query: {len(rows)} rows returned']}
        except RuntimeError as e:
            error_msg = f'Redshift execution failed: {e}'
            logger.error(error_msg)
            return {'query_results': [], 'error': error_msg, 'step_log': [f'❌ Redshift Query: {e}']}
    workflow = StateGraph(QueryState)
    workflow.add_node('planner', planner_agent.process)
    workflow.add_node('schema', schema_agent.process)
    workflow.add_node('sql', sql_agent.process)
    workflow.add_node('validator', validator_agent.process)
    workflow.add_node('redshift_query', redshift_query_node)
    workflow.add_node('python_code', python_agent.process)
    workflow.add_node('executor', executor_agent.process)
    workflow.add_node('insights', insights_agent.process)
    workflow.set_entry_point('planner')
    workflow.add_edge('planner', 'schema')
    workflow.add_edge('schema', 'sql')
    workflow.add_edge('sql', 'validator')

    def route_after_validation(state: QueryState) -> Literal['redshift_query', '__end__']:
        validation = state.get('sql_validation', {})
        is_valid = validation.get('is_valid', False)
        has_error = bool(state.get('error', ''))
        has_sql = bool(state.get('generated_sql', '').strip())
        if is_valid and has_sql and (not has_error):
            logger.info('Routing: validator PASSED → redshift_query')
            return 'redshift_query'
        else:
            logger.warning(f'Routing: validator FAILED → END | is_valid={is_valid} | has_sql={has_sql} | has_error={has_error}')
            return '__end__'
    workflow.add_conditional_edges('validator', route_after_validation, {'redshift_query': 'redshift_query', '__end__': END})
    workflow.add_edge('redshift_query', 'python_code')
    workflow.add_edge('python_code', 'executor')
    workflow.add_edge('executor', 'insights')
    workflow.add_edge('insights', END)
    compiled = workflow.compile()
    logger.info('LangGraph v2 compiled successfully | pipeline: planner→schema→sql→validator→redshift→python_code→executor→insights')
    return compiled
try:
    analytics_graph = build_graph()
    logger.info('analytics_graph ready')
except Exception as e:
    logger.warning(f'analytics_graph could not be built at startup (likely missing credentials): {e}. Will retry on first /api/query request.')
    analytics_graph = None