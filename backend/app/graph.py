import logging
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from .state import QueryState
from .utils.bedrock import BedrockClient
from .utils.redshift import RedshiftClient
from .agents.guardrail import GuardrailAgent
from .agents.router import RouterAgent
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
  guardrail_agent = GuardrailAgent()
  router_agent = RouterAgent(llm_client=llm_client)
  schema_agent = SchemaAgent(llm_client=llm_client, redshift_client=redshift_client)
  sql_agent = SQLAgent(llm_client=llm_client, redshift_client=redshift_client)
  python_agent = PythonAgent(llm_client=llm_client)
  executor_agent = ExecutorAgent()
  insights_agent = InsightsAgent(llm_client=llm_client)
  workflow = StateGraph(QueryState)
  workflow.add_node('guardrail', guardrail_agent.process)
  workflow.add_node('router', router_agent.process)
  workflow.add_node('schema', schema_agent.process)
  workflow.add_node('sql', sql_agent.process)
  workflow.add_node('python_code', python_agent.process)
  workflow.add_node('executor', executor_agent.process)
  workflow.add_node('insights', insights_agent.process)
  workflow.set_entry_point('guardrail')

  def route_after_guardrail(state: QueryState) -> Literal['router', '__end__']:
    if state.get('guardrail_status') == 'rejected':
      logger.warning('Routing: guardrail REJECTED → END')
      return '__end__'
    return 'router'

  def route_after_sql(state: QueryState) -> Literal['python_code', '__end__']:
    has_error = bool(state.get('error', ''))
    if not has_error:
      return 'python_code'
    else:
      logger.warning(f'Routing: sql FAILED → END | has_error={has_error}')
      return '__end__'

  def route_after_executor(state: QueryState) -> Literal['python_code', 'insights', '__end__']:
    has_error = bool(state.get('error', ''))
    attempts = state.get('python_attempts', 0)
    if has_error and attempts < 3:
      logger.warning(f'Routing: executor FAILED → Retrying python_code | Attempt {attempts}/3')
      return 'python_code'
    elif has_error:
      logger.error('Routing: executor FAILED → Insights (Fallback)')
      return 'insights'
    return 'insights'

  workflow.add_conditional_edges('guardrail', route_after_guardrail, {'router': 'router', '__end__': END})
  workflow.add_edge('router', 'schema')
  workflow.add_edge('schema', 'sql')
  workflow.add_conditional_edges('sql', route_after_sql, {'python_code': 'python_code', '__end__': END})
  workflow.add_edge('python_code', 'executor')
  workflow.add_conditional_edges('executor', route_after_executor, {'python_code': 'python_code', 'insights': 'insights', '__end__': END})
  workflow.add_edge('insights', END)
  compiled = workflow.compile()
  logger.info('LangGraph v3 compiled successfully (CoreSight Upgrade)')
  return compiled
try:
  analytics_graph = build_graph()
  logger.info('analytics_graph ready')
except Exception as e:
  logger.warning(f'analytics_graph could not be built at startup: {e}. Will retry on request.')
  analytics_graph = None