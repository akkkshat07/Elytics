from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Dict
from config.system_config import REDIS_URL, ML_KEYWORDS
from services.session_memory import session_memory
from util.graph import State, coder_node, narrator_node, _mark_node_latency, _check_for_cancellation, _assert_llm_client
logger = logging.getLogger(__name__)
_DS_KEYWORDS = set(ML_KEYWORDS)

async def adhoc_router_node(state: State) -> Dict[str, Any]:
    _check_for_cancellation(state, 'adhoc_router_node')
    node_start = time.perf_counter()
    user_query = state.get('input', '')
    session_id = state.get('session_id', '')
    query_lower = user_query.lower()
    is_ds = any((kw in query_lower for kw in _DS_KEYWORDS))
    adhoc_file = state.get('adhoc_file_metadata', {})
    file_names = adhoc_file.get('file_names', ['uploaded_data'])
    enhanced = user_query
    last_ctx = session_memory.get_last_context(session_id)
    if last_ctx and last_ctx.get('previous_enhanced_question'):
        enhanced = f"Follow-up to: {last_ctx['previous_enhanced_question']}\n\nCurrent question: {user_query}"
    out = {'route_decision': 'simple', 'query_routing_type': 'data_scientist' if is_ds else 'data_analyst', 'relevant_tables': file_names, 'table_count': len(file_names), 'table_briefs': {name: f'User-uploaded file: {name}' for name in file_names}, 'enhanced_question': enhanced, 'cache_hit': False, 'guard_status': 'relevant', 'terminate_graph': False, 'adhoc_mode': True, 'data_science_mode': is_ds, 'data_analyst_mode': not is_ds}
    out.update(_mark_node_latency(state, 'adhoc_router', node_start))
    logger.info('Adhoc router | session=%s | routing=%s | files=%s | enhanced_len=%d', session_id, 'DS' if is_ds else 'DA', file_names, len(enhanced))
    return out

def build_adhoc_graph_app():
    try:
        from langgraph.graph import StateGraph, END
        RedisSaver = None
        try:
            from langgraph.checkpoint.redis import RedisSaver as _RS
            RedisSaver = _RS
        except Exception:
            try:
                from langgraph.checkpoint import RedisSaver as _RS
                RedisSaver = _RS
            except Exception:
                RedisSaver = None
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except Exception:
            SqliteSaver = None
        try:
            from langgraph.checkpoint.memory import MemorySaver
        except Exception:
            MemorySaver = None
        try:
            from langchain_core.runnables import RunnableLambda
        except Exception as e:
            raise RuntimeError(f'LangChain core not available: {e}')
        try:
            import redis.asyncio as redis
        except Exception:
            import redis
    except Exception as e:
        raise RuntimeError(f'LangGraph/Redis not available: {e}. Install deps.')
    graph = StateGraph(State)
    from util.guard_node import guard_node
    graph.add_node('guard', RunnableLambda(guard_node).with_config(run_name='guard_node'))
    graph.add_node('adhoc_router', RunnableLambda(adhoc_router_node).with_config(run_name='adhoc_router_node'))
    graph.add_node('coder', RunnableLambda(coder_node).with_config(run_name='coder_node'))
    graph.add_node('narrator', RunnableLambda(narrator_node).with_config(run_name='narrator_node'))
    graph.set_entry_point('guard')

    def guard_decision(state: State) -> str:
        if state.get('terminate_graph'):
            logger.info('Adhoc guard decision: irrelevant → narrator')
            return 'narrator'
        return 'adhoc_router'
    graph.add_conditional_edges('guard', guard_decision, {'narrator': 'narrator', 'adhoc_router': 'adhoc_router'})
    graph.add_edge('adhoc_router', 'coder')

    def coder_router(state: State) -> str:
        err = (state.get('executor_error_text') or '').strip()
        attempts = int(state.get('execution_attempts') or 0)
        max_retries = 2
        if err and attempts < max_retries:
            logger.info('Adhoc coder retry: attempt %d (error: %s)', attempts + 1, err[:80])
            state['execution_attempts'] = attempts + 1
            return 'coder'
        return 'narrator'
    graph.add_conditional_edges('coder', coder_router, {'coder': 'coder', 'narrator': 'narrator'})
    graph.add_edge('narrator', END)
    import os
    checkpointer = None
    try:
        if 'redis' in dir() and RedisSaver is not None:
            redis_client = redis.from_url(REDIS_URL)
            checkpointer = RedisSaver(redis_client)
    except Exception as e:
        logger.warning('Adhoc graph: Redis checkpointer unavailable: %s', e)
    if checkpointer is None and SqliteSaver is not None:
        try:
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'data')
            os.makedirs(data_dir, exist_ok=True)
            sqlite_path = os.path.abspath(os.path.join(data_dir, 'adhoc_state.sqlite'))
            checkpointer = SqliteSaver(sqlite_path)
        except Exception as e:
            logger.warning('Adhoc graph: SqliteSaver unavailable: %s', e)
    if checkpointer is None and MemorySaver is not None:
        checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    logger.info('Ad-hoc LangGraph compiled (guard → adhoc_router → coder → narrator).')
    return app