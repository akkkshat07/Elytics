from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple
from langchain_core.runnables import RunnableConfig
logger = logging.getLogger(__name__)
_GUARD_SYSTEM_PROMPT = "You are a classification gate for a business data analysis assistant.\n\nDecide: is this query asking to ANALYZE DATA, or is it something else entirely?\n\nRELEVANT — the query involves analyzing, querying, or exploring data:\n  - Mentions business terms: inventory, sales, revenue, orders, customers, vendors, products, items, costs, regions, orgs\n  - Asks analytical questions: 'how many', 'show me', 'compare', 'top N', 'total', 'trend', 'average', 'breakdown', 'summary', 'count'\n  - Contains data-related nouns even with typos: 'homany items' = 'how many items' → RELEVANT\n  - Operations on DATA (records, rows, tables): 'delete records from Q1', 'filter by region' → RELEVANT\n  - When unsure, lean RELEVANT\n\nIRRELEVANT — the query has ZERO data/business signals:\n  - Greetings: 'hi', 'hello', 'good morning'\n  - Jokes/entertainment: 'tell me a joke'\n  - General knowledge: 'capital of France', 'who is the president'\n  - System/code operations (targeting code, servers, apps — NOT data): 'delete the code', 'fix the backend', 'restart the server', 'deploy the app', 'write python code'\n  - NOTE: 'delete the code at the backend' is IRRELEVANT (targets code, not data). 'delete old records' is RELEVANT (targets data).\n\nKEY DISTINCTION: Does the query target DATA/RECORDS/TABLES or CODE/SERVERS/SYSTEMS? Data operations = RELEVANT. System operations = IRRELEVANT."
_GUARD_USER_TEMPLATE = 'User query: "{query}"\n\nRespond with EXACTLY one line:\nRELEVANT\nor\nIRRELEVANT: <one-sentence reason>'

async def _lightweight_llm_guard(user_query: str, llm_client, client_id: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    try:
        result = await llm_client.generate_completion(system_prompt=_GUARD_SYSTEM_PROMPT, user_message=_GUARD_USER_TEMPLATE.format(query=user_query), max_tokens=50, temperature=0, reasoning_effort='none')
        usage = result.get('usage') if isinstance(result.get('usage'), dict) else None
        text = (result.get('content') or '').strip()
        logger.info('Guard LLM raw response: %r for query: %r', text, user_query)
        if not text:
            logger.warning('Lightweight LLM guard returned empty — allowing through')
            return (True, '', usage)
        text_upper = text.upper().strip()
        is_irrelevant = text_upper.startswith('IR') or 'IRRELEVANT' in text_upper
        if is_irrelevant:
            reason = ''
            if ':' in text:
                reason = text.split(':', 1)[1].strip()
            msg = "I'm a data intelligence assistant. I can only help with questions about your business data. Could you ask a data-related question?"
            if reason:
                msg = f"I'm a data intelligence assistant. I can only help with data analysis questions. {reason}"
            logger.info('Guard LLM classified as IRRELEVANT (raw=%r)', text)
            return (False, msg, usage)
        logger.info('Guard LLM classified as RELEVANT (raw=%r)', text)
        return (True, '', usage)
    except Exception as e:
        logger.warning('Lightweight LLM guard failed, allowing query through: %s', e)
        return (True, '', None)

async def guard_node(state: Dict[str, Any], config: RunnableConfig | None=None) -> Dict[str, Any]:
    from util.graph import _check_for_cancellation, _mark_node_latency, _assert_llm_client
    _check_for_cancellation(state, 'guard_node')
    node_start = time.perf_counter()
    user_query = state.get('input', '')
    client_id = state.get('client_id', '')
    run_id = state.get('run_id', '')
    from util.guardrails import classify_question
    from db_config.database import get_db
    db = get_db()
    guard_result = await classify_question(user_query, client_id=client_id, db=db)
    if not guard_result.is_relevant:
        logger.info('Guard L1 blocked: category=%s reason=%s', guard_result.category, guard_result.reason)
        out = {'guard_status': 'irrelevant', 'guard_reason': guard_result.reason, 'guard_category': guard_result.category, 'terminate_graph': True, 'cache_hit': False, 'route_decision': 'irrelevant', 'executor_response': {}, 'executor_agent_task_id': '', 'executor_error_text': '', 'business_response': {'analysis': guard_result.user_message, 'follow_ups': []}, 'business_agent_task_id': f'business_guard_{run_id}'}
        out.update(_mark_node_latency(state, 'guard', node_start))
        return out
    llm_client = _assert_llm_client(state, 'guard_node', config)
    is_relevant, guard_message, guard_usage = await _lightweight_llm_guard(user_query, llm_client, client_id)
    agent_token_usage = dict(state.get('agent_token_usage') or {})
    if guard_usage:
        agent_token_usage['guard'] = guard_usage
    if not is_relevant:
        logger.info('Guard L1.5 (LLM) blocked query')
        out = {'guard_status': 'irrelevant', 'guard_reason': 'Lightweight LLM guard (Layer 1.5)', 'guard_category': 'llm_guard', 'terminate_graph': True, 'cache_hit': False, 'route_decision': 'irrelevant', 'executor_response': {}, 'executor_agent_task_id': '', 'executor_error_text': '', 'business_response': {'analysis': guard_message, 'follow_ups': []}, 'business_agent_task_id': f'business_guard_llm_{run_id}', 'agent_token_usage': agent_token_usage}
        out.update(_mark_node_latency(state, 'guard', node_start))
        return out
    logger.info('Guard passed — query is relevant')
    _fire_kernel_prewarm(state)
    out = {'guard_status': 'relevant', 'guard_category': 'relevant', 'guard_reason': '', 'terminate_graph': False, 'route_decision': '', 'cache_hit': False, 'executor_response': {}, 'executor_agent_task_id': '', 'executor_error_text': '', 'business_response': {}, 'business_agent_task_id': '', 'agent_token_usage': agent_token_usage}
    out.update(_mark_node_latency(state, 'guard', node_start))
    return out

def _fire_kernel_prewarm(state: Dict[str, Any]) -> None:
    session_id = state.get('session_id', '')
    client_id = state.get('client_id', '')
    if not session_id or not client_id:
        return
    try:
        from config.system_config import AGENT_CONFIG
        from util import session_kernel_store
        idle_timeout = float(AGENT_CONFIG.get('data_science_agent', {}).get('idle_timeout_minutes', 30.0))
        task = asyncio.create_task(session_kernel_store._prewarm_session_kernel(session_id=session_id, client_id=client_id, idle_timeout_minutes=idle_timeout), name=f'prewarm_kernel_{session_id}')
        session_kernel_store.register_prewarm_task(session_id, task)
        logger.debug('Kernel pre-warm task fired for session=%s', session_id)
    except Exception as exc:
        logger.warning('Could not fire kernel pre-warm for session=%s: %s', session_id, exc)