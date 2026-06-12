from __future__ import annotations
import asyncio
import json as _json_mod
import traceback
from typing import Any, Dict, Optional, TypedDict, AsyncGenerator, List, TYPE_CHECKING
import os
import logging
import time
from pathlib import Path
from util.dataset_paths import resolve_xml_data_sources_dir
from services.graph_result_registry import peek_graph_final_payload, record_graph_final_payload
from config.system_config import USE_LANGGRAPH, REDIS_URL, should_use_knowledge_filtering, AGENT_CONFIG
from util.cancellation import cancellation_manager
from langchain_core.runnables import RunnableConfig
from langsmith import traceable
from util.llm_utils import LLMClient
logger = logging.getLogger(__name__)
_RESPONSE_MATCHER_CACHE: Dict[str, Dict[str, Any]] = {}
_RESPONSE_MATCHER_CACHE_TTL_SEC = 300.0

def _mark_node_latency(state: 'State', node_name: str, start_ts: float) -> Dict[str, Any]:
    elapsed_ms = round((time.perf_counter() - start_ts) * 1000.0, 2)
    latencies = dict(state.get('graph_latency_ms') or {})
    latencies[node_name] = elapsed_ms
    logger.info('[Latency] node=%s elapsed_ms=%.2f', node_name, elapsed_ms)
    return {'graph_latency_ms': latencies}

def _get_cached_response_matcher(client_id: str, dataset_id: str=''):
    from response_caching.response_matcher import ResponseMatcher
    cache_key = f"{client_id}:{dataset_id or ''}"
    now = time.time()
    cached = _RESPONSE_MATCHER_CACHE.get(cache_key)
    if cached and now - cached.get('created_at', 0.0) < _RESPONSE_MATCHER_CACHE_TTL_SEC:
        return cached['matcher']
    matcher = ResponseMatcher(client_id=client_id, dataset_id=dataset_id or None)
    matcher.initialize()
    _RESPONSE_MATCHER_CACHE[cache_key] = {'matcher': matcher, 'created_at': now}
    return matcher

def _check_for_cancellation(state: 'State', node_name: str):
    session_id = state.get('session_id')
    if cancellation_manager.is_cancelled(session_id):
        logger.warning(f'Cancellation signal received before executing {node_name}.')
        raise asyncio.CancelledError(f'Stream stopped by user request during {node_name}.')

def _convert_records_to_split_json(records: List[Dict[str, Any]], name: str='_generated_dataframe_') -> Dict[str, Any]:
    import pandas as pd
    try:
        df = pd.DataFrame(records)
        return {'name': name, 'json_data': df.to_json(orient='split', date_format='iso')}
    except Exception as e:
        logger.warning(f"Failed to convert records to split JSON for '{name}': {e}")
        if records:
            columns = list(records[0].keys())
            data = [[r.get(c) for c in columns] for r in records]
        else:
            columns, data = ([], [])
        return {'name': name, 'json_data': _json_mod.dumps({'columns': columns, 'data': data})}

def _rbac_access_denied_state(state: 'State', *, violated_tables: List[str] | None=None, message: str='This data is not accessible to you.') -> Dict[str, Any]:
    violated = violated_tables or []
    run_id = state.get('run_id', '') if isinstance(state, dict) else ''
    return {'executor_response': {'console_output': '', 'dataframes': [], 'plotly_charts': [], 'access_denied_message': message, 'denied_tables_violated': violated}, 'executor_agent_task_id': f'executor_rbac_denied_{run_id}', 'executor_error_text': '', 'business_response': {'analysis': message}, 'business_agent_task_id': f'business_rbac_denied_{run_id}', 'terminate_graph': True, 'cache_hit': False}

class State(TypedDict, total=False):
    user_id: str
    input: str
    session_id: str
    run_id: str
    client_id: str
    dataset_id: str
    datasource_context: Dict[str, Any]
    route_decision: str
    relevant_tables: List[str]
    table_count: int
    table_briefs: Dict[str, str]
    query_routing_type: str
    enhanced_question: str
    scout_results: Dict[str, Dict]
    scout_errors: Dict[str, str]
    data_brief: str
    identified_joins: List[Dict]
    code: str
    executor_response: Any
    executor_error_text: str
    data_science_mode: bool
    data_analyst_mode: bool
    business_response: Any
    summarized_response: Any
    router_task_id: str
    coder_task_id: str
    executor_agent_task_id: str
    business_agent_task_id: str
    data_science_agent_task_id: str
    execution_attempts: int
    error_context: str
    terminate_graph: bool
    cache_hit: bool
    planner_reference: str
    python_reference: str
    business_reference: str
    guard_status: str
    guard_reason: str
    guard_category: str
    semantic_signature: Dict[str, Any]
    best_candidate_info: Dict[str, Any]
    semantic_cache_match: Dict[str, Any]
    denied_tables_cache: List[str]
    all_tables_cache: List[str]
    adhoc_mode: bool
    adhoc_file_metadata: Dict[str, Any]
    adhoc_dataset_path: str
    agent_inputs: Dict[str, Any]
    agent_token_usage: Dict[str, Any]
    graph_latency_ms: Dict[str, float]
    resolved_prompts: Dict[str, str]
    active_persona: Dict[str, Any]

def _assert_llm_client(state: 'State', node_name: str, config: RunnableConfig | None=None) -> 'LLMClient':
    runtime_config = config or {}
    llm_client = runtime_config.get('configurable', {}).get('llm_client')
    if llm_client is not None:
        return llm_client
    try:
        from langchain_core.runnables.config import var_child_runnable_config
        child_config = var_child_runnable_config.get({})
        llm_client = child_config.get('configurable', {}).get('llm_client')
        if llm_client is not None:
            return llm_client
    except (ImportError, LookupError):
        pass
    llm_client = state.get('llm_client')
    if llm_client is None:
        raise ValueError(f"llm_client is REQUIRED for {node_name}. Pass it via config['configurable']['llm_client'] or state['llm_client'].")
    return llm_client

@traceable(name='router_node')
async def router_node(state: State, config: RunnableConfig | None=None) -> Dict[str, Any]:
    _check_for_cancellation(state, 'router_node')
    llm_client = _assert_llm_client(state, 'router_node', config)
    node_start = time.perf_counter()
    client_id = state.get('client_id')
    if not client_id:
        raise ValueError('client_id is REQUIRED in state for multi-tenant operation')
    run_id = state.get('run_id', '')
    user_question = state.get('input', '')
    from db_config.database import get_db, get_mongo_manager
    db = get_db()
    mongo_manager = get_mongo_manager()
    from util.xml_prompt_loader import load_client_prompt
    resolved_prompts: Dict[str, str] = dict(state.get('resolved_prompts') or {})
    denied_tables_cache = set(state.get('denied_tables_cache') or [])
    all_tables_cache = list(state.get('all_tables_cache') or [])

    async def _resolve_prompts() -> Dict[str, str]:
        if resolved_prompts:
            return resolved_prompts
        _prompt_paths = ['agents/router.xml', 'agents/data_science_agent.xml', 'agents/data_analyst_agent.xml', 'agents/business.xml']
        _results = await asyncio.gather(*[load_client_prompt(p, client_id, mongo_manager.db, use_formatting=False) for p in _prompt_paths], return_exceptions=True)
        _name_map = ['router', 'data_science_agent', 'data_analyst_agent', 'business']
        out: Dict[str, str] = {}
        for _name, _res in zip(_name_map, _results):
            if isinstance(_res, Exception):
                logger.warning("Failed to resolve prompt for '%s': %s", _name, _res)
            else:
                out[_name] = _res
        return out

    async def _resolve_rbac() -> tuple:
        _denied = set(denied_tables_cache)
        _all_tables = list(all_tables_cache)
        try:
            from services.table_permissions_service import get_denied_tables_for_user
            from db_config.mongo_server import get_db as get_mongo_db
            from util.knowledge_filter import extract_table_names_from_table_introductions
            from util.xml_prompt_loader import load_xml_prompt_raw
            _async_db = await get_mongo_db()
            if not _denied:
                _denied = await get_denied_tables_for_user(state.get('user_id', 'anonymous'), client_id, _async_db, dataset_id=state.get('dataset_id') or None)
            if not _all_tables:
                _ds = state.get('dataset_id') or ''
                intro_path = resolve_xml_data_sources_dir(client_id, _ds or None) / 'meta_information' / 'table_introductions.xml'
                if intro_path.exists():
                    table_intro_xml = load_xml_prompt_raw(intro_path)
                    _all_tables = extract_table_names_from_table_introductions(table_intro_xml)
        except Exception as rbac_err:
            logger.error(f'[RBAC] Error loading tables: {rbac_err}', exc_info=True)
        return (_denied, _all_tables)

    async def _resolve_session_context() -> tuple:
        _conversation_context = None
        _active_persona: Dict[str, Any] | None = None
        try:
            from services.session_memory import session_memory
            last_ctx = session_memory.get_last_context(state.get('session_id', ''))
            if last_ctx:
                _conversation_context = {'previous_enhanced_question': last_ctx.get('previous_enhanced_question', ''), 'previous_plan': last_ctx.get('previous_plan', '')}
            locked = session_memory.get_persona(state.get('session_id', ''))
            if locked and locked.get('slug'):
                from services.persona_service import resolve_persona_for_session
                from db_config.mongo_server import get_db as get_mongo_db
                _async_db = await get_mongo_db()
                _active_persona = await resolve_persona_for_session(locked['slug'], client_id, _async_db)
                if _active_persona:
                    logger.info('Persona resolved | session=%s | persona=%s', state.get('session_id'), _active_persona.get('slug'))
        except Exception as _persona_err:
            logger.warning('Persona resolution failed (non-fatal): %s', _persona_err)
        return (_conversation_context, _active_persona)

    async def _resolve_subscription() -> Dict[str, Any]:
        try:
            from services.subscription_service import get_client_subscription
            from db_config.mongo_server import get_db as get_mongo_db
            _async_db = await get_mongo_db()
            return await get_client_subscription(client_id, _async_db)
        except Exception as sub_err:
            logger.warning(f'Subscription pre-fetch failed: {sub_err}')
            return {}
    resolved_prompts, (denied_tables_cache, all_tables_cache), (conversation_context, active_persona), subscription = await asyncio.gather(_resolve_prompts(), _resolve_rbac(), _resolve_session_context(), _resolve_subscription())
    if denied_tables_cache and all_tables_cache:
        from services.table_permissions_service import get_allowed_tables
        allowed = get_allowed_tables(all_tables_cache, denied_tables_cache)
        if len(allowed) == 0:
            logger.warning(f"[RBAC] Zero allowed tables for user '{state.get('user_id')}' client '{client_id}'")
            out = _rbac_access_denied_state(state)
            out['route_decision'] = 'irrelevant'
            out['denied_tables_cache'] = sorted(denied_tables_cache)
            out['all_tables_cache'] = sorted(all_tables_cache)
            out['resolved_prompts'] = resolved_prompts
            out.update(_mark_node_latency(state, 'router', node_start))
            return out
    from agents.router_agent import RouterAgent
    agent = RouterAgent(client_id=client_id, db=mongo_manager, llm_client=llm_client, resolved_prompt=resolved_prompts.get('router'), dataset_id=state.get('dataset_id') or None, denied_table_keys=set(denied_tables_cache) if denied_tables_cache else set())
    from services.table_permissions_service import get_allowed_tables
    if denied_tables_cache and all_tables_cache:
        _allowed_tables = get_allowed_tables(all_tables_cache, denied_tables_cache)
    else:
        _allowed_tables = list(all_tables_cache)
    _is_followup = bool(conversation_context and conversation_context.get('previous_enhanced_question'))
    if len(_allowed_tables) == 1 and (not _is_followup) and (not await agent.cache_manager.cache_has_points()):
        logger.info('Router: single-table fast-path — skipping LLM classification (table=%s)', _allowed_tables)
        _table_briefs: Dict[str, Any] = {}
        try:
            from util.xml_prompt_loader import load_client_data_descriptions
            _descriptions = load_client_data_descriptions(client_id, dataset_id=state.get('dataset_id') or None)
            for _t in _allowed_tables:
                if _t in _descriptions:
                    _table_briefs[_t] = _descriptions[_t]
        except Exception as _desc_err:
            logger.warning('Fast-path: failed to load table briefs: %s', _desc_err)
        out = {'cache_hit': False, 'route_decision': 'simple', 'terminate_graph': False, 'query_routing_type': 'data_analyst', 'relevant_tables': _allowed_tables, 'table_count': len(_allowed_tables), 'table_briefs': _table_briefs, 'enhanced_question': user_question, 'semantic_signature': {}, 'semantic_cache_match': {}, 'best_candidate_info': None, 'agent_inputs': state.get('agent_inputs', {}), 'agent_token_usage': state.get('agent_token_usage', {}), 'planner_reference': '', 'python_reference': '', 'business_reference': '', 'denied_tables_cache': sorted(denied_tables_cache), 'all_tables_cache': sorted(all_tables_cache), 'resolved_prompts': resolved_prompts, 'active_persona': active_persona}
        out.update(_mark_node_latency(state, 'router', node_start))
        return out
    result = await agent.process(user_question=user_question, user_id=state.get('user_id', 'anonymous'), client_id=client_id, conversation_context=conversation_context, skip_rbac=True)
    agent_inputs = state.get('agent_inputs', {})
    agent_token_usage = state.get('agent_token_usage', {})
    if agent._last_inputs:
        agent_inputs['router'] = agent._last_inputs
    if agent._last_usage:
        agent_token_usage['router'] = agent._last_usage
    if hasattr(agent, '_embedding_usage') and agent._embedding_usage.get('total_tokens', 0) > 0:
        agent_token_usage['embedding'] = agent._embedding_usage
    if result.get('irrelevant_detected') or result.get('query_routing_type') == 'irrelevant':
        logger.info('Router: LLM flagged query as irrelevant (Layer 2)')
        user_msg = "I'm a data intelligence assistant and can only help with questions about your business data. Could you ask a data-related question?"
        out = {'guard_status': 'irrelevant', 'guard_reason': 'LLM-based irrelevance detection (Layer 2)', 'guard_category': 'off_topic', 'terminate_graph': True, 'cache_hit': False, 'route_decision': 'irrelevant', 'semantic_signature': result.get('semantic_signature', {}), 'query_routing_type': 'irrelevant', 'semantic_cache_match': result.get('semantic_cache_match', {}), 'enhanced_question': result.get('enhanced_question', user_question), 'executor_response': {}, 'executor_agent_task_id': '', 'executor_error_text': '', 'business_response': {'analysis': user_msg, 'follow_ups': []}, 'business_agent_task_id': f'business_guard_l2_{run_id}', 'agent_inputs': agent_inputs, 'agent_token_usage': agent_token_usage, 'denied_tables_cache': sorted(denied_tables_cache), 'all_tables_cache': sorted(all_tables_cache), 'resolved_prompts': resolved_prompts, 'active_persona': active_persona}
        out.update(_mark_node_latency(state, 'router', node_start))
        return out
    if denied_tables_cache and (not result.get('access_denied')):
        from services.table_permissions_service import check_tables_access
        relevant_tables_for_rbac = result.get('relevant_tables') or []
        violations = check_tables_access(relevant_tables_for_rbac, denied_tables_cache)
        if violations:
            logger.warning('[RBAC] Access denied — restricted tables: %s', violations)
            result['access_denied'] = True
            result['denied_tables_violated'] = violations
    if result.get('access_denied'):
        out = _rbac_access_denied_state(state, violated_tables=result.get('denied_tables_violated'))
        out['route_decision'] = 'irrelevant'
        out['semantic_signature'] = result.get('semantic_signature', {})
        out['semantic_cache_match'] = result.get('semantic_cache_match', {})
        out['agent_inputs'] = agent_inputs
        out['agent_token_usage'] = agent_token_usage
        out['denied_tables_cache'] = sorted(denied_tables_cache)
        out['all_tables_cache'] = sorted(all_tables_cache)
        out['resolved_prompts'] = resolved_prompts
        out['active_persona'] = active_persona
        out.update(_mark_node_latency(state, 'router', node_start))
        return out
    routing_type = result.get('query_routing_type', 'data_analyst')
    data_science_allowed = subscription.get('features', {}).get('advanced_agents', False)
    if routing_type == 'data_scientist' and (not data_science_allowed):
        routing_type = 'data_analyst'
        logger.info('DS restricted to Pro/Premium; client=%s plan=%s -> data_analyst', client_id, subscription.get('plan_name'))
    planner_reference = result.get('planner_reference', '')
    python_reference = ''
    business_reference = ''
    best_candidate = result.get('best_candidate_info')
    if best_candidate and best_candidate.get('question_id'):
        try:
            rm = _get_cached_response_matcher(client_id, state.get('dataset_id') or '')
            ref_data = rm.get_reference_responses(int(best_candidate['question_id']))
            if ref_data:
                from response_caching.response_matcher import format_reference_for_agent
                similarity = best_candidate.get('similarity', 0.0)
                if similarity >= rm.threshold_guide:
                    if not planner_reference:
                        planner_reference = format_reference_for_agent(ref_data, 'planner')
                    python_reference = format_reference_for_agent(ref_data, 'python')
                    business_reference = format_reference_for_agent(ref_data, 'business')
        except Exception as ref_err:
            logger.warning(f'Reference lookup failed: {ref_err}')
    if result.get('cache_hit') and result.get('cached_code') is not None:
        logger.info('Router: Cache HIT — routing to coder for re-execution')
        query_embedding = result.get('query_embedding')
        if query_embedding and run_id:
            try:
                from util.embedding_cache import embedding_cache
                embedding_cache.put(run_id, query_embedding)
            except Exception:
                pass
        cache_hit_tables = result.get('relevant_tables', [])
        cache_hit_briefs = {}
        if cache_hit_tables:
            try:
                from util.xml_prompt_loader import load_client_data_descriptions
                descriptions = load_client_data_descriptions(client_id, dataset_id=state.get('dataset_id') or None)
                for t in cache_hit_tables:
                    if t in descriptions:
                        cache_hit_briefs[t] = descriptions[t]
            except Exception as desc_err:
                logger.warning(f'Failed to load table briefs for cache-hit: {desc_err}')
        out = {'code': result['cached_code'], 'cache_hit': True, 'route_decision': 'cache_hit', 'terminate_graph': False, 'query_routing_type': routing_type, 'relevant_tables': cache_hit_tables, 'table_count': len(cache_hit_tables), 'table_briefs': cache_hit_briefs, 'enhanced_question': result.get('enhanced_question', user_question), 'semantic_signature': result.get('semantic_signature', {}), 'semantic_cache_match': result.get('semantic_cache_match', {}), 'agent_inputs': agent_inputs, 'agent_token_usage': agent_token_usage, 'planner_reference': planner_reference, 'python_reference': python_reference, 'business_reference': business_reference, 'denied_tables_cache': sorted(denied_tables_cache), 'all_tables_cache': sorted(all_tables_cache), 'resolved_prompts': resolved_prompts, 'active_persona': active_persona}
        out.update(_mark_node_latency(state, 'router', node_start))
        return out
    relevant_tables = result.get('relevant_tables', [])
    table_count = len(relevant_tables)
    query_embedding = result.get('query_embedding')
    if query_embedding and run_id:
        try:
            from util.embedding_cache import embedding_cache
            embedding_cache.put(run_id, query_embedding)
        except Exception:
            pass
    table_briefs = {}
    if table_count <= 2 and relevant_tables:
        try:
            from util.xml_prompt_loader import load_client_data_descriptions
            descriptions = load_client_data_descriptions(client_id, dataset_id=state.get('dataset_id') or None)
            for t in relevant_tables:
                if t in descriptions:
                    table_briefs[t] = descriptions[t]
        except Exception as desc_err:
            logger.warning(f'Failed to load table briefs: {desc_err}')
    route_decision = 'simple' if table_count <= 2 else 'complex'
    logger.info(f'Router: route={route_decision} | tables={table_count} | routing={routing_type} | tables={relevant_tables}')
    out = {'cache_hit': False, 'route_decision': route_decision, 'terminate_graph': False, 'query_routing_type': routing_type, 'relevant_tables': relevant_tables, 'table_count': table_count, 'table_briefs': table_briefs, 'enhanced_question': result.get('enhanced_question', user_question), 'semantic_signature': result.get('semantic_signature', {}), 'semantic_cache_match': result.get('semantic_cache_match', {}), 'best_candidate_info': best_candidate, 'agent_inputs': agent_inputs, 'agent_token_usage': agent_token_usage, 'planner_reference': planner_reference, 'python_reference': python_reference, 'business_reference': business_reference, 'denied_tables_cache': sorted(denied_tables_cache), 'all_tables_cache': sorted(all_tables_cache), 'resolved_prompts': resolved_prompts, 'active_persona': active_persona}
    out.update(_mark_node_latency(state, 'router', node_start))
    return out

@traceable(name='scout_node')
async def scout_node(state: State, config: RunnableConfig | None=None) -> AsyncGenerator[Dict[str, Any], None]:
    _check_for_cancellation(state, 'scout_node')
    llm_client = _assert_llm_client(state, 'scout_node', config)
    node_start = time.perf_counter()
    client_id = state.get('client_id')
    relevant_tables = state.get('relevant_tables', [])
    user_question = state.get('enhanced_question') or state.get('input', '')
    max_concurrent = AGENT_CONFIG.get('scout_agent', {}).get('max_concurrent_scouts', 8)
    if not relevant_tables:
        logger.warning('Scout node: no relevant tables — yielding empty results')
        out = {'scout_results': {}, 'scout_errors': {}}
        out.update(_mark_node_latency(state, 'scout', node_start))
        yield out
        return
    from util.xml_prompt_loader import load_client_data_descriptions
    descriptions = load_client_data_descriptions(client_id, dataset_id=state.get('dataset_id') or None)
    yield {'scout_progress': {'tables_scouted': 0, 'tables_total': len(relevant_tables)}}
    persona_scout_note = ''
    _SCOUT_MAX_RETRIES = int(AGENT_CONFIG.get('scout_agent', {}).get('max_retries', 2))
    _SCOUT_BACKOFF_BASE = 1.0

    def _scout_is_retryable(exc: Exception) -> bool:
        from util.llm_errors import LLMHardFailureError, _extract_status_code
        if isinstance(exc, LLMHardFailureError):
            return False
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return True
        status = _extract_status_code(exc)
        if status is None:
            return True
        if status == 429 or 500 <= status <= 504:
            return True
        return False

    @traceable(name='scout_single')
    async def run_single_scout(table_name: str, description_xml: str, scout_index: int, scout_total: int) -> Dict[str, Any]:
        prompt = f"""Analyze this table for the user's question. Return JSON only.\n\nQUESTION: {user_question}{persona_scout_note}\nTABLE: {table_name}\nDESCRIPTION:\n{description_xml}\n\nReturn JSON: {{"columns": ["relevant col names"], "types": {{"col": "type"}}, "join_keys": ["FK columns"], "notes": "important data notes"}}"""
        import json
        import re
        last_error: str = ''
        usage = None
        for attempt in range(_SCOUT_MAX_RETRIES + 1):
            try:
                result = await llm_client.generate_completion(system_prompt='You are a data table analyst. Return ONLY valid JSON.', user_message=prompt, json_mode=True)
                usage = result.get('usage')
                response = result.get('content', '')
                response_text = response.strip()
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()
                if not response_text.startswith('{'):
                    m = re.search('\\{.*\\}', response_text, re.DOTALL)
                    if m:
                        response_text = m.group(0)
                parsed = json.loads(response_text)
                return {'table': table_name, 'success': True, 'data': parsed, 'usage': usage}
            except Exception as e:
                last_error = str(e)
                is_last_attempt = attempt >= _SCOUT_MAX_RETRIES
                if is_last_attempt or not _scout_is_retryable(e):
                    if not is_last_attempt:
                        logger.warning("Scout hard failure for table '%s' (not retrying): %s", table_name, e)
                    else:
                        logger.warning("Scout failed for table '%s' after %d attempt(s): %s", table_name, attempt + 1, e)
                    return {'table': table_name, 'success': False, 'error': last_error, 'usage': usage}
                delay = _SCOUT_BACKOFF_BASE * 2 ** attempt
                logger.info("Scout transient failure for table '%s' (attempt %d/%d) — retrying in %.1fs: %s", table_name, attempt + 1, _SCOUT_MAX_RETRIES + 1, delay, e)
                await asyncio.sleep(delay)
        return {'table': table_name, 'success': False, 'error': last_error, 'usage': usage}
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_scout(table_name, desc_xml, scout_index, scout_total):
        async with semaphore:
            return await run_single_scout(table_name, desc_xml, scout_index, scout_total, langsmith_extra={'metadata': {'table_name': table_name, 'scout_index': scout_index, 'scout_total': scout_total}, 'tags': [f'scout_table:{table_name}', f'scout_index:{scout_index}']})
    tasks = []
    scout_total = len(relevant_tables)
    for scout_index, table_name in enumerate(relevant_tables, start=1):
        desc_xml = descriptions.get(table_name, f'Table: {table_name} (no description available)')
        tasks.append(bounded_scout(table_name, desc_xml, scout_index, scout_total))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    scout_results = {}
    scout_errors = {}
    scouted = 0
    for r, table_name in zip(results, relevant_tables):
        scouted += 1
        if isinstance(r, Exception):
            logger.error(f"Scout exception for table '{table_name}': {r}")
            scout_errors[table_name] = f'Unexpected exception: {r}'
        elif r.get('success'):
            scout_results[r['table']] = r['data']
        else:
            scout_errors[r['table']] = r.get('error', 'Unknown error')
        yield {'scout_progress': {'tables_scouted': scouted, 'tables_total': len(relevant_tables)}}
    logger.info(f'Scout complete: {len(scout_results)} succeeded, {len(scout_errors)} failed')
    scout_prompt_tokens = 0
    scout_completion_tokens = 0
    scout_call_count = 0
    scout_provider = None
    scout_model = None
    for r in results:
        if isinstance(r, Exception) or not isinstance(r, dict):
            continue
        u = r.get('usage')
        if u and isinstance(u, dict):
            scout_prompt_tokens += int(u.get('prompt_tokens', 0) or 0)
            scout_completion_tokens += int(u.get('completion_tokens', 0) or 0)
            scout_call_count += 1
            if not scout_provider:
                scout_provider = u.get('provider')
                scout_model = u.get('model')
    agent_token_usage = state.get('agent_token_usage', {})
    if scout_call_count > 0:
        agent_token_usage['scout'] = {'prompt_tokens': scout_prompt_tokens, 'completion_tokens': scout_completion_tokens, 'total_tokens': scout_prompt_tokens + scout_completion_tokens, 'call_count': scout_call_count, 'provider': scout_provider, 'model': scout_model}
    out = {'scout_results': scout_results, 'scout_errors': scout_errors, 'agent_token_usage': agent_token_usage}
    out.update(_mark_node_latency(state, 'scout', node_start))
    yield out

@traceable(name='collator_node')
async def collator_node(state: State) -> Dict[str, Any]:
    node_start = time.perf_counter()
    scout_results = state.get('scout_results', {})
    scout_errors = state.get('scout_errors', {})
    brief_parts = []
    all_join_keys: Dict[str, List[str]] = {}
    for table_name, data in scout_results.items():
        columns = data.get('columns', [])
        types = data.get('types', {})
        join_keys = data.get('join_keys', [])
        notes = data.get('notes', '')
        col_parts = []
        for col in columns:
            col_type = types.get(col, '')
            col_parts.append(f'{col} ({col_type})' if col_type else col)
        line = f"TABLE {table_name}: {', '.join(col_parts)}"
        if notes:
            line += f'\n  Notes: {notes}'
        brief_parts.append(line)
        if join_keys:
            all_join_keys[table_name] = join_keys
    identified_joins = []
    tables_list = list(all_join_keys.keys())
    for i in range(len(tables_list)):
        for j in range(i + 1, len(tables_list)):
            t1, t2 = (tables_list[i], tables_list[j])
            shared = set(all_join_keys[t1]) & set(all_join_keys[t2])
            for key in shared:
                identified_joins.append({'left': t1, 'right': t2, 'key': key})
    if identified_joins:
        join_lines = [f"{j['left']} JOIN {j['right']} ON {j['key']}" for j in identified_joins]
        brief_parts.append('JOINS: ' + '; '.join(join_lines))
    if scout_errors:
        warnings = [f'WARNING: Scout failed for {t}: {e}' for t, e in scout_errors.items()]
        brief_parts.extend(warnings)
    data_brief = '\n'.join(brief_parts)
    if len(data_brief) > 6000:
        data_brief = data_brief[:6000] + '\n... (truncated)'
    logger.info(f'Collator: data_brief={len(data_brief)} chars, joins={len(identified_joins)}')
    out = {'data_brief': data_brief, 'identified_joins': identified_joins}
    out.update(_mark_node_latency(state, 'collator', node_start))
    return out

@traceable(name='coder_node')
async def coder_node(state: State, config: RunnableConfig | None=None) -> AsyncGenerator[Dict[str, Any], None]:
    _check_for_cancellation(state, 'coder_node')
    llm_client = _assert_llm_client(state, 'coder_node', config)
    node_start = time.perf_counter()
    client_id = state.get('client_id')
    route_decision = state.get('route_decision', 'simple')
    routing_type = state.get('query_routing_type', 'data_analyst')
    current_attempts = int(state.get('execution_attempts') or 0)
    yield {'execution_attempts': current_attempts + 1}
    if state.get('cache_hit') and state.get('code') is not None:
        logger.info('Coder: Cache hit — re-executing cached code for fresh data')
        async for chunk in _execute_cached_code(state, llm_client, node_start):
            yield chunk
        return
    plan = ''
    if route_decision == 'complex':
        plan = state.get('data_brief', '')
    elif route_decision == 'simple':
        table_briefs = state.get('table_briefs', {})
        if table_briefs:
            plan = '\n'.join((f'TABLE {t}:\n{brief}' for t, brief in table_briefs.items()))
        else:
            tables = state.get('relevant_tables', [])
            plan = 'Available tables: ' + ', '.join(tables)
            if tables:
                logger.error('Coder: table_briefs empty for simple route — schema descriptions missing for %s. LLM will not have column information.', tables)
    if not plan:
        logger.warning('Coder: No plan available — using minimal context')
        plan = f"Answer the user's question: {state.get('enhanced_question') or state.get('input', '')}"
    persona = state.get('active_persona') or {}
    if persona.get('force_ds_agent') and routing_type != 'data_scientist':
        logger.info("Coder: persona '%s' forces DS agent (was '%s')", persona.get('slug'), routing_type)
        routing_type = 'data_scientist'
    logger.info(f'Coder: final routing decision: {routing_type} (plan length={len(plan)} chars)')
    if routing_type == 'data_scientist':
        async for chunk in _run_ds_analysis(state, plan, llm_client, node_start):
            yield chunk
    else:
        async for chunk in _run_da_analysis(state, plan, llm_client, node_start):
            yield chunk

async def _execute_cached_code(state: State, llm_client: LLMClient, node_start: float) -> AsyncGenerator[Dict[str, Any], None]:
    from agents.executor_agent import ExecutorAgent
    from db_config.database import get_db, get_mongo_manager
    client_id = state.get('client_id')
    code = state.get('code', '')
    run_id = state.get('run_id', '')
    rbac_violation = await _validate_code_rbac(state, code)
    if rbac_violation:
        yield rbac_violation
        return
    try:
        db = get_db()
        agent = ExecutorAgent(client_id=client_id, db=db, dataset_id=state.get('dataset_id') or None)
        result = await agent.process(user_id=state.get('user_id', 'anonymous'), planner_task_id=f'router_{run_id}', python_agent_task_id=f'coder_cache_{run_id}', generated_code=code, run_id=run_id, execution_attempts=state.get('execution_attempts', 0))
        out = {'executor_response': result.get('generated_response', ''), 'executor_agent_task_id': result.get('executor_agent_task_id', ''), 'executor_error_text': result.get('error_text', ''), 'cache_hit': True}
        out.update(_mark_node_latency(state, 'coder', node_start))
        record_graph_final_payload(run_id, out)
        yield out
    except Exception as e:
        logger.error(f'Cache re-execution failed: {e}', exc_info=True)
        out = {'executor_response': {'console_output': f'Cache re-execution failed: {str(e)}', 'dataframes': [], 'plotly_charts': []}, 'executor_error_text': str(e), 'cache_hit': True}
        out.update(_mark_node_latency(state, 'coder', node_start))
        record_graph_final_payload(run_id, out)
        yield out

async def _validate_code_rbac(state: State, code: str) -> Dict[str, Any] | None:
    denied_tables = state.get('denied_tables_cache') or []
    all_tables = state.get('all_tables_cache') or []
    if not denied_tables or not code:
        return None
    try:
        from util.knowledge_filter import extract_table_names_from_text
        from services.table_permissions_service import check_tables_access
        used_tables = extract_table_names_from_text(code, all_tables, max_tables=50)
        violations = check_tables_access(used_tables, denied_tables)
        if violations:
            logger.warning(f'[RBAC] Code references denied tables: {violations}')
            return _rbac_access_denied_state(state, violated_tables=violations)
    except Exception as e:
        logger.warning(f'RBAC code validation error: {e}')
    return None

async def _run_ds_analysis(state: State, plan: str, llm_client: LLMClient, node_start: float) -> AsyncGenerator[Dict[str, Any], None]:
    _check_for_cancellation(state, 'coder_node_ds')
    client_id = state.get('client_id')
    from db_config.mongo_server import get_db as get_mongo_db
    db = await get_mongo_db()
    user_query = state.get('enhanced_question') or state.get('input', '')
    ds_resolved_prompt = state.get('resolved_prompts', {}).get('data_science_agent') or ''
    coder_persona = state.get('active_persona') or {}
    if coder_persona.get('scout_context'):
        persona_block = f"\n\n---\nACTIVE AGENT PERSONA: {coder_persona.get('display_name', '')}\nYou are operating as a specialised agent for this domain. Apply the following domain context to your analysis, metric selection, and code generation. Prioritise columns, KPIs, and calculations that are relevant to this domain. If the actual data does not contain the domain-specific columns mentioned below, proceed normally with the columns that ARE available — never fail or produce empty output because expected domain columns are missing.\n{coder_persona['scout_context']}\n---"
        custom_marker = '\n\n' + '=' * 80
        if custom_marker in ds_resolved_prompt:
            idx = ds_resolved_prompt.index(custom_marker)
            ds_resolved_prompt = ds_resolved_prompt[:idx] + persona_block + ds_resolved_prompt[idx:]
        else:
            ds_resolved_prompt = ds_resolved_prompt + persona_block
        logger.info('DS coder: persona context injected | persona=%s', coder_persona.get('slug'))
    try:
        from agents.data_science_agent import DataScienceAgent
        agent = DataScienceAgent(client_id=client_id, db=db, llm_client=llm_client, resolved_prompt=ds_resolved_prompt or None, dataset_id=state.get('dataset_id') or None, session_id=state.get('session_id') or '')
        agent._user_id = state.get('user_id')
        all_code_snippets = []
        iteration_code_map: Dict[int, str] = {}
        execution_outputs = []
        step_descriptions = []
        final_result = None
        pipeline_error_message = None
        error_agent_usage = None
        ds_context = {}
        planned_tables = state.get('relevant_tables') or []
        if planned_tables:
            ds_context['planned_tables'] = planned_tables
        if state.get('adhoc_mode'):
            ds_context['adhoc_mode'] = True
        adhoc_path = state.get('adhoc_dataset_path')
        _code_tok_buf = ''
        _code_tok_step = None
        _CODE_TOK_FLUSH = 24
        try:
            async for event in agent.execute_analysis(user_query=user_query, plan=plan, dataset_path=adhoc_path, dataset_dict=None, context=ds_context):
                event_type = event.get('type')
                if event_type == 'iteration_start':
                    iteration = event.get('iteration')
                    reasoning = event.get('reasoning', '')
                    step_descriptions.append({'step': iteration, 'description': reasoning})
                    yield {'cell_start': True, 'step_num': iteration, 'total_steps': event.get('max_iterations'), 'description': reasoning, 'thinking': event.get('thinking', ''), 'details': []}
                elif event_type == 'code_token':
                    iteration = event.get('iteration')
                    if _code_tok_step is not None and iteration != _code_tok_step and _code_tok_buf:
                        yield {'cell_code_token': True, 'step_num': _code_tok_step, 'delta': _code_tok_buf, 'attempt': 1}
                        _code_tok_buf = ''
                    _code_tok_step = iteration
                    _code_tok_buf += event.get('delta', '')
                    if len(_code_tok_buf) >= _CODE_TOK_FLUSH:
                        yield {'cell_code_token': True, 'step_num': iteration, 'delta': _code_tok_buf, 'attempt': event.get('attempt', 1)}
                        _code_tok_buf = ''
                elif event_type == 'code_generated':
                    code = event.get('code', '')
                    iteration = event.get('iteration')
                    attempt = event.get('attempt', 1)
                    if _code_tok_buf:
                        yield {'cell_code_token': True, 'step_num': _code_tok_step, 'delta': _code_tok_buf, 'attempt': 1}
                        _code_tok_buf = ''
                    _code_tok_step = None
                    all_code_snippets.append(code)
                    if iteration is not None:
                        iteration_code_map[iteration] = code
                    yield {'cell_code': True, 'step_num': iteration, 'code': code, 'attempt': attempt}
                elif event_type == 'iteration_execution':
                    iteration = event.get('iteration')
                    has_error = bool(event.get('exception'))
                    attempt = event.get('attempt', 1)
                    execution_outputs.append({'step': iteration, 'stdout': event.get('stdout', ''), 'stderr': event.get('stderr', ''), 'exception': event.get('exception')})
                    yield {'cell_result': True, 'step_num': iteration, 'success': not has_error, 'stdout': event.get('stdout', ''), 'error': event.get('exception'), 'attempt': attempt}
                elif event_type == 'iteration_retry':
                    yield {'cell_retry': True, 'step_num': event.get('iteration'), 'attempt': event.get('attempt', 1), 'error': event.get('error', '')}
                elif event_type == 'iteration_complete':
                    yield {'cell_complete': True, 'step_num': event.get('iteration'), 'available_variables': event.get('available_variables', [])}
                elif event_type == 'error':
                    iteration = event.get('iteration', event.get('step_num'))
                    last_error = event.get('last_error', event.get('message', ''))
                    pipeline_error_message = event.get('message', 'Unknown error')
                    if '_agent_usage' in event:
                        error_agent_usage = event['_agent_usage']
                    if event.get('error_type'):
                        yield {'fatal_error': True, 'detail': pipeline_error_message, 'error_type': event['error_type']}
                    else:
                        yield {'cell_failed': True, 'step_num': iteration, 'last_error': last_error}
                elif event_type == 'final_result':
                    final_result = event
                    break
                elif event_type == 'status':
                    yield {'cell_status': True, 'message': event.get('message', '')}
        except asyncio.CancelledError:
            if not final_result:
                raise
            logger.warning('DS finalize: cancellation after final_result — proceeding with finalization')
        logger.info('DS finalize: has_result=%s', final_result is not None)
        combined_code = '\n\n# --- Code Execution Summary ---\n' + '\n\n'.join((f'# Iteration {i + 1}\n{c}' for i, c in enumerate(all_code_snippets))) if all_code_snippets else ''
        generated_text = final_result.get('text_output', '') if final_result else ''
        text_outputs = []
        kpis = final_result.get('kpis') if final_result else None
        if kpis:
            text_outputs.append({'name': '_kpis_', 'value': str(_round_kpi_values(kpis))})
        ds_summary = final_result.get('summary', '') if final_result else ''
        if ds_summary and ds_summary.strip() not in (generated_text.strip(), ''):
            text_outputs.append({'name': '_summary_', 'value': ds_summary})
        dataframes = await _extract_dataframes_from_result(final_result, client_id=client_id)
        plotly_charts = _extract_plotly_charts(final_result)
        console_output, business_console_output = _build_console_outputs(execution_outputs, iteration_code_map, step_descriptions, final_result)
        executor_response = {'console_output': console_output, 'business_console_output': business_console_output, 'dataframes': dataframes, 'plotly_charts': plotly_charts, 'matplotlib_images': [], 'text_outputs': text_outputs, 'analysis_steps': step_descriptions, 'ds_analysis': generated_text, 'artifact_registry': getattr(agent, '_artifact_registry', [])}
        error_text = _get_error_text(pipeline_error_message, execution_outputs)
        agent_token_usage = state.get('agent_token_usage', {})
        if final_result and '_agent_usage' in final_result:
            agent_token_usage['data_science'] = final_result['_agent_usage']
        out = {'code': combined_code, 'executor_response': executor_response, 'executor_error_text': error_text, 'data_science_mode': True, 'agent_token_usage': agent_token_usage}
        out.update(_mark_node_latency(state, 'coder', node_start))
        record_graph_final_payload(state.get('run_id', ''), out)
        yield out
    except Exception as e:
        logger.error(f'Error in DS coder: {e}\n{traceback.format_exc()}')
        agent_token_usage = state.get('agent_token_usage', {})
        if error_agent_usage:
            agent_token_usage['data_science'] = error_agent_usage
        out = {'code': '', 'executor_response': {'console_output': f'Data science execution failed: {str(e)}', 'business_console_output': '', 'dataframes': [], 'plotly_charts': [], 'matplotlib_images': [], 'text_outputs': [], 'ds_analysis': f'Analysis pipeline encountered an error during finalization: {str(e)}'}, 'executor_error_text': str(e), 'data_science_mode': True, 'agent_token_usage': agent_token_usage}
        out.update(_mark_node_latency(state, 'coder', node_start))
        record_graph_final_payload(state.get('run_id', ''), out)
        yield out

async def _run_da_analysis(state: State, plan: str, llm_client: LLMClient, node_start: float) -> AsyncGenerator[Dict[str, Any], None]:
    _check_for_cancellation(state, 'coder_node_da')
    client_id = state.get('client_id')
    from db_config.mongo_server import get_db as get_mongo_db
    db = await get_mongo_db()
    user_query = state.get('enhanced_question') or state.get('input', '')
    da_resolved_prompt = state.get('resolved_prompts', {}).get('data_analyst_agent') or ''
    coder_persona = state.get('active_persona') or {}
    if coder_persona.get('scout_context'):
        persona_block = f"\n\n---\nACTIVE AGENT PERSONA: {coder_persona.get('display_name', '')}\nYou are operating as a specialised agent for this domain. Apply the following domain context to your analysis, metric selection, and code generation. Prioritise columns, KPIs, and calculations that are relevant to this domain. If the actual data does not contain the domain-specific columns mentioned below, proceed normally with the columns that ARE available — never fail or produce empty output because expected domain columns are missing.\n{coder_persona['scout_context']}\n---"
        custom_marker = '\n\n' + '=' * 80
        if custom_marker in da_resolved_prompt:
            idx = da_resolved_prompt.index(custom_marker)
            da_resolved_prompt = da_resolved_prompt[:idx] + persona_block + da_resolved_prompt[idx:]
        else:
            da_resolved_prompt = da_resolved_prompt + persona_block
        logger.info('DA coder: persona context injected | persona=%s', coder_persona.get('slug'))
    try:
        from agents.data_analyst_agent import DataAnalystAgent
        agent = DataAnalystAgent(client_id=client_id, db=db, llm_client=llm_client, resolved_prompt=da_resolved_prompt or None, dataset_id=state.get('dataset_id') or None, session_id=state.get('session_id') or '')
        agent._user_id = state.get('user_id')
        all_code_snippets = []
        iteration_code_map: Dict[int, str] = {}
        iteration_reasoning_map: Dict[int, str] = {}
        execution_outputs = []
        final_result = None
        pipeline_error_message = None
        error_agent_usage = None
        da_context = {}
        planned_tables = state.get('relevant_tables') or []
        if planned_tables:
            da_context['planned_tables'] = planned_tables
        if state.get('adhoc_mode'):
            da_context['adhoc_mode'] = True
        adhoc_path = state.get('adhoc_dataset_path')
        _code_tok_buf = ''
        _code_tok_step = None
        _CODE_TOK_FLUSH = 24
        try:
            async for event in agent.execute_analysis(user_query=user_query, plan=plan, dataset_path=adhoc_path, dataset_dict=None, context=da_context):
                event_type = event.get('type')
                logger.info(f"DA event: type={event_type} iteration={event.get('iteration')} attempt={event.get('attempt')}")
                if event_type == 'iteration_start':
                    iteration = event.get('iteration')
                    reasoning = event.get('reasoning', '')
                    if iteration is not None:
                        iteration_reasoning_map[iteration] = reasoning
                    yield {'cell_start': True, 'step_num': iteration, 'total_steps': event.get('max_iterations'), 'description': reasoning, 'thinking': event.get('thinking', ''), 'details': []}
                elif event_type == 'code_token':
                    iteration = event.get('iteration')
                    if _code_tok_step is not None and iteration != _code_tok_step and _code_tok_buf:
                        yield {'cell_code_token': True, 'step_num': _code_tok_step, 'delta': _code_tok_buf, 'attempt': 1}
                        _code_tok_buf = ''
                    _code_tok_step = iteration
                    _code_tok_buf += event.get('delta', '')
                    if len(_code_tok_buf) >= _CODE_TOK_FLUSH:
                        yield {'cell_code_token': True, 'step_num': iteration, 'delta': _code_tok_buf, 'attempt': event.get('attempt', 1)}
                        _code_tok_buf = ''
                elif event_type == 'code_generated':
                    code = event.get('code', '')
                    iteration = event.get('iteration')
                    attempt = event.get('attempt', 1)
                    if _code_tok_buf:
                        yield {'cell_code_token': True, 'step_num': _code_tok_step, 'delta': _code_tok_buf, 'attempt': 1}
                        _code_tok_buf = ''
                    _code_tok_step = None
                    all_code_snippets.append(code)
                    if iteration is not None:
                        iteration_code_map[iteration] = code
                    yield {'cell_code': True, 'step_num': iteration, 'code': code, 'attempt': attempt}
                elif event_type == 'iteration_execution':
                    iteration = event.get('iteration')
                    has_error = bool(event.get('exception'))
                    attempt = event.get('attempt', 1)
                    execution_outputs.append({'step': iteration, 'stdout': event.get('stdout', ''), 'stderr': event.get('stderr', ''), 'exception': event.get('exception')})
                    yield {'cell_result': True, 'step_num': iteration, 'success': not has_error, 'stdout': event.get('stdout', ''), 'error': event.get('exception'), 'attempt': attempt}
                elif event_type == 'iteration_retry':
                    yield {'cell_retry': True, 'step_num': event.get('iteration'), 'attempt': event.get('attempt', 1), 'error': event.get('error', '')}
                elif event_type == 'iteration_complete':
                    yield {'cell_complete': True, 'step_num': event.get('iteration'), 'available_variables': event.get('available_variables', [])}
                elif event_type == 'error':
                    iteration = event.get('iteration', event.get('step_num'))
                    last_error = event.get('last_error', event.get('message', ''))
                    pipeline_error_message = event.get('message', 'Unknown error')
                    if '_agent_usage' in event:
                        error_agent_usage = event['_agent_usage']
                    if event.get('error_type'):
                        yield {'fatal_error': True, 'detail': pipeline_error_message, 'error_type': event['error_type']}
                    else:
                        yield {'cell_failed': True, 'step_num': iteration, 'last_error': last_error}
                elif event_type == 'final_result':
                    final_result = event
                    break
                elif event_type == 'status':
                    yield {'cell_status': True, 'message': event.get('message', '')}
        except asyncio.CancelledError:
            if not final_result:
                raise
            logger.warning('DA finalize: cancellation after final_result — proceeding with finalization')
        logger.info('DA finalize: has_result=%s', final_result is not None)
        combined_code = '\n\n# --- Data Analyst Code Summary ---\n' + '\n\n'.join((f'# Iteration {i + 1}\n{c}' for i, c in enumerate(all_code_snippets))) if all_code_snippets else ''
        plotly_charts = []
        dataframes = []
        text_outputs = []
        text_output = ''
        if final_result:
            viz_type = final_result.get('viz_type', 'chart_and_table')
            try:
                if viz_type not in ('kpi_card', 'table_only'):
                    plotly_charts = _extract_plotly_charts(final_result, name_prefix='_analyst_chart_')
            except Exception as _chart_err:
                logger.warning('DA coder: chart extraction failed: %s', _chart_err)
            try:
                table_data = final_result.get('table')
                if table_data and isinstance(table_data, list):
                    dataframes.append({'name': '_analyst_table_', 'data': table_data})
                generated_df = final_result.get('dataframe')
                if generated_df is not None and (not table_data):
                    dataframes = await _extract_dataframes_from_result(final_result, client_id=client_id)
            except Exception as _df_err:
                logger.warning('DA coder: dataframe extraction failed: %s', _df_err)
            try:
                kpis = final_result.get('kpis')
                if kpis:
                    text_outputs.append({'name': '_kpis_', 'value': str(_round_kpi_values(kpis))})
                summary = final_result.get('summary', '')
                text_output = final_result.get('text_output', '')
                if summary:
                    text_outputs.append({'name': '_summary_', 'value': summary})
            except Exception as _kpi_err:
                logger.warning('DA coder: KPI/summary extraction failed: %s', _kpi_err)
                text_output = final_result.get('text_output', '')
        if not dataframes and final_result:
            debug_fallback_enabled = os.getenv('CORESIGHT_ENABLE_DEBUG_RESULT_FALLBACK', 'false').strip().lower() in {'1', 'true', 'yes', 'on'}
            if debug_fallback_enabled:
                fallback_row = None
                if isinstance(final_result.get('kpis'), dict) and final_result.get('kpis'):
                    fallback_row = final_result.get('kpis')
                elif final_result.get('summary'):
                    fallback_row = {'summary': str(final_result.get('summary'))}
                elif final_result.get('text_output'):
                    fallback_row = {'summary': str(final_result.get('text_output'))}
                if fallback_row:
                    dataframes = [{'name': '_analyst_fallback_', 'data': [fallback_row]}]
        try:
            console_output, business_console_output = _build_console_outputs(execution_outputs, iteration_code_map, [{'step': k, 'description': v} for k, v in iteration_reasoning_map.items()], final_result)
        except Exception as _console_err:
            logger.warning('DA coder: console output build failed: %s', _console_err)
            console_output = '\n'.join((eo.get('stdout', '') for eo in execution_outputs if eo.get('stdout')))
            business_console_output = console_output
        executor_response = {'console_output': console_output, 'business_console_output': business_console_output, 'dataframes': dataframes, 'plotly_charts': plotly_charts, 'matplotlib_images': [], 'text_outputs': text_outputs, 'da_analysis': text_output, 'artifact_registry': getattr(agent, '_artifact_registry', [])}
        logger.info('DA coder executor_response: charts=%d dataframes=%d text_outputs=%d da_analysis=%s', len(plotly_charts), len(dataframes), len(text_outputs), bool(text_output))
        error_text = _get_error_text(pipeline_error_message, execution_outputs)
        agent_token_usage = state.get('agent_token_usage', {})
        if final_result and '_agent_usage' in final_result:
            agent_token_usage['data_analyst'] = final_result['_agent_usage']
        out = {'code': combined_code, 'executor_response': executor_response, 'executor_error_text': error_text, 'data_analyst_mode': True, 'agent_token_usage': agent_token_usage}
        out.update(_mark_node_latency(state, 'coder', node_start))
        record_graph_final_payload(state.get('run_id', ''), out)
        yield out
    except Exception as e:
        logger.error(f'Error in DA coder: {e}\n{traceback.format_exc()}')
        agent_token_usage = state.get('agent_token_usage', {})
        if error_agent_usage:
            agent_token_usage['data_analyst'] = error_agent_usage
        _fallback_console = '\n'.join((eo.get('stdout', '') for eo in execution_outputs if eo.get('stdout'))) if execution_outputs else ''
        out = {'code': '', 'executor_response': {'console_output': _fallback_console or f'Data analyst execution failed: {str(e)}', 'business_console_output': _fallback_console, 'dataframes': [], 'plotly_charts': [], 'matplotlib_images': [], 'text_outputs': [], 'da_analysis': f'Analysis pipeline encountered an error during finalization: {str(e)}'}, 'executor_error_text': str(e), 'data_analyst_mode': True, 'agent_token_usage': agent_token_usage}
        out.update(_mark_node_latency(state, 'coder', node_start))
        record_graph_final_payload(state.get('run_id', ''), out)
        yield out

def _round_kpi_values(kpis, decimals: int=2):
    if isinstance(kpis, dict):
        return {k: _round_kpi_values(v, decimals) for k, v in kpis.items()}
    if isinstance(kpis, (list, tuple)):
        return type(kpis)((_round_kpi_values(v, decimals) for v in kpis))
    if isinstance(kpis, float):
        return round(kpis, decimals)
    return kpis

async def _guard_dataframe_size(records: List[Dict[str, Any]], name: str, client_id: str='') -> Dict[str, Any]:
    import pandas as pd
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    _DF_MEMORY_LIMIT = 50 * 1024 * 1024
    _DF_PREVIEW_ROWS = 10000
    total_rows = len(records)
    is_truncated = False
    full_data_path = None
    try:
        df = pd.DataFrame(records)
        mem_bytes = int(df.memory_usage(deep=True).sum())
        if mem_bytes > _DF_MEMORY_LIMIT:
            try:
                base_output_dir = _Path(f'assets/clients/{client_id}/output/')
                data_objects_dir = base_output_dir / 'data_objects'
                data_objects_dir.mkdir(parents=True, exist_ok=True)
                timestamp = _dt.utcnow().strftime('%Y%m%d_%H%M%S_%f')
                parquet_path = data_objects_dir / f'{name}_{timestamp}.parquet'
                await asyncio.to_thread(df.to_parquet, parquet_path, index=False)
                full_data_path = str(parquet_path)
                is_truncated = True
                records = df.head(_DF_PREVIEW_ROWS).to_dict(orient='records')
                logger.info("Large FINAL_RESULT DataFrame '%s' (%d rows, %.1f MB) spilled to %s; preview truncated to %d rows.", name, total_rows, mem_bytes / (1024 * 1024), parquet_path, _DF_PREVIEW_ROWS)
            except Exception as spill_err:
                logger.warning("Failed to spill FINAL_RESULT DataFrame '%s' to parquet: %s — returning full data.", name, spill_err)
    except Exception:
        pass
    return {'name': name, 'data': records, 'total_rows': total_rows, 'is_truncated': is_truncated, 'full_data_path': full_data_path}

async def _extract_dataframes_from_result(final_result, client_id: str='') -> List[Dict]:
    if not final_result:
        return []
    table_data = final_result.get('table')
    if table_data and isinstance(table_data, list):
        return [await _guard_dataframe_size(table_data, '_ds_table_', client_id)]
    generated_df = final_result.get('dataframe')
    if generated_df is None:
        return []
    if isinstance(generated_df, list):
        if generated_df and isinstance(generated_df[0], dict) and ('name' not in generated_df[0] or 'data' not in generated_df[0]):
            return [await _guard_dataframe_size(generated_df, '_generated_dataframe_', client_id)]
        result = []
        for entry in generated_df:
            if isinstance(entry, dict) and 'data' in entry and isinstance(entry['data'], list):
                result.append(await _guard_dataframe_size(entry['data'], entry.get('name', '_generated_dataframe_'), client_id))
            else:
                result.append(entry)
        return result
    elif isinstance(generated_df, dict):
        if 'data' in generated_df and isinstance(generated_df['data'], list):
            return [await _guard_dataframe_size(generated_df['data'], generated_df.get('name', '_generated_dataframe_'), client_id)]
        return [generated_df]
    else:
        try:
            records = generated_df.to_dict(orient='records')
            return [await _guard_dataframe_size(records, '_generated_dataframe_', client_id)]
        except Exception:
            return [{'name': '_generated_dataframe_', 'data': str(generated_df)}]

def _extract_plotly_charts(final_result, name_prefix: str='_ds_chart_') -> List[Dict]:
    if not final_result:
        return []
    charts = []
    multi = final_result.get('charts')
    if multi and isinstance(multi, list):
        for mc in multi:
            if isinstance(mc, dict) and mc.get('figure'):
                charts.append({'name': mc.get('name', name_prefix), 'figure': mc['figure']})
    if not charts:
        chart_json = final_result.get('chart')
        if chart_json:
            charts = [{'name': name_prefix, 'figure': chart_json}]
    return charts

def _build_console_outputs(execution_outputs, iteration_code_map, step_descriptions, final_result):
    console_parts = []
    for eo in execution_outputs:
        step_num = eo['step']
        code = iteration_code_map.get(step_num)
        if code:
            console_parts.append(f'--- Step {step_num} Code ---\n{code}\n')
        if eo.get('stdout'):
            console_parts.append(f"--- Step {step_num} Output ---\n{eo['stdout']}")
        if eo.get('exception'):
            console_parts.append(f"Step {eo['step']} ERROR: {eo['exception']}")
    console_output = '\n'.join(console_parts)
    step_desc_map = {sd['step']: sd['description'] for sd in step_descriptions}
    business_console_parts = []
    for eo in execution_outputs:
        if eo.get('exception'):
            continue
        step_num = eo['step']
        stdout = eo.get('stdout', '')
        reasoning = step_desc_map.get(step_num, '')
        if stdout and stdout.strip():
            header = f'Step {step_num}'
            if reasoning:
                header += f' — {reasoning}'
            business_console_parts.append(f'--- {header} ---\n{stdout}')
    if final_result:
        final_summary = final_result.get('summary', '') or final_result.get('text_output', '') or final_result.get('prediction', '')
        if final_summary:
            business_console_parts.append(f'--- Final Answer ---\n{final_summary}')
    business_console_output = '\n'.join(business_console_parts) if business_console_parts else console_output
    return (console_output, business_console_output)

def _get_error_text(pipeline_error_message, execution_outputs):
    has_errors = pipeline_error_message or any((eo.get('exception') for eo in execution_outputs))
    if pipeline_error_message:
        return pipeline_error_message
    elif has_errors:
        return 'Some steps had execution errors'
    return ''

@traceable(name='narrator_node')
async def narrator_node(state: State, config: RunnableConfig | None=None) -> AsyncGenerator[Dict[str, Any], None]:
    _check_for_cancellation(state, 'narrator_node')
    node_start = time.perf_counter()
    if state.get('terminate_graph') and state.get('business_response'):
        logger.info('Narrator: terminate_graph set - returning precomputed response')
        from util.embedding_cache import embedding_cache
        run_id = state.get('run_id', '')
        if run_id:
            embedding_cache.remove(run_id)
        out = {'business_response': state.get('business_response'), 'business_agent_task_id': state.get('business_agent_task_id', f"business_rbac_denied_{state.get('run_id', '')}")}
        out.update(_mark_node_latency(state, 'narrator', node_start))
        record_graph_final_payload(run_id, out)
        yield out
        return
    if state.get('cache_hit'):
        logger.info('Narrator: Cache hit - generating fresh business insights')
    llm_client = _assert_llm_client(state, 'narrator_node', config)
    from agents.business_agent import BusinessAgent
    from db_config.database import get_db, get_mongo_manager
    client_id = state.get('client_id')
    if not client_id:
        raise ValueError('client_id is REQUIRED in state for multi-tenant operation')
    db = get_db()
    mongo_manager = get_mongo_manager()
    narrator_persona = state.get('active_persona') or {}
    narrator_persona_guidance = ''
    if narrator_persona.get('narrator_style'):
        narrator_persona_guidance = f"ACTIVE AGENT PERSONA — {narrator_persona.get('display_name', '')}\nYou are presenting insights as a specialised {narrator_persona.get('display_name', '')} agent.\nUse the domain vocabulary, KPIs, and framing below in EVERY section of your response (summary, metrics, insights, recommendations, follow_ups). This persona shapes your language and focus — it does NOT change which JSON sections you produce.\n\n{narrator_persona['narrator_style']}"
        logger.info('Narrator: persona guidance prepared for user message | persona=%s', narrator_persona.get('slug'))
    agent = BusinessAgent(client_id=client_id, db=mongo_manager, llm_client=llm_client, resolved_prompt=state.get('resolved_prompts', {}).get('business') or None)
    use_knowledge_filtering = should_use_knowledge_filtering(client_id, state.get('session_id', ''))
    executor_response_data = state.get('executor_response', {})
    run_id = state.get('run_id', '')
    if not executor_response_data and run_id:
        late_payload = peek_graph_final_payload(run_id)
        late_executor_response = late_payload.get('executor_response')
        if late_executor_response:
            executor_response_data = late_executor_response
            state['executor_response'] = late_executor_response
            if late_payload.get('code') and (not state.get('code')):
                state['code'] = late_payload['code']
            if late_payload.get('executor_error_text') and (not state.get('executor_error_text')):
                state['executor_error_text'] = late_payload['executor_error_text']
            if late_payload.get('executor_agent_task_id') and (not state.get('executor_agent_task_id')):
                state['executor_agent_task_id'] = late_payload['executor_agent_task_id']
            if late_payload.get('agent_inputs'):
                state['agent_inputs'] = {**(state.get('agent_inputs') or {}), **late_payload['agent_inputs']}
            if late_payload.get('agent_token_usage'):
                state['agent_token_usage'] = {**(state.get('agent_token_usage') or {}), **late_payload['agent_token_usage']}
            logger.info('Narrator recovered executor_response from graph result registry | run_id=%s', run_id)
    try:
        import json
        if isinstance(executor_response_data, str) and executor_response_data.strip().startswith('{'):
            parsed = json.loads(executor_response_data)
        else:
            parsed = executor_response_data if isinstance(executor_response_data, dict) else {}
    except Exception:
        parsed = {'console_output': executor_response_data, 'dataframes': []}
    plan_text = state.get('data_brief', '')
    if not plan_text:
        table_briefs = state.get('table_briefs', {})
        if table_briefs:
            plan_text = '\n'.join((f'TABLE {t}: {b}' for t, b in table_briefs.items()))
    adhoc_note = ''
    if state.get('adhoc_mode'):
        adhoc_file = state.get('adhoc_file_metadata', {})
        fname = adhoc_file.get('original_filename', 'uploaded file')
        adhoc_note = f"\n[CONTEXT: This analysis is based on a user-uploaded ad-hoc file '{fname}', not the organization's configured data sources. Frame insights around what the data shows rather than organizational context.]\n"
    stream = agent.process(executor_results=parsed, planner_response={'user_question': state.get('enhanced_question') or state.get('input'), 'plan': adhoc_note + plan_text if adhoc_note else plan_text}, reference_guidance=state.get('business_reference', '') or '', use_knowledge_filtering=use_knowledge_filtering, persona_guidance=narrator_persona_guidance)
    SENTINEL = '===STRUCTURED==='
    raw_response = ''
    structured_buf = ''
    narr_emitted_len = 0
    past_sentinel = False
    usage_info = None
    _narr_buf = ''
    _NARR_FLUSH = 24
    async for token, usage in stream:
        if token == '__USAGE__':
            if usage:
                usage_info = usage
            continue
        raw_response += token
        if past_sentinel:
            structured_buf += token
            continue
        idx = raw_response.find(SENTINEL)
        if idx == -1:
            safe_len = max(0, len(raw_response) - (len(SENTINEL) - 1))
            new = raw_response[narr_emitted_len:safe_len]
            if new:
                _narr_buf += new
                narr_emitted_len = safe_len
                if len(_narr_buf) >= _NARR_FLUSH:
                    yield {'business_token': _narr_buf}
                    _narr_buf = ''
        else:
            new = raw_response[narr_emitted_len:idx]
            if _narr_buf or new:
                yield {'business_token': _narr_buf + new}
                _narr_buf = ''
            narr_emitted_len = idx
            past_sentinel = True
            structured_buf += raw_response[idx + len(SENTINEL):]
    if _narr_buf:
        yield {'business_token': _narr_buf}
        _narr_buf = ''
    if not past_sentinel:
        tail = raw_response[narr_emitted_len:]
        if tail:
            yield {'business_token': tail}
        structured_for_parse = raw_response
    else:
        structured_for_parse = structured_buf
    processed_biz = await agent.process_raw_business_insights(structured_for_parse)
    agent_inputs = state.get('agent_inputs', {})
    agent_token_usage = state.get('agent_token_usage', {})
    if hasattr(agent, '_last_inputs'):
        agent_inputs['business'] = agent._last_inputs
    if hasattr(agent, '_last_usage') and agent._last_usage:
        agent_token_usage['business'] = agent._last_usage
    elif usage_info:
        agent_token_usage['business'] = usage_info
    elif hasattr(agent, 'llm_client') and hasattr(agent, '_last_inputs'):
        inputs = agent._last_inputs or {}
        try:
            agent_token_usage['business'] = agent.llm_client._estimate_usage(system_prompt=inputs.get('system_prompt', ''), user_message=inputs.get('user_message', ''), prior_messages=None, content=raw_response, provider=getattr(agent.llm_client, 'default_provider', None), model=getattr(agent.llm_client, 'default_model', None))
        except Exception:
            pass
    if 'business' not in agent_token_usage:
        agent_token_usage['business'] = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'provider': getattr(agent.llm_client, 'default_provider', None) if hasattr(agent, 'llm_client') else None, 'model': getattr(agent.llm_client, 'default_model', None) if hasattr(agent, 'llm_client') else None, 'missing_usage': True}
    from util.embedding_cache import embedding_cache
    if run_id:
        embedding_cache.remove(run_id)
    out = {'business_response': processed_biz, 'business_agent_task_id': f"business_{state.get('run_id', '')}", 'agent_inputs': agent_inputs, 'agent_token_usage': agent_token_usage}
    out.update(_mark_node_latency(state, 'narrator', node_start))
    record_graph_final_payload(run_id, out)
    yield out

@traceable(name='guard_node')
async def guard_node_traced(state: State, config: RunnableConfig | None=None) -> Dict[str, Any]:
    from util.guard_node import guard_node
    return await guard_node(state, config)

def build_graph_app():
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
            raise RuntimeError(f'LangChain core not available for RunnableLambda: {e}')
        try:
            import redis.asyncio as redis
        except Exception:
            import redis
    except Exception as e:
        raise RuntimeError(f'LangGraph/Redis not available: {e}. Install deps.')
    graph = StateGraph(State)
    graph.add_node('guard', RunnableLambda(guard_node_traced).with_config(run_name='guard_node'))
    graph.add_node('router', RunnableLambda(router_node).with_config(run_name='router_node'))
    graph.add_node('scout', RunnableLambda(scout_node).with_config(run_name='scout_node'))
    graph.add_node('collator', RunnableLambda(collator_node).with_config(run_name='collator_node'))
    graph.add_node('coder', RunnableLambda(coder_node).with_config(run_name='coder_node'))
    graph.add_node('narrator', RunnableLambda(narrator_node).with_config(run_name='narrator_node'))
    graph.set_entry_point('guard')

    def guard_decision(state: State) -> str:
        if state.get('terminate_graph'):
            logger.info('Guard decision: irrelevant → narrator')
            return 'narrator'
        return 'router'
    graph.add_conditional_edges('guard', guard_decision, {'narrator': 'narrator', 'router': 'router'})

    def router_decision(state: State) -> str:
        if state.get('terminate_graph'):
            logger.info('Router decision: terminate_graph → narrator')
            return 'narrator'
        route = state.get('route_decision', 'simple')
        if route == 'irrelevant':
            logger.info('Router decision: irrelevant → narrator')
            return 'narrator'
        elif route == 'cache_hit':
            logger.info('Router decision: cache_hit → coder')
            return 'coder'
        elif route == 'complex':
            logger.info(f"Router decision: complex ({state.get('table_count', 0)} tables) → scout")
            return 'scout'
        else:
            logger.info(f"Router decision: simple ({state.get('table_count', 0)} tables) → coder")
            return 'coder'
    graph.add_conditional_edges('router', router_decision, {'narrator': 'narrator', 'coder': 'coder', 'scout': 'scout'})
    graph.add_edge('scout', 'collator')
    graph.add_edge('collator', 'coder')
    graph.add_edge('coder', 'narrator')
    graph.add_edge('narrator', END)
    checkpointer = None
    try:
        if 'redis' in dir() and RedisSaver is not None:
            redis_client = redis.from_url(REDIS_URL)
            checkpointer = RedisSaver(redis_client)
    except Exception as e:
        logger.warning(f'Redis checkpointer unavailable: {e}. Falling back.')
    if checkpointer is None and SqliteSaver is not None:
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'assets', 'data'), exist_ok=True)
            sqlite_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'data', 'state.sqlite')
            sqlite_path = os.path.abspath(sqlite_path)
            checkpointer = SqliteSaver(sqlite_path)
            logger.info(f'Using SqliteSaver at {sqlite_path}')
        except Exception as e:
            logger.warning(f'SqliteSaver unavailable: {e}. Falling back to memory.')
    if checkpointer is None and MemorySaver is not None:
        checkpointer = MemorySaver()
        logger.info('Using MemorySaver (non-persistent) for checkpoints.')
    app = graph.compile(checkpointer=checkpointer)
    logger.info('LangGraph v2 app compiled (Router-Scouts-Coder-Narrator).')
    return app