from __future__ import annotations
import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from util.time_utils import utcnow
from typing import Any, AsyncGenerator, Callable, Dict, Awaitable, Optional
from services.session_memory import session_memory
from config.system_config import USE_LANGGRAPH
try:
    from util.graph import build_graph_app
except Exception:
    build_graph_app = None
try:
    from util.adhoc_graph import build_adhoc_graph_app
except Exception:
    build_adhoc_graph_app = None
from util.Mongodb import MongoDBManager
from response_caching.response_matcher import ResponseMatcher, format_reference_for_agent
from util.cancellation import cancellation_manager
from services.db_credentials_service import DBCredentialsService
from util.dataset_paths import assets_datasets_dir
from util.data_source import require_store_in_local
from services.graph_result_registry import discard_graph_final_payload, pop_graph_final_payload
_go_background_signals: Dict[str, asyncio.Event] = {}

def register_background_signal(run_id: str) -> asyncio.Event:
    event = asyncio.Event()
    _go_background_signals[run_id] = event
    return event

def trigger_background_signal(run_id: str) -> bool:
    event = _go_background_signals.get(run_id)
    if event:
        event.set()
        return True
    return False

def cleanup_background_signal(run_id: str):
    _go_background_signals.pop(run_id, None)

def _classify_exc_for_sse(exc: Exception) -> tuple:
    from util.llm_errors import LLMHardFailureError, _extract_status_code, _extract_retry_after
    if isinstance(exc, LLMHardFailureError):
        return ('content_policy', None)
    if isinstance(exc, asyncio.TimeoutError):
        return ('transient', None)
    status = _extract_status_code(exc)
    if status == 429:
        raw = _extract_retry_after(exc)
        return ('rate_limit', int(raw) if raw else None)
    if status and 500 <= status <= 504:
        return ('transient', None)
    if status in (401, 403):
        return ('system', None)
    return ('system', None)

@dataclass
class StreamState:
    user_id: str
    input: str
    session_id: str
    run_id: str
    user_input: Dict[str, Any]
    client_id: str
    route_decision: str = ''
    relevant_tables: list = field(default_factory=list)
    code: str = ''
    executor_agent_task_id: str = ''
    business_agent_task_id: str = ''
    result: str = ''
    executor_response: Dict[str, Any] = field(default_factory=dict)
    business_response: Dict[str, Any] = field(default_factory=dict)
    history: list = field(default_factory=list)
    files: Any = None
    tables: Any = None
    executor_error_text: str = ''
    error_context: str = ''
    cached_result: Dict[str, Any] = field(default_factory=dict)
    use_cached_result: bool = False
    enhanced_question: str = ''
    use_business_agent: bool = False
    adhoc_mode: bool = False
    adhoc_file_metadata: Dict[str, Any] = field(default_factory=dict)
    adhoc_dataset_path: str = ''
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_execution_time_seconds: float = 0.0
    graph_latency_ms: Dict[str, float] = field(default_factory=dict)
    guard_status: str = ''
    guard_category: str = ''
    agent_inputs: Dict[str, Any] = field(default_factory=dict)
    agent_token_usage: Dict[str, Any] = field(default_factory=dict)
    active_persona: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None
    completion_status: str = ''
    dataset_id: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def calculate_total_time(self) -> float:
        if self.end_time:
            delta = self.end_time - self.start_time
            self.total_execution_time_seconds = delta.total_seconds()
        return self.total_execution_time_seconds

class OrchestratorManager:
    MAX_RETRIES = 3

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mongo_manager = MongoDBManager()
        self.response_matcher = None
        self.graph_app = None
        self.adhoc_graph_app = None
        if USE_LANGGRAPH and build_graph_app:
            try:
                self.graph_app = build_graph_app()
                self.logger.debug('LangGraph app enabled and compiled.')
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.logger.error(f'Failed to initialize LangGraph app: {e}')
                self.graph_app = None
            if build_adhoc_graph_app:
                try:
                    self.adhoc_graph_app = build_adhoc_graph_app()
                    self.logger.debug('Ad-hoc LangGraph app compiled.')
                except Exception as e:
                    self.logger.error(f'Failed to initialize ad-hoc LangGraph app: {e}')
                    self.adhoc_graph_app = None

    def _get_response_matcher(self, client_id: str, dataset_id: Optional[str]=None) -> ResponseMatcher:
        return ResponseMatcher(client_id=client_id, dataset_id=dataset_id)

    @staticmethod
    def _has_meaningful_content(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, dict):
            return any((OrchestratorManager._has_meaningful_content(item) for item in value.values()))
        if isinstance(value, (list, tuple, set)):
            return any((OrchestratorManager._has_meaningful_content(item) for item in value))
        if isinstance(value, bool):
            return value
        return bool(value)

    @staticmethod
    def _estimate_payload_size(value: Any) -> int:
        try:
            return len(json.dumps(value, default=str, ensure_ascii=False))
        except Exception:
            return 0

    @classmethod
    def _should_replace_graph_response(cls, current: Any, candidate: Any) -> bool:
        current_has_content = cls._has_meaningful_content(current)
        candidate_has_content = cls._has_meaningful_content(candidate)
        if candidate_has_content and (not current_has_content):
            return True
        if candidate_has_content and current_has_content:
            return cls._estimate_payload_size(candidate) >= cls._estimate_payload_size(current)
        return False

    def _hydrate_state_from_registry_payload(self, state: StreamState, payload: Dict[str, Any]) -> list[tuple[str, Dict[str, Any]]]:
        if not isinstance(payload, dict) or not payload:
            return []
        if payload.get('code'):
            state.code = payload['code']
        if payload.get('agent_inputs'):
            state.agent_inputs.update(payload['agent_inputs'])
        if payload.get('agent_token_usage'):
            state.agent_token_usage.update(payload['agent_token_usage'])
        executor_candidate = payload.get('executor_response')
        if self._should_replace_graph_response(state.executor_response, executor_candidate):
            state.executor_response = executor_candidate
        if 'executor_error_text' in payload:
            candidate_error_text = payload.get('executor_error_text', '')
            if self._has_meaningful_content(candidate_error_text) or not state.executor_error_text:
                state.executor_error_text = candidate_error_text
        if payload.get('executor_agent_task_id'):
            state.executor_agent_task_id = payload['executor_agent_task_id']
        business_candidate = payload.get('business_response')
        if self._should_replace_graph_response(state.business_response, business_candidate):
            state.business_response = business_candidate
        if payload.get('business_agent_task_id'):
            state.business_agent_task_id = payload['business_agent_task_id']
        results: list[tuple[str, Dict[str, Any]]] = []
        if not getattr(state, '_executor_emitted', False) and (self._has_meaningful_content(state.executor_response) or self._has_meaningful_content(state.executor_error_text)):
            state._executor_emitted = True
            results.append(('executor', {'executor_response': state.executor_response, 'executor_error_text': state.executor_error_text, 'executor_agent_task_id': state.executor_agent_task_id, 'data_science_mode': bool(payload.get('data_science_mode')), 'data_analyst_mode': bool(payload.get('data_analyst_mode'))}))
        if not getattr(state, '_business_emitted', False) and self._has_meaningful_content(state.business_response):
            state._business_emitted = True
            results.append(('business', {'business_response': state.business_response, 'business_agent_task_id': state.business_agent_task_id}))
        return results

    async def stream(self, user_input: Dict[str, Any]) -> AsyncGenerator[str, None]:
        state, check_cancelled, sse_factory = await self._initialize_stream(user_input)
        
        # Intercept common greetings
        user_input_text = (user_input.get('input') or '').strip().lower()
        if user_input_text in ['hi', 'hello', 'hey', 'hii', 'greetings', 'sup', 'hola', 'namaste']:
            yield sse_factory('start', {'message': 'orchestration_started'})
            yield sse_factory('router_complete', {'route_decision': 'chat', 'relevant_tables': [], 'table_count': 0, 'query_routing_type': 'chat'})
            yield sse_factory('business', {'business_response': 'Hello! I am Elytics. How can I assist you with your analytics today?'})
            state.end_time = utcnow()
            total_time = state.calculate_total_time()
            greeting_response = 'Hello! I am Elytics. How can I assist you with your analytics today?'
            state.business_response = {'analysis': greeting_response}
            final_response = {'response': greeting_response, 'business_response': {'analysis': greeting_response}, 'total_time_seconds': total_time, 'latency_breakdown': {}, 'session_id': state.session_id, 'run_id': state.run_id, 'user_id': state.user_id}
            yield sse_factory('final', final_response)
            yield sse_factory('end', {'message': 'orchestration_complete', 'total_time_seconds': total_time, 'latency_breakdown': {}})
            asyncio.create_task(self.mongo_manager.save_conversation_async(formatted_response=final_response, user_id=state.user_id, input_text=state.input, userInput=state.user_input, state=state.to_dict(), status='completed', pending_conv_id=state.conversation_id))
            return

        run_in_background = user_input.get('run_in_background', False)
        bg_signal = register_background_signal(state.run_id)
        event_queue: asyncio.Queue = asyncio.Queue()
        graph_done = asyncio.Event()
        graph_error: Dict[str, Any] = {}
        graph_task: Optional[asyncio.Task] = None
        try:
            if not state.adhoc_mode:
                await self.mongo_manager.connect()
                _creds_svc = DBCredentialsService(self.mongo_manager.db)
                _requested = (user_input.get('dataset_id') or '').strip() or None
                if _requested:
                    _creds = await _creds_svc.get_credentials(state.client_id, db_type=None, decrypt_password=False, dataset_id=_requested)
                    if not _creds or not _creds.get('is_enabled', True):
                        error_detail = 'Invalid or disabled dataset_id for this client.'
                        yield sse_factory('error', {'detail': error_detail, 'error_type': 'invalid_dataset'})
                        state.end_time = utcnow()
                        state.calculate_total_time()
                        asyncio.create_task(self.mongo_manager.save_conversation_async(formatted_response={}, user_id=state.user_id, input_text=state.input, userInput=state.user_input, state=state.to_dict(), status='error', error=error_detail, pending_conv_id=state.conversation_id))
                        return
                    state.dataset_id = str(_creds.get('dataset_id') or _requested)
                else:
                    _active = await _creds_svc.get_active_datasets(state.client_id)
                    if not _active:
                        self.logger.warning('DB credentials missing for client_id=%s — aborting stream.', state.client_id)
                        error_detail = 'Database service is down. Please configure your database credentials.'
                        yield sse_factory('error', {'detail': error_detail, 'error_type': 'db_credentials_missing'})
                        state.end_time = utcnow()
                        state.calculate_total_time()
                        asyncio.create_task(self.mongo_manager.save_conversation_async(formatted_response={}, user_id=state.user_id, input_text=state.input, userInput=state.user_input, state=state.to_dict(), status='error', error=error_detail, pending_conv_id=state.conversation_id))
                        return
                    state.dataset_id = str(_active[0].get('dataset_id') or '')
                    _creds = await _creds_svc.get_credentials(state.client_id, db_type=None, decrypt_password=False, dataset_id=state.dataset_id or None)
                if _creds is None:
                    error_detail = 'Database service is down. Please configure your database credentials.'
                    yield sse_factory('error', {'detail': error_detail, 'error_type': 'db_credentials_missing'})
                    state.end_time = utcnow()
                    state.calculate_total_time()
                    asyncio.create_task(self.mongo_manager.save_conversation_async(formatted_response={}, user_id=state.user_id, input_text=state.input, userInput=state.user_input, state=state.to_dict(), status='error', error=error_detail, pending_conv_id=state.conversation_id))
                    return
                _db_type = _creds.get('db_type')
                try:
                    _store_local = require_store_in_local(_creds)
                except RuntimeError as e:
                    error_detail = str(e)
                    yield sse_factory('error', {'detail': error_detail, 'error_type': 'db_config_missing'})
                    state.end_time = utcnow()
                    state.calculate_total_time()
                    asyncio.create_task(self.mongo_manager.save_conversation_async(formatted_response={}, user_id=state.user_id, input_text=state.input, userInput=state.user_input, state=state.to_dict(), status='error', error=error_detail, pending_conv_id=state.conversation_id))
                    return
                _ssh_cfg = (_creds.get('additional_params') or {}).get('ssh') or {}
                _ssh_enabled = bool(_ssh_cfg.get('enabled'))
                self.logger.info('mode_selection client_id=%s dataset_id=%s db_type=%s store_in_local=%s ssh_enabled=%s', state.client_id, state.dataset_id, _db_type, _store_local, _ssh_enabled)
                if _ssh_enabled and _store_local and (_db_type == 'postgres'):
                    yield sse_factory('error', {'detail': "Selected SSH Postgres dataset is configured for local/parquet mode (store_in_local=true). Disable 'Store data locally' for live SQL mode.", 'error_type': 'ssh_live_mode_misconfigured'})
                    return
                _needs_parquet_assets = _db_type == 'file_upload' or _store_local
                if _needs_parquet_assets:
                    self.logger.info('parquet_mode_selected client_id=%s dataset_id=%s reason=%s', state.client_id, state.dataset_id, 'db_type=file_upload' if _db_type == 'file_upload' else 'store_in_local=true')
                if _needs_parquet_assets:
                    _datasets_dir = assets_datasets_dir(state.client_id, state.dataset_id or None)
                    if not _datasets_dir.exists():
                        _frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
                        _support_email = os.getenv('SUPPORT_EMAIL', 'support@coresight.ai')
                        _admin_link = f'{_frontend_url}/admin?tab=database'
                        self.logger.warning('Dataset assets directory missing for client_id=%s dataset_id=%s (expected %s) — aborting stream.', state.client_id, state.dataset_id, _datasets_dir)
                        yield sse_factory('error', {'detail': f'Dataset files could not be found, please retry uploading file / adding db credentials at {_admin_link}. If it fails please contact {_support_email}', 'error_type': 'assets_not_found'})
                        return
                else:
                    _assets_dir = Path(f'assets/clients/{state.client_id}')
                    if not _assets_dir.exists():
                        _assets_dir.mkdir(parents=True, exist_ok=True)
                        self.logger.info('Created missing assets directory for live mode client_id=%s at %s', state.client_id, _assets_dir)
            llm_client = await self._init_llm_client(state)
            if llm_client is None:
                yield sse_factory('llm_error', {'message': f'Failed to initialize LLM configuration for client {state.client_id}', 'type': 'error'})
                return
            if not llm_client._initialized and llm_client._init_error:
                yield sse_factory('llm_warning', {'message': str(llm_client._init_error), 'type': 'warning', 'fallback_provider': llm_client.default_provider, 'fallback_model': llm_client.default_model})
            self.logger.debug('Initialized shared LLMClient for graph | client_id=%s | provider=%s | model=%s', state.client_id, llm_client.default_provider, llm_client.default_model)
            yield sse_factory('start', {'message': 'orchestration_started'})
            active_graph = self.graph_app
            if state.adhoc_mode and self.adhoc_graph_app:
                active_graph = self.adhoc_graph_app
                self.logger.info('Using ad-hoc graph for session %s', state.session_id)
            if not active_graph:
                self.logger.error('Legacy orchestrator path is retired. Enable USE_LANGGRAPH to proceed.')
                yield sse_factory('error', {'detail': 'Legacy orchestrator path retired. Set USE_LANGGRAPH=true.'})
                return
            initial_state, config = self._build_graph_config(state, llm_client)
            state.conversation_id = await self.mongo_manager.save_conversation_pending_async(run_id=state.run_id, user_id=state.user_id, client_id=state.client_id, session_id=state.session_id, input_text=state.input, dataset_id=state.dataset_id or None, adhoc_quick_upload=state.adhoc_mode, adhoc_file_snapshot=state.adhoc_file_metadata or None)
            graph_task = asyncio.create_task(self._graph_task(initial_state, config, state, event_queue, graph_done, graph_error, active_graph))
            transitioned_to_background = False
            _last_heartbeat = time.time()
            _HEARTBEAT_INTERVAL = 30.0
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    if bg_signal.is_set() and (not transitioned_to_background):
                        sse_events = await self._attempt_background_transition(state, event_queue, graph_task, graph_done, graph_error, sse_factory)
                        if sse_events is not None:
                            transitioned_to_background = True
                            for sse in sse_events:
                                yield sse
                            return
                        else:
                            bg_signal.clear()
                    if graph_done.is_set() and event_queue.empty():
                        break
                    await check_cancelled()
                    _now = time.time()
                    if _now - _last_heartbeat >= _HEARTBEAT_INTERVAL:
                        yield sse_factory('heartbeat', {'timestamp': utcnow().isoformat(), 'phase': state.route_decision or 'processing'})
                        _last_heartbeat = _now
                    continue
                if event is None:
                    break
                event_type, event_data = event
                if bg_signal.is_set() and (not transitioned_to_background):
                    await event_queue.put(event)
                    sse_events = await self._attempt_background_transition(state, event_queue, graph_task, graph_done, graph_error, sse_factory)
                    if sse_events is not None:
                        transitioned_to_background = True
                        for sse in sse_events:
                            yield sse
                        return
                    else:
                        bg_signal.clear()
                if event_type == 'router_complete' and (not transitioned_to_background):
                    yield sse_factory(event_type, event_data)
                    self._update_state_from_event(state, event_type, event_data)
                    should_bg = self._should_run_background(route_decision=event_data.get('route_decision', ''), query_routing_type=event_data.get('query_routing_type', ''), query=state.input, user_requested=run_in_background)
                    if should_bg:
                        sse_events = await self._attempt_background_transition(state, event_queue, graph_task, graph_done, graph_error, sse_factory, route_decision=event_data.get('route_decision', ''), query_routing_type=event_data.get('query_routing_type', ''), message="This analysis will run in the background. You'll be notified when it's ready.")
                        if sse_events is not None:
                            transitioned_to_background = True
                            for sse in sse_events:
                                yield sse
                            return
                    continue
                if event_type == 'fatal_error':
                    self.logger.warning('Fatal graph error for client_id=%s error_type=%s — aborting stream.', state.client_id, event_data.get('error_type'))
                    yield sse_factory('error', {'detail': event_data.get('detail', 'A fatal error occurred.'), 'error_type': event_data.get('error_type', 'assets_not_found')})
                    return
                yield sse_factory(event_type, event_data)
                self._update_state_from_event(state, event_type, event_data)
                await check_cancelled()
            if graph_error:
                raise Exception(graph_error.get('error', 'Graph execution failed'))
            state.end_time = utcnow()
            total_time = state.calculate_total_time()
            self.logger.debug('Total execution time for session %s: %.2f seconds', state.session_id, total_time)
            final_response = self._finalize_stream(state)
            if state.use_cached_result:
                final_response['cached'] = True
            final_response['total_time_seconds'] = total_time
            latency_breakdown = self._build_latency_breakdown(state, total_time)
            final_response['latency_breakdown'] = latency_breakdown
            yield sse_factory('final', final_response)
            yield sse_factory('end', {'message': 'orchestration_complete', 'total_time_seconds': total_time, 'latency_breakdown': latency_breakdown})
            self.logger.debug('Saving conversation with agent_inputs keys: %s, agent_token_usage keys: %s', list(state.agent_inputs.keys()) if state.agent_inputs else [], list(state.agent_token_usage.keys()) if state.agent_token_usage else [])
            asyncio.create_task(self.mongo_manager.save_conversation_async(formatted_response=final_response, user_id=state.user_id, input_text=state.input, userInput=state.user_input, state=state.to_dict(), status='completed', pending_conv_id=state.conversation_id))
        except asyncio.CancelledError:
            if transitioned_to_background:
                try:
                    from db_config.mongo_server import get_db
                    from domains.conversation.service import ConversationService
                    db = await get_db()
                    await ConversationService(db).cancel_background(state.run_id, state.client_id)
                except Exception:
                    pass
            raise
        except Exception as e:
            state.end_time = utcnow()
            total_time = state.calculate_total_time()
            self.logger.error(f'Streaming orchestration error after {total_time:.2f} seconds: {e}', exc_info=True)
            error_type = graph_error.get('error_type') if graph_error else None
            retry_after = graph_error.get('retry_after_seconds') if graph_error else None
            if not error_type:
                error_type, retry_after = _classify_exc_for_sse(e)
            error_msg_lower = str(e).lower()
            if 'llm' in error_msg_lower or 'api_key' in error_msg_lower or 'quota' in error_msg_lower or 'genai' in error_msg_lower or 'google.api_core.exceptions' in error_msg_lower:
                safe_detail = "Language model service is unavailable or misconfigured. Please check API keys."
            elif 'db' in error_msg_lower or 'connection' in error_msg_lower or 'mongo' in error_msg_lower or 'postgres' in error_msg_lower or 'pymongo' in error_msg_lower:
                safe_detail = "Database service is down. Please configure your database credentials."
            else:
                safe_detail = "An internal service error occurred. Please try again later."
                
            error_payload: Dict[str, Any] = {'detail': safe_detail, 'total_time_seconds': total_time, 'error_type': error_type}
            if retry_after is not None:
                error_payload['retry_after_seconds'] = retry_after
            yield sse_factory('error', error_payload)
            asyncio.create_task(self.mongo_manager.save_conversation_async(formatted_response={}, user_id=state.user_id, input_text=state.input, userInput=state.user_input, state=state.to_dict(), status='error', error=str(e), pending_conv_id=state.conversation_id))
        finally:
            if graph_task and (not graph_task.done()) and (not transitioned_to_background):
                graph_task.cancel()
                try:
                    await graph_task
                except (asyncio.CancelledError, Exception):
                    pass
                if state.conversation_id:
                    try:
                        from db_config.mongo_server import get_db
                        from domains.conversation.service import ConversationService
                        db = await get_db()
                        await ConversationService(db).cancel(state.conversation_id, state.client_id)
                        self.logger.info('Marked orphaned conversation %s as cancelled (client disconnect)', state.conversation_id)
                    except Exception:
                        self.logger.warning('Failed to mark orphaned conversation %s as cancelled', state.conversation_id)
            cancellation_manager.cleanup(state.session_id)
            cleanup_background_signal(state.run_id)
            self.logger.debug('Removed stream reference for session %s', state.session_id)

    async def _graph_task(self, initial_state: Dict[str, Any], config: Dict[str, Any], state: StreamState, event_queue: asyncio.Queue, graph_done: asyncio.Event, graph_error: Dict[str, Any], graph_app=None) -> None:
        app = graph_app or self.graph_app
        try:
            async for event in app.astream_events(initial_state, config=config, version='v2'):
                processed = self._process_graph_event(event, state)
                for item in processed:
                    await event_queue.put(item)
            late_payload = pop_graph_final_payload(state.run_id)
            late_events = self._hydrate_state_from_registry_payload(state, late_payload)
            if late_events:
                self.logger.info('Recovered late graph payloads from registry | run_id=%s | events=%s', state.run_id, [event_type for event_type, _ in late_events])
                for item in late_events:
                    await event_queue.put(item)
            try:
                question_to_save = state.enhanced_question if state.enhanced_question else state.input
                session_memory.update_last_context(session_id=state.session_id, previous_user_question=question_to_save, previous_enhanced_instruction=state.route_decision, is_follow_up=False, enhanced_question=state.enhanced_question or '')
            except Exception:
                pass
        except asyncio.CancelledError:
            self.logger.debug('Graph task cancelled for session %s', state.session_id)
        except Exception as e:
            graph_error['error'] = str(e)
            error_type, retry_after = _classify_exc_for_sse(e)
            graph_error['error_type'] = error_type
            if retry_after is not None:
                graph_error['retry_after_seconds'] = retry_after
            self.logger.error(f'Graph task failed: {e}', exc_info=True)
        finally:
            discard_graph_final_payload(state.run_id)
            await event_queue.put(None)
            graph_done.set()

    def _process_graph_event(self, event: Dict[str, Any], state: StreamState) -> list:
        results = []
        kind = event.get('event', '')
        name = event.get('name', '')

        def _flatten_event_payloads(payload: Any) -> list[Dict[str, Any]]:
            fragments: list[Dict[str, Any]] = []
            if isinstance(payload, dict):
                fragments.append(payload)
                for value in payload.values():
                    fragments.extend(_flatten_event_payloads(value))
            elif isinstance(payload, (list, tuple)):
                for item in payload:
                    fragments.extend(_flatten_event_payloads(item))
            return fragments

        def _has_content(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                return bool(value.strip())
            if isinstance(value, (list, tuple)):
                return any((_has_content(item) for item in value))
            if isinstance(value, dict):
                return any((_has_content(item) for item in value.values()))
            if isinstance(value, bool):
                return value
            return bool(value)

        def _payload_size(value: Any) -> int:
            try:
                return len(json.dumps(value, default=str, ensure_ascii=False))
            except Exception:
                return 0

        def _should_replace_response(current: Any, candidate: Any) -> bool:
            current_has_content = _has_content(current)
            candidate_has_content = _has_content(candidate)
            if candidate_has_content and (not current_has_content):
                return True
            if candidate_has_content and current_has_content:
                return _payload_size(candidate) >= _payload_size(current)
            return False

        def _is_meaningful_executor_payload(payload: Dict[str, Any]) -> bool:
            executor_response = payload.get('executor_response')
            if _has_content(executor_response):
                return True
            if _has_content(payload.get('executor_error_text')):
                return True
            if 'executor_response' in payload and (payload.get('data_science_mode') or payload.get('data_analyst_mode')):
                return True
            return False

        def _is_meaningful_business_payload(payload: Dict[str, Any]) -> bool:
            business_response = payload.get('business_response')
            return _has_content(business_response)

        def _capture_executor_payload(payload: Dict[str, Any]) -> None:
            if 'executor_response' in payload:
                candidate_response = payload.get('executor_response', {})
                if _should_replace_response(state.executor_response, candidate_response):
                    state.executor_response = candidate_response
            if 'executor_error_text' in payload:
                candidate_error_text = payload.get('executor_error_text', '')
                if _has_content(candidate_error_text) or not state.executor_error_text:
                    state.executor_error_text = candidate_error_text
            state.executor_agent_task_id = payload.get('executor_agent_task_id', getattr(state, 'executor_agent_task_id', '') or f'coder_{state.run_id}')
            if 'code' in payload:
                state.code = payload.get('code', '')
            if payload.get('agent_inputs'):
                state.agent_inputs.update(payload['agent_inputs'])
            if payload.get('agent_token_usage'):
                state.agent_token_usage.update(payload['agent_token_usage'])

        def _capture_business_payload(payload: Dict[str, Any]) -> None:
            if 'business_response' in payload:
                candidate_response = payload.get('business_response', {})
                if _should_replace_response(state.business_response, candidate_response):
                    state.business_response = candidate_response
            state.business_agent_task_id = payload.get('business_agent_task_id', getattr(state, 'business_agent_task_id', '') or f'business_{state.run_id}')
            if payload.get('agent_inputs'):
                state.agent_inputs.update(payload['agent_inputs'])
            if payload.get('agent_token_usage'):
                state.agent_token_usage.update(payload['agent_token_usage'])

        def _capture_final_payloads_from_fragments(fragments: list[Dict[str, Any]], *, prefer_node_name: str, allow_emit: bool) -> None:
            nonlocal results
            for fragment in fragments:
                if _is_meaningful_executor_payload(fragment) and (prefer_node_name == 'any' or name == prefer_node_name or 'executor_response' in fragment or _has_content(fragment.get('executor_error_text'))):
                    _capture_executor_payload(fragment)
                    if allow_emit and (not getattr(state, '_executor_emitted', False)):
                        state._executor_emitted = True
                        results.append(('executor', {'executor_response': state.executor_response, 'executor_error_text': state.executor_error_text, 'executor_agent_task_id': state.executor_agent_task_id, 'data_science_mode': fragment.get('data_science_mode', False), 'data_analyst_mode': fragment.get('data_analyst_mode', False)}))
                if _is_meaningful_business_payload(fragment) and (prefer_node_name == 'any' or name == prefer_node_name or 'business_response' in fragment):
                    _capture_business_payload(fragment)
                    if allow_emit and (not getattr(state, '_business_emitted', False)):
                        state._business_emitted = True
                        results.append(('business', {'business_response': state.business_response, 'business_agent_task_id': state.business_agent_task_id}))
        if kind == 'on_chain_start':
            if name == 'guard_node':
                results.append(('router_start', {'message': 'Checking query...', 'isPreparing': True}))
            elif name == 'router_node':
                pass
            elif name == 'scout_node':
                results.append(('scout_start', {'message': 'Analyzing data sources...', 'isPreparing': True}))
            elif name == 'coder_node':
                results.append(('coder_start', {'message': 'Generating analysis...', 'isPreparing': True}))
            elif name == 'narrator_node':
                results.append(('narrator_start', {'message': 'Generating insights...', 'isPreparing': True}))
        elif kind == 'on_chain_stream':
            chunk = event.get('data', {}).get('chunk', {})
            stream_fragments = _flatten_event_payloads(chunk)
            _capture_final_payloads_from_fragments(stream_fragments, prefer_node_name='any', allow_emit=True)
            if name == 'scout_node' and chunk.get('scout_progress'):
                results.append(('scout_progress', {'tables_scouted': chunk.get('tables_scouted'), 'tables_total': chunk.get('tables_total')}))
            elif name == 'coder_node':
                if chunk.get('fatal_error'):
                    results.append(('fatal_error', {'detail': chunk.get('detail', 'A fatal error occurred.'), 'error_type': chunk.get('error_type', 'assets_not_found')}))
                elif chunk.get('cell_code_token'):
                    results.append(('cell_code_token', {'step_num': chunk.get('step_num'), 'delta': chunk.get('delta', ''), 'attempt': chunk.get('attempt', 1)}))
                elif chunk.get('cell_status'):
                    results.append(('cell_status', {'message': chunk.get('message', '')}))
                else:
                    for cell_key in ('cell_start', 'cell_code', 'cell_result', 'cell_retry', 'cell_complete', 'cell_failed'):
                        if chunk.get(cell_key):
                            data = {}
                            if cell_key == 'cell_start':
                                data = {'step_num': chunk.get('step_num'), 'total_steps': chunk.get('total_steps'), 'description': chunk.get('description', ''), 'thinking': chunk.get('thinking', ''), 'details': chunk.get('details', [])}
                            elif cell_key == 'cell_code':
                                data = {'step_num': chunk.get('step_num'), 'code': chunk.get('code', ''), 'attempt': chunk.get('attempt', 1)}
                            elif cell_key == 'cell_result':
                                data = {'step_num': chunk.get('step_num'), 'success': chunk.get('success', True), 'stdout': chunk.get('stdout', ''), 'error': chunk.get('error'), 'attempt': chunk.get('attempt', 1)}
                            elif cell_key == 'cell_retry':
                                data = {'step_num': chunk.get('step_num'), 'attempt': chunk.get('attempt', 1), 'error': chunk.get('error', '')}
                            elif cell_key == 'cell_complete':
                                data = {'step_num': chunk.get('step_num'), 'available_variables': chunk.get('available_variables', [])}
                            elif cell_key == 'cell_failed':
                                data = {'step_num': chunk.get('step_num'), 'last_error': chunk.get('last_error', '')}
                            results.append((cell_key, data))
                            break
            elif name == 'narrator_node':
                token = chunk.get('business_token')
                if token:
                    results.append(('business_token', {'token': token}))
        elif kind == 'on_chain_end':
            output = event.get('data', {}).get('output')
            if not output:
                return results
            end_fragments = _flatten_event_payloads(output)
            _capture_final_payloads_from_fragments(end_fragments, prefer_node_name='any', allow_emit=True)
            if isinstance(output, dict) and output.get('graph_latency_ms'):
                state.graph_latency_ms.update(output['graph_latency_ms'])
            if name == 'guard_node':
                if output.get('route_decision'):
                    state.route_decision = output['route_decision']
                state.guard_status = output.get('guard_status', '')
                state.guard_category = output.get('guard_category', '')
                if output.get('agent_token_usage'):
                    state.agent_token_usage.update(output['agent_token_usage'])
                if output.get('terminate_graph'):
                    results.append(('router_complete', {'route_decision': 'irrelevant', 'relevant_tables': [], 'table_count': 0, 'query_routing_type': 'irrelevant', 'guard_status': state.guard_status, 'guard_category': state.guard_category}))
            elif name == 'router_node':
                state.route_decision = output.get('route_decision', '')
                state.relevant_tables = output.get('relevant_tables', [])
                if 'enhanced_question' in output:
                    state.enhanced_question = output['enhanced_question']
                if output.get('active_persona'):
                    state.active_persona = output['active_persona']
                if output.get('agent_inputs'):
                    state.agent_inputs.update(output['agent_inputs'])
                if output.get('agent_token_usage'):
                    state.agent_token_usage.update(output['agent_token_usage'])
                results.append(('router_complete', {'route_decision': state.route_decision, 'relevant_tables': state.relevant_tables, 'table_count': len(state.relevant_tables), 'query_routing_type': output.get('query_routing_type', ''), 'guard_status': output.get('guard_status', state.guard_status), 'guard_category': output.get('guard_category', state.guard_category)}))
            elif name == 'scout_node':
                if output.get('agent_inputs'):
                    state.agent_inputs.update(output['agent_inputs'])
                if output.get('agent_token_usage'):
                    state.agent_token_usage.update(output['agent_token_usage'])
                results.append(('scout_complete', {'tables_analyzed': output.get('tables_analyzed', 0)}))
        return results

    def _should_run_background(self, route_decision: str, query_routing_type: str, query: str, user_requested: bool=False) -> bool:
        from config.system_config import ENABLE_BACKGROUND_JOBS, ML_KEYWORDS
        if not ENABLE_BACKGROUND_JOBS:
            return False
        if user_requested:
            return True
        if query_routing_type == 'data_scientist':
            query_lower = query.lower()
            return any((kw in query_lower for kw in ML_KEYWORDS))
        return False

    @staticmethod
    def _estimate_duration(route_decision: str, query_routing_type: str) -> int:
        if query_routing_type == 'data_scientist':
            return 600
        if route_decision == 'complex':
            return 180
        return 120

    async def _transition_to_background(self, state: StreamState, route_decision: str='', query_routing_type: str='') -> bool:
        try:
            from db_config.mongo_server import get_db
            from config.system_config import BACKGROUND_JOB_CONFIG
            from domains.conversation.service import ConversationService, BackgroundLimitError
            from services import redis_job_store
            db = await get_db()
            conv_service = ConversationService(db)
            estimated = self._estimate_duration(route_decision or state.route_decision or 'unknown', query_routing_type or state.route_decision or 'unknown')
            await conv_service.transition_to_background(run_id=state.run_id, client_id=state.client_id, estimated_duration_seconds=estimated, max_concurrent=BACKGROUND_JOB_CONFIG['max_concurrent_per_client'])
            now = utcnow().isoformat()
            await redis_job_store.store_active_job(run_id=state.run_id, client_id=state.client_id, user_id=state.user_id, session_id=state.session_id, input_text=state.input, route_decision=route_decision or state.route_decision or '', estimated_duration_seconds=estimated, created_at=now, started_at=now)
            return True
        except BackgroundLimitError as e:
            self.logger.warning(f'Background limit reached: {e}. Running synchronously.')
            return False
        except Exception as e:
            self.logger.error(f'Failed to transition to background: {e}', exc_info=True)
            return False

    async def _attempt_background_transition(self, state: StreamState, event_queue: asyncio.Queue, graph_task: asyncio.Task, graph_done: asyncio.Event, graph_error: Dict[str, Any], sse_factory: Callable, *, route_decision: str='', query_routing_type: str='', message: str="Moved to background. You'll be notified when it's ready.") -> Optional[list]:
        transitioned = await self._transition_to_background(state, route_decision=route_decision, query_routing_type=query_routing_type)
        if not transitioned:
            return None
        estimated = self._estimate_duration(route_decision or state.route_decision or 'unknown', query_routing_type or state.route_decision or 'unknown')
        asyncio.create_task(self._background_consumer(event_queue=event_queue, graph_task=graph_task, graph_done=graph_done, graph_error=graph_error, state=state))
        return [sse_factory('job_queued', {'run_id': state.run_id, 'message': message, 'estimated_duration_seconds': estimated}), sse_factory('end', {'message': 'backgrounded', 'run_id': state.run_id})]

    async def _background_consumer(self, event_queue: asyncio.Queue, graph_task: asyncio.Task, graph_done: asyncio.Event, graph_error: Dict[str, Any], state: StreamState) -> None:
        run_id = state.run_id
        bg_cancellation = cancellation_manager.register_job(run_id)
        try:
            from db_config.mongo_server import get_db
            from config.system_config import BACKGROUND_JOB_CONFIG
            from domains.conversation.service import ConversationService
            from domains.conversation.model import ConversationPhase
            from services import redis_job_store
            db = await get_db()
            conv_service = ConversationService(db)
            timeout = BACKGROUND_JOB_CONFIG['timeout_seconds']
            deadline = asyncio.get_event_loop().time() + timeout
            phase_map = {'scout_start': ConversationPhase.SCOUT, 'scout_complete': ConversationPhase.SCOUT, 'coder_start': ConversationPhase.CODER, 'cell_start': ConversationPhase.CODER, 'cell_complete': ConversationPhase.CODER, 'executor': ConversationPhase.CODER, 'narrator_start': ConversationPhase.NARRATOR, 'business': ConversationPhase.NARRATOR}
            while True:
                if bg_cancellation.is_set():
                    self.logger.info(f'Background conversation {run_id} cancelled')
                    graph_task.cancel()
                    state.completion_status = 'cancelled'
                    await conv_service.cancel_background(run_id, state.client_id)
                    await redis_job_store.mark_job_terminal(run_id=run_id, client_id=state.client_id, user_id=state.user_id)
                    if state.conversation_id:
                        await self.mongo_manager.save_conversation_async(formatted_response={}, user_id=state.user_id, input_text=state.input, userInput=state.user_input, state=state.to_dict(), status=state.completion_status, error='Cancelled by user', pending_conv_id=state.conversation_id)
                    return
                if asyncio.get_event_loop().time() > deadline:
                    self.logger.error(f'Background conversation {run_id} timed out after {timeout}s')
                    graph_task.cancel()
                    state.completion_status = 'failed'
                    await conv_service.fail_background(run_id, state.client_id, f'Timed out after {timeout} seconds')
                    await redis_job_store.mark_job_terminal(run_id=run_id, client_id=state.client_id, user_id=state.user_id)
                    if state.conversation_id:
                        await self.mongo_manager.save_conversation_async(formatted_response={}, user_id=state.user_id, input_text=state.input, userInput=state.user_input, state=state.to_dict(), status=state.completion_status, error=f'Timed out after {timeout} seconds', pending_conv_id=state.conversation_id)
                    return
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    if graph_done.is_set() and event_queue.empty():
                        break
                    continue
                if event is None:
                    break
                event_type, event_data = event
                self._update_state_from_event(state, event_type, event_data)
                if event_type in phase_map:
                    phase = phase_map[event_type]
                    message = event_data.get('message', '')
                    iteration = event_data.get('step_num', 0)
                    max_iterations = event_data.get('total_steps', 0)
                    await conv_service.update_background_progress(run_id, state.client_id, current_phase=phase, message=message, iteration=iteration, max_iterations=max_iterations)
                    await redis_job_store.update_job_progress(run_id=run_id, current_phase=phase.value, message=message, iteration=iteration, max_iterations=max_iterations)
            if graph_error:
                state.end_time = utcnow()
                state.calculate_total_time()
                state.completion_status = 'failed'
                error_msg = graph_error.get('error', 'Unknown error')
                await conv_service.fail_background(run_id, state.client_id, error_msg)
                await redis_job_store.mark_job_terminal(run_id=run_id, client_id=state.client_id, user_id=state.user_id)
                if state.conversation_id:
                    await self.mongo_manager.save_conversation_async(formatted_response={}, user_id=state.user_id, input_text=state.input, userInput=state.user_input, state=state.to_dict(), status=state.completion_status, error=error_msg, pending_conv_id=state.conversation_id)
                try:
                    from notifications.notification_service import create_notification
                    from notifications.notification_model import Notification
                    short_input = state.input[:80] + '…' if len(state.input) > 80 else state.input
                    await create_notification(Notification(client_id=state.client_id, user_id=state.user_id, type='response_error', title='Response Failed', message=f'Your query "{short_input}" could not be completed.', metadata={'run_id': run_id, 'conversation_id': state.conversation_id or ''}, target_role='any'))
                except Exception as _notify_err:
                    self.logger.warning(f'Failed to create response_error notification: {_notify_err}')
                return
            state.end_time = utcnow()
            state.calculate_total_time()
            state.completion_status = 'completed'
            final_response = self._finalize_stream(state)
            final_response['total_time_seconds'] = state.total_execution_time_seconds
            final_response['latency_breakdown'] = self._build_latency_breakdown(state, state.total_execution_time_seconds)
            await conv_service.complete_background(run_id, state.client_id)
            await redis_job_store.mark_job_terminal(run_id=run_id, client_id=state.client_id, user_id=state.user_id)
            await self.mongo_manager.save_conversation_async(formatted_response=final_response, user_id=state.user_id, input_text=state.input, userInput=state.user_input, state=state.to_dict(), status=state.completion_status, pending_conv_id=state.conversation_id)
            self.logger.info(f'Background conversation {run_id} completed in {state.total_execution_time_seconds:.1f}s')
            try:
                from notifications.notification_service import create_notification
                from notifications.notification_model import Notification
                short_input = state.input[:80] + '…' if len(state.input) > 80 else state.input
                await create_notification(Notification(client_id=state.client_id, user_id=state.user_id, type='response_generated', title='Response Ready', message=f'Your query "{short_input}" has been answered.', metadata={'run_id': run_id, 'conversation_id': state.conversation_id or ''}, target_role='any'))
            except Exception as _notify_err:
                self.logger.warning(f'Failed to create response_generated notification: {_notify_err}')
        except asyncio.CancelledError:
            self.logger.info(f'Background consumer for {run_id} was cancelled')
            state.completion_status = 'cancelled'
            try:
                from db_config.mongo_server import get_db
                from domains.conversation.service import ConversationService
                from services import redis_job_store
                db = await get_db()
                await ConversationService(db).cancel_background(run_id, state.client_id)
                await redis_job_store.mark_job_terminal(run_id=run_id, client_id=state.client_id, user_id=state.user_id)
                if state.conversation_id:
                    await self.mongo_manager.save_conversation_async(formatted_response={}, user_id=state.user_id, input_text=state.input, userInput=state.user_input, state=state.to_dict(), status=state.completion_status, error='Cancelled', pending_conv_id=state.conversation_id)
            except Exception:
                pass
        except Exception as e:
            self.logger.error(f'Background consumer for {run_id} failed: {e}', exc_info=True)
            state.completion_status = 'failed'
            try:
                from db_config.mongo_server import get_db
                from domains.conversation.service import ConversationService
                from services import redis_job_store
                db = await get_db()
                await ConversationService(db).fail_background(run_id, state.client_id, str(e))
                await redis_job_store.mark_job_terminal(run_id=run_id, client_id=state.client_id, user_id=state.user_id)
                if state.conversation_id:
                    await self.mongo_manager.save_conversation_async(formatted_response={}, user_id=state.user_id, input_text=state.input, userInput=state.user_input, state=state.to_dict(), status=state.completion_status, error=str(e), pending_conv_id=state.conversation_id)
            except Exception:
                pass
        finally:
            cancellation_manager.cleanup_job(run_id)

    def _update_state_from_event(self, state: StreamState, event_type: str, event_data: Dict[str, Any]) -> None:
        if event_type == 'router_complete':
            state.route_decision = event_data.get('route_decision', '')
            state.relevant_tables = event_data.get('relevant_tables', [])
        elif event_type == 'executor':
            state.executor_response = event_data.get('executor_response', {})
            state.executor_error_text = event_data.get('executor_error_text', '')
            state.executor_agent_task_id = event_data.get('executor_agent_task_id', '')
        elif event_type == 'business':
            state.business_response = event_data.get('business_response', {})
            state.business_agent_task_id = event_data.get('business_agent_task_id', '')

    async def _init_llm_client(self, state: StreamState):
        from util.llm_utils import LLMClient
        from db_config.database import get_mongo_manager
        mongo_manager = get_mongo_manager()
        llm_client = LLMClient(agent_name=None, client_id=state.client_id, db=mongo_manager, session_id=state.session_id, run_id=state.run_id, user_id=state.user_id)
        try:
            await llm_client._load_client_llm_config()
        except Exception as e:
            self.logger.error(f'Failed to load LLM configuration for client {state.client_id}: {e}', exc_info=True)
            return None
        return llm_client

    def _build_graph_config(self, state: StreamState, llm_client) -> tuple:
        initial_state = state.to_dict()
        initial_state.pop('llm_client', None)
        initial_state.pop('executor_response', None)
        initial_state.pop('code', None)
        initial_state.pop('business_response', None)
        safe_user = (state.user_id or 'anonymous').replace(' ', '_')
        short_run = state.run_id.split('-')[0]
        run_name = f'orchestrator:{state.session_id}:{short_run}:{safe_user}'
        config = {'configurable': {'thread_id': state.session_id, 'llm_client': llm_client}, 'run_name': run_name, 'tags': ['coresight', f'session:{state.session_id}', f'user:{safe_user}', f'run:{short_run}'], 'metadata': {'session_id': state.session_id, 'run_id': state.run_id, 'user_id': state.user_id, 'input': state.input[:500] if state.input else ''}}
        return (initial_state, config)

    async def stop_stream(self, session_id: str) -> bool:
        self.logger.debug('Received stop signal for session: %s', session_id)
        return cancellation_manager.signal(session_id)

    @staticmethod
    def _sanitize_for_json(obj):
        import math
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: OrchestratorManager._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [OrchestratorManager._sanitize_for_json(item) for item in obj]
        return obj

    def _create_sse_event(self, event_type: str, data: Dict[str, Any], session_id: str, run_id: str, event_id: int=0) -> str:
        payload = {**(data or {}), 'session_id': session_id, 'run_id': run_id}
        try:
            sanitized = self._sanitize_for_json(payload)
            json_str = json.dumps(sanitized, default=str, ensure_ascii=False)
            return f'id: {event_id}\nevent: {event_type}\ndata: {json_str}\n\n'
        except (TypeError, ValueError) as e:
            self.logger.error(f'Error serializing SSE event {event_type}: {e}, payload keys: {list(payload.keys())}')
            error_payload = {'error': 'Failed to serialize event data', 'event_type': event_type}
            return f'id: {event_id}\nevent: error\ndata: {json.dumps(error_payload)}\n\n'

    async def _initialize_stream(self, user_input: Dict[str, Any]) -> tuple[StreamState, Callable[[], Awaitable[None]], Callable[[str, Dict[str, Any]], str]]:
        session_id = user_input.get('session_id') or str(uuid.uuid4())
        user_input['session_id'] = session_id
        run_id = str(uuid.uuid4())
        user_input['run_id'] = run_id
        cancellation_event = cancellation_manager.register(session_id)
        client_id = user_input.get('client_id')
        if not client_id:
            raise ValueError('client_id is REQUIRED for multi-tenant operation. No default client exists. Every request must specify a valid client_id.')
        persona_slug = user_input.get('persona_slug')
        if persona_slug is not None:
            session_memory.set_persona(session_id, {'slug': persona_slug})
        adhoc_file = session_memory.get_adhoc_file(session_id)
        adhoc_mode = adhoc_file is not None
        adhoc_dataset_path = ''
        if adhoc_file:
            file_paths = adhoc_file.get('file_paths', [])
            adhoc_dataset_path = file_paths[0] if file_paths else ''
            self.logger.info('Ad-hoc mode detected | session=%s | file=%s', session_id, adhoc_file.get('original_filename'))
        state = StreamState(user_id=user_input['user_id'], input=user_input['input'], session_id=session_id, run_id=run_id, client_id=client_id, files=user_input.get('files'), user_input=user_input, use_business_agent=user_input.get('use_business_agent', False), adhoc_mode=adhoc_mode, adhoc_file_metadata=adhoc_file or {}, adhoc_dataset_path=adhoc_dataset_path, start_time=utcnow())

        async def check_cancelled():
            if cancellation_event.is_set():
                self.logger.info(f'Stream for session {session_id} cancelled.')
                raise asyncio.CancelledError('Stream stopped by user request.')
        _event_counter = {'n': 0}

        def sse_factory(event_type, data):
            _event_counter['n'] += 1
            return self._create_sse_event(event_type, data, session_id, run_id, _event_counter['n'])
        return (state, check_cancelled, sse_factory)

    async def _reference_check_phase(self, state: StreamState, check_cancelled: Callable[[], Awaitable[None]]) -> None:
        await check_cancelled()
        last_ctx = session_memory.get_last_context(state.session_id)
        if not last_ctx:
            self.logger.debug('New session detected: %s (fresh context)', state.session_id)
        await check_cancelled()
        user_question = state.input
        response_matcher = self._get_response_matcher(state.client_id, state.dataset_id or None)
        reference_data = await asyncio.to_thread(response_matcher.check_and_get_reference, user_question)
        if reference_data:
            state.planner_reference = format_reference_for_agent(reference_data, 'planner')
            state.python_reference = format_reference_for_agent(reference_data, 'python')
            state.business_reference = format_reference_for_agent(reference_data, 'business')
            similarity_score = reference_data.get('similarity_score', 0)
            cached_result = reference_data.get('cached_result', {})
            if similarity_score > 0.99 and cached_result:
                planner_from_cache = cached_result.get('planner_response', '')
                if isinstance(planner_from_cache, str):
                    planner_structured = {'plan': planner_from_cache if planner_from_cache else 'Analysis completed using cached data', 'is_follow_up': False, 'tables': []}
                else:
                    planner_structured = planner_from_cache if planner_from_cache else {'plan': 'Analysis completed using cached data', 'is_follow_up': False, 'tables': []}
                state.cached_result = {'planner_response': planner_structured, 'code': cached_result.get('code', '# Cached analysis - no code execution required'), 'executor_response': cached_result.get('executor_response', {'console_output': 'Cached result - no execution performed', 'dataframes': [], 'status': 'cached'}), 'business_response': cached_result.get('business_response', {'analysis': ' **Cached Analysis Results**\n\nThis response is based on a previously successful analysis for a very similar question (>99.9% similarity).\n\n **Key Insights**\n• Analysis completed using cached data\n• Results are based on historical successful execution\n• No new database queries were performed\n\n❓ **Follow-up Questions**\n• Would you like to run a fresh analysis?\n• Do you need more recent data?'})}
                state.use_cached_result = True
                self.logger.debug('Using cached result for similarity %.1f%%', similarity_score * 100)
            else:
                state.use_cached_result = False
                self.logger.debug('Reference guidance found but similarity not high enough for caching or cached data incomplete.')
        else:
            self.logger.debug('No similar question found. Proceeding without reference guidance.')

    def _build_latency_breakdown(self, state: StreamState, total_time_seconds: float) -> Dict[str, Any]:
        nodes = {k: round(float(v), 2) for k, v in (state.graph_latency_ms or {}).items()}
        total_ms = round(total_time_seconds * 1000.0, 2)
        accounted_ms = round(sum(nodes.values()), 2)
        other_ms = round(max(0.0, total_ms - accounted_ms), 2)

        def _pct(ms: float) -> float:
            return round(ms / total_ms * 100.0, 1) if total_ms > 0 else 0.0
        breakdown = {'total_wall_clock_ms': total_ms, 'accounted_ms': accounted_ms, 'other_ms': other_ms, 'nodes': {name: {'ms': ms, 'pct': _pct(ms)} for name, ms in sorted(nodes.items(), key=lambda kv: kv[1], reverse=True)}}
        summary = ' | '.join((f'{name}={ms:.0f}ms({_pct(ms):.0f}%)' for name, ms in sorted(nodes.items(), key=lambda kv: kv[1], reverse=True)))
        self.logger.info('[Latency] breakdown session=%s total=%.0fms accounted=%.0fms other=%.0fms | %s | other=%.0fms(%.0f%%)', state.session_id, total_ms, accounted_ms, other_ms, summary, other_ms, _pct(other_ms))
        return breakdown

    def _finalize_stream(self, state: StreamState) -> Dict[str, Any]:
        result = {'business_response': state.business_response, 'route_decision': state.route_decision, 'relevant_tables': state.relevant_tables, 'coder_response': state.code, 'executor_response': state.executor_response, 'executor_agent_task_id': state.executor_agent_task_id, 'business_agent_task_id': state.business_agent_task_id, 'dataset_id': state.dataset_id, 'session_id': state.session_id, 'run_id': state.run_id, 'user_id': state.user_id, 'total_execution_time_seconds': state.total_execution_time_seconds, 'start_time': state.start_time.isoformat(), 'end_time': state.end_time.isoformat() if state.end_time else None}
        if state.enhanced_question:
            result['enhanced_question'] = state.enhanced_question
            self.logger.debug('Including enhanced_question in final response for session %s', state.session_id)
        return result

    async def stream_business_only(self, data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        session_id = data.get('session_id') or str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        user_id = data.get('user_id', 'anonymous')
        input_query = data.get('input', '')
        executor_response = data.get('executor_response', {})
        summarized_response = data.get('summarized_response')
        report_id = data.get('report_id')
        motor_db = data.get('db')
        client_id = data.get('client_id')
        if not client_id:
            raise ValueError('client_id is REQUIRED for business insights generation')
        cancellation_event = cancellation_manager.register(session_id)
        _event_counter = {'n': 0}

        def sse_factory(event_type, data):
            _event_counter['n'] += 1
            return self._create_sse_event(event_type, data, session_id, run_id, _event_counter['n'])
        try:
            yield sse_factory('start', {'message': 'business_insights_started'})
            from util.llm_utils import LLMClient
            from db_config.database import get_mongo_manager
            mongo_manager = get_mongo_manager()
            llm_client = LLMClient(agent_name=None, client_id=client_id, db=mongo_manager, session_id=session_id, run_id=run_id, user_id=user_id)
            try:
                await llm_client._load_client_llm_config()
            except Exception as config_err:
                self.logger.error(f'Failed to load LLM configuration for client {client_id}: {config_err}', exc_info=True)
                yield sse_factory('llm_error', {'message': f'Failed to initialize LLM configuration: {str(config_err)}', 'type': 'error'})
                return
            from util.graph import narrator_node, State
            state: State = {'user_id': user_id, 'input': input_query, 'session_id': session_id, 'run_id': run_id, 'client_id': client_id, 'executor_response': executor_response, 'summarized_response': summarized_response, 'business_reference': '', 'llm_client': llm_client}
            yield sse_factory('business_start', {'message': 'business_started', 'isPreparing': True})
            async for update in narrator_node(state):
                if cancellation_event.is_set():
                    raise asyncio.CancelledError('Stream stopped by user request.')
                if 'business_token' in update:
                    yield sse_factory('business_token', {'token': update['business_token']})
                elif 'business_response' in update:
                    business_response = update['business_response']
                    yield sse_factory('business', {'business_response': business_response, 'business_agent_task_id': update.get('business_agent_task_id', f'business_{run_id}')})
                    try:
                        if report_id and motor_db is not None and isinstance(update.get('agent_token_usage'), dict):
                            from domains.dashboard.repository import DashboardRepository
                            from util.time_utils import utcnow
                            from util.token_usage_utils import normalize_agent_token_usage
                            normalized_agent_usage, total_token_usage = normalize_agent_token_usage(update.get('agent_token_usage') or {})
                            repo = DashboardRepository(motor_db)
                            await repo.append_report_usage_event(user_id=user_id, client_id=client_id, report_id=report_id, event={'at': utcnow(), 'action': 'insights', 'agent_token_usage': normalized_agent_usage, 'total_token_usage': total_token_usage})
                    except Exception as e:
                        self.logger.warning('Failed to persist dashboard usage event (insights) report=%s: %s', report_id, e)
            final_response = {'business_response': state.get('business_response', {}), 'session_id': session_id, 'run_id': run_id, 'user_id': user_id}
            yield sse_factory('final', final_response)
            yield sse_factory('end', {'message': 'business_insights_complete'})
        except asyncio.CancelledError:
            self.logger.debug('Business insights stream cancelled for session %s', session_id)
        except Exception as e:
            self.logger.error(f'Business insights streaming error: {e}', exc_info=True)
            error_type, retry_after = _classify_exc_for_sse(e)
            error_payload: Dict[str, Any] = {'detail': str(e), 'error_type': error_type}
            if retry_after is not None:
                error_payload['retry_after_seconds'] = retry_after
            yield sse_factory('error', error_payload)
        finally:
            cancellation_manager.cleanup(session_id)