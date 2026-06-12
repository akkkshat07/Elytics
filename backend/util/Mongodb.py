from __future__ import annotations
import os
import asyncio
import json
import uuid
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any, Callable, List
from contextlib import asynccontextmanager
from bson import encode
from dotenv import load_dotenv
from config.system_config import DEFAULT_LLM_PROVIDER, LLM_PROVIDERS
from util.time_utils import utcnow
from util.knowledge_filter import _approx_token_count
load_dotenv()

def estimate_object_size(obj: Any) -> int:
    try:
        json_str = json.dumps(obj, default=str)
        return len(json_str.encode('utf-8'))
    except Exception:
        return 0

def truncate_executor_response(executor_response: Any, max_size_mb: float=10.0) -> tuple[Any, bool]:
    if not executor_response:
        return (executor_response, False)
    max_size_bytes = int(max_size_mb * 1024 * 1024)
    current_size = estimate_object_size(executor_response)
    if current_size <= max_size_bytes:
        return (executor_response, False)
    if isinstance(executor_response, dict):
        truncated = {'console_output': executor_response.get('console_output', '')[:1000] + '... [truncated]' if executor_response.get('console_output') else '', 'status': executor_response.get('status', ''), 'dataframes': [{'name': df.get('name', ''), 'shape': df.get('shape', [0, 0]), 'columns': df.get('columns', [])[:10], 'preview': '... [data truncated for storage]'} for df in (executor_response.get('dataframes', []) or [])[:5]], 'plotly_charts': [{'title': chart.get('title', 'Chart') if isinstance(chart, dict) else 'Chart'} for chart in (executor_response.get('plotly_charts', []) or [])[:5]], 'matplotlib_images': [f'image_{i}' for i in range(len(executor_response.get('matplotlib_images', []) or []))], 'text_outputs': (executor_response.get('text_outputs', []) or [])[:3], '_truncated': True, '_original_size_mb': round(current_size / 1024 / 1024, 2)}
        return (truncated, True)
    else:
        truncated = str(executor_response)[:10000] + '... [truncated]'
        return (truncated, True)

def sanitize_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        import math
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    return obj

def normalize_conversation_doc(doc: dict) -> dict:
    ar = doc.setdefault('agent_responses', {})
    if not ar.get('planner') and doc.get('planner_response'):
        ar['planner'] = doc['planner_response']
    if not ar.get('python') and doc.get('coder_response'):
        ar['python'] = doc['coder_response']
    if not ar.get('business') and doc.get('business_response'):
        ar['business'] = doc['business_response']
    if not ar.get('executor') and doc.get('executor_response'):
        ar['executor'] = doc['executor_response']
    return doc

def _build_adhoc_file_snapshot(metadata: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(metadata, dict) or not metadata:
        return None
    original_filename = metadata.get('original_filename')
    if not original_filename:
        return None
    snapshot: Dict[str, Any] = {'original_filename': str(original_filename), 'file_size_bytes': metadata.get('file_size_bytes'), 'uploaded_at': metadata.get('uploaded_at'), 'sheet_count': metadata.get('sheet_count'), 'file_names': metadata.get('file_names') or [], 'session_id': metadata.get('session_id'), 'client_id': metadata.get('client_id')}
    return snapshot
EXTRA_TOKEN_KEYS = ['reasoning_tokens', 'cached_input_tokens', 'cache_creation_input_tokens', 'audio_input_tokens', 'audio_output_tokens', 'image_input_tokens', 'accepted_prediction_tokens', 'rejected_prediction_tokens', 'text_input_tokens', 'text_output_tokens']

def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

def _normalize_token_usage_map(token_usage_map: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, int]]:
    normalized_usage: Dict[str, Any] = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_provider_tokens = 0
    extra_totals: Dict[str, int] = {key: 0 for key in EXTRA_TOKEN_KEYS}
    for usage_key, usage in (token_usage_map or {}).items():
        if not usage:
            continue
        if not isinstance(usage, dict):
            normalized_usage[usage_key] = usage
            continue
        prompt_tokens = _safe_int(usage.get('prompt_tokens') or usage.get('input_tokens') or usage.get('prompt_token_count'))
        completion_tokens = _safe_int(usage.get('completion_tokens') or usage.get('output_tokens') or usage.get('candidates_token_count'))
        total_tokens_provider = _safe_int(usage.get('total_tokens_provider') or usage.get('total_tokens') or usage.get('total_token_count'))
        normalized = dict(usage)
        normalized['prompt_tokens'] = prompt_tokens
        normalized['completion_tokens'] = completion_tokens
        normalized['total_tokens'] = prompt_tokens + completion_tokens
        if total_tokens_provider:
            normalized['total_tokens_provider'] = total_tokens_provider
            total_provider_tokens += total_tokens_provider
        for key in EXTRA_TOKEN_KEYS:
            value = _safe_int(normalized.get(key))
            if value:
                normalized[key] = value
                extra_totals[key] += value
        normalized_usage[usage_key] = normalized
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
    totals = {'prompt_tokens': total_prompt_tokens, 'completion_tokens': total_completion_tokens, 'total_tokens': total_prompt_tokens + total_completion_tokens}
    if total_provider_tokens:
        totals['total_tokens_provider'] = total_provider_tokens
    for key, value in extra_totals.items():
        if value:
            totals[key] = value
    return (normalized_usage, totals)

def _calculate_estimated_cost_from_usage_map(normalized_usage: Dict[str, Any], key_field: str='per_agent_cost') -> Dict[str, Any]:
    try:
        from config.system_config import MODEL_PRICING
        total_cost = 0.0
        per_item_cost = {}
        for item_key, usage in (normalized_usage or {}).items():
            if not isinstance(usage, dict):
                continue
            model = usage.get('model') or ''
            pricing = MODEL_PRICING.get(model, MODEL_PRICING.get('_default', {}))
            input_cost = usage.get('prompt_tokens', 0) / 1000000 * pricing.get('input_per_1m', 0)
            output_cost = usage.get('completion_tokens', 0) / 1000000 * pricing.get('output_per_1m', 0)
            item_cost = round(input_cost + output_cost, 6)
            per_item_cost[item_key] = item_cost
            total_cost += item_cost
        return {'total_cost_usd': round(total_cost, 6), key_field: per_item_cost}
    except Exception:
        return {'total_cost_usd': 0.0, key_field: {}}

def _estimate_message_text_usage(text: str) -> Dict[str, Any]:
    estimated_tokens = _approx_token_count(text or '')
    return {'text_tokens_estimated': estimated_tokens, 'estimated': True}

def _build_response_discussion_source_snapshot(conversation_doc: Dict[str, Any], response_context: str='', parent_question: str='') -> Dict[str, Any]:
    snapshot_source = dict(conversation_doc or {})
    normalize_conversation_doc(snapshot_source)
    agent_responses = dict(snapshot_source.get('agent_responses') or {})
    executor_response, executor_truncated = truncate_executor_response(agent_responses.get('executor'), max_size_mb=4.0)
    python_response = agent_responses.get('python')
    if python_response and len(str(python_response)) > 200000:
        python_response = f'{str(python_response)[:200000]}\n... [truncated for discussion storage]'
    snapshot = {'conversation_id': str(snapshot_source.get('_id') or ''), 'run_id': snapshot_source.get('run_id'), 'session_id': snapshot_source.get('session_id'), 'input': snapshot_source.get('input'), 'status': snapshot_source.get('status'), 'route_decision': snapshot_source.get('route_decision'), 'created_at': snapshot_source.get('created_at'), 'response_context': response_context, 'parent_question': parent_question, 'agent_responses': {'planner': agent_responses.get('planner'), 'python': python_response, 'executor': executor_response, 'business': agent_responses.get('business')}, 'metadata': dict(snapshot_source.get('metadata') or {}), 'timing': dict(snapshot_source.get('timing') or {}), 'llm_config': dict(snapshot_source.get('llm_config') or {}), 'model': snapshot_source.get('model'), 'agent_inputs': dict(snapshot_source.get('agent_inputs') or {}), 'agent_token_usage': dict(snapshot_source.get('agent_token_usage') or {}), 'total_token_usage': dict(snapshot_source.get('total_token_usage') or {}), 'estimated_cost': dict(snapshot_source.get('estimated_cost') or {}), 'enhanced_question': snapshot_source.get('enhanced_question'), 'semantic_signature': snapshot_source.get('semantic_signature')}
    if executor_truncated:
        snapshot['metadata']['discussion_executor_response_truncated'] = True
    return snapshot

def _aggregate_discussion_total_usage(messages: List[Dict[str, Any]]) -> Dict[str, int]:
    usage_by_message = {}
    for message in messages or []:
        if message.get('role') != 'assistant':
            continue
        token_usage = message.get('token_usage')
        message_id = message.get('message_id') or str(uuid.uuid4())
        if isinstance(token_usage, dict) and token_usage:
            usage_by_message[message_id] = token_usage
    _, total_usage = _normalize_token_usage_map(usage_by_message)
    return total_usage

def _calculate_discussion_estimated_cost(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    usage_by_message = {}
    for message in messages or []:
        if message.get('role') != 'assistant':
            continue
        token_usage = message.get('token_usage')
        message_id = message.get('message_id') or str(uuid.uuid4())
        if isinstance(token_usage, dict) and token_usage:
            usage_by_message[message_id] = token_usage
    normalized_usage, _ = _normalize_token_usage_map(usage_by_message)
    return _calculate_estimated_cost_from_usage_map(normalized_usage, key_field='per_message_cost')

class MongoDBManager:

    def __init__(self):
        self.database_url: str = os.getenv('DATABASE_URL', 'mongodb://localhost:27017') or 'mongodb://localhost:27017'
        self.database_name: str = os.getenv('DATABASE_NAME', 'core-sight') or 'core-sight'
        self.client = None
        self.db = None
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        if not self.client:
            from db_config.mongo_server import start_mongodb_server
            shared_client = start_mongodb_server()
            if shared_client is not None:
                self.client = shared_client
                self.logger.info('MongoDBManager: reusing shared MongoDB client from mongo_server')
            else:
                self.client = AsyncIOMotorClient(self.database_url, maxPoolSize=50, minPoolSize=10, maxIdleTimeMS=60000, serverSelectionTimeoutMS=5000, connectTimeoutMS=10000, socketTimeoutMS=60000)
                self.logger.warning('MongoDBManager: shared client unavailable, created own client')
            self.db = self.client[self.database_name]
            try:
                server_info = await self.client.admin.command('isMaster')
                self.is_replica_set = 'setName' in server_info
                if self.is_replica_set:
                    self.logger.info(f"MongoDB replica set detected: {server_info.get('setName')}")
                else:
                    self.logger.warning('MongoDB is running in standalone mode. Transactions will be disabled. For production, use a replica set.')
            except Exception as e:
                self.logger.warning(f'Could not detect replica set status: {e}. Assuming standalone mode.')
                self.is_replica_set = False

    @asynccontextmanager
    async def transaction(self):
        await self.connect()
        if not getattr(self, 'is_replica_set', False):
            self.logger.debug('Skipping transaction (standalone MongoDB)')
            yield None
            return
        try:
            async with await self.client.start_session() as session:
                async with session.start_transaction():
                    yield session
        except Exception as e:
            if 'Transaction numbers' in str(e) or 'not supported' in str(e).lower():
                self.logger.warning('MongoDB transactions not supported (requires replica set). Operations will run without transaction safety.')
                yield None
            else:
                raise

    async def save_conversation_async(self, formatted_response, user_id, input_text, userInput, state, status, error=None, pending_conv_id: Optional[str]=None):
        try:
            await self.connect()
            collection = self.db.conversations
            total_time = state.get('total_execution_time_seconds', 0.0)
            start_time = state.get('start_time')
            end_time = state.get('end_time')
            session_id = state.get('session_id', userInput.get('session_id'))
            executor_response = state.get('executor_response')
            code = state.get('code', '')
            executor_truncated = False
            if executor_response:
                executor_response, executor_truncated = truncate_executor_response(executor_response, max_size_mb=10.0)
            code_truncated = False
            MAX_CODE_SIZE = 1024 * 1024
            if code and len(code) > MAX_CODE_SIZE:
                code = code[:MAX_CODE_SIZE] + '\n... [code truncated for storage]'
                code_truncated = True
                self.logger.warning(f"Code truncated from {len(state.get('code', '')) / 1024:.1f}KB to 1MB")
            active_persona = state.get('active_persona') or {}
            persona_snapshot = None
            if active_persona and active_persona.get('slug'):
                persona_snapshot = {'slug': active_persona.get('slug'), 'display_name': active_persona.get('display_name')}
            adhoc_snapshot = _build_adhoc_file_snapshot(state.get('adhoc_file_metadata'))
            conversation_data = {'run_id': state.get('run_id') or formatted_response.get('run_id'), 'user_id': user_id, 'session_id': session_id, 'dataset_id': state.get('dataset_id'), 'input': input_text, 'status': status, 'adhoc_quick_upload': bool(state.get('adhoc_mode', False)), 'route_decision': state.get('route_decision', ''), 'agent_responses': {'planner': state.get('planner_response'), 'python': code, 'executor': executor_response, 'business': state.get('business_response')}, 'metadata': {'attempts': state.get('attempts', 0), 'execution_attempts': state.get('execution_attempts', 0), 'executor_response_truncated': executor_truncated, 'code_truncated': code_truncated, 'cache_hit': state.get('cache_hit', False)}, 'timing': {'total_execution_time_seconds': total_time, 'start_time': start_time, 'end_time': end_time}, 'active_persona': persona_snapshot}
            if adhoc_snapshot:
                conversation_data['adhoc_file_snapshot'] = adhoc_snapshot
            if not pending_conv_id:
                conversation_data['created_at'] = utcnow()
            client_id = state.get('client_id') or userInput.get('client_id')
            if not client_id:
                error_msg = f'Cannot save conversation without client_id | session={session_id} | user={user_id} | This violates multi-tenant data isolation policy.'
                self.logger.error(f'CRITICAL: {error_msg}')
                raise ValueError(error_msg)
            conversation_data['client_id'] = client_id
            if not pending_conv_id:
                from services.subscription_service import check_conversation_limit
                is_allowed, current_count, limit = await check_conversation_limit(client_id, self.db)
                if not is_allowed:
                    self.logger.warning(f'Conversation limit exceeded during save | client_id={client_id} | used={current_count} | limit={limit} | session={session_id}')
                    return None
            try:
                default_config = await self.get_default_llm_config(client_id)
                if default_config and default_config.get('model'):
                    conversation_data['llm_config'] = {'config_id': default_config.get('config_id'), 'provider': default_config.get('provider'), 'model': default_config.get('model')}
                    conversation_data['model'] = default_config.get('model')
                else:
                    fallback_model = LLM_PROVIDERS.get(DEFAULT_LLM_PROVIDER, {}).get('default_model')
                    conversation_data['llm_config'] = {'config_id': None, 'provider': DEFAULT_LLM_PROVIDER, 'model': fallback_model}
                    conversation_data['model'] = fallback_model
            except Exception as e:
                self.logger.debug('Could not attach LLM config to conversation: %s', e)
            enhanced_q = state.get('enhanced_question')
            if enhanced_q:
                conversation_data['enhanced_question'] = enhanced_q
                self.logger.info(f'Saving enhanced_question to MongoDB: {enhanced_q[:100]}...')
            else:
                self.logger.info('No enhanced_question in state to save to MongoDB')
            semantic_signature = state.get('semantic_signature')
            if semantic_signature:
                conversation_data['semantic_signature'] = semantic_signature
                self.logger.info('Saving semantic_signature to MongoDB for cache feedback reuse')
            agent_inputs = state.get('agent_inputs', {})
            agent_token_usage = dict(state.get('agent_token_usage', {}) or {})

            def _safe_int(value: Any) -> int:
                if value is None:
                    return 0
                try:
                    return int(value)
                except (TypeError, ValueError):
                    try:
                        return int(float(value))
                    except (TypeError, ValueError):
                        return 0
            existing_business_usage = agent_token_usage.get('business')
            should_recover_business_usage = 'business' not in agent_token_usage or not isinstance(existing_business_usage, dict) or bool(existing_business_usage.get('missing_usage')) or (_safe_int(existing_business_usage.get('total_tokens')) <= 0)
            if state.get('business_response') and should_recover_business_usage and isinstance(agent_inputs, dict) and isinstance(agent_inputs.get('business'), dict):
                biz_inputs = agent_inputs.get('business', {})
                biz_response = state.get('business_response')
                biz_text = biz_response if isinstance(biz_response, str) else json.dumps(biz_response, default=str)
                prompt_tokens_est = _approx_token_count(f"{biz_inputs.get('system_prompt', '')}\n{biz_inputs.get('user_message', '')}")
                completion_tokens_est = _approx_token_count(biz_text or '')
                agent_token_usage['business'] = {'prompt_tokens': prompt_tokens_est, 'completion_tokens': completion_tokens_est, 'total_tokens': prompt_tokens_est + completion_tokens_est, 'provider': (state.get('llm_config') or {}).get('provider') or (conversation_data.get('llm_config') or {}).get('provider'), 'model': (state.get('llm_config') or {}).get('model') or (conversation_data.get('llm_config') or {}).get('model'), 'estimated': True, 'missing_usage_recovered': 'business_response_present_without_usage'}
            if agent_inputs:
                conversation_data['agent_inputs'] = agent_inputs
            if agent_token_usage:
                from util.token_usage_utils import normalize_agent_token_usage
                normalized_usage, total_token_usage = normalize_agent_token_usage(agent_token_usage)
                conversation_data['agent_token_usage'] = normalized_usage
                conversation_data['total_token_usage'] = total_token_usage
                try:
                    from config.system_config import MODEL_PRICING
                    total_cost = 0.0
                    per_agent_cost = {}
                    for agent_name, usage in normalized_usage.items():
                        if not isinstance(usage, dict):
                            continue
                        model = usage.get('model') or ''
                        pricing = MODEL_PRICING.get(model, MODEL_PRICING.get('_default', {}))
                        input_cost = usage.get('prompt_tokens', 0) / 1000000 * pricing.get('input_per_1m', 0)
                        output_cost = usage.get('completion_tokens', 0) / 1000000 * pricing.get('output_per_1m', 0)
                        agent_cost = round(input_cost + output_cost, 6)
                        per_agent_cost[agent_name] = agent_cost
                        total_cost += agent_cost
                    conversation_data['estimated_cost'] = {'total_cost_usd': round(total_cost, 6), 'per_agent_cost': per_agent_cost}
                except Exception as cost_err:
                    logger.warning(f'Failed to calculate cost: {cost_err}')
            if error:
                conversation_data['error'] = error
            MONGODB_MAX_SIZE = 16 * 1024 * 1024
            SAFE_SIZE_LIMIT = int(MONGODB_MAX_SIZE * 0.9)
            try:
                doc_size = len(encode(conversation_data))
                if doc_size > SAFE_SIZE_LIMIT:
                    self.logger.warning(f'Document size ({doc_size / 1024 / 1024:.2f}MB) still exceeds safe limit ({SAFE_SIZE_LIMIT / 1024 / 1024:.2f}MB) after pre-truncation. Removing executor entirely and truncating business response.')
                    conversation_data['agent_responses']['executor'] = {'_note': 'Executor response removed due to size constraints', 'status': 'truncated'}
                    business_resp = conversation_data['agent_responses'].get('business')
                    if business_resp and isinstance(business_resp, dict):
                        conversation_data['agent_responses']['business'] = {'analysis': business_resp.get('analysis', '')[:5000], '_note': 'Business response truncated due to size constraints'}
                    conversation_data['metadata']['executor_response_removed'] = True
                    conversation_data['metadata']['business_response_truncated'] = True
                    conversation_data['metadata']['original_size_mb'] = round(doc_size / 1024 / 1024, 2)
                    doc_size = len(encode(conversation_data))
                    self.logger.info(f'After aggressive truncation: {doc_size / 1024 / 1024:.2f}MB')
                else:
                    self.logger.info(f'Document size ({doc_size / 1024 / 1024:.2f}MB) is within limits. Storing execution result.')
            except Exception as size_check_error:
                self.logger.error(f'Size check failed: {size_check_error}. Applying emergency truncation.')
                conversation_data['agent_responses']['executor'] = None
                conversation_data['agent_responses']['business'] = {'analysis': 'Truncated due to size constraints'}
                conversation_data['metadata']['emergency_truncation'] = True
            if pending_conv_id:
                from bson import ObjectId
                await collection.update_one({'_id': ObjectId(pending_conv_id), 'client_id': client_id}, {'$set': {**conversation_data, 'updated_at': utcnow()}})
                self.logger.info(f"Conversation updated from pending - id={pending_conv_id}, run_id={conversation_data['run_id']}, time={total_time:.2f}s")
                return pending_conv_id
            else:
                result = await collection.insert_one(conversation_data)
                self.logger.info(f'Conversation saved - ID: {result.inserted_id}, Execution time: {total_time:.2f}s')
                return str(result.inserted_id)
        except Exception as save_error:
            self.logger.error(f"MongoDB save failed for session {state.get('session_id', 'unknown')}: {str(save_error)[:200]}")
            return None

    async def save_conversation_pending_async(self, run_id: str, user_id: str, client_id: str, session_id: str, input_text: str, route_decision: str='', dataset_id: Optional[str]=None, adhoc_quick_upload: bool=False, adhoc_file_snapshot: Optional[Dict[str, Any]]=None) -> Optional[str]:
        if not client_id:
            self.logger.error('save_conversation_pending_async: client_id is required')
            return None
        try:
            await self.connect()
            doc = {'run_id': run_id, 'user_id': user_id, 'client_id': client_id, 'session_id': session_id, 'dataset_id': dataset_id, 'input': input_text, 'status': 'pending', 'route_decision': route_decision, 'adhoc_quick_upload': bool(adhoc_quick_upload), 'created_at': utcnow(), 'agent_responses': {'planner': None, 'python': None, 'executor': None, 'business': None}, 'metadata': {}, 'timing': {}, 'llm_config': {}, 'agent_inputs': {}, 'agent_token_usage': {}, 'total_token_usage': {}, 'estimated_cost': {}}
            if adhoc_file_snapshot:
                doc['adhoc_file_snapshot'] = _build_adhoc_file_snapshot(adhoc_file_snapshot)
            result = await self.db.conversations.insert_one(doc)
            self.logger.info(f'Pending conversation saved: run_id={run_id}, id={result.inserted_id}')
            return str(result.inserted_id)
        except Exception as e:
            self.logger.error(f'Failed to save pending conversation run_id={run_id}: {e}')
            return None

    async def save_feedback_async(self, run_id: str, rating: str, comment: str, user_id: str, session=None):
        try:
            await self.connect()
            collection = self.db.conversations
            feedback_data = {'rating': rating, 'comment': comment, 'user_id': user_id, 'created_at': utcnow()}
            update_kwargs = {'session': session} if session else {}
            result = await collection.update_one({'run_id': run_id}, {'$set': {'feedback': feedback_data}}, **update_kwargs)
            if result.matched_count > 0:
                self.logger.info(f'Feedback saved for run_id: {run_id}')
                return run_id
            else:
                self.logger.warning(f'Could not find conversation to save feedback for run_id: {run_id}')
                return None
        except Exception as e:
            self.logger.error(f'Failed to save feedback for run_id {run_id}: {e}')
            return None

    async def get_response_discussion(self, *, conversation_id: str, source_run_id: str, user_id: str, client_id: str) -> Optional[Dict[str, Any]]:
        try:
            await self.connect()
            collection = self.db.response_discussions
            document = await collection.find_one({'conversation_id': conversation_id.lower(), 'source_run_id': source_run_id, 'user_id': user_id, 'client_id': client_id, 'is_deleted': {'$ne': True}})
            if not document:
                return None
            document['_id'] = str(document['_id'])
            return sanitize_floats(document)
        except Exception as e:
            self.logger.error(f'Failed to retrieve response discussion for conversation_id={conversation_id}, source_run_id={source_run_id}: {e}')
            return None

    async def save_response_discussion_turn(self, *, client_id: str, user_id: str, conversation_id: str, source_run_id: str, session_id: Optional[str], parent_question: str, response_context: str, source_conversation: Dict[str, Any], user_question: str, assistant_answer: str, assistant_usage: Optional[Dict[str, Any]]=None, llm_config: Optional[Dict[str, Any]]=None) -> Optional[Dict[str, Any]]:
        try:
            await self.connect()
            collection = self.db.response_discussions
            normalized_conversation_id = conversation_id.lower()
            now = utcnow()
            existing = await collection.find_one({'conversation_id': normalized_conversation_id, 'source_run_id': source_run_id, 'user_id': user_id, 'client_id': client_id, 'is_deleted': {'$ne': True}})
            discussion_doc: Dict[str, Any] = existing or {'client_id': client_id, 'user_id': user_id, 'session_id': session_id or source_conversation.get('session_id'), 'conversation_id': normalized_conversation_id, 'source_run_id': source_run_id, 'source_parent_question': parent_question, 'response_context': response_context, 'source_conversation_snapshot': _build_response_discussion_source_snapshot(source_conversation, response_context=response_context, parent_question=parent_question), 'status': 'active', 'created_at': now, 'updated_at': now, 'messages': [], 'total_token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, 'estimated_cost': {'total_cost_usd': 0.0, 'per_message_cost': {}}, 'llm_config': dict(llm_config or {}), 'model': (llm_config or {}).get('model'), 'metadata': {}}
            discussion_doc['updated_at'] = now
            discussion_doc['status'] = 'active'
            if session_id:
                discussion_doc['session_id'] = session_id
            if response_context:
                discussion_doc['response_context'] = response_context
            if parent_question:
                discussion_doc['source_parent_question'] = parent_question
            discussion_doc['source_conversation_snapshot'] = _build_response_discussion_source_snapshot(source_conversation, response_context=discussion_doc.get('response_context', ''), parent_question=discussion_doc.get('source_parent_question', ''))
            merged_llm_config = dict(discussion_doc.get('llm_config') or {})
            merged_llm_config.update({k: v for k, v in (llm_config or {}).items() if v is not None})
            if merged_llm_config:
                discussion_doc['llm_config'] = merged_llm_config
                discussion_doc['model'] = merged_llm_config.get('model')
            messages = list(discussion_doc.get('messages') or [])
            messages.append({'message_id': str(uuid.uuid4()), 'role': 'user', 'content': user_question, 'created_at': now, 'token_usage': _estimate_message_text_usage(user_question)})
            normalized_assistant_usage_map, _ = _normalize_token_usage_map({'discussion': assistant_usage or {}})
            assistant_message_usage = normalized_assistant_usage_map.get('discussion', {})
            if merged_llm_config:
                assistant_message_usage.setdefault('provider', merged_llm_config.get('provider'))
                assistant_message_usage.setdefault('model', merged_llm_config.get('model'))
            messages.append({'message_id': str(uuid.uuid4()), 'role': 'assistant', 'content': assistant_answer, 'created_at': now, 'token_usage': assistant_message_usage, 'provider': (merged_llm_config or {}).get('provider'), 'model': (merged_llm_config or {}).get('model')})
            discussion_doc['messages'] = messages
            discussion_doc['total_token_usage'] = _aggregate_discussion_total_usage(messages)
            discussion_doc['estimated_cost'] = _calculate_discussion_estimated_cost(messages)
            discussion_doc['metadata'] = {**dict(discussion_doc.get('metadata') or {}), 'message_count': len(messages), 'user_message_count': sum((1 for item in messages if item.get('role') == 'user')), 'assistant_message_count': sum((1 for item in messages if item.get('role') == 'assistant'))}
            if existing and existing.get('_id'):
                await collection.replace_one({'_id': existing['_id']}, discussion_doc)
                discussion_doc['_id'] = str(existing['_id'])
            else:
                result = await collection.insert_one(discussion_doc)
                discussion_doc['_id'] = str(result.inserted_id)
            return sanitize_floats(discussion_doc)
        except Exception as e:
            self.logger.error(f'Failed to save response discussion for conversation_id={conversation_id}, source_run_id={source_run_id}: {e}')
            return None

    async def get_conversation_by_run_id(self, run_id: str) -> Optional[Dict]:
        try:
            if not self.database_url:
                self.logger.error('Database URL not configured')
                return None
            await self.connect()
            collection = self.db.conversations
            conversation = await collection.find_one({'run_id': run_id})
            if conversation:
                if '_id' in conversation:
                    conversation['_id'] = str(conversation['_id'])
                normalize_conversation_doc(conversation)
                return sanitize_floats(conversation)
            else:
                self.logger.warning(f'Conversation not found for run_id: {run_id}')
                return None
        except Exception as e:
            self.logger.error(f'Failed to retrieve conversation for run_id {run_id}: {e}')
            return None

    async def get_execution_stats(self, user_id: str=None, days: int=30):
        try:
            if not self.database_url:
                self.logger.error('Database URL not configured')
                return None
            await self.connect()
            collection = self.db.conversations
            now = utcnow()
            match_criteria = {'created_at': {'$gte': now - timedelta(days=days)}, 'timing.total_execution_time_seconds': {'$exists': True}}
            if user_id:
                match_criteria['user_id'] = user_id
            pipeline = [{'$match': match_criteria}, {'$group': {'_id': None, 'avg_time': {'$avg': '$timing.total_execution_time_seconds'}, 'min_time': {'$min': '$timing.total_execution_time_seconds'}, 'max_time': {'$max': '$timing.total_execution_time_seconds'}, 'total_executions': {'$sum': 1}}}]
            result = await collection.aggregate(pipeline).to_list(length=1)
            if result:
                stats = result[0]
                return {'average_time_seconds': round(stats.get('avg_time', 0), 2), 'min_time_seconds': round(stats.get('min_time', 0), 2), 'max_time_seconds': round(stats.get('max_time', 0), 2), 'total_executions': stats.get('total_executions', 0), 'days_analyzed': days}
            else:
                return {'average_time_seconds': 0, 'min_time_seconds': 0, 'max_time_seconds': 0, 'total_executions': 0, 'days_analyzed': days}
        except Exception as e:
            self.logger.error(f'Failed to get execution stats: {e}')
            return None

    async def get_user_sessions(self, user_id: str=None, page: int=1, limit: int=20, client_id: str | None=None, is_admin: bool=False, include_user_info: bool=False):
        try:
            await self.connect()
            collection = self.db.conversations
            from bson import ObjectId
            match_filter = {}
            match_filter['is_deleted'] = {'$ne': True}
            match_filter['status'] = {'$ne': 'cancelled'}
            if user_id is not None:
                try:
                    if isinstance(user_id, str) and len(user_id) == 24:
                        obj_id = ObjectId(user_id)
                        match_filter['$or'] = [{'user_id': obj_id}, {'user_id': user_id}]
                    else:
                        match_filter['user_id'] = user_id
                except:
                    match_filter['user_id'] = user_id
            if client_id:
                match_filter['client_id'] = client_id
            pipeline = [{'$match': match_filter}, {'$sort': {'created_at': 1}}, {'$group': {'_id': '$session_id', 'user_id': {'$first': '$user_id'}, 'first_message': {'$first': '$input'}, 'first_created_at': {'$first': '$created_at'}, 'last_created_at': {'$last': '$created_at'}, 'message_count': {'$sum': 1}, 'completed_count': {'$sum': {'$cond': [{'$eq': ['$status', 'completed']}, 1, 0]}}, 'unread_count': {'$sum': {'$cond': [{'$eq': ['$is_read', False]}, 1, 0]}}, 'pending_count': {'$sum': {'$cond': [{'$and': [{'$in': ['$status', ['pending', 'running']]}, {'$ne': ['$is_background', True]}]}, 1, 0]}}}}, {'$sort': {'last_created_at': -1}}, {'$skip': (page - 1) * limit}, {'$limit': limit}]
            sessions = await collection.aggregate(pipeline).to_list(length=limit)
            count_pipeline = [{'$match': match_filter}, {'$group': {'_id': '$session_id'}}, {'$count': 'total'}]
            count_result = await collection.aggregate(count_pipeline).to_list(length=1)
            total_sessions = count_result[0]['total'] if count_result else 0
            users_map = {}
            if include_user_info:
                user_ids = set()
                for session in sessions:
                    user_id_val = session.get('user_id')
                    if user_id_val:
                        user_ids.add(str(user_id_val) if isinstance(user_id_val, ObjectId) else user_id_val)
                if user_ids:
                    user_query = []
                    for uid in user_ids:
                        if len(uid) == 24:
                            try:
                                user_query.append({'_id': ObjectId(uid)})
                            except:
                                pass
                        user_query.append({'_id': uid})
                    users_cursor = self.db.users.find({'$or': user_query if user_query else [{'_id': {'$in': list(user_ids)}}]}, {'_id': 1, 'email': 1, 'name': 1})
                    async for user in users_cursor:
                        user_id_str = str(user['_id'])
                        users_map[user_id_str] = {'email': user.get('email', 'Unknown'), 'name': user.get('name', user.get('email', 'Unknown User'))}
            formatted_sessions = []
            for session in sessions:
                session_data = {'session_id': session['_id'], 'title': session['first_message'][:100] if session.get('first_message') else 'Untitled', 'created_at': session['first_created_at'], 'last_updated': session['last_created_at'], 'message_count': session['message_count'], 'completed_count': session.get('completed_count', 0), 'has_unread': session.get('unread_count', 0) > 0, 'has_pending': session.get('pending_count', 0) > 0}
                session_user_id = str(session.get('user_id', '')) if session.get('user_id') else ''
                session_data['user_id'] = session_user_id
                if include_user_info:
                    user_info = users_map.get(session_user_id, {'email': 'Unknown', 'name': 'Unknown User'})
                    session_data['user_email'] = user_info['email']
                    session_data['user_name'] = user_info['name']
                formatted_sessions.append(session_data)
            return {'sessions': formatted_sessions, 'total': total_sessions, 'page': page, 'limit': limit}
        except Exception as e:
            self.logger.error(f'Failed to get user sessions: {e}')
            return None

    async def get_unread_session_count(self, *, client_id: str, user_id: str=None) -> int:
        try:
            await self.connect()
            collection = self.db.conversations
            from bson import ObjectId
            match_filter = {'is_read': False, 'is_deleted': {'$ne': True}, 'status': {'$ne': 'cancelled'}}
            if client_id:
                match_filter['client_id'] = client_id
            if user_id is not None:
                try:
                    if isinstance(user_id, str) and len(user_id) == 24:
                        obj_id = ObjectId(user_id)
                        match_filter['$or'] = [{'user_id': obj_id}, {'user_id': user_id}]
                    else:
                        match_filter['user_id'] = user_id
                except Exception:
                    match_filter['user_id'] = user_id
            pipeline = [{'$match': match_filter}, {'$group': {'_id': '$session_id'}}, {'$count': 'total'}]
            result = await collection.aggregate(pipeline).to_list(length=1)
            return result[0]['total'] if result else 0
        except Exception as e:
            self.logger.error(f'Failed to get unread session count: {e}')
            return 0

    async def get_session_conversations(self, user_id: str, session_id: str, *, include_pending: bool=True):
        try:
            await self.connect()
            collection = self.db.conversations
            from bson import ObjectId
            query = {'session_id': session_id}
            try:
                if isinstance(user_id, str) and len(user_id) == 24:
                    obj_id = ObjectId(user_id)
                    query['$or'] = [{'user_id': obj_id}, {'user_id': user_id}]
                else:
                    query['user_id'] = user_id
            except:
                query['user_id'] = user_id
            query['is_deleted'] = {'$ne': True}
            if include_pending:
                query['status'] = {'$ne': 'cancelled'}
            else:
                query['status'] = 'completed'
            cursor = collection.find(query).sort('created_at', 1).limit(500)
            conversations = await cursor.to_list(length=500)
            for conv in conversations:
                conv['id'] = str(conv['_id'])
                del conv['_id']
                normalize_conversation_doc(conv)
            return [sanitize_floats(conv) for conv in conversations]
        except Exception as e:
            self.logger.error(f'Failed to get session conversations: {e}')
            return None

    async def delete_session(self, user_id: str, session_id: str, session=None):
        try:
            await self.connect()
            collection = self.db.conversations
            delete_kwargs = {'session': session} if session else {}
            result = await collection.delete_many({'user_id': user_id, 'session_id': session_id}, **delete_kwargs)
            deleted_count = result.deleted_count
            self.logger.info(f'Deleted {deleted_count} conversations for session {session_id}')
            return deleted_count > 0
        except Exception as e:
            self.logger.error(f'Failed to delete session: {e}')
            return False

    async def delete_session_by_session_id(self, session_id: str, session=None) -> bool:
        try:
            await self.connect()
            collection = self.db.conversations
            delete_kwargs = {'session': session} if session else {}
            result = await collection.delete_many({'session_id': session_id}, **delete_kwargs)
            deleted_count = result.deleted_count
            self.logger.info(f'Deleted {deleted_count} conversations for session {session_id} (by session_id)')
            return deleted_count > 0
        except Exception as e:
            self.logger.error(f'Failed to delete session by session_id: {e}')
            return False

    async def delete_session_by_session_id_scoped(self, session_id: str, client_id: str | None, session=None) -> int:
        try:
            await self.connect()
            collection = self.db.conversations
            update_kwargs = {'session': session} if session else {}
            if client_id:
                filter_query = {'session_id': session_id, '$or': [{'client_id': client_id}, {'client_id': {'$exists': False}}, {'client_id': None}]}
            else:
                filter_query = {'session_id': session_id, '$or': [{'client_id': {'$exists': False}}, {'client_id': None}]}
            result = await collection.update_many(filter_query, {'$set': {'is_deleted': True, 'deleted_at': utcnow()}}, **update_kwargs)
            updated_count = result.modified_count
            self.logger.info(f'Soft delete: marked {updated_count} conversations as deleted for session {session_id} (client_id scope: {client_id})')
            return updated_count
        except Exception as e:
            self.logger.error(f'Failed to soft delete session by session_id (scoped): {e}')
            return 0

    async def update_session_title(self, user_id: str, session_id: str, title: str):
        try:
            await self.connect()
            collection = self.db.conversations
            first_conv = await collection.find_one({'user_id': user_id, 'session_id': session_id}, sort=[('created_at', 1)])
            if not first_conv:
                return False
            result = await collection.update_one({'_id': first_conv['_id']}, {'$set': {'metadata.custom_title': title}})
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f'Failed to update session title: {e}')
            return False

    async def get_session_metadata(self, session_id: str, user_id: str=None, client_id: str=None) -> Optional[Dict]:
        try:
            await self.connect()
            collection = self.db.conversations
            from bson import ObjectId
            query = {'session_id': session_id}
            if user_id:
                try:
                    if isinstance(user_id, str) and len(user_id) == 24:
                        obj_id = ObjectId(user_id)
                        query['$or'] = [{'user_id': obj_id}, {'user_id': user_id}]
                    else:
                        query['user_id'] = user_id
                except:
                    query['user_id'] = user_id
            if client_id:
                query['client_id'] = client_id
            session_conv = await collection.find_one(query, {'user_id': 1, 'client_id': 1, 'session_id': 1, '_id': 0})
            if session_conv:
                if 'user_id' in session_conv and isinstance(session_conv['user_id'], ObjectId):
                    session_conv['user_id'] = str(session_conv['user_id'])
                    self.logger.debug(f"Converted user_id ObjectId to string: {session_conv['user_id']}")
                elif 'user_id' in session_conv:
                    session_conv['user_id'] = str(session_conv['user_id'])
            return session_conv
        except Exception as e:
            self.logger.error(f'Failed to get session metadata: {e}', exc_info=True)
            return None

    async def get_conversation_by_run_id(self, run_id: str) -> Optional[Dict]:
        try:
            await self.connect()
            collection = self.db.conversations
            conversation = await collection.find_one({'run_id': run_id})
            if conversation:
                normalize_conversation_doc(conversation)
                return sanitize_floats(conversation)
            return conversation
        except Exception as e:
            self.logger.error(f'Failed to get conversation by run_id: {e}')
            return None

    async def get_llm_configurations(self, client_id: str) -> Optional[Dict]:
        try:
            await self.connect()
            collection = self.db.llm_configurations
            config_doc = await collection.find_one({'client_id': client_id})
            if config_doc and '_id' in config_doc:
                config_doc['_id'] = str(config_doc['_id'])
            return config_doc
        except Exception as e:
            self.logger.error(f'Failed to get LLM configurations for client {client_id}: {e}')
            return None

    async def save_llm_configuration(self, client_id: str, config_data: Dict[str, Any], user_id: str) -> Optional[str]:
        try:
            await self.connect()
            collection = self.db.llm_configurations
            import uuid
            config_id = config_data.get('config_id') or str(uuid.uuid4())
            config_entry = {'config_id': config_id, 'provider': config_data.get('provider'), 'model': config_data.get('model'), 'api_key': config_data.get('api_key'), 'is_default': config_data.get('is_default', False), 'is_active': config_data.get('is_active', True), 'is_platform': config_data.get('is_platform', False), 'is_deleted': False, 'created_at': config_data.get('created_at', utcnow()), 'validated_at': config_data.get('validated_at', utcnow())}
            existing = await collection.find_one({'client_id': client_id})
            if existing:
                if config_entry['is_default']:
                    await collection.update_one({'client_id': client_id}, {'$set': {'configurations.$[elem].is_default': False}}, array_filters=[{'elem.config_id': {'$ne': config_id}}])
                config_exists = any((c.get('config_id') == config_id for c in existing.get('configurations', [])))
                if config_exists:
                    result = await collection.update_one({'client_id': client_id, 'configurations.config_id': config_id}, {'$set': {'configurations.$': config_entry, 'updated_at': utcnow(), 'updated_by': user_id}})
                else:
                    result = await collection.update_one({'client_id': client_id}, {'$push': {'configurations': config_entry}, '$set': {'updated_at': utcnow(), 'updated_by': user_id}})
            else:
                new_doc = {'client_id': client_id, 'configurations': [config_entry], 'created_at': utcnow(), 'updated_at': utcnow(), 'updated_by': user_id}
                result = await collection.insert_one(new_doc)
            self.logger.info(f'LLM configuration saved for client {client_id}, config_id: {config_id}')
            return config_id
        except Exception as e:
            self.logger.error(f'Failed to save LLM configuration: {e}')
            return None

    async def set_default_llm_configuration(self, client_id: str, config_id: str) -> bool:
        try:
            await self.connect()
            collection = self.db.llm_configurations
            await collection.update_one({'client_id': client_id}, {'$set': {'configurations.$[].is_default': False}})
            result = await collection.update_one({'client_id': client_id, 'configurations.config_id': config_id}, {'$set': {'configurations.$.is_default': True, 'updated_at': utcnow()}})
            success = result.modified_count > 0
            if success:
                self.logger.info(f'Set config {config_id} as default for client {client_id}')
            return success
        except Exception as e:
            self.logger.error(f'Failed to set default LLM configuration: {e}')
            return False

    async def delete_llm_configuration(self, client_id: str, config_id: str) -> Dict[str, Any]:
        try:
            await self.connect()
            collection = self.db.llm_configurations
            config_doc = await collection.find_one({'client_id': client_id})
            if config_doc:
                for config in config_doc.get('configurations', []):
                    if config.get('config_id') == config_id:
                        if config.get('is_default') and (not config.get('is_deleted')):
                            self.logger.warning(f'Cannot delete default LLM configuration {config_id} for client {client_id}')
                            return {'success': False, 'error': 'You cannot delete currently activated LLM config. Activate another LLM config first to delete the current config.'}
                        break
            result = await collection.update_one({'client_id': client_id, 'configurations.config_id': config_id}, {'$set': {'configurations.$.is_deleted': True, 'updated_at': utcnow()}})
            success = result.modified_count > 0
            if success:
                self.logger.info(f'Soft deleted LLM configuration {config_id} for client {client_id}')
            return {'success': success}
        except Exception as e:
            self.logger.error(f'Failed to soft delete LLM configuration: {e}')
            return {'success': False, 'error': str(e)}

    async def create_default_llm_config_for_client(self, client_id: str, platform_config: Dict[str, Any]=None) -> Optional[str]:
        try:
            await self.connect()
            collection = self.db.llm_configurations
            existing = await collection.find_one({'client_id': client_id})
            if existing and existing.get('configurations'):
                self.logger.info(f'Client {client_id} already has LLM configurations, skipping default creation')
                return None
            if platform_config is None:
                meta_config = await self.get_llm_meta_config()
                if not meta_config:
                    self.logger.error('No LLM meta config in database, cannot create default config')
                    return None
                platform_configs = meta_config.get('platform_configs', [])
                default_config_id = meta_config.get('default_registration_config')
                if default_config_id:
                    platform_config = next((c for c in platform_configs if c.get('config_id') == default_config_id), None)
                    if not platform_config:
                        self.logger.warning(f"Configured default_registration_config '{default_config_id}' not found in platform_configs, falling back to first available")
                if not platform_config:
                    platform_config = platform_configs[0] if platform_configs else None
            if not platform_config:
                self.logger.error('No platform config available to create default LLM config')
                return None
            config_id = platform_config.get('config_id')
            now = utcnow()
            config_entry = {'config_id': config_id, 'provider': platform_config.get('provider'), 'model': platform_config.get('model'), 'model_label': platform_config.get('model_label'), 'api_key': None, 'is_default': True, 'is_active': True, 'is_platform': True, 'is_system': platform_config.get('is_system', True), 'is_deleted': False, 'created_at': now, 'validated_at': now}
            if existing:
                result = await collection.update_one({'client_id': client_id}, {'$push': {'configurations': config_entry}, '$set': {'updated_at': now, 'updated_by': 'system'}})
            else:
                new_doc = {'client_id': client_id, 'configurations': [config_entry], 'created_at': now, 'updated_at': now, 'updated_by': 'system'}
                result = await collection.insert_one(new_doc)
            self.logger.info(f"Created default LLM configuration for client {client_id}: {config_id} ({platform_config.get('provider')}/{platform_config.get('model')})")
            return config_id
        except Exception as e:
            self.logger.error(f'Failed to create default LLM configuration for client {client_id}: {e}')
            return None

    async def get_default_llm_config(self, client_id: str, agent_name: str=None) -> Optional[Dict]:
        try:
            await self.connect()
            collection = self.db.llm_configurations
            self.logger.info(f'Querying llm_configurations for client_id={client_id}')
            config_doc = await collection.find_one({'client_id': client_id})
            if not config_doc:
                self.logger.warning(f'No llm_configurations document found for client_id={client_id}')
                return None
            if not config_doc.get('configurations'):
                self.logger.warning(f"llm_configurations document exists but 'configurations' array is empty for client_id={client_id}")
                return None
            self.logger.info(f"Found {len(config_doc.get('configurations', []))} configuration(s) for client_id={client_id}")
            for config in config_doc.get('configurations', []):
                if config.get('is_deleted', False):
                    continue
                self.logger.info(f"Checking config: provider={config.get('provider')} | model={config.get('model')} | is_default={config.get('is_default')} | is_active={config.get('is_active')}")
                if config.get('is_default') and config.get('is_active'):
                    self.logger.info(f"✓ Found default config: {config.get('provider')}/{config.get('model')}")
                    return config
            self.logger.warning(f'No default+active config found for client_id={client_id}')
            return None
        except Exception as e:
            self.logger.error(f'Failed to get default LLM config for client {client_id}: {e}', exc_info=True)
            return None

    async def get_llm_meta_config(self) -> Optional[Dict]:
        try:
            await self.connect()
            collection = self.db.llm_meta_config
            config = await collection.find_one({'is_active': True})
            if not config:
                self.logger.warning('No active LLM meta configuration found in MongoDB')
                return None
            self.logger.info(f"Loaded LLM meta config version {config.get('version')}")
            return config
        except Exception as e:
            self.logger.error(f'Failed to get LLM meta config: {e}', exc_info=True)
            return None

    async def save_llm_meta_config(self, config_data: Dict[str, Any], updated_by: str) -> Optional[int]:
        try:
            await self.connect()
            collection = self.db.llm_meta_config
            async with self.transaction() as session:
                current_active = await collection.find_one({'is_active': True}, session=session)
                if current_active:
                    new_version = current_active.get('version', 0) + 1
                else:
                    new_version = 1
                await collection.update_many({'is_active': True}, {'$set': {'is_active': False, 'deactivated_at': utcnow(), 'deactivated_by': updated_by}}, session=session)
                new_config = {'version': new_version, 'provider_models': config_data.get('provider_models'), 'provider_meta': config_data.get('provider_meta'), 'model_tier': config_data.get('model_tier'), 'platform_configs': config_data.get('platform_configs'), 'platform_fallback_order': config_data.get('platform_fallback_order', []), 'default_registration_config': config_data.get('default_registration_config'), 'is_active': True, 'created_at': utcnow(), 'created_by': updated_by}
                result = await collection.insert_one(new_config, session=session)
                if not result.inserted_id:
                    raise Exception('Insert failed - no inserted_id returned')
                self.logger.info(f'Saved new LLM meta config version {new_version} | updated_by={updated_by} | id={result.inserted_id}')
                return new_version
        except Exception as e:
            self.logger.error(f'Failed to save LLM meta config (transaction rolled back): {e}', exc_info=True)
            return None

    async def get_llm_meta_config_history(self, limit: int=10, skip: int=0) -> Dict[str, Any]:
        try:
            await self.connect()
            collection = self.db.llm_meta_config
            total_count = await collection.count_documents({})
            cursor = collection.find().sort('created_at', -1).skip(skip).limit(limit)
            configs = await cursor.to_list(length=limit)
            for config in configs:
                config['_id'] = str(config['_id'])
            return {'configs': configs, 'total_count': total_count}
        except Exception as e:
            self.logger.error(f'Failed to get LLM meta config history: {e}', exc_info=True)
            return {'configs': [], 'total_count': 0}