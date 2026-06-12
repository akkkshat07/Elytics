from __future__ import annotations
from fastapi import APIRouter, Response, UploadFile, File, HTTPException, Form, Query, Depends, Request, Path, BackgroundTasks
from fastapi.responses import StreamingResponse
import logging
from util.Mongodb import MongoDBManager, normalize_conversation_doc
from services.orchestrator_manager import OrchestratorManager
from domains.conversation.service import ConversationService
from util.cancellation import cancellation_manager
from typing import Optional, Dict, Any, List
from datetime import datetime
from util.time_utils import utcnow
from pydantic import BaseModel, Field, field_validator, EmailStr
import math
from fastapi.encoders import jsonable_encoder
from typing import Literal
from auth.auth import create_access_token, create_refresh_token, decode_refresh_token, verify_password
from config.system_config import SUPER_ADMIN_EMAIL, SUPER_ADMIN_PASSWORD_HASH
from auth.token_blacklist import token_blacklist
from middleware.auth_middleware import require_auth, require_auth_flexible
from middleware.client_middleware import get_client_id_from_request
from util.metrics import record_query
from util.audit_logger import audit_login_success, audit_login_failure, audit_query_execution, audit_access_denied, AuditEventType, AuditSeverity, audit_logger
import time
import json
import re
import asyncio

async def _cancel_background_run(*, conv_service: ConversationService, client_id: str, run_id: str) -> bool:
    cancellation_manager.signal_job(run_id)
    return await conv_service.cancel_background(run_id=run_id, client_id=client_id)

async def _cancel_active_background_runs_for_session(*, db, conv_service: ConversationService, client_id: str, session_id: str, limit: int=200) -> int:
    cursor = db.conversations.find({'session_id': session_id, 'client_id': client_id, 'is_background': True, 'status': {'$in': ['pending', 'running']}, 'is_deleted': {'$ne': True}}, {'run_id': 1, '_id': 0})
    jobs = await cursor.to_list(length=limit)
    attempted = 0
    for job in jobs:
        run_id = (job or {}).get('run_id')
        if not run_id:
            continue
        attempted += 1
        try:
            await _cancel_background_run(conv_service=conv_service, client_id=client_id, run_id=run_id)
        except Exception:
            pass
    return attempted

def validate_id_parameter(param_value: str, param_name: str) -> str:
    if not param_value or len(param_value) > 100:
        raise HTTPException(status_code=400, detail=f'Invalid {param_name}: must be between 1 and 100 characters')
    if not re.match('^[a-zA-Z0-9_-]+$', param_value):
        raise HTTPException(status_code=400, detail=f'Invalid {param_name}: must contain only alphanumeric characters, hyphens, and underscores')
    return param_value

def format_discussion_history(messages: List[Dict[str, Any]], max_chars: int=6000) -> str:
    if not messages:
        return ''
    lines: List[str] = []
    for message in messages[-12:]:
        role = 'User' if message.get('role') == 'user' else 'Assistant'
        content = str(message.get('content') or '').strip()
        if content:
            lines.append(f'{role}: {content}')
    joined = '\n'.join(lines)
    if len(joined) <= max_chars:
        return joined
    return joined[-max_chars:]

class FeedbackSchema(BaseModel):
    run_id: str = Field(..., min_length=1, max_length=100, description='Conversation run ID (UUID format recommended)')
    rating: Literal['positive', 'negative'] = Field(..., description='Feedback rating (positive or negative only)')
    comment: Optional[str] = Field(None, max_length=2000, description='Optional feedback comment (max 2000 chars)')
    user_id: str = Field(..., min_length=1, max_length=100, description='User ID submitting feedback')

    @field_validator('run_id', 'user_id')
    @classmethod
    def validate_id_format(cls, v):
        if not re.match('^[a-zA-Z0-9_-]+$', v):
            raise ValueError(f'ID must contain only alphanumeric characters, hyphens, and underscores')
        return v

    @field_validator('comment')
    @classmethod
    def sanitize_comment(cls, v):
        if v:
            v = re.sub('[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f]', '', v)
            v = v.strip()
        return v if v else None

class StopStreamRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100, description='Session ID to stop streaming')
    run_id: Optional[str] = Field(None, min_length=1, max_length=100, description='Run ID to cancel (preferred when available)')

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if not re.match('^[a-zA-Z0-9_-]+$', v):
            raise ValueError('session_id must contain only alphanumeric characters, hyphens, and underscores')
        return v

    @field_validator('run_id')
    @classmethod
    def validate_run_id(cls, v):
        if not v:
            return None
        if not re.match('^[a-zA-Z0-9_-]+$', v):
            raise ValueError('run_id must contain only alphanumeric characters, hyphens, and underscores')
        return v

class AuthRequest(BaseModel):
    email: EmailStr = Field(..., description='User email address')
    password: str = Field(..., min_length=1, max_length=200, description='Password')

class UpdateSessionTitleRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100, description='Session ID to update')
    title: str = Field(..., min_length=1, max_length=500, description='New session title (max 500 chars)')
    user_id: str = Field(..., min_length=1, max_length=100, description='User ID requesting update')

    @field_validator('session_id', 'user_id')
    @classmethod
    def validate_id_format(cls, v):
        if not re.match('^[a-zA-Z0-9_-]+$', v):
            raise ValueError(f'ID must contain only alphanumeric characters, hyphens, and underscores')
        return v

    @field_validator('title')
    @classmethod
    def sanitize_title(cls, v):
        v = re.sub('[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f]', '', v)
        return v.strip()

class PinConversationRequest(BaseModel):
    conversation_id: str = Field(..., min_length=24, max_length=24, description='MongoDB conversation ObjectId')

    @field_validator('conversation_id')
    @classmethod
    def validate_conversation_id(cls, v):
        if not re.match('^[a-fA-F0-9]{24}$', v):
            raise ValueError('conversation_id must be a valid 24-character ObjectId')
        return v.lower()

class ResponseDiscussionRequest(BaseModel):
    conversation_id: str = Field(..., min_length=24, max_length=24, description='MongoDB conversation ObjectId for the linked assistant response')
    source_run_id: str = Field(..., min_length=1, max_length=100, description='Run ID for the linked assistant response')
    parent_question: Optional[str] = Field(default='', max_length=1000, description='Original user question that produced the assistant response')
    session_id: Optional[str] = Field(default='', max_length=100, description='Optional session ID for the linked assistant response')
    question: str = Field(..., min_length=1, max_length=1000, description='User question about the selected response')
    response_context: str = Field(..., min_length=1, max_length=20000, description='Text context extracted from the selected assistant response')

    @field_validator('conversation_id')
    @classmethod
    def validate_conversation_id(cls, v):
        if not re.match('^[a-fA-F0-9]{24}$', v or ''):
            raise ValueError('conversation_id must be a valid 24-character ObjectId')
        return v.lower()

    @field_validator('source_run_id', 'session_id')
    @classmethod
    def validate_link_ids(cls, v):
        v = (v or '').strip()
        if not v:
            return ''
        if not re.match('^[a-zA-Z0-9_-]+$', v):
            raise ValueError('IDs must contain only alphanumeric characters, hyphens, and underscores')
        return v

    @field_validator('question', 'response_context', 'parent_question')
    @classmethod
    def sanitize_text(cls, v):
        v = re.sub('[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f]', '', v or '')
        return v.strip()

class AgentsRouter:

    def __init__(self):
        self.router = APIRouter()
        self._add_agents_routes()
        self.orchestrator_manager = OrchestratorManager()
        self.logger = logging.getLogger(__name__)
        self.mongo_manager = MongoDBManager()

    def _add_agents_routes(self):

        @self.router.get('/agent_query_stream')
        async def agent_query_stream(request: Request, input: Optional[str]=Query(None), session_id: Optional[str]=Query(None), dataset_id: Optional[str]=Query(None, description='Dataset to run the query against'), run_in_background: bool=Query(False, description='Force query to run as background job'), persona_slug: Optional[str]=Query(None, description='Agent persona slug to activate for this session'), current_user: dict=Depends(require_auth_flexible())):
            start_time = time.time()
            error_occurred = False
            try:
                if not current_user:
                    raise HTTPException(status_code=401, detail='Authentication required. Please log in.')
                client_id = current_user.get('client_id')
                if not client_id:
                    self.logger.error(f"SECURITY: Token missing client_id for user {current_user.get('user_id', 'unknown')}")
                    raise HTTPException(status_code=403, detail='Invalid authentication token: missing client identifier. Please log in again.')
                self.logger.info(f"[MULTI-TENANT] Query from client='{client_id}', user='{current_user.get('user_id')}'")
                user_role = current_user.get('role', 'user')
                if user_role != 'super_admin':
                    from services.subscription_service import check_conversation_limit
                    from db_config.mongo_server import get_db
                    db = await get_db()
                    is_allowed, current_count, limit = await check_conversation_limit(client_id, db)
                    if not is_allowed:
                        error_msg = f'Conversation limit reached. You have used {current_count}/{limit} conversations. Please upgrade your plan.'
                        self.logger.warning(f'Conversation limit exceeded | client_id={client_id} | used={current_count} | limit={limit}')

                        async def error_stream():
                            error_data = {'type': 'error', 'message': error_msg, 'status': 'limit_exceeded', 'current_count': current_count, 'limit': limit}
                            yield f'event: limit_error\n'
                            yield f'data: {json.dumps(error_data)}\n\n'
                            yield f'event: end\n'
                            yield f"data: {json.dumps({'status': 'ended'})}\n\n"
                        headers = {'Cache-Control': 'no-cache', 'Connection': 'keep-alive'}
                        return StreamingResponse(error_stream(), media_type='text/event-stream', headers=headers)
                    if limit is not None and limit > 0:
                        after_this_query = current_count + 1
                        just_crossed_90 = after_this_query / limit >= 0.9 and current_count / limit < 0.9
                        if just_crossed_90:
                            try:
                                from notifications.notification_service import create_notification
                                from notifications.notification_model import Notification as _Notif
                                _uid = str(current_user.get('user_id') or current_user.get('_id') or '')
                                if _uid and client_id:
                                    remaining = limit - after_this_query
                                    await create_notification(_Notif(client_id=client_id, user_id=_uid, type='conversation_limit_warning', title='Conversation Quota Warning', message=f'You have used {after_this_query}/{limit} conversations this month ({remaining} remaining). Consider upgrading your plan.', metadata={'current_count': after_this_query, 'limit': limit}, target_role='any'))
                            except Exception as _ne:
                                self.logger.warning(f'Failed to send conversation_limit_warning notification: {_ne}')
                authenticated_user_id = current_user.get('user_id') or current_user.get('_id') or 'anonymous'
                data = {'input': input, 'files': None, 'user_id': authenticated_user_id, 'session_id': session_id or 'default', 'client_id': client_id, 'dataset_id': (dataset_id or '').strip() or None, 'attempts': 0, 'history': [], 'run_in_background': run_in_background, 'persona_slug': persona_slug}
                client_ip = request.client.host if request.client else 'unknown'
                asyncio.create_task(audit_query_execution(user_id=authenticated_user_id, client_id=client_id, query=input or '', execution_time=0.0, ip_address=client_ip))
                generator = self.orchestrator_manager.stream(data)
                headers = {'Cache-Control': 'no-cache', 'Connection': 'keep-alive'}
                return StreamingResponse(generator, media_type='text/event-stream', headers=headers)
            except Exception as e:
                error_occurred = True
                self.logger.error(f'Error starting agent query stream: {str(e)}')
                raise HTTPException(status_code=500, detail='Failed to start agent query stream')
            finally:
                duration_ms = (time.time() - start_time) * 1000
                record_query(client_id=client_id, duration_ms=duration_ms, error=error_occurred)

        @self.router.get('/personas/available')
        async def get_available_personas(current_user: dict=Depends(require_auth())):
            client_id = current_user.get('client_id')
            if not client_id:
                raise HTTPException(status_code=403, detail='client_id missing from token')
            try:
                from db_config.mongo_server import get_db as get_mongo_db
                from services.persona_service import get_client_personas
                from services.subscription_service import get_client_subscription
                db = await get_mongo_db()
                personas = await get_client_personas(client_id, db)
                subscription = await get_client_subscription(client_id, db)
                plan_name = subscription.get('plan_name', 'freemium')
                PLAN_ORDER = ['freemium', 'starter', 'pro', 'premium']
                plan_rank = PLAN_ORDER.index(plan_name) if plan_name in PLAN_ORDER else 0

                def tier_allowed(persona: dict) -> bool:
                    tiers = persona.get('subscription_tiers', [])
                    if not tiers:
                        return True
                    return any((PLAN_ORDER.index(t) <= plan_rank for t in tiers if t in PLAN_ORDER))
                return {'personas': [{'slug': p['slug'], 'display_name': p['display_name'], 'description': p.get('description', '')} for p in personas if tier_allowed(p)]}
            except Exception as e:
                self.logger.error('Error fetching available personas: %s', e, exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to fetch available personas')

        @self.router.post('/business_insights_only')
        async def business_insights_only(request: Request, current_user: dict=Depends(require_auth())):
            try:
                body = await request.json()
                if not body.get('input'):
                    raise HTTPException(status_code=422, detail='Missing required field: input')
                if not body.get('executor_response'):
                    raise HTTPException(status_code=422, detail='Missing required field: executor_response')
                client_id = current_user.get('client_id')
                if not client_id:
                    raise HTTPException(status_code=400, detail='client_id is required for multi-tenant operation. Please re-authenticate.')
                data = {'input': body.get('input'), 'executor_response': body.get('executor_response', {}), 'summarized_response': body.get('summarized_response'), 'user_id': current_user.get('user_id') or current_user.get('_id') or body.get('user_id', 'anonymous'), 'session_id': body.get('session_id', 'default'), 'client_id': client_id}
                generator = self.orchestrator_manager.stream_business_only(data)
                headers = {'Cache-Control': 'no-cache', 'Connection': 'keep-alive'}
                return StreamingResponse(generator, media_type='text/event-stream', headers=headers)
            except HTTPException:
                raise
            except json.JSONDecodeError as e:
                self.logger.error(f'Invalid JSON in request body: {str(e)}')
                raise HTTPException(status_code=422, detail=f'Invalid JSON in request body: {str(e)}')
            except Exception as e:
                self.logger.error(f'Error starting business insights stream: {str(e)}', exc_info=True)
                raise HTTPException(status_code=500, detail=f'Failed to start business insights stream: {str(e)}')

        @self.router.post('/discuss-response', tags=['Agents'])
        async def discuss_response(payload: ResponseDiscussionRequest, current_user: dict=Depends(require_auth())):
            try:
                from bson import ObjectId
                client_id = current_user.get('client_id')
                user_id = current_user.get('user_id')
                if not client_id:
                    raise HTTPException(status_code=400, detail='Missing client context. Please log in again.')
                if not user_id:
                    raise HTTPException(status_code=401, detail='Missing user context. Please log in again.')
                from db_config.mongo_server import get_db
                from util.llm_utils import LLMClient
                db = await get_db()
                source_conversation = await db.conversations.find_one({'_id': ObjectId(payload.conversation_id), 'run_id': payload.source_run_id, 'client_id': client_id, 'is_deleted': {'$ne': True}})
                if not source_conversation:
                    raise HTTPException(status_code=404, detail='The linked response could not be found for this user.')
                if str(source_conversation.get('user_id') or '').strip() != str(user_id).strip():
                    raise HTTPException(status_code=403, detail='Cannot discuss responses belonging to another user.')
                existing_thread = await self.mongo_manager.get_response_discussion(conversation_id=payload.conversation_id, source_run_id=payload.source_run_id, user_id=user_id, client_id=client_id)
                llm = LLMClient(agent_name='business_agent', client_id=client_id, db=db)
                await llm._load_client_llm_config()
                response_context = payload.response_context[:12000]
                discussion_history = format_discussion_history((existing_thread or {}).get('messages', []), max_chars=6000)
                system_prompt = 'You are answering follow-up questions about a single previously generated assistant response. Use only the provided response context. You may also use the prior discuss-thread history, but only when it is consistent with the same selected response. The response context may include code generation details, execution results, tables, charts, metrics, insights, recommendations, etc. Explain the technical logic when that context is present. Do not introduce new analysis, new data claims, or unrelated chat context. You may synthesize across the provided sections, but you must stay grounded in them. If the answer is not stated or cannot be reasonably inferred from the provided response context, say that the original response did not specify it. If the question is unrelated to the provided response, say you can only discuss that response. Keep the answer concise and direct.'
                user_message = f'RESPONSE CONTEXT:\n{response_context}\n\n' + (f'DISCUSS THREAD HISTORY:\n{discussion_history}\n\n' if discussion_history else '') + f'FOLLOW-UP QUESTION:\n{payload.question}'
                response = await llm.generate_completion(system_prompt=system_prompt, user_message=user_message, temperature=0.1, max_tokens=1200)
                answer = (response.get('content') or '').strip()
                if not answer:
                    answer = 'The original response did not provide enough detail to answer that.'
                llm_config = {'provider': (llm.client_config or {}).get('provider'), 'model': (llm.client_config or {}).get('model')}
                thread = await self.mongo_manager.save_response_discussion_turn(client_id=client_id, user_id=user_id, conversation_id=payload.conversation_id, source_run_id=payload.source_run_id, session_id=payload.session_id or source_conversation.get('session_id'), parent_question=payload.parent_question or '', response_context=response_context, source_conversation=source_conversation, user_question=payload.question, assistant_answer=answer, assistant_usage=response.get('usage'), llm_config=llm_config)
                return {'answer': answer, 'usage': response.get('usage'), 'thread': thread}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error discussing response: {e}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to discuss response')

        @self.router.get('/discuss-response/{conversation_id}/{source_run_id}', tags=['Agents'])
        async def get_discuss_response_thread(conversation_id: str=Path(..., min_length=24, max_length=24, regex='^[a-fA-F0-9]{24}$'), source_run_id: str=Path(..., min_length=1, max_length=100, regex='^[a-zA-Z0-9_-]+$'), current_user: dict=Depends(require_auth())):
            try:
                from bson import ObjectId
                normalized_conversation_id = conversation_id.strip().lower()
                source_run_id = validate_id_parameter(source_run_id, 'source_run_id')
                user_id = current_user.get('user_id')
                client_id = current_user.get('client_id')
                if not user_id or not client_id:
                    raise HTTPException(status_code=401, detail='User authentication context is incomplete')
                await self.mongo_manager.connect()
                db = self.mongo_manager.db
                if db is None:
                    raise HTTPException(status_code=500, detail='Database not available')
                source_conversation = await db.conversations.find_one({'_id': ObjectId(normalized_conversation_id), 'run_id': source_run_id, 'client_id': client_id, 'is_deleted': {'$ne': True}})
                if not source_conversation:
                    raise HTTPException(status_code=404, detail='Linked response not found for this user')
                if str(source_conversation.get('user_id') or '').strip() != str(user_id).strip():
                    raise HTTPException(status_code=403, detail="Cannot access another user's linked response")
                thread = await self.mongo_manager.get_response_discussion(conversation_id=normalized_conversation_id, source_run_id=source_run_id, user_id=user_id, client_id=client_id)
                if not thread:
                    return {'found': False, 'conversation_id': normalized_conversation_id, 'source_run_id': source_run_id, 'messages': [], 'total_token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}
                return {'found': True, 'thread': thread}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error loading response discussion thread: {str(e)}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to load discussion thread')

        @self.router.post('/feedback', tags=['Feedback'])
        async def submit_feedback(feedback: FeedbackSchema, current_user: dict=Depends(require_auth())):
            try:
                user_id = current_user['user_id']
                client_id = current_user.get('client_id')
                conversation = await self.mongo_manager.get_conversation_by_run_id(feedback.run_id)
                if not conversation:
                    raise HTTPException(status_code=404, detail='Conversation not found')
                if conversation['user_id'] != user_id:
                    raise HTTPException(status_code=403, detail="Cannot submit feedback for other users' conversations")
                if client_id and conversation.get('client_id') != client_id:
                    raise HTTPException(status_code=403, detail='Cannot submit feedback for conversations from other clients')
                await self.mongo_manager.connect()
                try:
                    async with self.mongo_manager.transaction() as session:
                        update_result = await self.mongo_manager.save_feedback_async(run_id=feedback.run_id, rating=feedback.rating, comment=feedback.comment or '', user_id=user_id, session=session)
                        if not update_result:
                            raise HTTPException(status_code=404, detail=f"Conversation with run_id '{feedback.run_id}' not found.")
                        if feedback.rating.lower() == 'positive':
                            try:
                                from response_caching.feedback_processor import process_feedback_from_run_id
                                success = await process_feedback_from_run_id(feedback.run_id, self.mongo_manager)
                                if success:
                                    self.logger.info(f'Successfully added question to vector database for run_id: {feedback.run_id}')
                                else:
                                    self.logger.warning(f'Vector DB indexing failed for run_id {feedback.run_id} (Qdrant may be unavailable). Feedback record will still be committed.')
                            except Exception as vec_error:
                                self.logger.warning(f'Vector DB processing raised an exception for run_id {feedback.run_id}: {vec_error}. Feedback record will still be committed.')
                except Exception as tx_error:
                    self.logger.error(f'Transaction failed for feedback {feedback.run_id}: {tx_error}')
                    if isinstance(tx_error, HTTPException):
                        raise tx_error
                    raise HTTPException(status_code=500, detail=f'An error occurred while saving feedback: {str(tx_error)}')
                if feedback.rating.lower() == 'positive':
                    try:
                        from notifications.notification_service import create_notification
                        from notifications.notification_model import Notification
                        notif_user_id = str(current_user.get('user_id') or current_user.get('_id') or '')
                        if notif_user_id and client_id:
                            short_q = (conversation.get('input', '') or '')[:80]
                            await create_notification(Notification(client_id=client_id, user_id=notif_user_id, type='query_cached', title='Query Saved as Cache', message=f'Your query "{short_q}" has been saved to the cache for faster future responses.', metadata={'run_id': feedback.run_id}, target_role='any'))
                    except Exception as _ne:
                        self.logger.warning(f'Failed to send query_cached notification: {_ne}')
                return {'status': 'success', 'message': 'Feedback recorded successfully.'}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f'An error occurred: {e}')

        @self.router.get('/datasets', tags=['Datasets'])
        async def list_enabled_datasets(current_user: dict=Depends(require_auth())):
            from db_config.mongo_server import get_db
            from services.db_credentials_service import DBCredentialsService
            client_id = current_user.get('client_id')
            if not client_id:
                raise HTTPException(status_code=403, detail='client_id missing from token')
            db = await get_db()
            svc = DBCredentialsService(db)
            rows = await svc.get_active_datasets(client_id)
            return {'success': True, 'datasets': rows}

        @self.router.get('/suggested-questions', tags=['Suggest'])
        async def get_suggested_questions(dataset_id: Optional[str]=Query(None, description="Load questions for this dataset's XML path"), current_user: dict=Depends(require_auth())):
            try:
                from pathlib import Path
                from util.dataset_paths import resolve_xml_data_sources_dir
                from defusedxml.ElementTree import parse
                client_id = current_user.get('client_id')
                if not client_id:
                    raise HTTPException(status_code=403, detail='client_id missing from token')
                ds_root = resolve_xml_data_sources_dir(client_id, dataset_id)
                questions_path = ds_root / 'suggested_questions.xml'
                if not questions_path.exists() and dataset_id:
                    questions_path = resolve_xml_data_sources_dir(client_id, None) / 'suggested_questions.xml'
                if not questions_path.exists():
                    self.logger.info(f'suggested_questions.xml not found for {client_id} at {questions_path}, returning empty list')
                    return {'questions': [], 'source': 'none', 'client_id': client_id}
                try:
                    from xml.etree.ElementTree import ParseError
                    tree = parse(questions_path)
                    root = tree.getroot()
                    questions = []
                    for question_elem in root.findall('.//question'):
                        question_text = question_elem.text or ''
                        question_id = question_elem.get('id', '')
                        category = question_elem.get('category', 'general')
                        if question_text:
                            questions.append({'id': question_id, 'text': question_text, 'category': category})
                    self.logger.info(f'Loaded {len(questions)} suggested questions for client {client_id}')
                    return {'questions': questions, 'source': 'client_specific', 'client_id': client_id}
                except ParseError as e:
                    self.logger.error(f'Error parsing suggested_questions.xml for {client_id}: {e}')
                    return {'questions': [], 'source': 'none', 'error': 'Failed to parse XML'}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error loading suggested questions: {e}', exc_info=True)
                return {'questions': [], 'source': 'none', 'error': str(e)}

        @self.router.get('/suggest_questions', tags=['Suggest'])
        async def suggest_questions(q: str=Query(default='', min_length=0), limit: int=Query(default=5, ge=1, le=20), dataset_id: Optional[str]=Query(None, description='Dataset to scope suggestions to'), current_user: dict=Depends(require_auth())):
            client_id = current_user.get('client_id')
            if not client_id:
                raise HTTPException(status_code=403, detail='client_id missing from token')
            if not q or not q.strip():
                return {'suggestions': []}
            try:
                from response_caching.suggester import suggest
                results = suggest(q.strip(), limit, client_id, dataset_id=dataset_id)
                return {'suggestions': results}
            except Exception as e:
                self.logger.error(f"suggest_questions error for client '{client_id}': {e}", exc_info=True)
                return {'suggestions': []}

        @self.router.get('/history', tags=['History'])
        async def get_conversation_history(page: int=1, limit: int=10, sort_order: Literal['desc', 'asc']='desc', feedback_filter: Optional[Literal['all', 'positive', 'negative', 'none']]='all', current_user: dict=Depends(require_auth())):
            if page < 1 or limit < 1:
                raise HTTPException(status_code=400, detail='Page and limit must be positive integers.')
            try:
                await self.mongo_manager.connect()
                db = self.mongo_manager.db
                if db is None:
                    raise HTTPException(status_code=500, detail='Database not available')
                collection = db.conversations
                user_id = current_user['user_id']
                client_id = current_user.get('client_id')
                if not client_id:
                    raise HTTPException(status_code=403, detail='client_id missing from token')
                base_filter = {'user_id': user_id, 'client_id': client_id, 'is_deleted': {'$ne': True}, 'status': {'$ne': 'cancelled'}}
                if feedback_filter == 'positive':
                    base_filter['feedback.rating'] = 'positive'
                elif feedback_filter == 'negative':
                    base_filter['feedback.rating'] = 'negative'
                elif feedback_filter == 'none':
                    base_filter['$or'] = [{'feedback': {'$exists': False}}, {'feedback': None}]
                total_records = await collection.count_documents(base_filter)
                if total_records == 0:
                    return {'data': [], 'currentPage': 1, 'totalPages': 0, 'totalRecords': 0, 'feedbackCounts': await get_feedback_counts(collection, user_id), 'appliedFilter': feedback_filter}
                skip = (page - 1) * limit
                total_pages = math.ceil(total_records / limit)
                sort_direction = -1 if sort_order == 'desc' else 1
                cursor = collection.find(base_filter).sort('created_at', sort_direction).skip(skip).limit(limit)
                history_data = await cursor.to_list(length=limit)
                for item in history_data:
                    item['id'] = str(item['_id'])
                    del item['_id']
                    normalize_conversation_doc(item)
                feedback_counts = await get_feedback_counts(collection, user_id)
                return jsonable_encoder({'data': history_data, 'currentPage': page, 'totalPages': total_pages, 'totalRecords': total_records, 'feedbackCounts': feedback_counts, 'appliedFilter': feedback_filter})
            except Exception as e:
                raise HTTPException(status_code=500, detail=f'An error occurred while fetching history: {e}')

        async def get_feedback_counts(collection, user_id: Optional[str]=None):
            try:
                user_match = {'user_id': user_id, 'is_deleted': {'$ne': True}} if user_id else {'is_deleted': {'$ne': True}}
                pipeline = [{'$match': user_match} if user_match else {'$match': {}}, {'$facet': {'all': [{'$count': 'count'}], 'positive': [{'$match': {'feedback.rating': 'positive'}}, {'$count': 'count'}], 'negative': [{'$match': {'feedback.rating': 'negative'}}, {'$count': 'count'}], 'none': [{'$match': {'$or': [{'feedback': {'$exists': False}}, {'feedback': None}]}}, {'$count': 'count'}]}}]
                result = await collection.aggregate(pipeline).to_list(length=1)
                if result:
                    counts = result[0]
                    return {'all': counts['all'][0]['count'] if counts['all'] else 0, 'positive': counts['positive'][0]['count'] if counts['positive'] else 0, 'negative': counts['negative'][0]['count'] if counts['negative'] else 0, 'none': counts['none'][0]['count'] if counts['none'] else 0}
                else:
                    return {'all': 0, 'positive': 0, 'negative': 0, 'none': 0}
            except Exception as e:
                return {'all': 0, 'positive': 0, 'negative': 0, 'none': 0}

        @self.router.post('/agent_stop_stream')
        async def agent_stop_stream(request: StopStreamRequest, current_user: dict=Depends(require_auth())):
            session_id = request.session_id
            run_id = request.run_id
            if not session_id:
                raise HTTPException(status_code=400, detail='session_id is required')
            try:
                stopped = await self.orchestrator_manager.stop_stream(session_id)
                try:
                    client_id = current_user.get('client_id')
                    if client_id:
                        await self.mongo_manager.connect()
                        db = self.mongo_manager.db
                        if db is not None:
                            collection = db.conversations
                            now = utcnow()
                            if run_id:
                                await collection.update_one({'run_id': run_id, 'client_id': client_id, 'status': {'$in': ['pending', 'running']}, 'is_deleted': {'$ne': True}}, {'$set': {'status': 'cancelled', 'updated_at': now, 'timing.end_time': now, 'metadata.cancel_reason': 'user_stop'}})
                            else:
                                from pymongo import ReturnDocument
                                await collection.find_one_and_update({'session_id': session_id, 'client_id': client_id, 'status': {'$in': ['pending', 'running']}, 'is_deleted': {'$ne': True}}, {'$set': {'status': 'cancelled', 'updated_at': now, 'timing.end_time': now, 'metadata.cancel_reason': 'user_stop'}}, sort=[('created_at', -1)], return_document=ReturnDocument.AFTER)
                except Exception as cancel_err:
                    self.logger.warning(f'Failed to persist cancelled status for session {session_id} (run_id={run_id}): {cancel_err}')
                if stopped:
                    return {'message': f'Stop signal sent for session {session_id}.'}
                return {'message': f'No active stream found for session {session_id}.'}
            except Exception as e:
                self.logger.error(f'Error stopping stream for session {session_id}: {str(e)}')
                raise HTTPException(status_code=500, detail='Failed to stop agent stream')

        @self.router.post('/auth')
        async def authenticate_user(request: AuthRequest, http_request: Request, background_tasks: BackgroundTasks):
            try:
                from db_config.mongo_server import get_db
                from auth.auth import verify_password
                client_ip = http_request.client.host if http_request.client else 'unknown'
                user_agent = http_request.headers.get('user-agent', 'unknown')
                email_lower = request.email.lower().strip()
                self.logger.info(f'Authentication attempt | email={email_lower} | ip={client_ip}')
                if email_lower == SUPER_ADMIN_EMAIL.lower() and verify_password(request.password, SUPER_ADMIN_PASSWORD_HASH):
                    token_data = {'user_id': 'super_admin', '_id': 'super_admin', 'email': SUPER_ADMIN_EMAIL, 'username': 'super_admin', 'client_id': 'super_admin', 'role': 'super_admin'}
                    from datetime import timedelta
                    token = create_access_token(data=token_data, expires_delta=timedelta(minutes=60))
                    refresh_token = create_refresh_token(data=token_data)
                    asyncio.create_task(audit_login_success(user_id='super_admin', client_id='super_admin', ip_address=client_ip, user_agent=user_agent))
                    self.logger.info(f'Super admin login successful | email={SUPER_ADMIN_EMAIL}')
                    return {'is_valid': 1, 'user_id': 'super_admin', 'token': token, 'refresh_token': refresh_token, 'user_data': {'user_id': 'super_admin', '_id': 'super_admin', 'email': SUPER_ADMIN_EMAIL, 'username': 'super_admin', 'full_name': 'Super Admin', 'client_id': 'super_admin', 'role': 'super_admin'}}
                import time
                perf_start = time.time()
                db = await get_db()
                self.logger.debug(f'[PERF] DB connection: {(time.time() - perf_start) * 1000:.2f}ms')
                perf_start = time.time()
                user = await db.users.find_one({'email': email_lower})
                self.logger.debug(f'[PERF] User query: {(time.time() - perf_start) * 1000:.2f}ms')
                if not user:
                    self.logger.warning(f'Authentication failed - user not found | email={email_lower}')
                    asyncio.create_task(audit_login_failure(email=email_lower, ip_address=client_ip, reason='User not found'))
                    return {'is_valid': 0, 'message': 'Invalid email or password'}
                perf_start = time.time()
                password_valid = verify_password(request.password, user.get('hashed_password', ''))
                self.logger.debug(f'[PERF] Password verify: {(time.time() - perf_start) * 1000:.2f}ms')
                if not password_valid:
                    self.logger.warning(f'Authentication failed - invalid password | email={email_lower}')
                    asyncio.create_task(audit_login_failure(email=email_lower, ip_address=client_ip, reason='Invalid password'))
                    return {'is_valid': 0, 'message': 'Invalid email or password'}
                if not user.get('is_active', True):
                    self.logger.warning(f'Authentication failed - account disabled | email={email_lower}')
                    asyncio.create_task(audit_login_failure(email=email_lower, ip_address=client_ip, reason='Account disabled'))
                    return {'is_valid': 0, 'message': 'Account is disabled. Please contact your administrator.'}
                if not user.get('is_email_verified', True):
                    self.logger.warning(f'Authentication failed - email not verified | email={email_lower}')
                    asyncio.create_task(audit_login_failure(email=email_lower, ip_address=client_ip, reason='Email not verified'))
                    return {'is_valid': 0, 'message': 'Please verify your email address before signing in. Check your inbox for the verification link.', 'email_not_verified': True, 'email': email_lower}
                user_id = str(user.get('_id'))
                email = user.get('email')
                username = user.get('username', '')
                full_name = user.get('full_name', email.split('@')[0] if email else 'User')
                client_id = user.get('client_id')
                role = user.get('role', 'user')
                if not client_id:
                    self.logger.error(f'Authentication failed - no client_id | email={email_lower}')
                    asyncio.create_task(audit_login_failure(email=email_lower, ip_address=client_ip, reason='User has no client_id (data integrity issue)'))
                    return {'is_valid': 0, 'message': 'Account configuration error. Please contact support.'}
                from services.tenant_service import get_tenant_status
                tenant_status = await get_tenant_status(client_id, db)
                if tenant_status in ['suspended', 'deleted']:
                    status_message = {'suspended': 'Your account has been suspended. Please contact support.', 'deleted': 'Your account has been deleted. Please contact support.'}.get(tenant_status, 'Your account is not active. Please contact support.')
                    self.logger.warning(f'Authentication blocked - tenant {tenant_status} | email={email_lower} | client_id={client_id}')
                    asyncio.create_task(audit_login_failure(email=email_lower, ip_address=client_ip, reason=f'Tenant status: {tenant_status}'))
                    return {'is_valid': 0, 'message': status_message}
                token_data = {'user_id': user_id, '_id': user_id, 'email': email, 'username': username, 'client_id': client_id, 'role': role}
                perf_start = time.time()
                token = create_access_token(data=token_data)
                refresh_token = create_refresh_token(data=token_data)
                self.logger.debug(f'[PERF] Token creation: {(time.time() - perf_start) * 1000:.2f}ms')
                asyncio.create_task(audit_login_success(user_id=user_id, client_id=client_id, ip_address=client_ip, user_agent=user_agent))
                self.logger.info(f'✅ Authentication successful | email={email} | client_id={client_id} | role={role}')
                user_config = user.get('config', {})
                return {'is_valid': 1, 'user_id': user_id, 'token': token, 'refresh_token': refresh_token, 'user_data': {'user_id': user_id, '_id': user_id, 'email': email, 'username': username, 'full_name': full_name, 'client_id': client_id, 'role': role, 'config': user_config}}
            except Exception as e:
                self.logger.error(f'Authentication error: {str(e)}', exc_info=True)
                asyncio.create_task(audit_login_failure(username=request.username, ip_address=client_ip, reason=f'Exception: {str(e)}'))
                raise HTTPException(status_code=500, detail='Authentication failed due to server error. Please try again.')

        @self.router.post('/auth/refresh', tags=['Authentication'])
        async def refresh_token(request: Request):
            try:
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    raise HTTPException(status_code=401, detail='Missing or invalid refresh token')
                refresh_token = auth_header.split(' ')[1]
                if await token_blacklist.is_refresh_token_blacklisted(refresh_token):
                    raise HTTPException(status_code=401, detail='Refresh token has been revoked')
                payload = decode_refresh_token(refresh_token)
                if not payload:
                    raise HTTPException(status_code=401, detail='Invalid refresh token')
                token_data = {'_id': payload.get('_id') or payload.get('user_id'), 'user_id': payload.get('_id') or payload.get('user_id'), 'email': payload.get('email'), 'username': payload.get('username'), 'client_id': payload.get('client_id'), 'role': payload.get('role')}
                new_access_token = create_access_token(data=token_data)
                new_refresh_token = create_refresh_token(data=token_data)
                from datetime import timedelta
                expires_at = utcnow() + timedelta(days=30)
                await token_blacklist.blacklist_refresh_token(refresh_token, expires_at)
                return {'token': new_access_token, 'refresh_token': new_refresh_token}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Token refresh error: {str(e)}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to refresh token')

        @self.router.post('/auth/logout', tags=['Authentication'])
        async def logout(request: Request, current_user: dict=Depends(require_auth())):
            try:
                auth_header = request.headers.get('Authorization')
                if auth_header and auth_header.startswith('Bearer '):
                    token = auth_header.split(' ')[1]
                    from datetime import timedelta
                    expires_at = utcnow() + timedelta(hours=8)
                    await token_blacklist.blacklist_token(token, expires_at)
                try:
                    body = await request.json()
                    refresh_token = body.get('refresh_token')
                    if refresh_token:
                        expires_at = utcnow() + timedelta(days=30)
                        await token_blacklist.blacklist_refresh_token(refresh_token, expires_at)
                except:
                    pass
                self.logger.info(f"User logged out | user_id={current_user.get('user_id', 'unknown')}")
                return {'message': 'Logged out successfully'}
            except Exception as e:
                self.logger.error(f'Logout error: {str(e)}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to logout')

        @self.router.put('/users/me/config', tags=['User Settings'])
        async def update_own_config(request: Request, config_data: Dict[str, Any], current_user: dict=Depends(require_auth())):
            try:
                from db_config.mongo_server import get_db
                user_email = current_user.get('email')
                if not user_email:
                    raise HTTPException(status_code=401, detail='User not authenticated')
                db = await get_db()
                user = await db.users.find_one({'email': user_email.lower()})
                if not user:
                    raise HTTPException(status_code=404, detail='User not found')
                if 'theme' in config_data and config_data['theme'] not in ['light', 'dark']:
                    raise HTTPException(status_code=400, detail="theme must be 'light' or 'dark'")
                if 'show_development_steps' in config_data and (not isinstance(config_data['show_development_steps'], bool)):
                    raise HTTPException(status_code=400, detail='show_development_steps must be a boolean')
                if 'business_insights_sections' in config_data:
                    valid_keys = {'summary', 'metrics', 'insights', 'recommendations', 'follow_ups', 'note'}
                    sections = config_data['business_insights_sections']
                    if not isinstance(sections, dict):
                        raise HTTPException(status_code=400, detail='business_insights_sections must be a dictionary')
                    for key in sections:
                        if key not in valid_keys:
                            raise HTTPException(status_code=400, detail=f"Invalid business_insights_sections key: {key}. Valid keys are: {', '.join(sorted(valid_keys))}")
                        if not isinstance(sections[key], bool):
                            raise HTTPException(status_code=400, detail=f'business_insights_sections.{key} must be a boolean')
                if 'pinned_query_ids' in config_data:
                    from bson import ObjectId
                    pinned_ids = config_data['pinned_query_ids']
                    if not isinstance(pinned_ids, list):
                        raise HTTPException(status_code=400, detail='pinned_query_ids must be an array of conversation IDs')
                    deduped_ids: List[str] = []
                    seen = set()
                    for raw_id in pinned_ids:
                        if not isinstance(raw_id, str):
                            raise HTTPException(status_code=400, detail='pinned_query_ids must contain only string values')
                        normalized = raw_id.strip().lower()
                        if not ObjectId.is_valid(normalized):
                            raise HTTPException(status_code=400, detail=f'Invalid conversation ID in pinned_query_ids: {raw_id}')
                        if normalized not in seen:
                            seen.add(normalized)
                            deduped_ids.append(normalized)
                    if len(deduped_ids) > 3:
                        raise HTTPException(status_code=400, detail='Maximum 3 pinned queries are allowed')
                    config_data['pinned_query_ids'] = deduped_ids
                existing_config = user.get('config', {})
                merged_config = {**existing_config, **config_data}
                await db.users.update_one({'email': user_email.lower()}, {'$set': {'config': merged_config, 'updated_at': utcnow()}})
                self.logger.info(f'User {user_email} updated their config')
                return {'success': True, 'message': 'Config updated successfully', 'config': merged_config}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error updating user config: {str(e)}', exc_info=True)
                raise HTTPException(status_code=500, detail=f'Failed to update config: {str(e)}')

        @self.router.get('/users/me/conversations/by-run/{run_id}', tags=['User Settings'])
        async def get_conversation_id_by_run_id(run_id: str=Path(..., min_length=1, max_length=100, regex='^[a-zA-Z0-9_-]+$'), current_user: dict=Depends(require_auth())):
            try:
                run_id = validate_id_parameter(run_id, 'run_id')
                await self.mongo_manager.connect()
                db = self.mongo_manager.db
                if db is None:
                    raise HTTPException(status_code=500, detail='Database not available')
                user_id = current_user.get('user_id')
                client_id = current_user.get('client_id')
                if not user_id or not client_id:
                    raise HTTPException(status_code=401, detail='User authentication context is incomplete')
                conversation = await db.conversations.find_one({'run_id': run_id, 'user_id': user_id, 'client_id': client_id, 'is_deleted': {'$ne': True}})
                if not conversation:
                    return {'found': False, 'run_id': run_id}
                return {'found': True, 'run_id': run_id, 'conversation_id': str(conversation.get('_id')), 'session_id': conversation.get('session_id', ''), 'enhanced_question': conversation.get('enhanced_question', ''), 'input': conversation.get('input', '')}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error resolving conversation by run_id: {str(e)}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to resolve conversation')

        @self.router.get('/users/me/pinned-queries', tags=['User Settings'])
        async def get_pinned_queries(current_user: dict=Depends(require_auth())):
            try:
                from bson import ObjectId
                from db_config.mongo_server import get_db
                user_email = current_user.get('email')
                user_id = current_user.get('user_id')
                client_id = current_user.get('client_id')
                if not user_email or not user_id or (not client_id):
                    raise HTTPException(status_code=401, detail='User not authenticated')
                db = await get_db()
                user = await db.users.find_one({'email': user_email.lower()})
                if not user:
                    raise HTTPException(status_code=404, detail='User not found')
                config = user.get('config', {}) or {}
                pinned_ids_raw = config.get('pinned_query_ids', []) or []
                valid_ids: List[str] = []
                seen = set()
                for conv_id in pinned_ids_raw:
                    if isinstance(conv_id, str):
                        normalized = conv_id.strip().lower()
                        if ObjectId.is_valid(normalized) and normalized not in seen:
                            valid_ids.append(normalized)
                            seen.add(normalized)
                    if len(valid_ids) == 3:
                        break
                if not valid_ids:
                    return {'pinned_questions': [], 'pinned_query_ids': []}
                object_ids = [ObjectId(conv_id) for conv_id in valid_ids]
                conversations = await db.conversations.find({'_id': {'$in': object_ids}, 'user_id': user_id, 'client_id': client_id, 'is_deleted': {'$ne': True}}).to_list(length=10)
                conv_map = {str(c['_id']).lower(): c for c in conversations}
                pinned_questions = []
                cleaned_ids: List[str] = []
                for conv_id in valid_ids:
                    conversation = conv_map.get(conv_id)
                    if not conversation:
                        continue
                    cleaned_ids.append(conv_id)
                    pinned_questions.append({'conversation_id': conv_id, 'run_id': conversation.get('run_id'), 'dataset_id': conversation.get('dataset_id'), 'enhanced_question': conversation.get('enhanced_question', '') or '', 'input': conversation.get('input', '') or '', 'pinned_at': conversation.get('created_at').isoformat() if conversation.get('created_at') else None})
                if cleaned_ids != valid_ids:
                    merged_config = {**config, 'pinned_query_ids': cleaned_ids}
                    await db.users.update_one({'email': user_email.lower()}, {'$set': {'config': merged_config, 'updated_at': utcnow()}})
                return {'pinned_questions': pinned_questions, 'pinned_query_ids': cleaned_ids}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error fetching pinned queries: {str(e)}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to fetch pinned queries')

        @self.router.post('/users/me/pinned-queries', tags=['User Settings'])
        async def pin_query(request_data: PinConversationRequest, background_tasks: BackgroundTasks, current_user: dict=Depends(require_auth())):
            try:
                from bson import ObjectId
                from db_config.mongo_server import get_db
                from response_caching.feedback_processor import process_feedback_from_conversation_id
                user_email = current_user.get('email')
                user_id = current_user.get('user_id')
                client_id = current_user.get('client_id')
                if not user_email or not user_id or (not client_id):
                    raise HTTPException(status_code=401, detail='User not authenticated')
                db = await get_db()
                user = await db.users.find_one({'email': user_email.lower()})
                if not user:
                    raise HTTPException(status_code=404, detail='User not found')
                conversation_id = request_data.conversation_id.strip().lower()
                conversation = await db.conversations.find_one({'_id': ObjectId(conversation_id), 'user_id': user_id, 'client_id': client_id, 'is_deleted': {'$ne': True}})
                if not conversation:
                    raise HTTPException(status_code=404, detail='Conversation not found for this user')
                existing_config = user.get('config', {}) or {}
                pinned_ids = existing_config.get('pinned_query_ids', []) or []
                normalized = []
                seen = set()
                for pid in pinned_ids:
                    if isinstance(pid, str):
                        val = pid.strip().lower()
                        if ObjectId.is_valid(val) and val not in seen:
                            normalized.append(val)
                            seen.add(val)
                    if len(normalized) == 3:
                        break
                if conversation_id in normalized:
                    return {'success': True, 'message': 'Query already pinned', 'pinned_query_ids': normalized}
                if len(normalized) >= 3:
                    raise HTTPException(status_code=400, detail='Maximum 3 pinned queries reached. Unpin one to continue.')
                updated_ids = [conversation_id, *normalized][:3]
                updated_config = {**existing_config, 'pinned_query_ids': updated_ids}
                await db.users.update_one({'email': user_email.lower()}, {'$set': {'config': updated_config, 'updated_at': utcnow()}})
                try:
                    from notifications.notification_service import create_notification
                    from notifications.notification_model import Notification as _Notif
                    notif_uid = str(user_id or '')
                    if notif_uid and client_id:
                        convo_input = (conversation.get('input', '') or '')[:80]
                        await create_notification(_Notif(client_id=client_id, user_id=notif_uid, type='query_pinned', title='Query Pinned', message=f'Your query "{convo_input}" has been pinned for quick access.', metadata={'conversation_id': conversation_id}, target_role='any'))
                except Exception as _ne:
                    self.logger.warning(f'Failed to send query_pinned notification: {_ne}')

                async def _warm_cache():
                    try:
                        await process_feedback_from_conversation_id(conversation_id, self.mongo_manager)
                    except Exception as warm_err:
                        self.logger.warning(f'Pinned cache warmup failed for conversation_id={conversation_id}: {warm_err}')
                background_tasks.add_task(_warm_cache)
                return {'success': True, 'message': 'Query pinned successfully', 'pinned_query_ids': updated_ids}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error pinning query: {str(e)}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to pin query')

        @self.router.delete('/users/me/pinned-queries/{conversation_id}', tags=['User Settings'])
        async def unpin_query(conversation_id: str=Path(..., min_length=24, max_length=24, regex='^[a-fA-F0-9]{24}$'), current_user: dict=Depends(require_auth())):
            try:
                from bson import ObjectId
                from db_config.mongo_server import get_db
                user_email = current_user.get('email')
                if not user_email:
                    raise HTTPException(status_code=401, detail='User not authenticated')
                normalized_id = conversation_id.strip().lower()
                if not ObjectId.is_valid(normalized_id):
                    raise HTTPException(status_code=400, detail='Invalid conversation_id')
                db = await get_db()
                user = await db.users.find_one({'email': user_email.lower()})
                if not user:
                    raise HTTPException(status_code=404, detail='User not found')
                existing_config = user.get('config', {}) or {}
                pinned_ids = existing_config.get('pinned_query_ids', []) or []
                updated_ids = [pid.strip().lower() for pid in pinned_ids if isinstance(pid, str) and pid.strip().lower() != normalized_id and ObjectId.is_valid(pid.strip())][:3]
                updated_config = {**existing_config, 'pinned_query_ids': updated_ids}
                await db.users.update_one({'email': user_email.lower()}, {'$set': {'config': updated_config, 'updated_at': utcnow()}})
                return {'success': True, 'message': 'Query unpinned successfully', 'pinned_query_ids': updated_ids}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error unpinning query: {str(e)}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to unpin query')

        @self.router.get('/check-client-data', tags=['Authentication'])
        async def check_client_data(current_user: dict=Depends(require_auth())):
            try:
                from db_config.mongo_server import get_db
                client_id = current_user.get('client_id')
                role = current_user.get('role', 'user')
                is_admin = role == 'admin'
                db = await get_db()
                db_credential = await db.db_credentials.find_one({'client_id': client_id})
                has_data = db_credential is not None
                config_type = db_credential.get('db_type') if db_credential else None
                self.logger.info(f"Client data check | client_id={client_id} | user={current_user.get('username')} | has_db_credentials={has_data} | db_type={config_type} | is_admin={is_admin}")
                return {'has_data': has_data, 'config_type': config_type, 'is_admin': is_admin, 'client_id': client_id}
            except Exception as e:
                self.logger.error(f'Error checking client data: {str(e)}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to check client data')

        @self.router.get('/sessions', tags=['Sessions'])
        async def get_user_sessions(page: int=Query(1, ge=1), limit: int=Query(20, ge=1, le=100), current_user: dict=Depends(require_auth())):
            try:
                user_role = current_user.get('role', 'user')
                client_id = current_user.get('client_id')
                is_admin = user_role in ['admin', 'super_admin']
                if is_admin:
                    result = await self.mongo_manager.get_user_sessions(user_id=None, page=page, limit=limit, client_id=client_id, is_admin=True, include_user_info=True)
                else:
                    user_id = current_user['user_id']
                    result = await self.mongo_manager.get_user_sessions(user_id=user_id, page=page, limit=limit, client_id=client_id, is_admin=False, include_user_info=False)
                if result is None:
                    raise HTTPException(status_code=500, detail='Failed to fetch sessions')
                return result
            except Exception as e:
                self.logger.error(f'Error fetching user sessions: {str(e)}')
                raise HTTPException(status_code=500, detail=f'Failed to fetch sessions: {str(e)}')

        @self.router.get('/sessions/unread-count', tags=['Sessions'])
        async def get_unread_session_count(current_user: dict=Depends(require_auth())):
            try:
                user_role = current_user.get('role', 'user')
                client_id = current_user.get('client_id')
                is_admin = user_role in ['admin', 'super_admin']
                user_id = None if is_admin else current_user.get('user_id')
                count = await self.mongo_manager.get_unread_session_count(client_id=client_id, user_id=user_id)
                return {'unread_session_count': count}
            except Exception as e:
                self.logger.error(f'Error fetching unread session count: {str(e)}')
                raise HTTPException(status_code=500, detail='Failed to fetch unread count')

        @self.router.get('/sessions/{session_id}', tags=['Sessions'])
        async def get_session_conversations(session_id: str=Path(..., min_length=1, max_length=100, regex='^[a-zA-Z0-9_-]+$'), include_pending: bool=Query(True, description='Include pending/running conversations in the session thread'), current_user: dict=Depends(require_auth())):
            try:
                session_id = validate_id_parameter(session_id, 'session_id')
                user_id = current_user.get('user_id') or current_user.get('_id')
                client_id = current_user.get('client_id')
                user_role = current_user.get('role', 'user')
                is_admin = user_role in ['admin', 'super_admin']
                session = await self.mongo_manager.get_session_metadata(session_id, user_id=user_id if not is_admin else None, client_id=client_id if is_admin else None)
                if not session:
                    raise HTTPException(status_code=404, detail='Session not found')
                session_user_id = session.get('user_id')
                if not session_user_id:
                    raise HTTPException(status_code=404, detail='Session metadata incomplete: missing user_id')
                from bson import ObjectId
                if isinstance(session_user_id, ObjectId):
                    session_user_id = str(session_user_id)
                else:
                    session_user_id = str(session_user_id)
                if isinstance(user_id, ObjectId):
                    user_id = str(user_id)
                else:
                    user_id = str(user_id)
                session_user_id = session_user_id.strip()
                user_id = user_id.strip()
                self.logger.info(f"Session ownership check | session_id={session_id} | session_user_id='{session_user_id}' (type: {type(session_user_id).__name__}) | token_user_id='{user_id}' (type: {type(user_id).__name__}) | match={session_user_id == user_id} | session_user_id_repr={repr(session_user_id)} | token_user_id_repr={repr(user_id)}")
                if not is_admin and session_user_id != user_id:
                    self.logger.error(f"Session access denied | session_id={session_id} | session_user_id='{session_user_id}' | token_user_id='{user_id}' | session_user_id_len={len(session_user_id)} | token_user_id_len={len(user_id)} | session_user_id_bytes={(session_user_id.encode('utf-8') if isinstance(session_user_id, str) else 'N/A')} | token_user_id_bytes={(user_id.encode('utf-8') if isinstance(user_id, str) else 'N/A')}")
                    raise HTTPException(status_code=403, detail=f"Cannot access other users' sessions (session_user_id: {session_user_id}, token_user_id: {user_id})")
                if client_id and session.get('client_id') != client_id:
                    raise HTTPException(status_code=403, detail='Cannot access sessions from other clients')
                conversations_user_id = session_user_id if is_admin and session_user_id != user_id else user_id
                conversations = await self.mongo_manager.get_session_conversations(conversations_user_id, session_id, include_pending=include_pending)
                if conversations is None:
                    raise HTTPException(status_code=500, detail='Failed to fetch conversations')
                return {'session_id': session_id, 'conversations': conversations, 'is_own_session': session_user_id == user_id}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error fetching session conversations: {str(e)}', exc_info=True)
                raise HTTPException(status_code=500, detail=f'Failed to fetch conversations: {str(e)}')

        @self.router.delete('/sessions/{session_id}', tags=['Sessions'])
        async def delete_session(session_id: str=Path(..., min_length=1, max_length=100, regex='^[a-zA-Z0-9_-]+$'), current_user: dict=Depends(require_auth())):
            try:
                session_id = validate_id_parameter(session_id, 'session_id')
                user_id = current_user.get('user_id') or current_user.get('_id')
                client_id = current_user.get('client_id')
                if not client_id:
                    raise HTTPException(status_code=401, detail='Authentication context incomplete')
                session = await self.mongo_manager.get_session_metadata(session_id, user_id=user_id)
                if not session:
                    raise HTTPException(status_code=404, detail='Session not found')
                session_client_id = session.get('client_id')
                if session_client_id and client_id and (session_client_id != client_id):
                    raise HTTPException(status_code=403, detail='Cannot delete sessions from other clients')
                session_user_id = session.get('user_id')
                if not session_user_id:
                    raise HTTPException(status_code=404, detail='Session metadata incomplete: missing user_id')
                from bson import ObjectId
                session_user_id = str(session_user_id).strip()
                user_id = str(user_id).strip()
                if session_user_id != user_id:
                    user_role = current_user.get('role', 'user')
                    self.logger.warning(f"User {current_user.get('email')} (role: {user_role}) attempted to delete session {session_id} owned by user_id {session_user_id} (requesting user_id: {user_id})")
                    raise HTTPException(status_code=403, detail="Cannot delete other users' sessions. Only the session owner can delete their own sessions.")
                try:
                    await self.orchestrator_manager.stop_stream(session_id)
                except Exception as stop_err:
                    self.logger.warning(f'Failed to stop foreground stream for session {session_id}: {stop_err}')
                try:
                    from db_config.mongo_server import get_db
                    db = await get_db()
                    conv_service = ConversationService(db)
                    await _cancel_active_background_runs_for_session(db=db, conv_service=conv_service, client_id=client_id, session_id=session_id, limit=200)
                except Exception as bg_cancel_err:
                    self.logger.warning(f'Failed to cancel background jobs for session {session_id}: {bg_cancel_err}')
                deleted_count = await self.mongo_manager.delete_session_by_session_id_scoped(session_id, client_id)
                if deleted_count == 0:
                    self.logger.info(f'Delete requested for session {session_id}: no matching documents in client scope ({client_id}).')
                else:
                    self.logger.info(f'Successfully deleted {deleted_count} conversations for session {session_id} (client_id={client_id}, user_id={user_id})')
                return {'message': 'Session deleted successfully', 'session_id': session_id}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error deleting session: {str(e)}')
                raise HTTPException(status_code=500, detail=f'Failed to delete session: {str(e)}')

        @self.router.patch('/sessions/{session_id}/title', tags=['Sessions'])
        async def update_session_title(request: UpdateSessionTitleRequest, session_id: str=Path(..., min_length=1, max_length=100, regex='^[a-zA-Z0-9_-]+$'), current_user: dict=Depends(require_auth())):
            try:
                session_id = validate_id_parameter(session_id, 'session_id')
                user_id = current_user.get('user_id') or current_user.get('_id')
                client_id = current_user.get('client_id')
                session = await self.mongo_manager.get_session_metadata(session_id, user_id=user_id)
                if not session:
                    raise HTTPException(status_code=404, detail='Session not found')
                session_user_id = session.get('user_id')
                if not session_user_id:
                    raise HTTPException(status_code=404, detail='Session metadata incomplete: missing user_id')
                from bson import ObjectId
                if isinstance(session_user_id, ObjectId):
                    session_user_id = str(session_user_id)
                else:
                    session_user_id = str(session_user_id)
                if isinstance(user_id, ObjectId):
                    user_id = str(user_id)
                else:
                    user_id = str(user_id)
                session_user_id = session_user_id.strip()
                user_id = user_id.strip()
                if session_user_id != user_id:
                    raise HTTPException(status_code=403, detail="Cannot update title for other users' sessions")
                if client_id and session.get('client_id') != client_id:
                    raise HTTPException(status_code=403, detail='Cannot update title for sessions from other clients')
                success = await self.mongo_manager.update_session_title(user_id, session_id, request.title)
                if not success:
                    raise HTTPException(status_code=500, detail='Failed to update session title')
                return {'message': 'Session title updated successfully'}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error updating session title: {str(e)}')
                raise HTTPException(status_code=500, detail=f'Failed to update title: {str(e)}')

        @self.router.post('/sessions/{session_id}/mark-read', tags=['Sessions'])
        async def mark_session_read(session_id: str=Path(..., min_length=1, max_length=100, regex='^[a-zA-Z0-9_-]+$'), current_user: dict=Depends(require_auth())):
            try:
                client_id = current_user.get('client_id')
                if not client_id:
                    raise HTTPException(status_code=401, detail='Authentication context incomplete')
                session_id = validate_id_parameter(session_id, 'session_id')
                from db_config.mongo_server import get_db
                db = await get_db()
                conv_service = ConversationService(db)
                await conv_service.mark_session_read(session_id=session_id, client_id=client_id)
                return {'ok': True}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error marking session read for {session_id}: {e}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to mark session read')

        @self.router.get('/client-vocab', tags=['Client Configuration'])
        async def get_client_vocab(current_user: dict=Depends(require_auth())):
            try:
                client_id = current_user.get('client_id')
                if not client_id:
                    return {'vocab': []}
                from db_config.database import get_db
                from services.schema_mapper import SchemaMapper
                db = get_db()
                schema_mapper = await SchemaMapper.create(client_id, db)
                guardrails = schema_mapper.get_guardrails_config()
                vocab = []
                facility_names = guardrails.get('facility_names', [])
                for facility in facility_names:
                    vocab.append({'content': facility, 'sounds_like': [facility.lower()]})
                product_terms = guardrails.get('product_terms', [])
                for product in product_terms:
                    vocab.append({'content': product, 'sounds_like': [product.lower()]})
                domain_keywords = guardrails.get('domain_keywords', [])
                for keyword in domain_keywords:
                    if keyword not in [v['content'] for v in vocab]:
                        vocab.append({'content': keyword, 'sounds_like': [keyword.lower()]})
                return {'vocab': vocab}
            except Exception as e:
                self.logger.error(f'Error loading client vocab: {e}')
                return {'vocab': []}

        @self.router.post('/internal/llm-query', tags=['Internal'])
        async def internal_llm_query(request: Request):
            try:
                data = await request.json()
                question = str(data.get('question', '')).strip()
                context = str(data.get('context', ''))
                client_id = str(data.get('client_id', 'default')).strip()
                if not question:
                    return {'answer': '', 'error': 'empty question'}
                await self.mongo_manager.connect()
                db = self.mongo_manager.db
                from util.llm_utils import LLMClient
                llm = LLMClient(agent_name='data_science_agent', client_id=client_id, db=db)
                system = 'You are a precise analytical assistant embedded in a data science pipeline. Answer the question directly based on the context. Be factual and brief — no preamble, no explanation, just the answer.'
                user_msg = f'Context:\n{context}\n\nQuestion: {question}' if context else question
                response = await llm.generate_completion(system_prompt=system, user_message=user_msg, temperature=0.1, max_tokens=600)
                return {'answer': (response.get('content') or '').strip(), 'error': response.get('error')}
            except Exception as e:
                self.logger.error(f'internal_llm_query error: {e}')
                return {'answer': '', 'error': str(e)}

        @self.router.post('/internal/live-sql-execute', tags=['Internal'])
        async def internal_live_sql_execute(request: Request):
            try:
                data = await request.json()
                client_id = str(data.get('client_id') or '').strip()
                dataset_id = str(data.get('dataset_id') or '').strip() or None
                mode = str(data.get('mode') or 'intent').strip().lower()
                user_query = str(data.get('user_query') or '').strip()
                query = str(data.get('query') or '').strip()
                max_retries = int(data.get('max_retries') or 2)
                if not client_id:
                    raise HTTPException(status_code=400, detail='client_id is required')
                if mode not in {'intent', 'query'}:
                    raise HTTPException(status_code=400, detail="mode must be 'intent' or 'query'")
                if mode == 'intent' and (not user_query):
                    raise HTTPException(status_code=400, detail='user_query is required for intent mode')
                if mode == 'query' and (not query):
                    raise HTTPException(status_code=400, detail='query is required for query mode')
                await self.mongo_manager.connect()
                db = self.mongo_manager.db
                from db_config.connection_pool_manager import ConnectionPoolManager
                from services.db_credentials_service import DBCredentialsService
                from agents.live_sql.retry_handler import execute_with_retries
                from util.llm_utils import LLMClient
                from util.data_source import require_store_in_local
                creds = await DBCredentialsService(db).get_credentials(client_id=client_id, db_type=None, decrypt_password=False, dataset_id=dataset_id)
                if not creds:
                    raise HTTPException(status_code=404, detail='No DB credentials found for requested dataset')
                db_type = (creds.get('db_type') or 'postgres').strip().lower()
                store_in_local = require_store_in_local(creds)
                ssh_cfg = (creds.get('additional_params') or {}).get('ssh') or {}
                ssh_enabled = bool(ssh_cfg.get('enabled'))
                schema_name = (creds.get('schema_name') or 'public').strip() or 'public'
                self.logger.info('live_sql_mode_selection client=%s dataset_id=%s mode=%s db_type=%s store_in_local=%s ssh_enabled=%s', client_id, dataset_id, mode, db_type, store_in_local, ssh_enabled)
                if store_in_local:
                    raise HTTPException(status_code=400, detail='Selected dataset is configured for local/parquet mode (store_in_local=true)')
                pool_manager = ConnectionPoolManager()
                connector = await pool_manager.get_connection(client_id=client_id, db=db, dataset_id=dataset_id)
                session_factory = connector.get_db()
                llm = LLMClient(agent_name='data_science_agent', client_id=client_id, db=db)

                async def _llm_complete(prompt: str) -> str:
                    response = await llm.generate_completion(system_prompt='You are a SQL generator. Return ONLY one SELECT SQL query. No markdown, no explanation.', user_message=prompt, temperature=0.0, max_tokens=900)
                    return str(response.get('content') or '').strip()
                result = await execute_with_retries(user_query=user_query if mode == 'intent' else query, session_factory=session_factory, llm_complete=_llm_complete, db_type=db_type, schema_name=schema_name, max_retries=max_retries if mode == 'intent' else 0, initial_query=query if mode == 'query' else None)
                df = result.get('dataframe')
                rows = df.to_dict(orient='records') if df is not None else []
                self.logger.info('live_sql_execution_result client=%s dataset_id=%s mode=%s status=%s row_count=%s query=%s', client_id, dataset_id, mode, result.get('status'), len(rows), result.get('query'))
                return {'status': result.get('status'), 'error': result.get('error'), 'query': result.get('query'), 'rows': rows, 'attempts': int(result.get('attempts') or 0)}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error('internal_live_sql_execute error: %s', e, exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get('/jobs', tags=['Background Jobs'])
        async def list_jobs(status: Optional[str]=Query(None, description='Filter by status: pending, running, completed, error, cancelled'), limit: int=Query(20, ge=1, le=100), offset: int=Query(0, ge=0), current_user: dict=Depends(require_auth())):
            try:
                client_id = current_user.get('client_id')
                user_id = current_user.get('user_id') or current_user.get('_id')
                if not client_id or not user_id:
                    raise HTTPException(status_code=401, detail='Authentication context incomplete')
                from db_config.mongo_server import get_db
                db = await get_db()
                conv_service = ConversationService(db)
                jobs = await conv_service.list_background(client_id=client_id, user_id=user_id, status=status, limit=limit, offset=offset)
                return {'jobs': jobs, 'count': len(jobs)}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error listing background jobs: {e}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to list background jobs')

        @self.router.get('/jobs/active', tags=['Background Jobs'])
        async def get_active_jobs(current_user: dict=Depends(require_auth())):
            try:
                client_id = current_user.get('client_id')
                user_id = current_user.get('user_id') or current_user.get('_id')
                if not client_id or not user_id:
                    raise HTTPException(status_code=401, detail='Authentication context incomplete')
                from services import redis_job_store
                jobs = await redis_job_store.get_active_jobs(client_id, user_id)
                if jobs is None:
                    from db_config.mongo_server import get_db
                    db = await get_db()
                    jobs = await ConversationService(db).get_active_background(client_id=client_id, user_id=user_id)
                return {'jobs': jobs, 'count': len(jobs)}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error fetching active jobs: {e}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to fetch active jobs')

        @self.router.post('/jobs/send-to-background', tags=['Background Jobs'])
        async def send_to_background(request: Request, current_user: dict=Depends(require_auth())):
            try:
                body = await request.json()
                run_id = body.get('run_id')
                if not run_id:
                    raise HTTPException(status_code=400, detail='run_id is required')
                run_id = validate_id_parameter(run_id, 'run_id')
                from services.orchestrator_manager import trigger_background_signal
                success = trigger_background_signal(run_id)
                if not success:
                    raise HTTPException(status_code=404, detail='No active stream found for this run_id')
                return {'status': 'signaled', 'run_id': run_id}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error sending to background: {e}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to send to background')

        @self.router.get('/jobs/{run_id}', tags=['Background Jobs'])
        async def get_job(run_id: str=Path(..., min_length=1, max_length=100), current_user: dict=Depends(require_auth())):
            try:
                client_id = current_user.get('client_id')
                if not client_id:
                    raise HTTPException(status_code=401, detail='Authentication context incomplete')
                run_id = validate_id_parameter(run_id, 'run_id')
                from db_config.mongo_server import get_db
                db = await get_db()
                conv_service = ConversationService(db)
                job = await conv_service.get_background_by_run_id(run_id=run_id, client_id=client_id)
                if not job:
                    raise HTTPException(status_code=404, detail='Background conversation not found')
                return job
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error fetching background conversation {run_id}: {e}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to fetch background conversation')

        @self.router.post('/jobs/{run_id}/cancel', tags=['Background Jobs'])
        async def cancel_job(run_id: str=Path(..., min_length=1, max_length=100), current_user: dict=Depends(require_auth())):
            try:
                client_id = current_user.get('client_id')
                if not client_id:
                    raise HTTPException(status_code=401, detail='Authentication context incomplete')
                run_id = validate_id_parameter(run_id, 'run_id')
                from db_config.mongo_server import get_db
                db = await get_db()
                conv_service = ConversationService(db)
                job = await conv_service.get_background_by_run_id(run_id=run_id, client_id=client_id)
                if not job:
                    raise HTTPException(status_code=404, detail='Background conversation not found')
                if job['status'] not in ('pending', 'running'):
                    raise HTTPException(status_code=400, detail=f"Cannot cancel conversation in '{job['status']}' status")
                cancelled = await _cancel_background_run(conv_service=conv_service, client_id=client_id, run_id=run_id)
                if not cancelled:
                    raise HTTPException(status_code=409, detail='Conversation already in terminal state')
                from services import redis_job_store
                await redis_job_store.mark_job_terminal(run_id=run_id, client_id=client_id, user_id=job.get('user_id', ''))
                self.logger.info(f"Background conversation {run_id} cancelled by user {current_user.get('user_id')}")
                return {'run_id': run_id, 'status': 'cancelled'}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error cancelling background conversation {run_id}: {e}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to cancel background conversation')

        @self.router.post('/jobs/{run_id}/notification-read', tags=['Background Jobs'])
        async def mark_notification_read(run_id: str=Path(..., min_length=1, max_length=100), current_user: dict=Depends(require_auth())):
            try:
                client_id = current_user.get('client_id')
                if not client_id:
                    raise HTTPException(status_code=401, detail='Authentication context incomplete')
                run_id = validate_id_parameter(run_id, 'run_id')
                from db_config.mongo_server import get_db
                db = await get_db()
                conv_service = ConversationService(db)
                updated = await conv_service.mark_notification_read(run_id=run_id, client_id=client_id)
                return {'run_id': run_id, 'notification_read': updated}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Error marking notification read for {run_id}: {e}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to mark notification read')

        @self.router.get('/dataframe-download', tags=['Agents'])
        async def download_dataframe(run_id: str=Query(..., description='run_id of the conversation'), df_name: str=Query(..., description='DataFrame variable name, e.g. _generated_dataframe_1_'), current_user: dict=Depends(require_auth())):
            import csv
            import io as _io
            import pandas as _pd
            from pathlib import Path as _Path
            client_id: str = current_user.get('client_id') or current_user.get('clientId', '')
            if not client_id:
                raise HTTPException(status_code=403, detail='client_id missing from token')
            mongo = self.mongo_manager
            doc = await mongo.db['conversations'].find_one({'run_id': run_id, 'client_id': client_id}, {'agent_responses.executor': 1})
            if not doc:
                raise HTTPException(status_code=404, detail='Conversation not found')
            executor_resp = (doc.get('agent_responses') or {}).get('executor') or {}
            dataframes = executor_resp.get('dataframes') or []
            parquet_path: Optional[str] = None
            for df_entry in dataframes:
                if df_entry.get('name') == df_name and df_entry.get('full_data_path'):
                    parquet_path = df_entry['full_data_path']
                    break
            if not parquet_path:
                raise HTTPException(status_code=404, detail='No full data file found for this DataFrame. It may not have been truncated.')
            p = _Path(parquet_path)
            if not p.exists():
                raise HTTPException(status_code=410, detail='Data file has been cleaned up and is no longer available.')
            df = await asyncio.to_thread(_pd.read_parquet, p)

            def _stream_csv(dataframe):
                buf = _io.StringIO()
                writer = csv.writer(buf)
                writer.writerow(dataframe.columns.tolist())
                yield buf.getvalue()
                buf.seek(0)
                buf.truncate(0)
                chunk_size = 500
                for start in range(0, len(dataframe), chunk_size):
                    chunk = dataframe.iloc[start:start + chunk_size]
                    for row in chunk.itertuples(index=False, name=None):
                        writer.writerow(row)
                    yield buf.getvalue()
                    buf.seek(0)
                    buf.truncate(0)
            safe_name = re.sub('[^a-zA-Z0-9_\\-]', '_', df_name)
            filename = f'{safe_name}_{run_id[:8]}.csv'
            return StreamingResponse(_stream_csv(df), media_type='text/csv', headers={'Content-Disposition': f'attachment; filename="{filename}"'})

        @self.router.post('/adhoc/upload')
        async def adhoc_upload_file(file: UploadFile=File(...), session_id: str=Form(...), current_user: dict=Depends(require_auth())):
            try:
                client_id = current_user.get('client_id')
                if not client_id:
                    raise HTTPException(status_code=401, detail='Authentication context incomplete')
                from services.adhoc_file_service import upload_file as adhoc_upload, AdhocFileError
                content = await file.read()
                metadata = await adhoc_upload(file_content=content, original_filename=file.filename or 'upload.csv', session_id=session_id, client_id=client_id)
                return {'filename': metadata['original_filename'], 'file_size': metadata['file_size_bytes'], 'file_names': metadata['file_names'], 'sheet_count': metadata['sheet_count'], 'session_id': session_id}
            except AdhocFileError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Ad-hoc upload failed: {e}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to upload file')

        @self.router.delete('/adhoc/file')
        async def adhoc_delete_file(session_id: str=Query(...), current_user: dict=Depends(require_auth())):
            try:
                client_id = current_user.get('client_id')
                if not client_id:
                    raise HTTPException(status_code=401, detail='Authentication context incomplete')
                from services.adhoc_file_service import delete_file as adhoc_delete
                deleted = await adhoc_delete(session_id=session_id, client_id=client_id)
                return {'success': deleted, 'session_id': session_id}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Ad-hoc delete failed: {e}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to delete file')

        @self.router.get('/adhoc/file-info')
        async def adhoc_file_info(session_id: str=Query(...), current_user: dict=Depends(require_auth())):
            try:
                client_id = current_user.get('client_id')
                if not client_id:
                    raise HTTPException(status_code=401, detail='Authentication context incomplete')
                from services.adhoc_file_service import get_file_metadata
                metadata = get_file_metadata(session_id)
                if metadata and metadata.get('client_id') != client_id:
                    return {'file': None}
                return {'file': metadata}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f'Ad-hoc file-info failed: {e}', exc_info=True)
                raise HTTPException(status_code=500, detail='Failed to get file info')
agents_router = AgentsRouter().router