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
from auth.auth import create_access_token, create_refresh_token, decode_refresh_token
from config.system_config import SUPER_ADMIN_EMAIL, SUPER_ADMIN_PASSWORD
from auth.token_blacklist import token_blacklist
from middleware.auth_middleware import require_auth, require_auth_flexible
from middleware.client_middleware import get_client_id_from_request  # MULTI-TENANT: Import client middleware
from util.metrics import record_query
from util.datasource_context import (
    build_client_datasource_catalog,
    normalize_datasource_context,
    resolve_client_metadata_path,
)
from util.audit_logger import (  # AUDIT: Import audit logging functions
    audit_login_success,
    audit_login_failure,
    audit_query_execution,
    audit_access_denied,
    AuditEventType,
    AuditSeverity,
    audit_logger
)
import time
import json
import re
import asyncio  # PERFORMANCE: For fire-and-forget audit logging


# SECURITY (Phase 2): Input validation helpers
def validate_id_parameter(param_value: str, param_name: str) -> str:
    """
    Validate ID parameters (session_id, run_id, user_id) to prevent injection attacks.
    
    Args:
        param_value: The ID value to validate
        param_name: The parameter name (for error messages)
        
    Returns:
        str: Validated ID value
        
    Raises:
        HTTPException: If validation fails
    """
    if not param_value or len(param_value) > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {param_name}: must be between 1 and 100 characters"
        )
    
    # Allow alphanumeric, hyphens, underscores only
    if not re.match(r'^[a-zA-Z0-9_-]+$', param_value):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {param_name}: must contain only alphanumeric characters, hyphens, and underscores"
        )
    
    return param_value


def format_discussion_history(messages: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """Format prior discuss-thread messages for scoped follow-up context."""
    if not messages:
        return ""

    lines: List[str] = []
    for message in messages[-12:]:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = str(message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")

    joined = "\n".join(lines)
    if len(joined) <= max_chars:
        return joined
    return joined[-max_chars:]


# SECURITY (Phase 2): Enhanced Pydantic schemas with validation
class FeedbackSchema(BaseModel):
    """
    Feedback submission schema with validation.
    
    Security: Validates run_id format, rating values, and limits comment length.
    """
    run_id: str = Field(
        ..., 
        min_length=1,
        max_length=100,
        description="Conversation run ID (UUID format recommended)"
    )
    rating: Literal['positive', 'negative'] = Field(
        ..., 
        description="Feedback rating (positive or negative only)"
    )
    comment: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional feedback comment (max 2000 chars)"
    )
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="User ID submitting feedback"
    )
    
    @field_validator('run_id', 'user_id')
    @classmethod
    def validate_id_format(cls, v):
        """Prevent injection attacks by validating ID format."""
        # Allow alphanumeric, hyphens, underscores only
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError(f"ID must contain only alphanumeric characters, hyphens, and underscores")
        return v
    
    @field_validator('comment')
    @classmethod
    def sanitize_comment(cls, v):
        """Sanitize comment to prevent injection attacks."""
        if v:
            # Remove null bytes and control characters
            v = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', v)
            v = v.strip()
        return v if v else None

class StopStreamRequest(BaseModel):
    """Stream stop request with validation."""
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Session ID to stop streaming"
    )
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        """Validate session ID format."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("session_id must contain only alphanumeric characters, hyphens, and underscores")
        return v

class AuthRequest(BaseModel):
    """Authentication request with validation."""
    email: EmailStr = Field(
        ...,
        description="User email address"
    )
    password: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Password"
    )

class UpdateSessionTitleRequest(BaseModel):
    """Session title update request with validation."""
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Session ID to update"
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="New session title (max 500 chars)"
    )
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="User ID requesting update"
    )
    
    @field_validator('session_id', 'user_id')
    @classmethod
    def validate_id_format(cls, v):
        """Validate ID format."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError(f"ID must contain only alphanumeric characters, hyphens, and underscores")
        return v
    
    @field_validator('title')
    @classmethod
    def sanitize_title(cls, v):
        """Sanitize title to prevent injection."""
        # Remove null bytes and control characters
        v = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', v)
        return v.strip()


class PinConversationRequest(BaseModel):
    """Pin conversation request payload."""
    conversation_id: str = Field(
        ...,
        min_length=24,
        max_length=24,
        description="MongoDB conversation ObjectId"
    )

    @field_validator('conversation_id')
    @classmethod
    def validate_conversation_id(cls, v):
        if not re.match(r'^[a-fA-F0-9]{24}$', v):
            raise ValueError("conversation_id must be a valid 24-character ObjectId")
        return v.lower()


class ResponseDiscussionRequest(BaseModel):
    """Scoped follow-up question about a single assistant response."""
    conversation_id: str = Field(
        ...,
        min_length=24,
        max_length=24,
        description="MongoDB conversation ObjectId for the linked assistant response",
    )
    source_run_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Run ID for the linked assistant response",
    )
    parent_question: Optional[str] = Field(
        default="",
        max_length=1000,
        description="Original user question that produced the assistant response",
    )
    session_id: Optional[str] = Field(
        default="",
        max_length=100,
        description="Optional session ID for the linked assistant response",
    )
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User question about the selected response",
    )
    response_context: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Text context extracted from the selected assistant response",
    )

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, v):
        if not re.match(r'^[a-fA-F0-9]{24}$', v or ""):
            raise ValueError("conversation_id must be a valid 24-character ObjectId")
        return v.lower()

    @field_validator("source_run_id", "session_id")
    @classmethod
    def validate_link_ids(cls, v):
        v = (v or "").strip()
        if not v:
            return ""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("IDs must contain only alphanumeric characters, hyphens, and underscores")
        return v

    @field_validator("question", "response_context", "parent_question")
    @classmethod
    def sanitize_text(cls, v):
        v = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", v or "")
        return v.strip()

class AgentsRouter:
    def __init__(self):
        self.router = APIRouter()
        self._add_agents_routes()
        self.orchestrator_manager = OrchestratorManager()
        self.logger = logging.getLogger(__name__)
        self.mongo_manager = MongoDBManager()

       
    def _add_agents_routes(self):
        async def resolve_request_datasource_context(
            *,
            client_id: str,
            requested_context: Optional[Dict[str, Any]] = None,
            session_id: Optional[str] = None,
            user_id: Optional[str] = None,
        ) -> Optional[Dict[str, Any]]:
            resolved_context = normalize_datasource_context(
                requested_context,
                client_id=client_id,
                allow_unavailable=False,
                fallback_to_default=True,
            )

            if session_id:
                existing_session = await self.mongo_manager.get_session_metadata(
                    session_id,
                    user_id=user_id,
                    client_id=client_id,
                )
                session_context = normalize_datasource_context(
                    (existing_session or {}).get("datasource_context"),
                    client_id=client_id,
                    allow_unavailable=True,
                    fallback_to_default=False,
                )
                if session_context:
                    if (
                        resolved_context
                        and resolved_context.get("datasource_key") != session_context.get("datasource_key")
                    ):
                        self.logger.info(
                            "Datasource context mismatch for existing session | session=%s | requested=%s | using=%s",
                            session_id,
                            resolved_context.get("datasource_key"),
                            session_context.get("datasource_key"),
                        )
                    return session_context

            return resolved_context

        @self.router.get("/agent_query_stream")
        async def agent_query_stream(
            request: Request,  # AUDIT: Added Request to get client IP
            input: Optional[str] = Query(None),
            session_id: Optional[str] = Query(None),
            datasource_key: Optional[str] = Query(None),
            business_unit: Optional[str] = Query(None),
            datasource_system: Optional[str] = Query(None),
            run_in_background: bool = Query(False, description="Force query to run as background job"),
            current_user: dict = Depends(require_auth_flexible())

        ):
            # Record metrics for this query
            start_time = time.time()
            error_occurred = False
            client_id = None
            
            try:
                # CRITICAL MULTI-TENANT SECURITY: Extract client_id from JWT
                # NO FALLBACK  - require explicit client_id from authenticated user
                if not current_user:
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required. Please log in."
                    )
                
                client_id = current_user.get("client_id")
                if not client_id:
                    # Token missing client_id - security violation
                    self.logger.error(
                        f"SECURITY: Token missing client_id for user {current_user.get('user_id', 'unknown')}"
                    )
                    raise HTTPException(
                        status_code=403,
                        detail="Invalid authentication token: missing client identifier. Please log in again."
                    )
                
                # Log the client access for audit trail
                self.logger.info(
                    f"[MULTI-TENANT] Query from client='{client_id}', user='{current_user.get('user_id')}'"
                )
                
                # Check conversation limit before executing query (super_admin bypasses)
                user_role = current_user.get("role", "user")
                if user_role != "super_admin":
                    from services.subscription_service import check_conversation_limit
                    from db_config.mongo_server import get_db

                    db = await get_db()
                    is_allowed, current_count, limit = await check_conversation_limit(client_id, db)

                    if not is_allowed:
                        error_msg = f"Conversation limit reached. You have used {current_count}/{limit} conversations. Please upgrade your plan."
                        self.logger.warning(
                            f"Conversation limit exceeded | client_id={client_id} | "
                            f"used={current_count} | limit={limit}"
                        )
                        # For SSE endpoints, send error event through stream instead of raising HTTPException
                        # This prevents EventSource from closing and being treated as auth failure
                        async def error_stream():
                            error_data = {
                                "type": "error",
                                "message": error_msg,
                                "status": "limit_exceeded",
                                "current_count": current_count,
                                "limit": limit
                            }
                            yield f"event: limit_error\n"
                            yield f"data: {json.dumps(error_data)}\n\n"
                            yield f"event: end\n"
                            yield f"data: {json.dumps({'status': 'ended'})}\n\n"

                        headers = {
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        }
                        return StreamingResponse(error_stream(), media_type="text/event-stream", headers=headers)
                
                # SECURITY: Always use authenticated user id from JWT token
                # Tokens always contain `_id`; older tokens may not contain `user_id`.
                authenticated_user_id = (
                    current_user.get("user_id")
                    or current_user.get("_id")
                    or "anonymous"
                )

                datasource_context = await resolve_request_datasource_context(
                    client_id=client_id,
                    requested_context={
                        "datasource_key": datasource_key,
                        "business_unit": business_unit,
                        "system": datasource_system,
                    },
                    session_id=session_id,
                    user_id=authenticated_user_id,
                )
                
                data = {
                    "input": input,
                    "files": None,
                    "user_id": authenticated_user_id,
                    "session_id": session_id or "default",
                    "client_id": client_id,  # MULTI-TENANT: Include client_id in data
                    "datasource_context": datasource_context,
                    "attempts": 0,
                    "history": [],
                    "run_in_background": run_in_background,
                }
                
                # AUDIT: Fire-and-forget for performance (don't await)
                client_ip = request.client.host if request.client else "unknown"
                asyncio.create_task(audit_query_execution(
                    user_id=authenticated_user_id,
                    client_id=client_id,
                    query=input or "",
                    execution_time=0.0,  # Will be updated in finally block
                    ip_address=client_ip
                ))
                
                generator = self.orchestrator_manager.stream(data)
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    # CORS handled by middleware
                }
                return StreamingResponse(generator, media_type="text/event-stream", headers=headers)
            except Exception as e:
                error_occurred = True
                self.logger.error(f"Error starting agent query stream: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to start agent query stream")
            finally:
                # Record query metrics
                duration_ms = (time.time() - start_time) * 1000
                record_query(client_id=client_id, duration_ms=duration_ms, error=error_occurred)
  
        @self.router.post("/business_insights_only")
        async def business_insights_only(
            request: Request,
            current_user: dict = Depends(require_auth())
        ):
            """Run only the business agent with existing executor results."""
            try:
                body = await request.json()
                
                # Validate required fields
                if not body.get("input"):
                    raise HTTPException(status_code=422, detail="Missing required field: input")
                if not body.get("executor_response"):
                    raise HTTPException(status_code=422, detail="Missing required field: executor_response")
                
                # MULTI-TENANT: Extract client_id from authenticated user
                client_id = current_user.get("client_id")
                if not client_id:
                    raise HTTPException(
                        status_code=400,
                        detail="client_id is required for multi-tenant operation. Please re-authenticate."
                    )
                
                data = {
                    "input": body.get("input"),
                    "executor_response": body.get("executor_response", {}),
                    "summarized_response": body.get("summarized_response"),
                    "user_id": current_user.get("user_id") or current_user.get("_id") or body.get("user_id", "anonymous"),
                    "session_id": body.get("session_id", "default"),
                    "client_id": client_id,  # MULTI-TENANT: Pass client_id to orchestrator
                }
                data["datasource_context"] = await resolve_request_datasource_context(
                    client_id=client_id,
                    requested_context=body.get("datasource_context") or {
                        "datasource_key": body.get("datasource_key"),
                        "business_unit": body.get("business_unit"),
                        "system": body.get("datasource_system"),
                    },
                    session_id=data["session_id"],
                    user_id=data["user_id"],
                )
                generator = self.orchestrator_manager.stream_business_only(data)
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
                return StreamingResponse(generator, media_type="text/event-stream", headers=headers)
            except HTTPException:
                raise
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in request body: {str(e)}")
                raise HTTPException(status_code=422, detail=f"Invalid JSON in request body: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error starting business insights stream: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to start business insights stream: {str(e)}")

        @self.router.post("/discuss-response", tags=["Agents"])
        async def discuss_response(
            payload: ResponseDiscussionRequest,
            current_user: dict = Depends(require_auth()),
        ):
            """Answer follow-up questions using only the selected response as context."""
            try:
                from bson import ObjectId
                client_id = current_user.get("client_id")
                user_id = current_user.get("user_id")
                if not client_id:
                    raise HTTPException(
                        status_code=400,
                        detail="Missing client context. Please log in again.",
                    )
                if not user_id:
                    raise HTTPException(status_code=401, detail="Missing user context. Please log in again.")

                from db_config.mongo_server import get_db
                from util.llm_utils import LLMClient

                db = await get_db()
                source_conversation = await db.conversations.find_one({
                    "_id": ObjectId(payload.conversation_id),
                    "run_id": payload.source_run_id,
                    "client_id": client_id,
                    "is_deleted": {"$ne": True},
                })
                if not source_conversation:
                    raise HTTPException(
                        status_code=404,
                        detail="The linked response could not be found for this user.",
                    )
                if str(source_conversation.get("user_id") or "").strip() != str(user_id).strip():
                    raise HTTPException(
                        status_code=403,
                        detail="Cannot discuss responses belonging to another user.",
                    )

                existing_thread = await self.mongo_manager.get_response_discussion(
                    conversation_id=payload.conversation_id,
                    source_run_id=payload.source_run_id,
                    user_id=user_id,
                    client_id=client_id,
                )
                llm = LLMClient(
                    agent_name="business_agent",
                    client_id=client_id,
                    db=db,
                )
                await llm._load_client_llm_config()

                response_context = payload.response_context[:12000]
                discussion_history = format_discussion_history(
                    (existing_thread or {}).get("messages", []),
                    max_chars=6000,
                )
                system_prompt = (
                    "You are answering follow-up questions about a single previously generated assistant response. "
                    "Use only the provided response context. "
                    "You may also use the prior discuss-thread history, but only when it is consistent with the same selected response. "
                    "The response context may include code generation details, execution results, tables, charts, metrics, insights, recommendations, etc. "
                    "Explain the technical logic when that context is present. "
                    "Do not introduce new analysis, new data claims, or unrelated chat context. "
                    "You may synthesize across the provided sections, but you must stay grounded in them. "
                    "If the answer is not stated or cannot be reasonably inferred from the provided response context, "
                    "say that the original response did not specify it. "
                    "If the question is unrelated to the provided response, say you can only discuss that response. "
                    "Keep the answer concise and direct."
                )
                user_message = (
                    f"RESPONSE CONTEXT:\n{response_context}\n\n"
                    + (
                        f"DISCUSS THREAD HISTORY:\n{discussion_history}\n\n"
                        if discussion_history
                        else ""
                    )
                    + 
                    f"FOLLOW-UP QUESTION:\n{payload.question}"
                )

                response = await llm.generate_completion(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    temperature=0.1,
                    max_tokens=1200,
                )

                answer = (response.get("content") or "").strip()
                if not answer:
                    answer = "The original response did not provide enough detail to answer that."

                llm_config = {
                    "provider": (llm.client_config or {}).get("provider"),
                    "model": (llm.client_config or {}).get("model"),
                }
                thread = await self.mongo_manager.save_response_discussion_turn(
                    client_id=client_id,
                    user_id=user_id,
                    conversation_id=payload.conversation_id,
                    source_run_id=payload.source_run_id,
                    session_id=payload.session_id or source_conversation.get("session_id"),
                    parent_question=payload.parent_question or "",
                    response_context=response_context,
                    source_conversation=source_conversation,
                    user_question=payload.question,
                    assistant_answer=answer,
                    assistant_usage=response.get("usage"),
                    llm_config=llm_config,
                )

                return {
                    "answer": answer,
                    "usage": response.get("usage"),
                    "thread": thread,
                }
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error discussing response: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to discuss response")

        @self.router.get("/discuss-response/{conversation_id}/{source_run_id}", tags=["Agents"])
        async def get_discuss_response_thread(
            conversation_id: str = Path(..., min_length=24, max_length=24, regex=r'^[a-fA-F0-9]{24}$'),
            source_run_id: str = Path(..., min_length=1, max_length=100, regex=r'^[a-zA-Z0-9_-]+$'),
            current_user: dict = Depends(require_auth()),
        ):
            """Load the persisted discuss-response thread for a specific assistant response."""
            try:
                from bson import ObjectId

                normalized_conversation_id = conversation_id.strip().lower()
                source_run_id = validate_id_parameter(source_run_id, "source_run_id")
                user_id = current_user.get("user_id")
                client_id = current_user.get("client_id")
                if not user_id or not client_id:
                    raise HTTPException(status_code=401, detail="User authentication context is incomplete")

                await self.mongo_manager.connect()
                db = self.mongo_manager.db
                if db is None:
                    raise HTTPException(status_code=500, detail="Database not available")

                source_conversation = await db.conversations.find_one({
                    "_id": ObjectId(normalized_conversation_id),
                    "run_id": source_run_id,
                    "client_id": client_id,
                    "is_deleted": {"$ne": True},
                })
                if not source_conversation:
                    raise HTTPException(status_code=404, detail="Linked response not found for this user")
                if str(source_conversation.get("user_id") or "").strip() != str(user_id).strip():
                    raise HTTPException(
                        status_code=403,
                        detail="Cannot access another user's linked response",
                    )

                thread = await self.mongo_manager.get_response_discussion(
                    conversation_id=normalized_conversation_id,
                    source_run_id=source_run_id,
                    user_id=user_id,
                    client_id=client_id,
                )
                if not thread:
                    return {
                        "found": False,
                        "conversation_id": normalized_conversation_id,
                        "source_run_id": source_run_id,
                        "messages": [],
                        "total_token_usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    }

                return {
                    "found": True,
                    "thread": thread,
                }
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error loading response discussion thread: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to load discussion thread")

        @self.router.post("/feedback", tags=["Feedback"])
        async def submit_feedback(
            feedback: FeedbackSchema,
            current_user: dict = Depends(require_auth())
        ):
            """
            Receives and saves user feedback for a conversation.
            
            Security: Validates conversation ownership before accepting feedback.
            Users can only submit feedback for their own conversations.
            """
            try:
                # Extract authenticated user info
                user_id = current_user["user_id"]
                client_id = current_user.get("client_id")
                
                # SECURITY: Verify conversation ownership before allowing feedback
                conversation = await self.mongo_manager.get_conversation_by_run_id(feedback.run_id)
                
                if not conversation:
                    raise HTTPException(status_code=404, detail="Conversation not found")
                
                # Validate ownership
                if conversation["user_id"] != user_id:
                    raise HTTPException(
                        status_code=403,
                        detail="Cannot submit feedback for other users' conversations"
                    )
                
                if client_id and conversation.get("client_id") != client_id:
                    raise HTTPException(
                        status_code=403,
                        detail="Cannot submit feedback for conversations from other clients"
                    )
                
                # Ensure the manager is connected
                await self.mongo_manager.connect()
                
                # Use transaction to ensure atomic feedback submission
                # Rollback if either feedback save or vector DB processing fails
                try:
                    async with self.mongo_manager.transaction() as session:
                        # Save feedback to conversation
                        update_result = await self.mongo_manager.save_feedback_async(
                            run_id=feedback.run_id,
                            rating=feedback.rating,
                            comment=feedback.comment or "",
                            user_id=user_id,  # Use validated user_id from token
                            session=session  # Transaction session
                        )
                        
                        if not update_result:
                            raise HTTPException(
                                status_code=404, 
                                detail=f"Conversation with run_id '{feedback.run_id}' not found."
                            )
                        
                        # Process positive feedback by adding to vector database
                        if feedback.rating.lower() == 'positive':
                            try:
                                from response_caching.feedback_processor import process_feedback_from_run_id
                                success = await process_feedback_from_run_id(feedback.run_id, self.mongo_manager)
                                if success:
                                    self.logger.info(f"Successfully added question to vector database for run_id: {feedback.run_id}")
                                else:
                                    # Raise exception to trigger rollback
                                    raise Exception("Failed to add question to vector database")
                            except Exception as vec_error:
                                self.logger.error(f"Error processing positive feedback for run_id {feedback.run_id}: {vec_error}")
                                # Rollback the transaction if vector DB processing fails
                                raise HTTPException(
                                    status_code=500,
                                    detail="Failed to process feedback. Please try again."
                                )
                        
                        # Transaction commits here if no exception raised
                        
                except Exception as tx_error:
                    # Transaction automatically rolled back
                    self.logger.error(f"Transaction failed for feedback {feedback.run_id}: {tx_error}")
                    # Re-raise HTTPException if it's already formatted
                    if isinstance(tx_error, HTTPException):
                        raise tx_error
                    # Otherwise wrap in generic error
                    raise HTTPException(
                        status_code=500,
                        detail=f"An error occurred while saving feedback: {str(tx_error)}"
                    )
                    
                return {"status": "success", "message": "Feedback recorded successfully."}
                
            except Exception as e:
                # In a real app, you'd have more robust logging here
                raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

        @self.router.get("/datasource-catalog", tags=["Suggest"])
        async def get_datasource_catalog(
            current_user: dict = Depends(require_auth())
        ):
            try:
                client_id = current_user.get("client_id")
                if not client_id:
                    raise HTTPException(status_code=403, detail="client_id missing from token")

                catalog = build_client_datasource_catalog(client_id)
                return {
                    **catalog,
                    "client_id": client_id,
                }
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error("Error loading datasource catalog: %s", e, exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to load datasource catalog")

        @self.router.get("/suggested-questions", tags=["Suggest"])
        async def get_suggested_questions(
            datasource_key: Optional[str] = Query(None),
            business_unit: Optional[str] = Query(None),
            datasource_system: Optional[str] = Query(None),
            current_user: dict = Depends(require_auth())
        ):
            """
            Get personalized suggested questions for the current client.
            Loads from xml_prompts/clients/{client_id}/data_sources/suggested_questions.xml
            
            Returns:
                List of suggested questions with categories, or empty list if file doesn't exist
            """
            datasource_context = None
            try:
                from pathlib import Path
                # SECURITY: Use defusedxml to prevent XXE attacks
                from defusedxml.ElementTree import parse
                
                client_id = current_user.get("client_id")
                if not client_id:
                    raise HTTPException(status_code=403, detail="client_id missing from token")
                
                datasource_context = normalize_datasource_context(
                    {
                        "datasource_key": datasource_key,
                        "business_unit": business_unit,
                        "system": datasource_system,
                    },
                    client_id=client_id,
                    allow_unavailable=False,
                    fallback_to_default=True,
                )

                # Load client-specific suggested questions from the selected datasource root.
                questions_path = resolve_client_metadata_path(
                    client_id,
                    "suggested_questions.xml",
                    datasource_context=datasource_context,
                    allow_legacy_when_context_missing=True,
                )

                if not questions_path or not questions_path.exists():
                    # Return empty list if file doesn't exist - don't show suggested questions
                    self.logger.info(f"suggested_questions.xml not found for {client_id} at {questions_path}, returning empty list")
                    return {
                        "questions": [],
                        "source": "none",
                        "client_id": client_id,
                        "datasource_context": datasource_context,
                    }
                
                # Parse XML
                try:
                    from xml.etree.ElementTree import ParseError
                    tree = parse(questions_path)
                    root = tree.getroot()
                    
                    questions = []
                    for question_elem in root.findall(".//question"):
                        question_text = question_elem.text or ""
                        question_id = question_elem.get("id", "")
                        category = question_elem.get("category", "general")
                        
                        if question_text:
                            questions.append({
                                "id": question_id,
                                "text": question_text,
                                "category": category
                            })
                    
                    self.logger.info(f"Loaded {len(questions)} suggested questions for client {client_id}")
                    
                    return {
                        "questions": questions,
                        "source": "client_specific",
                        "client_id": client_id,
                        "datasource_context": datasource_context,
                    }
                    
                except ParseError as e:
                    self.logger.error(f"Error parsing suggested_questions.xml for {client_id}: {e}")
                    # Return empty list on parse error
                    return {
                        "questions": [],
                        "source": "none",
                        "error": "Failed to parse XML",
                        "datasource_context": datasource_context,
                    }
                    
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error loading suggested questions: {e}", exc_info=True)
                # Return empty list on error
                return {
                    "questions": [],
                    "source": "none",
                    "error": str(e),
                }
        
        @self.router.get("/history", tags=["History"])
        async def get_conversation_history(
            page: int = 1, 
            limit: int = 10,
            sort_order: Literal['desc', 'asc'] = 'desc',
            feedback_filter: Optional[Literal['all', 'positive', 'negative', 'none']] = 'all',
            current_user: dict = Depends(require_auth())
        ):
            """
            Fetches paginated conversation history, sorted by the most recent.
            Supports filtering by feedback type (positive, negative, none, all).
            
            Security: Only returns conversations for the authenticated user and their client.
            user_id parameter removed - always uses authenticated user.
            """
            if page < 1 or limit < 1:
                raise HTTPException(status_code=400, detail="Page and limit must be positive integers.")

            try:
                await self.mongo_manager.connect()
                db = self.mongo_manager.db
                if db is None:
                    raise HTTPException(status_code=500, detail="Database not available")
                collection = db.conversations
                
                # SECURITY: Use authenticated user's credentials
                user_id = current_user["user_id"]
                client_id = current_user.get("client_id")
                
                if not client_id:
                    raise HTTPException(status_code=403, detail="client_id missing from token")
                
                # Build the filter query - ALWAYS filter by user and client
                base_filter = {
                    "user_id": user_id,      # Authenticated user only
                    "client_id": client_id,   # Their client only
                    "is_deleted": {"$ne": True}  # Exclude deleted conversations
                    
                }
                
                # Add feedback filter
                if feedback_filter == 'positive':
                    base_filter["feedback.rating"] = "positive"
                elif feedback_filter == 'negative':
                    base_filter["feedback.rating"] = "negative"
                elif feedback_filter == 'none':
                    # Match docs where feedback is absent (new schema) OR null (legacy)
                    base_filter["$or"] = [
                        {"feedback": {"$exists": False}},
                        {"feedback": None}
                    ]
                
                # Get total records for the filtered query
                total_records = await collection.count_documents(base_filter)
                if total_records == 0:
                    return {
                        "data": [],
                        "currentPage": 1,
                        "totalPages": 0,
                        "totalRecords": 0,
                        "feedbackCounts": await get_feedback_counts(collection, user_id),
                        "appliedFilter": feedback_filter
                    }
                
                # Calculate pagination
                skip = (page - 1) * limit
                total_pages = math.ceil(total_records / limit)
                
                # Define sort direction
                sort_direction = -1 if sort_order == 'desc' else 1
                
                # Fetch the filtered and paginated data from MongoDB
                cursor = collection.find(base_filter).sort("created_at", sort_direction).skip(skip).limit(limit)
                history_data = await cursor.to_list(length=limit)
                
                for item in history_data:
                    item["id"] = str(item["_id"])
                    del item["_id"]
                    normalize_conversation_doc(item)
                
                # Get feedback counts for all categories
                feedback_counts = await get_feedback_counts(collection, user_id)
                
                return jsonable_encoder({
                    "data": history_data,
                    "currentPage": page,
                    "totalPages": total_pages,
                    "totalRecords": total_records,
                    "feedbackCounts": feedback_counts,
                    "appliedFilter": feedback_filter
                })
                
            except Exception as e:
                # Add logging for production apps
                raise HTTPException(status_code=500, detail=f"An error occurred while fetching history: {e}")


        async def get_feedback_counts(collection, user_id: Optional[str] = None):
            """
            Helper function to get counts for all feedback categories.
            Returns a dictionary with counts for all, positive, negative, and none.
            Filters by user_id if provided.
            """
            try:
                # Build match filter for user
                user_match = {"user_id": user_id, "is_deleted": {"$ne": True}} if user_id else {"is_deleted": {"$ne": True}}
                
                pipeline = [
                    {"$match": user_match} if user_match else {"$match": {}},
                    {
                        "$facet": {
                            "all": [{"$count": "count"}],
                            "positive": [
                                {"$match": {"feedback.rating": "positive"}},
                                {"$count": "count"}
                            ],
                            "negative": [
                                {"$match": {"feedback.rating": "negative"}},
                                {"$count": "count"}
                            ],
                            "none": [
                                {"$match": {"$or": [
                                    {"feedback": {"$exists": False}},
                                    {"feedback": None}
                                ]}},
                                {"$count": "count"}
                            ]
                        }
                    }
                ]
                
                result = await collection.aggregate(pipeline).to_list(length=1)
                
                if result:
                    counts = result[0]
                    return {
                        "all": counts["all"][0]["count"] if counts["all"] else 0,
                        "positive": counts["positive"][0]["count"] if counts["positive"] else 0,
                        "negative": counts["negative"][0]["count"] if counts["negative"] else 0,
                        "none": counts["none"][0]["count"] if counts["none"] else 0
                    }
                else:
                    return {"all": 0, "positive": 0, "negative": 0, "none": 0}
                    
            except Exception as e:
                # Return default counts if aggregation fails
                return {"all": 0, "positive": 0, "negative": 0, "none": 0}

        
        @self.router.post("/agent_stop_stream")
        async def agent_stop_stream(
            request: StopStreamRequest,
            current_user: dict = Depends(require_auth())
        ):
            """Endpoint to signal a stream to stop."""
            session_id = request.session_id
            if not session_id:
                raise HTTPException(status_code=400, detail="session_id is required")
            
            try:
                stopped = await self.orchestrator_manager.stop_stream(session_id)
                if stopped:
                    return {"message": f"Stop signal sent for session {session_id}."}
                else:
                    return {"message": f"No active stream found for session {session_id}."}
            except Exception as e:
                self.logger.error(f"Error stopping stream for session {session_id}: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to stop agent stream")
        
        @self.router.post("/auth")
        async def authenticate_user(request: AuthRequest, http_request: Request, background_tasks: BackgroundTasks):
            """
            Generic multi-tenant authentication using MongoDB.
            
            Authenticates users from ANY client by:
            1. Querying MongoDB users collection by username
            2. Verifying password hash with bcrypt
            3. Extracting user's client_id, role, and other data
            4. Generating JWT token with user information
            
            AUDIT: Logs all authentication attempts (success and failure).
            SECURITY: No hardcoded users, no client-specific logic.
            """
            try:
                from db_config.mongo_server import get_db
                from auth.auth import verify_password
                
                # Extract client IP for audit logging
                client_ip = http_request.client.host if http_request.client else "unknown"
                user_agent = http_request.headers.get("user-agent", "unknown")
                
                # Normalize email to lowercase for consistent lookup
                email_lower = request.email.lower().strip()
                self.logger.info(f"Authentication attempt | email={email_lower} | ip={client_ip}")
                
                # ============================================================
                # SUPER-ADMIN: Check env credentials first (unified login)
                # ============================================================
                if email_lower == SUPER_ADMIN_EMAIL.lower() and request.password == SUPER_ADMIN_PASSWORD:
                    token_data = {
                        "user_id": "super_admin",
                        "_id": "super_admin",
                        "email": SUPER_ADMIN_EMAIL,
                        "username": "super_admin",
                        "client_id": "super_admin",
                        "role": "super_admin"
                    }
                    token = create_access_token(data=token_data)
                    refresh_token = create_refresh_token(data=token_data)
                    asyncio.create_task(audit_login_success(
                        user_id="super_admin",
                        client_id="super_admin",
                        ip_address=client_ip,
                        user_agent=user_agent
                    ))
                    self.logger.info(f"Super admin login successful | email={SUPER_ADMIN_EMAIL}")
                    return {
                        "is_valid": 1,
                        "user_id": "super_admin",
                        "token": token,
                        "refresh_token": refresh_token,
                        "user_data": {
                            "user_id": "super_admin",
                            "_id": "super_admin",
                            "email": SUPER_ADMIN_EMAIL,
                            "username": "super_admin",
                            "full_name": "Super Admin",
                            "client_id": "super_admin",
                            "role": "super_admin"
                        }
                    }
                
                # ============================================================
                # MONGODB AUTHENTICATION (Generic Multi-Tenant)
                # ============================================================
                
                # Get MongoDB database connection
                import time
                perf_start = time.time()
                db = await get_db()
                self.logger.debug(f"[PERF] DB connection: {(time.time() - perf_start)*1000:.2f}ms")
                
                # Find user by email in MongoDB
                perf_start = time.time()
                user = await db.users.find_one({"email": email_lower})
                self.logger.debug(f"[PERF] User query: {(time.time() - perf_start)*1000:.2f}ms")
                
                if not user:
                    # User not found
                    self.logger.warning(f"Authentication failed - user not found | email={email_lower}")
                    
                    # AUDIT: Fire-and-forget for performance (don't await)
                    asyncio.create_task(audit_login_failure(
                        email=email_lower,
                        ip_address=client_ip,
                        reason="User not found"
                    ))
                    
                    return {
                        "is_valid": 0,
                        "message": "Invalid email or password"
                    }
                
                # Verify password using bcrypt
                perf_start = time.time()
                password_valid = verify_password(request.password, user.get("hashed_password", ""))
                self.logger.debug(f"[PERF] Password verify: {(time.time() - perf_start)*1000:.2f}ms")
                
                if not password_valid:
                    # Invalid password
                    self.logger.warning(f"Authentication failed - invalid password | email={email_lower}")
                    
                    # AUDIT: Fire-and-forget for performance (don't await)
                    asyncio.create_task(audit_login_failure(
                        email=email_lower,
                        ip_address=client_ip,
                        reason="Invalid password"
                    ))
                    
                    return {
                        "is_valid": 0,
                        "message": "Invalid email or password"
                    }
                
                # Check if user account is active
                if not user.get("is_active", True):
                    self.logger.warning(f"Authentication failed - account disabled | email={email_lower}")
                    
                    # AUDIT: Fire-and-forget for performance (don't await)
                    asyncio.create_task(audit_login_failure(
                        email=email_lower,
                        ip_address=client_ip,
                        reason="Account disabled"
                    ))
                    
                    return {
                        "is_valid": 0,
                        "message": "Account is disabled. Please contact your administrator."
                    }

                # Check if email is verified
                if not user.get("is_email_verified", True):
                    self.logger.warning(f"Authentication failed - email not verified | email={email_lower}")

                    asyncio.create_task(audit_login_failure(
                        email=email_lower,
                        ip_address=client_ip,
                        reason="Email not verified"
                    ))

                    return {
                        "is_valid": 0,
                        "message": "Please verify your email address before signing in. Check your inbox for the verification link.",
                        "email_not_verified": True,
                        "email": email_lower
                    }

                # Extract user information
                user_id = str(user.get("_id"))
                email = user.get("email")
                username = user.get("username", "")  # Keep for backward compatibility, may be empty
                full_name = user.get("full_name", email.split("@")[0] if email else "User")
                client_id = user.get("client_id")
                role = user.get("role", "user")
                
                # Validate client_id exists (required for multi-tenant)
                if not client_id:
                    self.logger.error(f"Authentication failed - no client_id | email={email_lower}")
                    
                    # AUDIT: Fire-and-forget for performance (don't await)
                    asyncio.create_task(audit_login_failure(
                        email=email_lower,
                        ip_address=client_ip,
                        reason="User has no client_id (data integrity issue)"
                    ))
                    
                    return {
                        "is_valid": 0,
                        "message": "Account configuration error. Please contact support."
                    }
                
                # Check tenant status (block login if suspended or deleted)
                from services.tenant_service import get_tenant_status
                tenant_status = await get_tenant_status(client_id, db)
                if tenant_status in ["suspended", "deleted"]:
                    status_message = {
                        "suspended": "Your account has been suspended. Please contact support.",
                        "deleted": "Your account has been deleted. Please contact support."
                    }.get(tenant_status, "Your account is not active. Please contact support.")
                    
                    self.logger.warning(f"Authentication blocked - tenant {tenant_status} | email={email_lower} | client_id={client_id}")
                    
                    # AUDIT: Fire-and-forget for performance (don't await)
                    asyncio.create_task(audit_login_failure(
                        email=email_lower,
                        ip_address=client_ip,
                        reason=f"Tenant status: {tenant_status}"
                    ))
                    
                    return {
                        "is_valid": 0,
                        "message": status_message
                    }
                
                # Create JWT token with user data
                token_data = {
                    "user_id": user_id,
                    "_id": user_id,
                    "email": email,
                    "username": username,
                    "client_id": client_id,
                    "role": role
                }
                
                perf_start = time.time()
                token = create_access_token(data=token_data)
                refresh_token = create_refresh_token(data=token_data)
                self.logger.debug(f"[PERF] Token creation: {(time.time() - perf_start)*1000:.2f}ms")
                
                # AUDIT: Fire-and-forget for performance (don't await)
                # This allows the auth response to return immediately while logging happens in background
                asyncio.create_task(audit_login_success(
                    user_id=user_id,
                    client_id=client_id,
                    ip_address=client_ip,
                    user_agent=user_agent
                ))
                
                self.logger.info(
                    f"✅ Authentication successful | email={email} | "
                    f"client_id={client_id} | role={role}"
                )
                
                # Get user config if it exists
                user_config = user.get("config", {})
                
                # Return success response
                return {
                    "is_valid": 1,
                    "user_id": user_id,
                    "token": token,
                    "refresh_token": refresh_token,
                    "user_data": {
                        "user_id": user_id,
                        "_id": user_id,
                        "email": email,
                        "username": username,
                        "full_name": full_name,
                        "client_id": client_id,
                        "role": role,
                        "config": user_config  # Include user config in response
                    }
                }
                    
            except Exception as e:
                self.logger.error(f"Authentication error: {str(e)}", exc_info=True)
                
                # AUDIT: Fire-and-forget for performance (don't await)
                asyncio.create_task(audit_login_failure(
                    username=request.username,
                    ip_address=client_ip,
                    reason=f"Exception: {str(e)}"
                ))
                
                raise HTTPException(
                    status_code=500,
                    detail="Authentication failed due to server error. Please try again."
                )
        
        @self.router.post("/auth/refresh", tags=["Authentication"])
        async def refresh_token(request: Request):
            """
            Refresh access token using a valid refresh token.
            Returns a new access token and refresh token (rotation).
            """
            try:
                # Get refresh token from Authorization header
                auth_header = request.headers.get("Authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    raise HTTPException(status_code=401, detail="Missing or invalid refresh token")
                
                refresh_token = auth_header.split(" ")[1]
                
                # Check if refresh token is blacklisted
                if await token_blacklist.is_refresh_token_blacklisted(refresh_token):
                    raise HTTPException(status_code=401, detail="Refresh token has been revoked")
                
                # Decode refresh token
                payload = decode_refresh_token(refresh_token)
                if not payload:
                    raise HTTPException(status_code=401, detail="Invalid refresh token")
                
                # Create new tokens
                token_data = {
                    "_id": payload.get("_id") or payload.get("user_id"),
                    "user_id": payload.get("_id") or payload.get("user_id"),
                    "email": payload.get("email"),
                    "username": payload.get("username"),
                    "client_id": payload.get("client_id"),
                    "role": payload.get("role")
                }
                
                new_access_token = create_access_token(data=token_data)
                new_refresh_token = create_refresh_token(data=token_data)
                
                # Blacklist old refresh token (rotation)
                from datetime import timedelta
                expires_at = utcnow() + timedelta(days=30)
                await token_blacklist.blacklist_refresh_token(refresh_token, expires_at)
                
                return {
                    "token": new_access_token,
                    "refresh_token": new_refresh_token
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Token refresh error: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to refresh token")
        
        @self.router.post("/auth/logout", tags=["Authentication"])
        async def logout(request: Request, current_user: dict = Depends(require_auth())):
            """
            Logout user by blacklisting their tokens.
            """
            try:
                # Get token from Authorization header
                auth_header = request.headers.get("Authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    token = auth_header.split(" ")[1]
                    
                    # Blacklist access token
                    from datetime import timedelta
                    expires_at = utcnow() + timedelta(hours=8)
                    await token_blacklist.blacklist_token(token, expires_at)
                
                # If refresh token is provided in body, blacklist it too
                try:
                    body = await request.json()
                    refresh_token = body.get("refresh_token")
                    if refresh_token:
                        expires_at = utcnow() + timedelta(days=30)
                        await token_blacklist.blacklist_refresh_token(refresh_token, expires_at)
                except:
                    pass  # No refresh token provided, that's okay
                
                self.logger.info(f"User logged out | user_id={current_user.get('user_id', 'unknown')}")
                
                return {"message": "Logged out successfully"}
                
            except Exception as e:
                self.logger.error(f"Logout error: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to logout")
        
        @self.router.put("/users/me/config", tags=["User Settings"])
        async def update_own_config(
            request: Request,
            config_data: Dict[str, Any],
            current_user: dict = Depends(require_auth())
        ):
            """
            Update the current user's config (theme, development steps, etc.)
            Allows any authenticated user (user or admin) to update their own settings.
            Stores config in the user collection's config field in MongoDB.
            
            **Permissions**: Any authenticated user
            
            **Body**: Dictionary with config fields:
            - theme: "light" or "dark" (default: "light")
            - show_development_steps: boolean (default: true)
            - business_insights_sections: dict with keys (summary, metrics, insights, recommendations, follow_ups, note) and boolean values (all default: true)
            
            **Returns**: Updated config
            """
            try:
                from db_config.mongo_server import get_db
                
                user_email = current_user.get("email")
                if not user_email:
                    raise HTTPException(status_code=401, detail="User not authenticated")
                
                # Get database connection
                db = await get_db()
                
                # Find user in MongoDB
                user = await db.users.find_one({"email": user_email.lower()})
                if not user:
                    raise HTTPException(status_code=404, detail="User not found")
                
                # Validate config data
                if "theme" in config_data and config_data["theme"] not in ["light", "dark"]:
                    raise HTTPException(status_code=400, detail="theme must be 'light' or 'dark'")
                
                if "show_development_steps" in config_data and not isinstance(config_data["show_development_steps"], bool):
                    raise HTTPException(status_code=400, detail="show_development_steps must be a boolean")
                
                # Validate business_insights_sections structure
                if "business_insights_sections" in config_data:
                    valid_keys = {"summary", "metrics", "insights", "recommendations", "follow_ups", "note"}
                    sections = config_data["business_insights_sections"]
                    if not isinstance(sections, dict):
                        raise HTTPException(status_code=400, detail="business_insights_sections must be a dictionary")
                    for key in sections:
                        if key not in valid_keys:
                            raise HTTPException(status_code=400, detail=f"Invalid business_insights_sections key: {key}. Valid keys are: {', '.join(sorted(valid_keys))}")
                        if not isinstance(sections[key], bool):
                            raise HTTPException(status_code=400, detail=f"business_insights_sections.{key} must be a boolean")

                # Validate pinned_query_ids structure
                if "pinned_query_ids" in config_data:
                    from bson import ObjectId
                    pinned_ids = config_data["pinned_query_ids"]
                    if not isinstance(pinned_ids, list):
                        raise HTTPException(status_code=400, detail="pinned_query_ids must be an array of conversation IDs")
                    deduped_ids: List[str] = []
                    seen = set()
                    for raw_id in pinned_ids:
                        if not isinstance(raw_id, str):
                            raise HTTPException(status_code=400, detail="pinned_query_ids must contain only string values")
                        normalized = raw_id.strip().lower()
                        if not ObjectId.is_valid(normalized):
                            raise HTTPException(status_code=400, detail=f"Invalid conversation ID in pinned_query_ids: {raw_id}")
                        if normalized not in seen:
                            seen.add(normalized)
                            deduped_ids.append(normalized)
                    if len(deduped_ids) > 3:
                        raise HTTPException(status_code=400, detail="Maximum 3 pinned queries are allowed")
                    config_data["pinned_query_ids"] = deduped_ids
                
                # Get existing config or create new empty dict
                existing_config = user.get("config", {})
                
                # Merge new config with existing config
                merged_config = {**existing_config, **config_data}
                
                # Update user document in MongoDB - save config field
                await db.users.update_one(
                    {"email": user_email.lower()},
                    {"$set": {"config": merged_config, "updated_at": utcnow()}}
                )
                
                self.logger.info(f"User {user_email} updated their config")
                
                # Return updated config
                return {
                    "success": True,
                    "message": "Config updated successfully",
                    "config": merged_config
                }
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error updating user config: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

        @self.router.get("/users/me/conversations/by-run/{run_id}", tags=["User Settings"])
        async def get_conversation_id_by_run_id(
            run_id: str = Path(..., min_length=1, max_length=100, regex=r'^[a-zA-Z0-9_-]+$'),
            current_user: dict = Depends(require_auth())
        ):
            """Resolve persisted conversation ID from run_id after stream completion."""
            try:
                run_id = validate_id_parameter(run_id, "run_id")
                await self.mongo_manager.connect()
                db = self.mongo_manager.db
                if db is None:
                    raise HTTPException(status_code=500, detail="Database not available")

                user_id = current_user.get("user_id")
                client_id = current_user.get("client_id")
                if not user_id or not client_id:
                    raise HTTPException(status_code=401, detail="User authentication context is incomplete")

                conversation = await db.conversations.find_one({
                    "run_id": run_id,
                    "user_id": user_id,
                    "client_id": client_id,
                    "is_deleted": {"$ne": True},
                })
                if not conversation:
                    return {"found": False, "run_id": run_id}

                return {
                    "found": True,
                    "run_id": run_id,
                    "conversation_id": str(conversation.get("_id")),
                    "enhanced_question": conversation.get("enhanced_question", ""),
                    "input": conversation.get("input", ""),
                }
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error resolving conversation by run_id: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to resolve conversation")

        @self.router.get("/users/me/pinned-queries", tags=["User Settings"])
        async def get_pinned_queries(current_user: dict = Depends(require_auth())):
            """Fetch the current user's pinned query metadata."""
            try:
                from bson import ObjectId
                from db_config.mongo_server import get_db

                user_email = current_user.get("email")
                user_id = current_user.get("user_id")
                client_id = current_user.get("client_id")
                if not user_email or not user_id or not client_id:
                    raise HTTPException(status_code=401, detail="User not authenticated")

                db = await get_db()
                user = await db.users.find_one({"email": user_email.lower()})
                if not user:
                    raise HTTPException(status_code=404, detail="User not found")

                config = user.get("config", {}) or {}
                pinned_ids_raw = config.get("pinned_query_ids", []) or []

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
                    return {"pinned_questions": [], "pinned_query_ids": []}

                object_ids = [ObjectId(conv_id) for conv_id in valid_ids]
                conversations = await db.conversations.find({
                    "_id": {"$in": object_ids},
                    "user_id": user_id,
                    "client_id": client_id,
                    "is_deleted": {"$ne": True},
                }).to_list(length=10)
                conv_map = {str(c["_id"]).lower(): c for c in conversations}

                pinned_questions = []
                cleaned_ids: List[str] = []
                for conv_id in valid_ids:
                    conversation = conv_map.get(conv_id)
                    if not conversation:
                        continue
                    cleaned_ids.append(conv_id)
                    pinned_questions.append({
                        "conversation_id": conv_id,
                        "run_id": conversation.get("run_id"),
                        "enhanced_question": conversation.get("enhanced_question", "") or "",
                        "input": conversation.get("input", "") or "",
                        "pinned_at": conversation.get("created_at").isoformat() if conversation.get("created_at") else None,
                    })

                if cleaned_ids != valid_ids:
                    merged_config = {**config, "pinned_query_ids": cleaned_ids}
                    await db.users.update_one(
                        {"email": user_email.lower()},
                        {"$set": {"config": merged_config, "updated_at": utcnow()}}
                    )

                return {
                    "pinned_questions": pinned_questions,
                    "pinned_query_ids": cleaned_ids,
                }
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error fetching pinned queries: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to fetch pinned queries")

        @self.router.post("/users/me/pinned-queries", tags=["User Settings"])
        async def pin_query(
            request_data: PinConversationRequest,
            background_tasks: BackgroundTasks,
            current_user: dict = Depends(require_auth())
        ):
            """Pin a conversation query to user config (max 3, idempotent)."""
            try:
                from bson import ObjectId
                from db_config.mongo_server import get_db
                from response_caching.feedback_processor import process_feedback_from_conversation_id

                user_email = current_user.get("email")
                user_id = current_user.get("user_id")
                client_id = current_user.get("client_id")
                if not user_email or not user_id or not client_id:
                    raise HTTPException(status_code=401, detail="User not authenticated")

                db = await get_db()
                user = await db.users.find_one({"email": user_email.lower()})
                if not user:
                    raise HTTPException(status_code=404, detail="User not found")

                conversation_id = request_data.conversation_id.strip().lower()
                conversation = await db.conversations.find_one({
                    "_id": ObjectId(conversation_id),
                    "user_id": user_id,
                    "client_id": client_id,
                    "is_deleted": {"$ne": True},
                })
                if not conversation:
                    raise HTTPException(status_code=404, detail="Conversation not found for this user")

                existing_config = user.get("config", {}) or {}
                pinned_ids = existing_config.get("pinned_query_ids", []) or []
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
                    return {
                        "success": True,
                        "message": "Query already pinned",
                        "pinned_query_ids": normalized,
                    }

                if len(normalized) >= 3:
                    raise HTTPException(status_code=400, detail="Maximum 3 pinned queries reached. Unpin one to continue.")

                updated_ids = [conversation_id, *normalized][:3]
                updated_config = {**existing_config, "pinned_query_ids": updated_ids}
                await db.users.update_one(
                    {"email": user_email.lower()},
                    {"$set": {"config": updated_config, "updated_at": utcnow()}}
                )

                async def _warm_cache():
                    try:
                        await process_feedback_from_conversation_id(conversation_id, self.mongo_manager)
                    except Exception as warm_err:
                        self.logger.warning(f"Pinned cache warmup failed for conversation_id={conversation_id}: {warm_err}")

                background_tasks.add_task(_warm_cache)

                return {
                    "success": True,
                    "message": "Query pinned successfully",
                    "pinned_query_ids": updated_ids,
                }
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error pinning query: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to pin query")

        @self.router.delete("/users/me/pinned-queries/{conversation_id}", tags=["User Settings"])
        async def unpin_query(
            conversation_id: str = Path(..., min_length=24, max_length=24, regex=r'^[a-fA-F0-9]{24}$'),
            current_user: dict = Depends(require_auth())
        ):
            """Unpin a conversation from the current user."""
            try:
                from bson import ObjectId
                from db_config.mongo_server import get_db

                user_email = current_user.get("email")
                if not user_email:
                    raise HTTPException(status_code=401, detail="User not authenticated")

                normalized_id = conversation_id.strip().lower()
                if not ObjectId.is_valid(normalized_id):
                    raise HTTPException(status_code=400, detail="Invalid conversation_id")

                db = await get_db()
                user = await db.users.find_one({"email": user_email.lower()})
                if not user:
                    raise HTTPException(status_code=404, detail="User not found")

                existing_config = user.get("config", {}) or {}
                pinned_ids = existing_config.get("pinned_query_ids", []) or []
                updated_ids = [
                    pid.strip().lower()
                    for pid in pinned_ids
                    if isinstance(pid, str) and pid.strip().lower() != normalized_id and ObjectId.is_valid(pid.strip())
                ][:3]

                updated_config = {**existing_config, "pinned_query_ids": updated_ids}
                await db.users.update_one(
                    {"email": user_email.lower()},
                    {"$set": {"config": updated_config, "updated_at": utcnow()}}
                )

                return {
                    "success": True,
                    "message": "Query unpinned successfully",
                    "pinned_query_ids": updated_ids,
                }
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error unpinning query: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to unpin query")
        
        @self.router.get("/check-client-data", tags=["Authentication"])
        async def check_client_data(current_user: dict = Depends(require_auth())):
            """
            Check if client has db_credentials saved in MongoDB.
            Checks for any db_type (postgres, mysql, file_upload, etc.) in db_credentials collection.
            
            Returns:
                - has_data: bool - Whether client has any db_credentials saved
                - config_type: str - Type of configuration (db_type from db_credentials)
                - is_admin: bool - Whether user is admin
            """
            try:
                from db_config.mongo_server import get_db
                
                client_id = current_user.get("client_id")
                role = current_user.get("role", "user")
                is_admin = role == "admin"
                
                # Check if any db_credentials exist in MongoDB for this client
                db = await get_db()
                db_credential = await db.db_credentials.find_one({"client_id": client_id})
                
                has_data = db_credential is not None
                config_type = db_credential.get("db_type") if db_credential else None
                
                self.logger.info(
                    f"Client data check | client_id={client_id} | user={current_user.get('username')} | "
                    f"has_db_credentials={has_data} | db_type={config_type} | is_admin={is_admin}"
                )
                
                return {
                    "has_data": has_data,
                    "config_type": config_type,
                    "is_admin": is_admin,
                    "client_id": client_id
                }
                
            except Exception as e:
                self.logger.error(f"Error checking client data: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail="Failed to check client data"
                )
        
        @self.router.get("/sessions", tags=["Sessions"])
        async def get_user_sessions(
            page: int = Query(1, ge=1),
            limit: int = Query(20, ge=1, le=100),
            current_user: dict = Depends(require_auth())
        ):
            """
            Get chat sessions.
            
            - Regular users: Returns only their own sessions
            - Admins: Returns all sessions for users in their client
            """
            try:
                user_role = current_user.get("role", "user")
                client_id = current_user.get("client_id")
                
                # Check if user is admin
                is_admin = user_role in ["admin", "super_admin"]
                
                if is_admin:
                    # Admin: Get all sessions for the client
                    result = await self.mongo_manager.get_user_sessions(
                        user_id=None,  # None = get all users in client
                        page=page,
                        limit=limit,
                        client_id=client_id,
                        is_admin=True,
                        include_user_info=True  # Include user email/name
                    )
                else:
                    # Regular user: Get only their own sessions
                    user_id = current_user["user_id"]
                    result = await self.mongo_manager.get_user_sessions(
                        user_id=user_id,
                        page=page,
                        limit=limit,
                        client_id=client_id,
                        is_admin=False,
                        include_user_info=False
                    )
                
                if result is None:
                    raise HTTPException(status_code=500, detail="Failed to fetch sessions")
                return result
            except Exception as e:
                self.logger.error(f"Error fetching user sessions: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to fetch sessions: {str(e)}")
        
        @self.router.get("/sessions/{session_id}", tags=["Sessions"])
        async def get_session_conversations(
            session_id: str = Path(..., min_length=1, max_length=100, regex=r'^[a-zA-Z0-9_-]+$'),
            current_user: dict = Depends(require_auth())
        ):
            """
            Get all conversations for a specific session.
            Returns complete conversation history with all questions and responses.
            
            Security: Validates session ownership before returning data.
            Path parameter validation prevents injection attacks.
            """
            try:
                # SECURITY (Phase 2): Validate session_id format
                session_id = validate_id_parameter(session_id, "session_id")
                
                # Extract authenticated user info
                user_id = current_user.get("user_id") or current_user.get("_id")
                client_id = current_user.get("client_id")
                user_role = current_user.get("role", "user")

                # Check if user is admin
                is_admin = user_role in ["admin", "super_admin"]
                
                # Verify session ownership (pass None for user_id if admin to allow viewing any session)
                # Pass client_id to ensure admin can only view sessions from their client
                session = await self.mongo_manager.get_session_metadata(
                    session_id, 
                    user_id=user_id if not is_admin else None,
                    client_id=client_id if is_admin else None
                )
                
                if not session:
                    raise HTTPException(status_code=404, detail="Session not found")
                
                # Validate ownership
                session_user_id = session.get("user_id")
                if not session_user_id:
                    raise HTTPException(
                        status_code=404,
                        detail="Session metadata incomplete: missing user_id"
                    )
                
                # Convert both values to strings for comparison (handle ObjectId from both sources)
                from bson import ObjectId
                
                # Convert session_user_id to string
                if isinstance(session_user_id, ObjectId):
                    session_user_id = str(session_user_id)
                else:
                    session_user_id = str(session_user_id)
                
                # Convert token user_id to string (it might be ObjectId if decode_access_token converted it)
                if isinstance(user_id, ObjectId):
                    user_id = str(user_id)
                else:
                    user_id = str(user_id)
                
                # Normalize both values for comparison (strip whitespace)
                session_user_id = session_user_id.strip()
                user_id = user_id.strip()
                
                # Log ownership check (use info level so it shows up)
                self.logger.info(
                    f"Session ownership check | session_id={session_id} | "
                    f"session_user_id='{session_user_id}' (type: {type(session_user_id).__name__}) | "
                    f"token_user_id='{user_id}' (type: {type(user_id).__name__}) | "
                    f"match={session_user_id == user_id} | "
                    f"session_user_id_repr={repr(session_user_id)} | token_user_id_repr={repr(user_id)}"
                )
                
                # Allow admins to view other users' sessions, but regular users can only view their own
                if not is_admin and session_user_id != user_id:
                    self.logger.error(
                        f"Session access denied | session_id={session_id} | "
                        f"session_user_id='{session_user_id}' | token_user_id='{user_id}' | "
                        f"session_user_id_len={len(session_user_id)} | token_user_id_len={len(user_id)} | "
                        f"session_user_id_bytes={session_user_id.encode('utf-8') if isinstance(session_user_id, str) else 'N/A'} | "
                        f"token_user_id_bytes={user_id.encode('utf-8') if isinstance(user_id, str) else 'N/A'}"
                    )
                    raise HTTPException(
                        status_code=403,
                        detail=f"Cannot access other users' sessions (session_user_id: {session_user_id}, token_user_id: {user_id})"
                    )
                
                if client_id and session.get("client_id") != client_id:
                    raise HTTPException(
                        status_code=403,
                        detail="Cannot access sessions from other clients"
                    )
                
                # Now safe to retrieve conversations
                 # Use session's user_id when admin is viewing another user's session
                conversations_user_id = session_user_id if is_admin and session_user_id != user_id else user_id
                conversations = await self.mongo_manager.get_session_conversations(conversations_user_id, session_id)
                if conversations is None:
                    raise HTTPException(status_code=500, detail="Failed to fetch conversations")
                return {
                    "session_id": session_id,
                    "conversations": conversations,
                    "is_own_session": session_user_id == user_id,
                    "datasource_context": session.get("datasource_context"),
                    "datasource_key": session.get("datasource_key"),
                }
            except HTTPException:
                raise  # Re-raise HTTPExceptions as-is (don't convert to 500)
            except Exception as e:
                self.logger.error(f"Error fetching session conversations: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to fetch conversations: {str(e)}")
        
        @self.router.delete("/sessions/{session_id}", tags=["Sessions"])
        async def delete_session(
            session_id: str = Path(..., min_length=1, max_length=100, regex=r'^[a-zA-Z0-9_-]+$'),
            current_user: dict = Depends(require_auth())
        ):
            """
            Delete a chat session and all its conversations.
            
            Security: 
            - Validates session ownership before deletion (prevents admins from deleting other users' sessions)
            - Validates client_id to prevent cross-client deletion
            - Path parameter validation prevents injection attacks
            """
            try:
                # SECURITY (Phase 2): Validate session_id format
                session_id = validate_id_parameter(session_id, "session_id")
                # Extract authenticated user info
                user_id = current_user.get("user_id") or current_user.get("_id")
                client_id = current_user.get("client_id")
                
                # Verify session exists and belongs to the same client
                session = await self.mongo_manager.get_session_metadata(session_id, user_id=user_id)
                if not session:
                    raise HTTPException(status_code=404, detail="Session not found")
                # Verify client_id matches (prevent cross-client deletion)
                session_client_id = session.get("client_id")
                if session_client_id and client_id and session_client_id != client_id:
                    raise HTTPException(
                        status_code=403,
                        detail="Cannot delete sessions from other clients"
                    )  
                # SECURITY: Validate ownership - only the session owner can delete
                session_user_id = session.get("user_id")
                if not session_user_id:
                    raise HTTPException(
                        status_code=404,
                        detail="Session metadata incomplete: missing user_id"
                    )
                
                # Convert both values to strings for comparison (handle ObjectId from both sources)
                from bson import ObjectId
                # Convert both to strings (handles ObjectId and string formats)
                session_user_id = str(session_user_id).strip()
                user_id = str(user_id).strip()
                
                # Ownership check - only the session owner can delete
                if session_user_id != user_id:
                    user_role = current_user.get("role", "user")
                    self.logger.warning(
                        f"User {current_user.get('email')} (role: {user_role}) "
                        f"attempted to delete session {session_id} owned by user_id {session_user_id} "
                        f"(requesting user_id: {user_id})"
                    )
                    raise HTTPException(
                        status_code=403,
                        detail="Cannot delete other users' sessions. Only the session owner can delete their own sessions."
                    )

                # Perform scoped deletion to current client's data (plus legacy records without client_id)
                deleted_count = await self.mongo_manager.delete_session_by_session_id_scoped(session_id, client_id)
                if deleted_count == 0:
                    # Idempotent behavior: do not leak whether other clients had data; treat as success
                    self.logger.info(
                        f"Delete requested for session {session_id}: no matching documents in client scope ({client_id})."
                    )
                else:
                    self.logger.info(
                        f"Successfully deleted {deleted_count} conversations for session {session_id} (client_id={client_id}, user_id={user_id})"
                    )

                return {"message": "Session deleted successfully", "session_id": session_id}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error deleting session: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")
        
        @self.router.patch("/sessions/{session_id}/title", tags=["Sessions"])
        async def update_session_title(
            request: UpdateSessionTitleRequest,
            session_id: str = Path(..., min_length=1, max_length=100, regex=r'^[a-zA-Z0-9_-]+$'),
            current_user: dict = Depends(require_auth())
        ):
            """
            Update the title of a chat session.
            
            Security: Requires authentication and ownership validation.
            Path parameter validation prevents injection attacks.
            Users can only update titles for their own sessions.
            """
            try:
                # SECURITY (Phase 2): Validate session_id format
                session_id = validate_id_parameter(session_id, "session_id")
                
                # Extract authenticated user info
                user_id = current_user.get("user_id") or current_user.get("_id")
                client_id = current_user.get("client_id")
                
                # Verify session ownership before allowing update
                session = await self.mongo_manager.get_session_metadata(session_id, user_id=user_id)
                
                if not session:
                    raise HTTPException(status_code=404, detail="Session not found")
                
                # Validate ownership
                session_user_id = session.get("user_id")
                if not session_user_id:
                    raise HTTPException(
                        status_code=404,
                        detail="Session metadata incomplete: missing user_id"
                    )
                
                # Convert both values to strings for comparison (handle ObjectId from both sources)
                from bson import ObjectId
                
                # Convert session_user_id to string
                if isinstance(session_user_id, ObjectId):
                    session_user_id = str(session_user_id)
                else:
                    session_user_id = str(session_user_id)
                
                # Convert token user_id to string (it might be ObjectId if decode_access_token converted it)
                if isinstance(user_id, ObjectId):
                    user_id = str(user_id)
                else:
                    user_id = str(user_id)
                
                # Normalize both values for comparison (strip whitespace)
                session_user_id = session_user_id.strip()
                user_id = user_id.strip()
                
                if session_user_id != user_id:
                    raise HTTPException(
                        status_code=403, 
                        detail="Cannot update title for other users' sessions"
                    )
                
                if client_id and session.get("client_id") != client_id:
                    raise HTTPException(
                        status_code=403, 
                        detail="Cannot update title for sessions from other clients"
                    )
                
                # Now safe to update (use authenticated user_id, not request body)
                success = await self.mongo_manager.update_session_title(
                    user_id,  # Use authenticated user_id
                    session_id, 
                    request.title
                )
                
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to update session title")
                
                return {"message": "Session title updated successfully"}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error updating session title: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to update title: {str(e)}")

        @self.router.post("/sessions/{session_id}/mark-read", tags=["Sessions"])
        async def mark_session_read(
            session_id: str = Path(..., min_length=1, max_length=100, regex=r'^[a-zA-Z0-9_-]+$'),
            current_user: dict = Depends(require_auth()),
        ):
            """Mark all conversations in a session as read. Called fire-and-forget from the frontend."""
            try:
                client_id = current_user.get("client_id")
                if not client_id:
                    raise HTTPException(status_code=401, detail="Authentication context incomplete")

                session_id = validate_id_parameter(session_id, "session_id")

                from db_config.mongo_server import get_db
                db = await get_db()
                conv_service = ConversationService(db)

                await conv_service.mark_session_read(session_id=session_id, client_id=client_id)
                return {"ok": True}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error marking session read for {session_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to mark session read")

        @self.router.get("/client-vocab", tags=["Client Configuration"])
        async def get_client_vocab(
            current_user: dict = Depends(require_auth())
        ):
            """
            Get client-specific vocabulary for speech recognition.
            
            Returns vocabulary list formatted for Speechmatics API based on
            client's guardrails configuration (facility names, product terms, etc.)
            """
            try:
                client_id = current_user.get("client_id")
                if not client_id:
                    # Return empty vocab if no client_id
                    return {"vocab": []}
                
                # Get guardrails from SchemaMapper
                from db_config.database import get_db
                from services.schema_mapper import SchemaMapper
                
                db = get_db()
                schema_mapper = SchemaMapper(client_id, db)
                guardrails = schema_mapper.get_guardrails_config()
                
                # Build vocab list from guardrails
                vocab = []
                
                # Add facility names
                facility_names = guardrails.get("facility_names", [])
                for facility in facility_names:
                    vocab.append({
                        "content": facility,
                        "sounds_like": [facility.lower()]
                    })
                
                # Add product terms
                product_terms = guardrails.get("product_terms", [])
                for product in product_terms:
                    vocab.append({
                        "content": product,
                        "sounds_like": [product.lower()]
                    })
                
                # Add domain keywords (as potential vocab)
                domain_keywords = guardrails.get("domain_keywords", [])
                for keyword in domain_keywords:
                    if keyword not in [v["content"] for v in vocab]:  # Avoid duplicates
                        vocab.append({
                            "content": keyword,
                            "sounds_like": [keyword.lower()]
                        })
                
                return {"vocab": vocab}

            except Exception as e:
                self.logger.error(f"Error loading client vocab: {e}")
                # Return empty vocab on error to prevent breaking speech recognition
                return {"vocab": []}

        # P1 (RLM): Internal callback endpoint for kernel-side llm_query() function.
        # No authentication — this is local-only, called from the Jupyter kernel subprocess.
        @self.router.post("/internal/llm-query", tags=["Internal"])
        async def internal_llm_query(request: Request):
            """
            Internal LLM sub-call endpoint for the llm_query() kernel helper.
            Enables generated code to call back to the LLM for semantic reasoning
            (text classification, entity extraction, answer verification) that
            pure pandas/numpy cannot handle. See RLM paper (MIT CSAIL, Dec 2025).
            """
            try:
                data = await request.json()
                question = str(data.get("question", "")).strip()
                context = str(data.get("context", ""))
                client_id = str(data.get("client_id", "default")).strip()

                if not question:
                    return {"answer": "", "error": "empty question"}

                from util.llm_utils import LLMClient
                llm = LLMClient(
                    agent_name="data_science_agent",
                    client_id=client_id,
                    db=self.mongo_manager.get_db()
                )
                system = (
                    "You are a precise analytical assistant embedded in a data science pipeline. "
                    "Answer the question directly based on the context. Be factual and brief — "
                    "no preamble, no explanation, just the answer."
                )
                user_msg = (
                    f"Context:\n{context}\n\nQuestion: {question}"
                    if context else question
                )
                response = await llm.generate_completion(
                    system_prompt=system,
                    user_message=user_msg,
                    temperature=0.1,
                    max_tokens=600
                )
                return {
                    "answer": (response.get("content") or "").strip(),
                    "error": response.get("error")
                }
            except Exception as e:
                self.logger.error(f"internal_llm_query error: {e}")
                return {"answer": "", "error": str(e)}

        # ------------------------------------------------------------------
        # Background Job Endpoints
        # ------------------------------------------------------------------

        @self.router.get("/jobs", tags=["Background Jobs"])
        async def list_jobs(
            status: Optional[str] = Query(None, description="Filter by status: pending, running, completed, error, cancelled"),
            limit: int = Query(20, ge=1, le=100),
            offset: int = Query(0, ge=0),
            current_user: dict = Depends(require_auth()),
        ):
            """List background conversations for the current user, optionally filtered by status."""
            try:
                client_id = current_user.get("client_id")
                user_id = current_user.get("user_id") or current_user.get("_id")
                if not client_id or not user_id:
                    raise HTTPException(status_code=401, detail="Authentication context incomplete")

                from db_config.mongo_server import get_db
                db = await get_db()
                conv_service = ConversationService(db)

                jobs = await conv_service.list_background(
                    client_id=client_id,
                    user_id=user_id,
                    status=status,
                    limit=limit,
                    offset=offset,
                )
                return {"jobs": jobs, "count": len(jobs)}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error listing background jobs: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to list background jobs")

        @self.router.get("/jobs/active", tags=["Background Jobs"])
        async def get_active_jobs(
            current_user: dict = Depends(require_auth()),
        ):
            """Lightweight poll endpoint for active (pending+running) background conversations. Designed for 15s polling."""
            try:
                client_id = current_user.get("client_id")
                user_id = current_user.get("user_id") or current_user.get("_id")
                if not client_id or not user_id:
                    raise HTTPException(status_code=401, detail="Authentication context incomplete")

                from db_config.mongo_server import get_db
                db = await get_db()
                conv_service = ConversationService(db)

                jobs = await conv_service.get_active_background(client_id=client_id, user_id=user_id)
                return {"jobs": jobs, "count": len(jobs)}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error fetching active jobs: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to fetch active jobs")

        @self.router.post("/jobs/send-to-background", tags=["Background Jobs"])
        async def send_to_background(
            request: Request,
            current_user: dict = Depends(require_auth()),
        ):
            """Signal a currently streaming query to move to background processing."""
            try:
                body = await request.json()
                run_id = body.get("run_id")
                if not run_id:
                    raise HTTPException(status_code=400, detail="run_id is required")

                run_id = validate_id_parameter(run_id, "run_id")

                from services.orchestrator_manager import trigger_background_signal
                success = trigger_background_signal(run_id)
                if not success:
                    raise HTTPException(status_code=404, detail="No active stream found for this run_id")

                return {"status": "signaled", "run_id": run_id}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error sending to background: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to send to background")

        @self.router.get("/jobs/{run_id}", tags=["Background Jobs"])
        async def get_job(
            run_id: str = Path(..., min_length=1, max_length=100),
            current_user: dict = Depends(require_auth()),
        ):
            """Get full detail for a background conversation including result (for completed ones)."""
            try:
                client_id = current_user.get("client_id")
                if not client_id:
                    raise HTTPException(status_code=401, detail="Authentication context incomplete")

                run_id = validate_id_parameter(run_id, "run_id")

                from db_config.mongo_server import get_db
                db = await get_db()
                conv_service = ConversationService(db)

                job = await conv_service.get_background_by_run_id(run_id=run_id, client_id=client_id)
                if not job:
                    raise HTTPException(status_code=404, detail="Background conversation not found")
                return job
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error fetching background conversation {run_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to fetch background conversation")

        @self.router.post("/jobs/{run_id}/cancel", tags=["Background Jobs"])
        async def cancel_job(
            run_id: str = Path(..., min_length=1, max_length=100),
            current_user: dict = Depends(require_auth()),
        ):
            """Cancel an active background conversation."""
            try:
                client_id = current_user.get("client_id")
                if not client_id:
                    raise HTTPException(status_code=401, detail="Authentication context incomplete")

                run_id = validate_id_parameter(run_id, "run_id")

                from db_config.mongo_server import get_db
                db = await get_db()
                conv_service = ConversationService(db)

                # Verify conversation exists and belongs to client
                job = await conv_service.get_background_by_run_id(run_id=run_id, client_id=client_id)
                if not job:
                    raise HTTPException(status_code=404, detail="Background conversation not found")

                if job["status"] not in ("pending", "running"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot cancel conversation in '{job['status']}' status"
                    )

                # Signal cancellation to the background consumer
                cancellation_manager.signal_job(run_id)

                # Update DB status
                cancelled = await conv_service.cancel_background(run_id=run_id, client_id=client_id)
                if not cancelled:
                    raise HTTPException(status_code=409, detail="Conversation already in terminal state")

                self.logger.info(f"Background conversation {run_id} cancelled by user {current_user.get('user_id')}")
                return {"run_id": run_id, "status": "cancelled"}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error cancelling background conversation {run_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to cancel background conversation")

        @self.router.post("/jobs/{run_id}/notification-read", tags=["Background Jobs"])
        async def mark_notification_read(
            run_id: str = Path(..., min_length=1, max_length=100),
            current_user: dict = Depends(require_auth()),
        ):
            """Mark a completed background conversation's notification as read (dismisses toast on frontend)."""
            try:
                client_id = current_user.get("client_id")
                if not client_id:
                    raise HTTPException(status_code=401, detail="Authentication context incomplete")

                run_id = validate_id_parameter(run_id, "run_id")

                from db_config.mongo_server import get_db
                db = await get_db()
                conv_service = ConversationService(db)

                updated = await conv_service.mark_notification_read(run_id=run_id, client_id=client_id)
                return {"run_id": run_id, "notification_read": updated}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error marking notification read for {run_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to mark notification read")


        # ══════════════════════════════════════════════════════════════════
        # AD-HOC FILE UPLOAD ENDPOINTS
        # ══════════════════════════════════════════════════════════════════

        @self.router.post("/adhoc/upload")
        async def adhoc_upload_file(
            file: UploadFile = File(...),
            session_id: str = Form(...),
            current_user: dict = Depends(require_auth()),
        ):
            """Upload a CSV/Excel file for ad-hoc analysis in a session."""
            try:
                client_id = current_user.get("client_id")
                if not client_id:
                    raise HTTPException(status_code=401, detail="Authentication context incomplete")

                from services.adhoc_file_service import upload_file as adhoc_upload, AdhocFileError

                content = await file.read()
                metadata = await adhoc_upload(
                    file_content=content,
                    original_filename=file.filename or "upload.csv",
                    session_id=session_id,
                    client_id=client_id,
                )
                return {
                    "filename": metadata["original_filename"],
                    "file_size": metadata["file_size_bytes"],
                    "file_names": metadata["file_names"],
                    "sheet_count": metadata["sheet_count"],
                    "session_id": session_id,
                }
            except AdhocFileError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Ad-hoc upload failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to upload file")

        @self.router.delete("/adhoc/file")
        async def adhoc_delete_file(
            session_id: str = Query(...),
            current_user: dict = Depends(require_auth()),
        ):
            """Delete the ad-hoc file for a session."""
            try:
                client_id = current_user.get("client_id")
                if not client_id:
                    raise HTTPException(status_code=401, detail="Authentication context incomplete")

                from services.adhoc_file_service import delete_file as adhoc_delete

                deleted = await adhoc_delete(session_id=session_id, client_id=client_id)
                return {"success": deleted, "session_id": session_id}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Ad-hoc delete failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to delete file")

        @self.router.get("/adhoc/file-info")
        async def adhoc_file_info(
            session_id: str = Query(...),
            current_user: dict = Depends(require_auth()),
        ):
            """Get ad-hoc file metadata for a session (or null)."""
            try:
                client_id = current_user.get("client_id")
                if not client_id:
                    raise HTTPException(status_code=401, detail="Authentication context incomplete")

                from services.adhoc_file_service import get_file_metadata

                metadata = get_file_metadata(session_id)
                if metadata and metadata.get("client_id") != client_id:
                    return {"file": None}
                return {"file": metadata}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Ad-hoc file-info failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to get file info")


agents_router = AgentsRouter().router
