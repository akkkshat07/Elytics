from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ConversationStatus(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    ERROR = 'error'
    CANCELLED = 'cancelled'
ACTIVE_STATUSES = {ConversationStatus.PENDING, ConversationStatus.RUNNING}
TERMINAL_STATUSES = {ConversationStatus.COMPLETED, ConversationStatus.ERROR, ConversationStatus.CANCELLED}

class ConversationFeedbackRating(str, Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'

class ConversationAgentResponses(BaseModel):
    planner: Optional[Any] = None
    python: Optional[str] = None
    executor: Optional[Any] = None
    business: Optional[Any] = None

class ConversationMetadata(BaseModel):
    attempts: int = 0
    execution_attempts: int = 0
    executor_response_truncated: bool = False
    code_truncated: bool = False
    cache_hit: bool = False
    custom_title: Optional[str] = None
    discussion_executor_response_truncated: Optional[bool] = None
    executor_response_removed: Optional[bool] = None
    business_response_truncated: Optional[bool] = None
    original_size_mb: Optional[float] = None
    emergency_truncation: Optional[bool] = None

    class Config:
        extra = 'allow'

class ConversationTiming(BaseModel):
    total_execution_time_seconds: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class ConversationLLMConfig(BaseModel):
    config_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None

class ConversationTokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_tokens_provider: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    estimated: Optional[bool] = None
    missing_usage_recovered: Optional[str] = None
    reasoning_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    audio_input_tokens: Optional[int] = None
    audio_output_tokens: Optional[int] = None
    image_input_tokens: Optional[int] = None
    accepted_prediction_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None
    text_input_tokens: Optional[int] = None
    text_output_tokens: Optional[int] = None

    class Config:
        extra = 'allow'

class ConversationTotalTokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_tokens_provider: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    audio_input_tokens: Optional[int] = None
    audio_output_tokens: Optional[int] = None
    image_input_tokens: Optional[int] = None
    accepted_prediction_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None
    text_input_tokens: Optional[int] = None
    text_output_tokens: Optional[int] = None

    class Config:
        extra = 'allow'

class ConversationEstimatedCost(BaseModel):
    total_cost_usd: float = 0.0
    per_agent_cost: Dict[str, float] = Field(default_factory=dict)

class ConversationPhase(str, Enum):
    ROUTER = 'router'
    SCOUT = 'scout'
    CODER = 'coder'
    NARRATOR = 'narrator'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'

class ConversationProgress(BaseModel):
    current_phase: ConversationPhase = ConversationPhase.ROUTER
    message: str = 'Queued'
    iteration: int = 0
    max_iterations: int = 0

class ConversationFeedback(BaseModel):
    rating: ConversationFeedbackRating
    comment: str = ''
    user_id: str
    created_at: datetime

class Conversation(BaseModel):
    run_id: str
    user_id: str
    session_id: str
    client_id: str
    input: str
    status: ConversationStatus = ConversationStatus.PENDING
    route_decision: str = ''
    created_at: datetime
    agent_responses: ConversationAgentResponses = Field(default_factory=ConversationAgentResponses)
    metadata: ConversationMetadata = Field(default_factory=ConversationMetadata)
    timing: ConversationTiming = Field(default_factory=ConversationTiming)
    llm_config: ConversationLLMConfig = Field(default_factory=ConversationLLMConfig)
    model: Optional[str] = None
    enhanced_question: Optional[str] = None
    semantic_signature: Optional[Any] = None
    agent_inputs: Dict[str, Any] = Field(default_factory=dict)
    agent_token_usage: Dict[str, ConversationTokenUsage] = Field(default_factory=dict)
    total_token_usage: ConversationTotalTokenUsage = Field(default_factory=ConversationTotalTokenUsage)
    estimated_cost: ConversationEstimatedCost = Field(default_factory=ConversationEstimatedCost)
    error: Optional[str] = None
    is_background: bool = False
    progress: Optional[ConversationProgress] = None
    estimated_duration_seconds: Optional[int] = None
    notification_read: bool = False
    is_read: bool = True
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    feedback: Optional[ConversationFeedback] = None
    active_persona: Optional[Dict[str, Any]] = None
    is_deleted: Optional[bool] = None
    deleted_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        extra = 'allow'
        populate_by_name = True