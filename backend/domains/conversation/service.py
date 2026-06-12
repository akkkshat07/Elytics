import logging
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from domains.conversation.model import Conversation, ConversationPhase, ConversationStatus
from domains.conversation.repository import ConversationRepository
logger = logging.getLogger(__name__)

class BackgroundLimitError(Exception):

    def __init__(self, client_id: str, current_count: int, max_count: int):
        self.client_id = client_id
        self.current_count = current_count
        self.max_count = max_count
        super().__init__(f'Client {client_id} has {current_count}/{max_count} active background conversations. Wait for one to complete or cancel one.')

class ConversationService:

    def __init__(self, db: AsyncIOMotorDatabase):
        self.repo = ConversationRepository(db)

    async def create_pending(self, *, run_id: str, user_id: str, client_id: str, session_id: str, input_text: str, route_decision: str='') -> Optional[str]:
        return await self.repo.insert_pending(run_id=run_id, user_id=user_id, client_id=client_id, session_id=session_id, input_text=input_text, route_decision=route_decision)

    async def get_by_run_id(self, run_id: str) -> Optional[Conversation]:
        return await self.repo.find_by_run_id(run_id)

    async def get_by_id(self, conversation_id: str, client_id: str) -> Optional[Conversation]:
        return await self.repo.find_by_id(conversation_id, client_id)

    async def list_sessions(self, *, client_id: str, user_id: Optional[str]=None, page: int=1, limit: int=20) -> Dict[str, Any]:
        return await self.repo.list_sessions(client_id=client_id, user_id=user_id, page=page, limit=limit)

    async def get_session_conversations(self, session_id: str, user_id: str) -> list[Conversation]:
        return await self.repo.find_by_session(session_id, user_id)

    async def complete(self, conversation_id: str, client_id: str, update_data: Dict[str, Any]) -> bool:
        update_data['status'] = ConversationStatus.COMPLETED.value
        return await self.repo.update_by_id(conversation_id, client_id, update_data)

    async def fail(self, conversation_id: str, client_id: str, error: str) -> bool:
        return await self.repo.update_by_id(conversation_id, client_id, {'status': ConversationStatus.ERROR.value, 'error': error[:2000]})

    async def cancel(self, conversation_id: str, client_id: str) -> bool:
        return await self.repo.update_status(conversation_id, client_id, ConversationStatus.CANCELLED)

    async def save_feedback(self, run_id: str, rating: str, comment: str, user_id: str) -> bool:
        return await self.repo.save_feedback(run_id, rating, comment, user_id)

    async def rename_session(self, user_id: str, session_id: str, title: str) -> bool:
        return await self.repo.update_session_title(user_id, session_id, title)

    async def delete_session(self, session_id: str, client_id: str) -> int:
        return await self.repo.soft_delete_session(session_id, client_id)

    async def transition_to_background(self, *, run_id: str, client_id: str, estimated_duration_seconds: int=300, max_concurrent: int=2) -> bool:
        active_count = await self.repo.count_active_background(client_id)
        if active_count >= max_concurrent:
            raise BackgroundLimitError(client_id, active_count, max_concurrent)
        return await self.repo.mark_background_running(run_id, client_id, estimated_duration_seconds=estimated_duration_seconds)

    async def update_background_progress(self, run_id: str, client_id: str, *, current_phase: ConversationPhase, message: str='', iteration: int=0, max_iterations: int=0) -> bool:
        return await self.repo.update_progress(run_id, client_id, current_phase=current_phase, message=message, iteration=iteration, max_iterations=max_iterations)

    async def complete_background(self, run_id: str, client_id: str) -> bool:
        return await self.repo.complete_background(run_id, client_id)

    async def fail_background(self, run_id: str, client_id: str, error: str) -> bool:
        return await self.repo.fail_background(run_id, client_id, error)

    async def cancel_background(self, run_id: str, client_id: str) -> bool:
        return await self.repo.cancel_background(run_id, client_id)

    async def get_active_background(self, client_id: str, user_id: str) -> List[Dict[str, Any]]:
        return await self.repo.get_active_background(client_id, user_id)

    async def list_background(self, client_id: str, user_id: Optional[str]=None, status: Optional[str]=None, limit: int=20, offset: int=0) -> List[Dict[str, Any]]:
        return await self.repo.list_background(client_id, user_id, status, limit, offset)

    async def get_background_by_run_id(self, run_id: str, client_id: str) -> Optional[Dict[str, Any]]:
        return await self.repo.find_background_by_run_id(run_id, client_id)

    async def mark_notification_read(self, run_id: str, client_id: str) -> bool:
        return await self.repo.mark_notification_read(run_id, client_id)

    async def mark_session_read(self, session_id: str, client_id: str) -> bool:
        return await self.repo.mark_session_read(session_id, client_id)

    async def mark_stale_background_failed(self, stale_threshold_minutes: int=30) -> int:
        return await self.repo.mark_stale_background_failed(stale_threshold_minutes)

    async def get_execution_stats(self, *, client_id: str, user_id: Optional[str]=None, days: int=30) -> Dict[str, Any]:
        return await self.repo.get_execution_stats(client_id=client_id, user_id=user_id, days=days)