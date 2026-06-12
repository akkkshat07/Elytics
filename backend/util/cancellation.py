import asyncio
import logging
from typing import Dict, Union
logger = logging.getLogger(__name__)

class CancellationManager:

    def __init__(self):
        self._events: Dict[str, asyncio.Event] = {}

    def register(self, session_id: str) -> asyncio.Event:
        if session_id in self._events:
            logger.warning(f'Session {session_id} is already registered. Overwriting event.')
        event = asyncio.Event()
        self._events[session_id] = event
        logger.info(f'Registered cancellation event for session: {session_id}')
        return event

    def signal(self, session_id: str) -> bool:
        if session_id in self._events:
            self._events[session_id].set()
            logger.info(f'Signaled cancellation for session: {session_id}')
            return True
        logger.warning(f'Attempted to signal non-existent session: {session_id}')
        return False

    def is_cancelled(self, session_id: Union[str, None]) -> bool:
        if not session_id:
            return False
        event = self._events.get(session_id)
        return event.is_set() if event else False

    def cleanup(self, session_id: str):
        if session_id in self._events:
            del self._events[session_id]
            logger.info(f'Cleaned up cancellation event for session: {session_id}')

    def register_job(self, job_id: str) -> asyncio.Event:
        return self.register(job_id)

    def signal_job(self, job_id: str) -> bool:
        return self.signal(job_id)

    def cleanup_job(self, job_id: str):
        self.cleanup(job_id)
cancellation_manager = CancellationManager()