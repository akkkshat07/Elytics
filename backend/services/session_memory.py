from __future__ import annotations
from typing import Optional, Dict, Any
from threading import RLock
import json
from config.system_config import REDIS_URL
try:
    import redis
    _redis_client = redis.Redis.from_url(REDIS_URL) if REDIS_URL else None
    if _redis_client is not None:
        _redis_client.ping()
except Exception:
    _redis_client = None

class SessionMemory:

    def __init__(self) -> None:
        self._lock = RLock()
        self._last_context_by_session: Dict[str, Dict[str, Any]] = {}

    def _redis_key(self, session_id: str) -> str:
        return f'session:{session_id}:last_context'

    def _redis_count_key(self, session_id: str) -> str:
        return f'session:{session_id}:follow_up_count'

    def get_last_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        if _redis_client is not None:
            try:
                data = _redis_client.get(self._redis_key(session_id))
                if data:
                    return json.loads(data)
            except Exception:
                pass
        with self._lock:
            return self._last_context_by_session.get(session_id)

    def get_follow_up_count(self, session_id: str) -> int:
        if _redis_client is not None:
            try:
                count = _redis_client.get(self._redis_count_key(session_id))
                return int(count) if count else 0
            except Exception:
                pass
        with self._lock:
            ctx = self._last_context_by_session.get(session_id)
            return ctx.get('follow_up_count', 0) if ctx else 0

    def increment_follow_up_count(self, session_id: str) -> int:
        if _redis_client is not None:
            try:
                count = _redis_client.incr(self._redis_count_key(session_id))
                _redis_client.expire(self._redis_count_key(session_id), 604800)
                return int(count)
            except Exception:
                pass
        with self._lock:
            if session_id not in self._last_context_by_session:
                self._last_context_by_session[session_id] = {'follow_up_count': 0}
            self._last_context_by_session[session_id]['follow_up_count'] = self._last_context_by_session[session_id].get('follow_up_count', 0) + 1
            return self._last_context_by_session[session_id]['follow_up_count']

    def reset_follow_up_count(self, session_id: str) -> None:
        if _redis_client is not None:
            try:
                _redis_client.set(self._redis_count_key(session_id), '0', ex=604800)
                return
            except Exception:
                pass
        with self._lock:
            if session_id in self._last_context_by_session:
                self._last_context_by_session[session_id]['follow_up_count'] = 0

    def update_last_context(self, session_id: str, previous_user_question: str, previous_enhanced_instruction: str, is_follow_up: bool=False, enhanced_question: str='') -> None:
        import logging
        logger = logging.getLogger(__name__)
        if is_follow_up:
            follow_up_count = self.increment_follow_up_count(session_id)
            logger.info(f'Session {session_id}: Follow-up detected, count now {follow_up_count}')
        else:
            self.reset_follow_up_count(session_id)
            follow_up_count = 0
            logger.info(f'Session {session_id}: New question (not follow-up), reset count to 0')
        payload = {'previous_user_question': previous_user_question, 'previous_enhanced_instruction': previous_enhanced_instruction, 'previous_plan': previous_enhanced_instruction, 'previous_enhanced_question': enhanced_question or previous_user_question, 'follow_up_count': follow_up_count}
        if _redis_client is not None:
            try:
                _redis_client.set(self._redis_key(session_id), json.dumps(payload), ex=604800)
                return
            except Exception:
                pass
        with self._lock:
            if session_id not in self._last_context_by_session:
                self._last_context_by_session[session_id] = {}
            self._last_context_by_session[session_id].update(payload)

    def reset_context(self, session_id: str) -> None:
        if _redis_client is not None:
            try:
                _redis_client.delete(self._redis_key(session_id))
                _redis_client.delete(self._redis_count_key(session_id))
                return
            except Exception:
                pass
        with self._lock:
            if session_id in self._last_context_by_session:
                del self._last_context_by_session[session_id]

    def _redis_adhoc_key(self, session_id: str) -> str:
        return f'session:{session_id}:adhoc_file'

    def set_adhoc_file(self, session_id: str, metadata: Dict[str, Any]) -> None:
        if _redis_client is not None:
            try:
                _redis_client.set(self._redis_adhoc_key(session_id), json.dumps(metadata), ex=86400)
                return
            except Exception:
                pass
        with self._lock:
            ctx = self._last_context_by_session.setdefault(session_id, {})
            ctx['adhoc_file'] = metadata

    def get_adhoc_file(self, session_id: str) -> Optional[Dict[str, Any]]:
        if _redis_client is not None:
            try:
                data = _redis_client.get(self._redis_adhoc_key(session_id))
                if data:
                    return json.loads(data)
            except Exception:
                pass
        with self._lock:
            ctx = self._last_context_by_session.get(session_id)
            return ctx.get('adhoc_file') if ctx else None

    def clear_adhoc_file(self, session_id: str) -> None:
        if _redis_client is not None:
            try:
                _redis_client.delete(self._redis_adhoc_key(session_id))
                return
            except Exception:
                pass
        with self._lock:
            ctx = self._last_context_by_session.get(session_id)
            if ctx and 'adhoc_file' in ctx:
                del ctx['adhoc_file']

    def _redis_persona_key(self, session_id: str) -> str:
        return f'session:{session_id}:persona'

    def set_persona(self, session_id: str, persona: Optional[Dict[str, Any]]) -> bool:
        _SENTINEL = '__not_set__'
        if _redis_client is not None:
            try:
                key = self._redis_persona_key(session_id)
                import json as _json
                value = _json.dumps(persona)
                was_set = _redis_client.set(key, value, nx=True)
                return bool(was_set)
            except Exception:
                pass
        with self._lock:
            ctx = self._last_context_by_session.setdefault(session_id, {})
            if _SENTINEL not in ctx and 'persona' not in ctx:
                ctx['persona'] = persona
                return True
            return False

    def get_persona(self, session_id: str) -> Optional[Dict[str, Any]]:
        if _redis_client is not None:
            try:
                import json as _json
                data = _redis_client.get(self._redis_persona_key(session_id))
                if data is not None:
                    return _json.loads(data)
            except Exception:
                pass
        with self._lock:
            ctx = self._last_context_by_session.get(session_id)
            if ctx and 'persona' in ctx:
                return ctx['persona']
        return None
session_memory = SessionMemory()