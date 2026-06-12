import asyncio
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Set
from util.time_utils import utcnow
logger = logging.getLogger(__name__)

@dataclass
class LLMCallEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: str = ''
    session_id: str = ''
    run_id: str = ''
    client_id: str = ''
    user_id: str = ''
    agent: str = ''
    provider: str = ''
    model: str = ''
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    success: bool = True
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    is_load_test: bool = False
    source: str = 'ui'
    load_test_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

@dataclass
class LiveSession:
    session_id: str
    run_id: str
    client_id: str
    user_id: str
    source: str
    load_test_id: Optional[str]
    started_at: str
    last_event_at: str
    current_agent: str
    provider: str
    model: str
    call_count: int = 0
    total_latency_ms: int = 0
    error_count: int = 0
    status: str = 'active'
    finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class LLMMetricsService:
    RING_BUFFER_SIZE = 2000
    FLUSH_INTERVAL_SECONDS = 5
    SESSION_DONE_DISMISS_SECONDS = 30

    def __init__(self):
        self._ring: deque = deque(maxlen=self.RING_BUFFER_SIZE)
        self._pending_flush: List[LLMCallEvent] = []
        self._subscribers: Set[asyncio.Queue] = set()
        self._live_sessions: Dict[str, LiveSession] = {}
        self._flush_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._mongo: Any = None

    def start(self, mongo_manager: Any):
        self._mongo = mongo_manager
        self._flush_task = asyncio.create_task(self._flush_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info('LLMMetricsService started — flush every %ds', self.FLUSH_INTERVAL_SECONDS)

    async def stop(self):
        if self._flush_task:
            self._flush_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        await self._flush_to_mongo()

    async def emit(self, event: LLMCallEvent):
        if not event.ts:
            event.ts = utcnow().isoformat()
        self._ring.append(event)
        self._pending_flush.append(event)
        self._update_live_session(event)
        await self._broadcast(event)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers.add(q)
        logger.debug('SSE subscriber added — total: %d', len(self._subscribers))
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers.discard(q)
        logger.debug('SSE subscriber removed — total: %d', len(self._subscribers))

    async def sse_stream(self) -> AsyncGenerator[str, None]:
        q = self.subscribe()
        snapshot = {'type': 'snapshot', 'sessions': [s.to_dict() for s in self._live_sessions.values()]}
        yield f'data: {json.dumps(snapshot, default=str)}\n\n'
        try:
            while True:
                try:
                    event: LLMCallEvent = await asyncio.wait_for(q.get(), timeout=20.0)
                    payload = {'type': 'llm_call', 'event': event.to_dict()}
                    yield f'data: {json.dumps(payload, default=str)}\n\n'
                    sess = self._live_sessions.get(event.session_id)
                    if sess:
                        sess_payload = {'type': 'session_update', 'session': sess.to_dict()}
                        yield f'data: {json.dumps(sess_payload, default=str)}\n\n'
                except asyncio.TimeoutError:
                    yield ': heartbeat\n\n'
        except asyncio.CancelledError:
            pass
        finally:
            self.unsubscribe(q)

    async def _broadcast(self, event: LLMCallEvent):
        dead: Set[asyncio.Queue] = set()
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    dead.add(q)
        for q in dead:
            self._subscribers.discard(q)

    def _update_live_session(self, event: LLMCallEvent):
        now = utcnow().isoformat()
        sess = self._live_sessions.get(event.session_id)
        if sess is None:
            sess = LiveSession(session_id=event.session_id, run_id=event.run_id, client_id=event.client_id, user_id=event.user_id, source=event.source, load_test_id=event.load_test_id, started_at=now, last_event_at=now, current_agent=event.agent, provider=event.provider, model=event.model)
            self._live_sessions[event.session_id] = sess
        sess.last_event_at = now
        sess.current_agent = event.agent
        sess.provider = event.provider
        sess.model = event.model
        sess.call_count += 1
        sess.total_latency_ms += event.latency_ms
        if not event.success:
            sess.error_count += 1
        if event.agent in ('narrator', 'business') and event.success:
            sess.status = 'done'
            sess.finished_at = now

    def mark_session_done(self, session_id: str):
        sess = self._live_sessions.get(session_id)
        if sess and sess.status == 'active':
            sess.status = 'done'
            sess.finished_at = utcnow().isoformat()

    def get_live_sessions(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._live_sessions.values()]

    def get_ring_buffer_summary(self) -> Dict[str, Any]:
        events = list(self._ring)
        if not events:
            return {'total_calls': 0, 'providers': {}}
        by_provider: Dict[str, Dict] = {}
        for e in events:
            key = f'{e.provider}/{e.model}'
            if key not in by_provider:
                by_provider[key] = {'provider': e.provider, 'model': e.model, 'calls': 0, 'errors': 0, 'latencies': [], 'prompt_tokens': 0, 'completion_tokens': 0}
            p = by_provider[key]
            p['calls'] += 1
            if not e.success:
                p['errors'] += 1
            if e.latency_ms:
                p['latencies'].append(e.latency_ms)
            p['prompt_tokens'] += e.prompt_tokens
            p['completion_tokens'] += e.completion_tokens
        result_providers = []
        for p in by_provider.values():
            lats = sorted(p.pop('latencies'))
            n = len(lats)
            p['p50_ms'] = lats[int(n * 0.5)] if lats else 0
            p['p95_ms'] = lats[int(n * 0.95)] if lats else 0
            p['p99_ms'] = lats[int(n * 0.99)] if lats else 0
            p['error_rate'] = round(p['errors'] / p['calls'], 4) if p['calls'] else 0
            result_providers.append(p)
        return {'total_calls': len(events), 'providers': result_providers, 'live_sessions': len([s for s in self._live_sessions.values() if s.status == 'active'])}

    async def _flush_loop(self):
        while True:
            await asyncio.sleep(self.FLUSH_INTERVAL_SECONDS)
            await self._flush_to_mongo()

    async def _flush_to_mongo(self):
        if not self._pending_flush or not self._mongo:
            return
        batch = self._pending_flush[:]
        self._pending_flush.clear()
        try:
            docs = [e.to_dict() for e in batch]
            db = self._mongo.db
            await db.llm_call_metrics.insert_many(docs, ordered=False)
            logger.debug('Flushed %d LLM metric events to MongoDB', len(docs))
        except Exception as exc:
            logger.error('LLM metrics MongoDB flush failed: %s', exc)
            self._pending_flush = batch[-500:] + self._pending_flush

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(10)
            now = time.time()
            to_remove = []
            for sid, sess in self._live_sessions.items():
                if sess.status == 'done' and sess.finished_at:
                    try:
                        from datetime import datetime, timezone
                        finished = datetime.fromisoformat(sess.finished_at.replace('Z', '+00:00'))
                        age = (datetime.now(timezone.utc) - finished).total_seconds()
                        if age >= self.SESSION_DONE_DISMISS_SECONDS:
                            to_remove.append(sid)
                    except Exception:
                        pass
            for sid in to_remove:
                del self._live_sessions[sid]
                removal = {'type': 'session_removed', 'session_id': sid}
                for q in list(self._subscribers):
                    try:
                        q.put_nowait(removal)
                    except Exception:
                        pass
llm_metrics_service = LLMMetricsService()