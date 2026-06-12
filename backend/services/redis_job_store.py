import json
import logging
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
_BG_JOB_TTL = 86400
_BG_ACTIVE_TTL = 86400
_client = None

def _get_client():
    global _client
    if _client is None:
        try:
            from config.system_config import REDIS_URL
            if REDIS_URL:
                import redis.asyncio as aioredis
                _client = aioredis.from_url(REDIS_URL, decode_responses=True)
        except Exception as e:
            logger.debug('redis_job_store: async client unavailable: %s', e)
    return _client

def _job_key(run_id: str) -> str:
    return f'bg_job:{run_id}'

def _active_key(client_id: str, user_id: str) -> str:
    return f'bg_active:{client_id}:{user_id}'

async def store_active_job(*, run_id: str, client_id: str, user_id: str, session_id: str, input_text: str, route_decision: str, estimated_duration_seconds: int, created_at: str, started_at: str) -> None:
    redis = _get_client()
    if redis is None:
        return
    try:
        job_data = {'run_id': run_id, 'client_id': client_id, 'user_id': user_id, 'session_id': session_id, 'input': input_text[:100], 'route_decision': route_decision, 'status': 'running', 'estimated_duration_seconds': str(estimated_duration_seconds), 'created_at': created_at, 'started_at': started_at, 'progress': json.dumps({'current_phase': 'router', 'message': 'Running in background', 'iteration': 0, 'max_iterations': 0})}
        pipe = redis.pipeline()
        pipe.hset(_job_key(run_id), mapping=job_data)
        pipe.expire(_job_key(run_id), _BG_JOB_TTL)
        pipe.sadd(_active_key(client_id, user_id), run_id)
        pipe.expire(_active_key(client_id, user_id), _BG_ACTIVE_TTL)
        await pipe.execute()
        logger.debug('redis_job_store: stored active job run_id=%s', run_id)
    except Exception as e:
        logger.warning('redis_job_store.store_active_job failed run_id=%s: %s', run_id, e)

async def update_job_progress(*, run_id: str, current_phase: str, message: str, iteration: int=0, max_iterations: int=0) -> None:
    redis = _get_client()
    if redis is None:
        return
    try:
        progress = json.dumps({'current_phase': current_phase, 'message': message, 'iteration': iteration, 'max_iterations': max_iterations})
        await redis.hset(_job_key(run_id), 'progress', progress)
    except Exception as e:
        logger.debug('redis_job_store.update_job_progress failed run_id=%s: %s', run_id, e)

async def mark_job_terminal(*, run_id: str, client_id: str, user_id: str) -> None:
    redis = _get_client()
    if redis is None:
        return
    try:
        await redis.srem(_active_key(client_id, user_id), run_id)
        logger.debug('redis_job_store: removed active job run_id=%s', run_id)
    except Exception as e:
        logger.warning('redis_job_store.mark_job_terminal failed run_id=%s: %s', run_id, e)

async def get_active_jobs(client_id: str, user_id: str) -> Optional[List[Dict[str, Any]]]:
    redis = _get_client()
    if redis is None:
        return None
    try:
        run_ids = await redis.smembers(_active_key(client_id, user_id))
        if not run_ids:
            return []
        pipe = redis.pipeline()
        for run_id in run_ids:
            pipe.hgetall(_job_key(run_id))
        results = await pipe.execute()
        jobs: List[Dict[str, Any]] = []
        stale: List[str] = []
        for run_id, doc in zip(run_ids, results):
            if not doc:
                stale.append(run_id)
                continue
            if doc.get('progress'):
                try:
                    doc['progress'] = json.loads(doc['progress'])
                except Exception:
                    pass
            if doc.get('estimated_duration_seconds'):
                try:
                    doc['estimated_duration_seconds'] = int(doc['estimated_duration_seconds'])
                except Exception:
                    pass
            jobs.append(doc)
        if stale:
            try:
                await redis.srem(_active_key(client_id, user_id), *stale)
            except Exception:
                pass
        return jobs
    except Exception as e:
        logger.warning('redis_job_store.get_active_jobs failed client=%s: %s', client_id, e)
        return None