from __future__ import annotations
import time
import logging
from uuid import uuid4
from typing import Optional, Dict, Tuple
from fastapi import Request, status
from fastapi.responses import JSONResponse
from redis.exceptions import NoScriptError
import redis.asyncio as redis
import os
logger = logging.getLogger(__name__)
_DEFAULT_RATE_LIMITED_PREFIXES: set = {'/api/agents/agent_query_stream'}
_rate_limited_prefixes_cache: Optional[set] = None
_rate_limited_prefixes_cache_ts: float = 0.0
_RATE_LIMITED_PREFIXES_CACHE_TTL = 60

def invalidate_relaxed_prefixes_cache() -> None:
    global _rate_limited_prefixes_cache, _rate_limited_prefixes_cache_ts
    _rate_limited_prefixes_cache = None
    _rate_limited_prefixes_cache_ts = 0.0

async def _get_rate_limited_prefixes() -> set:
    global _rate_limited_prefixes_cache, _rate_limited_prefixes_cache_ts
    now = time.time()
    if _rate_limited_prefixes_cache is not None and now - _rate_limited_prefixes_cache_ts < _RATE_LIMITED_PREFIXES_CACHE_TTL:
        return _rate_limited_prefixes_cache
    try:
        from db_config.mongo_server import get_db
        db = await get_db()
        config = await db.rate_limit_config.find_one({'config_type': 'rate_limited_prefixes'})
        if config and 'prefixes' in config:
            _rate_limited_prefixes_cache = set(config['prefixes'])
        else:
            _rate_limited_prefixes_cache = _DEFAULT_RATE_LIMITED_PREFIXES
        _rate_limited_prefixes_cache_ts = now
        return _rate_limited_prefixes_cache
    except Exception as e:
        logger.warning(f'Failed to load rate-limited prefixes from DB: {e}. Using defaults.')
        if _rate_limited_prefixes_cache is not None:
            return _rate_limited_prefixes_cache
        return _DEFAULT_RATE_LIMITED_PREFIXES
RATE_LIMIT_LUA = "\nlocal key = KEYS[1]\nlocal window_start = tonumber(ARGV[1])\nlocal current_time = tonumber(ARGV[2])\nlocal max_requests = tonumber(ARGV[3])\nlocal member       = ARGV[4]\nlocal expire_secs  = tonumber(ARGV[5])\n\n-- Remove expired entries\nredis.call('ZREMRANGEBYSCORE', key, 0, window_start)\n\n-- Count current entries\nlocal count = redis.call('ZCARD', key)\n\n-- Only add if under the limit\nlocal added = 0\nif count < max_requests then\n    redis.call('ZADD', key, current_time, member)\n    added = 1\nend\n\n-- Refresh TTL\nredis.call('EXPIRE', key, expire_secs)\n\nreturn {count, added}\n"

class RateLimiter:
    _CB_INITIAL_COOLDOWN = 30
    _CB_MAX_COOLDOWN = 300
    _CB_BACKOFF_FACTOR = 2

    def __init__(self):
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self._redis_error_logged = False
        self._cb_open = False
        self._cb_last_failure = 0.0
        self._cb_cooldown = self._CB_INITIAL_COOLDOWN
        try:
            self.redis = redis.from_url(redis_url, encoding='utf-8', decode_responses=True, socket_connect_timeout=0.5, socket_timeout=0.5)
            self.enabled = True
            self._lua_sha: Optional[str] = None
            logger.info(f'Rate limiter initialized with Redis at {redis_url}')
        except Exception as e:
            logger.warning(f'Rate limiter Redis connection failed: {e}. Rate limiting disabled.')
            self.redis = None
            self.enabled = False
            self._cb_open = True
            self._cb_last_failure = time.time()
        self.default_limits = {'client': {'window': 60, 'max_requests': 200}, 'user': {'window': 60, 'max_requests': 30}, 'endpoint': {'window': 60, 'max_requests': 15}}

    def _is_circuit_open(self) -> bool:
        if not self._cb_open:
            return False
        elapsed = time.time() - self._cb_last_failure
        if elapsed >= self._cb_cooldown:
            return False
        return True

    def _record_failure(self):
        self._cb_open = True
        self._cb_last_failure = time.time()
        self._cb_cooldown = min(self._cb_cooldown * self._CB_BACKOFF_FACTOR, self._CB_MAX_COOLDOWN)
        if not self._redis_error_logged:
            logger.error(f'Redis unavailable — rate limiting disabled for {self._cb_cooldown}s (circuit breaker open)')
            self._redis_error_logged = True
        else:
            logger.warning(f'Redis still unavailable — cooldown extended to {self._cb_cooldown}s')

    def _record_success(self):
        if self._cb_open:
            logger.info('Redis recovered — rate limiting re-enabled (circuit breaker closed)')
        self._cb_open = False
        self._cb_cooldown = self._CB_INITIAL_COOLDOWN
        self._redis_error_logged = False

    async def _ensure_lua_loaded(self):
        if self._lua_sha is None:
            self._lua_sha = await self.redis.script_load(RATE_LIMIT_LUA)

    async def check_rate_limit(self, key: str, window: int, max_requests: int) -> Tuple[bool, Dict[str, int]]:
        if not self.enabled or not self.redis or self._is_circuit_open():
            return (True, {'limit': max_requests, 'remaining': max_requests, 'reset': int(time.time()) + window})
        try:
            current_time = time.time()
            window_start = current_time - window
            member = f'{current_time}:{uuid4().hex[:8]}'
            await self._ensure_lua_loaded()
            result = await self.redis.evalsha(self._lua_sha, 1, key, str(window_start), str(current_time), str(max_requests), member, str(window + 10))
            current_count = int(result[0])
            was_added = int(result[1])
            is_allowed = was_added == 1
            remaining = max(0, max_requests - current_count - (1 if is_allowed else 0))
            reset_time = int(current_time + window)
            self._record_success()
            return (is_allowed, {'limit': max_requests, 'remaining': remaining, 'reset': reset_time})
        except NoScriptError:
            self._lua_sha = None
            try:
                await self._ensure_lua_loaded()
                return await self.check_rate_limit(key, window, max_requests)
            except Exception:
                self._record_failure()
                return (True, {'limit': max_requests, 'remaining': max_requests, 'reset': int(time.time()) + window})
        except Exception as e:
            self._record_failure()
            return (True, {'limit': max_requests, 'remaining': max_requests, 'reset': int(time.time()) + window})

    async def check_all_limits(self, client_id: str, user_id: str, endpoint: str, rate_limits: Optional[Dict[str, int]]=None) -> Tuple[bool, Dict[str, Dict[str, int]]]:
        client_max = (rate_limits or {}).get('client_rpm', self.default_limits['client']['max_requests'])
        user_max = (rate_limits or {}).get('user_rpm', self.default_limits['user']['max_requests'])
        endpoint_max = (rate_limits or {}).get('endpoint_rpm', self.default_limits['endpoint']['max_requests'])
        window = 60
        client_key = f'ratelimit:client:{client_id}'
        client_allowed, client_info = await self.check_rate_limit(client_key, window, client_max)
        user_key = f'ratelimit:user:{client_id}:{user_id}'
        user_allowed, user_info = await self.check_rate_limit(user_key, window, user_max)
        endpoint_key = f'ratelimit:endpoint:{client_id}:{user_id}:{endpoint}'
        endpoint_allowed, endpoint_info = await self.check_rate_limit(endpoint_key, window, endpoint_max)
        is_allowed = client_allowed and user_allowed and endpoint_allowed
        all_limits = {'client': client_info, 'user': user_info, 'endpoint': endpoint_info}
        return (is_allowed, all_limits)

    async def close(self):
        if self.redis:
            await self.redis.close()
            logger.info('Rate limiter Redis connection closed')
rate_limiter = RateLimiter()

def _extract_identity_from_request(request: Request) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return (None, None, None)
    try:
        from auth.auth import decode_access_token
        token = auth_header.split(' ', 1)[1]
        payload = decode_access_token(token)
        if not payload:
            return (None, None, None)
        client_id = payload.get('client_id')
        user_id = str(payload.get('_id') or payload.get('user_id') or payload.get('sub', ''))
        role = payload.get('role')
        return (client_id, user_id or None, role)
    except Exception:
        return (None, None, None)

def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.client.host if request.client else '0.0.0.0'
_rate_limits_cache: Dict[str, Tuple[Dict[str, int], float]] = {}
_RATE_LIMITS_CACHE_TTL = 60

async def _get_cached_rate_limits(client_id: str) -> Optional[Dict[str, int]]:
    now = time.time()
    cached = _rate_limits_cache.get(client_id)
    if cached and cached[1] > now:
        return cached[0]
    try:
        from services.subscription_service import get_rate_limits
        limits = await get_rate_limits(client_id)
        _rate_limits_cache[client_id] = (limits, now + _RATE_LIMITS_CACHE_TTL)
        return limits
    except Exception as e:
        logger.warning(f'Failed to fetch subscription rate limits for {client_id}: {e}')
        if cached:
            return cached[0]
        return None

async def rate_limit_middleware(request: Request, call_next):
    if request.url.path in ['/health', '/metrics', '/api/health']:
        return await call_next(request)
    client_id, user_id, role = _extract_identity_from_request(request)
    if role == 'super_admin':
        return await call_next(request)
    client_ip = _get_client_ip(request)
    effective_client_id = client_id or f'ip:{client_ip}'
    effective_user_id = user_id or f'ip:{client_ip}'
    endpoint = request.url.path
    plan_rate_limits = None
    if client_id:
        plan_rate_limits = await _get_cached_rate_limits(client_id)
    rate_limited_prefixes = await _get_rate_limited_prefixes()
    is_rate_limited = any((endpoint == pfx or endpoint.startswith(pfx + '/') for pfx in rate_limited_prefixes))
    if not is_rate_limited:
        return await call_next(request)
    is_allowed, limits_info = await rate_limiter.check_all_limits(effective_client_id, effective_user_id, endpoint, rate_limits=plan_rate_limits)
    if not is_allowed:
        if limits_info['client']['remaining'] == 0:
            limit_type = 'client'
        elif limits_info['user']['remaining'] == 0:
            limit_type = 'user'
        else:
            limit_type = 'endpoint'
        reset_time = limits_info[limit_type]['reset']
        logger.warning(f'Rate limit exceeded | type={limit_type} | client={effective_client_id} | user={effective_user_id} | endpoint={endpoint}')
        try:
            from util.audit_logger import audit_rate_limit_exceeded
            await audit_rate_limit_exceeded(user_id=effective_user_id, client_id=effective_client_id, limit_type=limit_type, endpoint=endpoint, ip_address=client_ip)
        except Exception as e:
            logger.error(f'Failed to log rate limit violation to audit: {e}')
        retry_after = max(1, reset_time - int(time.time()))
        return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS, content={'error': 'Rate limit exceeded', 'limit_type': limit_type, 'retry_after': retry_after, 'message': f'Too many requests. Please try again after {retry_after} seconds.'}, headers={'X-RateLimit-Limit': str(limits_info[limit_type]['limit']), 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': str(reset_time), 'Retry-After': str(retry_after)})
    response = await call_next(request)
    most_restrictive = min(limits_info.values(), key=lambda x: x['remaining'])
    response.headers['X-RateLimit-Limit'] = str(most_restrictive['limit'])
    response.headers['X-RateLimit-Remaining'] = str(most_restrictive['remaining'])
    response.headers['X-RateLimit-Reset'] = str(most_restrictive['reset'])
    return response