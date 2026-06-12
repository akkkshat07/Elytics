from __future__ import annotations
import asyncio
import logging
import time
import uuid
from typing import Optional
logger = logging.getLogger(__name__)
_LUA_ACQUIRE = "\nlocal key     = KEYS[1]\nlocal max_c   = tonumber(ARGV[1])\nlocal ttl     = tonumber(ARGV[2])\nlocal member  = ARGV[3]\nlocal now     = tonumber(ARGV[4])\nlocal expiry  = now + ttl\n\n-- Remove stale slots (expired leases from crashed replicas)\nredis.call('ZREMRANGEBYSCORE', key, '-inf', now)\n\n-- Count active slots\nlocal count = redis.call('ZCARD', key)\nif count < max_c then\n    redis.call('ZADD', key, expiry, member)\n    return 1\nend\nreturn 0\n"
_LUA_RELEASE = "\nredis.call('ZREM', KEYS[1], ARGV[1])\nreturn 1\n"

class RedisDistributedSemaphore:

    def __init__(self, redis_client, key: str, max_count: int, ttl_seconds: int=600) -> None:
        self._redis = redis_client
        self._key = key
        self._max = max_count
        self._ttl = ttl_seconds
        self._member: Optional[str] = None
        self._sha_acquire: Optional[str] = None
        self._sha_release: Optional[str] = None

    async def _load_scripts(self) -> None:
        if self._sha_acquire is None:
            self._sha_acquire = await self._redis.script_load(_LUA_ACQUIRE)
        if self._sha_release is None:
            self._sha_release = await self._redis.script_load(_LUA_RELEASE)

    async def acquire(self, timeout: float=300.0) -> bool:
        await self._load_scripts()
        member = str(uuid.uuid4())
        deadline = time.monotonic() + timeout
        poll_interval = 1.0
        while time.monotonic() < deadline:
            now = int(time.time())
            try:
                result = await self._redis.evalsha(self._sha_acquire, 1, self._key, self._max, self._ttl, member, now)
            except Exception as exc:
                if 'NOSCRIPT' in str(exc):
                    self._sha_acquire = None
                    self._sha_release = None
                    await self._load_scripts()
                    continue
                raise
            if result == 1:
                self._member = member
                logger.debug('DistributedSemaphore[%s] acquired (member=%s)', self._key, member)
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(poll_interval, remaining))
        logger.warning('DistributedSemaphore[%s] timeout after %.0fs (max=%d)', self._key, timeout, self._max)
        return False

    async def release(self) -> None:
        if self._member is None:
            return
        try:
            await self._load_scripts()
            await self._redis.evalsha(self._sha_release, 1, self._key, self._member)
            logger.debug('DistributedSemaphore[%s] released (member=%s)', self._key, self._member)
        except Exception as exc:
            if 'NOSCRIPT' in str(exc):
                self._sha_release = None
                await self._load_scripts()
                await self._redis.evalsha(self._sha_release, 1, self._key, self._member)
            else:
                logger.warning('DistributedSemaphore[%s] release error: %s', self._key, exc)
        finally:
            self._member = None

    @property
    def max_count(self) -> int:
        return self._max

    async def current_count(self) -> int:
        now = int(time.time())
        await self._redis.zremrangebyscore(self._key, '-inf', now)
        return await self._redis.zcard(self._key)

class _AsyncioSemaphoreAdapter:

    def __init__(self, max_count: int, key: str) -> None:
        self._sem = asyncio.Semaphore(max_count)
        self._max = max_count
        self._key = key
        self._acquired_count: int = 0

    async def acquire(self, timeout: float=300.0) -> bool:
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=timeout)
            self._acquired_count += 1
            return True
        except asyncio.TimeoutError:
            logger.warning('Semaphore[%s] (asyncio fallback) timeout after %.0fs', self._key, timeout)
            return False

    async def release(self) -> None:
        if self._acquired_count > 0:
            self._sem.release()
            self._acquired_count -= 1

    @property
    def max_count(self) -> int:
        return self._max

    async def current_count(self) -> int:
        return self._max - self._sem._value
_redis_available: Optional[bool] = None

async def _probe_redis(redis_client) -> bool:
    try:
        await redis_client.ping()
        return True
    except Exception:
        return False

async def make_semaphore(key: str, max_count: int, ttl_seconds: int=600, redis_client=None) -> 'RedisDistributedSemaphore | _AsyncioSemaphoreAdapter':
    global _redis_available
    if redis_client is not None:
        if _redis_available is None:
            _redis_available = await _probe_redis(redis_client)
        if _redis_available:
            logger.info('Semaphore[%s]: using Redis distributed semaphore (max=%d)', key, max_count)
            return RedisDistributedSemaphore(redis_client, key, max_count, ttl_seconds)
    logger.info('Semaphore[%s]: using asyncio fallback semaphore (max=%d) — multi-replica safety not guaranteed', key, max_count)
    return _AsyncioSemaphoreAdapter(max_count, key)