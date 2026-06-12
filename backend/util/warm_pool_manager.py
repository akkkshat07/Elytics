import asyncio
import logging
import time
from typing import Optional
logger = logging.getLogger(__name__)
_PREWARM_CLIENT_ID = '__prewarm__'

class WarmPoolManager:

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        from config.system_config import KERNEL_BACKEND
        if KERNEL_BACKEND != 'kubernetes':
            logger.info('WarmPoolManager: not running (KERNEL_BACKEND=%s)', KERNEL_BACKEND)
            return
        logger.info('WarmPoolManager: starting background pre-warm loop')
        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop(), name='warm-pool-manager')

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task and (not self._task.done()):
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        logger.info('WarmPoolManager: stopped')

    async def _loop(self) -> None:
        from config.system_config import K8S_WARM_POOL_SIZE, K8S_WARM_POOL_TARGET, PREWARM_POLL_INTERVAL, PREWARM_IDLE_TIMEOUT_MINUTES, MAX_DS_CONTAINERS_ENV
        from util.kernel_manager import _kernel_pool, _get_pool_lock, _get_global_semaphore, KernelStatus
        logger.info('WarmPoolManager: target=%d, poll_interval=%ds, idle_timeout=%dm', K8S_WARM_POOL_TARGET, PREWARM_POLL_INTERVAL, PREWARM_IDLE_TIMEOUT_MINUTES)
        while not self._stop_event.is_set():
            try:
                await self._tick(kernel_pool=_kernel_pool, pool_lock=_get_pool_lock(), global_sem=_get_global_semaphore(), target=K8S_WARM_POOL_TARGET, idle_timeout_minutes=PREWARM_IDLE_TIMEOUT_MINUTES)
            except Exception as exc:
                logger.error('WarmPoolManager: tick error: %s', exc, exc_info=True)
            try:
                await asyncio.wait_for(asyncio.shield(self._stop_event.wait()), timeout=PREWARM_POLL_INTERVAL)
            except asyncio.TimeoutError:
                pass

    async def _tick(self, kernel_pool: dict, pool_lock: asyncio.Lock, global_sem, target: int, idle_timeout_minutes: float) -> None:
        from util.k8s_kernel_manager import K8sKernelManager
        from util.kernel_manager import KernelStatus
        async with pool_lock:
            pool = kernel_pool.get(_PREWARM_CLIENT_ID, [])
            alive = []
            for mgr in pool:
                idle_minutes = 0.0
                if mgr._last_activity_time is not None:
                    from datetime import datetime
                    idle_minutes = (datetime.now() - mgr._last_activity_time).total_seconds() / 60
                is_alive_cached = getattr(mgr, '_is_alive_cache_result', False)
                status_ok = mgr._status in (KernelStatus.RUNNING, KernelStatus.IDLE)
                pod_alive = is_alive_cached and status_ok
                if not pod_alive or idle_minutes >= idle_timeout_minutes:
                    pod_name = getattr(mgr, '_pod_name', 'unknown')
                    logger.info('WarmPoolManager: expiring pod %s (alive=%s, idle=%.1fm)', pod_name, pod_alive, idle_minutes)
                    asyncio.create_task(mgr.stop())
                else:
                    alive.append(mgr)
            kernel_pool[_PREWARM_CLIENT_ID] = alive
            current_count = len(alive)
            slots_needed = target - current_count
        if slots_needed <= 0:
            return
        try:
            active = await global_sem.current_count()
            from config.system_config import DATA_SCIENCE_CONTAINER_CONFIG
            max_c = DATA_SCIENCE_CONTAINER_CONFIG['max_concurrent_containers']
            available = max_c - active
        except Exception:
            available = 0
        can_start = min(slots_needed, max(0, available // 2))
        if can_start == 0:
            logger.debug('WarmPoolManager: cluster near capacity, skipping pre-warm (needed=%d, available=%d)', slots_needed, available)
            return
        logger.info('WarmPoolManager: starting %d pre-warm pod(s) (pool=%d, target=%d)', can_start, current_count, target)
        tasks = [asyncio.create_task(self._start_prewarm_pod(), name=f'prewarm-{i}') for i in range(can_start)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                logger.warning('WarmPoolManager: pre-warm pod failed: %s', res)

    async def _start_prewarm_pod(self) -> None:
        from util.k8s_kernel_manager import K8sKernelManager
        from util.kernel_manager import _kernel_pool, _get_pool_lock, KernelStatus
        from config.system_config import K8S_WARM_POOL_TARGET
        mgr = K8sKernelManager(client_id=_PREWARM_CLIENT_ID, idle_timeout_minutes=0, enable_idle_monitor=False)
        try:
            success = await mgr.start()
        except Exception as exc:
            logger.warning('WarmPoolManager: pod start failed: %s', exc)
            return
        if not success:
            logger.warning('WarmPoolManager: pod start returned False')
            return
        mgr._status = KernelStatus.IDLE
        async with _get_pool_lock():
            pool = _kernel_pool.setdefault(_PREWARM_CLIENT_ID, [])
            if len(pool) < K8S_WARM_POOL_TARGET:
                pool.append(mgr)
                logger.info('WarmPoolManager: pod %s added to pre-warm pool (pool size=%d)', mgr._pod_name, len(pool))
            else:
                asyncio.create_task(mgr.stop())

def _adopt_prewarm_pod(client_id: str) -> Optional['K8sKernelManager']:
    from util.kernel_manager import _kernel_pool, KernelStatus
    pool = _kernel_pool.get(_PREWARM_CLIENT_ID, [])
    while pool:
        mgr = pool.pop(0)
        is_alive_cached = mgr._is_alive_cache_result if hasattr(mgr, '_is_alive_cache_result') else False
        status_ok = mgr._status in (KernelStatus.RUNNING, KernelStatus.IDLE)
        if is_alive_cached and status_ok:
            mgr.client_id = client_id
            mgr._status = KernelStatus.RUNNING
            mgr.update_activity()
            logger.info("WarmPoolManager: pre-warm pod %s adopted by client '%s'", mgr._pod_name, client_id)
            return mgr
        asyncio.create_task(mgr.stop())
    return None