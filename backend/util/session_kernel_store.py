from __future__ import annotations
import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
logger = logging.getLogger(__name__)
SESSION_KERNEL_IDLE_TIMEOUT: int = int(os.getenv('SESSION_KERNEL_IDLE_TIMEOUT', '1800'))
MAX_SESSION_KERNELS: int = int(os.getenv('MAX_SESSION_KERNELS', '5'))

@dataclass
class SessionKernelEntry:
    kernel_manager: Any
    mcp_client: Any
    stdio_context_manager: Any
    mcp_context_manager: Any
    session_id: str
    client_id: str
    last_used: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_used = time.monotonic()

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_used

    @property
    def is_alive(self) -> bool:
        return self.kernel_manager is not None and getattr(self.kernel_manager, 'is_alive', False)
_session_entries: Dict[str, SessionKernelEntry] = {}
_pending_tasks: Dict[str, asyncio.Task] = {}
_store_lock: Optional[asyncio.Lock] = None

def _get_lock() -> asyncio.Lock:
    global _store_lock
    if _store_lock is None:
        _store_lock = asyncio.Lock()
    return _store_lock

def get_session_kernel(session_id: str) -> Optional[SessionKernelEntry]:
    if not session_id:
        return None
    entry = _session_entries.get(session_id)
    if entry is None:
        return None
    if not entry.is_alive:
        logger.warning('Session kernel for %s is dead — removing entry', session_id)
        _session_entries.pop(session_id, None)
        return None
    entry.touch()
    return entry

def set_session_kernel(entry: SessionKernelEntry) -> None:
    _session_entries[entry.session_id] = entry
    logger.info('Session kernel registered | session=%s client=%s', entry.session_id, entry.client_id)

def touch_session_kernel(session_id: str) -> None:
    entry = _session_entries.get(session_id)
    if entry:
        entry.touch()

def register_prewarm_task(session_id: str, task: asyncio.Task) -> None:
    _pending_tasks[session_id] = task

def pop_prewarm_task(session_id: str) -> Optional[asyncio.Task]:
    return _pending_tasks.pop(session_id, None)

async def evict_session_kernel(session_id: str, reason: str='eviction') -> None:
    entry = _session_entries.pop(session_id, None)
    if entry is None:
        return
    logger.info('Evicting session kernel | session=%s reason=%s', session_id, reason)
    await _teardown_entry(entry)

async def _teardown_entry(entry: SessionKernelEntry) -> None:
    if entry.mcp_context_manager is not None:
        try:
            await entry.mcp_context_manager.__aexit__(None, None, None)
        except Exception as exc:
            logger.warning('Error closing MCP client for session %s: %s', entry.session_id, exc)
    if entry.stdio_context_manager is not None:
        try:
            await entry.stdio_context_manager.__aexit__(None, None, None)
        except Exception as exc:
            logger.warning('Error closing MCP stdio for session %s: %s', entry.session_id, exc)
    if entry.kernel_manager is not None:
        try:
            from util.kernel_manager import release_kernel_manager
            await release_kernel_manager(entry.kernel_manager, use_pool=True)
        except Exception as exc:
            logger.warning('Error releasing kernel for session %s: %s', entry.session_id, exc)

async def enforce_session_limit() -> int:
    count = len(_session_entries)
    if count <= MAX_SESSION_KERNELS:
        return 0
    sorted_entries = sorted(_session_entries.items(), key=lambda kv: kv[1].last_used)
    excess = count - MAX_SESSION_KERNELS
    evicted = 0
    for sid, _entry in sorted_entries[:excess]:
        await evict_session_kernel(sid, reason=f'count_cap({count}>{MAX_SESSION_KERNELS})')
        evicted += 1
    return evicted

async def cleanup_idle_session_kernels() -> int:
    to_evict = [sid for sid, entry in list(_session_entries.items()) if entry.idle_seconds > SESSION_KERNEL_IDLE_TIMEOUT]
    for sid in to_evict:
        await evict_session_kernel(sid, reason=f'idle>{SESSION_KERNEL_IDLE_TIMEOUT}s')
    cap_evicted = await enforce_session_limit()
    total = len(to_evict) + cap_evicted
    if total:
        logger.info('Session kernel cleanup: evicted %d (idle=%d, cap=%d), remaining=%d', total, len(to_evict), cap_evicted, len(_session_entries))
    return total

async def cleanup_all_session_kernels() -> None:
    session_ids = list(_session_entries.keys())
    for sid in session_ids:
        await evict_session_kernel(sid, reason='app_shutdown')
    logger.info('Session kernel store: shutdown cleanup complete (%d entries)', len(session_ids))

async def _prewarm_session_kernel(session_id: str, client_id: str, idle_timeout_minutes: float=30.0) -> None:
    if not session_id or not client_id:
        return
    if get_session_kernel(session_id) is not None:
        logger.debug('Pre-warm skipped: session %s already has a live kernel', session_id)
        return
    logger.info('Pre-warm: starting kernel for session=%s client=%s', session_id, client_id)
    kernel_manager = None
    stdio_ctx = None
    try:
        from util.kernel_manager import get_kernel_manager
        kernel_manager = await get_kernel_manager(client_id=client_id, idle_timeout_minutes=idle_timeout_minutes, use_docker=False)
        success = await kernel_manager.start()
        if not success:
            logger.warning('Pre-warm: kernel start failed for session=%s — skipping MCP init', session_id)
            await kernel_manager.stop()
            return
        from config.system_config import MCP_SERVER_COMMAND
        from mcp.client.stdio import stdio_client, StdioServerParameters
        from util.mcp.client import McpClient
        kernel_url = kernel_manager.get_connection_url()
        base_cmd = MCP_SERVER_COMMAND.split()
        server_args = ['--jupyter-url', kernel_url, '--jupyter-token', '']
        full_args = base_cmd[1:] + server_args
        server_params = StdioServerParameters(command=base_cmd[0], args=full_args)
        stdio_ctx = stdio_client(server_params)
        read_stream, write_stream = await stdio_ctx.__aenter__()
        mcp_ctx = McpClient(read_stream, write_stream)
        mcp_client = await mcp_ctx.__aenter__()
        entry = SessionKernelEntry(kernel_manager=kernel_manager, mcp_client=mcp_client, stdio_context_manager=stdio_ctx, mcp_context_manager=mcp_ctx, session_id=session_id, client_id=client_id)
        set_session_kernel(entry)
        await enforce_session_limit()
        logger.info('Pre-warm complete: kernel+MCP ready | session=%s', session_id)
    except asyncio.CancelledError:
        logger.info('Pre-warm cancelled for session=%s', session_id)
        await _cleanup_partial(kernel_manager, stdio_ctx)
        raise
    except Exception as exc:
        logger.warning('Pre-warm failed for session=%s: %s — agent will do full init instead', session_id, exc)
        await _cleanup_partial(kernel_manager, stdio_ctx)

async def _cleanup_partial(kernel_manager: Any, stdio_ctx: Any) -> None:
    if stdio_ctx is not None:
        try:
            await stdio_ctx.__aexit__(None, None, None)
        except Exception:
            pass
    if kernel_manager is not None:
        try:
            from util.kernel_manager import release_kernel_manager
            await release_kernel_manager(kernel_manager, use_pool=True)
        except Exception:
            try:
                await kernel_manager.stop()
            except Exception:
                pass