import asyncio
import logging
import subprocess
import time
import socket
import os
import shutil
import signal
from typing import Optional, Dict, Any, Tuple
from enum import Enum
from datetime import datetime
logger = logging.getLogger(__name__)
from config.system_config import DATA_SCIENCE_CONTAINER_CONFIG, KERNEL_BACKEND, K8S_WARM_POOL_SIZE

def _get_local_config() -> Dict[str, Any]:
    from config.system_config import LOCAL_KERNEL_CONFIG
    return LOCAL_KERNEL_CONFIG
_global_semaphore = None
_client_semaphores: Dict[str, Any] = {}
_kernel_pool: Dict[str, list] = {}
_pool_lock: Optional[asyncio.Lock] = None

def _get_pool_lock() -> asyncio.Lock:
    global _pool_lock
    if _pool_lock is None:
        _pool_lock = asyncio.Lock()
    from typing import cast
    return cast(asyncio.Lock, _pool_lock)

def _get_redis_client():
    from config.system_config import USE_REDIS_SEMAPHORE, REDIS_URL
    if not USE_REDIS_SEMAPHORE:
        return None
    try:
        import redis.asyncio as aioredis
        return aioredis.from_url(REDIS_URL, decode_responses=True)
    except Exception as exc:
        logger.warning('Could not create Redis client for semaphore: %s', exc)
        return None

async def _get_global_semaphore():
    global _global_semaphore
    if _global_semaphore is None:
        if KERNEL_BACKEND == 'local':
            max_c = _get_local_config()['max_concurrent_kernels']
        else:
            max_c = DATA_SCIENCE_CONTAINER_CONFIG['max_concurrent_containers']
        from util.distributed_semaphore import make_semaphore
        _global_semaphore = await make_semaphore(key='coresight:kernel:global', max_count=max_c, ttl_seconds=600, redis_client=_get_redis_client())
        logger.info('Global kernel semaphore initialized: max %d total (backend=%s, type=%s)', max_c, KERNEL_BACKEND, type(_global_semaphore).__name__)
    return _global_semaphore

async def _get_client_semaphore(client_id: str):
    if client_id not in _client_semaphores:
        if KERNEL_BACKEND == 'local':
            max_per = _get_local_config()['max_per_client_kernels']
        else:
            max_per = DATA_SCIENCE_CONTAINER_CONFIG['max_per_client_containers']
        from util.distributed_semaphore import make_semaphore
        _client_semaphores[client_id] = await make_semaphore(key=f'coresight:kernel:client:{client_id}', max_count=max_per, ttl_seconds=600, redis_client=_get_redis_client())
        logger.info("Per-client semaphore for '%s': max %d (type=%s)", client_id, max_per, type(_client_semaphores[client_id]).__name__)
    return _client_semaphores[client_id]

class KernelStatus(Enum):
    STOPPED = 'stopped'
    STARTING = 'starting'
    RUNNING = 'running'
    STOPPING = 'stopping'
    IDLE = 'idle'
    ERROR = 'error'

class DockerKernelManager:

    def __init__(self, image_name: str='coresight-datascience:latest', client_id: str='default', idle_timeout_minutes: float=30.0, enable_idle_monitor: bool=True, volume_mounts: Optional[Dict[str, Dict[str, str]]]=None, environment: Optional[Dict[str, str]]=None):
        import docker
        from docker.errors import DockerException
        self.image_name = image_name
        self.client_id = client_id
        self.container_name = f'jupyter-{client_id}-{int(time.time())}'
        self.idle_timeout_minutes = idle_timeout_minutes
        self.enable_idle_monitor = enable_idle_monitor
        self._volume_mounts = volume_mounts or {}
        self._environment = environment or {}
        self._docker_client = docker.from_env()
        self._container = None
        self._host_port: Optional[int] = None
        self._status = KernelStatus.STOPPED
        self._last_activity_time = None
        self._idle_monitor_task = None
        self._global_semaphore_acquired = False
        self._client_semaphore_acquired = False
        self._mcp_server_process: Optional[subprocess.Popen] = None

    @property
    def data_dir(self) -> str:
        return '/data'

    @property
    def status(self) -> KernelStatus:
        return self._status

    @property
    def is_alive(self) -> bool:
        if not self._container:
            return False
        try:
            self._container.reload()
            return self._container.status == 'running'
        except Exception:
            return False

    def _find_free_port(self) -> int:
        for _ in range(20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                port = s.getsockname()[1]
            if port < 8000 or port > 8999:
                return port
        return port

    def copy_file_to_container(self, host_path: str, container_path: str) -> bool:
        if not self._container:
            return False
        try:
            import tarfile
            import io
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode='w') as tar:
                tar.add(host_path, arcname=os.path.basename(container_path))
            stream.seek(0)
            self._container.put_archive(path=os.path.dirname(container_path), data=stream)
            return True
        except Exception as e:
            logger.error(f'Failed to copy file to container: {e}')
            return False

    async def start(self) -> bool:
        if self.is_alive:
            logger.info(f'Container {self.container_name} already running.')
            return True
        timeout = DATA_SCIENCE_CONTAINER_CONFIG['semaphore_acquire_timeout']
        client_sem = await _get_client_semaphore(self.client_id)
        if not self._client_semaphore_acquired:
            logger.info(f'Client {self.client_id} at container limit. Queueing request (timeout {timeout}s)...')
            acquired = await client_sem.acquire(timeout=timeout)
            if not acquired:
                logger.warning(f"Per-client container limit reached for '{self.client_id}'. Container {self.container_name} rejected.")
                self._status = KernelStatus.ERROR
                return False
            self._client_semaphore_acquired = True
        global_sem = await _get_global_semaphore()
        if not self._global_semaphore_acquired:
            logger.info(f'Global container limit reached. Queueing request (timeout {timeout}s)...')
            acquired = await global_sem.acquire(timeout=timeout)
            if not acquired:
                if self._client_semaphore_acquired:
                    await client_sem.release()
                    self._client_semaphore_acquired = False
                logger.warning(f'Global container limit reached. Container {self.container_name} rejected.')
                self._status = KernelStatus.ERROR
                return False
            self._global_semaphore_acquired = True
        self._status = KernelStatus.STARTING
        try:
            existing_images = self._docker_client.images.list(name=self.image_name)
            if not existing_images:
                logger.warning(f'Image {self.image_name} not found. Attempting to build...')
                import os
                cwd = os.getcwd()
                dockerfile_name = 'Dockerfile.datascience'
                build_path = cwd
                if os.path.exists(os.path.join(cwd, dockerfile_name)):
                    build_path = cwd
                elif os.path.exists(os.path.join(cwd, 'coresight-backend', dockerfile_name)):
                    build_path = os.path.join(cwd, 'coresight-backend')
                else:
                    module_path = os.path.dirname(os.path.abspath(__file__))
                    backend_root = os.path.dirname(module_path)
                    if os.path.exists(os.path.join(backend_root, dockerfile_name)):
                        build_path = backend_root
                    else:
                        raise FileNotFoundError(f'Could not find {dockerfile_name} in {cwd} or {backend_root}')
                logger.info(f'Building image {self.image_name} from {build_path}...')
                self._docker_client.images.build(path=build_path, dockerfile=dockerfile_name, tag=self.image_name)
            self._host_port = self._find_free_port()
            cfg = DATA_SCIENCE_CONTAINER_CONFIG
            logger.info(f'Starting container {self.container_name} on port {self._host_port}...')
            container = self._docker_client.containers.run(self.image_name, name=self.container_name, ports={'8888/tcp': self._host_port}, detach=True, remove=True, mem_limit=cfg['mem_limit'], cpu_period=cfg['cpu_period'], cpu_quota=cfg['cpu_quota'], network_mode=cfg['network_mode'], read_only=cfg['read_only'], tmpfs=cfg['tmpfs'], volumes=self._volume_mounts if self._volume_mounts else None, environment=self._environment if self._environment else None)
            self._container = container
            import urllib.request
            import urllib.error
            jupyter_url = f'http://localhost:{self._host_port}/api/status'
            ready = False
            for attempt in range(60):
                if not self.is_alive:
                    await asyncio.sleep(0.5)
                    continue
                try:
                    req = urllib.request.Request(jupyter_url, method='GET')
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        if resp.status == 200:
                            ready = True
                            break
                except (urllib.error.URLError, ConnectionError, OSError):
                    pass
                await asyncio.sleep(0.5)
            if not ready:
                logger.warning(f'Jupyter API did not become ready within 30s on port {self._host_port}. Proceeding anyway — MCP may retry internally.')
            logger.info(f'Container started. Jupyter accessible at http://localhost:{self._host_port}')
            self._status = KernelStatus.RUNNING
            self._last_activity_time = datetime.now()
            if self.enable_idle_monitor:
                self._idle_monitor_task = asyncio.create_task(self._monitor_idle())
            return True
        except Exception as e:
            error_msg = str(e)
            if 'FileNotFoundError' in error_msg or 'Connection aborted' in error_msg:
                logger.error(f'Docker connection failed: {e}. Ensure Docker Desktop is running and socket permissions are correct. Try: sudo chmod 666 /var/run/docker.sock')
            else:
                logger.error(f'Failed to start kernel: {e}')
            await self.stop()
            self._status = KernelStatus.ERROR
            return False

    def get_connection_url(self) -> str:
        if not self._host_port:
            raise RuntimeError('Container not running')
        return f'http://localhost:{self._host_port}'

    async def recycle(self):
        try:
            import urllib.request
            import urllib.error
            base_url = self.get_connection_url()
            req = urllib.request.Request(base_url + '/api/kernels', method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                import json as _json
                kernels = _json.loads(resp.read().decode())
            for kernel in kernels:
                kid = kernel.get('id')
                if not kid:
                    continue
                restart_req = urllib.request.Request(f'{base_url}/api/kernels/{kid}/restart', data=b'{}', method='POST', headers={'Content-Type': 'application/json'})
                urllib.request.urlopen(restart_req, timeout=10)
                logger.debug(f'Restarted Jupyter kernel {kid} in {self.container_name}')
        except Exception as exc:
            logger.warning(f'Could not restart Jupyter kernels in {self.container_name}: {exc}. Container will still be pooled — agent re-bootstraps on reuse.')
        self._status = KernelStatus.IDLE
        self.update_activity()
        logger.info(f'Kernel {self.container_name} recycled to warm pool.')

    async def stop(self):
        self._status = KernelStatus.STOPPING
        if self._container:
            try:
                self._container.stop(timeout=2)
            except Exception as e:
                logger.warning(f'Error stopping container: {e}')
            finally:
                self._container = None
        if self._idle_monitor_task:
            self._idle_monitor_task.cancel()
        if self._global_semaphore_acquired:
            global_sem = await _get_global_semaphore()
            await global_sem.release()
            self._global_semaphore_acquired = False
        if self._client_semaphore_acquired:
            client_sem = await _get_client_semaphore(self.client_id)
            await client_sem.release()
            self._client_semaphore_acquired = False
        self._status = KernelStatus.STOPPED
        logger.info(f'Kernel {self.container_name} stopped.')

    async def _monitor_idle(self):
        try:
            while self.status in (KernelStatus.RUNNING, KernelStatus.IDLE):
                await asyncio.sleep(60)
                if self._last_activity_time:
                    delta = (datetime.now() - self._last_activity_time).total_seconds() / 60
                    if delta > self.idle_timeout_minutes:
                        logger.info(f'Kernel idle for {delta:.1f}m. Shutting down.')
                        if self.client_id in _kernel_pool:
                            if self in _kernel_pool[self.client_id]:
                                _kernel_pool[self.client_id].remove(self)
                        await self.stop()
                        break
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f'DockerKernelManager._monitor_idle crashed: {exc}', exc_info=True)

    def update_activity(self):
        self._last_activity_time = datetime.now()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

class LocalKernelManager:

    def __init__(self, client_id: str='default', idle_timeout_minutes: float=30.0, enable_idle_monitor: bool=True, volume_mounts: Optional[Dict[str, Dict[str, str]]]=None, environment: Optional[Dict[str, str]]=None, **_kwargs: Any):
        self.client_id = client_id
        self.container_name = f'jupyter-local-{client_id}-{int(time.time())}'
        self.idle_timeout_minutes = idle_timeout_minutes
        self.enable_idle_monitor = enable_idle_monitor
        self._environment = environment or {}
        self._host_data_dir: Optional[str] = None
        if volume_mounts:
            for host_path, mount_info in volume_mounts.items():
                if mount_info.get('bind') == '/data':
                    self._host_data_dir = host_path
                    break
        self._process: Optional[subprocess.Popen] = None
        self._host_port: Optional[int] = None
        self._status = KernelStatus.STOPPED
        self._last_activity_time: Optional[datetime] = None
        self._idle_monitor_task: Optional[asyncio.Task] = None
        self._global_semaphore_acquired = False
        self._client_semaphore_acquired = False

    @property
    def data_dir(self) -> str:
        return self._host_data_dir or '/data'

    @property
    def status(self) -> KernelStatus:
        return self._status

    @property
    def is_alive(self) -> bool:
        if not self._process:
            return False
        return self._process.poll() is None

    def _find_free_port(self) -> int:
        for _ in range(20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                port = s.getsockname()[1]
            if port < 8000 or port > 8999:
                return port
        return port

    def copy_file_to_container(self, host_path: str, container_path: str) -> bool:
        logger.debug(f'LocalKernelManager: skip copy_file_to_container (files accessible at {host_path})')
        return True

    async def start(self) -> bool:
        if self.is_alive:
            logger.info(f'Jupyter process {self.container_name} already running.')
            return True
        cfg = _get_local_config()
        timeout = cfg['semaphore_acquire_timeout']
        client_sem = await _get_client_semaphore(self.client_id)
        if not self._client_semaphore_acquired:
            acquired = await client_sem.acquire(timeout=timeout)
            if not acquired:
                logger.warning(f"Per-client kernel limit reached for '{self.client_id}'.")
                self._status = KernelStatus.ERROR
                return False
            self._client_semaphore_acquired = True
        global_sem = await _get_global_semaphore()
        if not self._global_semaphore_acquired:
            acquired = await global_sem.acquire(timeout=timeout)
            if not acquired:
                if self._client_semaphore_acquired:
                    await client_sem.release()
                    self._client_semaphore_acquired = False
                logger.warning(f'Global kernel limit reached.')
                self._status = KernelStatus.ERROR
                return False
            self._global_semaphore_acquired = True
        self._status = KernelStatus.STARTING
        try:
            self._host_port = self._find_free_port()
            import sys as _sys
            active_bin_dir = os.path.dirname(_sys.executable)
            jupyter_candidates = [os.path.join(active_bin_dir, 'jupyter'), os.path.join(active_bin_dir, 'jupyter.exe')]
            jupyter_bin = next((path for path in jupyter_candidates if os.path.exists(path)), None)
            if not jupyter_bin:
                jupyter_bin = shutil.which('jupyter') or 'jupyter'
            cmd = [jupyter_bin, 'server', f'--port={self._host_port}', '--ip=127.0.0.1', '--no-browser', '--ServerApp.token=', '--ServerApp.password=', '--ServerApp.disable_check_xsrf=True']
            env = os.environ.copy()
            env.update(self._environment)
            logger.info(f'Starting local Jupyter server on port {self._host_port} for {self.client_id}')
            popen_kwargs = {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL, 'env': env}
            if os.name == 'nt':
                popen_kwargs['creationflags'] = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
            elif hasattr(os, 'setsid'):
                popen_kwargs['preexec_fn'] = os.setsid
            self._process = subprocess.Popen(cmd, **popen_kwargs)
            import urllib.request
            import urllib.error
            jupyter_url = f'http://localhost:{self._host_port}/api/status'
            startup_timeout = cfg['jupyter_startup_timeout']
            max_attempts = startup_timeout * 2
            ready = False
            for _ in range(max_attempts):
                if not self.is_alive:
                    await asyncio.sleep(0.5)
                    continue
                try:
                    req = urllib.request.Request(jupyter_url, method='GET')
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        if resp.status == 200:
                            ready = True
                            break
                except (urllib.error.URLError, ConnectionError, OSError):
                    pass
                await asyncio.sleep(0.5)
            if not ready:
                logger.warning(f'Jupyter API did not become ready within {startup_timeout}s on port {self._host_port}. Proceeding anyway — MCP may retry internally.')
            logger.info(f'Local Jupyter server running at http://localhost:{self._host_port}')
            self._status = KernelStatus.RUNNING
            self._last_activity_time = datetime.now()
            if self.enable_idle_monitor:
                self._idle_monitor_task = asyncio.create_task(self._monitor_idle())
            return True
        except Exception as e:
            logger.error(f'Failed to start local Jupyter server: {e}')
            await self.stop()
            self._status = KernelStatus.ERROR
            return False

    def get_connection_url(self) -> str:
        if not self._host_port:
            raise RuntimeError('Jupyter server not running')
        return f'http://localhost:{self._host_port}'

    async def recycle(self):
        try:
            import urllib.request
            import urllib.error
            base_url = self.get_connection_url()
            req = urllib.request.Request(base_url + '/api/kernels', method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                import json as _json
                kernels = _json.loads(resp.read().decode())
            for kernel in kernels:
                kid = kernel.get('id')
                if not kid:
                    continue
                restart_req = urllib.request.Request(f'{base_url}/api/kernels/{kid}/restart', data=b'{}', method='POST', headers={'Content-Type': 'application/json'})
                urllib.request.urlopen(restart_req, timeout=10)
                logger.debug(f'Restarted Jupyter kernel {kid} in {self.container_name}')
        except Exception as exc:
            logger.warning(f'Could not restart Jupyter kernels in {self.container_name}: {exc}. Server will still be pooled — agent re-bootstraps on reuse.')
        self._status = KernelStatus.IDLE
        self.update_activity()
        logger.info(f'Kernel {self.container_name} recycled to warm pool.')

    async def stop(self):
        self._status = KernelStatus.STOPPING
        if self._process:
            try:
                if os.name == 'nt':
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=3)
                else:
                    pgid = os.getpgid(self._process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(pgid, signal.SIGKILL)
                        self._process.wait(timeout=3)
            except (ProcessLookupError, OSError):
                pass
            finally:
                self._process = None
        if self._idle_monitor_task:
            self._idle_monitor_task.cancel()
        if self._global_semaphore_acquired:
            global_sem = await _get_global_semaphore()
            await global_sem.release()
            self._global_semaphore_acquired = False
        if self._client_semaphore_acquired:
            client_sem = await _get_client_semaphore(self.client_id)
            await client_sem.release()
            self._client_semaphore_acquired = False
        self._status = KernelStatus.STOPPED
        logger.info(f'Kernel {self.container_name} stopped.')

    async def _monitor_idle(self):
        try:
            while self.status in (KernelStatus.RUNNING, KernelStatus.IDLE):
                await asyncio.sleep(60)
                if self._last_activity_time:
                    delta = (datetime.now() - self._last_activity_time).total_seconds() / 60
                    if delta > self.idle_timeout_minutes:
                        logger.info(f'Kernel idle for {delta:.1f}m. Shutting down.')
                        if self.client_id in _kernel_pool:
                            if self in _kernel_pool[self.client_id]:
                                _kernel_pool[self.client_id].remove(self)
                        await self.stop()
                        break
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f'LocalKernelManager._monitor_idle crashed: {exc}', exc_info=True)

    def update_activity(self):
        self._last_activity_time = datetime.now()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
KernelManager = Any

def _create_kernel_manager(client_id: str, **kwargs: Any) -> Any:
    kwargs.pop('use_docker', None)
    if KERNEL_BACKEND == 'local':
        return LocalKernelManager(client_id=client_id, **kwargs)
    elif KERNEL_BACKEND == 'kubernetes':
        from util.k8s_kernel_manager import K8sKernelManager
        return K8sKernelManager(client_id=client_id, **kwargs)
    else:
        return DockerKernelManager(client_id=client_id, **kwargs)

async def get_kernel_manager(client_id: str, use_pool: bool=True, **kwargs) -> Any:
    if KERNEL_BACKEND == 'local':
        timeout = _get_local_config()['semaphore_acquire_timeout']
    else:
        timeout = DATA_SCIENCE_CONTAINER_CONFIG['semaphore_acquire_timeout']
    start_time = asyncio.get_event_loop().time()
    while True:
        if use_pool:
            async with _get_pool_lock():
                if client_id in _kernel_pool and _kernel_pool[client_id]:
                    mgr = _kernel_pool[client_id].pop()
                    if mgr.is_alive:
                        logger.info(f'Reusing warm kernel {mgr.container_name} from pool for {client_id}')
                        mgr._status = KernelStatus.RUNNING
                        mgr.update_activity()
                        return mgr
                    else:
                        logger.warning(f'Kernel {mgr.container_name} in pool was dead. Discarding.')
                if KERNEL_BACKEND == 'kubernetes':
                    from util.warm_pool_manager import _adopt_prewarm_pod
                    prewarm_mgr = _adopt_prewarm_pod(client_id)
                    if prewarm_mgr is not None:
                        return prewarm_mgr
                _BOOT_KEY = '__boot_prewarm__'
                if _BOOT_KEY in _kernel_pool and _kernel_pool[_BOOT_KEY]:
                    mgr = _kernel_pool[_BOOT_KEY].pop()
                    if mgr.is_alive:
                        mgr.client_id = client_id
                        mgr._status = KernelStatus.RUNNING
                        mgr.update_activity()
                        logger.info(f'Adopted boot-prewarmed kernel {mgr.container_name} for {client_id}')
                        return mgr
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            raise RuntimeError(f'Timeout waiting for an available data science environment for {client_id}')
        logger.info(f'No warm kernel available. Creating new instance for {client_id} (backend={KERNEL_BACKEND})')
        return _create_kernel_manager(client_id=client_id, **kwargs)

async def release_kernel_manager(mgr: Any, use_pool: bool=True):
    if not mgr.is_alive or mgr.status == KernelStatus.ERROR or (not use_pool):
        if not use_pool:
            logger.info(f'Kernel {mgr.container_name} is not poolable. Stopping permanently.')
        else:
            logger.info(f'Kernel {mgr.container_name} is unhealthy. Stopping permanently.')
        await mgr.stop()
        return
    async with _get_pool_lock():
        if mgr.client_id not in _kernel_pool:
            _kernel_pool[mgr.client_id] = []
        pool_max = K8S_WARM_POOL_SIZE if KERNEL_BACKEND == 'kubernetes' else 3
        if len(_kernel_pool[mgr.client_id]) < pool_max:
            _kernel_pool[mgr.client_id].append(mgr)
            await mgr.recycle()
        else:
            logger.info(f'Warm pool full for {mgr.client_id}. Destroying kernel {mgr.container_name}.')
            await mgr.stop()