import asyncio
import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import yaml
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account
from kubernetes import client as k8s_client, config as k8s_config
from kubernetes.client.exceptions import ApiException
from kubernetes.config.kube_config import KUBE_CONFIG_DEFAULT_LOCATION
from config.system_config import K8S_DEFAULT_CPU_LIMIT, K8S_DEFAULT_CPU_REQUEST, K8S_DEFAULT_MEM_LIMIT, K8S_DEFAULT_MEM_REQUEST, K8S_CONTEXT, K8S_KERNEL_IMAGE, K8S_KERNEL_NAMESPACE, K8S_POD_TIMEOUT, K8S_WARM_POOL_SIZE, DATA_SCIENCE_CONTAINER_CONFIG, GCS_BUCKET, S3_ACCESS_KEY, S3_ENDPOINT_URL, S3_SECRET_KEY, STORAGE_BACKEND
from util.kernel_manager import KernelStatus, _get_global_semaphore, _get_client_semaphore, _kernel_pool, _get_pool_lock
logger = logging.getLogger(__name__)
_IS_ALIVE_CACHE_TTL: float = 5.0
_GKE_K8S_SCOPE = 'https://www.googleapis.com/auth/cloud-platform'

def _running_in_cluster() -> bool:
    return os.path.isfile('/var/run/secrets/kubernetes.io/serviceaccount/token')

def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def _prepend_to_path(candidate: str) -> None:
    if os.path.isdir(candidate):
        path_parts = os.environ.get('PATH', '').split(os.pathsep)
        if candidate not in path_parts:
            os.environ['PATH'] = candidate + os.pathsep + os.environ.get('PATH', '')

def _prepare_gke_cli_paths() -> None:
    gcloud_path = shutil.which('gcloud')
    if not gcloud_path or gcloud_path.startswith('/snap/'):
        for candidate in ['/usr/lib/google-cloud-sdk/bin', '/usr/bin']:
            if os.path.isfile(os.path.join(candidate, 'gcloud')):
                _prepend_to_path(candidate)
                break
    if not shutil.which('gke-gcloud-auth-plugin'):
        for candidate in ['/usr/lib/google-cloud-sdk/bin', os.path.expanduser('~/google-cloud-sdk/bin'), '/opt/homebrew/share/google-cloud-sdk/bin', '/usr/local/share/google-cloud-sdk/bin']:
            if os.path.isfile(os.path.join(candidate, 'gke-gcloud-auth-plugin')):
                _prepend_to_path(candidate)
                logger.info(f'Added {candidate} to PATH for gke-gcloud-auth-plugin')
                break

def _resolve_gke_service_account_path() -> Optional[Path]:
    env_candidates = [('GOOGLE_APPLICATION_CREDENTIALS', os.getenv('GOOGLE_APPLICATION_CREDENTIALS')), ('GCP_SA_KEY_FILE', os.getenv('GCP_SA_KEY_FILE'))]
    for env_name, env_value in env_candidates:
        if not env_value:
            continue
        resolved = Path(env_value).expanduser()
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f'Kubernetes credential file not found at {resolved}. Set GOOGLE_APPLICATION_CREDENTIALS or GCP_SA_KEY_FILE to a valid service-account JSON path.')
        os.environ.setdefault('GOOGLE_APPLICATION_CREDENTIALS', str(resolved))
        return resolved
    return None

def _resolve_kubeconfig_path() -> Path:
    raw_paths = os.getenv('KUBECONFIG', KUBE_CONFIG_DEFAULT_LOCATION)
    for candidate in raw_paths.split(os.pathsep):
        if not candidate:
            continue
        resolved = Path(candidate).expanduser()
        if resolved.is_file():
            return resolved
    default_path = Path(KUBE_CONFIG_DEFAULT_LOCATION).expanduser()
    raise FileNotFoundError(f'No kubeconfig file found. Checked {raw_paths!r} (default: {default_path}).')

def _load_kubeconfig_document(kubeconfig_path: Path) -> Dict[str, Any]:
    with kubeconfig_path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}

def _select_kubeconfig_context(kubeconfig: Dict[str, Any], context_name: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    selected_name = context_name or K8S_CONTEXT or kubeconfig.get('current-context')
    contexts = kubeconfig.get('contexts') or []
    for entry in contexts:
        if entry.get('name') == selected_name:
            return (selected_name, entry.get('context') or {})
    available = ', '.join(sorted((entry.get('name', '<unknown>') for entry in contexts)))
    raise RuntimeError(f"Kubernetes context {selected_name!r} not found in kubeconfig. Available contexts: {available or '<none>'}.")

def _get_named_kubeconfig_entry(entries: list, entry_name: str, entry_kind: str) -> Dict[str, Any]:
    for entry in entries:
        if entry.get('name') == entry_name:
            return entry.get(entry_kind) or {}
    raise RuntimeError(f'Kubernetes {entry_kind} {entry_name!r} not found in kubeconfig.')

def _write_k8s_ca_file(cluster_name: str, cluster_config: Dict[str, Any], kubeconfig_path: Path) -> Optional[str]:
    ca_data = cluster_config.get('certificate-authority-data')
    if ca_data:
        cert_bytes = base64.b64decode(ca_data)
        digest = hashlib.sha256(cert_bytes).hexdigest()[:16]
        ca_dir = Path.home() / '.kube' / 'coresight-ca'
        ca_dir.mkdir(parents=True, exist_ok=True)
        ca_path = ca_dir / f'{cluster_name}-{digest}.crt'
        ca_path.write_bytes(cert_bytes)
        return str(ca_path)
    ca_path_raw = cluster_config.get('certificate-authority')
    if not ca_path_raw:
        return None
    ca_path = Path(ca_path_raw).expanduser()
    if not ca_path.is_absolute():
        ca_path = (kubeconfig_path.parent / ca_path).resolve()
    if not ca_path.is_file():
        raise FileNotFoundError(f'Kubernetes CA certificate file not found at {ca_path}.')
    return str(ca_path)

def _load_k8s_config_from_service_account(kubeconfig_path: Path, credentials_path: Path, context_name: Optional[str]) -> str:
    kubeconfig = _load_kubeconfig_document(kubeconfig_path)
    selected_context_name, selected_context = _select_kubeconfig_context(kubeconfig, context_name)
    cluster_name = selected_context.get('cluster')
    if not cluster_name:
        raise RuntimeError(f'Kubernetes context {selected_context_name!r} does not define a cluster.')
    cluster_config = _get_named_kubeconfig_entry(kubeconfig.get('clusters') or [], cluster_name, 'cluster')
    server = cluster_config.get('server')
    if not server:
        raise RuntimeError(f'Kubernetes cluster {cluster_name!r} does not define an API server URL.')
    ca_cert_path = _write_k8s_ca_file(cluster_name, cluster_config, kubeconfig_path)
    if not ca_cert_path:
        raise RuntimeError(f'Kubernetes cluster {cluster_name!r} does not define a CA certificate.')
    credentials = service_account.Credentials.from_service_account_file(str(credentials_path), scopes=[_GKE_K8S_SCOPE])
    auth_request = GoogleAuthRequest()
    configuration = k8s_client.Configuration()
    configuration.host = server
    configuration.verify_ssl = True
    configuration.ssl_ca_cert = ca_cert_path

    def _refresh_api_key(config: k8s_client.Configuration) -> None:
        if credentials.expired or not config.api_key.get('authorization'):
            credentials.refresh(auth_request)
            config.api_key['authorization'] = credentials.token
    _refresh_api_key(configuration)
    configuration.api_key_prefix = {'authorization': 'Bearer'}
    configuration.refresh_api_key_hook = _refresh_api_key
    k8s_client.Configuration.set_default(configuration)
    return selected_context_name

def _load_k8s_config() -> None:
    try:
        k8s_config.load_incluster_config()
        logger.info('Loaded in-cluster Kubernetes configuration')
        return
    except k8s_config.ConfigException:
        pass
    kubeconfig_path = _resolve_kubeconfig_path()
    credentials_path = _resolve_gke_service_account_path()
    requested_context = K8S_CONTEXT or None
    if credentials_path is not None:
        selected_context = _load_k8s_config_from_service_account(kubeconfig_path=kubeconfig_path, credentials_path=credentials_path, context_name=requested_context)
        default_config = k8s_client.Configuration.get_default_copy()
        logger.info('Loaded Kubernetes config from service account: kubeconfig=%s, context=%s, host=%s', kubeconfig_path, selected_context, default_config.host)
        return
    _prepare_gke_cli_paths()
    k8s_config.load_kube_config(config_file=str(kubeconfig_path), context=requested_context)
    default_config = k8s_client.Configuration.get_default_copy()
    logger.info('Loaded Kubernetes config via kubeconfig exec auth: kubeconfig=%s, context=%s, host=%s', kubeconfig_path, requested_context or 'current-context', default_config.host)

class K8sKernelManager:

    def __init__(self, client_id: str='default', idle_timeout_minutes: float=30.0, enable_idle_monitor: bool=True, volume_mounts: Optional[Dict[str, Dict[str, str]]]=None, environment: Optional[Dict[str, str]]=None, **_kwargs: Any):
        _load_k8s_config()
        self.client_id = client_id
        self._timestamp = int(time.time())
        self.container_name = f'kernel-{client_id}-{self._timestamp}'
        self.idle_timeout_minutes = idle_timeout_minutes
        self.enable_idle_monitor = enable_idle_monitor
        self._extra_environment = environment or {}
        self._namespace = K8S_KERNEL_NAMESPACE
        self._image = K8S_KERNEL_IMAGE
        self._pod_name = self.container_name
        self._service_name = f'svc-{self.container_name}'
        self._core_api = k8s_client.CoreV1Api()
        self._status = KernelStatus.STOPPED
        self._last_activity_time: Optional[datetime] = None
        self._idle_monitor_task: Optional[asyncio.Task] = None
        self._in_cluster = _running_in_cluster()
        self._port_forward_proc: Optional[subprocess.Popen] = None
        self._local_port: Optional[int] = None
        self._pod_ip: Optional[str] = None
        self._service_created: bool = False
        self._global_semaphore_acquired: bool = False
        self._client_semaphore_acquired: bool = False
        self._is_alive_cache_time: float = 0.0
        self._is_alive_cache_result: bool = False
        self._pf_keepalive_task: Optional[asyncio.Task] = None

    @property
    def data_dir(self) -> str:
        return '/data'

    @property
    def status(self) -> KernelStatus:
        return self._status

    @property
    def is_alive(self) -> bool:
        if self._port_forward_proc is not None and self._port_forward_proc.poll() is not None:
            self._is_alive_cache_result = False
            self._is_alive_cache_time = time.monotonic()
            return False
        now = time.monotonic()
        if now - self._is_alive_cache_time < _IS_ALIVE_CACHE_TTL:
            return self._is_alive_cache_result
        try:
            pod = self._core_api.read_namespaced_pod(name=self._pod_name, namespace=self._namespace)
            result = pod.status.phase == 'Running'
        except ApiException:
            result = False
        except Exception:
            result = False
        self._is_alive_cache_result = result
        self._is_alive_cache_time = now
        return result

    def _build_env_vars(self) -> list:
        import os as _os
        secret_name = _os.environ.get('GCS_K8S_SECRET_NAME', '')
        env_vars = [k8s_client.V1EnvVar(name='S3_ENDPOINT_URL', value=S3_ENDPOINT_URL), k8s_client.V1EnvVar(name='GCS_BUCKET', value=GCS_BUCKET), k8s_client.V1EnvVar(name='CLIENT_DATA_PATH', value=f'clients/{self.client_id}/'), k8s_client.V1EnvVar(name='STORAGE_BACKEND', value=STORAGE_BACKEND)]
        if secret_name:
            env_vars.append(k8s_client.V1EnvVar(name='S3_ACCESS_KEY', value_from=k8s_client.V1EnvVarSource(secret_key_ref=k8s_client.V1SecretKeySelector(name=secret_name, key='access_key'))))
            env_vars.append(k8s_client.V1EnvVar(name='S3_SECRET_KEY', value_from=k8s_client.V1EnvVarSource(secret_key_ref=k8s_client.V1SecretKeySelector(name=secret_name, key='secret_key'))))
            logger.info("K8sKernelManager: injecting GCS credentials from K8s Secret '%s'", secret_name)
        else:
            env_vars.append(k8s_client.V1EnvVar(name='S3_ACCESS_KEY', value=S3_ACCESS_KEY))
            env_vars.append(k8s_client.V1EnvVar(name='S3_SECRET_KEY', value=S3_SECRET_KEY))
        for k, v in self._extra_environment.items():
            env_vars.append(k8s_client.V1EnvVar(name=k, value=v))
        return env_vars

    def _build_pod_spec(self) -> k8s_client.V1Pod:
        labels = {'app': 'coresight-kernel', 'client-id': self.client_id, 'managed-by': 'coresight-api'}
        container = k8s_client.V1Container(name='jupyter', image=self._image, ports=[k8s_client.V1ContainerPort(container_port=8888)], env=self._build_env_vars(), resources=k8s_client.V1ResourceRequirements(requests={'cpu': K8S_DEFAULT_CPU_REQUEST, 'memory': K8S_DEFAULT_MEM_REQUEST}, limits={'cpu': K8S_DEFAULT_CPU_LIMIT, 'memory': K8S_DEFAULT_MEM_LIMIT}), readiness_probe=k8s_client.V1Probe(http_get=k8s_client.V1HTTPGetAction(path='/api/status', port=8888), initial_delay_seconds=5, period_seconds=3, failure_threshold=10), liveness_probe=k8s_client.V1Probe(http_get=k8s_client.V1HTTPGetAction(path='/api/status', port=8888), initial_delay_seconds=30, period_seconds=10, failure_threshold=3), security_context=k8s_client.V1SecurityContext(run_as_non_root=True, run_as_user=1000, read_only_root_filesystem=False, allow_privilege_escalation=False, capabilities=k8s_client.V1Capabilities(drop=['ALL'])))
        pod = k8s_client.V1Pod(api_version='v1', kind='Pod', metadata=k8s_client.V1ObjectMeta(name=self._pod_name, namespace=self._namespace, labels=labels), spec=k8s_client.V1PodSpec(containers=[container], restart_policy='Never', topology_spread_constraints=[k8s_client.V1TopologySpreadConstraint(max_skew=1, topology_key='kubernetes.io/hostname', when_unsatisfiable='ScheduleAnyway', label_selector=k8s_client.V1LabelSelector(match_labels={'app': 'coresight-kernel'}))]))
        return pod

    def _build_service_spec(self) -> k8s_client.V1Service:
        return k8s_client.V1Service(api_version='v1', kind='Service', metadata=k8s_client.V1ObjectMeta(name=self._service_name, namespace=self._namespace), spec=k8s_client.V1ServiceSpec(type='ClusterIP', selector={'app': 'coresight-kernel', 'client-id': self.client_id, 'managed-by': 'coresight-api'}, ports=[k8s_client.V1ServicePort(port=8888, target_port=8888)]))

    async def start(self) -> bool:
        if self.is_alive:
            logger.info(f'Pod {self._pod_name} already running.')
            return True
        timeout = DATA_SCIENCE_CONTAINER_CONFIG['semaphore_acquire_timeout']
        client_sem = await _get_client_semaphore(self.client_id)
        if not self._client_semaphore_acquired:
            if await client_sem.current_count() >= client_sem.max_count:
                logger.info(f'Client {self.client_id} at K8s pod limit. Queuing request (timeout {timeout}s)...')
            acquired = await client_sem.acquire(timeout=timeout)
            if not acquired:
                logger.warning(f"Per-client K8s pod limit reached for '{self.client_id}'. Pod {self._pod_name} rejected after {timeout}s wait.")
                self._status = KernelStatus.ERROR
                return False
            self._client_semaphore_acquired = True
        global_sem = await _get_global_semaphore()
        if not self._global_semaphore_acquired:
            if await global_sem.current_count() >= global_sem.max_count:
                logger.info('Global K8s pod limit reached. Queuing request...')
            acquired = await global_sem.acquire(timeout=timeout)
            if not acquired:
                if self._client_semaphore_acquired:
                    await client_sem.release()
                    self._client_semaphore_acquired = False
                logger.warning(f'Global K8s pod limit reached. Pod {self._pod_name} rejected.')
                self._status = KernelStatus.ERROR
                return False
            self._global_semaphore_acquired = True
        self._status = KernelStatus.STARTING
        try:
            logger.info(f'Creating pod {self._pod_name} in namespace {self._namespace}')
            await asyncio.to_thread(self._core_api.create_namespaced_pod, namespace=self._namespace, body=self._build_pod_spec())
            if not self._in_cluster:
                logger.info(f'Creating service {self._service_name} in namespace {self._namespace}')
                await asyncio.to_thread(self._core_api.create_namespaced_service, namespace=self._namespace, body=self._build_service_spec())
                self._service_created = True
            logger.info(f'Waiting up to {K8S_POD_TIMEOUT}s for pod {self._pod_name} to be Running...')
            deadline = time.monotonic() + K8S_POD_TIMEOUT
            pod_running = False
            while time.monotonic() < deadline:
                try:
                    pod = await asyncio.to_thread(self._core_api.read_namespaced_pod, name=self._pod_name, namespace=self._namespace)
                    if pod.status.phase == 'Running':
                        self._pod_ip = pod.status.pod_ip
                        pod_running = True
                        break
                    if pod.status.phase in ('Failed', 'Unknown'):
                        raise RuntimeError(f'Pod {self._pod_name} entered phase {pod.status.phase}')
                except ApiException as exc:
                    if exc.status == 404:
                        pass
                    else:
                        raise
                await asyncio.sleep(2)
            if not pod_running:
                raise RuntimeError(f'Pod {self._pod_name} did not reach Running within {K8S_POD_TIMEOUT}s')
            if not self._in_cluster:
                await self._start_port_forward()
                self._pf_keepalive_task = asyncio.create_task(self._port_forward_keepalive(), name=f'pf-keepalive-{self._pod_name}')
            jupyter_url = self.get_connection_url() + '/api/status'
            logger.info(f'Polling Jupyter readiness at {jupyter_url}')
            jupyter_ready = False
            jupyter_deadline = time.monotonic() + 60

            def _probe_jupyter(url: str) -> bool:
                try:
                    req = urllib.request.Request(url, method='GET')
                    with urllib.request.urlopen(req, timeout=3) as resp:
                        return resp.status == 200
                except (urllib.error.URLError, ConnectionError, OSError):
                    return False
            while time.monotonic() < jupyter_deadline:
                if await asyncio.to_thread(_probe_jupyter, jupyter_url):
                    jupyter_ready = True
                    break
                await asyncio.sleep(1)
            if not jupyter_ready:
                logger.warning(f'Jupyter API did not become ready within 60s on pod {self._pod_name}. Proceeding anyway — MCP may retry internally.')
            self._status = KernelStatus.RUNNING
            self._last_activity_time = datetime.now()
            logger.info(f'K8s kernel {self._pod_name} running. URL: {self.get_connection_url()}')
            if self.enable_idle_monitor:
                self._idle_monitor_task = asyncio.create_task(self._monitor_idle())
            return True
        except Exception as exc:
            logger.error(f'Failed to start K8s kernel {self._pod_name}: {exc}')
            try:
                await self.stop()
            except Exception as stop_exc:
                logger.warning('Cleanup after failed K8s kernel start also failed for %s: %s', self._pod_name, stop_exc)
            self._status = KernelStatus.ERROR
            raise RuntimeError(f'K8s kernel start failed: {exc}') from exc

    async def _start_port_forward(self) -> None:
        max_attempts = 8
        attempt_timeout = 60
        kubectl_restart_delay = 4
        inter_attempt_delay = 5
        for attempt in range(1, max_attempts + 1):
            self._local_port = _find_free_port()

            def _launch_pf(port: int) -> subprocess.Popen:
                cmd = ['kubectl', 'port-forward', f'pod/{self._pod_name}', f'{port}:8888', '-n', self._namespace]
                logger.info(f'Starting port-forward (attempt {attempt}/{max_attempts}): localhost:{port} → {self._pod_name}:8888')
                return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._port_forward_proc = _launch_pf(self._local_port)
            verify_url = f'http://127.0.0.1:{self._local_port}/api/status'
            deadline = time.monotonic() + attempt_timeout
            tunnel_ok = False
            while time.monotonic() < deadline:
                if self._port_forward_proc.poll() is not None:
                    rc = self._port_forward_proc.returncode
                    remaining = deadline - time.monotonic()
                    logger.debug(f'port-forward subprocess exited (code {rc}), restarting (attempt {attempt}, {remaining:.0f}s remaining)...')
                    self._port_forward_proc = None
                    if remaining < kubectl_restart_delay + 2:
                        break
                    await asyncio.sleep(kubectl_restart_delay)
                    self._local_port = _find_free_port()
                    verify_url = f'http://127.0.0.1:{self._local_port}/api/status'
                    self._port_forward_proc = _launch_pf(self._local_port)
                    continue

                def _probe_tunnel(url: str) -> bool:
                    try:
                        req = urllib.request.Request(url, method='GET')
                        with urllib.request.urlopen(req, timeout=2) as resp:
                            return resp.status == 200
                    except (urllib.error.URLError, ConnectionError, OSError):
                        return False
                if await asyncio.to_thread(_probe_tunnel, verify_url):
                    tunnel_ok = True
                    break
                await asyncio.sleep(2)
            if tunnel_ok:
                logger.info(f'Port-forward tunnel verified on localhost:{self._local_port} (attempt {attempt}/{max_attempts})')
                return
            logger.warning(f'port-forward attempt {attempt}/{max_attempts} exhausted ({attempt_timeout}s window); will retry')
            self._stop_port_forward()
            if attempt < max_attempts:
                await asyncio.sleep(inter_attempt_delay)
        raise RuntimeError(f'Port-forward to {self._pod_name}:8888 failed after {max_attempts} attempts')

    def _stop_port_forward(self) -> None:
        if self._port_forward_proc is not None:
            try:
                self._port_forward_proc.terminate()
                self._port_forward_proc.wait(timeout=5)
                logger.info('Port-forward process terminated.')
            except Exception as exc:
                logger.warning(f'Error stopping port-forward: {exc}')
            finally:
                self._port_forward_proc = None
                self._local_port = None

    async def stop(self) -> None:
        self._status = KernelStatus.STOPPING
        self._stop_port_forward()
        if self._service_created:
            try:
                self._core_api.delete_namespaced_service(name=self._service_name, namespace=self._namespace)
                logger.info(f'Deleted service {self._service_name}')
            except ApiException as exc:
                if exc.status != 404:
                    logger.warning(f'Error deleting service {self._service_name}: {exc}')
            except Exception as exc:
                logger.warning('Unexpected error deleting service %s: %s', self._service_name, exc)
            self._service_created = False
        try:
            self._core_api.delete_namespaced_pod(name=self._pod_name, namespace=self._namespace)
            logger.info(f'Deleted pod {self._pod_name}')
        except ApiException as exc:
            if exc.status != 404:
                logger.warning(f'Error deleting pod {self._pod_name}: {exc}')
        except Exception as exc:
            logger.warning('Unexpected error deleting pod %s: %s', self._pod_name, exc)
        if self._idle_monitor_task:
            self._idle_monitor_task.cancel()
        if self._pf_keepalive_task:
            self._pf_keepalive_task.cancel()
            self._pf_keepalive_task = None
        if self._global_semaphore_acquired:
            global_sem = await _get_global_semaphore()
            await global_sem.release()
            self._global_semaphore_acquired = False
        if self._client_semaphore_acquired:
            client_sem = await _get_client_semaphore(self.client_id)
            await client_sem.release()
            self._client_semaphore_acquired = False
        self._status = KernelStatus.STOPPED
        logger.info(f'K8s kernel {self._pod_name} stopped.')

    def get_connection_url(self) -> str:
        if self._in_cluster:
            if self._pod_ip:
                return f'http://{self._pod_ip}:8888'
            return f'http://{self._service_name}.{self._namespace}.svc.cluster.local:8888'
        if self._local_port:
            return f'http://127.0.0.1:{self._local_port}'
        return f'http://{self._service_name}.{self._namespace}.svc.cluster.local:8888'

    async def recycle(self) -> None:
        try:
            base_url = self.get_connection_url()
            req = urllib.request.Request(base_url + '/api/kernels', method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                kernels = json.loads(resp.read().decode())
            for kernel in kernels:
                kid = kernel.get('id')
                if not kid:
                    continue
                restart_req = urllib.request.Request(f'{base_url}/api/kernels/{kid}/restart', data=b'{}', method='POST', headers={'Content-Type': 'application/json'})
                urllib.request.urlopen(restart_req, timeout=10)
                logger.debug(f'Restarted kernel {kid} on {self._pod_name}')
        except Exception as exc:
            logger.warning(f'Could not restart Jupyter kernels on {self._pod_name}: {exc}. Pod will still be pooled — the agent re-bootstraps credentials on reuse.')
        self._is_alive_cache_time = 0.0
        self._status = KernelStatus.IDLE
        self.update_activity()
        logger.info(f'K8s kernel {self._pod_name} recycled to warm pool.')

    def update_activity(self) -> None:
        self._last_activity_time = datetime.now()

    def copy_file_to_container(self, host_path: str, container_path: str) -> bool:
        logger.debug(f'K8sKernelManager: skip copy_file_to_container (use storage backend instead)')
        return False

    async def _monitor_idle(self) -> None:
        try:
            while self.status in (KernelStatus.RUNNING, KernelStatus.IDLE):
                await asyncio.sleep(60)
                if self._last_activity_time:
                    delta = (datetime.now() - self._last_activity_time).total_seconds() / 60
                    if delta > self.idle_timeout_minutes:
                        logger.info(f'K8s kernel {self._pod_name} idle for {delta:.1f}m. Shutting down.')
                        async with _get_pool_lock():
                            pool = _kernel_pool.get(self.client_id, [])
                            if self in pool:
                                pool.remove(self)
                        await self.stop()
                        break
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f'_monitor_idle crashed for {self._pod_name}: {exc}', exc_info=True)

    async def _port_forward_keepalive(self) -> None:
        try:
            while self._status in (KernelStatus.RUNNING, KernelStatus.IDLE):
                await asyncio.sleep(30)
                if self._in_cluster or self._port_forward_proc is None:
                    break
                if self._port_forward_proc.poll() is not None:
                    exit_code = self._port_forward_proc.returncode
                    logger.warning(f'Port-forward for {self._pod_name} died (exit {exit_code}). Restarting...')
                    self._stop_port_forward()
                    try:
                        await self._start_port_forward()
                        self._is_alive_cache_time = 0.0
                        logger.info(f'Port-forward for {self._pod_name} restarted successfully.')
                    except Exception as restart_exc:
                        logger.error(f'Failed to restart port-forward for {self._pod_name}: {restart_exc}. Marking kernel ERROR.')
                        self._status = KernelStatus.ERROR
                        break
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f'Port-forward keepalive crashed for {self._pod_name}: {exc}', exc_info=True)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()