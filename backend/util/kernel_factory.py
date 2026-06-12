import logging
from typing import Any
from config.system_config import KERNEL_BACKEND
logger = logging.getLogger(__name__)

def create_kernel_manager(client_id: str, **kwargs: Any) -> Any:
    kwargs.pop('use_docker', None)
    if KERNEL_BACKEND == 'local':
        from util.kernel_manager import LocalKernelManager
        return LocalKernelManager(client_id=client_id, **kwargs)
    elif KERNEL_BACKEND == 'kubernetes':
        from util.k8s_kernel_manager import K8sKernelManager
        return K8sKernelManager(client_id=client_id, **kwargs)
    else:
        from util.kernel_manager import DockerKernelManager
        return DockerKernelManager(client_id=client_id, **kwargs)