import os
import logging
from pathlib import Path
from typing import Optional
logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_client_cache_dir(client_id: str, dataset_id: Optional[str]=None) -> Path:
    project_root = get_project_root()
    base = project_root / 'assets' / 'clients' / client_id / 'response_caching'
    if dataset_id:
        return base / dataset_id
    return base

def get_client_db_collection_name(client_id: str) -> str:
    safe_id = ''.join((c if c.isalnum() else '_' for c in client_id))
    return f'responses_{safe_id}'

def get_client_vector_db_path(client_id: str) -> str:
    project_root = get_project_root()
    return str(project_root / 'assets' / 'data' / 'vector_db' / client_id)

def ensure_client_cache_infrastructure(client_id: str, dataset_id: Optional[str]=None) -> bool:
    try:
        cache_dir = get_client_cache_dir(client_id, dataset_id)
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created cache directory for client '{client_id}': {cache_dir}")
        else:
            logger.debug(f"Cache directory already exists for client '{client_id}': {cache_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to ensure infrastructure for client '{client_id}': {e}")
        return False