from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

def _legacy_data_sources_dir(client_id: str) -> Path:
    return Path('xml_prompts/clients') / client_id / 'data_sources'

def _scoped_data_sources_dir(client_id: str, dataset_id: str) -> Path:
    return _legacy_data_sources_dir(client_id) / dataset_id

def resolve_xml_data_sources_dir(client_id: str, dataset_id: Optional[str], for_write: bool=False) -> Path:
    legacy = _legacy_data_sources_dir(client_id)
    if dataset_id:
        scoped = _scoped_data_sources_dir(client_id, dataset_id)
        if for_write:
            return scoped
        intro_scoped = scoped / 'meta_information' / 'table_introductions.xml'
        intro_legacy = legacy / 'meta_information' / 'table_introductions.xml'
        if intro_scoped.exists():
            return scoped
        if intro_legacy.exists():
            return legacy
        return scoped
    return legacy

def assets_datasets_dir(client_id: str, dataset_id: Optional[str]) -> Path:
    base = Path(f'assets/clients/{client_id}/datasets')
    if dataset_id:
        return base / dataset_id
    return base

def assets_uploads_dir(client_id: str, dataset_id: Optional[str]) -> Path:
    base = Path(f'assets/clients/{client_id}/uploads')
    if dataset_id:
        return base / dataset_id
    return base

def response_caching_dir(client_id: str, dataset_id: Optional[str]) -> Path:
    base = Path('assets/clients') / client_id / 'response_caching'
    if dataset_id:
        return base / dataset_id
    return base

def storage_datasets_prefix(client_id: str, dataset_id: Optional[str]=None) -> str:
    from util.storage.backend import get_storage_backend, LocalStorageBackend
    backend = get_storage_backend()
    base = f'clients/{client_id}/datasets'
    if dataset_id:
        base = f'{base}/{dataset_id}'
    if isinstance(backend, LocalStorageBackend):
        return f'assets/{base}'
    return base

def storage_uploads_prefix(client_id: str, dataset_id: Optional[str]=None) -> str:
    from util.storage.backend import get_storage_backend, LocalStorageBackend
    backend = get_storage_backend()
    base = f'clients/{client_id}/uploads'
    if dataset_id:
        base = f'{base}/{dataset_id}'
    if isinstance(backend, LocalStorageBackend):
        return f'assets/{base}'
    return base

def storage_xml_data_sources_prefix(client_id: str, dataset_id: Optional[str]=None) -> str:
    from util.storage.backend import get_storage_backend, LocalStorageBackend
    backend = get_storage_backend()
    if isinstance(backend, LocalStorageBackend):
        base = f'xml_prompts/clients/{client_id}/data_sources'
    else:
        base = f'clients/{client_id}/xml_prompts/data_sources'
    if dataset_id:
        base = f'{base}/{dataset_id}'
    return base

def storage_xml_agents_prefix(client_id: str) -> str:
    from util.storage.backend import get_storage_backend, LocalStorageBackend
    backend = get_storage_backend()
    if isinstance(backend, LocalStorageBackend):
        return f'xml_prompts/clients/{client_id}/agents'
    return f'clients/{client_id}/xml_prompts/agents'

def storage_xml_domain_knowledge_prefix(client_id: str) -> str:
    from util.storage.backend import get_storage_backend, LocalStorageBackend
    backend = get_storage_backend()
    if isinstance(backend, LocalStorageBackend):
        return f'xml_prompts/clients/{client_id}/domain_knowledge'
    return f'clients/{client_id}/xml_prompts/domain_knowledge'

def storage_response_caching_prefix(client_id: str, dataset_id: Optional[str]=None) -> str:
    from util.storage.backend import get_storage_backend, LocalStorageBackend
    backend = get_storage_backend()
    base = f'clients/{client_id}/response_caching'
    if dataset_id:
        base = f'{base}/{dataset_id}'
    if isinstance(backend, LocalStorageBackend):
        return f'assets/{base}'
    return base

def storage_adhoc_uploads_prefix(client_id: str, session_id: Optional[str]=None) -> str:
    from util.storage.backend import get_storage_backend, LocalStorageBackend
    backend = get_storage_backend()
    base = f'clients/{client_id}/adhoc_uploads'
    if session_id:
        base = f'{base}/{session_id}'
    if isinstance(backend, LocalStorageBackend):
        return f'assets/{base}'
    return base

def storage_client_prefix(client_id: str) -> str:
    from util.storage.backend import get_storage_backend, LocalStorageBackend
    backend = get_storage_backend()
    if isinstance(backend, LocalStorageBackend):
        return f'assets/clients/{client_id}'
    return f'clients/{client_id}'

def effective_dataset_id(credentials_doc: Optional[Dict[str, Any]]) -> Optional[str]:
    if not credentials_doc:
        return None
    did = credentials_doc.get('dataset_id')
    if did is None or did == '':
        return None
    return str(did)

def normalize_credential_dict(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(doc)
    if 'dataset_id' not in out or not out['dataset_id']:
        out['dataset_id'] = None
    if 'dataset_name' not in out:
        out['dataset_name'] = out.get('db_name') or out.get('db_type') or 'Dataset'
    if 'is_enabled' not in out:
        out['is_enabled'] = True
    if 'display_order' not in out:
        out['display_order'] = 0
    return out