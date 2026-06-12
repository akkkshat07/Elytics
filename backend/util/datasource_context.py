from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
logger = logging.getLogger(__name__)
CLIENTS_PROMPTS_PATH = Path('xml_prompts/clients')
_VARIANT_PATTERN = re.compile('^(?P<business_unit>[a-z0-9-]+)_(?P<system>[a-z0-9-]+)$')
_RESERVED_METADATA_DIRS = {'data_descriptions', 'meta_information'}
_RESERVED_CLIENT_ROOT_DIRS = {'agents', 'domain_knowledge', 'data_sources', 'data_descriptions', 'meta_information'}

def _slugify_component(value: Any) -> str:
    if value is None:
        return ''
    normalized = re.sub('[^a-z0-9]+', '-', str(value).strip().lower())
    return normalized.strip('-')

def _format_label(value: str) -> str:
    if not value:
        return ''
    parts = []
    for part in value.replace('-', ' ').split():
        if len(part) <= 3 and part.isalpha():
            parts.append(part.upper())
        else:
            parts.append(part.capitalize())
    return ' '.join(parts)

def _normalize_datasource_key(value: Any) -> str:
    text = str(value or '').strip()
    if not text:
        return ''
    if '_' in text:
        left, right = text.split('_', 1)
        return build_datasource_key(left, right)
    return text if _VARIANT_PATTERN.match(text) else _slugify_component(text)

def build_datasource_key(business_unit: Any, system: Any) -> str:
    business_unit_key = _slugify_component(business_unit)
    system_key = _slugify_component(system)
    if not business_unit_key or not system_key:
        return ''
    return f'{business_unit_key}_{system_key}'

def normalize_datasource_context(context: Optional[Any], *, client_id: Optional[str]=None, allow_unavailable: bool=True, fallback_to_default: bool=True, clients_prompts_path: Optional[Path]=None) -> Optional[Dict[str, Any]]:
    raw_key = ''
    raw_business_unit = ''
    raw_system = ''
    if isinstance(context, str):
        raw_key = _normalize_datasource_key(context)
    elif isinstance(context, dict):
        raw_key = _normalize_datasource_key(context.get('datasource_key', ''))
        raw_business_unit = _slugify_component(context.get('business_unit', ''))
        raw_system = _slugify_component(context.get('system', ''))
    if not raw_key and raw_business_unit and raw_system:
        raw_key = build_datasource_key(raw_business_unit, raw_system)
    elif raw_key and (not raw_business_unit or not raw_system):
        match = _VARIANT_PATTERN.match(raw_key)
        if match:
            raw_business_unit = match.group('business_unit')
            raw_system = match.group('system')
    if raw_key and raw_business_unit and raw_system:
        return {'datasource_key': raw_key, 'business_unit': raw_business_unit, 'business_unit_label': _format_label(raw_business_unit), 'system': raw_system, 'system_label': _format_label(raw_system), 'available': True}
    return None

def get_datasource_namespace_suffix(context: Optional[Dict[str, Any]]) -> Optional[str]:
    normalized = normalize_datasource_context(context, allow_unavailable=True, fallback_to_default=False)
    if not normalized:
        return None
    return normalized.get('datasource_key') or None

def get_client_metadata_storage_root(client_id: str, *, datasource_context: Optional[Dict[str, Any]]=None, clients_prompts_path: Optional[Path]=None) -> Path:
    base_path = clients_prompts_path or CLIENTS_PROMPTS_PATH
    client_root = base_path / client_id
    normalized_context = normalize_datasource_context(datasource_context, client_id=client_id, allow_unavailable=True, fallback_to_default=False, clients_prompts_path=clients_prompts_path)
    if normalized_context:
        ds_key = normalized_context['datasource_key']
        new_path = client_root / ds_key / 'data_sources'
        old_path = client_root / 'data_sources' / ds_key
        if new_path.exists():
            return new_path
        if old_path.exists():
            return old_path
        return new_path
    return client_root / 'data_sources'

def resolve_client_metadata_root(client_id: str, *, datasource_context: Optional[Dict[str, Any]]=None, allow_legacy_when_context_missing: bool=True, clients_prompts_path: Optional[Path]=None) -> Optional[Path]:
    candidate_root = get_client_metadata_storage_root(client_id, datasource_context=datasource_context, clients_prompts_path=clients_prompts_path)
    normalized_context = normalize_datasource_context(datasource_context, client_id=client_id, allow_unavailable=True, fallback_to_default=False, clients_prompts_path=clients_prompts_path)
    if normalized_context:
        if candidate_root.exists():
            return candidate_root
        return None
    if allow_legacy_when_context_missing and candidate_root.exists():
        return candidate_root
    return None

def resolve_client_metadata_path(client_id: str, relative_parts: Iterable[str] | str, *, datasource_context: Optional[Dict[str, Any]]=None, allow_legacy_when_context_missing: bool=True, clients_prompts_path: Optional[Path]=None) -> Optional[Path]:
    metadata_root = resolve_client_metadata_root(client_id, datasource_context=datasource_context, allow_legacy_when_context_missing=allow_legacy_when_context_missing, clients_prompts_path=clients_prompts_path)
    if not metadata_root:
        return None
    if isinstance(relative_parts, str):
        candidate = metadata_root / relative_parts
    else:
        candidate = metadata_root.joinpath(*relative_parts)
    return candidate if candidate.exists() else None

def resolve_client_metadata_dir(client_id: str, relative_parts: Iterable[str] | str, *, datasource_context: Optional[Dict[str, Any]]=None, allow_legacy_when_context_missing: bool=True, clients_prompts_path: Optional[Path]=None) -> Optional[Path]:
    candidate = resolve_client_metadata_path(client_id, relative_parts, datasource_context=datasource_context, allow_legacy_when_context_missing=allow_legacy_when_context_missing, clients_prompts_path=clients_prompts_path)
    if candidate and candidate.is_dir():
        return candidate
    return None