from __future__ import annotations
from copy import deepcopy
from threading import Lock
from typing import Any, Dict
_REGISTRY: Dict[str, Dict[str, Any]] = {}
_LOCK = Lock()
_MERGE_DICT_KEYS = {'agent_inputs', 'agent_token_usage'}

def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any((_has_content(item) for item in value.values()))
    if isinstance(value, (list, tuple, set)):
        return any((_has_content(item) for item in value))
    if isinstance(value, bool):
        return value
    return bool(value)

def record_graph_final_payload(run_id: str, payload: Dict[str, Any]) -> None:
    if not run_id or not isinstance(payload, dict):
        return
    with _LOCK:
        existing = dict(_REGISTRY.get(run_id) or {})
        for key, value in payload.items():
            if key in _MERGE_DICT_KEYS and isinstance(value, dict):
                merged = dict(existing.get(key) or {})
                merged.update(deepcopy(value))
                existing[key] = merged
                continue
            if _has_content(value) or key not in existing:
                existing[key] = deepcopy(value)
        _REGISTRY[run_id] = existing

def peek_graph_final_payload(run_id: str) -> Dict[str, Any]:
    if not run_id:
        return {}
    with _LOCK:
        return deepcopy(_REGISTRY.get(run_id) or {})

def pop_graph_final_payload(run_id: str) -> Dict[str, Any]:
    if not run_id:
        return {}
    with _LOCK:
        return deepcopy(_REGISTRY.pop(run_id, {}) or {})

def discard_graph_final_payload(run_id: str) -> None:
    if not run_id:
        return
    with _LOCK:
        _REGISTRY.pop(run_id, None)