from __future__ import annotations
from typing import Any, Dict, List, Tuple
EXTRA_TOKEN_KEYS = ['reasoning_tokens', 'cached_input_tokens', 'cache_creation_input_tokens', 'audio_input_tokens', 'audio_output_tokens', 'image_input_tokens', 'accepted_prediction_tokens', 'rejected_prediction_tokens', 'text_input_tokens', 'text_output_tokens']

def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

def normalize_agent_token_usage(agent_token_usage: Dict[str, Any] | None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    agent_token_usage = agent_token_usage or {}
    normalized_usage: Dict[str, Any] = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_provider_tokens = 0
    extra_totals: Dict[str, int] = {key: 0 for key in EXTRA_TOKEN_KEYS}
    for agent_name, usage in agent_token_usage.items():
        if not usage:
            continue
        if not isinstance(usage, dict):
            normalized_usage[agent_name] = usage
            continue
        prompt_tokens = _safe_int(usage.get('prompt_tokens') or usage.get('input_tokens') or usage.get('prompt_token_count'))
        completion_tokens = _safe_int(usage.get('completion_tokens') or usage.get('output_tokens') or usage.get('candidates_token_count'))
        total_tokens_provider = _safe_int(usage.get('total_tokens_provider') or usage.get('total_tokens') or usage.get('total_token_count'))
        total_tokens_derived = prompt_tokens + completion_tokens
        normalized = dict(usage)
        normalized['prompt_tokens'] = prompt_tokens
        normalized['completion_tokens'] = completion_tokens
        normalized['total_tokens'] = total_tokens_derived
        if total_tokens_provider:
            normalized['total_tokens_provider'] = total_tokens_provider
            total_provider_tokens += total_tokens_provider
        for key in EXTRA_TOKEN_KEYS:
            value = _safe_int(normalized.get(key))
            if value:
                normalized[key] = value
                extra_totals[key] += value
        normalized_usage[agent_name] = normalized
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
    total_token_usage: Dict[str, Any] = {'prompt_tokens': total_prompt_tokens, 'completion_tokens': total_completion_tokens, 'total_tokens': total_prompt_tokens + total_completion_tokens}
    if total_provider_tokens:
        total_token_usage['total_tokens_provider'] = total_provider_tokens
    for key, value in extra_totals.items():
        if value:
            total_token_usage[key] = value
    return (normalized_usage, total_token_usage)

def usage_event_step_prompt_completion_totals(ev: Dict[str, Any]) -> Tuple[int, int, int]:
    tu = ev.get('total_token_usage') if isinstance(ev.get('total_token_usage'), dict) else {}
    pt = _safe_int(tu.get('prompt_tokens'))
    ct = _safe_int(tu.get('completion_tokens'))
    tot_direct = _safe_int(tu.get('total_tokens')) or pt + ct
    if tot_direct > 0 or pt > 0 or ct > 0:
        total = tot_direct if tot_direct > 0 else pt + ct
        return (total, pt, ct)
    _norm_agents, tu2 = normalize_agent_token_usage(ev.get('agent_token_usage'))
    p2 = _safe_int(tu2.get('prompt_tokens'))
    c2 = _safe_int(tu2.get('completion_tokens'))
    t2 = _safe_int(tu2.get('total_tokens')) or p2 + c2
    return (t2, p2, c2)

def aggregate_dashboard_report_usage_totals(events: List[Dict[str, Any]] | None) -> Dict[str, Any]:
    evs = [e for e in events or [] if isinstance(e, dict)]
    prompt_sum = 0
    completion_sum = 0
    display_total_sum = 0
    for ev in evs:
        t, p, c = usage_event_step_prompt_completion_totals(ev)
        prompt_sum += p
        completion_sum += c
        display_total_sum += t if t > 0 else p + c
    return {'total_tokens': display_total_sum, 'prompt_tokens': prompt_sum, 'completion_tokens': completion_sum, 'usage_event_count': len(evs)}