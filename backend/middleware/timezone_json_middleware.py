from __future__ import annotations
import json
import math
from typing import Any, Mapping
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from util.time_utils import format_datetime_for_tz, parse_client_timezone, parse_iso_datetime
_TZ_HEADER_CANDIDATES = ('X-Client-Timezone', 'X-Timezone', 'Timezone')

def _get_client_tz(request: Request) -> Any:
    tz_name = request.cookies.get('client_timezone')
    if not tz_name:
        for h in _TZ_HEADER_CANDIDATES:
            tz_name = request.headers.get(h)
            if tz_name:
                break
    if not tz_name:
        tz_name = request.query_params.get('tz') or request.query_params.get('timezone')
    return parse_client_timezone(tz_name)

def _key_looks_like_timestamp(key: str) -> bool:
    k = key.lower()
    return k.endswith('_at') or k.endswith('_time') or k.endswith('_timestamp') or (k in {'timestamp', 'generated_at', 'expires_at', 'validated_at', 'deleted_at'}) or ('timestamp' in k)

def _convert_timestamps(obj: Any, tz: Any, parent_key: str | None=None) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str):
                out[k] = _convert_timestamps(v, tz, parent_key=k)
            else:
                out[k] = _convert_timestamps(v, tz, parent_key=parent_key)
        return out
    if isinstance(obj, list):
        return [_convert_timestamps(v, tz, parent_key=parent_key) for v in obj]
    if isinstance(obj, str) and parent_key and _key_looks_like_timestamp(parent_key):
        dt = parse_iso_datetime(obj)
        if dt is None:
            return obj
        return format_datetime_for_tz(dt, tz)
    return obj

async def timezone_json_middleware(request: Request, call_next):
    tz = _get_client_tz(request)
    response = await call_next(request)
    if isinstance(response, StreamingResponse):
        return response
    content_type = (response.headers.get('content-type') or '').lower()
    if 'application/json' not in content_type:
        return response
    body = b''
    async for chunk in response.body_iterator:
        body += chunk
    if not body:
        return response
    try:
        payload = json.loads(body)
    except Exception:
        return Response(content=body, status_code=response.status_code, headers=dict(response.headers), media_type=response.media_type, background=response.background)
    converted = _convert_timestamps(payload, tz)
    new_body = json.dumps(converted, ensure_ascii=False).encode('utf-8')
    headers: dict[str, str] = dict(response.headers)
    headers.pop('content-length', None)
    return Response(content=new_body, status_code=response.status_code, headers=headers, media_type='application/json', background=response.background)