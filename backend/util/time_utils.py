from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Optional
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def parse_client_timezone(tz_name: Optional[str]) -> Any:
    if not tz_name:
        return timezone.utc
    tz_name = tz_name.strip()
    if not tz_name or tz_name.upper() == 'UTC':
        return timezone.utc
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return timezone.utc

def parse_iso_datetime(value: str) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace('Z', '+00:00'))
    except Exception:
        return None
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def format_datetime_for_tz(dt: datetime, tz: Any) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(tz).replace(microsecond=0)
    return dt.isoformat()