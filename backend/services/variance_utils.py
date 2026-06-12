from __future__ import annotations
import asyncio
from datetime import date, timedelta
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

def _bm_start(business_month: str) -> date:
    year, month = (int(x) for x in business_month.split('-'))
    if month == 1:
        return date(year - 1, 12, 26)
    return date(year, month - 1, 26)

def _prev_business_month(business_month: str) -> str:
    year, month = (int(x) for x in business_month.split('-'))
    if month == 1:
        return f'{year - 1}-12'
    return f'{year}-{month - 1:02d}'

def _resolve_prev_month_report_date(business_month: str, report_date: str) -> Tuple[str, str]:
    rd = date.fromisoformat(report_date)
    current_start = _bm_start(business_month)
    day_offset = (rd - current_start).days
    prev_bm = _prev_business_month(business_month)
    prev_start = _bm_start(prev_bm)
    prev_rd = prev_start + timedelta(days=max(day_offset, 0))
    return (prev_bm, prev_rd.isoformat())

def _resolve_dates(business_month: str, report_date: Optional[str]) -> Dict[str, Any]:
    current_rd = report_date or date.today().isoformat()
    bm_start = _bm_start(business_month)
    rd = date.fromisoformat(current_rd)
    day_offset = (rd - bm_start).days
    first_day = day_offset <= 0
    prev_day_rd = None if first_day else (rd - timedelta(days=1)).isoformat()
    prev_bm, prev_month_rd = _resolve_prev_month_report_date(business_month, current_rd)
    return {'current_rd': current_rd, 'prev_day_rd': prev_day_rd, 'prev_bm': prev_bm, 'prev_month_rd': prev_month_rd, 'first_day_of_bm': first_day}

def _growth_pct(curr: Any, prev: Any) -> Optional[float]:
    try:
        c = float(curr or 0)
        p = float(prev or 0)
    except (TypeError, ValueError):
        return None
    if p == 0:
        return None
    return round((c - p) / abs(p) * 100, 1)

def _is_numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and (not isinstance(v, bool))

def _augment_kpi_dict(current_kpis: Dict[str, Any], prev_day_kpis: Optional[Dict[str, Any]], prev_month_kpis: Optional[Dict[str, Any]]) -> None:
    numeric_keys = [k for k, v in list(current_kpis.items()) if _is_numeric(v)]
    for k in numeric_keys:
        curr_v = current_kpis[k]
        if prev_day_kpis is not None:
            pd_v = prev_day_kpis.get(k)
            current_kpis[f'prev_day_{k}'] = pd_v if _is_numeric(pd_v) else None
            current_kpis[f'prev_day_delta_pct_{k}'] = _growth_pct(curr_v, pd_v)
        else:
            current_kpis[f'prev_day_{k}'] = None
            current_kpis[f'prev_day_delta_pct_{k}'] = None
        if prev_month_kpis is not None:
            pm_v = prev_month_kpis.get(k)
            current_kpis[f'prev_month_{k}'] = pm_v if _is_numeric(pm_v) else None
            current_kpis[f'prev_month_delta_pct_{k}'] = _growth_pct(curr_v, pm_v)
        else:
            current_kpis[f'prev_month_{k}'] = None
            current_kpis[f'prev_month_delta_pct_{k}'] = None

def _index_rows(rows: List[Dict[str, Any]], key_cols: Iterable[str]) -> Dict[Tuple, Dict[str, Any]]:
    keys = tuple(key_cols)
    return {tuple((r.get(c) for c in keys)): r for r in rows}

def _augment_rows(current_rows: List[Dict[str, Any]], prev_day_rows: Optional[List[Dict[str, Any]]], prev_month_rows: Optional[List[Dict[str, Any]]], key_cols: Iterable[str]) -> None:
    pd_idx = _index_rows(prev_day_rows, key_cols) if prev_day_rows else {}
    pm_idx = _index_rows(prev_month_rows, key_cols) if prev_month_rows else {}
    keys = tuple(key_cols)
    for row in current_rows:
        rk = tuple((row.get(c) for c in keys))
        pd_row = pd_idx.get(rk) or {}
        pm_row = pm_idx.get(rk) or {}
        numeric_keys = [k for k, v in list(row.items()) if _is_numeric(v) and k not in keys]
        for k in numeric_keys:
            if prev_day_rows is not None:
                row[f'prev_day_delta_pct_{k}'] = _growth_pct(row[k], pd_row.get(k))
            else:
                row[f'prev_day_delta_pct_{k}'] = None
            if prev_month_rows is not None:
                row[f'prev_month_delta_pct_{k}'] = _growth_pct(row[k], pm_row.get(k))
            else:
                row[f'prev_month_delta_pct_{k}'] = None

def _build_meta(dates: Dict[str, Any], business_month: str) -> Dict[str, Any]:
    return {'current_business_month': business_month, 'current_report_date': dates['current_rd'], 'prev_day_report_date': dates['prev_day_rd'], 'prev_business_month': dates['prev_bm'], 'prev_month_report_date': dates['prev_month_rd'], 'first_day_of_bm': dates['first_day_of_bm']}

async def compute_with_variance(fetch_async: Callable[..., Awaitable[Dict[str, Any]]], *, segments: List[str], business_month: str, report_date: Optional[str], extra_kwargs: Optional[Dict[str, Any]]=None, merge: str='kpi', rows_key: str='data', kpi_key: str='kpis', row_key_cols: Iterable[str]=()) -> Dict[str, Any]:
    extra = extra_kwargs or {}
    dates = _resolve_dates(business_month, report_date)
    current_t = fetch_async(segments, business_month, report_date=dates['current_rd'], **extra)
    prev_month_t = fetch_async(segments, dates['prev_bm'], report_date=dates['prev_month_rd'], **extra)
    if dates['prev_day_rd']:
        prev_day_t = fetch_async(segments, business_month, report_date=dates['prev_day_rd'], **extra)
        current, prev_day, prev_mo = await asyncio.gather(current_t, prev_day_t, prev_month_t)
    else:
        current, prev_mo = await asyncio.gather(current_t, prev_month_t)
        prev_day = None
    _apply_merge(current, prev_day, prev_mo, merge, kpi_key, rows_key, row_key_cols)
    current.setdefault('meta', {})['variance'] = _build_meta(dates, business_month)
    return current

async def compute_with_month_variance(fetch_async: Callable[..., Awaitable[Dict[str, Any]]], *, segments: List[str], business_month: str, extra_kwargs: Optional[Dict[str, Any]]=None, merge: str='kpi', rows_key: str='data', kpi_key: str='kpis', row_key_cols: Iterable[str]=()) -> Dict[str, Any]:
    extra = extra_kwargs or {}
    prev_bm = _prev_business_month(business_month)
    current, prev_mo = await asyncio.gather(fetch_async(segments, business_month, **extra), fetch_async(segments, prev_bm, **extra))
    _apply_merge(current, None, prev_mo, merge, kpi_key, rows_key, row_key_cols)
    current.setdefault('meta', {})['variance'] = {'current_business_month': business_month, 'prev_business_month': prev_bm, 'prev_day_report_date': None, 'first_day_of_bm': False}
    return current

def _apply_merge(current: Dict[str, Any], prev_day: Optional[Dict[str, Any]], prev_mo: Optional[Dict[str, Any]], merge: str, kpi_key: str, rows_key: str, row_key_cols: Iterable[str]) -> None:
    if merge == 'kpi':
        curr_kpis = current.get(kpi_key) or {}
        pd_kpis = (prev_day or {}).get(kpi_key) if prev_day else None
        pm_kpis = (prev_mo or {}).get(kpi_key) if prev_mo else None
        _augment_kpi_dict(curr_kpis, pd_kpis, pm_kpis)
        current[kpi_key] = curr_kpis
    elif merge == 'rows':
        curr_rows = current.get(rows_key) or []
        pd_rows = (prev_day or {}).get(rows_key) if prev_day else None
        pm_rows = (prev_mo or {}).get(rows_key) if prev_mo else None
        _augment_rows(curr_rows, pd_rows, pm_rows, row_key_cols)
        current[rows_key] = curr_rows
    else:
        raise ValueError(f'Unknown merge mode: {merge!r}')