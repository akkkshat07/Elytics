from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from services import dsr_service, lead_indicators_service
from services.report_db_utils import build_segment_queries, _engine_for, excluded_designations_clause, master_table, run_query
from services.variance_utils import compute_with_variance, compute_with_month_variance, _prev_business_month
logger = logging.getLogger(__name__)
_redis = None
_redis_init_attempted = False

def _get_redis():
    global _redis, _redis_init_attempted
    if _redis_init_attempted:
        return _redis
    _redis_init_attempted = True
    try:
        url = os.getenv('REDIS_URL')
        if not url:
            return None
        import redis.asyncio as aioredis
        _redis = aioredis.from_url(url, decode_responses=True)
    except Exception as e:
        logger.debug('ai_insights: Redis unavailable: %s', e)
        _redis = None
    return _redis

async def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    redis = _get_redis()
    if redis is None:
        return None
    try:
        raw = await redis.get(key)
        if raw:
            logger.info('cache_get | HIT | key=%s', key)
            return json.loads(raw)
        logger.info('cache_get | MISS | key=%s', key)
        return None
    except Exception:
        return None

async def _cache_set(key: str, value: Dict[str, Any], ttl: int) -> None:
    redis = _get_redis()
    if redis is None:
        return
    try:
        await redis.setex(key, ttl, json.dumps(value, default=str))
    except Exception:
        pass

def _get_rep_silence_sync(segments: List[str], business_month: str, limit: int=50) -> List[Dict[str, Any]]:
    t0 = time.monotonic()
    logger.info('get_rep_silence | bm=%s segments=%s limit=%d', business_month, segments, limit)
    seg_queries = build_segment_queries(segments)
    prev_bm = _prev_business_month(business_month)
    excl = excluded_designations_clause()
    frames = []
    for db_name, seg_filter in seg_queries:
        engine = _engine_for(db_name)
        tbl = master_table(db_name)
        sql = f"\n            SELECT p.uuid, p.emp_name, p.emp_zone as zone, p.emp_state as state,\n                   p.rsm_name as team_leader, p.designation,\n                   p.mtd_p3_nsv as prev_p3\n            FROM {tbl} p\n            LEFT JOIN {tbl} c ON p.uuid = c.uuid\n              AND c.record_type = 'EMPLOYEE' AND c.business_month = :bm\n            WHERE p.record_type = 'EMPLOYEE' AND p.business_month = :prev_bm\n              AND p.designation NOT IN ('NSM', 'SFA') {(seg_filter.replace('segment', 'p.segment') if seg_filter else '')}\n              AND COALESCE(p.mtd_p3_nsv, 0) > 0\n              AND (c.uuid IS NULL OR COALESCE(c.mtd_p3_nsv, 0) = 0)\n            ORDER BY p.mtd_p3_nsv DESC\n            LIMIT :limit\n        "
        try:
            df = run_query(engine, sql, {'bm': business_month, 'prev_bm': prev_bm, 'limit': limit})
            if not df.empty:
                frames.append(df)
        except Exception:
            logger.exception('Rep silence query failed for %s', db_name)
    if not frames:
        logger.warning('get_rep_silence | no silent reps for bm=%s', business_month)
        return []
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('prev_p3', ascending=False).head(limit)
    logger.info('get_rep_silence | done | %d silent reps | %.2fs', len(combined), time.monotonic() - t0)
    return combined.fillna('').to_dict(orient='records')

async def get_rep_silence(segments: List[str], business_month: str, limit: int=50):
    return await asyncio.to_thread(_get_rep_silence_sync, segments, business_month, limit)

def _fmt_inr(value: Any) -> str:
    try:
        v = float(value or 0)
    except (TypeError, ValueError):
        return '--'
    if v >= 10000000.0:
        return f'₹{v / 10000000.0:.2f}Cr'
    if v >= 100000.0:
        return f'₹{v / 100000.0:.2f}L'
    if v >= 1000.0:
        return f'₹{v / 1000.0:.1f}K'
    return f'₹{v:.0f}'

def _severity_for(delta: Optional[float]) -> str:
    if delta is None:
        return 'neutral'
    if delta > 0.5:
        return 'positive'
    if delta < -0.5:
        return 'negative'
    return 'neutral'

def _card(card_id: str, kind: str, severity: str, icon: str, headline: str, sub: str, metric: Dict[str, Any], drill: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    return {'id': card_id, 'kind': kind, 'severity': severity, 'icon': icon, 'headline': headline, 'sub': sub, 'metric': metric, 'drill': drill}

async def _gather_ranked_data(segments: List[str], business_month: str, report_date: Optional[str]) -> Dict[str, Any]:
    t0 = time.monotonic()
    logger.info('gather_ranked_data | bm=%s segments=%s report_date=%s', business_month, segments, report_date)
    summary_t = compute_with_variance(dsr_service.get_summary, segments=segments, business_month=business_month, report_date=report_date, merge='kpi', kpi_key='kpis')
    rsm_t = compute_with_variance(dsr_service.get_rsm_summary, segments=segments, business_month=business_month, report_date=report_date, merge='rows', rows_key='data', row_key_cols=('rsm_name', 'zone'))
    state_t = compute_with_month_variance(lead_indicators_service.get_state_wise, segments=segments, business_month=business_month, merge='rows', rows_key='data', row_key_cols=('state', 'state_group'))
    summary, rsm, state = await asyncio.gather(summary_t, rsm_t, state_t)
    logger.info('gather_ranked_data | done | %.2fs', time.monotonic() - t0)
    return {'summary': summary, 'rsm': rsm, 'state': state}

def _rank_rows(rows: List[Dict[str, Any]], metric_col: str, value_col: str, n: int=5):
    have_delta = [r for r in rows if r.get(metric_col) is not None]
    sorted_rows = sorted(have_delta, key=lambda r: r[metric_col], reverse=True)
    top = [r for r in sorted_rows if r[metric_col] > 0][:n]
    bottom = list(reversed([r for r in sorted_rows if r[metric_col] < 0][-n:]))
    return (top, bottom)

def _build_ribbon(data: Dict[str, Any], silence_count: int) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    summary = data.get('summary', {}) or {}
    kpis = summary.get('kpis', {}) or {}
    meta = (summary.get('meta') or {}).get('variance', {}) or {}
    p3 = kpis.get('total_p3_nsv')
    p3_lm = kpis.get('prev_month_total_p3_nsv')
    p3_d = kpis.get('prev_month_delta_pct_total_p3_nsv')
    if p3 is not None:
        cards.append(_card('headline_p3', kind='highlight', severity=_severity_for(p3_d), icon='trending-up' if (p3_d or 0) >= 0 else 'trending-down', headline=f'P3 NSV at {_fmt_inr(p3)}', sub=f"{('+' if (p3_d or 0) >= 0 else '')}{p3_d}% vs {_fmt_inr(p3_lm)} last month" if p3_d is not None else 'No comparable last month', metric={'value': p3_d if p3_d is not None else 0, 'unit': 'pct', 'direction': 'up' if (p3_d or 0) >= 0 else 'down'}, drill={'tab': 'overview', 'filter': {}}))
    state_rows = (data.get('state') or {}).get('data') or []
    top_states, bottom_states = _rank_rows(state_rows, 'prev_month_delta_pct_mtd_p3_nsv', 'mtd_p3_nsv', n=3)
    if top_states:
        s = top_states[0]
        d = s.get('prev_month_delta_pct_mtd_p3_nsv')
        cards.append(_card('top_state', kind='highlight', severity='positive', icon='trending-up', headline=f"{s.get('state', '—')} surging +{d}%", sub=f"{_fmt_inr(s.get('mtd_p3_nsv'))} P3 NSV this BM", metric={'value': d, 'unit': 'pct', 'direction': 'up'}, drill={'tab': 'state-wise', 'filter': {'state': s.get('state')}}))
    if bottom_states:
        s = bottom_states[0]
        d = s.get('prev_month_delta_pct_mtd_p3_nsv')
        cards.append(_card('bottom_state', kind='warning', severity='negative', icon='trending-down', headline=f"{s.get('state', '—')} down {d}%", sub=f"{_fmt_inr(s.get('mtd_p3_nsv'))} P3 NSV this BM", metric={'value': d, 'unit': 'pct', 'direction': 'down'}, drill={'tab': 'state-wise', 'filter': {'state': s.get('state')}}))
    rsm_rows = (data.get('rsm') or {}).get('data') or []
    top_rsm, bottom_rsm = _rank_rows(rsm_rows, 'prev_month_delta_pct_p3_nsv', 'p3_nsv', n=3)
    if top_rsm:
        r = top_rsm[0]
        d = r.get('prev_month_delta_pct_p3_nsv')
        cards.append(_card('top_rsm', kind='highlight', severity='positive', icon='award', headline=f"RSM {r.get('rsm_name', '—')} up +{d}%", sub=f"{_fmt_inr(r.get('p3_nsv'))} P3 NSV  •  {r.get('zone', '—')}", metric={'value': d, 'unit': 'pct', 'direction': 'up'}, drill={'tab': 'detailed', 'filter': {'rsm': r.get('rsm_name')}}))
    if silence_count > 0:
        cards.append(_card('rep_silence', kind='churn', severity='negative', icon='alert-triangle', headline=f'{silence_count} reps went silent', sub='Billed P3 last month, zero this month', metric={'value': silence_count, 'unit': 'count', 'direction': 'down'}, drill={'tab': 'insights', 'filter': {'section': 'silence'}}))
    return cards[:5]

def _build_full(data: Dict[str, Any], silence_rows: List[Dict[str, Any]], narrative: bool) -> Dict[str, Any]:
    summary = data.get('summary', {}) or {}
    state_rows = (data.get('state') or {}).get('data') or []
    rsm_rows = (data.get('rsm') or {}).get('data') or []
    top_states, bottom_states = _rank_rows(state_rows, 'prev_month_delta_pct_mtd_p3_nsv', 'mtd_p3_nsv', n=5)
    top_rsm, bottom_rsm = _rank_rows(rsm_rows, 'prev_month_delta_pct_p3_nsv', 'p3_nsv', n=5)
    payload: Dict[str, Any] = {'summary_kpis': summary.get('kpis', {}), 'top_states': top_states, 'bottom_states': bottom_states, 'top_rsms': top_rsm, 'bottom_rsms': bottom_rsm, 'rep_silence': {'count': len(silence_rows), 'lost_p3_nsv': round(sum((float(r.get('prev_p3') or 0) for r in silence_rows)), 2), 'reps': silence_rows[:25]}}
    if narrative:
        payload['narrative'] = _compose_narrative(payload)
    return payload

def _compose_narrative(payload: Dict[str, Any]) -> str:
    parts: List[str] = []
    k = payload.get('summary_kpis') or {}
    p3 = k.get('total_p3_nsv')
    p3_d = k.get('prev_month_delta_pct_total_p3_nsv')
    if p3 is not None and p3_d is not None:
        direction = 'up' if p3_d >= 0 else 'down'
        parts.append(f'P3 NSV is at {_fmt_inr(p3)}, {direction} {abs(p3_d)}% vs last month.')
    top = payload.get('top_states') or []
    if top:
        s = top[0]
        parts.append(f"{s.get('state')} leads growth at +{s.get('prev_month_delta_pct_mtd_p3_nsv')}% ({_fmt_inr(s.get('mtd_p3_nsv'))}).")
    bot = payload.get('bottom_states') or []
    if bot:
        s = bot[0]
        parts.append(f"{s.get('state')} is the biggest drag at {s.get('prev_month_delta_pct_mtd_p3_nsv')}%.")
    silence = payload.get('rep_silence') or {}
    if silence.get('count'):
        parts.append(f"{silence['count']} reps stopped billing P3 this month — a {_fmt_inr(silence.get('lost_p3_nsv'))} addressable gap.")
    if not parts:
        return 'Not enough data yet to surface insights.'
    return ' '.join(parts)

def _cache_key(scope: str, segments: List[str], business_month: str, report_date: Optional[str], narrative: bool=False) -> str:
    seg_key = ','.join(sorted(segments))
    rd = report_date or datetime.utcnow().date().isoformat()
    return f'insights:{scope}:{seg_key}:{business_month}:{rd}:n{int(narrative)}'

async def compute_ribbon_insights(segments: List[str], business_month: str, report_date: Optional[str]) -> Dict[str, Any]:
    t0 = time.monotonic()
    logger.info('compute_ribbon_insights | bm=%s segments=%s report_date=%s', business_month, segments, report_date)
    key = _cache_key('ribbon', segments, business_month, report_date)
    cached = await _cache_get(key)
    if cached is not None:
        logger.info('compute_ribbon_insights | served from cache | %.2fs', time.monotonic() - t0)
        return cached
    data, silence = await asyncio.gather(_gather_ranked_data(segments, business_month, report_date), get_rep_silence(segments, business_month, limit=200))
    cards = _build_ribbon(data, silence_count=len(silence))
    payload = {'generated_at': datetime.utcnow().isoformat() + 'Z', 'business_month': business_month, 'cards': cards}
    await _cache_set(key, payload, ttl=300)
    logger.info('compute_ribbon_insights | done | %d cards | %.2fs', len(cards), time.monotonic() - t0)
    return payload

async def compute_full_insights(segments: List[str], business_month: str, report_date: Optional[str], narrative: bool=False) -> Dict[str, Any]:
    t0 = time.monotonic()
    logger.info('compute_full_insights | bm=%s segments=%s report_date=%s narrative=%s', business_month, segments, report_date, narrative)
    key = _cache_key('full', segments, business_month, report_date, narrative)
    cached = await _cache_get(key)
    if cached is not None:
        logger.info('compute_full_insights | served from cache | %.2fs', time.monotonic() - t0)
        return cached
    data, silence = await asyncio.gather(_gather_ranked_data(segments, business_month, report_date), get_rep_silence(segments, business_month, limit=200))
    payload = _build_full(data, silence, narrative=narrative)
    payload['generated_at'] = datetime.utcnow().isoformat() + 'Z'
    payload['business_month'] = business_month
    ttl = 3600 if narrative else 300
    await _cache_set(key, payload, ttl=ttl)
    logger.info('compute_full_insights | done | %.2fs', time.monotonic() - t0)
    return payload