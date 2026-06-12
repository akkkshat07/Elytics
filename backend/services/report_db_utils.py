from __future__ import annotations
import os
import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sqlalchemy import create_engine, text
logger = logging.getLogger(__name__)
_DB_URL_NE = os.getenv('CORESIGHT_MASTER_NE_URL', 'postgresql://esme_user:esme_password@localhost:5432/coresight_master_ne')
_DB_URL_BH = os.getenv('CORESIGHT_MASTER_BH_URL', 'postgresql://esme_user:esme_password@localhost:5432/coresight_master_bh')
VALID_SEGMENTS = {'ne_retail', 'ne_parlor', 'bh'}
ALL_SEGMENTS = 'ne_retail,ne_parlor,bh'
RSM_DESIGNATIONS = ('RSM', 'BDM R')
TL_DESIGNATIONS = ('ASM', 'BDM', 'TSM', 'ASE', 'KAM', 'SSE')
EXCLUDED_DESIGNATIONS = ('NSM', 'SFA')
_engines: Dict[str, Any] = {}

def _engine_for(db: str):
    if db not in _engines:
        url = _DB_URL_NE if db == 'ne' else _DB_URL_BH
        _engines[db] = create_engine(url, pool_pre_ping=True, pool_size=3, max_overflow=10)
    return _engines[db]

def master_table(db: str) -> str:
    return 'master_ne' if db == 'ne' else 'master_bh'

def parse_segments(segments: Optional[str]) -> List[str]:
    if not segments:
        return list(VALID_SEGMENTS)
    parts = [s.strip().lower() for s in segments.split(',') if s.strip()]
    valid = [s for s in parts if s in VALID_SEGMENTS]
    return valid if valid else list(VALID_SEGMENTS)

def build_segment_queries(segments: List[str]) -> List[Tuple[str, str]]:
    queries = []
    ne_segs = []
    for seg in segments:
        if seg == 'ne_retail':
            ne_segs.append('Retail')
        elif seg == 'ne_parlor':
            ne_segs.append('Parlor')
        elif seg == 'bh':
            queries.append(('bh', ''))
    if len(ne_segs) == 2:
        queries.append(('ne', ''))
    elif 'Parlor' in ne_segs:
        queries.append(('ne', "AND segment = 'Parlor'"))
    elif 'Retail' in ne_segs:
        queries.append(('ne', "AND segment != 'Parlor'"))
    return queries

def segment_clause(seg_filter: str) -> str:
    return seg_filter

def zone_clause(zone: Optional[str]) -> str:
    if zone:
        return 'AND emp_zone = :zone'
    return ''

def excluded_designations_clause() -> str:
    return "AND designation NOT IN ('NSM', 'SFA')"

def rsm_designation_clause() -> str:
    return "AND (rsm_designation IS NULL OR rsm_designation NOT IN ('NSM', 'SFA'))"

def date_clause(report_date: Optional[str], single_day: bool=False) -> str:
    if not report_date:
        return ''
    if single_day:
        return 'AND order_date::date = :report_date'
    return 'AND order_date::date <= :report_date'

def run_query(engine, sql: str, params: dict=None) -> pd.DataFrame:
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        return pd.DataFrame(result.fetchall(), columns=result.keys())
_DATE_COLS = {'order_date', 'report_date', 'deactive_date'}
CHUNK_SIZE = 10000

def stream_base_data_to_sheet(ws, segments: List[str], business_month: str, report_date: Optional[str]=None, single_day: bool=False) -> int:
    import time
    t0 = time.monotonic()
    logger.info('stream_base_data | bm=%s segments=%s report_date=%s single_day=%s', business_month, segments, report_date, single_day)
    seg_queries = build_segment_queries(segments)
    dt = date_clause(report_date, single_day) if report_date else ''
    header_written = False
    total_rows = 0
    for db_name, seg_filter in seg_queries:
        engine = _engine_for(db_name)
        tbl = master_table(db_name)
        sql = f'\n            SELECT *\n            FROM {tbl}\n            WHERE business_month = :bm {dt} {seg_filter}\n            ORDER BY record_type, emp_zone, rsm_name, emp_name, order_date\n        '
        params: dict = {'bm': business_month}
        if report_date:
            params['report_date'] = report_date
        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            columns = list(result.keys())
            if not header_written:
                ws.append(columns)
                header_written = True
            batch = []
            for row in result:
                values = []
                for col, val in zip(columns, row):
                    if col in _DATE_COLS and val is not None:
                        try:
                            values.append(val.date() if hasattr(val, 'date') else val)
                        except Exception:
                            values.append(val)
                    else:
                        values.append(val)
                batch.append(values)
                if len(batch) >= CHUNK_SIZE:
                    for r in batch:
                        ws.append(r)
                    total_rows += len(batch)
                    batch.clear()
            if batch:
                for r in batch:
                    ws.append(r)
                total_rows += len(batch)
    if not header_written:
        ws.append(['No data'])
    logger.info('stream_base_data | done | %d rows | %.2fs', total_rows, time.monotonic() - t0)
    return total_rows