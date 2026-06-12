from __future__ import annotations
import csv
import io
import logging
import re
from datetime import date, datetime
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
import defusedxml.ElementTree as ET
from defusedxml.ElementTree import parse
from pydantic import BaseModel, Field
from sqlalchemy import text
from db_config.connectors.postgres_connector import PostgresConnector
from services.db_credentials_service import DBCredentialsService
from util.data_source import require_store_in_local
from util.dataset_paths import resolve_xml_data_sources_dir
logger = logging.getLogger(__name__)
LIVE_SQL_DB_TYPES = frozenset({'postgres', 'mysql'})
MAX_PAGE_SIZE = 500
DEFAULT_PAGE_SIZE = 100
EXPORT_ROW_CAP = 500000
EXPORT_CHUNK_SIZE = 5000
SIMPLE_IDENT_RE = re.compile('^[a-zA-Z_][a-zA-Z0-9_]*$')

class ViewReportFilter(BaseModel):
    column: str = Field(..., min_length=1, max_length=128)
    op: str = Field(..., min_length=1, max_length=32)
    value: Optional[Any] = None
    value_to: Optional[Any] = None

class ViewReportSort(BaseModel):
    column: str = Field(..., min_length=1, max_length=128)
    direction: str = Field(default='asc', pattern='^(?i)(asc|desc)$')

class ViewReportQueryRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1, max_length=128)
    table_name: str = Field(..., min_length=1, max_length=256)
    db_schema: str = Field(default='public', max_length=128)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)
    sort: Optional[ViewReportSort] = None
    filters: List[ViewReportFilter] = Field(default_factory=list)

class ViewReportExportRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1, max_length=128)
    table_name: str = Field(..., min_length=1, max_length=256)
    db_schema: str = Field(default='public', max_length=128)
    sort: Optional[ViewReportSort] = None
    filters: List[ViewReportFilter] = Field(default_factory=list)

@dataclass
class TableCatalogEntry:
    dataset_id: str
    dataset_name: str
    table_name: str
    schema: str
    columns: List[Dict[str, str]] = field(default_factory=list)

    @property
    def entry_id(self) -> str:
        return f'{self.dataset_id}::{self.table_name}'

    @property
    def label(self) -> str:
        return f'{self.dataset_name} - {self.table_name}'

    def column_names(self) -> List[str]:
        return [c['name'] for c in self.columns if c.get('name')]

class LiveTableViewError(Exception):

    def __init__(self, message: str, status_code: int=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

def _assert_safe_ident(name: str) -> None:
    if not name or not str(name).strip():
        raise LiveTableViewError(f'Invalid identifier: {name!r}')
    if '\x00' in name or ';' in name or '--' in name:
        raise LiveTableViewError(f'Invalid identifier: {name!r}')

def _quote_ident(name: str, dialect: str='postgres') -> str:
    _assert_safe_ident(name)
    if dialect == 'postgres':
        escaped = str(name).replace('"', '""')
        return f'"{escaped}"'
    if SIMPLE_IDENT_RE.match(name):
        return f'`{name}`'
    escaped = str(name).replace('`', '``')
    return f'`{escaped}`'

def _map_sdtype_to_filter_kind(sdtype: str) -> str:
    s = (sdtype or 'text').lower()
    if s in ('numerical', 'number', 'integer', 'float'):
        return 'number'
    if s in ('datetime', 'date'):
        return 'date'
    return 'text'

def _is_date_like_column(col_meta: Dict[str, str]) -> bool:
    sdtype = (col_meta.get('sdtype') or '').lower()
    data_type = (col_meta.get('data_type') or '').lower()
    if sdtype in ('datetime', 'date'):
        return True
    return 'date' in data_type or 'timestamp' in data_type

def _coerce_filter_param(val: Any, col_meta: Dict[str, str]) -> Any:
    if val is None or not _is_date_like_column(col_meta):
        return val
    if isinstance(val, datetime):
        return val
    if isinstance(val, date):
        data_type = (col_meta.get('data_type') or '').lower()
        if 'timestamp' in data_type:
            return datetime.combine(val, datetime.min.time())
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return val
        try:
            if 'T' in s:
                return datetime.fromisoformat(s.replace('Z', '+00:00'))
            parsed = date.fromisoformat(s[:10])
        except ValueError as e:
            raise LiveTableViewError(f'Invalid date value: {val!r}') from e
        data_type = (col_meta.get('data_type') or '').lower()
        if 'timestamp' in data_type:
            return datetime.combine(parsed, datetime.min.time())
        return parsed
    return val

def _load_dataset_table_catalog(client_id: str, dataset_id: str, dataset_name: str) -> List[TableCatalogEntry]:
    base_dir = resolve_xml_data_sources_dir(client_id, dataset_id)
    meta_dir = base_dir / 'meta_information'
    desc_dir = base_dir / 'data_descriptions'
    intros_map: Dict[str, str] = {}
    intros_file = meta_dir / 'table_introductions.xml'
    if intros_file.exists():
        try:
            tree = parse(intros_file)
            intros_map = {elem.get('table_name'): (elem.text or '').strip() for elem in tree.getroot().findall('.//table_introduction') if elem.get('table_name')}
        except ET.ParseError as e:
            logger.warning('Failed to parse table_introductions for %s/%s: %s', client_id, dataset_id, e)
    tables_dict: Dict[str, List[Dict[str, str]]] = {}
    if desc_dir.exists():
        for file_path in glob(str(desc_dir / '*_description.xml')):
            table_name = Path(file_path).name.replace('_description.xml', '')
            try:
                tree = parse(file_path)
                root = tree.getroot()
                cols: List[Dict[str, str]] = []
                for col in root.findall('.//column'):
                    col_name = col.get('name')
                    if not col_name:
                        continue
                    data_type = (col.get('data_type') or '').lower()
                    sdtype = 'text'
                    if any((x in data_type for x in ('int', 'numeric', 'decimal', 'double', 'float', 'real'))):
                        sdtype = 'numerical'
                    elif 'date' in data_type or 'time' in data_type:
                        sdtype = 'datetime'
                    elif 'bool' in data_type:
                        sdtype = 'categorical'
                    cols.append({'name': col_name, 'data_type': data_type, 'sdtype': sdtype})
                tables_dict[table_name] = cols
            except ET.ParseError as e:
                logger.warning('Failed parsing %s: %s', file_path, e)
    for table_name in intros_map:
        if table_name not in tables_dict:
            tables_dict[table_name] = []
    entries: List[TableCatalogEntry] = []
    for table_name, columns in tables_dict.items():
        entries.append(TableCatalogEntry(dataset_id=dataset_id, dataset_name=dataset_name, table_name=table_name, schema='public', columns=columns))
    return entries

def build_where_clause(filters: List[ViewReportFilter], allowed_columns: Dict[str, Dict[str, str]], dialect: str='postgres') -> Tuple[str, Dict[str, Any]]:
    if not filters:
        return ('', {})
    parts: List[str] = []
    params: Dict[str, Any] = {}
    param_idx = 0
    for f in filters:
        col = f.column
        if col not in allowed_columns:
            raise LiveTableViewError(f'Unknown filter column: {col}')
        qcol = _quote_ident(col, dialect)
        op = (f.op or '').lower()
        if op == 'is_null':
            parts.append(f'{qcol} IS NULL')
            continue
        if op == 'is_not_null':
            parts.append(f'{qcol} IS NOT NULL')
            continue
        col_meta = allowed_columns[col]
        if op == 'between':
            if f.value is None or f.value_to is None:
                raise LiveTableViewError(f"Filter 'between' requires value and value_to for {col}")
            k1, k2 = (f'p{param_idx}', f'p{param_idx + 1}')
            param_idx += 2
            params[k1] = _coerce_filter_param(f.value, col_meta)
            params[k2] = _coerce_filter_param(f.value_to, col_meta)
            parts.append(f'{qcol} BETWEEN :{k1} AND :{k2}')
            continue
        if f.value is None or (isinstance(f.value, str) and (not f.value.strip())):
            continue
        key = f'p{param_idx}'
        param_idx += 1
        val = _coerce_filter_param(f.value, col_meta)
        if op == 'eq':
            parts.append(f'{qcol} = :{key}')
            params[key] = val
        elif op == 'neq':
            parts.append(f'{qcol} <> :{key}')
            params[key] = val
        elif op == 'contains':
            parts.append(f'CAST({qcol} AS TEXT) ILIKE :{key}')
            params[key] = f'%{val}%'
        elif op == 'starts_with':
            parts.append(f'CAST({qcol} AS TEXT) ILIKE :{key}')
            params[key] = f'{val}%'
        elif op == 'gt':
            parts.append(f'{qcol} > :{key}')
            params[key] = val
        elif op == 'gte':
            parts.append(f'{qcol} >= :{key}')
            params[key] = val
        elif op == 'lt':
            parts.append(f'{qcol} < :{key}')
            params[key] = val
        elif op == 'lte':
            parts.append(f'{qcol} <= :{key}')
            params[key] = val
        elif op == 'in':
            if not isinstance(val, list):
                raise LiveTableViewError(f"Filter 'in' requires a list value for {col}")
            if not val:
                parts.append('1=0')
                continue
            placeholders = []
            for i, item in enumerate(val):
                k = f'p{param_idx}'
                param_idx += 1
                params[k] = item
                placeholders.append(f':{k}')
            parts.append(f"{qcol} IN ({', '.join(placeholders)})")
        else:
            raise LiveTableViewError(f'Unsupported filter op: {op}')
    if not parts:
        return ('', {})
    return (' WHERE ' + ' AND '.join(parts), params)

class LiveTableViewService:

    def __init__(self, db):
        self._db = db
        self._creds = DBCredentialsService(db)

    async def list_viewable_tables(self, client_id: str) -> List[Dict[str, Any]]:
        datasets = await self._creds.get_active_datasets(client_id)
        out: List[Dict[str, Any]] = []
        for ds in datasets:
            if ds.get('store_in_local'):
                continue
            db_type = (ds.get('db_type') or '').lower()
            if db_type not in LIVE_SQL_DB_TYPES:
                continue
            dataset_id = str(ds.get('dataset_id') or '')
            if not dataset_id:
                continue
            dataset_name = ds.get('dataset_name') or dataset_id
            catalog_entries = _load_dataset_table_catalog(client_id, dataset_id, dataset_name)
            if db_type == 'postgres':
                try:
                    connector = await self._get_pooled_connector(client_id, dataset_id)
                    synced: List[TableCatalogEntry] = []
                    for entry in catalog_entries:
                        try:
                            synced.append(await self._sync_columns_with_database(connector, entry))
                        except LiveTableViewError as e:
                            logger.warning('view-reports catalog sync skipped %s/%s: %s', dataset_id, entry.table_name, e.message)
                            synced.append(entry)
                    catalog_entries = synced
                except Exception as e:
                    logger.warning('view-reports catalog sync failed for dataset %s: %s', dataset_id, e)
            for entry in catalog_entries:
                out.append({'id': entry.entry_id, 'label': entry.label, 'dataset_id': entry.dataset_id, 'dataset_name': entry.dataset_name, 'table_name': entry.table_name, 'schema': entry.schema, 'columns': entry.columns})
        return out

    async def _resolve_catalog_entry(self, client_id: str, dataset_id: str, table_name: str) -> TableCatalogEntry:
        datasets = await self._creds.get_active_datasets(client_id)
        ds = next((d for d in datasets if str(d.get('dataset_id')) == dataset_id), None)
        if not ds:
            raise LiveTableViewError('Dataset not found or not enabled', 404)
        if ds.get('store_in_local'):
            raise LiveTableViewError('Dataset is not a live connection', 400)
        db_type = (ds.get('db_type') or '').lower()
        if db_type not in LIVE_SQL_DB_TYPES:
            raise LiveTableViewError(f'Unsupported database type: {db_type}', 400)
        dataset_name = ds.get('dataset_name') or dataset_id
        catalog = _load_dataset_table_catalog(client_id, dataset_id, dataset_name)
        entry = next((e for e in catalog if e.table_name == table_name), None)
        if not entry:
            raise LiveTableViewError('Table not found in metadata catalog', 404)
        return entry

    async def _get_credentials(self, client_id: str, dataset_id: str) -> Dict[str, Any]:
        creds = await self._creds.get_credentials(client_id=client_id, dataset_id=dataset_id, decrypt_password=True)
        if not creds:
            raise LiveTableViewError('Database credentials not found', 404)
        if require_store_in_local(creds):
            raise LiveTableViewError('Dataset is not a live connection', 400)
        return creds

    def _build_dsn(self, creds: Dict[str, Any]) -> str:
        db_url = creds.get('db_url')
        if db_url:
            return db_url
        db_type = creds.get('db_type', 'postgres')
        host = creds.get('db_host', '')
        port = creds.get('db_port', 5432)
        name = creds.get('db_name', '')
        user = creds.get('db_username', '')
        password = creds.get('db_password', '')
        if db_type == 'mysql':
            return f'mysql+aiomysql://{user}:{password}@{host}:{port}/{name}'
        return f'postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}'

    def _ssh_config(self, creds: Dict[str, Any]) -> Optional[dict]:
        additional = creds.get('additional_params') or {}
        ssh = additional.get('ssh')
        return ssh if isinstance(ssh, dict) else None

    @staticmethod
    def _column_meta_from_db_row(column_name: str, data_type: str) -> Dict[str, str]:
        dt = (data_type or '').lower()
        sdtype = 'text'
        if any((x in dt for x in ('int', 'numeric', 'decimal', 'double', 'float', 'real'))):
            sdtype = 'numerical'
        elif 'date' in dt or 'time' in dt:
            sdtype = 'datetime'
        return {'name': column_name, 'data_type': dt, 'sdtype': sdtype}

    @staticmethod
    def reconcile_columns_with_database(catalog_columns: List[Dict[str, str]], db_columns: List[Dict[str, str]]) -> List[Dict[str, str]]:
        db_by_name = {c['name']: c for c in db_columns if c.get('name')}
        if not db_by_name:
            return []
        catalog_by_name = {c['name']: c for c in catalog_columns if c.get('name')}
        catalog_order = [c['name'] for c in catalog_columns if c.get('name')]
        db_order = [c['name'] for c in db_columns if c.get('name')]
        if catalog_order:
            ordered_names = [n for n in catalog_order if n in db_by_name]
            ordered_names.extend((n for n in db_order if n not in catalog_by_name))
        else:
            ordered_names = db_order
        merged: List[Dict[str, str]] = []
        for name in ordered_names:
            base = dict(db_by_name[name])
            if name in catalog_by_name:
                cat = catalog_by_name[name]
                base['data_type'] = cat.get('data_type') or base.get('data_type', '')
                base['sdtype'] = cat.get('sdtype') or base.get('sdtype', 'text')
            merged.append(base)
        return merged

    async def _sync_columns_with_database(self, connector: PostgresConnector, entry: TableCatalogEntry) -> TableCatalogEntry:
        schema = entry.schema or 'public'
        session_factory = connector.get_db()
        async with session_factory() as session:
            result = await session.execute(text('\n                    SELECT column_name, data_type\n                    FROM information_schema.columns\n                    WHERE table_schema = :schema AND table_name = :table_name\n                    ORDER BY ordinal_position\n                    '), {'schema': schema, 'table_name': entry.table_name})
            rows = result.mappings().all()
        if not rows:
            raise LiveTableViewError(f'Table {schema}.{entry.table_name} not found in database', 404)
        db_columns = [self._column_meta_from_db_row(row['column_name'], row['data_type'] or '') for row in rows]
        entry.columns = self.reconcile_columns_with_database(entry.columns, db_columns)
        if not entry.columns:
            raise LiveTableViewError('No columns available for table', 400)
        return entry

    async def _get_pooled_connector(self, client_id: str, dataset_id: str):
        from db_config.connection_pool_manager import ConnectionPoolManager
        creds = await self._get_credentials(client_id, dataset_id)
        db_type = (creds.get('db_type') or 'postgres').lower()
        if db_type != 'postgres':
            raise LiveTableViewError('Only PostgreSQL live tables are supported in v1', 400)
        pool = ConnectionPoolManager()
        return await pool.get_connection(client_id, self._db, dataset_id=dataset_id)

    def _qualified_table(self, schema: str, table_name: str, dialect: str='postgres') -> str:
        return f'{_quote_ident(schema, dialect)}.{_quote_ident(table_name, dialect)}'

    async def query_table(self, client_id: str, request: ViewReportQueryRequest) -> Dict[str, Any]:
        entry = await self._resolve_catalog_entry(client_id, request.dataset_id, request.table_name)
        schema = request.db_schema or entry.schema or 'public'
        connector = await self._get_pooled_connector(client_id, request.dataset_id)
        entry = await self._sync_columns_with_database(connector, entry)
        allowed = {c['name']: c for c in entry.columns}
        col_names = entry.column_names()
        for f in request.filters:
            if f.column not in allowed:
                raise LiveTableViewError(f'Unknown filter column: {f.column}')
        dialect = 'postgres'
        where_sql, params = build_where_clause(request.filters, allowed, dialect)
        qtable = self._qualified_table(schema, entry.table_name, dialect)
        select_cols = ', '.join((_quote_ident(c, dialect) for c in col_names))
        order_sql = ''
        if request.sort and request.sort.column in allowed:
            direction = 'DESC' if request.sort.direction.lower() == 'desc' else 'ASC'
            order_sql = f' ORDER BY {_quote_ident(request.sort.column, dialect)} {direction}'
        offset = (request.page - 1) * request.page_size
        count_params = dict(params)
        data_params = {**params, 'limit': request.page_size, 'offset': offset}
        count_sql = f'SELECT COUNT(*) AS cnt FROM {qtable}{where_sql}'
        data_sql = f'SELECT {select_cols} FROM {qtable}{where_sql}{order_sql} LIMIT :limit OFFSET :offset'
        session_factory = connector.get_db()
        async with session_factory() as session:
            count_result = await session.execute(text(count_sql), count_params)
            total_rows = int(count_result.scalar() or 0)
            data_result = await session.execute(text(data_sql), data_params)
            rows_raw = data_result.fetchall()
        rows = [[_serialize_cell(row[i]) for i in range(len(col_names))] for row in rows_raw]
        total_pages = max(1, (total_rows + request.page_size - 1) // request.page_size)
        return {'columns': col_names, 'rows': rows, 'page': request.page, 'page_size': request.page_size, 'total_rows': total_rows, 'total_pages': total_pages}

    async def export_table_csv(self, client_id: str, request: ViewReportExportRequest) -> Tuple[str, AsyncIterator[bytes]]:
        entry = await self._resolve_catalog_entry(client_id, request.dataset_id, request.table_name)
        schema = request.db_schema or entry.schema or 'public'
        dataset_name = entry.dataset_name
        safe_name = re.sub('[^\\w\\-]+', '_', f'{dataset_name}-{entry.table_name}').strip('_')
        filename = f"{safe_name or 'export'}.csv"
        connector = await self._get_pooled_connector(client_id, request.dataset_id)
        entry = await self._sync_columns_with_database(connector, entry)
        allowed = {c['name']: c for c in entry.columns}
        col_names = entry.column_names()
        dialect = 'postgres'
        where_sql, params = build_where_clause(request.filters, allowed, dialect)
        qtable = self._qualified_table(schema, entry.table_name, dialect)
        select_cols = ', '.join((_quote_ident(c, dialect) for c in col_names))
        order_sql = ''
        if request.sort and request.sort.column in allowed:
            direction = 'DESC' if request.sort.direction.lower() == 'desc' else 'ASC'
            order_sql = f' ORDER BY {_quote_ident(request.sort.column, dialect)} {direction}'
        count_sql = f'SELECT COUNT(*) AS cnt FROM {qtable}{where_sql}'
        session_factory = connector.get_db()
        async with session_factory() as session:
            count_result = await session.execute(text(count_sql), dict(params))
            total_rows = int(count_result.scalar() or 0)
        if total_rows > EXPORT_ROW_CAP:
            raise LiveTableViewError(f'Export exceeds maximum of {EXPORT_ROW_CAP:,} rows ({total_rows:,} matched). Apply filters to reduce the result set.', 400)

        async def _stream() -> AsyncIterator[bytes]:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(col_names)
            yield buf.getvalue().encode('utf-8')
            buf.seek(0)
            buf.truncate(0)
            offset = 0
            chunk_params_base = dict(params)
            while offset < total_rows:
                chunk_params = {**chunk_params_base, 'limit': EXPORT_CHUNK_SIZE, 'offset': offset}
                data_sql = f'SELECT {select_cols} FROM {qtable}{where_sql}{order_sql} LIMIT :limit OFFSET :offset'
                async with session_factory() as session:
                    result = await session.execute(text(data_sql), chunk_params)
                    batch = result.fetchall()
                if not batch:
                    break
                for row in batch:
                    writer.writerow([_serialize_cell(row[i]) for i in range(len(col_names))])
                yield buf.getvalue().encode('utf-8')
                buf.seek(0)
                buf.truncate(0)
                offset += len(batch)
        return (filename, _stream())

def _serialize_cell(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, 'isoformat'):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value