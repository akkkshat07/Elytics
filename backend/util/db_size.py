import os
import logging
import asyncio
from typing import Optional
from sqlalchemy import text
logger = logging.getLogger(__name__)
MAX_DB_SIZE_BYTES: int = int(os.getenv('MAX_UPLOAD_SIZE', 200 * 1024 * 1024))

async def get_database_size_bytes(db_type: str, db_host: str, db_port: int, db_name: str, db_username: str, db_password: str, db_url: Optional[str]=None) -> int:
    try:
        if db_type == 'postgres':
            return await _postgres_size(db_host, db_port, db_name, db_username, db_password, db_url)
        elif db_type == 'mysql':
            return await _mysql_size(db_host, db_port, db_name, db_username, db_password, db_url)
        elif db_type == 'mongodb':
            return await _mongodb_size(db_host, db_port, db_name, db_username, db_password, db_url)
        elif db_type == 'sap_hana':
            return await _sap_hana_size(db_host, db_port, db_name, db_username, db_password, db_url)
        elif db_type == 'sap_oracle':
            return await _oracle_size(db_host, db_port, db_name, db_username, db_password, db_url)
        elif db_type == 'sap_sybase':
            return await _sybase_size(db_host, db_port, db_name, db_username, db_password, db_url)
        else:
            logger.warning('get_database_size_bytes: unsupported db_type=%s', db_type)
            return 0
    except Exception as exc:
        logger.warning('Failed to query database size for db_type=%s: %s', db_type, exc, exc_info=True)
        return 0

def format_size_mb(size_bytes: int) -> str:
    return f'{size_bytes / (1024 * 1024):.2f}'

async def _postgres_size(db_host: str, db_port: int, db_name: str, db_username: str, db_password: str, db_url: Optional[str]) -> int:
    from sqlalchemy.ext.asyncio import create_async_engine
    dsn = db_url if db_url and 'asyncpg' in db_url else f'postgresql+asyncpg://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
    engine = create_async_engine(dsn)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text('SELECT pg_database_size(current_database())'))
            row = result.scalar()
            return int(row) if row else 0
    finally:
        await engine.dispose()

async def _mysql_size(db_host: str, db_port: int, db_name: str, db_username: str, db_password: str, db_url: Optional[str]) -> int:
    from sqlalchemy.ext.asyncio import create_async_engine
    dsn = db_url if db_url and 'aiomysql' in db_url else f'mysql+aiomysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
    engine = create_async_engine(dsn)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text('SELECT SUM(data_length + index_length) FROM information_schema.tables WHERE table_schema = DATABASE()'))
            row = result.scalar()
            return int(row) if row else 0
    finally:
        await engine.dispose()

async def _mongodb_size(db_host: str, db_port: int, db_name: str, db_username: str, db_password: str, db_url: Optional[str]) -> int:
    from motor.motor_asyncio import AsyncIOMotorClient
    if db_url and ('mongodb+srv' in db_url or 'mongodb.net' in db_url):
        uri = db_url
    else:
        uri = f'mongodb+srv://{db_username}:{db_password}@{db_host}/{db_name}'
    client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
    try:
        db = client[db_name]
        stats = await db.command('dbStats')
        return int(stats.get('dataSize', 0))
    finally:
        client.close()

async def _sap_hana_size(db_host: str, db_port: int, db_name: str, db_username: str, db_password: str, db_url: Optional[str]) -> int:
    import re
    from hdbcli import dbapi
    if db_url:
        match = re.search('hana://.+:.+@(.+):(\\d+)', db_url)
        if match:
            db_host, db_port = (match.group(1), int(match.group(2)))

    def _query():
        conn = dbapi.connect(address=db_host, port=int(db_port), user=db_username, password=db_password)
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT SUM(TABLE_SIZE) FROM M_TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA')
            row = cursor.fetchone()
            return int(row[0]) if row and row[0] else 0
        finally:
            conn.close()
    return await asyncio.to_thread(_query)

async def _oracle_size(db_host: str, db_port: int, db_name: str, db_username: str, db_password: str, db_url: Optional[str]) -> int:
    from sqlalchemy.ext.asyncio import create_async_engine
    dsn = db_url if db_url and 'oracledb' in db_url else f'oracle+oracledb_async://{db_username}:{db_password}@{db_host}:{db_port}/?service_name={db_name}'
    engine = create_async_engine(dsn)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text('SELECT SUM(bytes) FROM user_segments'))
            row = result.scalar()
            return int(row) if row else 0
    finally:
        await engine.dispose()

async def _sybase_size(db_host: str, db_port: int, db_name: str, db_username: str, db_password: str, db_url: Optional[str]) -> int:
    from sqlalchemy import create_engine
    from db_config.connectors.sybase_connector import build_sybase_url
    import pyodbc
    odbc_conn_str = build_sybase_url(db_host, db_port, db_username, db_password, db_name)

    def _query():
        conn = pyodbc.connect(odbc_conn_str)
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT SUM(reserved_pages) * (@@maxpagesize) FROM sysusages WHERE dbid = db_id(db_name())')
            row = cursor.fetchone()
            return int(row[0]) if row and row[0] else 0
        finally:
            conn.close()
    return await asyncio.to_thread(_query)