import logging
import pandas as pd
from typing import Optional, Dict, Any
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from util.time_utils import utcnow
logger = logging.getLogger(__name__)

async def _validate_identifiers(session, is_oracle: bool, is_mysql: bool, is_hana: bool, schema: str, table_name: str, columns: Optional[list]) -> None:
    if is_oracle:
        table_query = text('SELECT TABLE_NAME FROM ALL_TABLES WHERE OWNER = :schema')
    elif is_mysql:
        table_query = text('SELECT TABLE_NAME FROM information_schema.tables WHERE TABLE_SCHEMA = :schema')
    else:
        table_query = text('SELECT table_name FROM information_schema.tables WHERE table_schema = :schema')
    result = await session.execute(table_query, {'schema': schema})
    allowed_tables = {row[0] for row in result.fetchall()}
    lookup_table = table_name.upper() if is_oracle else table_name
    if lookup_table not in allowed_tables:
        raise ValueError(f"Table '{table_name}' does not exist in schema '{schema}'")
    if columns:
        if is_oracle:
            col_query = text('SELECT COLUMN_NAME FROM ALL_TAB_COLUMNS WHERE OWNER = :schema AND TABLE_NAME = :table_name')
        elif is_mysql:
            col_query = text('SELECT COLUMN_NAME FROM information_schema.columns WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table_name')
        else:
            col_query = text('SELECT column_name FROM information_schema.columns WHERE table_schema = :schema AND table_name = :table_name')
        col_result = await session.execute(col_query, {'schema': schema, 'table_name': lookup_table})
        allowed_columns = {row[0] for row in col_result.fetchall()}
        if is_oracle:
            allowed_columns = {c.upper() for c in allowed_columns}
        invalid = [c for c in columns if (c.upper() if is_oracle else c) not in allowed_columns]
        if invalid:
            raise ValueError(f"Column(s) {invalid} do not exist in table '{table_name}'")

def read_sql_table(table_name: str, schema: str='public', columns: Optional[list]=None, session_factory=None) -> pd.DataFrame:
    if session_factory is None:
        raise RuntimeError('Database connection not available. Please ensure store_in_local=False and database credentials are configured.')
    try:
        import asyncio

        async def _read_table():
            async with session_factory() as session:
                is_mysql = session.bind.dialect.name == 'mysql'
                is_hana = session.bind.dialect.name == 'hana'
                is_oracle = session.bind.dialect.name == 'oracle'
                actual_schema = schema
                if is_mysql and (schema is None or schema == 'public'):
                    db_name_result = await session.execute(text('SELECT DATABASE()'))
                    actual_schema = db_name_result.scalar()
                elif is_hana and (schema is None or schema == 'public'):
                    schema_result = await session.execute(text('SELECT CURRENT_SCHEMA FROM DUMMY'))
                    actual_schema = schema_result.scalar()
                elif is_oracle and (schema is None or schema == 'public'):
                    schema_result = await session.execute(text('SELECT USER FROM DUAL'))
                    actual_schema = schema_result.scalar()
                await _validate_identifiers(session, is_oracle, is_mysql, is_hana, actual_schema, table_name, columns)
                if is_mysql:
                    if columns:
                        cols_str = ', '.join([f'`{col}`' for col in columns])
                        query = text(f'SELECT {cols_str} FROM `{table_name}`')
                    else:
                        query = text(f'SELECT * FROM `{table_name}`')
                elif is_hana:
                    if columns:
                        cols_str = ', '.join([f'"{col}"' for col in columns])
                        query = text(f'SELECT {cols_str} FROM "{actual_schema}"."{table_name}"')
                    else:
                        query = text(f'SELECT * FROM "{actual_schema}"."{table_name}"')
                elif is_oracle:
                    if columns:
                        cols_str = ', '.join([f'"{col}"' for col in columns])
                        query = text(f'SELECT {cols_str} FROM "{actual_schema}"."{table_name}"')
                        try:
                            result = await session.execute(query)
                            rows = result.mappings().fetchall()
                        except Exception as e:
                            error_str = str(e)
                            if 'ORA-00904' in error_str or 'invalid identifier' in error_str.lower():
                                logger.warning(f'Column(s) not found in table {actual_schema}.{table_name}. Requested columns: {columns}. Error: {error_str[:200]}. Falling back to SELECT * and filtering to available columns.')
                                query = text(f'SELECT * FROM "{actual_schema}"."{table_name}"')
                                result = await session.execute(query)
                                rows = result.mappings().fetchall()
                                if rows:
                                    df = pd.DataFrame(rows)
                                    df.columns = [col.upper() for col in df.columns]
                                    available_cols = [col.upper() for col in df.columns]
                                    requested_cols_upper = [col.upper() for col in columns]
                                    missing_cols = [col for col in requested_cols_upper if col not in available_cols]
                                    if missing_cols:
                                        logger.warning(f'Fallback successful: Returning all {len(df.columns)} available columns from {actual_schema}.{table_name}. Requested columns {columns} had {len(missing_cols)} missing: {missing_cols}. Available columns: {available_cols[:30]}...')
                                    else:
                                        logger.info(f'Fallback successful: Returning all {len(df.columns)} available columns from {actual_schema}.{table_name}.')
                                    return df
                                else:
                                    rows = []
                            else:
                                raise
                    else:
                        query = text(f'SELECT * FROM "{actual_schema}"."{table_name}"')
                        result = await session.execute(query)
                        rows = result.mappings().fetchall()
                else:
                    if columns:
                        cols_str = ', '.join([f'"{col}"' for col in columns])
                        query = text(f'SELECT {cols_str} FROM "{actual_schema}"."{table_name}"')
                    else:
                        query = text(f'SELECT * FROM "{actual_schema}"."{table_name}"')
                    result = await session.execute(query)
                    rows = result.mappings().fetchall()
                if rows:
                    return pd.DataFrame(rows)
                else:
                    if is_oracle:
                        metadata_query = text('\n                            SELECT COLUMN_NAME as column_name, DATA_TYPE as data_type \n                            FROM ALL_TAB_COLUMNS \n                            WHERE OWNER = :schema AND TABLE_NAME = :table_name\n                            ORDER BY COLUMN_ID\n                        ')
                    else:
                        metadata_query = text('\n                            SELECT column_name, data_type \n                            FROM information_schema.columns \n                            WHERE table_schema = :schema AND table_name = :table_name\n                            ORDER BY ordinal_position\n                        ')
                    col_result = await session.execute(metadata_query, {'schema': actual_schema, 'table_name': table_name})
                    col_rows = col_result.mappings().fetchall()
                    if col_rows:
                        col_names = [row['column_name'] for row in col_rows]
                        if is_oracle:
                            col_names = [col.upper() for col in col_names]
                        return pd.DataFrame(columns=col_names)
                    return pd.DataFrame()
        try:
            return asyncio.run(_read_table())
        except Exception as e:
            raise RuntimeError(f'Failed to execute async database operation: {str(e)}') from e
    except Exception as e:
        logger.error(f'Error reading table {schema}.{table_name}: {e}')
        raise RuntimeError(f'Failed to read table {schema}.{table_name}: {str(e)}')

def read_sql_query(sql_query: str, session_factory=None) -> pd.DataFrame:
    if session_factory is None:
        raise RuntimeError('Database connection not available. Please ensure store_in_local=False and database credentials are configured.')
    try:
        import asyncio

        async def _execute_query():
            async with session_factory() as session:
                is_oracle = session.bind.dialect.name == 'oracle'
                query = text(sql_query)
                result = await session.execute(query)
                rows = result.mappings().fetchall()
                if rows:
                    df = pd.DataFrame(rows)
                    if is_oracle:
                        df.columns = [col.upper() for col in df.columns]
                    return df
                else:
                    return pd.DataFrame()
        try:
            return asyncio.run(_execute_query())
        except Exception as e:
            raise RuntimeError(f'Failed to execute async database operation: {str(e)}') from e
    except Exception as e:
        logger.error(f'Error executing SQL query: {e}')
        raise RuntimeError(f'Failed to execute SQL query: {str(e)}')

def format_date_for_sap(date_input) -> int:
    if isinstance(date_input, int):
        return date_input
    if isinstance(date_input, str) and date_input.isdigit() and (len(date_input) == 8):
        return int(date_input)
    if isinstance(date_input, str):
        try:
            dt = datetime.strptime(date_input, '%Y-%m-%d')
        except ValueError:
            try:
                return int(date_input)
            except ValueError:
                raise ValueError(f"Invalid date format: {date_input}. Use 'YYYY-MM-DD' or 'YYYYMMDD'")
    elif isinstance(date_input, datetime):
        dt = date_input
    else:
        raise ValueError(f'Unsupported date type: {type(date_input)}')
    return int(dt.strftime('%Y%m%d'))

def verify_db_connection(session_factory=None) -> Dict[str, Any]:
    if session_factory is None:
        raise RuntimeError('Database connection not available. Please ensure store_in_local=False and database credentials are configured.')
    try:
        import asyncio

        async def _verify():
            async with session_factory() as session:
                is_mysql = session.bind.dialect.name == 'mysql'
                is_hana = session.bind.dialect.name == 'hana'
                if is_mysql:
                    db_name_query = 'SELECT database()'
                elif is_hana:
                    db_name_query = 'SELECT CURRENT_SCHEMA FROM DUMMY'
                else:
                    db_name_query = 'SELECT current_database()'
                db_name_result = await session.execute(text(db_name_query))
                db_name = db_name_result.scalar()
                if is_hana:
                    version_query = 'SELECT VERSION FROM SYS.M_DATABASE'
                else:
                    version_query = 'SELECT version()'
                version_result = await session.execute(text(version_query))
                version = version_result.scalar()
                if is_hana:
                    timestamp_query = 'SELECT CURRENT_TIMESTAMP FROM DUMMY'
                else:
                    timestamp_query = 'SELECT NOW()'
                timestamp_result = await session.execute(text(timestamp_query))
                db_timestamp = timestamp_result.scalar()
                conn_type = 'LIVE_POSTGRESQL'
                if is_mysql:
                    conn_type = 'LIVE_MYSQL'
                elif is_hana:
                    conn_type = 'LIVE_SAPHANA'
                return {'connected': True, 'database_name': db_name, 'server_version': version, 'current_timestamp': str(db_timestamp), 'connection_time': utcnow().isoformat(), 'connection_type': conn_type}
        try:
            return asyncio.run(_verify())
        except Exception as e:
            raise RuntimeError(f'Failed to verify database connection: {str(e)}') from e
    except Exception as e:
        logger.error(f'Error verifying database connection: {e}')
        raise RuntimeError(f'Failed to verify database connection: {str(e)}')