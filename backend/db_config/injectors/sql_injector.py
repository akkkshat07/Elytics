from __future__ import annotations
import logging
import asyncio
import pandas as pd
from typing import Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from db_config.connectors.postgres_connector import PostgresConnector
from db_config.connectors.mysql_connector import MySQLConnector
from db_config.connectors.oracle_connector import OracleConnector
from db_config.connectors.sap_hana_connector import SAPHANAConnector
from db_config.connectors.sybase_connector import SybaseConnector
logger = logging.getLogger(__name__)

class SQLInjector:

    def __init__(self, db_type: str, db_credential: Optional[Dict[str, Any]]=None, connector: Optional[PostgresConnector | MySQLConnector | SAPHANAConnector | OracleConnector | SybaseConnector]=None):
        if db_type not in ['postgres', 'mysql', 'sap_oracle', 'sap_hana', 'sap_sybase']:
            raise ValueError(f"db_type must be 'postgres', 'mysql', 'sap_oracle', 'sap_hana' or 'sap_sybase', got '{db_type}'")
        self.db_type = db_type
        self._connected = False
        if connector:
            self.connector = connector
            self.db_credential = db_credential or {}
            self.db_url = getattr(connector, 'db_url', None)
            if hasattr(connector, '_engine') and connector._engine is not None:
                self._connected = True
        else:
            if not db_credential:
                raise ValueError('db_credential is required if connector is not provided')
            self.db_credential = db_credential
            self.connector = None
            db_url = db_credential.get('db_url')
            if not db_url:
                db_host = db_credential.get('db_host')
                db_port = db_credential.get('db_port')
                db_name = db_credential.get('db_name')
                db_username = db_credential.get('db_username')
                db_password = db_credential.get('db_password')
                if db_type == 'sap_sybase':
                    if not all([db_host, db_port, db_username, db_password]):
                        raise ValueError('Either db_url must be provided, or all of db_host, db_port, db_username, and db_password must be provided (db_name is optional for sap_sybase).')
                elif not all([db_host, db_port, db_name, db_username, db_password]):
                    raise ValueError('Either db_url must be provided, or all of db_host, db_port, db_name, db_username, and db_password must be provided.')
                if db_type == 'postgres':
                    db_url = f'postgresql+asyncpg://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
                elif db_type == 'mysql':
                    db_url = f'mysql+aiomysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
                elif db_type == 'sap_oracle':
                    db_url = f'oracle+oracledb://{db_username}:{db_password}@{db_host}:{db_port}/?service_name={db_name}'
                elif db_type == 'sap_hana':
                    db_url = f'hana://{db_username}:{db_password}@{db_host}:{db_port}'
                    if int(db_port) == 443:
                        db_url += '?encrypt=true&sslValidateCertificate=false'
                elif db_type == 'sap_sybase':
                    from db_config.connectors.sybase_connector import build_sybase_url
                    db_url = build_sybase_url(db_host, db_port, db_username, db_password, db_name)
            self.db_url = db_url
            _ssh_config = (db_credential.get('additional_params') or {}).get('ssh')
            if db_type == 'postgres':
                self.connector = PostgresConnector(self.db_url, ssh_config=_ssh_config)
            elif db_type == 'mysql':
                self.connector = MySQLConnector(self.db_url, ssh_config=_ssh_config)
            elif db_type == 'sap_oracle':
                self.connector = OracleConnector(self.db_url, ssh_config=_ssh_config)
            elif db_type == 'sap_hana':
                self.connector = SAPHANAConnector(self.db_url)
            elif db_type == 'sap_sybase':
                self.connector = SybaseConnector(self.db_url)

    async def connect(self) -> None:
        if self._connected and self.connector:
            return
        if not self.connector:
            raise RuntimeError('Connector not initialized')
        await self.connector.connect()
        self._connected = True
        logger.info(f'SQLInjector connected to {self.db_type}')

    async def disconnect(self) -> None:
        if self.connector and self._connected:
            await self.connector.disconnect()
            self._connected = False
            logger.info(f'SQLInjector disconnected from {self.db_type}')

    def read_table(self, table_name: str, schema: str='public') -> Dict[str, Any]:
        if not self._connected or not self.connector:
            raise RuntimeError('Not connected to database. Call connect() first.')

        async def _read_table_metadata():
            session_factory = self.connector.get_db()
            async with session_factory() as session:
                is_mysql = session.bind.dialect.name == 'mysql'
                is_oracle = session.bind.dialect.name == 'oracle'
                is_hana = session.bind.dialect.name == 'hana'
                is_sybase = session.bind.dialect.name == 'sybase'
                actual_schema = schema
                if is_mysql and (schema is None or schema == 'public'):
                    db_name_result = await session.execute(text('SELECT DATABASE()'))
                    actual_schema = db_name_result.scalar()
                elif is_oracle and (schema is None or schema == 'public'):
                    schema_result = await session.execute(text('SELECT USER FROM DUAL'))
                    actual_schema = schema_result.scalar()
                if is_mysql:
                    metadata_query = text('\n                        SELECT \n                            column_name,\n                            data_type,\n                            is_nullable,\n                            column_default,\n                            character_maximum_length,\n                            numeric_precision,\n                            numeric_scale\n                        FROM information_schema.columns\n                        WHERE table_schema = :schema AND table_name = :table_name\n                        ORDER BY ordinal_position\n                    ')
                elif is_oracle:
                    if actual_schema:
                        actual_schema = actual_schema.upper()
                    metadata_query = text('\n                        SELECT \n                            column_name as "column_name",\n                            data_type as "data_type",\n                            nullable as "is_nullable",\n                            data_default as "column_default",\n                            data_length as "character_maximum_length",\n                            data_precision as "numeric_precision",\n                            data_scale as "numeric_scale"\n                        FROM all_tab_columns\n                        WHERE table_name = UPPER(:table_name)\n                        ' + ('AND owner = :schema' if actual_schema else '') + '\n                        ORDER BY column_id\n                    ')
                elif is_hana:
                    if actual_schema:
                        actual_schema = actual_schema.upper()
                    metadata_query = text('\n                        SELECT \n                            COLUMN_NAME as "column_name",\n                            DATA_TYPE_NAME as "data_type",\n                            IS_NULLABLE as "is_nullable",\n                            DEFAULT_VALUE as "column_default",\n                            LENGTH as "character_maximum_length",\n                            SCALE as "numeric_scale",\n                            0 as "numeric_precision" -- HANA separates these differently sometimes, simplistic fallback\n                        FROM SYS.TABLE_COLUMNS\n                        WHERE TABLE_NAME = UPPER(:table_name)\n                        ' + ('AND SCHEMA_NAME = :schema' if actual_schema else '') + '\n                        ORDER BY POSITION\n                    ')
                elif is_sybase:
                    if actual_schema:
                        actual_schema = actual_schema.upper()
                    metadata_query = text('\n                        SELECT \n                            c.name as "column_name",\n                            t.name as "data_type",\n                            CASE WHEN c.status & 8 = 8 THEN \'NO\' ELSE \'YES\' END as "is_nullable",\n                            NULL as "column_default",\n                            c.length as "character_maximum_length",\n                            c.scale as "numeric_scale",\n                            c.prec as "numeric_precision"\n                        FROM syscolumns c\n                        INNER JOIN systypes t ON c.type = t.type\n                        INNER JOIN sysobjects o ON c.id = o.id\n                        WHERE o.name = :table_name\n                        ' + ('AND o.uid = USER_ID(:schema)' if actual_schema else '') + '\n                        ORDER BY c.colid\n                    ')
                else:
                    metadata_query = text('\n                        SELECT \n                            column_name,\n                            data_type,\n                            is_nullable,\n                            column_default,\n                            character_maximum_length,\n                            numeric_precision,\n                            numeric_scale\n                        FROM information_schema.columns\n                        WHERE table_schema = :schema AND table_name = :table_name\n                        ORDER BY ordinal_position\n                    ')
                query_params = {'table_name': table_name}
                if actual_schema:
                    query_params['schema'] = actual_schema
                result = await session.execute(metadata_query, query_params)
                rows = result.mappings().fetchall()
                if not rows:
                    raise ValueError(f"Table '{actual_schema}.{table_name}' not found")
                columns = []
                for row in rows:
                    is_nullable = row['is_nullable']
                    if isinstance(is_nullable, str):
                        is_nullable = is_nullable.upper() in ('YES', 'Y', 'TRUE')
                    columns.append({'column_name': row['column_name'], 'data_type': row['data_type'], 'is_nullable': bool(is_nullable), 'column_default': row['column_default'], 'character_maximum_length': row['character_maximum_length'], 'numeric_precision': row['numeric_precision'], 'numeric_scale': row['numeric_scale']})
                return {'table_name': table_name, 'schema': actual_schema, 'columns': columns, 'column_count': len(columns)}
        try:
            return asyncio.run(_read_table_metadata())
        except Exception as e:
            logger.error(f'Error reading table metadata for {schema}.{table_name}: {e}')
            raise RuntimeError(f'Failed to read table metadata: {str(e)}')

    def query_sql(self, sql_query: str) -> pd.DataFrame:
        if not self._connected or not self.connector:
            raise RuntimeError('Not connected to database. Call connect() first.')

        async def _execute_query():
            session_factory = self.connector.get_db()
            async with session_factory() as session:
                query = text(sql_query)
                result = await session.execute(query)
                rows = result.mappings().fetchall()
                if rows:
                    return pd.DataFrame(rows)
                else:
                    return pd.DataFrame()
        try:
            return asyncio.run(_execute_query())
        except Exception as e:
            logger.error(f'Error executing SQL query: {e}')
            raise RuntimeError(f'Failed to execute SQL query: {str(e)}')

    def test_connection(self) -> Dict[str, Any]:
        if not self._connected or not self.connector:
            raise RuntimeError('Not connected to database. Call connect() first.')

        async def _verify():
            session_factory = self.connector.get_db()
            async with session_factory() as session:
                is_mysql = session.bind.dialect.name == 'mysql'
                if is_mysql:
                    db_name_query = 'SELECT database()'
                elif self.db_type == 'sap_oracle':
                    db_name_query = "SELECT sys_context('USERENV', 'DB_NAME') FROM dual"
                elif self.db_type == 'sap_hana':
                    db_name_query = 'SELECT DATABASE_NAME FROM SYS.M_DATABASE'
                elif self.db_type == 'sap_sybase':
                    db_name_query = 'SELECT DB_NAME()'
                else:
                    db_name_query = 'SELECT current_database()'
                db_name_result = await session.execute(text(db_name_query))
                db_name = db_name_result.scalar()
                if self.db_type == 'oracle':
                    version_query = 'SELECT version FROM v$instance'
                elif self.db_type == 'sap_hana':
                    version_query = 'SELECT VERSION FROM SYS.M_DATABASE'
                elif self.db_type == 'sap_sybase':
                    version_query = 'SELECT @@version'
                else:
                    version_query = 'SELECT version()'
                version_result = await session.execute(text(version_query))
                version = version_result.scalar()
                if is_mysql:
                    timestamp_query = 'SELECT NOW()'
                elif self.db_type == 'oracle':
                    timestamp_query = 'SELECT CURRENT_TIMESTAMP FROM dual'
                elif self.db_type == 'sap_hana':
                    timestamp_query = 'SELECT CURRENT_TIMESTAMP FROM DUMMY'
                elif self.db_type == 'sap_sybase':
                    timestamp_query = 'SELECT GETDATE()'
                else:
                    timestamp_query = 'SELECT NOW()'
                timestamp_result = await session.execute(text(timestamp_query))
                db_timestamp = timestamp_result.scalar()
                conn_type = 'LIVE_MYSQL' if is_mysql else 'LIVE_POSTGRESQL'
                if self.db_type == 'oracle':
                    conn_type = 'LIVE_ORACLE'
                if self.db_type == 'sap_hana':
                    conn_type = 'LIVE_SAP_HANA'
                if self.db_type == 'sap_sybase':
                    conn_type = 'LIVE_SAP_SYBASE'
                return {'connected': True, 'database_name': db_name, 'server_version': version, 'current_timestamp': str(db_timestamp), 'connection_type': conn_type, 'db_type': self.db_type}
        try:
            return asyncio.run(_verify())
        except Exception as e:
            logger.error(f'Error verifying database connection: {e}')
            raise RuntimeError(f'Failed to verify database connection: {str(e)}')