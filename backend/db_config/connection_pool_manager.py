from __future__ import annotations
import logging
import asyncio
from typing import Dict, Optional, List
from db_config.connectors.postgres_connector import PostgresConnector
from db_config.connectors.mysql_connector import MySQLConnector
from db_config.connectors.sap_hana_connector import SAPHANAConnector
from db_config.connectors.oracle_connector import OracleConnector
from db_config.connectors.sybase_connector import SybaseConnector
from services.db_credentials_service import DBCredentialsService
logger = logging.getLogger(__name__)

class ConnectionPoolManager:
    _instance: Optional['ConnectionPoolManager'] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._pools: Dict[str, PostgresConnector | MySQLConnector | SAPHANAConnector | OracleConnector | SybaseConnector] = {}
        self._initialized = True

    async def get_connection(self, client_id: str, db, dataset_id: Optional[str]=None) -> PostgresConnector | MySQLConnector | SAPHANAConnector | OracleConnector | SybaseConnector:
        pool_key = f'{client_id}:{dataset_id}' if dataset_id else client_id
        if pool_key in self._pools:
            connector = self._pools[pool_key]
            try:
                if connector._engine is not None:
                    return connector
            except Exception:
                logger.warning(f"Stale connection detected for pool key '{pool_key}', recreating...")
                del self._pools[pool_key]
        async with self._lock:
            if pool_key in self._pools:
                return self._pools[pool_key]
            service = DBCredentialsService(db)
            credentials = await service.get_credentials(client_id=client_id, db_type=None, decrypt_password=True, dataset_id=dataset_id)
            if not credentials:
                raise RuntimeError(f'No database credentials found for client {client_id}. Please configure database credentials first.')
            db_type = credentials.get('db_type', 'postgres')
            db_url = credentials.get('db_url')
            if not db_url:
                db_host = credentials.get('db_host')
                db_port = credentials.get('db_port')
                db_name = credentials.get('db_name')
                db_username = credentials.get('db_username')
                db_password = credentials.get('db_password')
                if db_type == 'sap_sybase':
                    if not all([db_host, db_port, db_username, db_password]):
                        raise RuntimeError(f'Incomplete database credentials for client {client_id}. Missing required fields (db_name is optional for sap_sybase).')
                elif not all([db_host, db_port, db_name, db_username, db_password]):
                    raise RuntimeError(f'Incomplete database credentials for client {client_id}. Missing required fields.')
                if db_type == 'postgres':
                    db_url = f'postgresql+asyncpg://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
                elif db_type == 'mysql':
                    db_url = f'mysql+aiomysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
                elif db_type == 'sap_hana':
                    db_url = f'hana://{db_username}:{db_password}@{db_host}:{db_port}'
                elif db_type == 'sap_oracle':
                    db_url = f'oracle+oracledb_async://{db_username}:{db_password}@{db_host}:{db_port}/?service_name={db_name}'
                elif db_type == 'sap_sybase':
                    from db_config.connectors.sybase_connector import build_sybase_url
                    db_url = build_sybase_url(db_host, db_port, db_username, db_password, db_name)
                else:
                    raise RuntimeError(f'Unsupported database type for connection: {db_type}')
            additional_params_creds = credentials.get('additional_params') or {}
            ssh_config = additional_params_creds.get('ssh')
            try:
                if db_type == 'postgres':
                    connector = PostgresConnector(db_url, ssh_config=ssh_config)
                elif db_type == 'mysql':
                    connector = MySQLConnector(db_url, ssh_config=ssh_config)
                elif db_type == 'sap_hana':
                    connector = SAPHANAConnector(db_url)
                elif db_type == 'sap_oracle':
                    connector = OracleConnector(db_url, ssh_config=ssh_config)
                elif db_type == 'sap_sybase':
                    connector = SybaseConnector(db_url)
                else:
                    raise RuntimeError(f'Unsupported database type: {db_type}')
                await connector.connect()
            except Exception as e:
                logger.error(f'Failed to connect to {db_type} for client {client_id}: {e}', exc_info=True)
                raise RuntimeError(f'Failed to establish {db_type} connection for client {client_id}: {str(e)}')
            self._pools[pool_key] = connector
            return connector

    async def disconnect_client(self, client_id: str) -> None:
        async with self._lock:
            keys_to_remove: List[str] = [k for k in self._pools if k == client_id or k.startswith(f'{client_id}:')]
            for key in keys_to_remove:
                connector = self._pools[key]
                try:
                    await connector.disconnect()
                except Exception as e:
                    logger.error(f'Error disconnecting pool key {key}: {e}')
                del self._pools[key]

    async def disconnect_all(self) -> None:
        async with self._lock:
            for client_id, connector in list(self._pools.items()):
                try:
                    await connector.disconnect()
                except Exception as e:
                    logger.error(f'Error disconnecting client {client_id}: {e}')
            self._pools.clear()

    def has_connection(self, client_id: str, dataset_id: Optional[str]=None) -> bool:
        pool_key = f'{client_id}:{dataset_id}' if dataset_id else client_id
        return pool_key in self._pools

    def get_connection_count(self) -> int:
        return len(self._pools)