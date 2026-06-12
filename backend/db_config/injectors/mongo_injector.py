import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from util.time_utils import utcnow
from db_config.connectors.mongo_connector import MongoConnector
from motor.motor_asyncio import AsyncIOMotorDatabase
logger = logging.getLogger(__name__)

class MongoInjector:

    def __init__(self, db_credential: Dict[str, Any]):
        db_type = db_credential.get('db_type')
        if db_type != 'mongodb':
            raise ValueError(f"db_type must be 'mongodb', got '{db_type}'")
        self.db_credential = db_credential
        self.connector: Optional[MongoConnector] = None
        self._connected = False
        db_url = db_credential.get('db_url')
        if not db_url:
            db_host = db_credential.get('db_host')
            db_port = db_credential.get('db_port')
            db_name = db_credential.get('db_name')
            db_username = db_credential.get('db_username')
            db_password = db_credential.get('db_password')
            if not all([db_host, db_name, db_username, db_password]):
                raise ValueError('Either db_url must be provided, or all of db_host, db_name, db_username, and db_password must be provided.')
            if 'mongodb+srv' in str(db_host) or 'mongodb.net' in str(db_host):
                db_url = f'mongodb+srv://{db_username}:{db_password}@{db_host}/{db_name}'
            else:
                port = db_port or 27017
                db_url = f'mongodb://{db_username}:{db_password}@{db_host}:{port}/{db_name}'
        self.db_url = db_url
        self.db_name = db_credential.get('db_name', '')
        _ssh_config = (db_credential.get('additional_params') or {}).get('ssh')
        self.connector = MongoConnector(self.db_url, self.db_name, ssh_config=_ssh_config)

    async def connect(self) -> None:
        if self._connected and self.connector:
            return
        if not self.connector:
            raise RuntimeError('Connector not initialized')
        await self.connector.connect()
        self._connected = True
        logger.info('MongoInjector connected to MongoDB')

    async def disconnect(self) -> None:
        if self.connector and self._connected:
            await self.connector.disconnect()
            self._connected = False
            logger.info('MongoInjector disconnected from MongoDB')

    def _get_db(self) -> AsyncIOMotorDatabase:
        if not self._connected or not self.connector:
            raise RuntimeError('Not connected to database. Call connect() first.')
        return self.connector.get_db()

    def find(self, collection_name: str, filter_query: Optional[Dict[str, Any]]=None, projection: Optional[Dict[str, Any]]=None, sort: Optional[List[tuple]]=None, limit: Optional[int]=None) -> List[Dict[str, Any]]:

        async def _find():
            db = self._get_db()
            collection = db[collection_name]
            find_kwargs = {}
            if projection:
                find_kwargs['projection'] = projection
            cursor = collection.find(filter_query or {}, **find_kwargs)
            if sort:
                cursor = cursor.sort(sort)
            if limit:
                cursor = cursor.limit(limit)
            results = await cursor.to_list(length=limit or 10000)
            return results
        try:
            return asyncio.run(_find())
        except Exception as e:
            logger.error(f"Error executing find on collection '{collection_name}': {e}")
            raise RuntimeError(f'Failed to execute find: {str(e)}')

    def find_one(self, collection_name: str, filter_query: Optional[Dict[str, Any]]=None, projection: Optional[Dict[str, Any]]=None) -> Optional[Dict[str, Any]]:

        async def _find_one():
            db = self._get_db()
            collection = db[collection_name]
            result = await collection.find_one(filter_query or {}, projection)
            return result
        try:
            return asyncio.run(_find_one())
        except Exception as e:
            logger.error(f"Error executing find_one on collection '{collection_name}': {e}")
            raise RuntimeError(f'Failed to execute find_one: {str(e)}')

    def aggregate(self, collection_name: str, aggregate_queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        async def _aggregate():
            db = self._get_db()
            collection = db[collection_name]
            cursor = collection.aggregate(aggregate_queries)
            results = await cursor.to_list(length=10000)
            return results
        try:
            return asyncio.run(_aggregate())
        except Exception as e:
            logger.error(f"Error executing aggregate on collection '{collection_name}': {e}")
            raise RuntimeError(f'Failed to execute aggregate: {str(e)}')

    def verify_db_connection(self) -> Dict[str, Any]:

        async def _verify():
            db = self._get_db()
            result = await db.command('ping')
            server_info = await db.client.server_info()
            stats = await db.command('dbStats')
            return {'connected': True, 'database_name': self.db_name, 'server_version': server_info.get('version', 'unknown'), 'current_timestamp': utcnow().isoformat(), 'connection_type': 'LIVE_MONGODB', 'ping_result': result, 'database_size': stats.get('dataSize', 0)}
        try:
            return asyncio.run(_verify())
        except Exception as e:
            logger.error(f'Error verifying MongoDB connection: {e}')
            raise RuntimeError(f'Failed to verify MongoDB connection: {str(e)}')