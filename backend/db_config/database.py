import os
import logging
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI
from contextlib import asynccontextmanager
from db_config.connectors.base import DatabaseConnector
from db_config.connectors.mongo_connector import MongoConnector
from db_config.connectors.postgres_connector import PostgresConnector
load_dotenv()
logger = logging.getLogger(__name__)
DATABASE_PROVIDER = 'mongo'
DATABASE_URL = os.getenv('DATABASE_URL', 'mongodb://localhost:27017')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'core-sight')
POSTGRES_URL = os.getenv('POSTGRES_DATABASE_URL', 'postgresql+asyncpg://postgres:postgres@localhost:5432/postgres')
_connector: Optional[DatabaseConnector] = None
_mongo_manager: Optional['MongoDBManager'] = None

def _build_connector() -> DatabaseConnector:
    if DATABASE_PROVIDER == 'mongo':
        return MongoConnector(DATABASE_URL, DATABASE_NAME)
    if DATABASE_PROVIDER == 'postgres':
        return PostgresConnector(POSTGRES_URL)
    raise ValueError(f"Unsupported database provider '{DATABASE_PROVIDER}'")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _connector, _mongo_manager
    _connector = _build_connector()
    logger.info('Initializing %s database connector', _connector.name)
    await _connector.connect()
    if DATABASE_PROVIDER == 'mongo':
        from util.Mongodb import MongoDBManager
        _mongo_manager = MongoDBManager()
        await _mongo_manager.connect()
        logger.info('Initialized MongoDBManager for LLM config access')
    if DATABASE_PROVIDER == 'mongo' and _mongo_manager and (_mongo_manager.db is not None):
        try:
            from db_config.indexes import ensure_indexes
            await ensure_indexes(_mongo_manager.db)
        except Exception as _ie:
            logger.warning('Index creation failed (non-fatal): %s', _ie)
    try:
        from services.llm_metrics_service import llm_metrics_service
        if _mongo_manager:
            llm_metrics_service.start(_mongo_manager)
            logger.info('LLM metrics service started')
    except Exception as _me:
        logger.warning('Failed to start LLM metrics service: %s', _me)
    try:
        yield
    finally:
        try:
            from services.llm_metrics_service import llm_metrics_service
            await llm_metrics_service.stop()
        except Exception:
            pass
        await _connector.disconnect()
        _connector = None
        if _mongo_manager:
            if _mongo_manager.client:
                _mongo_manager.client.close()
            _mongo_manager = None

def get_db():
    if _connector is None:
        raise RuntimeError('Database connector not initialized. Ensure FastAPI lifespan runs.')
    return _connector.get_db()

def get_mongo_manager():
    if _mongo_manager is None:
        raise RuntimeError('MongoDBManager not initialized. Ensure FastAPI lifespan runs with mongo provider.')
    return _mongo_manager