import os
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError
import logging
from dotenv import load_dotenv
logger = logging.getLogger(__name__)

class DatabaseManager:

    def __init__(self, connection_string: str=None):
        load_dotenv()
        if not connection_string:
            db_user = os.getenv('DB_USERNAME')
            db_password = os.getenv('DB_PASSWORD')
            db_host = os.getenv('DB_HOST', 'localhost')
            db_port = os.getenv('DB_PORT', '1521')
            db_service = os.getenv('DB_SERVICE_NAME', 'ORCL')
            connection_string = f'oracle+cx_oracle://{db_user}:{db_password}@{db_host}:{db_port}/?service_name={db_service}'
        self.connection_string = connection_string
        self.engine = self._create_engine(connection_string)
        self.Session = scoped_session(sessionmaker(bind=self.engine))

    def _create_engine(self, connection_string: str) -> Engine:
        engine_kwargs = {'pool_size': 5, 'max_overflow': 10, 'pool_timeout': 30, 'pool_recycle': 1800, 'pool_pre_ping': True, 'echo': False}
        try:
            engine = create_engine(connection_string, **engine_kwargs)
            with engine.connect() as conn:
                conn.execute(text('SELECT 1 FROM DUAL'))
            logger.info('Successfully connected to the database')
            return engine
        except Exception as e:
            logger.error(f'Failed to connect to the database: {e}')
            raise

    def execute_query(self, query: str, params: Optional[Dict]=None) -> List[Dict]:
        session = None
        try:
            session = self.Session()
            query = query.strip()
            if query.endswith(';'):
                query = query[:-1].strip()
            logger.debug(f'Executing query: {query}')
            result = session.execute(text(query), params or {})
            if result.returns_rows is False:
                return []
            columns = list(result.keys())
            return [dict(zip(columns, row)) for row in result.fetchall()]
        except SQLAlchemyError as e:
            if session:
                session.rollback()
            logger.error(f'Error executing query: {e}\nQuery: {query}')
            raise
        finally:
            if session:
                session.close()

    def close(self):
        self.Session.remove()
        self.engine.dispose()
        logger.info('Database connections closed')
db_manager = None

def get_db_manager() -> DatabaseManager:
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager