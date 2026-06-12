import logging
import asyncio
from typing import Optional, Any
from sqlalchemy import create_engine, text, URL
from sqlalchemy.orm import sessionmaker
from db_config.connectors.base import DatabaseConnector
logger = logging.getLogger(__name__)

def build_sybase_url(db_host: str, db_port: int, db_username: str, db_password: str, db_name: Optional[str]=None, driver: str='Adaptive Server Enterprise') -> str:
    odbc_conn_str = f'DRIVER={{{driver}}};SERVER={db_host};PORT={db_port};UID={db_username};PWD={db_password};'
    if db_name:
        odbc_conn_str += f'DATABASE={db_name};'
    return odbc_conn_str

class SybaseConnector(DatabaseConnector):
    name = 'sybase'

    def __init__(self, dsn: str, schema: Optional[str]=None):
        super().__init__(dsn, schema)
        self._engine = None
        self._session_maker = None

    async def connect(self) -> None:
        if self._engine is not None:
            return
        try:
            from urllib.parse import urlparse, parse_qs, unquote
            odbc_conn_str = None
            if 'odbc_connect=' in self.dsn:
                parsed = urlparse(self.dsn)
                query_params = parse_qs(parsed.query)
                odbc_conn_str = unquote(query_params.get('odbc_connect', [None])[0])
                if not odbc_conn_str:
                    raise ValueError('Could not extract odbc_connect from URL')
            elif self.dsn.startswith('sybase+pyodbc://') or self.dsn.startswith('sybase://'):
                parsed = urlparse(self.dsn)
                db_host = parsed.hostname
                if not db_host:
                    raise ValueError(f'Could not parse hostname from URL: {self.dsn}')
                db_port = parsed.port or 5000
                db_username = parsed.username or ''
                db_password = parsed.password or ''
                db_name = parsed.path.lstrip('/') if parsed.path and parsed.path != '/' else None
                odbc_conn_str = build_sybase_url(db_host=db_host, db_port=db_port, db_username=db_username, db_password=db_password, db_name=db_name)
                logger.info(f'Parsed SQLAlchemy URL and built ODBC connection string')
            elif self.dsn.startswith('DRIVER='):
                odbc_conn_str = self.dsn
            else:
                logger.warning(f'Unknown DSN format, attempting to use as-is: {self.dsn[:50]}...')
                odbc_conn_str = self.dsn
            safe_conn_str = odbc_conn_str
            if 'PWD=' in safe_conn_str:
                import re
                safe_conn_str = re.sub('PWD=[^;]+', 'PWD=***', safe_conn_str)
            logger.info(f'Sybase ODBC connection string: {safe_conn_str}')
            import pyodbc

            def pyodbc_creator():
                return pyodbc.connect(odbc_conn_str)
            logger.info('Attempting to connect using custom pyodbc creator function')
            self._engine = create_engine('sybase+pyodbc://', creator=pyodbc_creator, pool_size=5, max_overflow=10, pool_timeout=30, pool_recycle=1800, pool_pre_ping=True, echo=False)
            self._session_maker = sessionmaker(bind=self._engine)
            connection = self._engine.connect()
            try:
                connection.execute(text('SELECT 1'))
            finally:
                connection.close()
            logger.info('✅ Sybase connector initialized')
        except Exception as exc:
            logger.error('Failed to initialize Sybase connector: %s', exc, exc_info=True)
            raise

    async def disconnect(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_maker = None
            logger.info(' Sybase connector closed')

    def get_db(self):
        if self._session_maker is None:
            raise RuntimeError('Sybase connector not initialized. Call connect() first.')

        class AsyncSessionWrapper:

            def __init__(self, sync_session):
                self._session = sync_session

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await asyncio.to_thread(self._session.close)

            async def execute(self, statement, params=None, **kwargs):
                return await asyncio.to_thread(self._session.execute, statement, params, **kwargs)

            def __getattr__(self, name):
                return getattr(self._session, name)

        def async_session_factory():
            sync_session = self._session_maker()
            return AsyncSessionWrapper(sync_session)
        return async_session_factory