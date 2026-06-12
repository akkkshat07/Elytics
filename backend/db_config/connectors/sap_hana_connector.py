import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_config.connectors.base import DatabaseConnector
logger = logging.getLogger(__name__)

class SAPHANAConnector(DatabaseConnector):

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._engine = None
        self._session_maker = None

    async def connect(self):
        try:
            self._engine = create_engine(self.db_url)
            connection = self._engine.connect()
            connection.close()
            self._session_maker = sessionmaker(bind=self._engine)
            logger.info('Successfully connected to SAP HANA')
        except Exception as e:
            logger.error(f'Error connecting to SAP HANA: {e}')
            raise e

    async def disconnect(self):
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info('Disconnected from SAP HANA')

    def get_db(self):
        if not self._session_maker:
            raise Exception('Database not connected')
        import asyncio

        class AsyncSessionWrapper:

            def __init__(self, sync_session):
                self._session = sync_session

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await asyncio.to_thread(self._session.close)

            async def execute(self, statement, params=None, **kwargs):
                return await asyncio.to_thread(self._session.execute, statement, params, **kwargs)

            def mappings(self):
                return self

            def __getattr__(self, name):
                return getattr(self._session, name)

        def async_session_factory():
            sync_session = self._session_maker()
            return AsyncSessionWrapper(sync_session)
        return async_session_factory