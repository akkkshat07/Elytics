import logging
from typing import Optional
from urllib.parse import urlparse
from util.sshtunnel_compat import ensure_sshtunnel_compat
ensure_sshtunnel_compat()
from sshtunnel import SSHTunnelForwarder
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from db_config.connectors.base import DatabaseConnector
logger = logging.getLogger(__name__)

class MySQLConnector(DatabaseConnector):
    name = 'mysql'

    def __init__(self, dsn: str, schema: Optional[str]=None, ssh_config: Optional[dict]=None):
        super().__init__(dsn, schema)
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._ssh_config = ssh_config
        self._tunnel = None

    async def connect(self) -> None:
        if self._engine is not None:
            return
        try:
            dsn_to_use = self.dsn
            if self._ssh_config and self._ssh_config.get('enabled'):
                parsed = urlparse(self.dsn)
                self._tunnel = SSHTunnelForwarder((self._ssh_config['host'], self._ssh_config.get('port', 22)), ssh_username=self._ssh_config['username'], ssh_password=self._ssh_config.get('password'), ssh_pkey=self._ssh_config.get('private_key'), remote_bind_address=(parsed.hostname, parsed.port))
                self._tunnel.start()
                local_port = self._tunnel.local_bind_port
                if parsed.hostname and parsed.port:
                    dsn_to_use = self.dsn.replace(f'{parsed.hostname}:{parsed.port}', f'127.0.0.1:{local_port}')
                logger.info(f'SSH tunnel established, using local port {local_port}')
            self._engine = create_async_engine(dsn_to_use, future=True, pool_size=5, max_overflow=10, pool_recycle=1800)
            self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False, class_=AsyncSession)
            async with self._engine.begin() as conn:
                await conn.execute(text('SELECT 1'))
            logger.info('✅ MySQL connector initialized')
        except Exception as exc:
            logger.error('Failed to initialize MySQL connector: %s', exc, exc_info=True)
            raise

    async def disconnect(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
        if self._tunnel is not None:
            self._tunnel.stop()
            self._tunnel = None
            logger.info('SSH tunnel closed')
        logger.info(' MySQL connector closed')

    def get_db(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            raise RuntimeError('MySQL connector not initialized. Call connect() first.')
        return self._session_factory