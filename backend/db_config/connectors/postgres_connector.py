import logging
from typing import Optional
from urllib.parse import urlparse
from util.sshtunnel_compat import ensure_sshtunnel_compat
ensure_sshtunnel_compat()
from sshtunnel import SSHTunnelForwarder
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from db_config.connectors.base import DatabaseConnector
logger = logging.getLogger(__name__)

class PostgresConnector(DatabaseConnector):
    name = 'postgres'

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
                tunnel_kwargs = {'ssh_address_or_host': (self._ssh_config['host'], self._ssh_config.get('port', 22)), 'ssh_username': self._ssh_config.get('username', ''), 'remote_bind_address': (parsed.hostname, parsed.port or 5432), 'local_bind_address': ('127.0.0.1', 0), 'allow_agent': False, 'host_pkey_directories': []}
                if self._ssh_config.get('auth_method') == 'private_key':
                    pkey = self._ssh_config.get('private_key') or self._ssh_config.get('private_key_content')
                    if pkey:
                        tunnel_kwargs['ssh_pkey'] = pkey
                    passphrase = self._ssh_config.get('private_key_passphrase')
                    if passphrase:
                        tunnel_kwargs['ssh_private_key_password'] = passphrase
                else:
                    tunnel_kwargs['ssh_password'] = self._ssh_config.get('password') or ''
                self._tunnel = SSHTunnelForwarder(**tunnel_kwargs)
                self._tunnel.start()
                local_port = self._tunnel.local_bind_port
                dsn_to_use = self.dsn
                if parsed.hostname:
                    remote_port = parsed.port or 5432
                    if parsed.port:
                        dsn_to_use = dsn_to_use.replace(f'{parsed.hostname}:{parsed.port}', f'127.0.0.1:{local_port}')
                    else:
                        dsn_to_use = dsn_to_use.replace(parsed.hostname, f'127.0.0.1:{local_port}', 1)
                logger.info(f'SSH tunnel established, using local port {local_port}')
            self._engine = create_async_engine(dsn_to_use, future=True, pool_size=5, max_overflow=10, pool_recycle=1800)
            self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False, class_=AsyncSession)
            async with self._engine.begin() as conn:
                await conn.run_sync(lambda _: None)
            logger.info('✅ Postgres connector initialized')
        except Exception as exc:
            logger.error('Failed to initialize Postgres connector: %s', exc, exc_info=True)
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
        logger.info(' Postgres connector closed')

    def get_db(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            raise RuntimeError('Postgres connector not initialized. Call connect() first.')
        return self._session_factory