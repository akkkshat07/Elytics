import logging
from typing import Optional
from urllib.parse import urlparse
from util.sshtunnel_compat import ensure_sshtunnel_compat
ensure_sshtunnel_compat()
from sshtunnel import SSHTunnelForwarder
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from db_config.connectors.base import DatabaseConnector
logger = logging.getLogger(__name__)

class MongoConnector(DatabaseConnector):
    name = 'mongo'

    def __init__(self, dsn: str, db_name: str, ssh_config: Optional[dict]=None):
        super().__init__(dsn, db_name)
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
        self._ssh_config = ssh_config
        self._tunnel = None

    async def connect(self) -> None:
        if self._client is not None:
            return
        try:
            dsn_to_use = self.dsn
            if self._ssh_config and self._ssh_config.get('enabled'):
                parsed = urlparse(self.dsn)
                remote_host = parsed.hostname
                remote_port = parsed.port or 27017
                self._tunnel = SSHTunnelForwarder((self._ssh_config['host'], self._ssh_config.get('port', 22)), ssh_username=self._ssh_config['username'], ssh_password=self._ssh_config.get('password'), ssh_pkey=self._ssh_config.get('private_key'), remote_bind_address=(remote_host, remote_port))
                self._tunnel.start()
                local_port = self._tunnel.local_bind_port
                if remote_host and remote_port:
                    dsn_to_use = self.dsn.replace(f'{remote_host}:{remote_port}', f'127.0.0.1:{local_port}')
                logger.info(f'SSH tunnel established, using local port {local_port}')
            self._client = AsyncIOMotorClient(dsn_to_use, maxPoolSize=50, minPoolSize=10, maxIdleTimeMS=60000, serverSelectionTimeoutMS=5000, connectTimeoutMS=10000, socketTimeoutMS=60000)
            self._db = self._client[self.db_name] if self.db_name else self._client.get_default_database()
            logger.info('✅ Mongo connector initialized (db=%s)', self.db_name)
        except Exception as exc:
            logger.error('Failed to initialize Mongo connector: %s', exc, exc_info=True)
            raise

    async def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
        if self._tunnel is not None:
            self._tunnel.stop()
            self._tunnel = None
            logger.info('SSH tunnel closed')
        logger.info(' Mongo connector closed')

    def get_db(self) -> AsyncIOMotorDatabase:
        if self._db is None:
            raise RuntimeError('Mongo connector not initialized. Call connect() first.')
        return self._db