import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
from util.time_utils import utcnow
logger = logging.getLogger(__name__)
CACHE_TTL_SECONDS = 300

@dataclass
class ClientConfig:
    client_id: str
    name: str
    enabled: bool = True
    inherits_from: Optional[str] = None
    prompt_base_path: str = 'xml_prompts/base'
    customizations: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    business_insights_sections: Dict[str, bool] = field(default_factory=lambda: {'summary': True, 'metrics': True, 'insights': True, 'recommendations': True, 'follow_ups': True, 'note': True})
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {'client_id': self.client_id, 'name': self.name, 'enabled': self.enabled, 'inherits_from': self.inherits_from, 'prompt_base_path': self.prompt_base_path, 'customizations': self.customizations, 'metadata': self.metadata, 'business_insights_sections': self.business_insights_sections, 'created_at': self.created_at, 'updated_at': self.updated_at, 'created_by': self.created_by, 'updated_by': self.updated_by}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClientConfig':
        return cls(client_id=data.get('client_id'), name=data.get('name'), enabled=data.get('enabled', True), inherits_from=data.get('inherits_from'), prompt_base_path=data.get('prompt_base_path', 'xml_prompts/base'), customizations=data.get('customizations', {}), metadata=data.get('metadata', {}), business_insights_sections=data.get('business_insights_sections', {'summary': True, 'metrics': True, 'insights': True, 'recommendations': True, 'follow_ups': True, 'note': True}), created_at=data.get('created_at'), updated_at=data.get('updated_at'), created_by=data.get('created_by'), updated_by=data.get('updated_by'))

@dataclass
class CachedConfig:
    config: ClientConfig
    cached_at: datetime
    ttl_seconds: int = CACHE_TTL_SECONDS

    def is_expired(self) -> bool:
        expiry_time = self.cached_at + timedelta(seconds=self.ttl_seconds)
        return utcnow() > expiry_time

class ClientConfigManager:

    def __init__(self, db):
        self.db = db
        self._cache: Dict[str, CachedConfig] = {}
        self._lock = asyncio.Lock()
        logger.info('ClientConfigManager initialized')

    async def get_client_config(self, client_id: str, use_cache: bool=True) -> Optional[ClientConfig]:
        try:
            if use_cache and client_id in self._cache:
                cached = self._cache[client_id]
                if not cached.is_expired():
                    logger.debug(f"Cache hit for client '{client_id}'")
                    return cached.config
                else:
                    logger.debug(f"Cache expired for client '{client_id}'")
                    del self._cache[client_id]
            logger.debug(f"Fetching config for client '{client_id}' from database")
            config_data = await self.db.client_configs.find_one({'client_id': client_id})
            if not config_data:
                logger.warning(f"Client config not found for '{client_id}'")
                return None
            if '_id' in config_data:
                del config_data['_id']
            config = ClientConfig.from_dict(config_data)
            if use_cache:
                async with self._lock:
                    self._cache[client_id] = CachedConfig(config=config, cached_at=utcnow())
                logger.debug(f"Cached config for client '{client_id}'")
            return config
        except Exception as e:
            logger.error(f"Error getting client config for '{client_id}': {e}")
            return None

    async def get_all_clients(self, enabled_only: bool=False, skip: int=0, limit: int=100) -> List[ClientConfig]:
        try:
            query_filter = {}
            if enabled_only:
                query_filter['enabled'] = True
            cursor = self.db.client_configs.find(query_filter).skip(skip).limit(limit)
            configs = []
            async for config_data in cursor:
                if '_id' in config_data:
                    del config_data['_id']
                config = ClientConfig.from_dict(config_data)
                configs.append(config)
            logger.info(f'Retrieved {len(configs)} client configs')
            return configs
        except Exception as e:
            logger.error(f'Error getting all clients: {e}')
            return []

    async def create_client_config(self, client_id: str, name: str, enabled: bool=True, inherits_from: Optional[str]=None, customizations: Optional[Dict[str, Any]]=None, metadata: Optional[Dict[str, Any]]=None, created_by: str='system') -> Optional[ClientConfig]:
        try:
            existing = await self.db.client_configs.find_one({'client_id': client_id})
            if existing:
                logger.error(f"Client '{client_id}' already exists")
                return None
            if inherits_from:
                parent_config = await self.get_client_config(inherits_from, use_cache=False)
                if not parent_config:
                    logger.error(f"Parent client '{inherits_from}' not found")
                    return None
            config = ClientConfig(client_id=client_id, name=name, enabled=enabled, inherits_from=inherits_from, customizations=customizations or {}, metadata=metadata or {}, created_at=utcnow(), updated_at=utcnow(), created_by=created_by, updated_by=created_by)
            result = await self.db.client_configs.insert_one(config.to_dict())
            if result.inserted_id:
                logger.info(f"Created client config for '{client_id}'")
                async with self._lock:
                    self._cache[client_id] = CachedConfig(config=config, cached_at=utcnow())
                return config
            else:
                logger.error(f"Failed to insert client config for '{client_id}'")
                return None
        except Exception as e:
            logger.error(f"Error creating client config for '{client_id}': {e}")
            return None

    async def update_client_config(self, client_id: str, updates: Dict[str, Any], updated_by: str='system') -> bool:
        try:
            existing = await self.db.client_configs.find_one({'client_id': client_id})
            if not existing:
                logger.error(f"Client '{client_id}' not found for update")
                return False
            updates['updated_at'] = utcnow()
            updates['updated_by'] = updated_by
            if 'client_id' in updates:
                del updates['client_id']
            if 'inherits_from' in updates and updates['inherits_from']:
                parent_config = await self.get_client_config(updates['inherits_from'], use_cache=False)
                if not parent_config:
                    logger.error(f"Parent client '{updates['inherits_from']}' not found")
                    return False
            result = await self.db.client_configs.update_one({'client_id': client_id}, {'$set': updates})
            if result.modified_count > 0:
                logger.info(f"Updated client config for '{client_id}'")
                if client_id in self._cache:
                    async with self._lock:
                        del self._cache[client_id]
                    logger.debug(f"Invalidated cache for '{client_id}'")
                return True
            else:
                logger.warning(f"No changes made to client '{client_id}'")
                return True
        except Exception as e:
            logger.error(f"Error updating client config for '{client_id}': {e}")
            return False

    async def delete_client_config(self, client_id: str) -> bool:
        try:
            result = await self.db.client_configs.update_one({'client_id': client_id}, {'$set': {'enabled': False, 'updated_at': utcnow()}})
            if result.modified_count > 0:
                logger.info(f"Soft deleted client '{client_id}'")
                if client_id in self._cache:
                    async with self._lock:
                        del self._cache[client_id]
                return True
            else:
                logger.warning(f"Client '{client_id}' not found or already disabled")
                return False
        except Exception as e:
            logger.error(f"Error deleting client config for '{client_id}': {e}")
            return False

    async def hard_delete_client_config(self, client_id: str) -> bool:
        try:
            result = await self.db.client_configs.delete_one({'client_id': client_id})
            if result.deleted_count > 0:
                logger.warning(f"HARD DELETED client '{client_id}' from database")
                if client_id in self._cache:
                    async with self._lock:
                        del self._cache[client_id]
                return True
            else:
                logger.warning(f"Client '{client_id}' not found for deletion")
                return False
        except Exception as e:
            logger.error(f"Error hard deleting client config for '{client_id}': {e}")
            return False

    async def invalidate_cache(self, client_id: Optional[str]=None) -> None:
        async with self._lock:
            if client_id:
                if client_id in self._cache:
                    del self._cache[client_id]
                    logger.debug(f"Invalidated cache for '{client_id}'")
            else:
                self._cache.clear()
                logger.info('Invalidated all client config cache')

    async def get_client_count(self, enabled_only: bool=False) -> int:
        try:
            query_filter = {}
            if enabled_only:
                query_filter['enabled'] = True
            count = await self.db.client_configs.count_documents(query_filter)
            return count
        except Exception as e:
            logger.error(f'Error getting client count: {e}')
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        total_cached = len(self._cache)
        expired_count = sum((1 for cached in self._cache.values() if cached.is_expired()))
        active_count = total_cached - expired_count
        return {'total_cached': total_cached, 'active_cached': active_count, 'expired_cached': expired_count, 'ttl_seconds': CACHE_TTL_SECONDS}
_client_config_manager: Optional[ClientConfigManager] = None

def initialize_client_config_manager(db) -> ClientConfigManager:
    global _client_config_manager
    _client_config_manager = ClientConfigManager(db)
    logger.info('Global ClientConfigManager initialized')
    return _client_config_manager

def get_client_config_manager() -> Optional[ClientConfigManager]:
    if _client_config_manager is None:
        logger.warning('ClientConfigManager not initialized. Call initialize_client_config_manager first.')
    return _client_config_manager

async def get_config(client_id: str) -> Optional[ClientConfig]:
    manager = get_client_config_manager()
    if manager:
        return await manager.get_client_config(client_id)
    return None
__all__ = ['ClientConfig', 'CachedConfig', 'ClientConfigManager', 'initialize_client_config_manager', 'get_client_config_manager', 'get_config', 'CACHE_TTL_SECONDS']