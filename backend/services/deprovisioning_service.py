import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from util.time_utils import utcnow
from motor.motor_asyncio import AsyncIOMotorDatabase
from db_config.mongo_server import get_db
from util.audit_logger import audit_logger, AuditEventType, AuditSeverity
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent.parent
XML_PROMPTS_DIR = BASE_DIR / 'xml_prompts' / 'clients'
ASSETS_DIR = BASE_DIR / 'assets' / 'data' / 'clients'

async def cleanup_mongodb_data(client_id: str, db: Optional[AsyncIOMotorDatabase]=None) -> Dict[str, Any]:
    if db is None:
        db = await get_db()
    cleanup_results = {'users_deleted': 0, 'conversations_deleted': 0, 'subscriptions_deleted': 0, 'client_configs_deleted': 0, 'other_collections': {}}
    try:
        users_result = await db.users.delete_many({'client_id': client_id})
        cleanup_results['users_deleted'] = users_result.deleted_count
        logger.info(f'Deleted {users_result.deleted_count} users for client {client_id}')
        conversations_result = await db.conversations.delete_many({'client_id': client_id})
        cleanup_results['conversations_deleted'] = conversations_result.deleted_count
        logger.info(f'Deleted {conversations_result.deleted_count} conversations for client {client_id}')
        subscriptions_result = await db.subscriptions.delete_many({'client_id': client_id})
        cleanup_results['subscriptions_deleted'] = subscriptions_result.deleted_count
        logger.info(f'Deleted {subscriptions_result.deleted_count} subscriptions for client {client_id}')
        client_configs_result = await db.client_configs.delete_many({'client_id': client_id})
        cleanup_results['client_configs_deleted'] = client_configs_result.deleted_count
        logger.info(f'Deleted {client_configs_result.deleted_count} client_configs for client {client_id}')
        other_collections_to_check = ['audit_logs', 'feedback', 'sessions', 'cache_entries']
        for collection_name in other_collections_to_check:
            try:
                collection = getattr(db, collection_name, None)
                if collection:
                    result = await collection.delete_many({'client_id': client_id})
                    if result.deleted_count > 0:
                        cleanup_results['other_collections'][collection_name] = result.deleted_count
                        logger.info(f'Deleted {result.deleted_count} documents from {collection_name} for client {client_id}')
            except Exception as e:
                logger.warning(f'Error cleaning up collection {collection_name} for client {client_id}: {e}')
        logger.info(f'MongoDB cleanup completed for client {client_id}')
        return cleanup_results
    except Exception as e:
        logger.error(f'Error during MongoDB cleanup for client {client_id}: {e}', exc_info=True)
        raise

async def cleanup_file_system_data(client_id: str) -> Dict[str, Any]:
    cleanup_results = {'xml_prompts_deleted': False, 'assets_deleted': False, 'errors': []}
    try:
        xml_prompts_path = XML_PROMPTS_DIR / client_id
        if xml_prompts_path.exists() and xml_prompts_path.is_dir():
            shutil.rmtree(xml_prompts_path)
            cleanup_results['xml_prompts_deleted'] = True
            logger.info(f'Deleted XML prompts directory for client {client_id}: {xml_prompts_path}')
        else:
            logger.warning(f'XML prompts directory not found for client {client_id}: {xml_prompts_path}')
        assets_path = ASSETS_DIR / client_id
        if assets_path.exists() and assets_path.is_dir():
            shutil.rmtree(assets_path)
            cleanup_results['assets_deleted'] = True
            logger.info(f'Deleted assets directory for client {client_id}: {assets_path}')
        else:
            logger.warning(f'Assets directory not found for client {client_id}: {assets_path}')
        logger.info(f'File system cleanup completed for client {client_id}')
        return cleanup_results
    except Exception as e:
        logger.error(f'Error during file system cleanup for client {client_id}: {e}', exc_info=True)
        cleanup_results['errors'].append(str(e))
        raise

async def initiate_deprovisioning(client_id: str, hard_delete: bool=True, initiated_by: Optional[str]=None, db: Optional[AsyncIOMotorDatabase]=None) -> Dict[str, Any]:
    if db is None:
        db = await get_db()
    logger.warning(f"⚠️  DEPROVISIONING INITIATED for client {client_id} by {initiated_by or 'system'}")
    deprovisioning_results = {'client_id': client_id, 'initiated_at': utcnow(), 'initiated_by': initiated_by or 'system', 'hard_delete': hard_delete, 'mongodb_cleanup': {}, 'filesystem_cleanup': {}, 'success': False, 'errors': []}
    try:
        try:
            mongodb_results = await cleanup_mongodb_data(client_id, db)
            deprovisioning_results['mongodb_cleanup'] = mongodb_results
        except Exception as e:
            error_msg = f'MongoDB cleanup failed: {str(e)}'
            logger.error(error_msg)
            deprovisioning_results['errors'].append(error_msg)
            raise
        try:
            filesystem_results = await cleanup_file_system_data(client_id)
            deprovisioning_results['filesystem_cleanup'] = filesystem_results
        except Exception as e:
            error_msg = f'File system cleanup failed: {str(e)}'
            logger.error(error_msg)
            deprovisioning_results['errors'].append(error_msg)
        await audit_logger.log_event(event_type=AuditEventType.METADATA_TABLE_INTRODUCTION_UPDATED, severity=AuditSeverity.CRITICAL, user_id=initiated_by or 'system', client_id=client_id, details={'action': 'tenant_deprovisioning', 'hard_delete': hard_delete, 'results': deprovisioning_results})
        deprovisioning_results['success'] = True
        deprovisioning_results['completed_at'] = utcnow()
        logger.warning(f'✅ DEPROVISIONING COMPLETED for client {client_id}')
        return deprovisioning_results
    except Exception as e:
        deprovisioning_results['success'] = False
        deprovisioning_results['error'] = str(e)
        logger.error(f'❌ DEPROVISIONING FAILED for client {client_id}: {e}', exc_info=True)
        raise

async def get_deprovisioning_status(client_id: str, db: Optional[AsyncIOMotorDatabase]=None) -> Dict[str, Any]:
    if db is None:
        db = await get_db()
    client_config = await db.client_configs.find_one({'client_id': client_id})
    xml_prompts_path = XML_PROMPTS_DIR / client_id
    xml_prompts_exists = xml_prompts_path.exists() and xml_prompts_path.is_dir()
    assets_path = ASSETS_DIR / client_id
    assets_exists = assets_path.exists() and assets_path.is_dir()
    return {'client_id': client_id, 'deprovisioned': client_config is None, 'mongodb_data_exists': client_config is not None, 'xml_prompts_exists': xml_prompts_exists, 'assets_exists': assets_exists, 'fully_deprovisioned': client_config is None and (not xml_prompts_exists) and (not assets_exists)}