import logging
from typing import Dict, Any, Optional
from datetime import datetime
from util.time_utils import utcnow
from motor.motor_asyncio import AsyncIOMotorDatabase
from db_config.mongo_server import get_db
from util.audit_logger import audit_logger, AuditEventType, AuditSeverity
logger = logging.getLogger(__name__)
VALID_STATUSES = ['active', 'suspended', 'deleted']
STATUS_TRANSITIONS = {'active': ['suspended', 'deleted'], 'suspended': ['active', 'deleted'], 'deleted': []}

def validate_status_transition(old_status: str, new_status: str) -> bool:
    if old_status not in VALID_STATUSES:
        logger.warning(f'Invalid old status: {old_status}')
        return False
    if new_status not in VALID_STATUSES:
        logger.warning(f'Invalid new status: {new_status}')
        return False
    allowed_transitions = STATUS_TRANSITIONS.get(old_status, [])
    return new_status in allowed_transitions

async def get_tenant_status(client_id: str, db: Optional[AsyncIOMotorDatabase]=None) -> Optional[str]:
    if db is None:
        db = await get_db()
    try:
        client_config = await db.client_configs.find_one({'client_id': client_id})
        if not client_config:
            return None
        return client_config.get('status', 'active')
    except Exception as e:
        logger.error(f'Error getting tenant status for {client_id}: {e}')
        return None

async def update_tenant_status(client_id: str, new_status: str, reason: Optional[str]=None, updated_by: Optional[str]=None, db: Optional[AsyncIOMotorDatabase]=None) -> Dict[str, Any]:
    if db is None:
        db = await get_db()
    if new_status not in VALID_STATUSES:
        raise ValueError(f'Invalid status: {new_status}. Must be one of {VALID_STATUSES}')
    current_status = await get_tenant_status(client_id, db)
    if current_status is None:
        raise ValueError(f'Client {client_id} not found')
    if not validate_status_transition(current_status, new_status):
        raise ValueError(f"Invalid status transition from '{current_status}' to '{new_status}'. Allowed transitions: {STATUS_TRANSITIONS.get(current_status, [])}")
    update_doc = {'status': new_status, 'updated_at': utcnow()}
    if new_status == 'suspended':
        update_doc['suspended_at'] = utcnow()
    elif new_status == 'deleted':
        update_doc['deleted_at'] = utcnow()
    elif new_status == 'active':
        update_doc['suspended_at'] = None
        update_doc['deleted_at'] = None
    result = await db.client_configs.update_one({'client_id': client_id}, {'$set': update_doc})
    if result.modified_count == 0:
        logger.warning(f'No documents updated for client {client_id}')
    await audit_logger.log_event(event_type=AuditEventType.METADATA_TABLE_INTRODUCTION_UPDATED, severity=AuditSeverity.INFO, user_id=updated_by or 'system', client_id=client_id, details={'action': 'tenant_status_update', 'old_status': current_status, 'new_status': new_status, 'reason': reason})
    logger.info(f'Tenant status updated: {client_id} from {current_status} to {new_status}')
    return {'success': True, 'client_id': client_id, 'old_status': current_status, 'new_status': new_status, 'updated_at': update_doc['updated_at']}

async def suspend_tenant(client_id: str, reason: Optional[str]=None, updated_by: Optional[str]=None, db: Optional[AsyncIOMotorDatabase]=None) -> Dict[str, Any]:
    return await update_tenant_status(client_id, 'suspended', reason, updated_by, db)

async def activate_tenant(client_id: str, reason: Optional[str]=None, updated_by: Optional[str]=None, db: Optional[AsyncIOMotorDatabase]=None) -> Dict[str, Any]:
    return await update_tenant_status(client_id, 'active', reason, updated_by, db)

async def soft_delete_tenant(client_id: str, reason: Optional[str]=None, updated_by: Optional[str]=None, db: Optional[AsyncIOMotorDatabase]=None) -> Dict[str, Any]:
    return await update_tenant_status(client_id, 'deleted', reason, updated_by, db)