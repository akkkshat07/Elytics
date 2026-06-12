from __future__ import annotations
import logging
from datetime import datetime, timedelta
from util.time_utils import utcnow
from typing import Optional, Dict, Any, List
from enum import Enum
import asyncio
from util.Mongodb import MongoDBManager
logger = logging.getLogger(__name__)

class AuditEventType(str, Enum):
    AUTH_LOGIN_SUCCESS = 'auth.login.success'
    AUTH_LOGIN_FAILURE = 'auth.login.failure'
    AUTH_LOGOUT = 'auth.logout'
    AUTH_TOKEN_REFRESH = 'auth.token.refresh'
    AUTH_TOKEN_INVALID = 'auth.token.invalid'
    REGISTRATION_SUCCESS = 'registration.success'
    REGISTRATION_FAILURE = 'registration.failure'
    REGISTRATION_RATE_LIMITED = 'registration.rate.limited'
    REGISTRATION_DUPLICATE_ATTEMPT = 'registration.duplicate.attempt'
    EMAIL_VERIFICATION_SENT = 'email.verification.sent'
    EMAIL_VERIFICATION_SUCCESS = 'email.verification.success'
    EMAIL_VERIFICATION_FAILURE = 'email.verification.failure'
    EMAIL_VERIFICATION_EXPIRED = 'email.verification.expired'
    AUTHZ_ACCESS_DENIED = 'authz.access.denied'
    AUTHZ_PRIVILEGE_ESCALATION = 'authz.privilege.escalation'
    AUTHZ_CROSS_TENANT_ATTEMPT = 'authz.cross.tenant.attempt'
    ADMIN_USER_CREATED = 'admin.user.created'
    ADMIN_USER_UPDATED = 'admin.user.updated'
    ADMIN_USER_DELETED = 'admin.user.deleted'
    ADMIN_CLIENT_CREATED = 'admin.client.created'
    ADMIN_CLIENT_UPDATED = 'admin.client.updated'
    ADMIN_CLIENT_DELETED = 'admin.client.deleted'
    ADMIN_ROLE_CHANGED = 'admin.role.changed'
    ADMIN_PASSWORD_RESET = 'admin.password.reset'
    ADMIN_CLIENT_DEPROVISIONED = 'admin.client.deprovisioned'
    CONFIG_LLM_UPDATED = 'config.llm.updated'
    CONFIG_RATE_LIMIT_UPDATED = 'config.rate_limit.updated'
    DATA_QUERY_EXECUTED = 'data.query.executed'
    DATA_EXPORT_REQUESTED = 'data.export.requested'
    DATA_EXPORT_COMPLETED = 'data.export.completed'
    FILE_UPLOAD = 'file.upload'
    EXPLORATION_RUN = 'exploration.run'
    SECURITY_RATE_LIMIT_EXCEEDED = 'security.rate.limit.exceeded'
    SECURITY_INJECTION_ATTEMPT = 'security.injection.attempt'
    SECURITY_INVALID_INPUT = 'security.invalid.input'
    CONFIG_PROMPT_UPDATED = 'config.prompt.updated'
    CONFIG_SETTINGS_CHANGED = 'config.settings.changed'
    DB_CREDENTIALS_SAVED = 'db.credentials.saved'
    DB_CREDENTIALS_LOADED = 'db.credentials.loaded'
    DB_CREDENTIALS_DELETED = 'db.credentials.deleted'
    DB_CREDENTIALS_UPDATED = 'db.credentials.updated'
    METADATA_TABLE_INTRODUCTION_UPDATED = 'metadata.table_introduction.updated'
    METADATA_COLUMN_DESCRIPTION_UPDATED = 'metadata.column_description.updated'
    SESSION_CREATED = 'session.created'
    SESSION_DELETED = 'session.deleted'
    SESSION_TITLE_UPDATED = 'session.title.updated'
    FEEDBACK_SUBMITTED = 'feedback.submitted'

class AuditSeverity(str, Enum):
    INFO = 'info'
    WARNING = 'warning'
    CRITICAL = 'critical'
    EMERGENCY = 'emergency'

class AuditLogger:

    def __init__(self):
        self.mongo_manager = MongoDBManager()
        self.retention_days = 365
        self.queue = asyncio.Queue()
        self.enabled = True
        logger.info('Audit logger initialized')

    async def log_event(self, event_type: AuditEventType, severity: AuditSeverity, user_id: Optional[str]=None, client_id: Optional[str]=None, details: Optional[Dict[str, Any]]=None, ip_address: Optional[str]=None, user_agent: Optional[str]=None):
        if not self.enabled:
            return
        try:
            event = {'event_type': event_type.value, 'severity': severity.value, 'timestamp': utcnow(), 'user_id': user_id, 'client_id': client_id, 'ip_address': ip_address, 'user_agent': user_agent, 'details': details or {}, 'retention_until': utcnow() + timedelta(days=self.retention_days)}
            await self._write_to_db(event)
            if severity in [AuditSeverity.CRITICAL, AuditSeverity.EMERGENCY]:
                logger.warning(f'[AUDIT {severity.value.upper()}] {event_type.value} | user={user_id} | client={client_id} | ip={ip_address} | details={details}')
        except Exception as e:
            logger.error(f'Failed to log audit event: {e}', exc_info=True)

    async def _write_to_db(self, event: Dict[str, Any]):
        try:
            await self.mongo_manager.connect()
            if self.mongo_manager.db is None:
                logger.error('MongoDB connection not available for audit logging')
                return
            collection = self.mongo_manager.db.audit_logs
            await collection.insert_one(event)
        except Exception as e:
            logger.error(f'Failed to write audit log to DB: {e}')

    async def query_events(self, start_time: Optional[datetime]=None, end_time: Optional[datetime]=None, event_types: Optional[List[AuditEventType]]=None, severity: Optional[AuditSeverity]=None, user_id: Optional[str]=None, client_id: Optional[str]=None, limit: int=100) -> List[Dict[str, Any]]:
        try:
            await self.mongo_manager.connect()
            if self.mongo_manager.db is None:
                logger.error('MongoDB connection not available for audit query')
                return []
            collection = self.mongo_manager.db.audit_logs
            query = {}
            if start_time or end_time:
                query['timestamp'] = {}
                if start_time:
                    query['timestamp']['$gte'] = start_time
                if end_time:
                    query['timestamp']['$lte'] = end_time
            if event_types:
                query['event_type'] = {'$in': [et.value for et in event_types]}
            if severity:
                query['severity'] = severity.value
            if user_id:
                query['user_id'] = user_id
            if client_id:
                query['client_id'] = client_id
            cursor = collection.find(query).sort('timestamp', -1).limit(limit)
            events = await cursor.to_list(length=limit)
            for event in events:
                event['_id'] = str(event['_id'])
            return events
        except Exception as e:
            logger.error(f'Failed to query audit events: {e}', exc_info=True)
            return []

    async def cleanup_old_logs(self):
        try:
            await self.mongo_manager.connect()
            collection = self.mongo_manager.db.audit_logs
            result = await collection.delete_many({'retention_until': {'$lt': utcnow()}})
            if result.deleted_count > 0:
                logger.info(f'Cleaned up {result.deleted_count} expired audit logs')
            return result.deleted_count
        except Exception as e:
            logger.error(f'Failed to cleanup old audit logs: {e}', exc_info=True)
            return 0

    async def get_statistics(self, client_id: Optional[str]=None, days: int=30) -> Dict[str, Any]:
        try:
            await self.mongo_manager.connect()
            collection = self.mongo_manager.db.audit_logs
            match_criteria = {'timestamp': {'$gte': utcnow() - timedelta(days=days)}}
            if client_id:
                match_criteria['client_id'] = client_id
            pipeline_by_type = [{'$match': match_criteria}, {'$group': {'_id': '$event_type', 'count': {'$sum': 1}}}, {'$sort': {'count': -1}}]
            pipeline_by_severity = [{'$match': match_criteria}, {'$group': {'_id': '$severity', 'count': {'$sum': 1}}}]
            by_type = await collection.aggregate(pipeline_by_type).to_list(length=None)
            by_severity = await collection.aggregate(pipeline_by_severity).to_list(length=None)
            return {'period_days': days, 'client_id': client_id, 'events_by_type': {item['_id']: item['count'] for item in by_type}, 'events_by_severity': {item['_id']: item['count'] for item in by_severity}, 'total_events': sum((item['count'] for item in by_type))}
        except Exception as e:
            logger.error(f'Failed to get audit statistics: {e}', exc_info=True)
            return {}
audit_logger = AuditLogger()

async def audit_login_success(user_id: str, client_id: str, ip_address: str, user_agent: str):
    await audit_logger.log_event(AuditEventType.AUTH_LOGIN_SUCCESS, AuditSeverity.INFO, user_id=user_id, client_id=client_id, ip_address=ip_address, user_agent=user_agent)

async def audit_login_failure(email: str, ip_address: str, reason: str):
    await audit_logger.log_event(AuditEventType.AUTH_LOGIN_FAILURE, AuditSeverity.WARNING, details={'email': email, 'reason': reason}, ip_address=ip_address)

async def audit_access_denied(user_id: str, client_id: str, resource: str, reason: str):
    await audit_logger.log_event(AuditEventType.AUTHZ_ACCESS_DENIED, AuditSeverity.WARNING, user_id=user_id, client_id=client_id, details={'resource': resource, 'reason': reason})

async def audit_admin_action(admin_user_id: str, client_id: str, action: AuditEventType, target: str, details: Dict[str, Any]):
    await audit_logger.log_event(action, AuditSeverity.INFO, user_id=admin_user_id, client_id=client_id, details={'target': target, **details})

async def audit_rate_limit_exceeded(user_id: str, client_id: str, limit_type: str, endpoint: str, ip_address: str):
    await audit_logger.log_event(AuditEventType.SECURITY_RATE_LIMIT_EXCEEDED, AuditSeverity.WARNING, user_id=user_id, client_id=client_id, details={'limit_type': limit_type, 'endpoint': endpoint}, ip_address=ip_address)

async def audit_query_execution(user_id: str, client_id: str, query: str, execution_time: float, ip_address: str):
    await audit_logger.log_event(AuditEventType.DATA_QUERY_EXECUTED, AuditSeverity.INFO, user_id=user_id, client_id=client_id, details={'query': query[:500], 'execution_time_seconds': execution_time}, ip_address=ip_address)