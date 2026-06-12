from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from domains.conversation.model import ACTIVE_STATUSES, TERMINAL_STATUSES, Conversation, ConversationPhase, ConversationStatus
from util.time_utils import utcnow
logger = logging.getLogger(__name__)
COLLECTION = 'conversations'

class _AuditedCollection:

    def __init__(self, collection):
        self._col = collection

    def _stamp(self, update: dict) -> dict:
        if '$set' in update:
            return {**update, '$set': {**update['$set'], 'updated_at': utcnow()}}
        return update

    async def update_one(self, filter, update, **kwargs):
        return await self._col.update_one(filter, self._stamp(update), **kwargs)

    async def update_many(self, filter, update, **kwargs):
        return await self._col.update_many(filter, self._stamp(update), **kwargs)

    def __getattr__(self, name):
        return getattr(self._col, name)

def _to_model(doc: Dict[str, Any]) -> Conversation:
    if doc.get('_id'):
        doc['id'] = str(doc.pop('_id'))
    return Conversation.model_validate(doc)

class ConversationRepository:

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = _AuditedCollection(db[COLLECTION])

    async def insert(self, conversation: Conversation) -> Optional[str]:
        try:
            doc = conversation.model_dump(mode='python')
            result = await self.collection.insert_one(doc)
            logger.info('Conversation inserted | run_id=%s | _id=%s', conversation.run_id, result.inserted_id)
            return str(result.inserted_id)
        except Exception as e:
            logger.error('Failed to insert conversation run_id=%s: %s', conversation.run_id, e)
            return None

    async def insert_pending(self, *, run_id: str, user_id: str, client_id: str, session_id: str, input_text: str, route_decision: str='') -> Optional[str]:
        if not client_id:
            logger.error('insert_pending: client_id is required')
            return None
        try:
            doc = {'run_id': run_id, 'user_id': user_id, 'client_id': client_id, 'session_id': session_id, 'input': input_text, 'status': ConversationStatus.PENDING.value, 'route_decision': route_decision, 'created_at': utcnow(), 'agent_responses': {'planner': None, 'python': None, 'executor': None, 'business': None}, 'metadata': {}, 'timing': {}, 'llm_config': {}, 'agent_inputs': {}, 'agent_token_usage': {}, 'total_token_usage': {}, 'estimated_cost': {}}
            result = await self.collection.insert_one(doc)
            logger.info('Pending conversation inserted | run_id=%s | _id=%s', run_id, result.inserted_id)
            return str(result.inserted_id)
        except Exception as e:
            logger.error('Failed to insert pending conversation run_id=%s: %s', run_id, e)
            return None

    async def find_by_run_id(self, run_id: str) -> Optional[Conversation]:
        try:
            doc = await self.collection.find_one({'run_id': run_id})
            if not doc:
                return None
            return _to_model(doc)
        except Exception as e:
            logger.error('Failed to find conversation run_id=%s: %s', run_id, e)
            return None

    async def find_by_id(self, conversation_id: str, client_id: str) -> Optional[Conversation]:
        try:
            doc = await self.collection.find_one({'_id': ObjectId(conversation_id), 'client_id': client_id})
            if not doc:
                return None
            return _to_model(doc)
        except Exception as e:
            logger.error('Failed to find conversation _id=%s: %s', conversation_id, e)
            return None

    async def find_by_session(self, session_id: str, user_id: str, *, include_deleted: bool=False) -> List[Conversation]:
        try:
            query: Dict[str, Any] = {'session_id': session_id, 'user_id': user_id}
            if not include_deleted:
                query['is_deleted'] = {'$ne': True}
            cursor = self.collection.find(query).sort('created_at', 1).limit(500)
            docs = await cursor.to_list(length=500)
            return [_to_model(doc) for doc in docs]
        except Exception as e:
            logger.error('Failed to find conversations for session=%s: %s', session_id, e)
            return []

    async def get_session_metadata(self, session_id: str, *, user_id: Optional[str]=None, client_id: Optional[str]=None) -> Optional[Dict[str, Any]]:
        try:
            query: Dict[str, Any] = {'session_id': session_id}
            if user_id:
                query['user_id'] = user_id
            if client_id:
                query['client_id'] = client_id
            doc = await self.collection.find_one(query, {'user_id': 1, 'client_id': 1, 'session_id': 1, '_id': 0})
            if doc and isinstance(doc.get('user_id'), ObjectId):
                doc['user_id'] = str(doc['user_id'])
            return doc
        except Exception as e:
            logger.error('Failed to get session metadata session=%s: %s', session_id, e)
            return None

    async def list_sessions(self, *, client_id: str, user_id: Optional[str]=None, page: int=1, limit: int=20) -> Dict[str, Any]:
        try:
            match_filter: Dict[str, Any] = {'client_id': client_id, 'is_deleted': {'$ne': True}}
            if user_id:
                match_filter['user_id'] = user_id
            pipeline = [{'$match': match_filter}, {'$sort': {'created_at': 1}}, {'$group': {'_id': '$session_id', 'user_id': {'$first': '$user_id'}, 'first_message': {'$first': '$input'}, 'first_created_at': {'$first': '$created_at'}, 'last_activity': {'$max': {'$ifNull': ['$updated_at', '$created_at']}}, 'message_count': {'$sum': 1}, 'unread_count': {'$sum': {'$cond': [{'$eq': ['$is_read', False]}, 1, 0]}}}}, {'$sort': {'last_activity': -1}}, {'$skip': (page - 1) * limit}, {'$limit': limit}]
            sessions = await self.collection.aggregate(pipeline).to_list(length=limit)
            count_pipeline = [{'$match': match_filter}, {'$group': {'_id': '$session_id'}}, {'$count': 'total'}]
            count_result = await self.collection.aggregate(count_pipeline).to_list(length=1)
            total = count_result[0]['total'] if count_result else 0
            formatted = []
            for s in sessions:
                formatted.append({'session_id': s['_id'], 'user_id': str(s.get('user_id', '')), 'title': (s.get('first_message') or 'Untitled')[:100], 'created_at': s['first_created_at'], 'last_updated': s['last_activity'], 'message_count': s['message_count'], 'has_unread': s.get('unread_count', 0) > 0})
            return {'sessions': formatted, 'total': total, 'page': page, 'limit': limit}
        except Exception as e:
            logger.error('Failed to list sessions for client=%s: %s', client_id, e)
            return {'sessions': [], 'total': 0, 'page': page, 'limit': limit}

    async def update_by_id(self, conversation_id: str, client_id: str, update_data: Dict[str, Any]) -> bool:
        try:
            result = await self.collection.update_one({'_id': ObjectId(conversation_id), 'client_id': client_id}, {'$set': update_data})
            return result.modified_count > 0
        except Exception as e:
            logger.error('Failed to update conversation _id=%s: %s', conversation_id, e)
            return False

    async def update_status(self, conversation_id: str, client_id: str, status: ConversationStatus, **extra_fields: Any) -> bool:
        update_data: Dict[str, Any] = {'status': status.value, **extra_fields}
        return await self.update_by_id(conversation_id, client_id, update_data)

    async def save_feedback(self, run_id: str, rating: str, comment: str, user_id: str) -> bool:
        try:
            feedback = {'rating': rating, 'comment': comment, 'user_id': user_id, 'created_at': utcnow()}
            result = await self.collection.update_one({'run_id': run_id}, {'$set': {'feedback': feedback}})
            if result.matched_count > 0:
                logger.info('Feedback saved for run_id=%s', run_id)
                return True
            logger.warning('No conversation found for feedback run_id=%s', run_id)
            return False
        except Exception as e:
            logger.error('Failed to save feedback run_id=%s: %s', run_id, e)
            return False

    async def update_session_title(self, user_id: str, session_id: str, title: str) -> bool:
        try:
            first = await self.collection.find_one({'user_id': user_id, 'session_id': session_id}, sort=[('created_at', 1)])
            if not first:
                return False
            result = await self.collection.update_one({'_id': first['_id']}, {'$set': {'metadata.custom_title': title}})
            return result.modified_count > 0
        except Exception as e:
            logger.error('Failed to update session title session=%s: %s', session_id, e)
            return False

    async def soft_delete_session(self, session_id: str, client_id: str) -> int:
        try:
            filter_query: Dict[str, Any] = {'session_id': session_id, '$or': [{'client_id': client_id}, {'client_id': {'$exists': False}}, {'client_id': None}]}
            result = await self.collection.update_many(filter_query, {'$set': {'is_deleted': True, 'deleted_at': utcnow()}})
            logger.info('Soft-deleted %d conversations | session=%s | client=%s', result.modified_count, session_id, client_id)
            return result.modified_count
        except Exception as e:
            logger.error('Failed to soft-delete session=%s: %s', session_id, e)
            return 0

    async def hard_delete_session(self, user_id: str, session_id: str) -> int:
        try:
            result = await self.collection.delete_many({'user_id': user_id, 'session_id': session_id})
            logger.info('Hard-deleted %d conversations | session=%s', result.deleted_count, session_id)
            return result.deleted_count
        except Exception as e:
            logger.error('Failed to hard-delete session=%s: %s', session_id, e)
            return 0

    async def count_active_background(self, client_id: str) -> int:
        try:
            return await self.collection.count_documents({'client_id': client_id, 'is_background': True, 'status': {'$in': [s.value for s in ACTIVE_STATUSES]}})
        except Exception as e:
            logger.error('Failed to count active background conversations: %s', e)
            return 0

    async def mark_background_running(self, run_id: str, client_id: str, *, estimated_duration_seconds: int=300) -> bool:
        now = utcnow()
        try:
            result = await self.collection.update_one({'run_id': run_id, 'client_id': client_id}, {'$set': {'is_background': True, 'status': ConversationStatus.RUNNING.value, 'started_at': now, 'estimated_duration_seconds': estimated_duration_seconds, 'progress': {'current_phase': ConversationPhase.ROUTER.value, 'message': 'Running in background', 'iteration': 0, 'max_iterations': 0}}})
            return result.modified_count > 0
        except Exception as e:
            logger.error('Failed to mark background running run_id=%s: %s', run_id, e)
            return False

    async def update_progress(self, run_id: str, client_id: str, *, current_phase: ConversationPhase, message: str='', iteration: int=0, max_iterations: int=0) -> bool:
        try:
            result = await self.collection.update_one({'run_id': run_id, 'client_id': client_id, 'status': {'$in': [s.value for s in ACTIVE_STATUSES]}}, {'$set': {'progress.current_phase': current_phase.value, 'progress.message': message, 'progress.iteration': iteration, 'progress.max_iterations': max_iterations}})
            return result.modified_count > 0
        except Exception as e:
            logger.debug('Failed to update progress run_id=%s: %s', run_id, e)
            return False

    async def get_active_background(self, client_id: str, user_id: str) -> List[Dict[str, Any]]:
        try:
            cursor = self.collection.find({'client_id': client_id, 'user_id': user_id, 'is_background': True, 'status': {'$in': [s.value for s in ACTIVE_STATUSES]}, 'is_deleted': {'$ne': True}}, {'_id': 0, 'run_id': 1, 'session_id': 1, 'status': 1, 'progress': 1, 'input': 1, 'created_at': 1, 'started_at': 1, 'estimated_duration_seconds': 1, 'route_decision': 1}).sort('created_at', -1)
            docs = await cursor.to_list(length=50)
            for doc in docs:
                if doc.get('input'):
                    doc['input'] = doc['input'][:100]
            return docs
        except Exception as e:
            logger.error('Failed to get active background conversations: %s', e)
            return []

    async def list_background(self, client_id: str, user_id: Optional[str]=None, status: Optional[str]=None, limit: int=20, offset: int=0) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {'client_id': client_id, 'is_background': True, 'is_deleted': {'$ne': True}}
        if user_id:
            query['user_id'] = user_id
        if status and status in {s.value for s in ConversationStatus}:
            query['status'] = status
        try:
            cursor = self.collection.find(query, {'_id': 0}).sort('created_at', -1).skip(offset).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error('Failed to list background conversations: %s', e)
            return []

    async def find_background_by_run_id(self, run_id: str, client_id: str) -> Optional[Dict[str, Any]]:
        try:
            doc = await self.collection.find_one({'run_id': run_id, 'client_id': client_id, 'is_background': True, 'is_deleted': {'$ne': True}}, {'_id': 0})
            return doc
        except Exception as e:
            logger.error('Failed to find background conversation run_id=%s: %s', run_id, e)
            return None

    async def complete_background(self, run_id: str, client_id: str) -> bool:
        now = utcnow()
        try:
            result = await self.collection.update_one({'run_id': run_id, 'client_id': client_id, 'status': {'$in': [s.value for s in ACTIVE_STATUSES]}}, {'$set': {'status': ConversationStatus.COMPLETED.value, 'completed_at': now, 'is_read': False, 'progress.current_phase': ConversationPhase.COMPLETED.value, 'progress.message': 'Analysis complete'}})
            if result.modified_count > 0:
                logger.info('Background conversation completed | run_id=%s', run_id)
                return True
            return False
        except Exception as e:
            logger.error('Failed to complete background conversation run_id=%s: %s', run_id, e)
            return False

    async def fail_background(self, run_id: str, client_id: str, error: str) -> bool:
        now = utcnow()
        try:
            result = await self.collection.update_one({'run_id': run_id, 'client_id': client_id, 'status': {'$in': [s.value for s in ACTIVE_STATUSES]}}, {'$set': {'status': ConversationStatus.ERROR.value, 'completed_at': now, 'error': error[:2000], 'progress.current_phase': ConversationPhase.FAILED.value, 'progress.message': f'Failed: {error[:200]}'}})
            if result.modified_count > 0:
                logger.warning('Background conversation failed | run_id=%s | %s', run_id, error[:200])
                return True
            return False
        except Exception as e:
            logger.error('Failed to mark background conversation failed run_id=%s: %s', run_id, e)
            return False

    async def cancel_background(self, run_id: str, client_id: str) -> bool:
        now = utcnow()
        try:
            result = await self.collection.update_one({'run_id': run_id, 'client_id': client_id, 'status': {'$in': [s.value for s in ACTIVE_STATUSES]}}, {'$set': {'status': ConversationStatus.CANCELLED.value, 'completed_at': now, 'progress.current_phase': ConversationPhase.CANCELLED.value, 'progress.message': 'Cancelled by user'}})
            if result.modified_count > 0:
                logger.info('Background conversation cancelled | run_id=%s', run_id)
                return True
            return False
        except Exception as e:
            logger.error('Failed to cancel background conversation run_id=%s: %s', run_id, e)
            return False

    async def mark_notification_read(self, run_id: str, client_id: str) -> bool:
        try:
            result = await self.collection.update_one({'run_id': run_id, 'client_id': client_id}, {'$set': {'notification_read': True}})
            return result.modified_count > 0
        except Exception as e:
            logger.error('Failed to mark notification read run_id=%s: %s', run_id, e)
            return False

    async def mark_session_read(self, session_id: str, client_id: str) -> bool:
        try:
            result = await self.collection.update_many({'session_id': session_id, 'client_id': client_id, 'is_read': False}, {'$set': {'is_read': True}})
            logger.info('Marked session read | session=%s | updated=%d', session_id, result.modified_count)
            return True
        except Exception as e:
            logger.error('Failed to mark session read session=%s: %s', session_id, e)
            return False

    async def mark_stale_background_failed(self, stale_threshold_minutes: int=30) -> int:
        cutoff = utcnow() - timedelta(minutes=stale_threshold_minutes)
        try:
            result = await self.collection.update_many({'is_background': True, 'status': {'$in': [s.value for s in ACTIVE_STATUSES]}, 'created_at': {'$lt': cutoff}}, {'$set': {'status': ConversationStatus.ERROR.value, 'completed_at': utcnow(), 'error': 'Server restart: conversation was interrupted before completion.', 'progress.current_phase': ConversationPhase.FAILED.value, 'progress.message': 'Interrupted by server restart'}})
            count = result.modified_count
            if count > 0:
                logger.warning('Marked %d stale background conversations as failed (threshold: %dmin)', count, stale_threshold_minutes)
            return count
        except Exception as e:
            logger.error('Failed to mark stale background conversations: %s', e)
            return 0

    async def get_execution_stats(self, *, client_id: str, user_id: Optional[str]=None, days: int=30) -> Dict[str, Any]:
        try:
            from datetime import timedelta
            match: Dict[str, Any] = {'client_id': client_id, 'created_at': {'$gte': utcnow() - timedelta(days=days)}, 'timing.total_execution_time_seconds': {'$exists': True}}
            if user_id:
                match['user_id'] = user_id
            pipeline = [{'$match': match}, {'$group': {'_id': None, 'avg_time': {'$avg': '$timing.total_execution_time_seconds'}, 'min_time': {'$min': '$timing.total_execution_time_seconds'}, 'max_time': {'$max': '$timing.total_execution_time_seconds'}, 'total_executions': {'$sum': 1}}}]
            result = await self.collection.aggregate(pipeline).to_list(length=1)
            if result:
                s = result[0]
                return {'average_time_seconds': round(s.get('avg_time', 0), 2), 'min_time_seconds': round(s.get('min_time', 0), 2), 'max_time_seconds': round(s.get('max_time', 0), 2), 'total_executions': s.get('total_executions', 0), 'days_analyzed': days}
            return {'average_time_seconds': 0, 'min_time_seconds': 0, 'max_time_seconds': 0, 'total_executions': 0, 'days_analyzed': days}
        except Exception as e:
            logger.error('Failed to get execution stats: %s', e)
            return {}