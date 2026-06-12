from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from domains.dashboard.model import DashboardDocument, DashboardReport
from util.time_utils import utcnow
from util.token_usage_utils import aggregate_dashboard_report_usage_totals
logger = logging.getLogger(__name__)
COLLECTION = 'dashboard_reports'

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

class DashboardRepository:

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = _AuditedCollection(db[COLLECTION])

    async def ensure_indexes(self) -> None:
        try:
            await self.collection.create_index([('user_id', 1), ('client_id', 1)], unique=True)
            logger.info('Dashboard reports indexes ensured')
        except Exception as e:
            logger.error('Failed to create dashboard indexes: %s', e)

    async def get_by_user(self, user_id: str, client_id: str) -> Optional[Dict[str, Any]]:
        try:
            doc = await self.collection.find_one({'user_id': user_id, 'client_id': client_id}, {'_id': 0})
            return doc
        except Exception as e:
            logger.error('Failed to fetch dashboard user=%s client=%s: %s', user_id, client_id, e)
            return None

    async def list_report_usage_for_conversation(self, client_id: str, conversation_id: str) -> List[Dict[str, Any]]:
        cid = (conversation_id or '').strip()
        if not cid:
            return []
        id_match: List[Dict[str, Any]] = [{'reports.source_conversation_id': cid}]
        if len(cid) == 24 and all((c in '0123456789abcdefABCDEF' for c in cid)):
            try:
                id_match.append({'reports.source_conversation_id': ObjectId(cid)})
            except Exception:
                pass
        pipeline: List[Dict[str, Any]] = [{'$match': {'client_id': client_id}}, {'$unwind': '$reports'}, {'$match': {'$or': id_match}}, {'$sort': {'reports.created_at': -1}}, {'$project': {'_id': 0, 'dashboard_user_id': '$user_id', 'report_id': '$reports.report_id', 'title': '$reports.title', 'usage_events': {'$ifNull': ['$reports.usage_events', []]}, 'usage_totals': '$reports.usage_totals', 'report_created_at': '$reports.created_at', 'report_updated_at': '$reports.updated_at'}}]
        try:
            cursor = self.collection.aggregate(pipeline)
            return await cursor.to_list(length=None)
        except Exception as e:
            logger.error('list_report_usage_for_conversation failed client=%s conv=%s: %s', client_id, cid, e, exc_info=True)
            return []

    async def batch_report_usage_for_conversations(self, client_id: str, conversation_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        if not conversation_ids:
            return {}
        str_ids: List[str] = []
        obj_ids: List[ObjectId] = []
        for cid in conversation_ids:
            cid = (cid or '').strip()
            if not cid:
                continue
            str_ids.append(cid)
            if len(cid) == 24 and all((c in '0123456789abcdefABCDEF' for c in cid)):
                try:
                    obj_ids.append(ObjectId(cid))
                except Exception:
                    pass
        if not str_ids:
            return {}
        id_match: Dict[str, Any] = {'$in': str_ids + obj_ids} if obj_ids else {'$in': str_ids}
        pipeline: List[Dict[str, Any]] = [{'$match': {'client_id': client_id}}, {'$unwind': '$reports'}, {'$match': {'reports.source_conversation_id': id_match}}, {'$sort': {'reports.created_at': -1}}, {'$project': {'_id': 0, 'source_conversation_id': {'$toString': '$reports.source_conversation_id'}, 'dashboard_user_id': '$user_id', 'report_id': '$reports.report_id', 'title': '$reports.title', 'usage_events': {'$ifNull': ['$reports.usage_events', []]}, 'usage_totals': '$reports.usage_totals', 'report_created_at': '$reports.created_at', 'report_updated_at': '$reports.updated_at'}}]
        try:
            cursor = self.collection.aggregate(pipeline)
            rows = await cursor.to_list(length=None)
        except Exception as e:
            logger.error('batch_report_usage_for_conversations failed client=%s: %s', client_id, e, exc_info=True)
            return {}
        result: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in str_ids}
        for row in rows:
            src = (row.get('source_conversation_id') or '').strip()
            if src in result:
                result[src].append(row)
        return result

    async def get_or_create(self, user_id: str, client_id: str) -> Dict[str, Any]:
        doc = await self.get_by_user(user_id, client_id)
        if doc is not None:
            return doc
        now = utcnow()
        new_doc: Dict[str, Any] = {'user_id': user_id, 'client_id': client_id, 'reports': [], 'created_at': now, 'updated_at': now}
        try:
            await self.collection.insert_one({**new_doc})
            logger.info('Dashboard created user=%s client=%s', user_id, client_id)
            return new_doc
        except Exception as e:
            logger.error('Failed to create dashboard user=%s: %s', user_id, e)
            doc = await self.get_by_user(user_id, client_id)
            return doc or new_doc

    async def add_report(self, user_id: str, client_id: str, report: DashboardReport) -> bool:
        try:
            report_dict = report.model_dump(mode='python')
            result = await self.collection.update_one({'user_id': user_id, 'client_id': client_id}, {'$push': {'reports': report_dict}}, upsert=True)
            return result.modified_count > 0 or result.upserted_id is not None
        except Exception as e:
            logger.error('Failed to add report user=%s: %s', user_id, e)
            return False

    async def delete_report(self, user_id: str, client_id: str, report_id: str) -> bool:
        try:
            result = await self.collection.update_one({'user_id': user_id, 'client_id': client_id}, {'$pull': {'reports': {'report_id': report_id}}})
            return result.modified_count > 0
        except Exception as e:
            logger.error('Failed to delete report %s user=%s: %s', report_id, user_id, e)
            return False

    async def update_report_fields(self, user_id: str, client_id: str, report_id: str, updates: Dict[str, Any]) -> bool:
        try:
            set_payload = {f'reports.$.{k}': v for k, v in updates.items()}
            result = await self.collection.update_one({'user_id': user_id, 'client_id': client_id, 'reports.report_id': report_id}, {'$set': set_payload})
            return result.modified_count > 0
        except Exception as e:
            logger.error('Failed to update report %s user=%s: %s', report_id, user_id, e)
            return False

    async def _recompute_report_usage_totals(self, user_id: str, client_id: str, report_id: str) -> bool:
        doc = await self.collection.find_one({'user_id': user_id, 'client_id': client_id})
        if not doc:
            return False
        for r in doc.get('reports', []):
            if r.get('report_id') == report_id:
                events = r.get('usage_events') or []
                totals = aggregate_dashboard_report_usage_totals(events)
                return await self.update_report_fields(user_id, client_id, report_id, {'usage_totals': totals})
        return False

    async def append_report_usage_event(self, user_id: str, client_id: str, report_id: str, event: Dict[str, Any]) -> bool:
        try:
            result = await self.collection.update_one({'user_id': user_id, 'client_id': client_id, 'reports.report_id': report_id}, {'$push': {'reports.$.usage_events': event}})
            if result.modified_count > 0:
                try:
                    synced = await self._recompute_report_usage_totals(user_id, client_id, report_id)
                    if not synced:
                        logger.warning('append_report_usage_event: could not sync usage_totals report=%s user=%s', report_id, user_id)
                except Exception as sync_err:
                    logger.warning('append_report_usage_event: usage_totals recompute failed report=%s: %s', report_id, sync_err)
            return result.modified_count > 0
        except Exception as e:
            logger.error('Failed to append usage event report=%s user=%s client=%s: %s', report_id, user_id, client_id, e)
            return False

    async def reorder_reports(self, user_id: str, client_id: str, ordered_ids: List[str]) -> bool:
        try:
            doc = await self.get_by_user(user_id, client_id)
            if not doc:
                return False
            existing: Dict[str, Any] = {r['report_id']: r for r in doc.get('reports', [])}
            reordered = []
            for idx, rid in enumerate(ordered_ids):
                if rid in existing:
                    entry = {**existing[rid], 'order': idx}
                    reordered.append(entry)
            result = await self.collection.update_one({'user_id': user_id, 'client_id': client_id}, {'$set': {'reports': reordered}})
            return result.modified_count > 0
        except Exception as e:
            logger.error('Failed to reorder reports user=%s: %s', user_id, e)
            return False