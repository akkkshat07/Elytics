from __future__ import annotations
import asyncio
import csv
import json
import logging
import os
from datetime import datetime, timedelta
from util.time_utils import utcnow
from typing import Dict, List, Optional, Any
from enum import Enum
import io
import zipfile
from util.Mongodb import MongoDBManager
logger = logging.getLogger(__name__)

class ExportFormat(str, Enum):
    JSON = 'json'
    CSV = 'csv'
    ZIP = 'zip'

class ExportStatus(str, Enum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    EXPIRED = 'expired'

class DataExportService:

    def __init__(self):
        self.mongo_manager = MongoDBManager()
        self.export_dir = os.getenv('EXPORT_DIR', './data_exports')
        self.export_retention_days = 7
        os.makedirs(self.export_dir, exist_ok=True)
        logger.info(f'Data export service initialized | export_dir={self.export_dir}')

    async def export_user_data(self, user_id: str, client_id: str, format: ExportFormat=ExportFormat.JSON, include_sections: Optional[List[str]]=None) -> Dict[str, Any]:
        try:
            await self.mongo_manager.connect()
            job_id = f"export_{user_id}_{client_id}_{utcnow().strftime('%Y%m%d_%H%M%S')}"
            export_job = {'job_id': job_id, 'user_id': user_id, 'client_id': client_id, 'format': format.value, 'include_sections': include_sections or ['conversations', 'feedback', 'sessions', 'queries'], 'status': ExportStatus.PENDING.value, 'created_at': utcnow(), 'expires_at': utcnow() + timedelta(days=self.export_retention_days), 'file_path': None, 'file_size': None, 'error': None}
            await self.mongo_manager.db.data_export_jobs.insert_one(export_job)
            asyncio.create_task(self._process_export_job(job_id))
            logger.info(f'Data export job created | job_id={job_id} | user={user_id} | client={client_id} | format={format.value}')
            return {'job_id': job_id, 'status': ExportStatus.PENDING.value, 'message': 'Export job created. Processing in background.', 'estimated_time': '1-5 minutes', 'check_status_url': f'/api/admin/data-export/{job_id}'}
        except Exception as e:
            logger.error(f'Failed to create export job: {e}', exc_info=True)
            raise

    async def _process_export_job(self, job_id: str):
        try:
            await self.mongo_manager.connect()
            await self.mongo_manager.db.data_export_jobs.update_one({'job_id': job_id}, {'$set': {'status': ExportStatus.PROCESSING.value, 'started_at': utcnow()}})
            job = await self.mongo_manager.db.data_export_jobs.find_one({'job_id': job_id})
            if not job:
                logger.error(f'Export job not found: {job_id}')
                return
            user_id = job['user_id']
            client_id = job['client_id']
            format = ExportFormat(job['format'])
            sections = job['include_sections']
            data = {}
            if 'conversations' in sections:
                data['conversations'] = await self._get_user_conversations(user_id, client_id)
            if 'feedback' in sections:
                data['feedback'] = await self._get_user_feedback(user_id, client_id)
            if 'sessions' in sections:
                data['sessions'] = await self._get_user_sessions(user_id, client_id)
            if 'queries' in sections:
                data['queries'] = await self._get_user_queries(user_id, client_id)
            data['export_metadata'] = {'user_id': user_id, 'client_id': client_id, 'export_date': utcnow().isoformat(), 'format': format.value, 'sections_included': sections, 'record_counts': {section: len(data.get(section, [])) for section in sections}}
            file_path = os.path.join(self.export_dir, f'{job_id}.{format.value}')
            if format == ExportFormat.JSON:
                await self._export_as_json(data, file_path)
            elif format == ExportFormat.CSV:
                await self._export_as_csv(data, file_path)
            elif format == ExportFormat.ZIP:
                await self._export_as_zip(data, file_path)
            file_size = os.path.getsize(file_path)
            await self.mongo_manager.db.data_export_jobs.update_one({'job_id': job_id}, {'$set': {'status': ExportStatus.COMPLETED.value, 'completed_at': utcnow(), 'file_path': file_path, 'file_size': file_size}})
            logger.info(f"Export job completed | job_id={job_id} | file_size={file_size} bytes | records={data['export_metadata']['record_counts']}")
        except Exception as e:
            logger.error(f'Export job failed | job_id={job_id} | error={e}', exc_info=True)
            await self.mongo_manager.db.data_export_jobs.update_one({'job_id': job_id}, {'$set': {'status': ExportStatus.FAILED.value, 'error': str(e), 'failed_at': utcnow()}})

    async def _get_user_conversations(self, user_id: str, client_id: str) -> List[Dict]:
        try:
            conversations = await self.mongo_manager.db.conversation_history.find({'user_id': user_id, 'client_id': client_id}).to_list(length=None)
            for conv in conversations:
                conv['_id'] = str(conv['_id'])
            return conversations
        except Exception as e:
            logger.error(f'Failed to get conversations: {e}')
            return []

    async def _get_user_feedback(self, user_id: str, client_id: str) -> List[Dict]:
        try:
            conversations = await self.mongo_manager.db.conversation_history.find({'user_id': user_id, 'client_id': client_id, 'feedback': {'$exists': True}}).to_list(length=None)
            feedback_list = []
            for conv in conversations:
                if conv.get('feedback'):
                    feedback_list.append({'run_id': conv.get('run_id'), 'session_id': conv.get('session_id'), 'query': conv.get('query'), 'feedback': conv.get('feedback'), 'timestamp': conv.get('timestamp')})
            return feedback_list
        except Exception as e:
            logger.error(f'Failed to get feedback: {e}')
            return []

    async def _get_user_sessions(self, user_id: str, client_id: str) -> List[Dict]:
        try:
            sessions = await self.mongo_manager.db.conversation_history.aggregate([{'$match': {'user_id': user_id, 'client_id': client_id}}, {'$group': {'_id': '$session_id', 'session_id': {'$first': '$session_id'}, 'first_query': {'$first': '$query'}, 'message_count': {'$sum': 1}, 'first_timestamp': {'$min': '$timestamp'}, 'last_timestamp': {'$max': '$timestamp'}}}, {'$sort': {'first_timestamp': -1}}]).to_list(length=None)
            return sessions
        except Exception as e:
            logger.error(f'Failed to get sessions: {e}')
            return []

    async def _get_user_queries(self, user_id: str, client_id: str) -> List[Dict]:
        try:
            queries = await self.mongo_manager.db.conversation_history.find({'user_id': user_id, 'client_id': client_id}, {'run_id': 1, 'session_id': 1, 'query': 1, 'timestamp': 1, 'response_time': 1}).to_list(length=None)
            for q in queries:
                q['_id'] = str(q['_id'])
            return queries
        except Exception as e:
            logger.error(f'Failed to get queries: {e}')
            return []

    async def _export_as_json(self, data: Dict, file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    async def _export_as_csv(self, data: Dict, file_path: str):
        conversations = data.get('conversations', [])
        if not conversations:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('No conversation data available\n')
            return
        fieldnames = set()
        for conv in conversations:
            fieldnames.update(conv.keys())
        fieldnames = sorted(list(fieldnames))
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(conversations)

    async def _export_as_zip(self, data: Dict, file_path: str):
        with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for section_name, section_data in data.items():
                if section_name != 'export_metadata':
                    json_content = json.dumps(section_data, indent=2, default=str)
                    zipf.writestr(f'{section_name}.json', json_content)
            metadata_content = json.dumps(data['export_metadata'], indent=2, default=str)
            zipf.writestr('metadata.json', metadata_content)

    async def get_export_status(self, job_id: str) -> Optional[Dict]:
        try:
            await self.mongo_manager.connect()
            job = await self.mongo_manager.db.data_export_jobs.find_one({'job_id': job_id})
            if job:
                job['_id'] = str(job['_id'])
                return job
            return None
        except Exception as e:
            logger.error(f'Failed to get export status: {e}')
            return None

    async def cleanup_expired_exports(self) -> int:
        try:
            await self.mongo_manager.connect()
            expired_jobs = await self.mongo_manager.db.data_export_jobs.find({'expires_at': {'$lt': utcnow()}, 'status': ExportStatus.COMPLETED.value}).to_list(length=None)
            cleaned_count = 0
            for job in expired_jobs:
                if job.get('file_path') and os.path.exists(job['file_path']):
                    try:
                        os.remove(job['file_path'])
                        logger.info(f"Deleted expired export file: {job['file_path']}")
                    except Exception as e:
                        logger.error(f"Failed to delete file {job['file_path']}: {e}")
                await self.mongo_manager.db.data_export_jobs.update_one({'job_id': job['job_id']}, {'$set': {'status': ExportStatus.EXPIRED.value}})
                cleaned_count += 1
            if cleaned_count > 0:
                logger.info(f'Cleaned up {cleaned_count} expired exports')
            return cleaned_count
        except Exception as e:
            logger.error(f'Failed to cleanup expired exports: {e}')
            return 0
data_export_service = DataExportService()