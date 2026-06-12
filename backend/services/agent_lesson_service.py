import difflib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from util.time_utils import utcnow
logger = logging.getLogger(__name__)
CATEGORY_LABELS = {'A': 'Data quirks', 'B': 'Data quirks', 'C': 'Relationships', 'D': 'Filters', 'E': 'Performance', 'F': 'Business', 'G': 'Visualization', 'H': 'Environment'}
MAX_LESSONS_PER_CLIENT = 150
DEDUP_SIMILARITY_THRESHOLD = 0.85
STALE_DAYS = 90

class AgentLessonService:
    COLLECTION = 'agent_lessons'

    def __init__(self, db: Any):
        self.db = db
        self.col = db[self.COLLECTION]

    async def save_lesson(self, client_id: str, lesson_dict_or_type=None, category: str='A', lesson_text: str='', tables_involved: Optional[List[str]]=None, source: str='error_recovery', source_error_type: str='', source_query_hint: str='', confidence: float=0.9) -> Optional[str]:
        if isinstance(lesson_dict_or_type, dict):
            d = lesson_dict_or_type
            return await self._save_lesson_impl(client_id=client_id, lesson_type=d.get('lesson_type', 'general'), category=d.get('category', 'A'), lesson_text=d.get('lesson', ''), tables_involved=d.get('tables_involved'), source=d.get('source', 'error_recovery'), source_error_type=d.get('source_error_type', ''), source_query_hint=d.get('source_query_hint', ''), confidence=d.get('confidence', 0.9))
        lesson_type = lesson_dict_or_type or 'general'
        return await self._save_lesson_impl(client_id=client_id, lesson_type=lesson_type, category=category, lesson_text=lesson_text, tables_involved=tables_involved, source=source, source_error_type=source_error_type, source_query_hint=source_query_hint, confidence=confidence)

    async def _save_lesson_impl(self, client_id: str, lesson_type: str='general', category: str='A', lesson_text: str='', tables_involved: Optional[List[str]]=None, source: str='error_recovery', source_error_type: str='', source_query_hint: str='', confidence: float=0.9) -> Optional[str]:
        try:
            is_novel = await self._is_novel(client_id, category, lesson_text)
            if not is_novel:
                return None
            count = await self.col.count_documents({'client_id': client_id})
            if count >= MAX_LESSONS_PER_CLIENT:
                await self._evict_one(client_id)
            doc = {'client_id': client_id, 'lesson_type': lesson_type, 'category': category, 'lesson': lesson_text, 'tables_involved': tables_involved or [], 'source': source, 'source_error_type': source_error_type, 'source_query_hint': source_query_hint[:100], 'hit_count': 0, 'confidence': confidence, 'created_at': utcnow(), 'last_hit_at': None}
            result = await self.col.insert_one(doc)
            logger.info("Lesson saved [%s/%s] for client '%s': %s", category, source, client_id, lesson_text[:80])
            return str(result.inserted_id)
        except Exception as e:
            logger.warning('Failed to save lesson: %s', e)
            return None

    async def get_lessons(self, client_id: str, tables: Optional[List[str]]=None, categories: Optional[List[str]]=None, max_count: int=50) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {'client_id': client_id}
        if categories:
            query['category'] = {'$in': categories}
        cursor = self.col.find(query, {'_id': 0}).sort([('confidence', -1), ('hit_count', -1)]).limit(max_count)
        lessons = await cursor.to_list(length=max_count)
        if tables and lessons:
            tables_lower = {t.lower() for t in tables}
            relevant = []
            general = []
            for lesson in lessons:
                involved = {t.lower() for t in lesson.get('tables_involved', [])}
                if not involved or involved & tables_lower:
                    relevant.append(lesson)
                else:
                    general.append(lesson)
            lessons = relevant + general[:max(0, max_count - len(relevant))]
        return lessons

    async def format_lessons_for_prompt(self, client_id: str, tables: Optional[List[str]]=None, max_tokens: int=1500) -> str:
        lessons = await self.get_lessons(client_id, tables=tables)
        if not lessons:
            return ''
        groups: Dict[str, List[str]] = {}
        for lesson in lessons:
            cat = lesson.get('category', 'A')
            label = CATEGORY_LABELS.get(cat, 'Other')
            groups.setdefault(label, []).append(lesson['lesson'])
        lines: List[str] = []
        approx_tokens = 0
        for label, items in groups.items():
            section_header = f'{label}:'
            header_cost = len(section_header) // 4
            if approx_tokens + header_cost > max_tokens:
                break
            lines.append(section_header)
            approx_tokens += header_cost
            for item in items:
                line = f'- {item}'
                cost = len(line) // 4
                if approx_tokens + cost > max_tokens:
                    break
                lines.append(line)
                approx_tokens += cost
        try:
            lesson_ids = [l.get('lesson') for l in lessons[:len(lines)]]
            if lesson_ids:
                await self.col.update_many({'client_id': client_id, 'lesson': {'$in': lesson_ids}}, {'$inc': {'hit_count': 1}, '$set': {'last_hit_at': utcnow()}})
        except Exception:
            pass
        return '\n'.join(lines)

    async def cleanup_stale_lessons(self, client_id: str) -> int:
        cutoff = utcnow() - timedelta(days=STALE_DAYS)
        result = await self.col.delete_many({'client_id': client_id, 'hit_count': 0, 'created_at': {'$lt': cutoff}})
        if result.deleted_count:
            logger.info("Cleaned up %d stale lessons for client '%s'", result.deleted_count, client_id)
        return result.deleted_count

    async def delete_lessons_for_client(self, client_id: str) -> int:
        result = await self.col.delete_many({'client_id': client_id})
        return result.deleted_count

    async def delete_lessons_by_source(self, client_id: str, source: str) -> int:
        result = await self.col.delete_many({'client_id': client_id, 'source': source})
        return result.deleted_count

    async def _is_novel(self, client_id: str, category: str, new_text: str) -> bool:
        existing = await self.col.find({'client_id': client_id, 'category': category}, {'lesson': 1}).to_list(length=200)
        for doc in existing:
            existing_text = doc.get('lesson', '')
            ratio = difflib.SequenceMatcher(None, new_text.lower(), existing_text.lower()).ratio()
            if ratio >= DEDUP_SIMILARITY_THRESHOLD:
                await self.col.update_one({'_id': doc['_id']}, {'$inc': {'hit_count': 1}, '$set': {'last_hit_at': utcnow()}})
                logger.debug('Dedup: lesson matched existing (ratio=%.2f): %s', ratio, existing_text[:60])
                return False
        return True

    async def _evict_one(self, client_id: str) -> None:
        doc = await self.col.find_one({'client_id': client_id}, sort=[('hit_count', 1), ('created_at', 1)])
        if doc:
            await self.col.delete_one({'_id': doc['_id']})
            logger.info("Evicted lesson for client '%s': %s", client_id, doc.get('lesson', '')[:60])