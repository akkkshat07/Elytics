from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)

class UserPreferenceService:
    COLLECTION = 'user_preferences'
    PREFERENCE_LABELS = {'chart_type': 'Chart type', 'viz_style': 'Visualization style', 'detail_level': 'Detail level', 'limit': 'Result limit', 'number_format': 'Number format', 'time_granularity': 'Time granularity', 'grouping_preference': 'Grouping'}

    def __init__(self, db: Any):
        self.db = db
        self.col = db[self.COLLECTION]

    async def save_preference(self, client_id: str, user_id: str, preference_type: str, value: str, learned_from: str='', confidence: float=0.8) -> None:
        try:
            await self.col.update_one({'client_id': client_id, 'user_id': user_id, 'preference_type': preference_type}, {'$set': {'value': value, 'confidence': confidence, 'learned_from': learned_from, 'updated_at': datetime.utcnow()}, '$setOnInsert': {'created_at': datetime.utcnow()}}, upsert=True)
        except Exception as e:
            logger.debug('Failed to save user preference: %s', e)

    async def save_preferences_batch(self, client_id: str, user_id: str, preferences: List[Dict[str, Any]]) -> None:
        for pref in preferences:
            await self.save_preference(client_id=client_id, user_id=user_id, preference_type=pref['preference_type'], value=pref['value'], learned_from=pref.get('learned_from', ''), confidence=pref.get('confidence', 0.8))

    async def get_preferences(self, client_id: str, user_id: str) -> Dict[str, str]:
        try:
            cursor = self.col.find({'client_id': client_id, 'user_id': user_id}, {'preference_type': 1, 'value': 1, '_id': 0})
            result = {}
            async for doc in cursor:
                result[doc['preference_type']] = doc['value']
            return result
        except Exception as e:
            logger.debug('Failed to get user preferences: %s', e)
            return {}

    async def format_for_prompt(self, client_id: str, user_id: str, current_query_prefs: Optional[Dict[str, str]]=None, max_tokens: int=300) -> str:
        stored = await self.get_preferences(client_id, user_id)
        if not stored and (not current_query_prefs):
            return ''
        merged = {**stored}
        if current_query_prefs:
            merged.update(current_query_prefs)
        lines = []
        for ptype, value in merged.items():
            label = self.PREFERENCE_LABELS.get(ptype, ptype)
            source = '(current query)' if current_query_prefs and ptype in current_query_prefs else '(learned)'
            lines.append(f'- {label}: {value} {source}')
        text = '\n'.join(lines)
        if len(text) // 4 > max_tokens:
            text = text[:max_tokens * 4]
        return text

    async def delete_preferences(self, client_id: str, user_id: str) -> int:
        try:
            result = await self.col.delete_many({'client_id': client_id, 'user_id': user_id})
            return result.deleted_count
        except Exception as e:
            logger.debug('Failed to delete user preferences: %s', e)
            return 0