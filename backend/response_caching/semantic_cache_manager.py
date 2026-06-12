import asyncio
import logging
import json
import os
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
from response_caching.models import SemanticSignature, GlobalCacheKey, TenantCacheKey
from response_caching.response_matcher import ResponseMatcher
from config.system_config import VECTOR_DB_CONFIG
logger = logging.getLogger(__name__)

class SemanticCacheManager:
    _MATCHER_CACHE: Dict[str, ResponseMatcher] = {}

    def __init__(self, client_id: str, dataset_id: Optional[str]=None):
        self.client_id = client_id
        self.dataset_id = dataset_id
        cache_key = f"{client_id}:{dataset_id or ''}"
        if cache_key in self._MATCHER_CACHE:
            self.response_matcher = self._MATCHER_CACHE[cache_key]
        else:
            self.response_matcher = ResponseMatcher(client_id=client_id, dataset_id=dataset_id)
            self._MATCHER_CACHE[cache_key] = self.response_matcher
        self.signature_exact_threshold = 1.0
        self.signature_similarity_threshold = 0.9
        if not self.response_matcher._collection_ready:
            self.response_matcher.initialize()
        self.embedding_threshold = self.response_matcher.threshold_guide
        logger.info(f'SemanticCacheManager initialized for client: {client_id}')

    async def cache_has_points(self) -> bool:
        try:
            import asyncio
            from util.qdrant_utils import count_points
            if not self.response_matcher._collection_ready:
                if not self.response_matcher.initialize():
                    return False
            count = await asyncio.to_thread(count_points, self.response_matcher.collection_name, self.client_id)
            return bool(count and count > 0)
        except Exception as e:
            logger.warning("cache_has_points check failed for '%s' (treating as empty): %s", getattr(self.response_matcher, 'collection_name', '?'), e)
            return False

    async def get_top_candidates(self, query_embedding: List[float], limit: int=5, include_cached_payload: bool=False) -> List[Dict[str, Any]]:
        try:
            if not self.response_matcher._collection_ready:
                if not self.response_matcher.initialize():
                    logger.warning('ResponseMatcher not ready — returning empty candidates')
                    return []
            from util.qdrant_utils import search_vectors
            results = search_vectors(collection_name=self.response_matcher.collection_name, query_vector=query_embedding, limit=limit, client_id=self.client_id)
            if not results:
                return []
            candidates = []
            for r in results:
                question_id = r['payload'].get('question_id')
                cached_result = None
                planner_plan = None
                has_cached_code = bool(r['payload'].get('cached_code'))
                if include_cached_payload and question_id is not None:
                    ref = self.response_matcher.get_reference_responses(question_id)
                    if ref:
                        cached_code = ref.get('cached_code', '')
                        planner_text = ref.get('planner_agent_response', '')
                        planner_resp = ref.get('cached_planner_response') or {'plan': planner_text}
                        if planner_text:
                            planner_plan = planner_resp
                        if cached_code:
                            cached_result = {'planner_response': planner_resp, 'code': cached_code}
                            has_cached_code = True
                candidates.append({'index': len(candidates), 'question': r['document'], 'similarity': round(r['score'], 6) if r['score'] else 0.0, 'question_id': question_id, 'has_cached_code': has_cached_code, '_cached_result': cached_result, '_planner_plan': planner_plan})
            logger.info(f"Retrieved {len(candidates)} candidates from Qdrant (top similarity: {candidates[0]['similarity']:.3f})" if candidates else 'Retrieved 0 candidates from Qdrant')
            return candidates
        except ValueError as e:
            if 'not found' in str(e).lower():
                logger.warning("Qdrant collection '%s' missing despite _collection_ready=True — resetting and returning empty candidates: %s", self.response_matcher.collection_name, e)
                self.response_matcher._collection_ready = False
            else:
                logger.error(f'Error retrieving top candidates: {e}', exc_info=True)
            return []
        except Exception as e:
            logger.error(f'Error retrieving top candidates: {e}', exc_info=True)
            return []

    async def lookup(self, semantic_signature: SemanticSignature, query_embedding: List[float], user_question: str) -> Optional[Dict[str, Any]]:
        try:
            if not self.response_matcher._collection_ready:
                if not self.response_matcher.initialize():
                    logger.error('Failed to initialize ResponseMatcher')
                    return None
            exact_match = await self._lookup_exact_signature(semantic_signature)
            if exact_match:
                logger.info(f'Exact signature match found for cache key: {semantic_signature.to_cache_key()}')
                return {'matched': True, 'cached_code': exact_match.get('cached_code'), 'matched_signature': exact_match.get('signature'), 'match_type': 'exact', 'matched_question': exact_match.get('matched_question'), 'matched_question_id': exact_match.get('matched_question_id')}
            similarity_match = await self._lookup_similarity_signature(semantic_signature, query_embedding)
            if similarity_match:
                logger.info(f"High similarity signature match found (score: {similarity_match.get('similarity_score', 0):.3f})")
                return {'matched': True, 'cached_code': similarity_match.get('cached_code'), 'matched_signature': similarity_match.get('signature'), 'match_type': 'signature', 'similarity_score': similarity_match.get('similarity_score'), 'matched_question': similarity_match.get('matched_question'), 'matched_question_id': similarity_match.get('matched_question_id')}
            logger.debug('Attempting embedding similarity lookup...')
            embedding_match = await self._lookup_embedding_similarity(query_embedding, user_question)
            if embedding_match:
                logger.info(f"Embedding similarity match found (score: {embedding_match.get('similarity_score', 0):.3f})")
                return {'matched': True, 'cached_code': embedding_match.get('cached_code'), 'matched_signature': None, 'match_type': 'embedding', 'similarity_score': embedding_match.get('similarity_score'), 'matched_question': embedding_match.get('matched_question'), 'matched_question_id': embedding_match.get('matched_question_id')}
            logger.info(f"No semantic cache match found for query: '{user_question[:80]}...' (signature: {semantic_signature.to_cache_key()})")
            return None
        except Exception as e:
            logger.error(f'Error in semantic cache lookup: {e}', exc_info=True)
            return None

    async def _lookup_exact_signature(self, semantic_signature: SemanticSignature) -> Optional[Dict[str, Any]]:
        try:
            cache_key = semantic_signature.to_cache_key()
            if not self.response_matcher._collection_ready:
                return None
            from util.qdrant_utils import scroll_by_filter
            try:
                results = scroll_by_filter(collection_name=self.response_matcher.collection_name, filter_dict={'cache_key': cache_key}, limit=1, client_id=self.client_id)
                if results:
                    payload = results[0]['payload']
                    question_id = payload.get('question_id')
                    matched_question = payload.get('question') or results[0].get('document', '')
                    if question_id is not None:
                        reference_responses = self.response_matcher.get_reference_responses(question_id)
                        if reference_responses and reference_responses.get('cached_code'):
                            result = {'cached_code': reference_responses['cached_code'], 'signature': semantic_signature.to_dict(), 'matched_question': matched_question, 'matched_question_id': question_id}
                            return result
            except Exception as e:
                logger.debug(f'Exact signature lookup failed (may not have cache_key payload): {e}')
            return None
        except Exception as e:
            logger.warning(f'Error in exact signature lookup: {e}')
            return None

    async def _lookup_similarity_signature(self, semantic_signature: SemanticSignature, query_embedding: List[float]) -> Optional[Dict[str, Any]]:
        try:
            if not self.response_matcher._collection_ready:
                return None
            from util.qdrant_utils import scroll_all
            try:
                all_results = scroll_all(collection_name=self.response_matcher.collection_name, limit=1000, client_id=self.client_id)
                if not all_results:
                    return None
                best_match = None
                best_score = 0.0
                for entry in all_results:
                    payload = entry['payload']
                    sig_json = payload.get('semantic_signature_json')
                    if not sig_json:
                        continue
                    try:
                        stored_sig_data = json.loads(sig_json) if isinstance(sig_json, str) else sig_json
                        stored_sig = SemanticSignature.from_dict(stored_sig_data)
                        similarity = semantic_signature.similarity_score(stored_sig)
                        if similarity >= self.signature_similarity_threshold and similarity > best_score:
                            best_score = similarity
                            question_id = payload.get('question_id')
                            matched_question = payload.get('question') or entry.get('document', '')
                            if question_id is not None:
                                reference_responses = self.response_matcher.get_reference_responses(question_id)
                                if reference_responses and reference_responses.get('cached_code'):
                                    best_match = {'cached_code': reference_responses['cached_code'], 'signature': stored_sig.to_dict(), 'similarity_score': similarity, 'matched_question': matched_question, 'matched_question_id': question_id}
                    except Exception as e:
                        logger.debug(f'Error comparing signature: {e}')
                        continue
                return best_match
            except Exception as e:
                logger.debug(f'Similarity signature lookup failed: {e}')
                return None
        except Exception as e:
            logger.warning(f'Error in similarity signature lookup: {e}')
            return None

    async def _lookup_embedding_similarity(self, query_embedding: List[float], user_question: str) -> Optional[Dict[str, Any]]:
        try:
            if not self.response_matcher._collection_ready:
                if not self.response_matcher.initialize():
                    logger.warning('ResponseMatcher initialization failed for embedding lookup')
                    return None
            logger.debug(f"Performing embedding similarity lookup for: '{user_question[:80]}...'")
            from util.qdrant_utils import search_vectors
            try:
                results = search_vectors(collection_name=self.response_matcher.collection_name, query_vector=query_embedding, limit=5, client_id=self.client_id)
                if not results:
                    return None
                best_match = None
                best_similarity = 0.0
                for result in results:
                    base_similarity = result['score']
                    stored_question = result['document']
                    payload = result['payload']
                    adjusted_similarity = self.response_matcher.calculate_adjusted_similarity(user_question, stored_question, base_similarity)
                    if adjusted_similarity >= self.embedding_threshold and adjusted_similarity > best_similarity:
                        question_id = payload.get('question_id')
                        if question_id is not None:
                            reference_responses = self.response_matcher.get_reference_responses(question_id)
                            if reference_responses and reference_responses.get('cached_code'):
                                best_similarity = adjusted_similarity
                                best_match = {'cached_code': reference_responses['cached_code'], 'similarity_score': adjusted_similarity, 'question_id': question_id, 'matched_question': stored_question, 'matched_question_id': question_id}
                                logger.info(f"Found embedding match: similarity={adjusted_similarity:.3f}, question_id={question_id}, matched_question='{stored_question[:60]}...'")
                return best_match
            except Exception as qdrant_error:
                logger.warning(f'Direct Qdrant query failed, falling back to ResponseMatcher: {qdrant_error}')
                similar_matches = self.response_matcher.find_similar_question(user_question)
                if not similar_matches:
                    return None
                best_match = similar_matches[0]
                similarity = best_match.get('similarity', 0.0)
                if similarity >= self.embedding_threshold:
                    question_id = best_match.get('question_id')
                    matched_question = best_match.get('question')
                    if question_id is not None:
                        reference_responses = self.response_matcher.get_reference_responses(question_id)
                        if reference_responses and reference_responses.get('cached_code'):
                            return {'cached_code': reference_responses['cached_code'], 'similarity_score': similarity, 'matched_question': matched_question, 'matched_question_id': question_id}
            return None
        except Exception as e:
            logger.warning(f'Error in embedding similarity lookup: {e}', exc_info=True)
            return None

    async def store(self, semantic_signature: SemanticSignature, query_embedding: List[float], user_question: str, executor_response: Dict[str, Any], planner_response: Optional[Dict]=None, python_code: Optional[str]=None):
        try:
            from response_caching.cache_manager import CacheManager
            import pandas as pd
            cache_mgr = CacheManager(client_id=self.client_id, dataset_id=self.dataset_id)
            try:
                df, source = await asyncio.to_thread(cache_mgr._load_dataset)
            except FileNotFoundError:
                df = pd.DataFrame(columns=['no', 'question', 'user_id', 'planner_agent_response', 'python_agent_response', 'business_agent_response', 'cached_code', 'semantic_signature_json', 'cache_key'])
                source = 'new'
            global_key = GlobalCacheKey.from_embedding(semantic_signature, query_embedding)
            cache_key = global_key.to_string()
            existing_idx = None
            if 'question' in df.columns:
                exact_match = df[df['question'].str.lower() == user_question.lower()]
                if not exact_match.empty:
                    existing_idx = exact_match.index[0]
                elif 'cache_key' in df.columns:
                    cache_match = df[df['cache_key'] == cache_key]
                    if not cache_match.empty:
                        existing_idx = cache_match.index[0]
            sig_json = json.dumps(semantic_signature.to_dict())
            planner_response_text = ''
            if planner_response:
                if isinstance(planner_response, dict):
                    planner_response_text = planner_response.get('plan', json.dumps(planner_response))
                else:
                    planner_response_text = str(planner_response)
            new_row = {'question': user_question, 'user_id': '', 'planner_agent_response': planner_response_text, 'python_agent_response': python_code or '', 'business_agent_response': '', 'cached_code': python_code or '', 'semantic_signature_json': sig_json, 'cache_key': cache_key}
            if existing_idx is not None:
                for col, val in new_row.items():
                    if col in df.columns:
                        df.at[existing_idx, col] = val
                question_id = int(df.at[existing_idx, 'no'])
                logger.info(f'Updated existing cache entry (question_id: {question_id})')
            else:
                new_row['no'] = int(df['no'].max() + 1) if 'no' in df.columns and len(df) > 0 else 1
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                question_id = new_row['no']
                logger.info(f'Added new cache entry (question_id: {question_id})')
            await asyncio.to_thread(cache_mgr._write_dataset, df)
            from util.qdrant_utils import ensure_collection, upsert_points
            from util.embedding_utils import get_embedding_dimension
            try:
                await asyncio.to_thread(ensure_collection, self.response_matcher.collection_name, get_embedding_dimension(), self.client_id)
                question_id_str = f'q_{hashlib.sha256(user_question.lower().encode()).hexdigest()[:16]}'
                payload = {'question_id': int(question_id), 'question': user_question, 'semantic_variant': semantic_signature.semantic_variant, 'operation_type': semantic_signature.operation_type, 'cache_key': cache_key, 'semantic_signature_json': sig_json, 'planner_response': planner_response_text[:1000] if planner_response_text else '', 'python_response': (python_code or '')[:1000], 'business_response': '', 'cached_code': (python_code or '')[:5000]}
                await asyncio.to_thread(upsert_points, self.response_matcher.collection_name, [question_id_str], [query_embedding], [payload], [user_question], self.client_id)
                logger.info(f'Stored in Qdrant with ID: {question_id_str}')
            except Exception as qdrant_error:
                logger.error(f'Failed to store in Qdrant: {qdrant_error}')
            logger.info(f"Successfully stored semantic cache entry for client '{self.client_id}': cache_key={cache_key}, question_id={question_id}")
        except Exception as e:
            logger.error(f'Error storing semantic cache entry: {e}', exc_info=True)
            import traceback
            logger.error(traceback.format_exc())