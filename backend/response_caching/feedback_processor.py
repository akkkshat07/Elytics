import asyncio
import os
import pandas as pd
import logging
import re
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import hashlib
from response_caching.config_manager import get_client_cache_dir, get_client_vector_db_path, get_client_db_collection_name, ensure_client_cache_infrastructure
from services.subscription_service import get_client_subscription
logger = logging.getLogger(__name__)

class FeedbackProcessor:

    def __init__(self, client_id: str, dataset_id: Optional[str]=None):
        if not client_id:
            raise ValueError('client_id is required for FeedbackProcessor')
        self.client_id = client_id
        self.dataset_id = dataset_id
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config.system_config import VECTOR_DB_CONFIG
        self.config = VECTOR_DB_CONFIG
        ensure_client_cache_infrastructure(client_id, dataset_id)
        cache_dir = get_client_cache_dir(client_id, dataset_id)
        self.csv_path = str(cache_dir / 'correct_responses.csv')
        self.parquet_path = str(cache_dir / 'correct_responses.parquet')
        base_collection = get_client_db_collection_name(client_id)
        safe_ds = ''.join((c if c.isalnum() else '_' for c in dataset_id or ''))
        self.collection_name = f'{base_collection}_{safe_ds}' if safe_ds else base_collection
        logger.info(f"FeedbackProcessor initialized for client: {client_id}, dataset: {dataset_id or 'default'}")
        logger.info(f'  CSV Path: {self.csv_path}')
        logger.info(f'  Collection: {self.collection_name}')

    def initialize(self) -> bool:
        try:
            from util.embedding_utils import get_embedding_dimension
            from util.qdrant_utils import ensure_collection
            dim = get_embedding_dimension()
            ensure_collection(self.collection_name, dim, client_id=self.client_id)
            logger.info(f'Feedback processor initialized successfully for client: {self.client_id}')
            return True
        except Exception as e:
            logger.warning(f"Qdrant init failed for client '{self.client_id}' (non-fatal): {e}")
            return True

    def _normalize_question(self, text: str) -> str:
        return (str(text) if pd.notna(text) else '').strip().lower()

    def _question_id(self, question_text: str) -> str:
        normalized = self._normalize_question(question_text)
        digest = hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
        return f'q_{digest}'

    def _get_next_question_number(self) -> int:
        try:
            if os.path.exists(self.csv_path):
                df = pd.read_csv(self.csv_path)
                if not df.empty and 'no' in df.columns:
                    return int(df['no'].max()) + 1
            return 1
        except Exception as e:
            logger.warning(f'Could not determine next question number: {e}')
            return 1

    def get_cache_count(self) -> int:
        try:
            if os.path.exists(self.csv_path):
                df = pd.read_csv(self.csv_path)
                return len(df) if not df.empty else 0
            return 0
        except Exception as e:
            logger.warning(f'Could not get cache count: {e}')
            return 0

    def evict_oldest(self) -> None:
        try:
            if not os.path.exists(self.csv_path):
                return
            df = pd.read_csv(self.csv_path)
            if df.empty:
                return
            first_row = df.iloc[0]
            question = first_row.get('question', '')
            if not question or (isinstance(question, float) and pd.isna(question)):
                question = ''
            question_id = self._question_id(str(question).strip())
            try:
                from util.qdrant_utils import delete_by_ids, collection_exists
                if collection_exists(self.collection_name, client_id=self.client_id):
                    delete_by_ids(self.collection_name, [question_id], client_id=self.client_id)
                    logger.info(f'Evicted oldest cached question from Qdrant: {question_id}')
            except Exception as e:
                logger.warning(f'Qdrant delete during evict_oldest: {e}')
            df_rest = df.iloc[1:].reset_index(drop=True)
            df_rest.to_csv(self.csv_path, index=False)
            try:
                parquet_path = Path(self.csv_path).with_suffix('.parquet')
                df_rest.to_parquet(parquet_path, index=False)
            except Exception as pe:
                logger.error(f'Failed to update Parquet during evict_oldest: {pe}')
        except Exception as e:
            logger.error(f'Failed to evict oldest: {e}')

    def _add_to_csv(self, question: str, planner_response: str, python_response: str, business_response: str, cached_code: str='', cached_planner_response: str='', cached_executor_response: str='', cached_business_response: str='', semantic_signature_json: str='', cache_key: str='', user_id: Optional[str]=None, created_at: Optional[str]=None) -> int:
        try:
            question_no = self._get_next_question_number()
            new_row = {'no': question_no, 'question': question, 'planner_agent_response': planner_response, 'python_agent_response': python_response, 'business_agent_response': business_response, 'cached_code': cached_code, 'cached_planner_response': cached_planner_response, 'cached_executor_response': cached_executor_response, 'cached_business_response': cached_business_response, 'semantic_signature_json': semantic_signature_json, 'cache_key': cache_key, 'user_id': user_id if user_id else '', 'created_at': created_at if created_at else ''}
            all_columns = ['no', 'question', 'planner_agent_response', 'python_agent_response', 'business_agent_response', 'cached_code', 'cached_planner_response', 'cached_executor_response', 'cached_business_response', 'semantic_signature_json', 'cache_key', 'user_id', 'created_at']
            if os.path.exists(self.csv_path):
                df = pd.read_csv(self.csv_path)
                for col in all_columns:
                    if col not in df.columns:
                        df[col] = ''
            else:
                df = pd.DataFrame(columns=all_columns)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(self.csv_path, index=False)
            logger.info(f'Added question {question_no} to CSV: {self.csv_path}')
            try:
                parquet_path = Path(self.csv_path).with_suffix('.parquet')
                df.to_parquet(parquet_path, index=False)
                logger.info(f'Updated Parquet file: {parquet_path}')
            except Exception as pe:
                logger.error(f'Failed to update Parquet: {pe}')
            try:
                from response_caching.semantic_cache_manager import SemanticCacheManager
                cache_key = f"{self.client_id}:{self.dataset_id or ''}"
                if cache_key in SemanticCacheManager._MATCHER_CACHE:
                    SemanticCacheManager._MATCHER_CACHE[cache_key].responses_df = None
                    logger.info("Invalidated ResponseMatcher responses_df for client '%s', dataset '%s' after write.", self.client_id, self.dataset_id or 'default')
            except Exception as _e:
                logger.warning(f'Could not invalidate ResponseMatcher cache: {_e}')
            try:
                from response_caching.suggester import warm_reload
                warm_reload(self.client_id, dataset_id=self.dataset_id)
            except Exception as _e:
                logger.warning(f'Could not reload suggester cache after CSV write: {_e}')
            return question_no
        except Exception as e:
            logger.error(f'Failed to add question to CSV: {e}')
            raise

    def _add_to_vector_db(self, question: str, question_no: int, planner_response: str, python_response: str, business_response: str, cached_code: str='', user_id: Optional[str]=None) -> bool:
        try:
            from util.embedding_utils import generate_embedding
            from util.qdrant_utils import upsert_points, ensure_collection, get_by_ids
            from util.embedding_utils import get_embedding_dimension
            embedding = generate_embedding(question)
            question_id = self._question_id(question)
            ensure_collection(self.collection_name, get_embedding_dimension(), client_id=self.client_id)
            try:
                from response_caching.response_matcher import ResponseMatcher as _RM
                _tmp_rm = _RM.__new__(_RM)
                _tmp_rm.locations = set()
                operation_type = _tmp_rm.extract_query_pattern(question) or ''
            except Exception:
                operation_type = ''
            payload = {'question_id': question_no, 'question': question, 'operation_type': operation_type, 'planner_response': planner_response[:1000] if planner_response else '', 'python_response': python_response[:1000] if python_response else '', 'business_response': business_response[:1000] if business_response else '', 'cached_code': cached_code[:5000] if cached_code else '', 'user_id': user_id if user_id else ''}
            upsert_points(collection_name=self.collection_name, ids=[question_id], vectors=[embedding], payloads=[payload], documents=[question], client_id=self.client_id)
            logger.info(f'Added question {question_no} to Qdrant with ID: {question_id}')
            return True
        except Exception as e:
            logger.error(f'Failed to add question to vector database: {e}')
            return False

    def process_positive_feedback(self, conversation_data: Dict[str, Any], user_id: Optional[str]=None, created_at: Optional[str]=None) -> bool:
        try:
            if not self.initialize():
                return False
            user_question = (conversation_data.get('user_input') or conversation_data.get('input') or '').strip()
            enhanced_question = (conversation_data.get('enhanced_question') or '').strip()
            question_to_cache = enhanced_question if enhanced_question else user_question
            if not question_to_cache:
                logger.error('No user question found in conversation data')
                return False
            if enhanced_question:
                logger.info('Using enhanced_question for cache: %s...', enhanced_question[:80] + '...' if len(enhanced_question) > 80 else enhanced_question)
            planner_data = conversation_data.get('planner_response', {})
            logger.info(f"Processing planner_data type: {type(planner_data)}, has 'plan' key: {('plan' in planner_data if isinstance(planner_data, dict) else False)}")
            if isinstance(planner_data, dict):
                planner_response = planner_data.get('plan', '')
                if not planner_response:
                    logger.warning(f"No 'plan' field in planner_data: {list(planner_data.keys())}")
                    planner_response = str(planner_data)
            else:
                planner_response = str(planner_data)
                logger.info(f'planner_data is string, length: {len(planner_response)}')
            planner_response = re.sub('```(?:json)?\\s*', '', planner_response)
            planner_response = re.sub('```\\s*$', '', planner_response)
            if planner_response.strip().startswith('{') and planner_response.strip().endswith('}'):
                try:
                    parsed = json.loads(planner_response)
                    if isinstance(parsed, dict) and 'plan' in parsed:
                        planner_response = parsed['plan']
                        logger.info('Extracted plan from JSON string representation')
                except json.JSONDecodeError as e:
                    logger.warning(f'Failed to parse JSON string: {e}')
                    pass
            planner_response = planner_response.strip()
            logger.info(f"Final planner_response length: {len(planner_response)}, starts with: {(planner_response[:50] if planner_response else 'EMPTY')}...")
            if isinstance(planner_data, dict):
                cached_planner_response = json.dumps(planner_data, ensure_ascii=False)
            elif planner_data is None:
                cached_planner_response = ''
            else:
                cached_planner_response = str(planner_data)
            cached_code = conversation_data.get('code', '')
            python_response = cached_code
            business_data = conversation_data.get('business_response', {})
            if isinstance(business_data, dict):
                business_response = business_data.get('analysis', business_data.get('business_insights', ''))
            else:
                business_response = str(business_data)
            executor_data = conversation_data.get('executor_response', {})
            if isinstance(executor_data, (dict, list)):
                cached_executor_response = json.dumps(executor_data, ensure_ascii=False)
            elif executor_data is None:
                cached_executor_response = ''
            else:
                cached_executor_response = str(executor_data)
            if isinstance(business_data, (dict, list)):
                cached_business_response = json.dumps(business_data, ensure_ascii=False)
            elif business_data is None:
                cached_business_response = ''
            else:
                cached_business_response = str(business_data)
            question_id = self._question_id(question_to_cache)
            csv_has_entry = False
            existing_question_no = None
            try:
                if os.path.exists(self.csv_path):
                    _df = pd.read_csv(self.csv_path)
                    _match = _df[_df['question'].str.strip().str.lower() == question_to_cache.strip().lower()]
                    if not _match.empty:
                        csv_has_entry = True
                        existing_question_no = int(_match.iloc[0]['no'])
            except Exception:
                pass
            if csv_has_entry:
                try:
                    from util.qdrant_utils import get_by_ids, collection_exists
                    if collection_exists(self.collection_name, client_id=self.client_id):
                        existing = get_by_ids(self.collection_name, [question_id], client_id=self.client_id)
                        if existing:
                            logger.info(f"Question already fully cached (CSV + Qdrant): '{question_to_cache[:50]}' (no={existing_question_no})")
                            return True
                        else:
                            logger.warning(f'Question #{existing_question_no} in CSV but missing from Qdrant — re-indexing into Qdrant.')
                            qdrant_ok = self._add_to_vector_db(question=question_to_cache, question_no=existing_question_no, planner_response=planner_response, python_response=python_response, business_response=business_response, cached_code=cached_code, user_id=user_id)
                            if not qdrant_ok:
                                logger.warning(f'Qdrant re-index failed for question #{existing_question_no}')
                            return True
                except Exception:
                    return True
            question_no = self._add_to_csv(question=question_to_cache, planner_response=planner_response, python_response=python_response, business_response=business_response, cached_code=cached_code, cached_planner_response=cached_planner_response, cached_executor_response=cached_executor_response, cached_business_response=cached_business_response, user_id=user_id, created_at=created_at)
            logger.info(f"Successfully cached question #{question_no} for: '{question_to_cache[:80]}' (cached_code length: {len(cached_code)})")
            qdrant_ok = self._add_to_vector_db(question=question_to_cache, question_no=question_no, planner_response=planner_response, python_response=python_response, business_response=business_response, cached_code=cached_code, user_id=user_id)
            if not qdrant_ok:
                logger.warning(f"Qdrant indexing failed for question #{question_no} — question is saved in CSV but won't appear in similarity search until Qdrant recovers.")
            return True
        except Exception as e:
            logger.error(f'Failed to process positive feedback: {e}')
            return False

def get_total_cache_count(client_id: str) -> int:
    base_dir = get_client_cache_dir(client_id)
    if not base_dir.exists():
        return 0
    total = 0
    root_csv = base_dir / 'correct_responses.csv'
    if root_csv.exists():
        try:
            total += len(pd.read_csv(root_csv))
        except Exception as e:
            logger.warning(f'Could not read root cache CSV for count: {e}')
    for sub in base_dir.iterdir():
        if sub.is_dir() and sub.name != 'VectorDb':
            ds_csv = sub / 'correct_responses.csv'
            if ds_csv.exists():
                try:
                    total += len(pd.read_csv(ds_csv))
                except Exception as e:
                    logger.warning(f"Could not read dataset cache CSV '{sub.name}' for count: {e}")
    return total

async def _process_feedback_from_conversation(conversation: Dict[str, Any], mongo_manager) -> bool:
    try:
        if not conversation:
            return False
        user_id = conversation.get('user_id', '')
        created_at = None
        raw_created = conversation.get('created_at')
        if raw_created:
            if hasattr(raw_created, 'isoformat'):
                created_at = raw_created.isoformat()
            else:
                created_at = str(raw_created)
        logger.info(f'Extracted user_id from conversation: {user_id}, created_at: {created_at}')
        client_id = conversation.get('client_id')
        if not client_id:
            logger.error('CRITICAL: No client_id found for conversation. Cannot cache response.')
            return False
        logger.info(f'Processing feedback for client: {client_id}')
        agent_responses = conversation.get('agent_responses', {})
        enhanced_q = conversation.get('enhanced_question')
        regular_input = conversation.get('input', '')
        if enhanced_q:
            logger.info(f'Using enhanced_question for caching: {enhanced_q[:100]}...')
        else:
            logger.info(f'No enhanced_question found, using regular input: {regular_input[:100]}...')
        conversation_data = {'user_input': regular_input, 'enhanced_question': enhanced_q or '', 'planner_response': agent_responses.get('planner', {}), 'code': agent_responses.get('python', ''), 'business_response': agent_responses.get('business', {}), 'executor_response': agent_responses.get('executor', {})}
        dataset_id = conversation.get('dataset_id')
        processor = FeedbackProcessor(client_id=client_id, dataset_id=dataset_id)
        await mongo_manager.connect()
        subscription = await get_client_subscription(client_id, mongo_manager.db)
        max_cached = subscription.get('features', {}).get('max_cached_questions')
        if max_cached is not None and processor.initialize():
            if get_total_cache_count(client_id) >= max_cached:
                await asyncio.to_thread(processor.evict_oldest)
        return await asyncio.to_thread(processor.process_positive_feedback, conversation_data, user_id, created_at)
    except Exception as e:
        logger.error(f'Failed to process feedback from conversation: {e}')
        return False

async def process_feedback_from_run_id(run_id: str, mongo_manager) -> bool:
    try:
        conversation = await mongo_manager.get_conversation_by_run_id(run_id)
        if not conversation:
            logger.error(f'Conversation not found for run_id: {run_id}')
            return False
        return await _process_feedback_from_conversation(conversation, mongo_manager)
    except Exception as e:
        logger.error(f'Failed to process feedback from run_id {run_id}: {e}')
        return False

async def process_feedback_from_conversation_id(conversation_id: str, mongo_manager) -> bool:
    try:
        from bson import ObjectId
        if not ObjectId.is_valid(conversation_id):
            logger.error(f'Invalid conversation_id for cache warmup: {conversation_id}')
            return False
        await mongo_manager.connect()
        collection = mongo_manager.db.conversations
        conversation = await collection.find_one({'_id': ObjectId(conversation_id)})
        if not conversation:
            logger.error(f'Conversation not found for conversation_id: {conversation_id}')
            return False
        return await _process_feedback_from_conversation(conversation, mongo_manager)
    except Exception as e:
        logger.error(f'Failed to process feedback from conversation_id {conversation_id}: {e}')
        return False