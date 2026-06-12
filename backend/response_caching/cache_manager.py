import os
import sys
import pandas as pd
import logging
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()
logger = logging.getLogger(__name__)

def _sanitize_text(value) -> str:
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    return str(value)

class CacheManager:

    def __init__(self, client_id: Optional[str]=None, dataset_id: Optional[str]=None):
        from config.system_config import VECTOR_DB_CONFIG
        self.config = VECTOR_DB_CONFIG
        self.client_id = client_id
        self.dataset_id = dataset_id
        if client_id:
            from response_caching.config_manager import get_client_cache_dir, get_client_db_collection_name, ensure_client_cache_infrastructure
            ensure_client_cache_infrastructure(client_id, dataset_id)
            cache_dir = get_client_cache_dir(client_id, dataset_id)
            self.csv_path = str(cache_dir / 'correct_responses.csv')
            self.parquet_path = str(cache_dir / 'correct_responses.parquet')
            base_collection = get_client_db_collection_name(client_id)
            safe_ds = ''.join((c if c.isalnum() else '_' for c in dataset_id or ''))
            self.collection_name = f'{base_collection}_{safe_ds}' if safe_ds else base_collection
            logger.info(f"CacheManager initialized for client: {client_id}, dataset: {dataset_id or 'default'}")
            logger.info(f'  CSV Path: {self.csv_path}')
            logger.info(f'  Parquet Path: {self.parquet_path}')
            logger.info(f'  Collection: {self.collection_name}')
        else:
            self.csv_path = self.config['source_csv']
            self.collection_name = self.config['collection_name']
            try:
                self.parquet_path = str(Path(self.csv_path).with_suffix('.parquet'))
            except Exception:
                self.parquet_path = os.path.join(os.path.dirname(self.csv_path), 'correct_responses.parquet')
            logger.warning('No client_id provided - using default global paths (not recommended for production)')

    def _get_question_id(self, question_text: str) -> str:
        normalized = (question_text or '').strip().lower()
        digest = hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
        return f'q_{digest}'

    def _init_vector_db(self) -> bool:
        try:
            from util.qdrant_utils import collection_exists
            return collection_exists(self.collection_name, client_id=self.client_id)
        except Exception as e:
            logger.error(f'Error checking Qdrant collection: {e}')
            return False

    def _load_dataset(self) -> Tuple[pd.DataFrame, str]:
        last_error = None
        if os.path.exists(self.parquet_path):
            try:
                df = pd.read_parquet(self.parquet_path)
                return (df, 'parquet')
            except Exception as e:
                last_error = e
                logger.warning(f'Failed to read Parquet during dataset load: {e}')
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                return (df, 'csv')
            except Exception as e:
                last_error = e
                logger.warning(f'Failed to read CSV during dataset load: {e}')
        error_msg = f'No cache files found. CSV: {self.csv_path}, Parquet: {self.parquet_path}'
        if last_error:
            error_msg += f'. Last error: {last_error}'
        raise FileNotFoundError(error_msg)

    def _write_dataset(self, df: pd.DataFrame) -> None:
        if 'no' in df.columns:
            try:
                df['no'] = df['no'].astype(int)
            except Exception:
                pass
        df.to_csv(self.csv_path, index=False)
        df.to_parquet(self.parquet_path, index=False)

    def list_questions(self, page: int=1, page_size: int=50, search_query: Optional[str]=None, id_min: Optional[int]=None, id_max: Optional[int]=None, sort_by: Optional[str]=None, sort_order: Optional[str]=None, storage_filter: Optional[str]=None, date_from: Optional[str]=None, date_to: Optional[str]=None) -> Dict:
        try:
            try:
                df, source = self._load_dataset()
                csv_df = pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else None
            except FileNotFoundError as fnf:
                logger.info(f"No cache files found for client '{self.client_id}': {fnf}")
                return {'success': True, 'data': [], 'pagination': {'page': page, 'page_size': page_size, 'total_items': 0, 'total_pages': 0}, 'source': 'none'}
            except Exception as e:
                logger.error(f'Failed loading dataset: {e}')
                return {'success': False, 'error': str(e), 'data': []}
            filtered_df = df.copy()
            if search_query:
                mask = filtered_df['question'].str.contains(search_query, case=False, na=False)
                filtered_df = filtered_df[mask]
            if id_min is not None:
                filtered_df = filtered_df[filtered_df['no'] >= id_min]
            if id_max is not None:
                filtered_df = filtered_df[filtered_df['no'] <= id_max]
            if 'created_at' in filtered_df.columns and (date_from or date_to):
                created_at_series = filtered_df['created_at'].astype(str).str.strip()
                filtered_df = filtered_df[created_at_series != '']
                created_at_series = filtered_df['created_at'].astype(str).str.strip()
                if date_from:
                    filtered_df = filtered_df[created_at_series >= date_from]
                    created_at_series = filtered_df['created_at'].astype(str).str.strip()
                if date_to:
                    filtered_df = filtered_df[created_at_series <= date_to]
            csv_ids: set = set()
            if csv_df is not None and 'no' in csv_df.columns:
                try:
                    csv_ids = set(csv_df['no'].astype(int).tolist())
                except Exception:
                    csv_ids = set(csv_df['no'].tolist())
            try:
                parquet_ids = set(df['no'].astype(int).tolist())
            except Exception:
                parquet_ids = set(df['no'].tolist())
            vector_presence = {}
            if self._init_vector_db():
                try:
                    from util.qdrant_utils import get_by_ids
                    vector_ids = []
                    id_map = {}
                    for _, row in filtered_df.iterrows():
                        question_text = str(row.get('question', '') or '')
                        question_id = int(row.get('no'))
                        vector_id = self._get_question_id(question_text)
                        vector_ids.append(vector_id)
                        id_map[vector_id] = question_id
                    if vector_ids:
                        results = get_by_ids(self.collection_name, vector_ids, client_id=self.client_id)
                        existing_string_ids = set()
                        for r in results:
                            sid = r['payload'].get('_string_id', '')
                            if sid:
                                existing_string_ids.add(sid)
                        for vector_id, qid in id_map.items():
                            vector_presence[qid] = vector_id in existing_string_ids
                except Exception as vector_error:
                    logger.warning(f'Failed to determine vector presence: {vector_error}')
                    vector_presence = {}
            if storage_filter:
                if storage_filter.lower() == 'csv':
                    filtered_df = filtered_df[filtered_df['no'].isin(csv_ids)]
                elif storage_filter.lower() == 'parquet':
                    filtered_df = filtered_df[filtered_df['no'].isin(parquet_ids)]
                elif storage_filter.lower() == 'vector_db':
                    vector_db_ids = [qid for qid, present in vector_presence.items() if present]
                    filtered_df = filtered_df[filtered_df['no'].isin(vector_db_ids)]
            if sort_by:
                valid_sort_fields = ['id', 'question', 'user_id']
                if sort_by not in valid_sort_fields:
                    sort_by = 'id'
                if sort_by == 'id':
                    sort_column = 'no'
                else:
                    sort_column = sort_by
                if sort_column not in filtered_df.columns:
                    if sort_column == 'user_id':
                        filtered_df['user_id'] = ''
                    else:
                        sort_by = 'id'
                        sort_column = 'no'
                if sort_order and sort_order.lower() == 'asc':
                    ascending = True
                else:
                    ascending = False
                if sort_by == 'id' and sort_order is None:
                    ascending = False
                elif sort_by != 'id' and sort_order is None:
                    ascending = True
                filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending, na_position='last')
            else:
                filtered_df = filtered_df.sort_values(by='no', ascending=False, na_position='last')
            total_items = len(filtered_df)
            total_pages = (total_items + page_size - 1) // page_size
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_df = filtered_df.iloc[start_idx:end_idx]
            page_vector_presence = {}
            for qid in page_df['no'].tolist():
                page_vector_presence[qid] = vector_presence.get(qid, False)
            questions = []
            for _, row in page_df.iterrows():
                question_id = int(row['no'])
                user_id = ''
                if 'user_id' in row.index:
                    user_id_val = row.get('user_id', '')
                    if pd.notna(user_id_val) and str(user_id_val).strip():
                        user_id = str(user_id_val).strip()
                created_at = None
                if 'created_at' in row.index:
                    created_at_val = row.get('created_at', '')
                    if pd.notna(created_at_val) and str(created_at_val).strip():
                        created_at = str(created_at_val).strip()
                questions.append({'id': question_id, 'question': str(row['question']), 'user_id': user_id, 'created_at': created_at, 'has_planner': pd.notna(row.get('planner_agent_response')), 'has_python': pd.notna(row.get('python_agent_response')), 'has_business': pd.notna(row.get('business_agent_response')), 'present_in_csv': question_id in csv_ids, 'present_in_parquet': question_id in parquet_ids, 'present_in_vector_db': page_vector_presence.get(question_id, False)})
            return {'success': True, 'data': questions, 'pagination': {'page': page, 'page_size': page_size, 'total_items': total_items, 'total_pages': total_pages}, 'source': source}
        except Exception as e:
            logger.error(f'Error listing questions: {e}')
            return {'success': False, 'error': str(e), 'data': []}

    def search_questions(self, query: str) -> Dict:
        result = self.list_questions(page=1, page_size=100, search_query=query)
        if result.get('success') and result.get('data'):
            return {'success': True, 'data': result['data'], 'count': len(result['data'])}
        return result

    def get_question_detail(self, question_id: int) -> Dict:
        try:
            df, _ = self._load_dataset()
        except FileNotFoundError:
            return {'success': False, 'error': 'Question not found', 'code': 'not_found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
        matches = df[df['no'] == question_id]
        if matches.empty:
            return {'success': False, 'error': 'Question not found', 'code': 'not_found'}
        row = matches.iloc[0]
        detail = {'id': int(row['no']), 'question': _sanitize_text(row.get('question')), 'planner_agent_response': _sanitize_text(row.get('planner_agent_response')), 'python_agent_response': _sanitize_text(row.get('python_agent_response')), 'business_agent_response': _sanitize_text(row.get('business_agent_response'))}
        detail['present_in_csv'] = os.path.exists(self.csv_path)
        detail['present_in_parquet'] = os.path.exists(self.parquet_path)
        detail['present_in_vector_db'] = False
        if self._init_vector_db():
            try:
                from util.qdrant_utils import scroll_by_filter
                results = scroll_by_filter(self.collection_name, {'question_id': int(question_id)}, limit=1, client_id=self.client_id)
                detail['present_in_vector_db'] = bool(results)
            except Exception as vector_error:
                logger.warning(f'Failed to determine vector presence for question {question_id}: {vector_error}')
        return {'success': True, 'data': detail}

    def preview_delete(self, question_ids: List[int]) -> Dict:
        try:
            if os.path.exists(self.parquet_path):
                df = pd.read_parquet(self.parquet_path)
            elif os.path.exists(self.csv_path):
                df = pd.read_csv(self.csv_path)
            else:
                return {'success': False, 'error': 'No cache file found'}
            matching = df[df['no'].isin(question_ids)]
            not_found = set(question_ids) - set(matching['no'].tolist())
            questions_to_delete = []
            for _, row in matching.iterrows():
                questions_to_delete.append({'id': int(row['no']), 'question': str(row['question'])})
            return {'success': True, 'questions_to_delete': questions_to_delete, 'count': len(questions_to_delete), 'not_found_ids': list(not_found), 'total_before': len(df), 'total_after': len(df) - len(questions_to_delete)}
        except Exception as e:
            logger.error(f'Error previewing delete: {e}')
            return {'success': False, 'error': str(e)}

    def delete_questions(self, question_ids: List[int]) -> Dict:
        result = {'success': False, 'deleted_from_csv': 0, 'deleted_from_parquet': 0, 'deleted_from_vector_db': 0, 'errors': []}
        try:
            questions_to_delete: List[str] = []
            csv_deleted = 0
            if os.path.exists(self.csv_path):
                try:
                    df_csv = pd.read_csv(self.csv_path)
                    original_count = len(df_csv)
                    questions_to_delete = df_csv[df_csv['no'].isin(question_ids)]['question'].tolist()
                    df_csv = df_csv[~df_csv['no'].isin(question_ids)]
                    csv_deleted = original_count - len(df_csv)
                    df_csv.to_csv(self.csv_path, index=False)
                    result['deleted_from_csv'] = csv_deleted
                    logger.info(f'Deleted {csv_deleted} questions from CSV')
                except Exception as e:
                    error_msg = f'CSV deletion error: {str(e)}'
                    result['errors'].append(error_msg)
                    logger.error(error_msg)
            parquet_deleted = 0
            if os.path.exists(self.parquet_path):
                try:
                    df_parquet = pd.read_parquet(self.parquet_path)
                    original_count = len(df_parquet)
                    df_parquet = df_parquet[~df_parquet['no'].isin(question_ids)]
                    parquet_deleted = original_count - len(df_parquet)
                    df_parquet.to_parquet(self.parquet_path, index=False)
                    result['deleted_from_parquet'] = parquet_deleted
                    logger.info(f'Deleted {parquet_deleted} questions from Parquet')
                except Exception as e:
                    error_msg = f'Parquet deletion error: {str(e)}'
                    result['errors'].append(error_msg)
                    logger.error(error_msg)
            vector_deleted = 0
            if self._init_vector_db():
                try:
                    from util.qdrant_utils import delete_by_ids, count_points
                    try:
                        before_count = count_points(self.collection_name, client_id=self.client_id)
                    except Exception:
                        before_count = None
                    vector_ids = [self._get_question_id(q) for q in questions_to_delete if q]
                    if vector_ids:
                        delete_by_ids(self.collection_name, vector_ids, client_id=self.client_id)
                    try:
                        after_count = count_points(self.collection_name, client_id=self.client_id)
                        if before_count is not None and after_count is not None:
                            vector_deleted = max(before_count - after_count, 0)
                        else:
                            vector_deleted = len(vector_ids) if vector_ids else 0
                    except Exception:
                        vector_deleted = len(vector_ids) if vector_ids else 0
                    result['deleted_from_vector_db'] = vector_deleted
                    logger.info(f'Vector DB delete summary -> ids_removed: {len(vector_ids)}, count_delta: {vector_deleted}')
                except Exception as e:
                    error_msg = f'Vector DB deletion error: {str(e)}'
                    result['errors'].append(error_msg)
                    logger.error(error_msg)
            result['success'] = csv_deleted > 0 or parquet_deleted > 0 or vector_deleted > 0
            return result
        except Exception as e:
            logger.error(f'Error deleting questions: {e}')
            result['errors'].append(str(e))
            return result

    def update_question(self, question_id: int, updates: Dict[str, Optional[str]]) -> Dict:
        try:
            df, _ = self._load_dataset()
        except Exception as e:
            return {'success': False, 'error': str(e)}
        matches = df[df['no'] == question_id]
        if matches.empty:
            return {'success': False, 'error': 'Question not found', 'code': 'not_found'}
        row_index = matches.index[0]
        original_row = matches.iloc[0].copy()
        allowed_fields = {'question', 'planner_agent_response', 'python_agent_response', 'business_agent_response'}
        applied_updates: Dict[str, str] = {}
        for field, value in updates.items():
            if field not in allowed_fields or value is None:
                continue
            if field not in df.columns:
                df[field] = ''
            df.at[row_index, field] = value
            applied_updates[field] = value
        if not applied_updates:
            return {'success': False, 'error': 'No valid fields provided', 'code': 'invalid_request'}
        new_question = _sanitize_text(df.at[row_index, 'question']) if 'question' in df.columns else ''
        old_question = _sanitize_text(original_row.get('question'))
        question_changed = new_question != old_question
        if not new_question.strip():
            return {'success': False, 'error': 'Question text cannot be empty', 'code': 'invalid_request'}
        embedding = None
        try:
            from util.embedding_utils import generate_embedding
            embedding = generate_embedding(new_question)
        except Exception as embedding_error:
            logger.error(f'Failed generating embedding during update: {embedding_error}')
            return {'success': False, 'error': f'Failed to generate embedding: {embedding_error}'}
        try:
            self._write_dataset(df)
        except Exception as write_error:
            logger.error(f'Failed to persist dataset update: {write_error}')
            return {'success': False, 'error': f'Failed to persist updates: {write_error}'}
        vector_updated = False
        vector_errors: List[str] = []
        if self._init_vector_db():
            try:
                from util.qdrant_utils import upsert_points, delete_by_ids
                from util.embedding_utils import get_embedding_dimension
                payload = {'question_id': int(question_id), 'question': new_question, 'planner_response': _sanitize_text(df.at[row_index, 'planner_agent_response'])[:1000] if 'planner_agent_response' in df.columns else '', 'python_response': _sanitize_text(df.at[row_index, 'python_agent_response'])[:1000] if 'python_agent_response' in df.columns else '', 'business_response': _sanitize_text(df.at[row_index, 'business_agent_response'])[:1000] if 'business_agent_response' in df.columns else ''}
                old_vector_id = self._get_question_id(old_question)
                new_vector_id = self._get_question_id(new_question)
                if question_changed:
                    try:
                        delete_by_ids(self.collection_name, [old_vector_id], client_id=self.client_id)
                    except Exception:
                        pass
                upsert_points(collection_name=self.collection_name, ids=[new_vector_id], vectors=[embedding], payloads=[payload], documents=[new_question], client_id=self.client_id)
                vector_updated = True
            except Exception as vector_error:
                logger.error(f'Failed updating vector DB: {vector_error}')
                vector_errors.append(str(vector_error))
        detail = self.get_question_detail(question_id)
        if not detail.get('success'):
            detail.setdefault('errors', []).extend(vector_errors)
            detail['updated_fields'] = list(applied_updates.keys())
            detail['vector_updated'] = vector_updated
            return detail
        detail.setdefault('data', {})
        detail['data']['updated_fields'] = list(applied_updates.keys())
        detail['data']['vector_updated'] = vector_updated
        if vector_errors:
            detail['data']['vector_errors'] = vector_errors
        detail['success'] = True
        return detail

    def get_stats(self) -> Dict:
        try:
            stats = {'success': True, 'total_questions': 0, 'csv_exists': False, 'parquet_exists': False, 'vector_db_exists': False, 'csv_count': 0, 'parquet_count': 0, 'vector_db_count': 0, 'csv_size_mb': 0, 'parquet_size_mb': 0}
            if os.path.exists(self.csv_path):
                stats['csv_exists'] = True
                stats['csv_size_mb'] = round(os.path.getsize(self.csv_path) / (1024 * 1024), 2)
                df = pd.read_csv(self.csv_path)
                stats['csv_count'] = len(df)
            if os.path.exists(self.parquet_path):
                stats['parquet_exists'] = True
                stats['parquet_size_mb'] = round(os.path.getsize(self.parquet_path) / (1024 * 1024), 2)
                df = pd.read_parquet(self.parquet_path)
                stats['parquet_count'] = len(df)
            if self._init_vector_db():
                stats['vector_db_exists'] = True
                try:
                    from util.qdrant_utils import count_points
                    stats['vector_db_count'] = count_points(self.collection_name, client_id=self.client_id)
                except Exception:
                    stats['vector_db_count'] = 0
            stats['total_questions'] = max(stats['csv_count'], stats['parquet_count'], stats['vector_db_count'])
            return stats
        except Exception as e:
            logger.error(f'Error getting stats: {e}')
            return {'success': False, 'error': str(e)}

    def rebuild_vector_db(self) -> Dict:
        try:
            from response_caching.create_vector_db import create_vector_database
            client_id = self.client_id
            if client_id:
                logger.info(f'Starting vector database rebuild for client: {client_id}')
            else:
                logger.info('Starting vector database rebuild (global mode)')
            success = create_vector_database(client_id=client_id, dataset_id=self.dataset_id)
            if success:
                return {'success': True, 'message': 'Vector database rebuilt successfully'}
            else:
                return {'success': False, 'error': 'Failed to rebuild vector database'}
        except Exception as e:
            logger.error(f'Error rebuilding vector DB: {e}')
            return {'success': False, 'error': str(e)}