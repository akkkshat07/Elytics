import os
import sys
import pandas as pd
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Optional, List, Tuple
import json
from response_caching.config_manager import get_client_cache_dir, get_client_vector_db_path, get_client_db_collection_name
__all__ = ['ResponseMatcher', 'format_reference_for_agent', 'get_reference_guidance']
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger(__name__)

class ResponseMatcher:

    def __init__(self, client_id: Optional[str]=None, dataset_id: Optional[str]=None):
        from config.system_config import VECTOR_DB_CONFIG
        self.config = VECTOR_DB_CONFIG
        self.client_id = client_id
        self.dataset_id = dataset_id
        if client_id:
            safe_ds = ''.join((c if c.isalnum() else '_' for c in dataset_id or ''))
            base_collection = get_client_db_collection_name(client_id)
            self.collection_name = f'{base_collection}_{safe_ds}' if safe_ds else base_collection
            cache_dir = get_client_cache_dir(client_id, dataset_id)
            self.csv_path = str(cache_dir / 'correct_responses.csv')
            self.parquet_path = str(cache_dir / 'correct_responses.parquet')
            logger.info(f'ResponseMatcher initialized for client: {client_id}')
            logger.info(f'  Collection: {self.collection_name}')
            logger.info(f'  CSV Path: {self.csv_path}')
        else:
            self.collection_name = self.config['collection_name']
            self.csv_path = self.config['source_csv']
            self.parquet_path = str(Path(self.csv_path).with_suffix('.parquet'))
            logger.warning('No client_id provided - using default global paths (not recommended)')
        self.similarity_threshold = self.config.get('similarity_threshold', 0.1)
        self.threshold_return = float(self.config.get('threshold_return', 0.995))
        self.threshold_guide = float(self.config.get('threshold_guide', 0.75))
        self.locations = set()
        self.db = None
        self.api_key_env_var = self.config.get('api_key_env_var', 'OPENAI_API_KEY')
        self._collection_ready = False
        self.responses_df = None

    async def load_locations_from_schema(self, db=None):
        if not self.client_id:
            logger.debug('No client_id set, skipping location loading')
            return
        try:
            from services.schema_mapper import SchemaMapper
            schema_mapper = await SchemaMapper.create(self.client_id, db or self.db)
            guardrails = schema_mapper.get_guardrails_config()
            facility_names = guardrails.get('facility_names', [])
            if facility_names:
                self.locations = set((name.lower() for name in facility_names))
                logger.info(f"Loaded {len(self.locations)} locations from SchemaMapper guardrails for client '{self.client_id}'")
            else:
                logger.debug(f"No facility names found in guardrails for client '{self.client_id}'")
        except Exception as e:
            logger.warning(f"Failed to load locations from SchemaMapper for client '{self.client_id}': {e}")

    def initialize(self, db=None) -> bool:
        try:
            if db:
                self.db = db
            if self.client_id:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.load_locations_from_schema(db))
                except RuntimeError:
                    logger.debug('[ResponseMatcher] No running event loop; skipping location pre-load')
            try:
                from util.qdrant_utils import ensure_collection
                from util.embedding_utils import get_embedding_dimension
                ensure_collection(self.collection_name, get_embedding_dimension(), client_id=self.client_id)
                self._collection_ready = True
            except Exception as qdrant_err:
                logger.warning("Qdrant collection '%s' could not be ensured (non-fatal): %s", self.collection_name, qdrant_err)
                self._collection_ready = False
            if os.path.exists(self.parquet_path):
                try:
                    self.responses_df = pd.read_parquet(self.parquet_path)
                    logger.info(f'Loaded {len(self.responses_df)} reference responses from Parquet: {self.parquet_path}')
                except Exception as e:
                    logger.error(f'Failed reading Parquet at {self.parquet_path}: {e}')
                    self.responses_df = None
            if self.responses_df is None:
                if os.path.exists(self.csv_path):
                    self.responses_df = pd.read_csv(self.csv_path)
                    logger.info(f'Loaded {len(self.responses_df)} reference responses from CSV: {self.csv_path}')
                else:
                    logger.error(f'Responses dataset not found. Looked for Parquet: {self.parquet_path} and CSV: {self.csv_path}')
                    return False
            logger.info('Response matcher initialized successfully')
            return True
        except Exception as e:
            logger.error(f'Error initializing response matcher: {e}')
            return False

    def extract_location(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for loc in self.locations:
            if loc in text_lower:
                return loc
        return None

    def normalize_question(self, text: str) -> str:
        location = self.extract_location(text)
        if location:
            return text.lower().replace(location, 'LOCATION')
        return text.lower()

    def extract_query_pattern(self, text: str) -> Optional[str]:
        t = text.lower()
        patterns = [('top_n', ['top ', 'bottom ', 'highest ', 'lowest ', 'best ', 'worst ', 'rank']), ('trend', ['trend', 'over time', 'month by month', 'year over year', 'quarterly', 'weekly', 'daily', 'historical']), ('compare', ['compar', 'versus', ' vs ', 'difference between', 'contrast']), ('list', ['list ', 'show all', 'give me all', 'what are all', 'enumerate']), ('aggregate', ['total ', 'sum ', 'count ', 'average ', 'avg ', 'how many', 'how much'])]
        for operation_type, keywords in patterns:
            if any((kw in t for kw in keywords)):
                return operation_type
        return None

    def normalize_category(self, text: str) -> str:
        text = text.lower()
        replacements = {'corp id': 'corporate id'}
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def calculate_adjusted_similarity(self, query: str, stored: str, base_similarity: float) -> float:
        query_norm_cat = self.normalize_category(query)
        stored_norm_cat = self.normalize_category(stored)
        if query_norm_cat == stored_norm_cat:
            return 1.0
        query_norm = self.normalize_question(query)
        stored_norm = self.normalize_question(stored)
        query_loc = self.extract_location(query)
        stored_loc = self.extract_location(stored)
        boost = 0.0
        if query_norm == stored_norm:
            return 0.995
        key_phrases = ['group wise', 'category wise', 'trend', 'distribution', 'breakdown', 'summary', 'top 10', 'show me', 'what are', 'what is', 'compare', 'versus', 'vs', 'value of', 'total value', 'count of']
        query_lower = query_norm_cat
        stored_lower = stored_norm_cat
        matching_phrases = sum((1 for phrase in key_phrases if phrase in query_lower and phrase in stored_lower))
        if matching_phrases > 0:
            phrase_boost = min(0.25, matching_phrases * 0.08)
            boost = max(boost, phrase_boost)
        categories = ['project spares', 'workshop spares']
        for category in categories:
            if category in query_lower and category in stored_lower:
                boost = max(boost, 0.15)
        if query_loc and stored_loc:
            if query_loc == stored_loc:
                boost = max(boost, 0.2)
            else:
                boost = max(boost, 0.1)
        if query_lower.startswith('what is') and stored_lower.startswith('what is') or (query_lower.startswith('show me') and stored_lower.startswith('show me')):
            boost = max(boost, 0.12)
        adjusted_sim = min(0.99, base_similarity + boost)
        if base_similarity > 0.85:
            adjusted_sim = min(0.99, adjusted_sim + 0.05)
        return max(base_similarity, adjusted_sim)

    def find_similar_question(self, query_text: str, query_embedding: Optional[List[float]]=None) -> Optional[List[Dict]]:
        if not self._collection_ready:
            if not self.initialize():
                return None
        try:
            from util.qdrant_utils import search_vectors
            if query_embedding is None:
                from util.embedding_utils import generate_embedding
                query_embedding = generate_embedding(query_text)
            query_pattern = self.extract_query_pattern(query_text)
            qdrant_filter = {'operation_type': query_pattern} if query_pattern else None
            if query_pattern:
                logger.debug(f'Applying query_pattern filter: operation_type={query_pattern!r}')
            results = search_vectors(collection_name=self.collection_name, query_vector=query_embedding, limit=3, query_filter=qdrant_filter, client_id=self.client_id)
            if results:
                similar_questions = []
                logger.info(f"\nQuery: '{query_text}'")
                logger.info('Top matches found:')
                for i, result in enumerate(results):
                    base_similarity = result['score']
                    stored_question = result['document']
                    payload = result['payload']
                    adjusted_similarity = self.calculate_adjusted_similarity(query_text, stored_question, base_similarity or 0.0)
                    logger.info(f"Match {i + 1}: '{stored_question}'")
                    logger.info(f'  Base similarity: {base_similarity:.3f}')
                    logger.info(f'  Adjusted similarity: {adjusted_similarity:.3f}')
                    if adjusted_similarity >= self.similarity_threshold:
                        similar_questions.append({'question': stored_question, 'similarity': adjusted_similarity, 'base_similarity': base_similarity, 'distance': 1.0 - base_similarity if base_similarity is not None else None, 'question_id': payload.get('question_id'), 'metadata': payload})
                    else:
                        logger.info(f'  -> Excluded: Below threshold ({self.similarity_threshold})')
                if not similar_questions:
                    logger.info('No matches met the similarity threshold')
                return similar_questions if similar_questions else None
            return None
        except Exception as e:
            logger.error(f'Error finding similar question: {e}')
            return None

    def get_reference_responses(self, question_id: int) -> Optional[Dict]:
        if self.responses_df is None:
            logger.info('responses_df is None — reloading from disk before lookup (question_id=%s).', question_id)
            try:
                if os.path.exists(self.parquet_path):
                    self.responses_df = pd.read_parquet(self.parquet_path)
                elif os.path.exists(self.csv_path):
                    self.responses_df = pd.read_csv(self.csv_path)
                else:
                    logger.error('Responses dataset not found at %s or %s', self.parquet_path, self.csv_path)
                    return None
            except Exception as reload_err:
                logger.error('DataFrame reload failed for question_id=%s: %s', question_id, reload_err)
                return None
        try:
            matching_rows = self.responses_df[self.responses_df['no'] == question_id]
            if matching_rows.empty:
                logger.info(f'question_id {question_id} not in loaded DataFrame — reloading from disk (stale cache).')
                try:
                    if os.path.exists(self.parquet_path):
                        self.responses_df = pd.read_parquet(self.parquet_path)
                    elif os.path.exists(self.csv_path):
                        self.responses_df = pd.read_csv(self.csv_path)
                    matching_rows = self.responses_df[self.responses_df['no'] == question_id]
                except Exception as reload_err:
                    logger.warning(f'DataFrame reload failed: {reload_err}')
            if matching_rows.empty:
                logger.warning(f'No reference responses found for question ID: {question_id}')
                return None
            row = matching_rows.iloc[0]
            result = {'question': row['question'], 'planner_agent_response': row['planner_agent_response'] if pd.notna(row['planner_agent_response']) else '', 'python_agent_response': row['python_agent_response'] if pd.notna(row['python_agent_response']) else '', 'business_agent_response': row['business_agent_response'] if pd.notna(row['business_agent_response']) else ''}
            if 'semantic_signature_json' in row and pd.notna(row['semantic_signature_json']):
                try:
                    sig_data = row['semantic_signature_json']
                    if isinstance(sig_data, str):
                        result['semantic_signature_json'] = json.loads(sig_data)
                    else:
                        result['semantic_signature_json'] = sig_data
                except Exception as e:
                    logger.debug(f'Failed to parse semantic signature: {e}')
            if 'cached_code' in row:
                _cached_code_val = row['cached_code']
                if pd.notna(_cached_code_val):
                    result['cached_code'] = str(_cached_code_val)
                    logger.debug(f"Found cached_code: {result['cached_code'][:100]}...")
                else:
                    result['cached_code'] = ''
            if 'cached_planner_response' in row and pd.notna(row['cached_planner_response']):
                logger.debug(f"Found cached_planner_response: {str(row['cached_planner_response'])[:100]}...")
                try:
                    result['cached_planner_response'] = json.loads(row['cached_planner_response']) if isinstance(row['cached_planner_response'], str) else row['cached_planner_response']
                except (json.JSONDecodeError, TypeError):
                    result['cached_planner_response'] = {'plan': str(row['cached_planner_response']), 'is_follow_up': False, 'tables': []}
            if 'cached_executor_response' in row and pd.notna(row['cached_executor_response']):
                try:
                    result['cached_executor_response'] = json.loads(row['cached_executor_response']) if isinstance(row['cached_executor_response'], str) else row['cached_executor_response']
                except (json.JSONDecodeError, TypeError):
                    result['cached_executor_response'] = {'console_output': str(row['cached_executor_response'])}
            if 'cached_business_response' in row and pd.notna(row['cached_business_response']):
                logger.debug(f"Found cached_business_response: {str(row['cached_business_response'])[:100]}...")
                try:
                    result['cached_business_response'] = json.loads(row['cached_business_response']) if isinstance(row['cached_business_response'], str) else row['cached_business_response']
                except (json.JSONDecodeError, TypeError):
                    result['cached_business_response'] = {'analysis': str(row['cached_business_response'])}
            return result
        except Exception as e:
            logger.error(f'Error retrieving reference responses: {e}')
            return None

    def check_and_get_reference(self, user_question: str, query_embedding: Optional[List[float]]=None) -> Optional[Dict]:
        try:
            if self.responses_df is not None:
                user_location = self.extract_location(user_question)
                user_question_lower = user_question.lower()
                exact_matches = []
                for _, row in self.responses_df.iterrows():
                    stored_question = str(row['question'])
                    stored_location = self.extract_location(stored_question)
                    if user_question_lower == stored_question.lower() and user_location == stored_location:
                        exact_matches.append(row)
                if exact_matches:
                    exact_match = exact_matches[0]
                    exact_match_id = exact_match['no']
                    logger.info(f'\nFound exact match (question_id: {exact_match_id})')
                    logger.info(f"User location: {user_location}, Stored location: {self.extract_location(exact_match['question'])}")
                    reference_responses = self.get_reference_responses(exact_match_id)
                    if reference_responses:
                        return {'user_question': user_question, 'similar_question': exact_match['question'], 'similarity_score': 1.0, 'question_id': exact_match_id, 'reference_responses': reference_responses, 'duplicate_mode': 'return', 'is_exact_match': True}
            similar_matches = self.find_similar_question(user_question, query_embedding=query_embedding)
            if not similar_matches:
                logger.info(f'No similar questions found (similarity < {self.threshold_guide})')
                return None
            similar_matches.sort(key=lambda x: x['similarity'], reverse=True)
            best_match = similar_matches[0]
            logger.info(f'\nProcessing results:')
            logger.info(f"Best match: '{best_match['question']}' with similarity: {best_match['similarity']:.3f}")
            if len(similar_matches) > 1:
                logger.info('Additional matches:')
                for idx, match in enumerate(similar_matches[1:], 2):
                    logger.info(f"  {idx}. '{match['question']}' (similarity: {match['similarity']:.3f})")
            question_id = best_match['question_id']
            reference_responses = self.get_reference_responses(question_id) if question_id is not None else None
            if not reference_responses:
                return None
            sim = float(best_match['similarity'] or 0.0)
            logger.info(f'\nEvaluating thresholds:')
            logger.info(f'Similarity score: {sim:.3f}')
            logger.info(f'Return threshold: {self.threshold_return}')
            logger.info(f'Guide threshold: {self.threshold_guide}')
            user_location = self.extract_location(user_question)
            best_match_location = self.extract_location(best_match['question'])
            logger.info(f'Location comparison:')
            logger.info(f'  User question location: {user_location}')
            logger.info(f'  Best match location: {best_match_location}')
            if sim >= 0.995 and user_location == best_match_location:
                duplicate_mode = 'return'
                logger.info('Mode: RETURN (very high similarity with matching location)')
                if sim == 1.0:
                    logger.info('Exact match detected!')
                else:
                    logger.info(f'Very high similarity match: {sim:.3f}')
            elif sim >= self.threshold_guide:
                duplicate_mode = 'guide'
                logger.info('Mode: GUIDE (showing similar questions)')
                if user_location != best_match_location:
                    logger.info('Note: Locations differ - using as guidance only')
            else:
                duplicate_mode = 'none'
                logger.info('Mode: NONE (no sufficient matches)')
            result = {'user_question': user_question, 'similar_question': best_match['question'], 'similarity_score': best_match['similarity'], 'question_id': question_id, 'reference_responses': reference_responses, 'duplicate_mode': duplicate_mode}
            if duplicate_mode == 'guide':
                result['top_similar_questions'] = [{'question': match['question'], 'similarity': match['similarity']} for match in similar_matches]
            if sim > 0.99:
                cached_result = {}
                if 'cached_planner_response' in reference_responses:
                    cached_result['planner_response'] = reference_responses['cached_planner_response']
                if 'cached_code' in reference_responses:
                    cached_result['code'] = reference_responses['cached_code']
                if 'cached_executor_response' in reference_responses:
                    cached_result['executor_response'] = reference_responses['cached_executor_response']
                if 'cached_business_response' in reference_responses:
                    cached_result['business_response'] = reference_responses['cached_business_response']
                if cached_result:
                    result['cached_result'] = cached_result
                    present_keys = ', '.join(sorted(cached_result.keys()))
                    logger.info(f'High similarity match (>{sim:.1%}) - cached artifacts present: [{present_keys}]')
                else:
                    logger.info(f'High similarity match (>{sim:.1%}) - no cached artifacts present')
            elif duplicate_mode == 'guide' and len(similar_matches) > 1:
                additional_matches = [f"{m['question'][:50]}... ({m['similarity']:.1%})" for m in similar_matches[1:]]
                logger.info(f'Guide mode with {len(similar_matches)} similar questions:')
                for idx, match in enumerate(additional_matches, 2):
                    logger.info(f'  {idx}. {match}')
            return result
        except Exception as e:
            logger.error(f'Error in check_and_get_reference: {e}')
            return None

def format_reference_for_agent(reference_data: Dict, agent_type: str) -> str:
    if not reference_data:
        return ''
    ref_responses = reference_data['reference_responses']
    similar_question = reference_data['similar_question']
    similarity_score = reference_data['similarity_score']
    agent_key_map = {'planner': 'planner_agent_response', 'python': 'python_agent_response', 'business': 'business_agent_response'}
    agent_key = agent_key_map.get(agent_type)
    if not agent_key or not ref_responses.get(agent_key):
        return ''
    reference_response = ref_responses[agent_key]
    top_questions_text = ''
    if reference_data.get('top_similar_questions') and reference_data['duplicate_mode'] == 'guide':
        questions_list = [f"- {q['question']} ({q['similarity']:.1%})" for q in reference_data['top_similar_questions']]
        top_questions_text = '\n**Top Similar Questions Found:**\n' + '\n'.join(questions_list) + '\n'
    match_type = 'exact match' if reference_data.get('is_exact_match') else 'highly similar'
    sim_text = '(exact match)' if reference_data.get('is_exact_match') else f'(similarity: {similarity_score:.1%})'
    reference_text = f"\n## REFERENCE RESPONSE GUIDANCE ({('Use Exactly As-Is' if reference_data.get('is_exact_match') else 'Open-book, adapt minimally')})\n\nA previously asked question is an {match_type} {sim_text}:\n**{('Matching' if reference_data.get('is_exact_match') else 'Similar')} Question:** {similar_question}\n{(top_questions_text if not reference_data.get('is_exact_match') else '')}\nYou must base your response on the following prior correct answer, treating this as an open-book 'cheating' aid to ensure grounding and eliminate hallucinations. Replicate the structure and style; only adjust the parts that differ in the new query.\n\nTypical minimal adaptations include (examples):\n- If segment range changed (e.g., '5 to 10 years' vs '0 to 5 years'), update filters, numbers, and labels accordingly.\n- If facility/location changed (e.g., 'Location A' to 'Location B'), update filters and outputs for the new location while preserving the same analysis format.\n- If grouping or threshold changed, adjust those parameters but do not reinvent the approach.\n\nReference answer to mimic (structure, tone, level of detail):\n\n```\n{reference_response}\n```\n\nCRITICAL INSTRUCTIONS:\n- Generate your answer by following the above reference closely. This is an open-book assistance; do not deviate in structure unless necessary.\n- Only modify values, filters, and names that must change to fit the current question. Do not rewrite the entire response.\n- If any required data appears missing, state the assumption explicitly and proceed, still mirroring the reference format.\n"
    return reference_text.strip()

def get_reference_guidance(user_question: str, agent_type: str) -> str:
    matcher = ResponseMatcher()
    reference_data = matcher.check_and_get_reference(user_question)
    return format_reference_for_agent(reference_data, agent_type)

def test_raw_embedding_similarity(text1: str, text2: str) -> float:
    from util.embedding_utils import generate_embedding
    import numpy as np
    emb1 = generate_embedding(text1)
    emb2 = generate_embedding(text2)
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(similarity)