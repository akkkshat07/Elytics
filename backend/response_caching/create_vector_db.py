import os
import argparse
import pandas as pd
import logging
from pathlib import Path
import time
from dotenv import load_dotenv
import hashlib
import json
from response_caching.config_manager import get_client_cache_dir, get_client_db_collection_name, ensure_client_cache_infrastructure
load_dotenv()
logger = logging.getLogger(__name__)

def create_vector_database(client_id: str=None, dataset_id: str=None):
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.system_config import VECTOR_DB_CONFIG
    if client_id:
        ensure_client_cache_infrastructure(client_id, dataset_id)
        CSV_PATH = str(get_client_cache_dir(client_id, dataset_id) / 'correct_responses.csv')
        base_collection = get_client_db_collection_name(client_id)
        safe_ds = ''.join((c if c.isalnum() else '_' for c in dataset_id or ''))
        COLLECTION_NAME = f'{base_collection}_{safe_ds}' if safe_ds else base_collection
        logger.info(f"Multi-tenant mode: Creating VectorDB for client '{client_id}'{(f', dataset {dataset_id}' if dataset_id else '')}")
        logger.info(f'  CSV: {CSV_PATH}')
        logger.info(f'  Collection: {COLLECTION_NAME}')
    else:
        CSV_PATH = VECTOR_DB_CONFIG['source_csv']
        COLLECTION_NAME = VECTOR_DB_CONFIG['collection_name']
        logger.warning('No client_id provided - using global paths (deprecated)')
    try:
        from util.embedding_utils import generate_embedding, get_embedding_dimension
        from util.qdrant_utils import recreate_collection, upsert_points, search_vectors, count_points
        api_key_env = VECTOR_DB_CONFIG.get('api_key_env_var', 'OPENAI_API_KEY')
        api_key = os.getenv(api_key_env)
        if not api_key:
            logger.error(f'{api_key_env} environment variable not set')
            logger.info(f"Please set your API key: export {api_key_env}='your-key-here'")
            return False
        logger.info(f"Embedding provider: {VECTOR_DB_CONFIG.get('embedding_provider', 'openai')}")
        logger.info(f"Embedding model: {VECTOR_DB_CONFIG.get('embedding_model', 'text-embedding-3-small')}")
        logger.info(f'Reading CSV file: {CSV_PATH}')
        df = pd.read_csv(CSV_PATH)
        logger.info(f'Loaded {len(df)} rows from CSV')

        def _normalize_question(text: str) -> str:
            return (str(text) if pd.notna(text) else '').strip().lower()
        df['question_norm'] = df['question'].apply(_normalize_question)
        before_dedupe = len(df)
        df = df.sort_values('no').drop_duplicates(subset=['question_norm'], keep='last')
        after_dedupe = len(df)
        if after_dedupe < before_dedupe:
            logger.info(f'Removed {before_dedupe - after_dedupe} duplicate question rows based on normalized text')
        questions = df['question'].tolist()
        logger.info(f'Prepared {len(questions)} unique questions for embedding')
        dim = get_embedding_dimension()
        logger.info(f'Recreating Qdrant collection: {COLLECTION_NAME} (dim={dim})')
        recreate_collection(COLLECTION_NAME, dim, client_id=client_id)
        logger.info('Generating embeddings...')
        embeddings = []
        for i, question in enumerate(questions):
            try:
                embedding = generate_embedding(question)
                embeddings.append(embedding)
                logger.info(f'Generated embedding {i + 1}/{len(questions)}: {question[:50]}...')
                time.sleep(0.1)
            except Exception as e:
                logger.error(f'Error generating embedding for question {i + 1}: {e}')
                raise

        def _question_id(question_text: str) -> str:
            normalized = (question_text or '').strip().lower()
            digest = hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
            return f'q_{digest}'
        ids = [_question_id(q) for q in questions]
        payloads = []
        for i, row in df.iterrows():
            payload = {'question_id': int(row['no']), 'question': row['question'], 'planner_response': row['planner_agent_response'][:1000] if pd.notna(row['planner_agent_response']) else '', 'python_response': row['python_agent_response'][:1000] if pd.notna(row['python_agent_response']) else '', 'business_response': row['business_agent_response'][:1000] if pd.notna(row['business_agent_response']) else ''}
            if 'cached_code' in row and pd.notna(row['cached_code']):
                payload['cached_code'] = str(row['cached_code'])[:5000]
            if 'semantic_signature_json' in row and pd.notna(row['semantic_signature_json']):
                try:
                    sig_data = row['semantic_signature_json']
                    if isinstance(sig_data, str):
                        sig_dict = json.loads(sig_data)
                    else:
                        sig_dict = sig_data
                    payload['semantic_variant'] = str(sig_dict.get('semantic_variant', ''))
                    payload['operation_type'] = str(sig_dict.get('operation_type', ''))
                    payload['semantic_signature_json'] = json.dumps(sig_dict) if not isinstance(sig_data, str) else sig_data
                    from response_caching.models import SemanticSignature
                    sig = SemanticSignature.from_dict(sig_dict)
                    payload['cache_key'] = sig.to_cache_key()
                except Exception as e:
                    logger.warning(f'Failed to parse semantic signature for row {i}: {e}')
            payloads.append(payload)
        logger.info('Upserting embeddings into Qdrant collection...')
        batch_size = 100
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            upsert_points(collection_name=COLLECTION_NAME, ids=ids[start:end], vectors=embeddings[start:end], payloads=payloads[start:end], documents=questions[start:end], client_id=client_id)
            logger.info(f'Upserted batch {start + 1}-{end}/{len(ids)}')
        logger.info(f'Successfully added {len(questions)} questions to vector database')
        count = count_points(COLLECTION_NAME, client_id=client_id)
        logger.info(f'Collection now contains {count} documents')
        logger.info('Testing similarity search...')
        test_query = 'show top items'
        test_embedding = generate_embedding(test_query)
        results = search_vectors(collection_name=COLLECTION_NAME, query_vector=test_embedding, limit=3, client_id=client_id)
        logger.info(f"Test query: '{test_query}'")
        logger.info('Top 3 similar questions:')
        for i, result in enumerate(results):
            logger.info(f"  {i + 1}. {result['document']} (score: {result['score']:.3f})")
        try:
            parquet_path = Path(CSV_PATH).with_suffix('.parquet')
            df.to_parquet(parquet_path, index=False)
            logger.info(f'Exported correct_responses to Parquet at: {parquet_path}')
        except Exception as pe:
            logger.error(f'Failed to export Parquet: {pe}')
        logger.info('Vector database creation and Parquet export completed successfully!')
        return True
    except Exception as e:
        logger.error(f'Error creating vector database: {e}')
        return False

def query_similar_questions(query_text, n_results=5):
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.system_config import VECTOR_DB_CONFIG
    COLLECTION_NAME = VECTOR_DB_CONFIG['collection_name']
    try:
        from util.embedding_utils import generate_embedding
        from util.qdrant_utils import search_vectors
        query_embedding = generate_embedding(query_text)
        results = search_vectors(collection_name=COLLECTION_NAME, query_vector=query_embedding, limit=n_results)
        return {'query': query_text, 'similar_questions': [r['document'] for r in results], 'metadata': [r['payload'] for r in results], 'scores': [r['score'] for r in results]}
    except Exception as e:
        logger.error(f'Error querying vector database: {e}')
        return None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create VectorDB for response caching', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  # Multi-tenant mode (recommended):\n  python create_vector_db.py --client_id iffco\n  python create_vector_db.py --client_id hcl\n\n  # Per-dataset collection (matches runtime cache isolation):\n  python create_vector_db.py --client_id account-1 --dataset-id <uuid>\n\n  # Legacy mode (deprecated):\n  python create_vector_db.py\n        ')
    parser.add_argument('--client_id', type=str, required=False, help="Client identifier for multi-tenant isolation (e.g., 'iffco', 'hcl')")
    parser.add_argument('--dataset-id', type=str, required=False, default=None, dest='dataset_id', help='Dataset id for per-dataset CSV path and Qdrant collection name (omit for client-wide legacy collection)')
    args = parser.parse_args()
    success = create_vector_database(client_id=args.client_id, dataset_id=args.dataset_id)
    if success:
        logger.info('Vector database created successfully')
    else:
        logger.error('Failed to create vector database')