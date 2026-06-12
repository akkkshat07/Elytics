import os
import sys
import shutil
from dotenv import load_dotenv
load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
DATA_DIR = os.path.join(ASSETS_DIR, 'data')
PROMPTS_DIR = os.path.join(BASE_DIR, 'prompts')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
USE_LANGGRAPH = os.getenv('USE_LANGGRAPH', 'false').lower() == 'true'
USE_TIERED_PROMPTS = os.getenv('USE_TIERED_PROMPTS', 'true').lower() == 'true'
STREAM_CODE_TOKENS = os.getenv('STREAM_CODE_TOKENS', 'true').lower() == 'true'
USE_KNOWLEDGE_FILTERING = os.getenv('USE_KNOWLEDGE_FILTERING', 'false').lower() == 'true'
USE_KNOWLEDGE_SUMMARIZATION = os.getenv('USE_KNOWLEDGE_SUMMARIZATION', 'false').lower() == 'true'
MAX_KNOWLEDGE_TOKENS = int(os.getenv('MAX_KNOWLEDGE_TOKENS', '5000'))
MAX_CODING_KNOWLEDGE_TOKENS = int(os.getenv('MAX_CODING_KNOWLEDGE_TOKENS', '4000'))
MAX_LESSONS_TOKENS = int(os.getenv('MAX_LESSONS_TOKENS', '1500'))
MAX_DATA_PROFILE_TOKENS = int(os.getenv('MAX_DATA_PROFILE_TOKENS', '500'))
MAX_USER_PREFERENCES_TOKENS = int(os.getenv('MAX_USER_PREFERENCES_TOKENS', '300'))
KNOWLEDGE_FILTERING_AB_TEST = {'enabled': os.getenv('KNOWLEDGE_FILTERING_AB_TEST', 'false').lower() == 'true', 'test_clients': [c.strip() for c in os.getenv('KNOWLEDGE_FILTERING_TEST_CLIENTS', '').split(',') if c.strip()], 'test_percentage': float(os.getenv('KNOWLEDGE_FILTERING_TEST_PERCENTAGE', '0'))}

def should_use_knowledge_filtering(client_id: str, session_id: str='') -> bool:
    if not KNOWLEDGE_FILTERING_AB_TEST['enabled']:
        return USE_KNOWLEDGE_FILTERING
    if client_id in KNOWLEDGE_FILTERING_AB_TEST['test_clients']:
        return True
    if KNOWLEDGE_FILTERING_AB_TEST['test_percentage'] > 0 and session_id:
        import hashlib
        hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        return hash_value % 100 < KNOWLEDGE_FILTERING_AB_TEST['test_percentage']
    return USE_KNOWLEDGE_FILTERING
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
LANGSMITH_TRACING = os.getenv('LANGSMITH_TRACING', 'false').lower() == 'true'
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2', 'false').lower() == 'true' or LANGSMITH_TRACING
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
LANGSMITH_PROJECT = os.getenv('LANGSMITH_PROJECT', 'coresight')
LANGSMITH_ENDPOINT = os.getenv('LANGSMITH_ENDPOINT')
if LANGSMITH_TRACING and os.getenv('LANGCHAIN_TRACING_V2') is None:
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
if LANGSMITH_API_KEY and os.getenv('LANGCHAIN_API_KEY') is None:
    os.environ['LANGCHAIN_API_KEY'] = LANGSMITH_API_KEY
if LANGSMITH_PROJECT and os.getenv('LANGCHAIN_PROJECT') is None:
    os.environ['LANGCHAIN_PROJECT'] = LANGSMITH_PROJECT
if LANGSMITH_ENDPOINT and os.getenv('LANGCHAIN_ENDPOINT') is None:
    os.environ['LANGCHAIN_ENDPOINT'] = LANGSMITH_ENDPOINT
MCP_SERVER_PROVIDER = os.getenv('MCP_SERVER_PROVIDER', 'jupyter-mcp-server')
VENV_BIN_PATH = os.path.dirname(sys.executable)
_mcp_candidate = os.path.join(VENV_BIN_PATH, MCP_SERVER_PROVIDER)
MCP_SERVER_COMMAND = _mcp_candidate if os.path.isfile(_mcp_candidate) else shutil.which(MCP_SERVER_PROVIDER) or _mcp_candidate
DATA_SOURCE = {'enabled': True, 'type': 'sap_oracle', 'default_schema': 'SCM_AI', 'sample_limit': 1000, 'connection': {'username': os.getenv('DB_USERNAME'), 'password': os.getenv('DB_PASSWORD'), 'host': os.getenv('DB_HOST'), 'port': os.getenv('DB_PORT', '1521'), 'service_name': os.getenv('DB_SERVICE_NAME')}}
DEFAULT_LLM_PROVIDER = 'gemini'
LLM_PROVIDERS = {'openai': {'api_key_env_var': 'OPENAI_API_KEY', 'default_model': 'gpt-5-chat-latest', 'stt_model': 'whisper-1'}, 'groq': {'api_key_env_var': 'GROQ_API_KEY', 'default_model': 'openai/gpt-oss-120b', 'stt_model': 'whisper-1'}, 'gemini': {'api_key_env_var': 'GEMINI_API_KEY', 'default_model': 'gemini-3.1-pro-preview'}, 'claude': {'api_key_env_var': 'ANTHROPIC_API_KEY', 'default_model': 'claude-sonnet-4-20250514'}}

def gemini_use_vertex_ai() -> bool:
    return os.getenv('GEMINI_USE_VERTEX', '').strip().lower() in ('1', 'true', 'yes')
STT_CONFIG = {'provider': 'openai', 'model_name': LLM_PROVIDERS['openai']['stt_model']}
AGENT_CONFIG = {'explorer_data_desc': {'llm_provider': DEFAULT_LLM_PROVIDER, 'model_name': None, 'temperature': 0.0, 'max_sample_rows': 10, 'max_column_value_length': 200, 'max_total_prompt_length': 30000, 'max_columns_to_show': None, 'max_output_tokens': 2000, 'max_concurrent_llm_calls': 8}, 'explorer_table_intro': {'llm_provider': DEFAULT_LLM_PROVIDER, 'model_name': None, 'temperature': 0.0, 'max_sample_rows': 10, 'max_column_value_length': 200, 'max_total_prompt_length': 20000, 'max_columns_to_show': None, 'max_output_tokens': 2000, 'max_concurrent_llm_calls': 8}, 'explorer_data_profile': {'llm_provider': DEFAULT_LLM_PROVIDER, 'model_name': None, 'temperature': 0.0, 'max_output_tokens': 1500}, 'explorer_agent': {'llm_provider': DEFAULT_LLM_PROVIDER, 'model_name': None, 'reasoning_effort': None, 'temperature': 0.7, 'max_tokens': 2500}, 'executor_agent': {'execution_timeout': int(os.getenv('EXECUTOR_TIMEOUT', '600'))}, 'business_agent': {'prompt_file': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'xml_prompts', 'base', 'agents', 'business.xml'), 'llm_provider': DEFAULT_LLM_PROVIDER, 'model_name': None, 'reasoning_effort': None, 'temperature': 0.0}, 'router_agent': {'prompt_file': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'xml_prompts', 'base', 'agents', 'router.xml'), 'llm_provider': DEFAULT_LLM_PROVIDER, 'model_name': None, 'temperature': 0.0, 'max_tokens': 1000, 'followup_second_pass_similarity_threshold': 0.72}, 'scout_agent': {'temperature': 0.0, 'max_tokens': 500, 'max_concurrent_scouts': 8}, 'data_science_agent': {'prompt_file': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'xml_prompts', 'base', 'agents', 'data_science_agent.xml'), 'llm_provider': DEFAULT_LLM_PROVIDER, 'model_name': None, 'reasoning_effort': None, 'temperature': 0.0, 'max_iterations': 8, 'max_retries_per_iteration': 3, 'retry_temperatures': [0.0, 0.25, 0.5], 'doom_loop_threshold': 3, 'timeout_per_execution': 600, 'idle_timeout_minutes': 30.0, 'context_compaction_interval': 3, 'code_preview_max_chars': 600, 'output_preview_max_chars': 800, 'output_storage_max_chars': 1000, 'string_values_top_n': 5, 'max_journal_detail_entries': 1}, 'data_analyst_agent': {'prompt_file': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'xml_prompts', 'base', 'agents', 'data_analyst_agent.xml'), 'llm_provider': DEFAULT_LLM_PROVIDER, 'model_name': None, 'reasoning_effort': None, 'temperature': 0.0, 'max_iterations': 6, 'max_retries_per_iteration': 3, 'retry_temperatures': [0.0, 0.2, 0.4], 'timeout_per_execution': 300, 'idle_timeout_minutes': 30.0, 'context_compaction_interval': 3, 'code_preview_max_chars': 600, 'output_preview_max_chars': 800, 'output_storage_max_chars': 1000, 'string_values_top_n': 5, 'max_journal_detail_entries': 1, 'doom_loop_threshold': 3, 'always_generate_chart': True, 'always_generate_table': True}}
ENABLE_BACKGROUND_JOBS = os.getenv('ENABLE_BACKGROUND_JOBS', 'true').lower() == 'true'
BACKGROUND_JOB_CONFIG = {'max_concurrent_per_client': int(os.getenv('BG_MAX_CONCURRENT_PER_CLIENT', '2')), 'timeout_seconds': int(os.getenv('BG_JOB_TIMEOUT', '900')), 'stale_threshold_minutes': int(os.getenv('BG_STALE_THRESHOLD_MINUTES', '30')), 'poll_interval_seconds': 15, 'completed_job_ttl_days': int(os.getenv('BG_COMPLETED_TTL_DAYS', '90'))}
ML_KEYWORDS = ['predict', 'forecast', 'train', 'model', 'regression', 'classification', 'cluster', 'neural', 'xgboost', 'random forest', 'time series', 'arima', 'prophet', 'lstm', 'deep learning', 'machine learning', 'gradient boosting', 'decision tree', 'cross validation', 'hyperparameter', 'ensemble', 'svm', 'knn', 'k-means']
KERNEL_BACKEND = os.getenv('KERNEL_BACKEND', 'kubernetes')
STORAGE_BACKEND = os.getenv('STORAGE_BACKEND', 'gcs')
GCS_BUCKET = os.getenv('GCS_BUCKET', 'coresight-data')
GCS_PROJECT_ID = os.getenv('GCS_PROJECT_ID', '')
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'https://storage.googleapis.com')
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY', '')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY', '')
K8S_KERNEL_NAMESPACE = os.getenv('K8S_KERNEL_NAMESPACE', 'coresight-kernels')
K8S_KERNEL_IMAGE = os.getenv('K8S_KERNEL_IMAGE', 'coresight-datascience:latest')
K8S_CONTEXT = os.getenv('K8S_CONTEXT', '')
K8S_DEFAULT_CPU_REQUEST = os.getenv('K8S_DEFAULT_CPU_REQUEST', '500m')
K8S_DEFAULT_CPU_LIMIT = os.getenv('K8S_DEFAULT_CPU_LIMIT', '1000m')
K8S_DEFAULT_MEM_REQUEST = os.getenv('K8S_DEFAULT_MEM_REQUEST', '1Gi')
K8S_DEFAULT_MEM_LIMIT = os.getenv('K8S_DEFAULT_MEM_LIMIT', '2Gi')
K8S_POD_TIMEOUT = int(os.getenv('K8S_POD_TIMEOUT', '120'))
K8S_WARM_POOL_SIZE = int(os.getenv('K8S_WARM_POOL_SIZE', '2'))
K8S_WARM_POOL_TARGET = int(os.getenv('K8S_WARM_POOL_TARGET', str(K8S_WARM_POOL_SIZE * 4)))
PREWARM_POLL_INTERVAL = int(os.getenv('PREWARM_POLL_INTERVAL', '30'))
PREWARM_IDLE_TIMEOUT_MINUTES = float(os.getenv('PREWARM_IDLE_TIMEOUT_MINUTES', '20'))
MAX_DS_CONTAINERS_ENV = 'MAX_DS_CONTAINERS'
USE_REDIS_SEMAPHORE = os.getenv('USE_REDIS_SEMAPHORE', 'true').lower() == 'true'
LOCAL_KERNEL_CONFIG = {'max_concurrent_kernels': int(os.getenv('MAX_LOCAL_KERNELS', '15')), 'max_per_client_kernels': int(os.getenv('MAX_LOCAL_PER_CLIENT', '5')), 'semaphore_acquire_timeout': int(os.getenv('LOCAL_SEMAPHORE_TIMEOUT', '300')), 'jupyter_startup_timeout': int(os.getenv('LOCAL_JUPYTER_TIMEOUT', '30'))}
DATA_SCIENCE_CONTAINER_CONFIG = {'max_concurrent_containers': int(os.getenv('MAX_DS_CONTAINERS', '15')), 'max_per_client_containers': int(os.getenv('MAX_DS_PER_CLIENT', '5')), 'semaphore_acquire_timeout': int(os.getenv('DS_SEMAPHORE_TIMEOUT', '300')), 'mem_limit': os.getenv('DS_CONTAINER_MEM_LIMIT', '1g'), 'cpu_period': int(os.getenv('DS_CONTAINER_CPU_PERIOD', '100000')), 'cpu_quota': int(os.getenv('DS_CONTAINER_CPU_QUOTA', '50000')), 'network_mode': os.getenv('DS_CONTAINER_NETWORK', 'bridge'), 'read_only': False, 'tmpfs': {'/tmp': 'size=256m'}}
ADHOC_FILE_CONFIG = {'max_file_size_mb': int(os.getenv('ADHOC_MAX_FILE_SIZE_MB', '20')), 'allowed_extensions': ['.csv', '.xlsx', '.xls'], 'base_dir': os.path.join(ASSETS_DIR, 'clients'), 'file_ttl_hours': int(os.getenv('ADHOC_FILE_TTL_HOURS', '1'))}
RETRY_CONFIG = {'execution_max_retries': 2, 'validation_max_retries': 2}
VECTOR_DB_CONFIG = {'enabled': True, 'vector_db_provider': 'qdrant', 'qdrant_url': os.getenv('QDRANT_URL', 'http://localhost:6333'), 'qdrant_host': os.getenv('QDRANT_HOST', 'localhost'), 'qdrant_port': int(os.getenv('QDRANT_PORT', '6333')), 'db_path': os.path.join(DATA_DIR, 'vector_db'), 'collection_name': 'client_questions', 'embedding_provider': DEFAULT_LLM_PROVIDER, 'embedding_model': 'gemini-embedding-001' if DEFAULT_LLM_PROVIDER == 'gemini' else 'text-embedding-3-small', 'embedding_dimension': 3072 if DEFAULT_LLM_PROVIDER == 'gemini' else 1536, 'api_key_env_var': 'GEMINI_API_KEY' if DEFAULT_LLM_PROVIDER == 'gemini' else 'OPENAI_API_KEY', 'similarity_threshold': 0.95, 'threshold_return': float(os.getenv('VECTOR_DB_THRESHOLD_RETURN', 0.995)), 'threshold_guide': float(os.getenv('VECTOR_DB_THRESHOLD_GUIDE', 0.75)), 'max_results': 5, 'source_csv': os.path.join(DATA_DIR, 'correct_responses', 'correct_responses.csv'), 'clients_data_dir': os.path.join(DATA_DIR, 'clients')}
MODEL_PRICING = {'gpt-5-chat-latest': {'input_per_1m': 2.0, 'output_per_1m': 8.0}, 'gpt-4.1-mini': {'input_per_1m': 0.4, 'output_per_1m': 1.6}, 'gpt-4o': {'input_per_1m': 2.5, 'output_per_1m': 10.0}, 'gpt-4o-mini': {'input_per_1m': 0.15, 'output_per_1m': 0.6}, 'gpt-4.1-nano': {'input_per_1m': 0.1, 'output_per_1m': 0.4}, 'o3-mini': {'input_per_1m': 1.1, 'output_per_1m': 4.4}, 'gemini-3.1-pro-preview': {'input_per_1m': 1.25, 'output_per_1m': 5.0}, 'gemini-2.5-flash-preview-05-20': {'input_per_1m': 0.15, 'output_per_1m': 0.6}, 'gemini-2.5-flash': {'input_per_1m': 0.15, 'output_per_1m': 0.6}, 'gemini-2.0-flash': {'input_per_1m': 0.1, 'output_per_1m': 0.4}, 'gemini-1.5-flash': {'input_per_1m': 0.075, 'output_per_1m': 0.3}, 'claude-sonnet-4-20250514': {'input_per_1m': 3.0, 'output_per_1m': 15.0}, 'claude-3-5-haiku-latest': {'input_per_1m': 0.8, 'output_per_1m': 4.0}, 'openai/gpt-oss-120b': {'input_per_1m': 0.0, 'output_per_1m': 0.0}, 'text-embedding-3-small': {'input_per_1m': 0.02, 'output_per_1m': 0.0}, 'text-embedding-3-large': {'input_per_1m': 0.13, 'output_per_1m': 0.0}, 'gemini-embedding-001': {'input_per_1m': 0.01, 'output_per_1m': 0.0}, '_default': {'input_per_1m': 1.0, 'output_per_1m': 4.0}}
SUPER_ADMIN_EMAIL = os.getenv('SUPER_ADMIN_EMAIL', 'superadmin@coresight.com')
SUPER_ADMIN_PASSWORD_HASH = os.getenv('SUPER_ADMIN_PASSWORD_HASH', '')