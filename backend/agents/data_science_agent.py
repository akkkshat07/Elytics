from __future__ import annotations
import asyncio
import difflib
import hashlib
import json
import logging
import os
import re
import signal
import sys
import tempfile
from datetime import datetime
from langsmith import traceable
from util.time_utils import utcnow
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import base64
from io import StringIO
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.system_config import AGENT_CONFIG, LLM_PROVIDERS, DEFAULT_LLM_PROVIDER, MCP_SERVER_COMMAND
from util.llm_utils import LLMClient
from util.xml_prompt_loader import load_xml_prompt_raw, load_client_prompt, load_client_data_descriptions, BASE_PROMPTS_PATH, CLIENTS_PROMPTS_PATH
from util.dataset_paths import resolve_xml_data_sources_dir, assets_datasets_dir, storage_datasets_prefix
from util.kernel_manager import get_kernel_manager, release_kernel_manager, DockerKernelManager, LocalKernelManager
from util.kernel_factory import create_kernel_manager
from util.mcp.client import McpClient, McpError, McpTimeoutError
from util.notebook_builder import NotebookBuilder
from mcp.client.stdio import stdio_client, StdioServerParameters
import traceback
logger = logging.getLogger(__name__)
_LIVE_DB_TYPES = {'postgres', 'mysql', 'mongodb', 'sqlserver', 'sap_oracle', 'sap_hana', 'sap_sybase'}

def _get_duckdb_bootstrap_code(client_id: str, dataset_id: Optional[str]=None) -> str:
    from config.system_config import STORAGE_BACKEND
    if STORAGE_BACKEND != 'gcs':
        return ''
    prefix = f'clients/{client_id}/datasets'
    if dataset_id:
        prefix = f'{prefix}/{dataset_id}'
    return f'''\n# DuckDB GCS bootstrap (auto-injected by CoreSight)\nimport os as _os_\nimport duckdb\n_coresight_conn = duckdb.connect()\n_coresight_conn.execute("LOAD httpfs;")\n_ep_ = _os_.environ.get('S3_ENDPOINT_URL', 'https://storage.googleapis.com').replace('https://', '').replace('http://', '')\n_ak_ = _os_.environ.get('S3_ACCESS_KEY', '')\n_sk_ = _os_.environ.get('S3_SECRET_KEY', '')\n_bucket_ = _os_.environ.get('GCS_BUCKET', '')\nif not _ak_ or not _sk_:\n    raise RuntimeError("GCS HMAC credentials (S3_ACCESS_KEY / S3_SECRET_KEY) not set in kernel environment.")\n_coresight_conn.execute(f"SET s3_endpoint='{{_ep_}}';")\n_coresight_conn.execute(f"SET s3_access_key_id='{{_ak_}}';")\n_coresight_conn.execute(f"SET s3_secret_access_key='{{_sk_}}';")\n_coresight_conn.execute("SET s3_url_style='path';")\n_coresight_conn.execute("SET s3_use_ssl=true;")\n_prefix_ = '{prefix}'\n\ndef query_parquet(file_pattern: str, sql_query: str = None) -> 'pd.DataFrame':\n    """Query parquet files from GCS using DuckDB.\n\n    Args:\n        file_pattern: File name or glob (e.g., 'sales.parquet' or '*.parquet')\n        sql_query: Optional SQL query. Use {{TABLE}} as placeholder for the\n                   parquet path. If None, selects all rows.\n    """\n    gcs_path = f"s3://{{_bucket_}}/{{_prefix_}}/{{file_pattern}}"\n    if sql_query:\n        _rp_ = f"read_parquet('{{gcs_path}}')"\n        _sql_ = sql_query\n        # Handle both {{{{TABLE}}}} (double-brace, legacy) and {{TABLE}} (single-brace, from LLM f-strings)\n        # Check double-brace FIRST because single-brace is a substring of double-brace\n        if '{{{{TABLE}}}}' in _sql_:\n            _sql_ = _sql_.replace('{{{{TABLE}}}}', _rp_)\n        elif '{{TABLE}}' in _sql_:\n            _sql_ = _sql_.replace('{{TABLE}}', _rp_)\n        return _coresight_conn.execute(_sql_).fetchdf()\n    return _coresight_conn.execute(f"SELECT * FROM read_parquet('{{gcs_path}}')").fetchdf()\n'''

def _check_jupyter_mcp_timeout_patch():
    try:
        from patch_jupyter_mcp import find_server_py, check_patch
        if not check_patch(find_server_py()):
            logger.warning('jupyter_mcp_server execute_code timeout is capped at 60s. ML/forecasting code will be killed early. Fix: conda activate coresight && python patch_jupyter_mcp.py')
    except Exception:
        pass
_check_jupyter_mcp_timeout_patch()

class DataScienceAgent:

    def __init__(self, agent_name: str='data_science_agent', provided_config: Optional[Dict]=None, client_id: str=None, db: Any=None, notebook_output_dir: str='test_outputs', llm_client: Optional[LLMClient]=None, resolved_prompt: Optional[str]=None, dataset_id: Optional[str]=None, session_id: Optional[str]=None):
        if not client_id:
            raise ValueError('client_id is REQUIRED for multi-tenant operation. No default client exists. Every request must specify a valid client_id.')
        self.agent_name = agent_name
        self.client_id = client_id
        self.session_id = session_id or ''
        self.dataset_id = dataset_id
        self._session_owned: bool = False
        self.db = db
        self.config = provided_config or AGENT_CONFIG.get(self.agent_name, {})
        if llm_client is None:
            raise ValueError(f'llm_client is REQUIRED for {self.agent_name}. When using agents in the graph, pass the shared LLMClient from state.')
        self.llm_client = llm_client
        self.base_prompt = self._load_system_prompt(resolved_prompt=resolved_prompt)
        self.kernel_manager: Optional[LocalKernelManager] = None
        self.mcp_client: Optional[McpClient] = None
        self._stdio_context_manager = None
        self._mcp_context_manager = None
        self._stdio_process: Any = None
        self._stdio_process_pid: Optional[int] = None
        self.execution_history: List[Dict] = []
        self.variables_state: Dict[str, Any] = {}
        self._data_dir: str = '/data'
        self._planned_tables: List[str] = []
        self._is_live_db: bool = False
        self.db_credentials_env: Dict[str, str] = {}
        self._dataset_volume_mounted: bool = False
        self.notebook_output_dir = notebook_output_dir
        self.notebook_builder: Optional[NotebookBuilder] = None
        self.usage_stats = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'models': set()}
        self.llm_provider = self.config.get('llm_provider', DEFAULT_LLM_PROVIDER)
        self.max_iterations = self.config.get('max_iterations', 15)
        self.temperature = self.config.get('temperature', 0.0)
        self.reasoning_effort = self.config.get('reasoning_effort', None)
        self.timeout_per_execution = self.config.get('timeout_per_execution', 180)
        self.idle_timeout_minutes = self.config.get('idle_timeout_minutes', 30.0)
        self.max_result_rows = int(self.config.get('max_result_rows', 500))
        self.max_retries_per_iteration = self.config.get('max_retries_per_iteration', 3)
        self.retry_temperatures = self.config.get('retry_temperatures', [0.0, 0.25, 0.5])
        self.context_compaction_interval = self.config.get('context_compaction_interval', 3)
        self.code_preview_max_chars = self.config.get('code_preview_max_chars', 1200)
        self.output_preview_max_chars = self.config.get('output_preview_max_chars', 1500)
        self.output_storage_max_chars = self.config.get('output_storage_max_chars', 1000)
        self.string_values_top_n = self.config.get('string_values_top_n', 5)
        self.max_journal_detail_entries = self.config.get('max_journal_detail_entries', 1)
        self.doom_loop_threshold: int = self.config.get('doom_loop_threshold', 3)
        self._recent_failed_codes: List[str] = []
        self._static_system_context: Optional[str] = None
        self._cached_lessons_text: Optional[str] = None
        self._cached_prefs_text: Optional[str] = None
        self._artifact_registry: List[Dict[str, Any]] = []
        self._profiled_datasets: set = set()
        self._previous_profile_shapes: Dict[str, Any] = {}
        configured_model = self.config.get('model_name')
        if configured_model:
            self.model = configured_model
        else:
            provider_cfg = LLM_PROVIDERS.get(self.llm_provider, {})
            self.model = provider_cfg.get('default_model', 'gpt-4')
        logger.info(f"DataScienceAgent initialized for client '{client_id}' | provider={self.llm_provider}, model={self.model}, max_iterations={self.max_iterations}, temperature={self.temperature}, timeout={self.timeout_per_execution}s")

    @property
    def _is_gcs(self) -> bool:
        from config.system_config import STORAGE_BACKEND
        return STORAGE_BACKEND == 'gcs'

    def _read_parquet_instruction(self, path: str, columns: Optional[List[str]]=None) -> str:
        if self._is_gcs:
            fname = path.rsplit('/', 1)[-1] if '/' in path else path
            if columns:
                col_str = ', '.join((f"'{c}'" for c in columns))
                return f"""query_parquet('{fname}', "SELECT {col_str} FROM {{TABLE}}")"""
            return f"query_parquet('{fname}')"
        else:
            if columns:
                return f"pd.read_parquet(r'{path}', columns={columns})"
            return f"pd.read_parquet(r'{path}')"

    def _loaded_datasets_prompt(self, loaded_datasets: List[Dict]) -> List[str]:
        lines: List[str] = []
        if not loaded_datasets:
            return lines
        is_gcs = any((ds.get('gcs') for ds in loaded_datasets))
        if is_gcs:
            lines.append('LOADED DATASETS (stored in GCS — use query_parquet() to load):')
            for ds in loaded_datasets:
                lines.append(f"  - '{ds['path']}' → load as: {ds['variable']} = query_parquet('{ds['path']}')")
            lines.append('')
            lines.append('⚠️ CRITICAL: Files are in cloud storage, NOT on the local filesystem.')
            lines.append('DO NOT use pd.read_parquet() or pd.read_csv() — they will fail with FileNotFoundError.')
            lines.append("ALWAYS use query_parquet('filename.parquet') which reads from GCS via DuckDB.")
            lines.append("Example: df = query_parquet('bom.parquet')")
            lines.append('For SQL: df = query_parquet(\'bom.parquet\', "SELECT col1, col2 FROM {TABLE} WHERE col1 > 10")')
            lines.append('For column selection: df = query_parquet(\'bom.parquet\', "SELECT col1, col2 FROM {TABLE}")')
        else:
            lines.append("LOADED DATASETS (⚠️ CRITICAL: when calling pd.read_parquet/read_csv, use the EXACT absolute path below — NEVER a bare filename like 'data.parquet'):")
            for ds in loaded_datasets:
                lines.append(f"  - path='{ds.get('path', '?')}' variable={ds.get('variable', '?')} format={ds.get('format', '?')}")
        lines.append('')
        return lines

    def _performance_rules_prompt(self) -> List[str]:
        if self._is_gcs:
            return ['PERFORMANCE RULES:', '- For LARGE tables (>100K rows): select needed columns via SQL: query_parquet(\'file.parquet\', "SELECT col1, col2 FROM {TABLE}")', "- For SMALL lookup/dimension tables (<10K rows): load ALL columns: query_parquet('file.parquet')", "- When in doubt about which columns you'll need, load MORE columns — reloading wastes iterations", '- Do NOT add .sample() — always process the FULL dataset for accurate results', '- Only sample if the user EXPLICITLY asks for it in their query', '- Use vectorized operations, not loops.']
        else:
            return ['PERFORMANCE RULES:', "- For LARGE tables (>100K rows): select needed columns: pd.read_parquet(path, columns=['col1','col2'])", '- For SMALL lookup/dimension tables (<10K rows): load ALL columns to avoid repeated KeyErrors', "- When in doubt about which columns you'll need, load MORE columns — reloading wastes iterations", '- Do NOT add .sample() — always process the FULL dataset for accurate results', '- Only sample if the user EXPLICITLY asks for it in their query', '- Use vectorized operations, not loops.']

    def _matches_planned_tables(self, filename: str) -> bool:
        if not self._planned_tables:
            return True
        fname_upper = filename.upper()
        return any((table.upper() in fname_upper for table in self._planned_tables))

    def _get_raw_db(self):
        if self.db is None:
            return None
        return getattr(self.db, 'db', self.db) if type(self.db).__name__ == 'MongoDBManager' else self.db

    def _load_knowledge_for_coding(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {'table_introductions_xml': '', 'data_descriptions': {}, 'domain_terminology': '', 'client_data_profile': ''}
        try:
            client_base = CLIENTS_PROMPTS_PATH / self.client_id
            ds_root = resolve_xml_data_sources_dir(self.client_id, self.dataset_id)
            intro_path = ds_root / 'meta_information' / 'table_introductions.xml'
            if not intro_path.exists():
                intro_path = Path(BASE_PROMPTS_PATH) / 'data_sources' / 'meta_information' / 'table_introductions.xml'
            if intro_path.exists():
                result['table_introductions_xml'] = load_xml_prompt_raw(intro_path)
            desc_dir = ds_root / 'data_descriptions'
            if desc_dir.exists():
                result['data_descriptions'] = load_client_data_descriptions(client_id=self.client_id, dataset_id=self.dataset_id)
            term_path = client_base / 'domain_knowledge' / 'terminology.xml'
            if not term_path.exists():
                term_path = Path(BASE_PROMPTS_PATH) / 'domain_knowledge' / 'terminology.xml'
            if term_path.exists():
                result['domain_terminology'] = load_xml_prompt_raw(term_path)
            profile_path = ds_root / 'meta_information' / 'client_data_profile.xml'
            if profile_path.exists():
                result['client_data_profile'] = load_xml_prompt_raw(profile_path)
            logger.info('Knowledge for coding | client=%s | intros=%d chars, descs=%d tables, terms=%d chars, profile=%d chars', self.client_id, len(result['table_introductions_xml']), len(result['data_descriptions']), len(result['domain_terminology']), len(result.get('client_data_profile', '')))
        except Exception as e:
            logger.warning('Failed to load knowledge for coding: %s', e)
        return result

    async def _build_static_system_context(self, user_query: str, plan_guidance: str, execution_context: Dict[str, Any]) -> str:
        parts: List[str] = []
        parts.append(self.base_prompt or '')
        parts.append('\n\nYou are a recursive data science agent. You observe outputs, decide the next step, and iterate until the analysis is complete. You MUST respond in the exact plain-text RESPONSE FORMAT described below — no JSON, no markdown fences, no extra commentary.')
        parts.append('')
        live_sql_mode = bool(execution_context.get('live_sql_mode'))
        if live_sql_mode:
            db_type = (execution_context.get('db_type', '') or '').strip().lower()
            logger.info('LIVE SQL prompt: db_type=%s (postgres_quoting=%s)', db_type, db_type in ('postgres', 'postgresql'))
            parts.append('LIVE SQL MODE:')
            parts.append(f"- Database type: {db_type or 'unknown'}. A connection object `conn` is available.")
            parts.append('- Query data with: df = pd.read_sql("SELECT ... FROM table", conn)')
            if db_type in ('postgres', 'postgresql'):
                parts.extend(['- CRITICAL: PostgreSQL requires double-quoting ALL column names in SQL.', '  WRONG:   SELECT Order_Date FROM dsr_master  ← will fail', '  CORRECT: SELECT "Order_Date" FROM dsr_master  ← must quote', '  Example: df = pd.read_sql(\'SELECT "Order_Date", "P3_NSV" FROM dsr_master WHERE "BusinessMonth" = \\\'2025-05\\\'\', conn)'])
            else:
                parts.append('  Example: df = pd.read_sql("SELECT * FROM table_name LIMIT 100", conn)')
            parts.extend(['- Each table has DIFFERENT columns. Only use columns listed for that table in the Analyst Guidance.', '- NEVER use pd.read_parquet(), pd.read_csv(), file paths, glob, or local files.', ''])
        file_schemas = execution_context.get('file_schemas', {})
        if file_schemas:
            parts.append('FILE SCHEMAS (EXACT column names — use ONLY these, case-sensitive):')
            for fname, schema in file_schemas.items():
                rows_info = f", {schema['num_rows']:,} rows" if schema.get('num_rows') else ''
                parts.append(f"  {fname} ({schema.get('path', '')}{rows_info})")
                parts.append(f"    columns = {schema.get('columns', [])}")
                if schema.get('types'):
                    parts.append(f"    types   = {schema.get('types', {})}")
            parts.append('')
            parts.append('⚠️ CRITICAL: The plan guidance may use WRONG column names (e.g. LAST_ISSUE_DATE). ALWAYS use the EXACT column names from FILE SCHEMAS above instead. ' + ('When using query_parquet(), select ONLY columns from the schema.' if self._is_gcs else 'When using pd.read_parquet(columns=[...]), use ONLY names from the schema.'))
            parts.append('')
        knowledge_ctx = execution_context.get('knowledge_context', {})
        if knowledge_ctx and (file_schemas or live_sql_mode):
            from util.knowledge_filter import compress_table_introductions_for_coding, compress_data_descriptions_for_coding, compress_terminology_for_coding, _approx_token_count
            from config.system_config import MAX_CODING_KNOWLEDGE_TOKENS
            schema_tables = [Path(f).stem for f in file_schemas.keys()] if file_schemas else list(getattr(self, '_planned_tables', []) or [])
            knowledge_lines: list = []
            budget = MAX_CODING_KNOWLEDGE_TOKENS
            intros = compress_table_introductions_for_coding(knowledge_ctx.get('table_introductions_xml', ''), schema_tables)
            if intros:
                cost = _approx_token_count(intros)
                if cost <= budget:
                    knowledge_lines.append('TABLE DESCRIPTIONS:')
                    knowledge_lines.append(intros)
                    budget -= cost
            descs = compress_data_descriptions_for_coding(knowledge_ctx.get('data_descriptions', {}), schema_tables)
            if descs:
                cost = _approx_token_count(descs)
                if cost <= budget:
                    knowledge_lines.append('')
                    knowledge_lines.append('COLUMN DESCRIPTIONS (use to select correct columns):')
                    knowledge_lines.append(descs)
                    budget -= cost
            terms = compress_terminology_for_coding(knowledge_ctx.get('domain_terminology', ''))
            if terms:
                cost = _approx_token_count(terms)
                if cost <= budget:
                    knowledge_lines.append('')
                    knowledge_lines.append('DOMAIN TERMINOLOGY:')
                    knowledge_lines.append(terms)
            if knowledge_lines:
                parts.append('BUSINESS KNOWLEDGE (understand what the data means):')
                parts.extend(knowledge_lines)
                parts.append('')
        client_profile = knowledge_ctx.get('client_data_profile', '')
        if client_profile:
            from config.system_config import MAX_DATA_PROFILE_TOKENS
            profile_cost = len(client_profile) // 4
            if profile_cost <= MAX_DATA_PROFILE_TOKENS:
                parts.append('CLIENT DATA PROFILE (formatting & locale guidance):')
                parts.append(client_profile)
                parts.append('')
        try:
            raw_db = self._get_raw_db()
            if raw_db and (not self._cached_lessons_text):
                from services.agent_lesson_service import AgentLessonService
                from config.system_config import MAX_LESSONS_TOKENS
                lesson_svc = AgentLessonService(raw_db)
                planned_tables = getattr(self, '_planned_tables', None)
                schema_tables = [Path(f).stem for f in file_schemas.keys()] if file_schemas else None
                filter_tables = planned_tables or schema_tables
                self._cached_lessons_text = await lesson_svc.format_lessons_for_prompt(self.client_id, tables=filter_tables, max_tokens=MAX_LESSONS_TOKENS)
        except Exception as le:
            logger.debug('Lesson injection skipped: %s', le)
        if self._cached_lessons_text:
            parts.append('LEARNED PATTERNS (from prior analyses — follow these strictly):')
            parts.append(self._cached_lessons_text)
            parts.append('')
        try:
            raw_db = self._get_raw_db()
            user_id = getattr(self, '_user_id', None)
            if raw_db and user_id and (not self._cached_prefs_text):
                from services.user_preference_service import UserPreferenceService
                from services.preference_extractor import PreferenceExtractor
                from config.system_config import MAX_USER_PREFERENCES_TOKENS
                pref_svc = UserPreferenceService(raw_db)
                current_prefs = PreferenceExtractor.extract_as_dict(user_query) if user_query else {}
                self._cached_prefs_text = await pref_svc.format_for_prompt(self.client_id, user_id, current_query_prefs=current_prefs, max_tokens=MAX_USER_PREFERENCES_TOKENS)
        except Exception:
            pass
        if self._cached_prefs_text:
            parts.append('USER PREFERENCES (respect these for visualization and formatting):')
            parts.append(self._cached_prefs_text)
            parts.append('')
        data_profile = execution_context.get('data_profile', {})
        if file_schemas and len(file_schemas) > 1:
            parts.append('MULTI-TABLE JOIN CONTEXT:')
            col_to_files: Dict[str, list] = {}
            file_row_counts: Dict[str, int] = {}
            for fname, schema in file_schemas.items():
                for col in schema.get('columns', []):
                    col_to_files.setdefault(col, []).append(fname)
                if schema.get('num_rows') is not None:
                    file_row_counts[fname] = schema['num_rows']
                stem = Path(fname).stem
                stem_lower = stem.lower().replace('-', '_')
                for ds_name, prof in data_profile.items():
                    ds_lower = ds_name.lower().replace('-', '_')
                    if ds_lower == stem_lower or ds_lower.endswith(stem_lower) or stem_lower.endswith(ds_lower) or (stem_lower in ds_lower) or (ds_lower in stem_lower):
                        shape = prof.get('shape', [0])
                        file_row_counts[fname] = shape[0] if shape else 0
            shared_cols = {col: files for col, files in col_to_files.items() if len(files) > 1}
            if shared_cols:
                parts.append('  Shared columns (potential join keys):')
                for col, files in shared_cols.items():
                    parts.append(f"    {col}: appears in {', '.join(files)}")

            def _cols_near_match(a: str, b: str) -> bool:
                if a == b:
                    return False
                if a in b or b in a:
                    return True
                for suffix in ('_ID', '_NAME', '_CODE', '_KEY', '_NUM'):
                    if a.endswith(suffix) and b.endswith(suffix):
                        base_a = a[:-len(suffix)].rstrip('_')
                        base_b = b[:-len(suffix)].rstrip('_')
                        if base_a and base_b and (base_a in base_b or base_b in base_a):
                            return True
                return False
            all_cols_by_file = {fname: set(s.get('columns', [])) for fname, s in file_schemas.items()}
            near_matches = []
            fnames_list = list(all_cols_by_file.keys())
            for i in range(len(fnames_list)):
                for j in range(i + 1, len(fnames_list)):
                    for col_a in all_cols_by_file[fnames_list[i]]:
                        for col_b in all_cols_by_file[fnames_list[j]]:
                            if _cols_near_match(col_a, col_b):
                                near_matches.append((col_a, fnames_list[i], col_b, fnames_list[j]))
            if near_matches:
                parts.append('  Near-match columns (VERIFY overlap before joining — names differ):')
                for col_a, f_a, col_b, f_b in near_matches[:10]:
                    parts.append(f'    {col_a} ({f_a}) ↔ {col_b} ({f_b})')
            for fname, rows in file_row_counts.items():
                if rows < 10000:
                    load_hint = f"Load ALL columns: query_parquet('{fname}')" if self._is_gcs else f'Load ALL columns: pd.read_parquet(path) with NO columns= parameter.'
                    parts.append(f"  ⚠️ CRITICAL: {fname} is a small table ({rows} rows) — IGNORE the plan's column selection for this file. {load_hint}")
            if not any((rows < 10000 for rows in file_row_counts.values())):
                for fname, schema in file_schemas.items():
                    n_cols = len(schema.get('columns', []))
                    if n_cols <= 6 and fname not in file_row_counts:
                        parts.append(f'  ℹ️ {fname} has only {n_cols} columns — likely a small lookup table. Load ALL columns to avoid needing to reload.')
            parts.append("\n  ⚠️ MULTI-TABLE JOIN RULE: Before joining two tables, you MUST:\n    1. For small lookup/dimension tables: load ALL columns (do NOT use columns= parameter)\n    2. BEFORE joining, verify join key overlap:\n       overlap = set(df_a['col_a'].unique()) & set(df_b['col_b'].unique())\n       print(f'Overlap: {len(overlap)} common values')\n    3. If overlap is 0, try OTHER candidate columns — check near-matches above\n    4. Column names may differ (e.g., ORGANIZATION_ID ↔ INV_ORG_ID) — check VALUES, not just names\n    5. Use the column pair with the HIGHEST overlap for the join\n    6. FILTER-BY-VALUE (same rules apply): When using a value from table A to filter table B (e.g., .isin(), == comparison):\n       - Verify the looked-up value EXISTS in the target column BEFORE filtering\n       - If 0 rows result, the value maps to a DIFFERENT column in table B\n       - Print unique values in candidate columns to find the correct mapping")
            parts.append('')
        loaded_datasets = execution_context.get('loaded_datasets', [])
        if loaded_datasets:
            parts.extend(self._loaded_datasets_prompt(loaded_datasets))
        performance_rules = ["- Push filters and aggregations into SQL; don't load full tables.", '- Use pd.read_sql(sql, conn) for every query; never return raw SQL only.', '- Do NOT use parquet/CSV/file operations in live SQL mode.', '- Use vectorized pandas operations on returned DataFrames.'] if live_sql_mode else ["- For LARGE tables (>100K rows): select needed columns: pd.read_parquet(path, columns=['col1','col2'])", '- For SMALL lookup/dimension tables (<10K rows): load ALL columns to avoid repeated KeyErrors', "- When in doubt about which columns you'll need, load MORE columns — reloading wastes iterations", '- Do NOT add .sample() — always process the FULL dataset for accurate results', '- Only sample if the user EXPLICITLY asks for it in their query', '- Use vectorized operations, not loops.']
        parts.extend(['RULES:', "- ITERATION 1 PLANNING: In your very first execution step (Iteration 1), your `code` MUST ONLY initialize three Python objects: `TASKS = [...]`, `COMPLETED_TASKS = []`, and `_VAR_INTENT_ = {}`. Break down the user query into a detailed, tech-heavy `TASKS` list based on the PROVIDED ANALYST GUIDANCE. Every task MUST include the table names and specific columns/filters mentioned in the guidance (e.g., `[Step 1] Fetch TotalValue from table_name where OrderDate >= '2025-01-01'`). Do NOT include SQL, data loading, or analysis logic in Iteration 1 — planning only. Begin task execution from Iteration 2. IMPORTANT: The column names and data types are ALREADY provided in the Analyst Guidance above. Do NOT waste iterations inspecting schema, verifying columns, or probing data types — trust the provided schema and start querying data directly in Iteration 2.", "- INTENT REGISTRY: Starting Iteration 2, for every meaningful DataFrame/variable you create, add a 1-sentence entry to `_VAR_INTENT_` (e.g., `_VAR_INTENT_['df_sales'] = 'Filtered 2025 sales by region'`).", '- STATE PROGRESSION: After every iteration that completes a task, append the completed task string to `COMPLETED_TASKS` (e.g., `COMPLETED_TASKS.append(TASKS[0])`).', '- COMPLETION GATE: Do NOT set FINAL_RESULT until every entry in `TASKS` is also in `COMPLETED_TASKS`. Premature completion ruins accuracy.', '- Each iteration builds on previous ones. Variables from prior iterations are alive in memory.', '- Do NOT re-import libraries or re-load data that was already loaded.', '- Do NOT repeat code from prior iterations — reference existing variables.', '- Add print() statements to show intermediate and final results.', '- If this is the LAST step of the analysis, store your primary result in FINAL_RESULT.', '  Example: FINAL_RESULT = result_df  or  FINAL_RESULT = {"key": value}', "- Set FINAL_RESULT BEFORE declaring action='done'.", '', 'WORKFLOW GUIDANCE (adapt based on query complexity):', '- For SIMPLE queries (1-2 tables, clear columns): load → compute → FINAL_RESULT in 3-4 iterations.', '- For COMPLEX queries (3+ tables, joins needed): load → merge → compute → FINAL_RESULT in 5-6 iterations.', '- You MAY combine related operations in one cell (e.g., load + inspect, or aggregate + visualize).', "- Before any groupby/filter, quickly check the relevant column: print(df['col'].nunique()) or print(df['col'].unique()[:10])", "- If a merge/join produces 0 rows, investigate immediately — don't proceed with empty data.", '- MERGE VERIFICATION: After any pd.merge/join, immediately print:', '  1. Result shape vs input shapes (row explosion = wrong keys)', "  2. Check for '_x'/'_y' column suffixes (= overlapping non-key columns, likely wrong join keys)", '  3. Sample 2-3 rows to sanity-check the joined data', '', 'CODE QUALITY RULES:', '- Keep cells to 30-40 lines MAX.', "- NEVER re-import libraries or re-load data that's already in AVAILABLE VARIABLES.", "- NEVER reference variables that don't exist — check AVAILABLE VARIABLES above.", '- Every cell MUST end with print() showing what was produced.', "- If a previous cell produced a DataFrame, USE it — don't rebuild it.", '', *self._performance_rules_prompt(), '', 'FILTER VERIFICATION RULES:', "- After EVERY filter (.isin(), .query(), boolean indexing, pd.merge), IMMEDIATELY check: print(f'Filtered: {result.shape[0]} rows')", '- If 0 rows: DO NOT proceed. Instead:', '  1. Print unique values in BOTH source and target filter columns', '  2. Check if you used the wrong column (e.g., ORG_ID vs INV_ORG_ID)', '  3. Try alternative columns ending in _ID, _CODE, _KEY, _NUM', '- When using a value from table A to filter table B:', "  1. Print the lookup value: print(f'Lookup: {value}')", '  2. Verify the value EXISTS in the target column before filtering', '  3. Column names often DIFFER between tables for the same entity', '  4. Values may also differ: ORG_ID=81 does NOT mean ORGANIZATION_ID=81', '- NEVER silently accept 0 rows and proceed to the next step', '', 'RESPONSE FORMAT — output these lines in this EXACT order, plain text, no JSON, no markdown fences:', "ACTION: code        (use 'done' instead when the analysis is finished)", "REASONING: ≤6 words, creative step title (e.g. 'Pulling in the RFQ data', 'Linking RFQs to their organization', 'Spotting the top revenue drivers', 'Assembling the final picture'). No column/file/table names. No generic labels like Loading datasets or Merging datasets.", 'THINKING: 1-2 sentences — What data do I have, what do I still need, and what will this cell do?', 'CODE:', "<python code to execute on the lines that follow; write nothing after the code. Omit the CODE: line and everything after it when ACTION is 'done'>", '', 'RULES FOR THE RESPONSE FORMAT:', '- ACTION, REASONING and THINKING are each a SINGLE line.', "- Everything after the line 'CODE:' is taken verbatim as the Python cell — do NOT wrap it in quotes, JSON, or markdown fences."])
        return '\n'.join(parts)

    def _build_journal_entry(self, iteration: int, reasoning: str, new_vars: Dict[str, Any], prev_vars: Dict[str, Any]) -> str:
        created = []
        for name, info in new_vars.items():
            if name in prev_vars:
                continue
            if isinstance(info, dict) and info.get('type') == 'DataFrame':
                shape = info.get('shape', [])
                shape_str = f'{shape[0]}×{shape[1]}' if len(shape) >= 2 else str(shape)
                created.append(f'{name} (DataFrame, {shape_str})')
            elif isinstance(info, dict):
                created.append(f"{name} ({info.get('type', '?')})")
            else:
                created.append(f'{name} ({info})')
        vars_part = ', '.join(created[:5]) if created else 'no new variables'
        return f'Step {iteration}: {reasoning} → {vars_part}'

    def _compute_profile_delta(self, current_profile: Dict[str, Any]) -> Dict[str, Any]:
        delta: Dict[str, Any] = {}
        current_keys = set(current_profile.keys())
        top_n = self.string_values_top_n
        for key in current_keys - self._profiled_datasets:
            prof = {}
            src = current_profile[key]
            for field in ('shape', 'columns', 'dtypes', 'null_counts', 'sample_row'):
                if src.get(field):
                    prof[field] = src[field]
            if src.get('string_values'):
                capped = {}
                for col_name, sv in src['string_values'].items():
                    capped[col_name] = {'unique_count': sv.get('unique_count', '?'), 'top_values': sv.get('top_values', [])[:top_n]}
                prof['string_values'] = capped
            delta[key] = {'status': 'new', 'profile': prof}
        for key in current_keys & self._profiled_datasets:
            prev_shape = self._previous_profile_shapes.get(key)
            curr_shape = current_profile[key].get('shape')
            if prev_shape != curr_shape:
                delta[key] = {'status': 'changed', 'shape_before': prev_shape, 'shape_after': curr_shape, 'columns': current_profile[key].get('columns')}
        self._profiled_datasets = current_keys.copy()
        self._previous_profile_shapes = {k: v.get('shape') for k, v in current_profile.items()}
        return delta

    def _register_artifact(self, iteration: int, reasoning: str, new_vars: Dict[str, Any], prev_vars: Dict[str, Any]) -> None:
        new_variables = {}
        modified_variables = {}
        for name, info in new_vars.items():
            if name not in prev_vars:
                new_variables[name] = info if isinstance(info, dict) else {'type': str(info)}
            elif isinstance(info, dict) and isinstance(prev_vars.get(name), dict):
                prev_shape = prev_vars[name].get('shape')
                curr_shape = info.get('shape')
                if prev_shape != curr_shape:
                    modified_variables[name] = {'type': info.get('type', '?'), 'shape_before': prev_shape, 'shape_after': curr_shape}
        self._artifact_registry.append({'iteration': iteration, 'reasoning': reasoning, 'new_variables': new_variables, 'modified_variables': modified_variables})

    def _build_dynamic_user_message(self, user_query: str, plan_guidance: str, execution_context: Dict[str, Any], iteration: int) -> str:
        parts: List[str] = []
        if execution_context.get('adhoc_mode'):
            parts.append('IMPORTANT: You are analyzing a USER-UPLOADED ad-hoc file. Focus on exploratory data analysis. Do NOT reference any other tables or data sources — only use the uploaded file(s). The file schema and sample data are available from the kernel.')
            parts.append('')
        parts.append(f'USER QUERY: {user_query}')
        parts.append('')
        parts.append('PLAN GUIDANCE (use as direction, not a rigid checklist):')
        parts.append(plan_guidance)
        parts.append('')
        journal = execution_context.get('execution_journal', [])
        if journal:
            parts.append('EXECUTION JOURNAL (DO NOT repeat any of this — all variables are alive in kernel):')
            for entry in journal:
                parts.append(f'  {entry}')
            parts.append('')
            completed = execution_context.get('completed_iterations', [])
            detail_count = self.max_journal_detail_entries
            if completed:
                recent = completed[-detail_count:]
                parts.append('LAST ITERATION DETAIL:')
                for item in recent:
                    iter_id = item.get('iteration', '?')
                    code_preview = item.get('code', '')
                    if code_preview and len(code_preview) > self.code_preview_max_chars:
                        code_preview = code_preview[:self.code_preview_max_chars] + '\n# ...[truncated]'
                    output_preview = item.get('output', '')
                    if output_preview and len(output_preview) > self.output_preview_max_chars:
                        output_preview = output_preview[:self.output_preview_max_chars] + '\n...[truncated]'
                    if code_preview:
                        parts.append(f'  Code (Step {iter_id}):\n{code_preview}')
                    if output_preview:
                        parts.append(f'  Output (Step {iter_id}):\n{output_preview}')
                parts.append('')
        else:
            parts.append('No iterations completed yet — this is the first iteration.')
            parts.append('')
        available_vars = execution_context.get('available_variables', {})
        if available_vars:
            vars_lines = []
            dataframes = []
            for name, info in available_vars.items():
                if isinstance(info, dict):
                    type_str = info.get('type', 'Unknown')
                    if type_str == 'DataFrame':
                        dataframes.append(name)
                    details = ''
                    if 'columns' in info:
                        cols = info['columns']
                        if len(cols) > 10:
                            cols_str = str(cols[:10])[:-1] + ', ...]'
                        else:
                            cols_str = str(cols)
                        details = f' columns={cols_str}'
                    if 'shape' in info:
                        details += f" shape={info['shape']}"
                    intent_str = f''' intent="{info['intent']}"''' if 'intent' in info else ''
                    value_str = ''
                    if name in ('TASKS', 'COMPLETED_TASKS', '_VAR_INTENT_') and 'value' in info:
                        value_str = f" value={info['value']}"
                    vars_lines.append(f'- {name} ({type_str}){details}{intent_str}{value_str}')
                else:
                    vars_lines.append(f'- {name} ({info})')
            parts.append(f'AVAILABLE VARIABLES:\n' + '\n'.join(vars_lines))
            if dataframes and 'df' not in dataframes:
                if len(dataframes) == 1:
                    parts.append(f"\n⚠️ CRITICAL: The dataframe is named '{dataframes[0]}'. DO NOT use 'df'. Use '{dataframes[0]}' instead.")
                else:
                    parts.append(f"\n⚠️ CRITICAL: Available dataframes: {', '.join(dataframes)}. DO NOT use 'df' unless it is defined.")
            parts.append('⚠️ CRITICAL: These are the ONLY variables in memory. Data is NOT preloaded as table-name globals (e.g., IFFCO_INV_AI_CONS does NOT exist as a variable). Use ONLY the exact variable names listed above.')
            if 'FINAL_RESULT' in available_vars:
                parts.append('\n STOP — FINAL_RESULT IS ALREADY SET IN THE KERNEL. Your analysis is COMPLETE. You MUST return action: "done" immediately. Do NOT generate more code. Do NOT re-compute or re-set FINAL_RESULT. The answer has already been produced.')
            parts.append('')
        data_profile = execution_context.get('data_profile', {})
        if data_profile:
            delta = self._compute_profile_delta(data_profile)
            if delta:
                parts.append('DATASET PROFILE CHANGES:')
                for ds_name, info in delta.items():
                    if info['status'] == 'new':
                        prof = info['profile']
                        parts.append(f'  NEW: {ds_name}')
                        if prof.get('shape'):
                            parts.append(f"    shape   = {prof['shape']}")
                        if prof.get('columns'):
                            parts.append(f"    columns = {prof['columns']}")
                        if prof.get('dtypes'):
                            parts.append(f"    dtypes  = {prof['dtypes']}")
                        if prof.get('null_counts'):
                            parts.append(f"    nulls   = {prof['null_counts']}")
                        if prof.get('sample_row'):
                            parts.append(f"    sample  = {str(prof['sample_row'][0])[:200]}")
                        if prof.get('string_values'):
                            parts.append(f'    string_values:')
                            for col_name, sv in prof['string_values'].items():
                                unique_ct = sv.get('unique_count', '?')
                                top_vals = sv.get('top_values', [])
                                parts.append(f'      {col_name} ({unique_ct} unique): {top_vals}')
                    elif info['status'] == 'changed':
                        parts.append(f"  CHANGED: {ds_name} shape {info.get('shape_before')} → {info.get('shape_after')}")
                has_string_values = any((info['status'] == 'new' and info.get('profile', {}).get('string_values') for info in delta.values()))
                if has_string_values:
                    parts.append('⚠️ STRING FILTER RULE: When filtering on a string column, first EXPLORE with str.contains(), then REVIEW unique values, then FILTER.')
                parts.append('USE ONLY THESE EXACT COLUMN NAMES — case-sensitive, no variation.')
                parts.append('')
            elif not journal:
                parts.append('DATASET PROFILE (loaded DataFrames):')
                for ds_name, prof in data_profile.items():
                    parts.append(f'  {ds_name}:')
                    if prof.get('shape'):
                        parts.append(f"    shape   = {prof['shape']}")
                    if prof.get('columns'):
                        parts.append(f"    columns = {prof['columns']}")
                    if prof.get('dtypes'):
                        parts.append(f"    dtypes  = {prof['dtypes']}")
                    if prof.get('null_counts'):
                        parts.append(f"    nulls   = {prof['null_counts']}")
                    if prof.get('sample_row'):
                        parts.append(f"    sample  = {str(prof['sample_row'][0])[:200]}")
                    if prof.get('string_values'):
                        parts.append(f'    string_values:')
                        for col_name, sv in prof['string_values'].items():
                            unique_ct = sv.get('unique_count', '?')
                            top_vals = sv.get('top_values', [])[:self.string_values_top_n]
                            suffix = f' ... ({unique_ct} unique total)' if unique_ct and int(str(unique_ct)) > self.string_values_top_n else ''
                            parts.append(f'      {col_name} ({unique_ct} unique): {top_vals}{suffix}')
                parts.append('USE ONLY THESE EXACT COLUMN NAMES — case-sensitive, no variation.')
                has_sv = any((prof.get('string_values') for prof in data_profile.values()))
                if has_sv:
                    parts.append('⚠️ STRING FILTER RULE: When filtering on a string column, first EXPLORE with str.contains(), then REVIEW unique values, then FILTER.')
                parts.append('')
        failed_iters = execution_context.get('failed_iterations', [])
        if failed_iters:
            parts.append('⚠️ FAILED ITERATIONS (these approaches ALREADY FAILED — learn from them):')
            for fi in failed_iters[-3:]:
                parts.append(f"  Iteration {fi['iteration']} FAILED:")
                parts.append(f"    Error: {fi['error'][:300]}")
                if fi.get('code_snippet'):
                    parts.append(f"    Attempted code (snippet): {fi['code_snippet']}")
            parts.append('  → Do NOT repeat these failed approaches. Fix the SPECIFIC error and continue from AVAILABLE VARIABLES.')
            parts.append('')
        warnings = execution_context.get('warnings', [])
        if warnings:
            zero_row_warnings = [w for w in warnings if 'ZERO_ROW' in w or 'became empty' in w]
            other_warnings = [w for w in warnings if w not in zero_row_warnings]
            if zero_row_warnings:
                parts.append('⚠️ CRITICAL — ZERO-ROW RESULTS DETECTED:')
                for w in zero_row_warnings:
                    parts.append(f'  ❌ {w}')
                parts.append('  ACTION REQUIRED: Re-examine filter columns and values. Try alternative columns. Print unique values to verify.')
                parts.append('')
            if other_warnings:
                parts.append('WARNINGS from prior iterations:')
                for w in other_warnings:
                    parts.append(f'  - {w}')
                parts.append('')
        remaining = self.max_iterations - iteration
        parts.append(f'ITERATION: {iteration} / {self.max_iterations}')
        if remaining <= 2:
            parts.append(f'⚠️ URGENT: Only {remaining} iteration(s) remaining. You MUST assemble FINAL_RESULT in this iteration. Use whatever data you have — a partial answer is better than no answer.')
        elif remaining <= self.max_iterations // 2:
            parts.append(f'Note: {remaining} iterations remaining. If you have enough data, proceed to computation and FINAL_RESULT.')
        parts.extend(['', 'YOUR TASK:', 'Based on the user query, plan guidance, execution journal and outputs,', 'decide what to do NEXT.'])
        return '\n'.join(parts)

    async def _fetch_db_credentials(self) -> None:
        try:
            from services.db_credentials_service import DBCredentialsService
            from util.data_source import require_store_in_local
            actual_db = getattr(self.db, 'db', self.db) if type(self.db).__name__ == 'MongoDBManager' else self.db
            service = DBCredentialsService(actual_db)
            credentials = await service.get_credentials(self.client_id, db_type=None, decrypt_password=True, dataset_id=self.dataset_id)
            if not credentials:
                raise RuntimeError('store_in_local not configured in DB Configs')
            db_type = credentials.get('db_type', '')
            store_in_local = require_store_in_local(credentials)
            ssh_cfg = (credentials.get('additional_params') or {}).get('ssh') or {}
            ssh_enabled = bool(ssh_cfg.get('enabled'))
            if store_in_local or db_type == 'file_upload':
                if ssh_enabled and db_type == 'postgres':
                    raise RuntimeError("SSH Postgres credential is configured with store_in_local=true. Disable 'Store data locally' to run SSH live SQL mode.")
                self._is_live_db = False
                self.db_credentials_env = {}
                logger.info('Parquet/local mode for client=%s dataset_id=%s db_type=%s store_in_local=%s ssh_enabled=%s', self.client_id, self.dataset_id, db_type, store_in_local, ssh_enabled)
                return
            if db_type not in _LIVE_DB_TYPES:
                raise RuntimeError(f'Unsupported live DB type: {db_type}')
            db_url = credentials.get('db_url') or ''
            db_host = credentials.get('db_host') or ''
            db_password = credentials.get('db_password') or ''
            db_user = credentials.get('db_username') or ''
            if not db_url and (not db_host or not db_password or (not db_user)):
                raise RuntimeError(f'Incomplete DB credentials for client {self.client_id}. Please re-save credentials via the DB configuration page.')
            self._is_live_db = True
            import json as _json_mod
            self.db_credentials_env = {'CS_CLIENT_ID': self.client_id or '', 'CS_DATASET_ID': self.dataset_id or '', 'CS_DB_TYPE': db_type or '', 'CS_DB_HOST': db_host, 'CS_DB_PORT': str(credentials.get('db_port') or ''), 'CS_DB_NAME': credentials.get('db_name') or '', 'CS_DB_USER': db_user, 'CS_DB_PASSWORD': db_password, 'CS_DB_URL': db_url, 'CS_DB_SCHEMA': credentials.get('schema_name') or 'public', 'CS_SSH_ENABLED': 'true' if ssh_enabled else 'false', 'CS_SSH_TUNNEL': _json_mod.dumps(ssh_cfg) if ssh_enabled else '{}'}
            logger.info('Live DB mode for client=%s dataset_id=%s type=%s ssh_enabled=%s', self.client_id, self.dataset_id, db_type, ssh_enabled)
        except Exception as e:
            logger.error(f'Error fetching DB credentials: {e}')
            raise

    def _load_system_prompt(self, resolved_prompt: Optional[str]=None) -> str:
        if resolved_prompt:
            return resolved_prompt
        relative_path = f'agents/{self.agent_name}.xml'
        if self.db is not None and self.client_id:
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    return loop.run_until_complete(load_client_prompt(relative_path, self.client_id, self.db, use_formatting=False))
            except Exception as e:
                logger.warning('Client-aware prompt loading failed for %s (client=%s), falling back to base: %s', self.agent_name, self.client_id, e)
        try:
            prompt_path = Path(BASE_PROMPTS_PATH) / 'agents' / f'{self.agent_name}.xml'
            if prompt_path.exists():
                with open(prompt_path, 'r') as f:
                    return f.read()
        except Exception as e:
            logger.warning(f'Could not load XML prompt: {e}')
        return 'You are an expert Data Science Agent. Generate clean, executable Python code for data analysis, predictions, and iterative refinement in a Jupyter environment.'

    def _update_usage(self, usage: Optional[Dict[str, Any]]) -> None:
        if not usage:
            return

        def _safe_int(value: Any) -> int:
            if value is None:
                return 0
            try:
                return int(value)
            except (TypeError, ValueError):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return 0
        prompt_tokens = _safe_int(usage.get('prompt_tokens') or usage.get('input_tokens') or usage.get('prompt_token_count'))
        completion_tokens = _safe_int(usage.get('completion_tokens') or usage.get('output_tokens') or usage.get('candidates_token_count'))
        self.usage_stats['prompt_tokens'] = self.usage_stats.get('prompt_tokens', 0) + prompt_tokens
        self.usage_stats['completion_tokens'] = self.usage_stats.get('completion_tokens', 0) + completion_tokens
        self.usage_stats['total_tokens'] = self.usage_stats.get('total_tokens', 0) + (prompt_tokens + completion_tokens)
        extra_token_keys = ['reasoning_tokens', 'cached_input_tokens', 'cache_creation_input_tokens', 'audio_input_tokens', 'audio_output_tokens', 'image_input_tokens', 'accepted_prediction_tokens', 'rejected_prediction_tokens', 'text_input_tokens', 'text_output_tokens', 'total_tokens_provider']
        for key in extra_token_keys:
            value = _safe_int(usage.get(key))
            if value:
                self.usage_stats[key] = self.usage_stats.get(key, 0) + value
        if usage.get('model'):
            self.usage_stats['models'].add(usage['model'])

    def _parse_plan_steps(self, plan: str) -> List[Dict[str, Any]]:
        import re
        steps = []
        if isinstance(plan, list):
            plan = '\n'.join((str(s) for s in plan))
        lines = plan.strip().split('\n')
        current_step = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            step_match = re.match('^(\\d+)\\.\\s+(.+)$', line)
            if step_match:
                if current_step:
                    steps.append(current_step)
                step_num = int(step_match.group(1))
                description = step_match.group(2)
                current_step = {'step_num': step_num, 'description': description, 'details': []}
            elif line.startswith(('-', '*', '•')) and current_step:
                detail = line.lstrip('-*• ').strip()
                if detail:
                    current_step['details'].append(detail)
        if current_step:
            steps.append(current_step)
        logger.info(f'Parsed {len(steps)} steps from plan')
        return steps

    async def execute_analysis(self, user_query: str, plan: str, dataset_path: Optional[str]=None, dataset_dict: Optional[Dict]=None, context: Optional[Dict]=None) -> AsyncGenerator[Dict, None]:
        try:
            self.notebook_builder = NotebookBuilder(output_dir=self.notebook_output_dir, name_prefix='analysis')
            self.notebook_builder.add_markdown_cell(f"# Data Science Analysis\n\n**Query:** {user_query}\n\n**Generated:** {utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n---\n\n## Guidance\n\n{plan}")
            self.notebook_builder.save()
            yield (await self._stream_event('status', {'message': f'Analysis started — working iteratively (max {self.max_iterations} iterations)', 'max_iterations': self.max_iterations, 'notebook_path': str(self.notebook_builder.filepath)}))
            is_adhoc_request = bool((context or {}).get('adhoc_mode'))
            if is_adhoc_request:
                self._is_live_db = False
                self.db_credentials_env = {}
            else:
                await self._fetch_db_credentials()
            yield (await self._stream_event('status', {'message': 'Initializing Jupyter kernel...'}))
            await self._initialize_kernel()
            if self._session_owned:
                await self._clear_stale_kernel_sentinels()
            if self._is_live_db:
                yield (await self._stream_event('status', {'message': 'Injecting SQL query helpers...'}))
                await self._inject_sql_query_helpers()
                loaded_datasets = []
            else:
                yield (await self._stream_event('status', {'message': 'Loading dataset...'}))
                try:
                    loaded_datasets = await self._load_dataset_to_kernel(dataset_path, dataset_dict)
                except FileNotFoundError as e:
                    _frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
                    _support_email = os.getenv('SUPPORT_EMAIL', 'support@coresight.ai')
                    _admin_link = f'{_frontend_url}/admin?tab=database'
                    yield (await self._stream_event('error', {'message': f'Assets could not be found for the selected dataset, please retry uploading file / adding db credentials at {_admin_link}. If it fails please contact {_support_email}', 'error_type': 'assets_not_found', 'detail': str(e)}))
                    return
            kernel_vars = await self._get_kernel_variables()
            actual_vars = list(kernel_vars.keys()) if kernel_vars else []
            if not self._is_live_db and (not actual_vars) and (not loaded_datasets):
                try:
                    from config.system_config import STORAGE_BACKEND as _sb_force
                    if _sb_force == 'gcs':
                        from agents.data_science_agent import _get_duckdb_bootstrap_code
                        data_prefix = storage_datasets_prefix(self.client_id, self.dataset_id)
                        bootstrap = _get_duckdb_bootstrap_code(self.client_id, self.dataset_id)
                        if bootstrap:
                            _bootstrap_result = await self._execute_code(bootstrap)
                            if _bootstrap_result.get('exception'):
                                logger.error('DuckDB GCS bootstrap failed in force-load: %s', _bootstrap_result['exception'])
                                raise RuntimeError(f"DuckDB GCS bootstrap failed: {_bootstrap_result['exception']}. Ensure the kernel image has duckdb with httpfs pre-installed (see Dockerfile.datascience).")
                        try:
                            from util.storage.backend import get_storage_backend
                            storage = get_storage_backend()
                            gcs_files = await storage.list_files(data_prefix)
                            parquet_files = [f for f in gcs_files if f.endswith('.parquet')]
                            if parquet_files:
                                loaded_datasets = []
                                for pf in parquet_files:
                                    fname = pf.rsplit('/', 1)[-1] if '/' in pf else pf
                                    safe_name = fname.replace(' ', '_').replace('-', '_').replace('.parquet', '')
                                    loaded_datasets.append({'path': fname, 'variable': f'df_{safe_name}' if len(parquet_files) > 1 else 'df', 'format': 'parquet', 'gcs': True})
                                logger.info(f'GCS force-load: found {len(parquet_files)} parquet files')
                        except Exception as e:
                            logger.warning(f'GCS force-load file listing failed: {e}')
                    else:
                        client_data_dir = assets_datasets_dir(self.client_id, self.dataset_id)
                        if client_data_dir.exists():
                            parquet_files = list(client_data_dir.glob('*.parquet'))
                            if parquet_files:
                                main_file = parquet_files[0]
                                container_path = str(main_file)
                                if self.kernel_manager:
                                    self.kernel_manager.copy_file_to_container(str(main_file), container_path)
                                logger.info(f'Force-loading dataset: {container_path}')
                                force_code = f"""\nimport pandas as pd\ntry:\n    df = pd.read_parquet(r'{container_path}')\n    print(f"Force-loaded dataset from {container_path}")\n    print(f"Shape: {{df.shape}}")\nexcept Exception as e:\n    print(f"Force-load failed: {{e}}")\n"""
                                await self._execute_code(force_code)
                                kernel_vars = await self._get_kernel_variables()
                                actual_vars = list(kernel_vars.keys())
                                loaded_datasets = [{'path': container_path, 'variable': 'df', 'format': 'parquet'}]
                except Exception as e:
                    logger.warning(f'Force loading failed: {e}')
            if not actual_vars:
                logger.warning('No variables found in kernel after dataset loading')
            execution_context = {'user_query': user_query, 'plan_guidance': plan, 'dataset_path': dataset_path, 'available_variables': kernel_vars if kernel_vars else {}, 'completed_iterations': [], 'execution_journal': [], 'context': context or {}, 'loaded_datasets': loaded_datasets, 'live_sql_mode': self._is_live_db, 'db_type': (self.db_credentials_env or {}).get('CS_DB_TYPE', '') if self._is_live_db else '', 'warnings': []}
            planned_tables = (context or {}).get('planned_tables') or []
            if planned_tables:
                self._planned_tables = [str(t).strip() for t in planned_tables if str(t).strip()]
                logger.info('[PromptScope] data_science planned_tables=%s', self._planned_tables)
            yield (await self._stream_event('status', {'message': 'Injecting llm_query() helper...'}))
            await self._inject_llm_query_helper()
            if self._is_live_db:
                execution_context['file_schemas'] = {}
                execution_context['data_profile'] = {}
            else:
                yield (await self._stream_event('status', {'message': 'Reading file schemas...'}))
                execution_context['file_schemas'] = await self._probe_parquet_schemas(loaded_datasets, dataset_path)
                yield (await self._stream_event('status', {'message': 'Profiling dataset...'}))
                execution_context['data_profile'] = await self._probe_dataset_profile()
            is_adhoc = (context or {}).get('adhoc_mode', False)
            if is_adhoc:
                execution_context['knowledge_context'] = {}
                execution_context['adhoc_mode'] = True
                logger.info('Adhoc mode: skipping backend knowledge loading')
            else:
                try:
                    execution_context['knowledge_context'] = self._load_knowledge_for_coding()
                except Exception as e:
                    logger.warning('Knowledge loading failed (non-fatal): %s', e)
                    execution_context['knowledge_context'] = {}
            iteration = 0
            status = 'continue'
            consecutive_failures = 0
            MAX_CONSECUTIVE_FAILURES = self.doom_loop_threshold
            self._recent_failed_codes = []
            while iteration < self.max_iterations and status == 'continue':
                iteration += 1
                if iteration > 1 and 'FINAL_RESULT' in execution_context.get('available_variables', {}):
                    logger.info('Iteration %d: FINAL_RESULT already in kernel from previous iteration — stopping immediately.', iteration)
                    if self.notebook_builder:
                        self.notebook_builder.add_markdown_cell('## Analysis Complete\n\nFINAL_RESULT was set in a previous iteration — no further iterations needed.')
                        self.notebook_builder.save()
                    status = 'done'
                    break
                yield (await self._stream_event('status', {'message': f'Iteration {iteration}/{self.max_iterations} — deciding next action...'}))
                _early_iteration_start = False
                try:
                    if iteration == 1:
                        try:
                            decision = await self._generate_technical_plan(user_query=user_query, plan_guidance=plan, execution_context=execution_context)
                        except Exception as plan_err:
                            logger.warning('Iteration 1: technical-plan call failed (%s) — falling back to _decide_next_action.', plan_err)
                            decision = await self._decide_next_action(user_query=user_query, plan_guidance=plan, execution_context=execution_context, iteration=iteration)
                    else:
                        from config.system_config import STREAM_CODE_TOKENS, USE_TIERED_PROMPTS
                        if STREAM_CODE_TOKENS and USE_TIERED_PROMPTS:
                            decision = None
                            async for _ev in self._decide_next_action_streaming(user_query=user_query, plan_guidance=plan, execution_context=execution_context, iteration=iteration):
                                if 'action_header' in _ev:
                                    _hdr = _ev['action_header']
                                    if _hdr.get('action') == 'code':
                                        yield (await self._stream_event('iteration_start', {'iteration': iteration, 'max_iterations': self.max_iterations, 'reasoning': _hdr.get('reasoning', ''), 'thinking': _hdr.get('thinking', '')}))
                                        _early_iteration_start = True
                                elif _ev.get('action_token_kind') == 'code':
                                    yield (await self._stream_event('code_token', {'iteration': iteration, 'delta': _ev.get('delta', ''), 'attempt': 1}))
                                elif 'decision' in _ev:
                                    decision = _ev['decision']
                            if decision is None:
                                decision = await self._decide_next_action(user_query=user_query, plan_guidance=plan, execution_context=execution_context, iteration=iteration)
                        else:
                            decision = await self._decide_next_action(user_query=user_query, plan_guidance=plan, execution_context=execution_context, iteration=iteration)
                except Exception as e:
                    logger.error(f'Iteration {iteration}: _decide_next_action failed: {e}')
                    yield (await self._stream_event('error', {'message': f'Decision-making failed at iteration {iteration}: {e}'}))
                    status = 'error'
                    break
                action = decision.get('action', 'code')
                reasoning = decision.get('reasoning', '')
                thinking = decision.get('thinking', '')
                code = decision.get('code', '')
                if action == 'done':
                    logger.info(f'Iteration {iteration}: LLM declared DONE — {reasoning}')
                    yield (await self._stream_event('iteration_complete', {'iteration': iteration, 'action': 'done', 'reasoning': reasoning}))
                    if self.notebook_builder:
                        self.notebook_builder.add_markdown_cell(f'## ✅ Analysis Complete (iteration {iteration})\n\n{reasoning}')
                        self.notebook_builder.save()
                    status = 'done'
                    break
                if not _early_iteration_start:
                    yield (await self._stream_event('iteration_start', {'iteration': iteration, 'max_iterations': self.max_iterations, 'reasoning': reasoning, 'thinking': thinking}))
                if self.notebook_builder:
                    self.notebook_builder.add_markdown_cell(f'## Iteration {iteration}: {reasoning}')
                    self.notebook_builder.save()
                logger.info(f'Iteration {iteration}/{self.max_iterations}: {reasoning}')
                iteration_success = False
                last_error = None
                _stashed_failed_code = None
                _stashed_error_type = None
                for attempt in range(self.max_retries_per_iteration):
                    try:
                        if attempt > 0:
                            code = await self._regenerate_code_after_error(user_query=user_query, plan_guidance=plan, execution_context=execution_context, iteration=iteration, failed_code=code, error=last_error, attempt=attempt)
                        if not code:
                            last_error = 'Failed to generate code'
                            continue
                        validation_err = self._validate_code_syntax(code, execution_context.get('available_variables', {}))
                        if validation_err:
                            logger.warning(f'Iteration {iteration}: AST validation failed: {validation_err}')
                            last_error = f'Code validation error: {validation_err}'
                            self._recent_failed_codes.append(code)
                            continue
                        if self._detect_doom_loop(code):
                            doom_msg = f'Doom loop detected at iteration {iteration}: the last {self.doom_loop_threshold} failed attempts used nearly identical code. Aborting to prevent wasted compute.'
                            logger.warning(doom_msg)
                            yield (await self._stream_event('error', {'message': doom_msg, 'iteration': iteration, 'last_error': last_error or ''}))
                            status = 'error'
                            break
                        yield (await self._stream_event('code_generated', {'iteration': iteration, 'code': code, 'attempt': attempt + 1}))
                        if self.notebook_builder:
                            self.notebook_builder.add_code_cell(code)
                            self.notebook_builder.save()
                        execution_result = await self._execute_code(code)
                        _raw_stdout = execution_result.get('stdout', '')
                        _clean_stdout = '\n'.join((line for line in _raw_stdout.splitlines() if not line.startswith('__LIVE_SQL_LOG__:')))
                        _MAX_SSE_STDOUT = 5000
                        if len(_clean_stdout) > _MAX_SSE_STDOUT:
                            _clean_stdout = _clean_stdout[:_MAX_SSE_STDOUT] + '\n...[output truncated]...'
                        yield (await self._stream_event('iteration_execution', {'iteration': iteration, 'attempt': attempt + 1, 'stdout': _clean_stdout, 'stderr': execution_result.get('stderr', ''), 'exception': execution_result.get('exception')}))
                        if self.notebook_builder and execution_result.get('stdout'):
                            self.notebook_builder.add_output_to_last_cell(execution_result.get('stdout', ''))
                            self.notebook_builder.save()
                        detected_error = execution_result.get('exception')
                        if not detected_error and self._stdout_contains_error(execution_result.get('stdout', '')):
                            detected_error = self._extract_error_from_stdout(execution_result.get('stdout', ''))
                            execution_result['exception'] = detected_error
                            logger.info(f'Iteration {iteration}: detected error in stdout: {detected_error[:150]}')
                        if detected_error:
                            last_error = detected_error
                            _stashed_failed_code = code
                            _stashed_error_type, _, _ = self._classify_error(detected_error)
                            logger.warning(f'Iteration {iteration} failed (attempt {attempt + 1}): {last_error[:200]}')
                            if _stashed_error_type == 'FILE_NOT_FOUND':
                                _frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
                                _support_email = os.getenv('SUPPORT_EMAIL', 'support@coresight.ai')
                                _admin_link = f'{_frontend_url}/admin?tab=database'
                                yield (await self._stream_event('error', {'message': f'Assets could not be found, please retry uploading file / adding db credentials at {_admin_link}. If it fails please contact {_support_email}', 'error_type': 'assets_not_found'}))
                                status = 'error'
                                break
                            self._recent_failed_codes.append(code)
                            if len(self._recent_failed_codes) > self.doom_loop_threshold * 2:
                                self._recent_failed_codes = self._recent_failed_codes[-self.doom_loop_threshold:]
                            if attempt == 0:
                                diag_code = await self._generate_diagnostic_code(code, last_error, execution_context.get('available_variables', {}))
                                if diag_code:
                                    diag_result = await self._execute_code(diag_code)
                                    diag_output = diag_result.get('stdout', '')
                                    if diag_output:
                                        last_error += f'\n\nDIAGNOSTIC OUTPUT:\n{diag_output[:300]}'
                            if attempt < self.max_retries_per_iteration - 1:
                                yield (await self._stream_event('iteration_retry', {'iteration': iteration, 'attempt': attempt + 1, 'error': last_error, 'message': 'Retrying with error feedback...'}))
                            continue
                        iteration_success = True
                        self._recent_failed_codes = []
                        try:
                            raw_db = self._get_raw_db()
                            if raw_db:
                                from services.lesson_extractor import LessonExtractor
                                from services.agent_lesson_service import AgentLessonService
                                _lesson_svc = AgentLessonService(raw_db)
                                if attempt > 0 and _stashed_failed_code:
                                    lessons = LessonExtractor.extract_from_error_recovery(error_type=_stashed_error_type or 'UNKNOWN', error_text=last_error or '', failed_code=_stashed_failed_code, fixed_code=code, file_schemas=execution_context.get('file_schemas', {}))
                                    if lessons:
                                        logger.info('Lesson hook 1 (error recovery): extracted %d lesson(s)', len(lessons))
                                    for lsn in lessons:
                                        await _lesson_svc.save_lesson(self.client_id, lsn)
                                pattern_lessons = LessonExtractor.extract_from_code_pattern(code)
                                if pattern_lessons:
                                    logger.info('Lesson hook 2 (code pattern): extracted %d lesson(s)', len(pattern_lessons))
                                for lsn in pattern_lessons:
                                    await _lesson_svc.save_lesson(self.client_id, lsn)
                        except Exception as _le:
                            logger.debug('Lesson extraction skipped: %s', _le)
                        fr_exists = await self._check_final_result_in_kernel()
                        prev_vars = execution_context.get('available_variables', {})
                        new_vars = await self._get_kernel_variables()
                        if new_vars:
                            execution_context['available_variables'] = new_vars
                        elif prev_vars:
                            logger.warning('Iteration %d: _get_kernel_variables() returned empty but previous vars existed (%d vars). Keeping previous variables.', iteration, len(prev_vars))
                            new_vars = prev_vars
                        if fr_exists and 'FINAL_RESULT' not in new_vars:
                            logger.warning('Iteration %d: FINAL_RESULT exists in kernel but _get_kernel_variables() missed it!', iteration)
                            new_vars['FINAL_RESULT'] = {'type': 'dict'}
                            execution_context['available_variables'] = new_vars
                        existing_profile_keys = set(execution_context.get('data_profile', {}).keys())
                        current_df_names = {name for name, info in new_vars.items() if isinstance(info, dict) and info.get('type') == 'DataFrame'}
                        if not existing_profile_keys or current_df_names != existing_profile_keys:
                            new_profile = await self._probe_dataset_profile()
                            if new_profile:
                                execution_context['data_profile'] = new_profile
                                logger.info(f'Re-profiled after iteration {iteration}: new={current_df_names - existing_profile_keys}')
                                try:
                                    raw_db = self._get_raw_db()
                                    if raw_db:
                                        from services.lesson_extractor import LessonExtractor
                                        from services.agent_lesson_service import AgentLessonService
                                        profile_lessons = LessonExtractor.extract_from_data_profile(new_profile, execution_context.get('file_schemas', {}))
                                        if profile_lessons:
                                            logger.info('Lesson hook 3 (data profile): extracted %d lesson(s)', len(profile_lessons))
                                            _lsvc = AgentLessonService(raw_db)
                                            for lsn in profile_lessons:
                                                await _lsvc.save_lesson(self.client_id, lsn)
                                except Exception:
                                    pass
                        is_valid, validation_issue = await self._validate_step_output({'step_num': iteration, 'description': reasoning}, new_vars, prev_vars)
                        if not is_valid and attempt < self.max_retries_per_iteration - 1:
                            diag_code = self._generate_zero_row_diagnostic(validation_issue, new_vars)
                            diag_output = ''
                            if diag_code:
                                diag_result = await self._execute_code(diag_code)
                                diag_output = diag_result.get('stdout', '')[:500]
                            last_error = f'ZERO_ROW_RESULT: {validation_issue}. The filter/join produced an empty DataFrame. This likely means wrong column or wrong values were used for filtering. Re-check which column in the target table corresponds to the lookup value. Try alternative columns.'
                            if diag_output:
                                last_error += f'\n\nDIAGNOSTIC (unique values in related columns):\n{diag_output}'
                            logger.warning(f'Iteration {iteration}: zero-row self-correction triggered: {validation_issue}')
                            iteration_success = False
                            yield (await self._stream_event('iteration_retry', {'iteration': iteration, 'attempt': attempt + 1, 'error': last_error, 'message': 'Zero-row result detected — retrying with diagnostic context...'}))
                            continue
                        elif not is_valid:
                            logger.warning(f'Iteration {iteration} silent failure (no retries left): {validation_issue}')
                            execution_context['warnings'].append(f'Iteration {iteration}: ZERO_ROW_RESULT: {validation_issue}')
                        explosion_warnings = self._detect_row_explosion(new_vars, prev_vars)
                        for w in explosion_warnings:
                            logger.warning('Iteration %d: %s', iteration, w)
                            execution_context['warnings'].append(f'Iteration {iteration}: {w}')
                        raw_output = execution_result.get('stdout', '')
                        if len(raw_output) > self.output_storage_max_chars:
                            half = self.output_storage_max_chars // 2
                            raw_output = raw_output[:half] + '\n...[truncated]...\n' + raw_output[-half:]
                        execution_context['completed_iterations'].append({'iteration': iteration, 'reasoning': reasoning, 'thinking': thinking, 'code': code, 'output': raw_output, 'variables': new_vars})
                        execution_context.setdefault('execution_journal', []).append(self._build_journal_entry(iteration, reasoning, new_vars, prev_vars))
                        self._register_artifact(iteration, reasoning, new_vars, prev_vars)
                        yield (await self._stream_event('iteration_complete', {'iteration': iteration, 'reasoning': reasoning, 'available_variables': list(new_vars.keys())}))
                        from config.system_config import USE_TIERED_PROMPTS
                        if not USE_TIERED_PROMPTS:
                            completed = execution_context['completed_iterations']
                            if len(completed) % self.context_compaction_interval == 0 and len(completed) >= self.context_compaction_interval:
                                try:
                                    n = self.context_compaction_interval
                                    batch = completed[-n:]
                                    summary_text = await self._summarize_completed_steps(batch)
                                    execution_context['completed_iterations'] = completed[:-n] + [{'iteration': f"summary({batch[0]['iteration']}-{batch[-1]['iteration']})", 'reasoning': summary_text, 'code': '', 'output': '', 'variables': new_vars}]
                                    logger.info(f"Context compacted: summarized iterations {batch[0]['iteration']}-{batch[-1]['iteration']}")
                                except Exception as compact_err:
                                    logger.warning(f'Context compaction skipped: {compact_err}')
                        logger.info(f'Iteration {iteration} completed successfully')
                        consecutive_failures = 0
                        logger.debug('Iteration %d: checking for FINAL_RESULT in new_vars. Keys: %s', iteration, list(new_vars.keys()) if new_vars else 'EMPTY')
                        if 'FINAL_RESULT' in new_vars:
                            logger.info(f'Iteration {iteration}: FINAL_RESULT detected in kernel — auto-declaring done.')
                            if self.notebook_builder:
                                self.notebook_builder.add_markdown_cell(f'## ✅ Analysis Complete (iteration {iteration})\n\nFINAL_RESULT was set — stopping.')
                                self.notebook_builder.save()
                            status = 'done'
                        break
                    except Exception as e:
                        last_error = str(e)
                        logger.error(f'Error in iteration {iteration}, attempt {attempt + 1}: {e}')
                        if attempt < self.max_retries_per_iteration - 1:
                            yield (await self._stream_event('iteration_retry', {'iteration': iteration, 'attempt': attempt + 1, 'error': str(e)}))
                if status == 'error':
                    break
                if not iteration_success:
                    consecutive_failures += 1
                    execution_context.setdefault('failed_iterations', []).append({'iteration': iteration, 'error': (last_error or 'unknown')[:500], 'code_snippet': (code or '')[:300]})
                    logger.warning(f'Iteration {iteration} failed after {self.max_retries_per_iteration} attempts (consecutive failures: {consecutive_failures})')
                    if self.notebook_builder:
                        self.notebook_builder.add_markdown_cell(f"❌ **Iteration {iteration} FAILED** after {self.max_retries_per_iteration} attempts.\n\nLast error: `{(last_error[:300] if last_error else 'Unknown')}`")
                        self.notebook_builder.save()
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        yield (await self._stream_event('error', {'message': f'{consecutive_failures} consecutive iterations failed. Stopping to prevent wasted compute.', 'iteration': iteration, 'last_error': last_error}))
                        status = 'error'
                        break
            completed_count = len(execution_context['completed_iterations'])
            if status == 'error' or completed_count == 0:
                failure_msg = f'Analysis incomplete — {completed_count} iterations completed, stopped due to errors.'
                final_result = {'prediction': failure_msg, 'text_output': failure_msg, 'dataframe': None, 'iterations_completed': completed_count, 'timestamp': utcnow().isoformat(), 'pipeline_failed': True, '_agent_usage': {k: list(v) if isinstance(v, set) else v for k, v in self.usage_stats.items()}}
                yield (await self._stream_event('final_result', final_result))
                if self.notebook_builder:
                    self.notebook_builder.add_markdown_cell(f'---\n\n## ❌ Analysis Aborted\n\n{failure_msg}')
                    self.notebook_builder.save()
            else:
                if self.kernel_manager:
                    self.kernel_manager.update_activity()
                yield (await self._stream_event('status', {'message': 'Fetching result data...'}))
                final_df_records = await self._fetch_generated_dataframe()
                logger.info(f'[DS_AGENT] _fetch_generated_dataframe → {type(final_df_records).__name__}, len={(len(final_df_records) if final_df_records else None)}')
                all_charts = await self._fetch_all_generated_charts()
                logger.info(f'[DS_AGENT] _fetch_all_generated_charts → {len(all_charts)} charts')
                if not all_charts:
                    single = await self._fetch_generated_chart()
                    if single:
                        all_charts = [{'name': '_generated_plotly_fig_', 'figure': single}]
                yield (await self._stream_event('status', {'message': 'Generating final result...'}))
                try:
                    final_result = await self._generate_final_result(execution_context)
                except Exception as gen_err:
                    logger.warning(f'[DS_AGENT] _generate_final_result failed ({type(gen_err).__name__}: {gen_err}); using fallback summary')
                    _partial_usage = {k: list(v) if isinstance(v, set) else v for k, v in self.usage_stats.items()}
                    final_result = {'prediction': 'Analysis complete. Results are shown in the table below.', 'text_output': 'Analysis complete. Results are shown in the table below.', 'dataframe': None, 'iterations_completed': completed_count, 'timestamp': utcnow().isoformat(), '_agent_usage': _partial_usage}
                final_result['dataframe'] = final_df_records
                if all_charts:
                    final_result['charts'] = all_charts
                    final_result['chart'] = all_charts[0]['figure']
                logger.info('DS finalize: final_result keys=%s has_chart=%s has_table=%s', list(final_result.keys()) if isinstance(final_result, dict) else type(final_result).__name__, bool(final_result.get('chart')) if isinstance(final_result, dict) else False, bool(final_result.get('table')) if isinstance(final_result, dict) else False)
                yield (await self._stream_event('final_result', final_result))
                if self.notebook_builder:
                    summary_text = final_result.get('text_output', final_result.get('prediction', ''))
                    self.notebook_builder.add_markdown_cell(f'---\n\n## Final Result\n\n{summary_text}')
                    nb_path = self.notebook_builder.save()
                    final_result['notebook_path'] = str(nb_path)
            yield (await self._stream_event('status', {'message': f'Analysis complete ({completed_count} iterations)' if status != 'error' else 'Analysis incomplete due to errors', 'notebook_path': str(self.notebook_builder.filepath) if self.notebook_builder else None}))
        except Exception as e:
            logger.error(f'Error in execute_analysis: {e}\n{traceback.format_exc()}')
            partial_usage = {k: list(v) if isinstance(v, set) else v for k, v in self.usage_stats.items()} if hasattr(self, 'usage_stats') and self.usage_stats else {}
            yield (await self._stream_event('error', {'message': str(e), 'traceback': traceback.format_exc(), '_agent_usage': partial_usage}))
        finally:
            await self._cleanup_kernel()

    async def _generate_technical_plan(self, user_query: str, plan_guidance: str, execution_context: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = "You are a Senior Technical Data Architect. Your goal is to translate a business query and analyst guidance into a mandatory technical strategy for a recursive data science agent.\n\nRULES:\n1. Output ONLY valid JSON with keys: 'thinking' and 'code'.\n2. Your 'code' MUST ONLY initialize the following variables:\n   TASKS = [...]  # A list of 5-8 strings\n   COMPLETED_TASKS = []\n   _VAR_INTENT_ = {}\n3. Every string in 'TASKS' MUST be 'verbose' and tech-heavy. Include EXACT table names, column names,    and specific sargable filter values (e.g. 'Load TotalValue from NE_DMSSaleReportTable where OrderDate >= 2025-01-01')    from the PROVIDED guidance.\n4. Do NOT include any SQL loading or analysis logic in the code. This is a PLANNING-ONLY step.\n5. Do NOT include any explanation or markdown fences."
        user_message = f'USER QUERY: {user_query}\n\nANALYST GUIDANCE:\n{plan_guidance}\n\nBased on the guidance, create a 5-8 step verbose technical plan that implements the strategy exactly as described.'
        response = await self.llm_client.generate_completion(system_prompt=system_prompt, user_message=user_message, temperature=0.1)
        self._update_usage(response.get('usage'))
        content = (response.get('content') or '').strip()
        try:
            if content.startswith('```'):
                content = re.sub('```[a-z]*\\n|```', '', content).strip()
            parsed = json.loads(content)
            return {'action': 'code', 'thinking': parsed.get('thinking', 'Initializing high-fidelity technical plan.'), 'reasoning': 'Technical Planning Step', 'code': parsed.get('code', '')}
        except Exception as e:
            logger.error(f'Planning step JSON parse failed: {e}')
            raise ValueError(f'Technical planning failed: {e}')

    @traceable(name='coder_decide_action')
    async def _decide_next_action(self, user_query: str, plan_guidance: str, execution_context: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        from config.system_config import USE_TIERED_PROMPTS
        if USE_TIERED_PROMPTS:
            if self._static_system_context is None:
                self._static_system_context = await self._build_static_system_context(user_query, plan_guidance, execution_context)
            system_prompt = self._static_system_context
            user_message = self._build_dynamic_user_message(user_query, plan_guidance, execution_context, iteration)
            response = await self.llm_client.generate_completion(system_prompt=system_prompt, user_message=user_message, temperature=self.temperature)
            self._update_usage(response.get('usage'))
            content = (response.get('content') or '').strip()
            return self._parse_decision_response(content)
        return await self._decide_next_action_legacy(user_query, plan_guidance, execution_context, iteration)

    async def _decide_next_action_streaming(self, user_query: str, plan_guidance: str, execution_context: Dict[str, Any], iteration: int) -> AsyncGenerator[Dict[str, Any], None]:
        if self._static_system_context is None:
            self._static_system_context = await self._build_static_system_context(user_query, plan_guidance, execution_context)
        system_prompt = self._static_system_context
        user_message = self._build_dynamic_user_message(user_query, plan_guidance, execution_context, iteration)
        full = ''
        in_code = False
        code_body_start = 0
        code_emitted_len = 0
        tokens_seen = 0
        try:
            async for token, usage in self.llm_client.generate_completion_stream(system_prompt=system_prompt, user_message=user_message, temperature=self.temperature):
                if token == '__USAGE__':
                    if usage:
                        self._update_usage(usage)
                    continue
                if isinstance(token, str) and token.startswith('ERROR:'):
                    if tokens_seen == 0:
                        logger.warning('Streaming decision failed before any token (%s); falling back to one-shot decision.', token)
                        decision = await self._decide_next_action(user_query, plan_guidance, execution_context, iteration)
                        yield {'decision': decision}
                        return
                    logger.warning('Streaming decision errored mid-stream: %s', token)
                    break
                tokens_seen += 1
                full += token
                if not in_code:
                    match = self._CODE_SENTINEL_RE.search(full)
                    if match:
                        in_code = True
                        code_body_start = match.end()
                        action, reasoning, thinking = self._parse_decision_header(full[:match.start()])
                        yield {'action_header': {'action': action or 'code', 'reasoning': reasoning, 'thinking': thinking}, 'iteration': iteration}
                        initial = full[code_body_start:]
                        if initial:
                            code_emitted_len = len(initial)
                            yield {'action_token_kind': 'code', 'delta': initial, 'iteration': iteration}
                else:
                    code_so_far = full[code_body_start:]
                    new = code_so_far[code_emitted_len:]
                    if new:
                        code_emitted_len = len(code_so_far)
                        yield {'action_token_kind': 'code', 'delta': new, 'iteration': iteration}
        except Exception as e:
            if tokens_seen == 0:
                logger.warning('Streaming decision raised before any token (%s); falling back to one-shot decision.', e)
                decision = await self._decide_next_action(user_query, plan_guidance, execution_context, iteration)
                yield {'decision': decision}
                return
            logger.warning('Streaming decision raised mid-stream: %s', e)
        yield {'decision': self._parse_decision_response(full.strip())}

    async def _decide_next_action_legacy(self, user_query: str, plan_guidance: str, execution_context: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        prompt_parts = []
        prompt_parts.append(f'USER QUERY: {user_query}')
        prompt_parts.append('')
        prompt_parts.append('PLAN GUIDANCE (use as direction, not a rigid checklist):')
        prompt_parts.append(plan_guidance)
        prompt_parts.append('')
        file_schemas = execution_context.get('file_schemas', {})
        if file_schemas:
            prompt_parts.append('FILE SCHEMAS (EXACT column names — use ONLY these, case-sensitive):')
            for fname, schema in file_schemas.items():
                rows_info = f", {schema['num_rows']:,} rows" if schema.get('num_rows') else ''
                prompt_parts.append(f"  {fname} ({schema.get('path', '')}{rows_info})")
                prompt_parts.append(f"    columns = {schema.get('columns', [])}")
                if schema.get('types'):
                    prompt_parts.append(f"    types   = {schema.get('types', {})}")
            prompt_parts.append('')
            prompt_parts.append('⚠️ CRITICAL: The plan guidance may use WRONG column names (e.g. LAST_ISSUE_DATE). ALWAYS use the EXACT column names from FILE SCHEMAS above instead. ' + ('When using query_parquet(), select ONLY columns from the schema.' if self._is_gcs else 'When using pd.read_parquet(columns=[...]), use ONLY names from the schema.'))
            prompt_parts.append('')
        knowledge_ctx = execution_context.get('knowledge_context', {})
        if knowledge_ctx and file_schemas:
            from util.knowledge_filter import compress_table_introductions_for_coding, compress_data_descriptions_for_coding, compress_terminology_for_coding, _approx_token_count
            from config.system_config import MAX_CODING_KNOWLEDGE_TOKENS
            schema_tables = [Path(f).stem for f in file_schemas.keys()]
            knowledge_lines: list = []
            budget = MAX_CODING_KNOWLEDGE_TOKENS
            intros = compress_table_introductions_for_coding(knowledge_ctx.get('table_introductions_xml', ''), schema_tables)
            if intros:
                cost = _approx_token_count(intros)
                if cost <= budget:
                    knowledge_lines.append('TABLE DESCRIPTIONS:')
                    knowledge_lines.append(intros)
                    budget -= cost
            descs = compress_data_descriptions_for_coding(knowledge_ctx.get('data_descriptions', {}), schema_tables)
            if descs:
                cost = _approx_token_count(descs)
                if cost <= budget:
                    knowledge_lines.append('')
                    knowledge_lines.append('COLUMN DESCRIPTIONS (use to select correct columns):')
                    knowledge_lines.append(descs)
                    budget -= cost
            terms = compress_terminology_for_coding(knowledge_ctx.get('domain_terminology', ''))
            if terms:
                cost = _approx_token_count(terms)
                if cost <= budget:
                    knowledge_lines.append('')
                    knowledge_lines.append('DOMAIN TERMINOLOGY:')
                    knowledge_lines.append(terms)
            if knowledge_lines:
                prompt_parts.append('BUSINESS KNOWLEDGE (understand what the data means):')
                prompt_parts.extend(knowledge_lines)
                prompt_parts.append('')
        try:
            raw_db = self._get_raw_db()
            if raw_db:
                from services.agent_lesson_service import AgentLessonService
                from config.system_config import MAX_LESSONS_TOKENS
                lesson_svc = AgentLessonService(raw_db)
                planned_tables = getattr(self, '_planned_tables', None)
                schema_tables = [Path(f).stem for f in file_schemas.keys()] if file_schemas else None
                filter_tables = planned_tables or schema_tables
                lessons_text = await lesson_svc.format_lessons_for_prompt(self.client_id, tables=filter_tables, max_tokens=MAX_LESSONS_TOKENS)
                if lessons_text:
                    prompt_parts.append('LEARNED PATTERNS (from prior analyses — follow these strictly):')
                    prompt_parts.append(lessons_text)
                    prompt_parts.append('')
        except Exception as le:
            logger.debug('Lesson injection skipped: %s', le)
        client_profile = knowledge_ctx.get('client_data_profile', '')
        if client_profile:
            from config.system_config import MAX_DATA_PROFILE_TOKENS
            profile_cost = len(client_profile) // 4
            if profile_cost <= MAX_DATA_PROFILE_TOKENS:
                prompt_parts.append('CLIENT DATA PROFILE (formatting & locale guidance):')
                prompt_parts.append(client_profile)
                prompt_parts.append('')
        try:
            raw_db = self._get_raw_db()
            user_id = getattr(self, '_user_id', None)
            if raw_db and user_id:
                from services.user_preference_service import UserPreferenceService
                from services.preference_extractor import PreferenceExtractor
                from config.system_config import MAX_USER_PREFERENCES_TOKENS
                pref_svc = UserPreferenceService(raw_db)
                current_prefs = PreferenceExtractor.extract_as_dict(user_query) if user_query else {}
                prefs_text = await pref_svc.format_for_prompt(self.client_id, user_id, current_query_prefs=current_prefs, max_tokens=MAX_USER_PREFERENCES_TOKENS)
                if prefs_text:
                    prompt_parts.append('USER PREFERENCES (respect these for visualization and formatting):')
                    prompt_parts.append(prefs_text)
                    prompt_parts.append('')
        except Exception:
            pass
        data_profile = execution_context.get('data_profile', {})
        if file_schemas and len(file_schemas) > 1:
            prompt_parts.append('MULTI-TABLE JOIN CONTEXT:')
            col_to_files: Dict[str, list] = {}
            file_row_counts: Dict[str, int] = {}
            for fname, schema in file_schemas.items():
                for col in schema.get('columns', []):
                    col_to_files.setdefault(col, []).append(fname)
                if schema.get('num_rows') is not None:
                    file_row_counts[fname] = schema['num_rows']
                stem = Path(fname).stem
                stem_lower = stem.lower().replace('-', '_')
                for ds_name, prof in data_profile.items():
                    ds_lower = ds_name.lower().replace('-', '_')
                    if ds_lower == stem_lower or ds_lower.endswith(stem_lower) or stem_lower.endswith(ds_lower) or (stem_lower in ds_lower) or (ds_lower in stem_lower):
                        shape = prof.get('shape', [0])
                        file_row_counts[fname] = shape[0] if shape else 0
            shared_cols = {col: files for col, files in col_to_files.items() if len(files) > 1}
            if shared_cols:
                prompt_parts.append('  Shared columns (potential join keys):')
                for col, files in shared_cols.items():
                    prompt_parts.append(f"    {col}: appears in {', '.join(files)}")

            def _cols_near_match(a: str, b: str) -> bool:
                if a == b:
                    return False
                if a in b or b in a:
                    return True
                for suffix in ('_ID', '_NAME', '_CODE', '_KEY', '_NUM'):
                    if a.endswith(suffix) and b.endswith(suffix):
                        base_a = a[:-len(suffix)].rstrip('_')
                        base_b = b[:-len(suffix)].rstrip('_')
                        if base_a and base_b and (base_a in base_b or base_b in base_a):
                            return True
                return False
            all_cols_by_file = {fname: set(s.get('columns', [])) for fname, s in file_schemas.items()}
            near_matches = []
            fnames_list = list(all_cols_by_file.keys())
            for i in range(len(fnames_list)):
                for j in range(i + 1, len(fnames_list)):
                    for col_a in all_cols_by_file[fnames_list[i]]:
                        for col_b in all_cols_by_file[fnames_list[j]]:
                            if _cols_near_match(col_a, col_b):
                                near_matches.append((col_a, fnames_list[i], col_b, fnames_list[j]))
            if near_matches:
                prompt_parts.append('  Near-match columns (VERIFY overlap before joining — names differ):')
                for col_a, f_a, col_b, f_b in near_matches[:10]:
                    prompt_parts.append(f'    {col_a} ({f_a}) ↔ {col_b} ({f_b})')
            small_tables = []
            for fname, rows in file_row_counts.items():
                if rows < 10000:
                    small_tables.append(fname)
                    load_hint = f"Load ALL columns: query_parquet('{fname}')" if self._is_gcs else f'Load ALL columns: pd.read_parquet(path) with NO columns= parameter.'
                    prompt_parts.append(f"  ⚠️ CRITICAL: {fname} is a small table ({rows} rows) — IGNORE the plan's column selection for this file. {load_hint}")
            if not small_tables:
                for fname, schema in file_schemas.items():
                    n_cols = len(schema.get('columns', []))
                    if n_cols <= 6 and fname not in file_row_counts:
                        prompt_parts.append(f'  ℹ️ {fname} has only {n_cols} columns — likely a small lookup table. Load ALL columns to avoid needing to reload.')
            prompt_parts.append("\n  ⚠️ MULTI-TABLE JOIN RULE: Before joining two tables, you MUST:\n    1. For small lookup/dimension tables: load ALL columns (do NOT use columns= parameter)\n    2. BEFORE joining, verify join key overlap:\n       overlap = set(df_a['col_a'].unique()) & set(df_b['col_b'].unique())\n       print(f'Overlap: {len(overlap)} common values')\n    3. If overlap is 0, try OTHER candidate columns — check near-matches above\n    4. Column names may differ (e.g., ORGANIZATION_ID ↔ INV_ORG_ID) — check VALUES, not just names\n    5. Use the column pair with the HIGHEST overlap for the join\n    6. FILTER-BY-VALUE (same rules apply): When using a value from table A to filter table B (e.g., .isin(), == comparison):\n       - Verify the looked-up value EXISTS in the target column BEFORE filtering\n       - If 0 rows result, the value maps to a DIFFERENT column in table B\n       - Print unique values in candidate columns to find the correct mapping")
            prompt_parts.append('')
        if not data_profile:
            data_profile = execution_context.get('data_profile', {})
        if data_profile:
            prompt_parts.append('DATASET PROFILE (loaded DataFrames):')
            for ds_name, prof in data_profile.items():
                prompt_parts.append(f'  {ds_name}:')
                if prof.get('shape'):
                    prompt_parts.append(f"    shape   = {prof['shape']}")
                if prof.get('columns'):
                    prompt_parts.append(f"    columns = {prof['columns']}")
                if prof.get('dtypes'):
                    prompt_parts.append(f"    dtypes  = {prof['dtypes']}")
                if prof.get('null_counts'):
                    prompt_parts.append(f"    nulls   = {prof['null_counts']}")
                if prof.get('sample_row'):
                    prompt_parts.append(f"    sample  = {str(prof['sample_row'][0])[:200]}")
                if prof.get('string_values'):
                    prompt_parts.append(f'    string_values (actual values in string columns):')
                    for col_name, sv in prof['string_values'].items():
                        unique_ct = sv.get('unique_count', '?')
                        top_vals = sv.get('top_values', [])[:10]
                        suffix = f' ... ({unique_ct} unique total)' if unique_ct > 10 else ''
                        prompt_parts.append(f'      {col_name} ({unique_ct} unique): {top_vals}{suffix}')
            prompt_parts.append('USE ONLY THESE EXACT COLUMN NAMES — case-sensitive, no variation.')
            has_string_values = any((prof.get('string_values') for prof in data_profile.values()))
            if has_string_values:
                prompt_parts.append("⚠️ STRING FILTER RULE: When the user's query involves filtering on a string column, you MUST follow this iterative discovery process:\n  1. EXPLORE: Print unique values matching the user's keyword using str.contains(r'keyword', case=False, na=False), then print the matching unique values and their counts.\n  2. REVIEW & DECIDE: In the NEXT iteration, look at the discovered values. Decide which ones are relevant to the user's question. Not all matches may be relevant (e.g., 'cement mixer' is NOT cement inventory).\n  3. FILTER: Apply the final filter using the verified values.\nRefer to the string_values above for initial awareness of what's in each column.")
            prompt_parts.append('')
        available_vars = execution_context.get('available_variables', {})
        if available_vars:
            vars_lines = []
            dataframes = []
            for name, info in available_vars.items():
                if isinstance(info, dict):
                    type_str = info.get('type', 'Unknown')
                    if type_str == 'DataFrame':
                        dataframes.append(name)
                    details = ''
                    if 'columns' in info:
                        cols = info['columns']
                        if len(cols) > 10:
                            cols_str = str(cols[:10])[:-1] + ', ...]'
                        else:
                            cols_str = str(cols)
                        details = f' columns={cols_str}'
                    if 'shape' in info:
                        details += f" shape={info['shape']}"
                    intent_str = f''' intent="{info['intent']}"''' if 'intent' in info else ''
                    value_str = ''
                    if name in ('TASKS', 'COMPLETED_TASKS', '_VAR_INTENT_') and 'value' in info:
                        value_str = f" value={info['value']}"
                    vars_lines.append(f'- {name} ({type_str}){details}{intent_str}{value_str}')
                else:
                    vars_lines.append(f'- {name} ({info})')
            prompt_parts.append(f'AVAILABLE VARIABLES:\n' + '\n'.join(vars_lines))
            if dataframes and 'df' not in dataframes:
                if len(dataframes) == 1:
                    prompt_parts.append(f"\n⚠️ CRITICAL: The dataframe is named '{dataframes[0]}'. DO NOT use 'df'. Use '{dataframes[0]}' instead.")
                else:
                    prompt_parts.append(f"\n⚠️ CRITICAL: Available dataframes: {', '.join(dataframes)}. DO NOT use 'df' unless it is defined.")
            prompt_parts.append('⚠️ CRITICAL: These are the ONLY variables in memory. Data is NOT preloaded as table-name globals (e.g., IFFCO_INV_AI_CONS does NOT exist as a variable). Use ONLY the exact variable names listed above.')
            if 'FINAL_RESULT' in available_vars:
                prompt_parts.append('\n STOP — FINAL_RESULT IS ALREADY SET IN THE KERNEL. Your analysis is COMPLETE. You MUST return action: "done" immediately. Do NOT generate more code. Do NOT re-compute or re-set FINAL_RESULT. The answer has already been produced.')
            prompt_parts.append('')
        loaded_datasets = execution_context.get('loaded_datasets', [])
        if loaded_datasets:
            prompt_parts.extend(self._loaded_datasets_prompt(loaded_datasets))
        failed_iters = execution_context.get('failed_iterations', [])
        if failed_iters:
            prompt_parts.append('⚠️ FAILED ITERATIONS (these approaches ALREADY FAILED — learn from them):')
            for fi in failed_iters[-3:]:
                prompt_parts.append(f"  Iteration {fi['iteration']} FAILED:")
                prompt_parts.append(f"    Error: {fi['error'][:300]}")
                if fi.get('code_snippet'):
                    prompt_parts.append(f"    Attempted code (snippet): {fi['code_snippet']}")
            prompt_parts.append('  → Do NOT repeat these failed approaches. Fix the SPECIFIC error (e.g., wrong variable name, missing merge step) and continue from AVAILABLE VARIABLES.')
            prompt_parts.append('')
        completed = execution_context.get('completed_iterations', [])
        if completed:
            accomplished_lines = ['ACCOMPLISHED SO FAR (DO NOT REPEAT ANY OF THIS — all variables are alive in kernel):']
            for item in completed:
                iter_id = item.get('iteration', '?')
                reasoning_str = item.get('reasoning', '')
                accomplished_lines.append(f'  - Iteration {iter_id}: {reasoning_str}')
            current_dfs = [f"  - {name} ({info.get('type', '?')}, shape={info.get('shape', '?')})" for name, info in available_vars.items() if isinstance(info, dict) and info.get('type') == 'DataFrame']
            if current_dfs:
                accomplished_lines.append('  Current DataFrames in memory:')
                accomplished_lines.extend(current_dfs)
            accomplished_lines.append('  ⚠️ Write ONLY new incremental code. Do NOT re-load, re-import, or re-compute anything above.')
            prompt_parts.extend(accomplished_lines)
            prompt_parts.append('')
            prompt_parts.append('COMPLETED ITERATIONS (detailed):')
            for item in completed:
                iter_id = item.get('iteration', '?')
                reasoning_str = item.get('reasoning', '')
                thinking_str = item.get('thinking', '')
                code_preview = item.get('code', '')
                if code_preview and len(code_preview) > self.code_preview_max_chars:
                    code_preview = code_preview[:self.code_preview_max_chars] + '\n# ...[truncated]'
                output_preview = item.get('output', '')
                if output_preview and len(output_preview) > self.output_preview_max_chars:
                    output_preview = output_preview[:self.output_preview_max_chars] + '\n...[truncated]'
                prompt_parts.append(f'  --- Iteration {iter_id}: {reasoning_str} ---')
                if thinking_str:
                    prompt_parts.append(f'  Thinking: {thinking_str}')
                if code_preview:
                    prompt_parts.append(f'  Code:\n{code_preview}')
                if output_preview:
                    prompt_parts.append(f'  Output:\n{output_preview}')
            prompt_parts.append('')
        else:
            prompt_parts.append('No iterations completed yet — this is the first iteration.')
            prompt_parts.append('')
        warnings = execution_context.get('warnings', [])
        if warnings:
            zero_row_warnings = [w for w in warnings if 'ZERO_ROW' in w or 'became empty' in w]
            other_warnings = [w for w in warnings if w not in zero_row_warnings]
            if zero_row_warnings:
                prompt_parts.append('⚠️ CRITICAL — ZERO-ROW RESULTS DETECTED (you MUST address these):')
                for w in zero_row_warnings:
                    prompt_parts.append(f'  ❌ {w}')
                prompt_parts.append('  ACTION REQUIRED: Do NOT proceed with empty DataFrames. Re-examine filter columns and values. Try alternative columns (e.g., INV_ORG_ID instead of ORGANIZATION_ID). Print unique values to verify.')
                prompt_parts.append('')
            if other_warnings:
                prompt_parts.append('WARNINGS from prior iterations:')
                for w in other_warnings:
                    prompt_parts.append(f'  - {w}')
                prompt_parts.append('')
        remaining = self.max_iterations - iteration
        convergence_note = ''
        if remaining <= 2:
            convergence_note = f'⚠️ URGENT: Only {remaining} iteration(s) remaining. You MUST assemble FINAL_RESULT in this iteration. Use whatever data you have — a partial answer is better than no answer.'
        elif remaining <= self.max_iterations // 2:
            convergence_note = f'Note: {remaining} iterations remaining. If you have enough data, proceed to computation and FINAL_RESULT.'
        prompt_parts.extend([f'ITERATION: {iteration} / {self.max_iterations}', *([convergence_note] if convergence_note else []), '', 'YOUR TASK:', 'Based on the user query, plan guidance, completed iterations and their outputs,', 'decide what to do NEXT. You have two options:', '', '1. action="code" — Write Python code for the next logical step.', '2. action="done" — Declare the analysis complete (use ONLY when the user query is fully answered).', '', 'RULES:', "- ITERATION 1 PLANNING: In your very first execution step (Iteration 1), your `code` MUST ONLY initialize three Python objects: `TASKS = [...]`, `COMPLETED_TASKS = []`, and `_VAR_INTENT_ = {}`. Break down the user query into a detailed, tech-heavy `TASKS` list based on the PROVIDED ANALYST GUIDANCE. Every task MUST include the table names and specific columns/filters mentioned in the guidance (e.g., `[Step 1] Fetch TotalValue from table_name where OrderDate >= '2025-01-01'`). Do NOT include SQL, data loading, or analysis logic in Iteration 1 — planning only. Begin task execution from Iteration 2. IMPORTANT: The column names and data types are ALREADY provided in the Analyst Guidance above. Do NOT waste iterations inspecting schema, verifying columns, or probing data types — trust the provided schema and start querying data directly in Iteration 2.", "- INTENT REGISTRY: Starting Iteration 2, for every meaningful DataFrame/variable you create, add a 1-sentence entry to `_VAR_INTENT_` (e.g., `_VAR_INTENT_['df_sales'] = 'Filtered 2025 sales by region'`).", '- STATE PROGRESSION: After every iteration that completes a task, append the completed task string to `COMPLETED_TASKS` (e.g., `COMPLETED_TASKS.append(TASKS[0])`).', '- COMPLETION GATE: Do NOT set FINAL_RESULT until every entry in `TASKS` is also in `COMPLETED_TASKS`. Premature completion ruins accuracy.', '- Each iteration builds on previous ones. Variables from prior iterations are alive in memory.', '- Do NOT re-import libraries or re-load data that was already loaded.', '- Do NOT repeat code from prior iterations — reference existing variables.', '- Add print() statements to show intermediate and final results.', '- If this is the LAST step of the analysis, store your primary result in FINAL_RESULT.', '  Example: FINAL_RESULT = result_df  or  FINAL_RESULT = {"key": value}', "- Set FINAL_RESULT BEFORE declaring action='done'.", '', 'WORKFLOW GUIDANCE (adapt based on query complexity):', '- For SIMPLE queries (1-2 tables, clear columns): load → compute → FINAL_RESULT in 3-4 iterations.', '- For COMPLEX queries (3+ tables, joins needed): load → merge → compute → FINAL_RESULT in 5-6 iterations.', '- You MAY combine related operations in one cell (e.g., load + inspect, or aggregate + visualize).', "- Before any groupby/filter, quickly check the relevant column: print(df['col'].nunique()) or print(df['col'].unique()[:10])", "- If a merge/join produces 0 rows, investigate immediately — don't proceed with empty data.", '- MERGE VERIFICATION: After any pd.merge/join, immediately print:', '  1. Result shape vs input shapes (row explosion = wrong keys)', "  2. Check for '_x'/'_y' column suffixes (= overlapping non-key columns, likely wrong join keys)", '  3. Sample 2-3 rows to sanity-check the joined data', '', 'CODE QUALITY RULES:', '- Keep cells to 30-40 lines MAX.', "- NEVER re-import libraries or re-load data that's already in AVAILABLE VARIABLES.", "- NEVER reference variables that don't exist — check AVAILABLE VARIABLES above.", '- Every cell MUST end with print() showing what was produced.', "- If a previous cell produced a DataFrame, USE it — don't rebuild it.", '', *self._performance_rules_prompt(), '', 'FILTER VERIFICATION RULES:', "- After EVERY filter (.isin(), .query(), boolean indexing, pd.merge), IMMEDIATELY check: print(f'Filtered: {result.shape[0]} rows')", '- If 0 rows: DO NOT proceed. Instead:', '  1. Print unique values in BOTH source and target filter columns', '  2. Check if you used the wrong column (e.g., ORG_ID vs INV_ORG_ID)', '  3. Try alternative columns ending in _ID, _CODE, _KEY, _NUM', '- When using a value from table A to filter table B:', "  1. Print the lookup value: print(f'Lookup: {value}')", '  2. Verify the value EXISTS in the target column before filtering', '  3. Column names often DIFFER between tables for the same entity', '  4. Values may also differ: ORG_ID=81 does NOT mean ORGANIZATION_ID=81', '- NEVER silently accept 0 rows and proceed to the next step', '', 'RESPONSE FORMAT — output these lines in this EXACT order, plain text, no JSON, no markdown fences:', "ACTION: code        (use 'done' instead when the analysis is finished)", "REASONING: ≤6 words, creative step title (e.g. 'Pulling in the RFQ data', 'Linking RFQs to their organization', 'Spotting the top revenue drivers', 'Assembling the final picture'). No column/file/table names. No generic labels like Loading datasets or Merging datasets.", 'THINKING: 1-2 sentences — What data do I have, what do I still need, and what will this cell do?', 'CODE:', "<python code to execute on the lines that follow; write nothing after the code. Omit the CODE: line and everything after it when ACTION is 'done'>", '', 'RULES FOR THE RESPONSE FORMAT:', '- ACTION, REASONING and THINKING are each a SINGLE line.', "- Everything after the line 'CODE:' is taken verbatim as the Python cell — do NOT wrap it in quotes, JSON, or markdown fences."])
        full_prompt = '\n'.join(prompt_parts)
        system_prompt = (self.base_prompt or '') + '\n\nYou are a recursive data science agent. You observe outputs, decide the next step, and iterate until the analysis is complete. You MUST respond in the exact plain-text RESPONSE FORMAT described below — no JSON, no markdown fences, no extra commentary.'
        response = await self.llm_client.generate_completion(system_prompt=system_prompt, user_message=full_prompt, temperature=self.temperature)
        self._update_usage(response.get('usage'))
        content = (response.get('content') or '').strip()
        decision = self._parse_decision_response(content)
        return decision
    _CODE_SENTINEL_RE = re.compile('(?:\\A|\\n)[ \\t]*CODE:[ \\t]*\\r?\\n?')

    def _parse_decision_response(self, content: str) -> Dict[str, Any]:
        if content.startswith('```json'):
            content = content[len('```json'):].strip()
        if content.startswith('```'):
            content = content[3:].strip()
        if content.endswith('```'):
            content = content[:-3].strip()
        if content.lstrip().startswith('{'):
            try:
                parsed = json.loads(content)
                action = parsed.get('action', 'code')
                if action not in ('code', 'done'):
                    logger.warning(f"Unknown action '{action}', defaulting to 'code'")
                    action = 'code'
                return {'action': action, 'reasoning': parsed.get('reasoning', ''), 'thinking': parsed.get('thinking', ''), 'code': parsed.get('code', '')}
            except json.JSONDecodeError:
                pass
        return self._parse_plaintext_decision(content)

    def _parse_plaintext_decision(self, content: str) -> Dict[str, Any]:
        match = self._CODE_SENTINEL_RE.search(content)
        if match:
            header = content[:match.start()]
            code = content[match.end():].strip()
        else:
            header = content
            code = ''
        action, reasoning, thinking = self._parse_decision_header(header)
        if not action and (not match):
            logger.warning(f'Decision response had no ACTION/CODE header, treating as raw code: {content[:100]!r}')
            return {'action': 'code', 'reasoning': 'LLM returned raw code (no decision header)', 'thinking': '', 'code': content.strip()}
        if action not in ('code', 'done'):
            if action:
                logger.warning(f"Unknown action '{action}', defaulting to 'code'")
            action = 'code'
        return {'action': action, 'reasoning': reasoning, 'thinking': thinking, 'code': code}

    @staticmethod
    def _parse_decision_header(header: str) -> Tuple[str, str, str]:
        action = ''
        reasoning = ''
        thinking = ''
        for line in (header or '').splitlines():
            stripped = line.strip()
            low = stripped.lower()
            if low.startswith('action:'):
                action = stripped.split(':', 1)[1].strip().lower()
            elif low.startswith('reasoning:'):
                reasoning = stripped.split(':', 1)[1].strip()
            elif low.startswith('thinking:'):
                thinking = stripped.split(':', 1)[1].strip()
        return (action, reasoning, thinking)

    async def _regenerate_code_after_error(self, user_query: str, plan_guidance: str, execution_context: Dict[str, Any], iteration: int, failed_code: str, error: str, attempt: int) -> str:
        prompt_parts = []
        prompt_parts.append(f'USER QUERY: {user_query}')
        prompt_parts.append('')
        available_vars = execution_context.get('available_variables', {})
        if available_vars:
            vars_lines = []
            dataframes = []
            for name, info in available_vars.items():
                if isinstance(info, dict):
                    type_str = info.get('type', 'Unknown')
                    if type_str == 'DataFrame':
                        dataframes.append(name)
                    details = ''
                    if 'columns' in info:
                        cols_str = str(info['columns'][:10]) if len(info['columns']) > 10 else str(info['columns'])
                        details = f' columns={cols_str}'
                    if 'shape' in info:
                        details += f" shape={info['shape']}"
                    vars_lines.append(f'- {name} ({type_str}){details}')
                else:
                    vars_lines.append(f'- {name} ({info})')
            prompt_parts.append('AVAILABLE VARIABLES:\n' + '\n'.join(vars_lines))
            if dataframes and 'df' not in dataframes:
                prompt_parts.append(f"⚠️ CRITICAL: Use '{dataframes[0]}' NOT 'df'.")
            prompt_parts.append('⚠️ Data is NOT preloaded as table-name globals. Only variables listed in AVAILABLE VARIABLES exist.')
            prompt_parts.append('')
        file_schemas = execution_context.get('file_schemas', {})
        if file_schemas:
            prompt_parts.append('FILE SCHEMAS (EXACT column names — use ONLY these):')
            for fname, schema in file_schemas.items():
                prompt_parts.append(f"  {fname}: columns={schema.get('columns', [])}")
            prompt_parts.append('⚠️ Use ONLY these exact column names. Do NOT use names from the plan.')
            prompt_parts.append('')
        loaded_datasets = execution_context.get('loaded_datasets', [])
        if loaded_datasets:
            prompt_parts.append("LOADED DATASETS (CRITICAL: if you reload a file, use these EXACT absolute paths — NEVER use a bare filename like 'data.parquet', it WILL raise FileNotFoundError):")
            for ds in loaded_datasets:
                p = ds.get('path', '?')
                v = ds.get('variable', '?')
                fmt = ds.get('format', '?')
                prompt_parts.append(f"  - path='{p}'  variable={v}  format={fmt}")
            prompt_parts.append('')
        data_profile = execution_context.get('data_profile', {})
        if data_profile:
            prompt_parts.append('DATASET PROFILE:')
            for ds_name, prof in data_profile.items():
                prompt_parts.append(f"  {ds_name}: columns={prof.get('columns', [])}, shape={prof.get('shape', [])}")
            prompt_parts.append('')
        error_type, repair_hint, _ = self._classify_error(error)
        prompt_parts.append(f'FAILED CODE (attempt {attempt}):')
        prompt_parts.append(failed_code[:800])
        prompt_parts.append('')
        error_limit = 2000 if error_type == 'COLUMN_NOT_FOUND' else 800
        prompt_parts.append(f'ERROR [{error_type}]:')
        prompt_parts.append(error[:error_limit])
        prompt_parts.append('')
        prompt_parts.append(f'REQUIRED FIX: {repair_hint}')
        prompt_parts.append("Identify the SPECIFIC error and fix it. If the approach was correct but a variable/column name was wrong, fix ONLY the name. Do NOT reload data that's already in memory. Do NOT start over from scratch. Use the EXACT variable names from AVAILABLE VARIABLES above.")
        if error_type == 'MISSING_COLUMN' and file_schemas:
            import re as _re
            key_match = _re.search('KeyError:\\s*[\'\\"]([^\'\\"]+)[\'\\"]', error)
            if key_match:
                missing_col = key_match.group(1)
                for fname, schema in file_schemas.items():
                    if missing_col in schema.get('columns', []):
                        ds_path = schema.get('path', fname)
                        if self._is_gcs:
                            prompt_parts.append(f"""\n⚠️ RECOVERY HINT: Column '{missing_col}' EXISTS in {fname} but was not loaded.\nRELOAD the file with the missing column included:\n  df = query_parquet('{fname}', "SELECT *, '{missing_col}' FROM {{TABLE}}")\nOr load ALL columns: query_parquet('{fname}')""")
                        else:
                            prompt_parts.append(f"\n⚠️ RECOVERY HINT: Column '{missing_col}' EXISTS in {fname} but was not loaded.\nRELOAD the file with the missing column included:\n  df = pd.read_parquet('{ds_path}', columns=[...existing..., '{missing_col}'])\nOr load ALL columns for small tables: pd.read_parquet('{ds_path}')")
                        break
        lessons_text = self._cached_lessons_text
        if not lessons_text:
            try:
                raw_db = self._get_raw_db()
                if raw_db:
                    from services.agent_lesson_service import AgentLessonService
                    lesson_svc = AgentLessonService(raw_db)
                    schema_tables = [Path(f).stem for f in file_schemas.keys()] if file_schemas else None
                    lessons_text = await lesson_svc.format_lessons_for_prompt(self.client_id, tables=schema_tables, max_tokens=800)
            except Exception:
                pass
        if lessons_text:
            prompt_parts.append('LEARNED PATTERNS (from prior analyses — follow these):')
            prompt_parts.append(lessons_text)
        prompt_parts.append('')
        prompt_parts.append('Return ONLY corrected Python code. No explanations, no markdown fences.')
        full_prompt = '\n'.join(prompt_parts)
        temp = self.retry_temperatures[min(attempt, len(self.retry_temperatures) - 1)]
        code = await self._generate_code_for_step(full_prompt, temperature_override=temp)
        return code

    async def _initialize_kernel(self) -> None:
        if self.session_id:
            from util import session_kernel_store
            pending_task = session_kernel_store.pop_prewarm_task(self.session_id)
            if pending_task is not None and (not pending_task.done()):
                logger.info('Awaiting pre-warmed kernel for session=%s', self.session_id)
                try:
                    await asyncio.wait_for(asyncio.shield(pending_task), timeout=35.0)
                except Exception as exc:
                    logger.warning('Pre-warm task timed out or failed for session=%s: %s — falling back to fresh init', self.session_id, exc)
            entry = session_kernel_store.get_session_kernel(self.session_id)
            if entry is not None:
                self.kernel_manager = entry.kernel_manager
                self.mcp_client = entry.mcp_client
                self._stdio_context_manager = entry.stdio_context_manager
                self._mcp_context_manager = entry.mcp_context_manager
                self._session_owned = True
                logger.info('Session kernel reused for session=%s (pre-warmed or follow-up)', self.session_id)
                return
        try:
            self.kernel_manager = await get_kernel_manager(client_id=self.client_id, idle_timeout_minutes=self.idle_timeout_minutes, environment=self.db_credentials_env if self._is_live_db else None, use_docker=False)
            success = await self.kernel_manager.start()
            if not success:
                raise RuntimeError('Failed to start local Jupyter kernel')
            kernel_url = self.kernel_manager.get_connection_url()
            logger.info(f'Connecting MCP to local Jupyter kernel at {kernel_url}')
            base_cmd = MCP_SERVER_COMMAND.split()
            server_args = ['--jupyter-url', kernel_url, '--jupyter-token', '']
            full_args = base_cmd[1:] + server_args
            server_params = StdioServerParameters(command=base_cmd[0], args=full_args)
            self._stdio_context_manager = stdio_client(server_params)
            read_stream, write_stream = await self._stdio_context_manager.__aenter__()
            self._capture_stdio_process()
            self._mcp_context_manager = McpClient(read_stream, write_stream)
            self.mcp_client = await self._mcp_context_manager.__aenter__()
            logger.info('MCP kernel and client initialized successfully')
            if self.session_id:
                from util import session_kernel_store
                entry = session_kernel_store.SessionKernelEntry(kernel_manager=self.kernel_manager, mcp_client=self.mcp_client, stdio_context_manager=self._stdio_context_manager, mcp_context_manager=self._mcp_context_manager, session_id=self.session_id, client_id=self.client_id)
                session_kernel_store.set_session_kernel(entry)
                self._session_owned = True
        except Exception as e:
            logger.error(f'Failed to initialize MCP kernel: {e}')
            raise RuntimeError(f'MCP initialization failed: {e}') from e

    async def _clear_stale_kernel_sentinels(self) -> None:
        cleanup_code = "# ── Session kernel reset: clear stale sentinels ──\nfor _v_ in ['FINAL_RESULT', 'TASKS', 'COMPLETED_TASKS']:\n    if _v_ in globals():\n        del globals()[_v_]\ndel _v_\nprint('Session sentinels cleared')"
        try:
            result = await self._execute_code(cleanup_code)
            logger.info('Cleared stale kernel sentinels for session=%s | stdout=%s', self.session_id, (result.get('stdout') or '').strip())
        except Exception as exc:
            logger.warning('Failed to clear stale kernel sentinels for session=%s: %s', self.session_id, exc)

    async def _load_dataset_to_kernel(self, dataset_path: Optional[str], dataset_dict: Optional[Dict]) -> List[Dict[str, str]]:
        code = ''
        loaded_datasets: List[Dict[str, str]] = []
        if dataset_path:
            p = Path(dataset_path)
            if self.kernel_manager:
                self.kernel_manager.copy_file_to_container(dataset_path, dataset_path)
            fmt = 'parquet' if p.suffix.lower() == '.parquet' else 'csv'
            loaded_datasets.append({'path': str(p), 'variable': 'df', 'format': fmt})
        elif dataset_dict is not None:
            import pandas as pd
            if isinstance(dataset_dict, pd.DataFrame):
                df_str = dataset_dict.to_json(orient='split')
            else:
                df_str = json.dumps(dataset_dict)
            code = f"""\nimport pandas as pd\nimport json\ndf = pd.read_json(json.loads({repr(df_str)}), orient='split')\nprint(f"Loaded dataset with shape: {{df.shape}}")\nprint(f"Columns: {{df.columns.tolist()}}")\n"""
            loaded_datasets.append({'path': '<in-memory>', 'variable': 'df', 'format': 'dict'})
        else:
            from config.system_config import STORAGE_BACKEND
            try:
                if STORAGE_BACKEND == 'gcs':
                    data_prefix = storage_datasets_prefix(self.client_id, self.dataset_id)
                    bootstrap = _get_duckdb_bootstrap_code(self.client_id, self.dataset_id)
                    if bootstrap:
                        _bootstrap_result = await self._execute_code(bootstrap)
                        if _bootstrap_result.get('exception'):
                            raise RuntimeError(f"DuckDB GCS bootstrap failed: {_bootstrap_result['exception']}. Ensure the kernel image has duckdb with httpfs pre-installed (see Dockerfile.datascience).")
                    try:
                        from util.storage.backend import get_storage_backend
                        storage = get_storage_backend()
                        gcs_files = await storage.list_files(data_prefix)
                        parquet_files = [f for f in gcs_files if f.endswith('.parquet')]
                        if parquet_files:
                            for pf in parquet_files:
                                fname = pf.rsplit('/', 1)[-1] if '/' in pf else pf
                                safe_name = fname.replace(' ', '_').replace('-', '_').replace('.parquet', '')
                                loaded_datasets.append({'path': fname, 'variable': f'df_{safe_name}' if len(parquet_files) > 1 else 'df', 'format': 'parquet', 'gcs': True})
                        else:
                            loaded_datasets.append({'path': data_prefix, 'variable': 'df', 'format': 'parquet', 'gcs': True})
                    except Exception as e:
                        logger.warning(f'Failed to list GCS files: {e}')
                        loaded_datasets.append({'path': data_prefix, 'variable': 'df', 'format': 'parquet', 'gcs': True})
                    logger.info(f'GCS dataset prefix staged for LLM: {data_prefix}')
                else:
                    client_data_dir = assets_datasets_dir(self.client_id, self.dataset_id)
                    if client_data_dir.exists():
                        all_files = list(client_data_dir.glob('*.parquet')) + list(client_data_dir.glob('*.csv'))
                        if all_files:
                            for f in all_files:
                                fmt = 'parquet' if f.suffix.lower() == '.parquet' else 'csv'
                                safe_name = f.stem.replace(' ', '_').replace('-', '_').replace('.', '_')
                                loaded_datasets.append({'path': str(f), 'variable': safe_name, 'format': fmt})
                                logger.info(f'Staged dataset for LLM: {f.name} → {f}')
                        else:
                            logger.warning(f'No datasets found in {client_data_dir}')
                            if self.dataset_id:
                                raise FileNotFoundError(f'No dataset files found for client_id={self.client_id} dataset_id={self.dataset_id} under {client_data_dir}')
                    else:
                        logger.warning(f'Client dataset directory not found: {client_data_dir}')
                        if self.dataset_id:
                            raise FileNotFoundError(f'Dataset directory missing for client_id={self.client_id} dataset_id={self.dataset_id}: {client_data_dir}')
            except Exception as e:
                logger.error(f'Error discovering client datasets: {e}')
                if isinstance(e, FileNotFoundError):
                    raise
            return loaded_datasets
        if code:
            try:
                result = await self._execute_code(code)
                stdout = result.get('stdout', '')
                if 'TIMEOUT ERROR' in stdout or 'execution exceeded' in stdout.lower():
                    logger.error(f'Dataset loading TIMED OUT: {stdout[:200]}')
                elif result.get('exception'):
                    logger.error(f"Dataset loading failed: {result['exception']}")
                    logger.error(f"stderr: {result.get('stderr', '')}")
                else:
                    logger.info(f'Dataset loaded. Output: {stdout[:200]}')
            except Exception as e:
                logger.error(f'Failed to load dataset: {e}')
                raise
        return loaded_datasets

    def _build_step_code_prompt(self, step: Dict[str, Any], user_query: str, execution_context: Dict, attempt: int=0, last_error: Optional[str]=None) -> str:
        step_num = step['step_num']
        description = step['description']
        details = step.get('details', [])
        completed_steps = execution_context.get('completed_iterations', execution_context.get('completed_steps', []))
        available_vars = execution_context.get('available_variables', ['df'])
        loaded_datasets = execution_context.get('loaded_datasets', [])
        live_sql_mode = bool(execution_context.get('live_sql_mode'))
        prompt_parts = [f'USER QUERY: {user_query}', '']
        if live_sql_mode:
            db_type = (execution_context.get('db_type', '') or '').strip().lower()
            prompt_parts.append('LIVE SQL MODE: Use pd.read_sql(sql, conn) to query data.')
            if db_type in ('postgres', 'postgresql'):
                prompt_parts.append('- PostgreSQL: MUST double-quote ALL column names: "Order_Date", "P3_NSV", "BusinessMonth" etc.')
            prompt_parts.extend(['- Each table has DIFFERENT columns. Only use columns listed for that table in the Analyst Guidance.', '- NEVER use pd.read_parquet, file paths, glob, or local file operations.', ''])
        if not live_sql_mode and (not loaded_datasets) and (step_num == 1):
            try:
                client_data_dir = assets_datasets_dir(self.client_id, self.dataset_id)
                if client_data_dir.exists():
                    parquet_files = list(client_data_dir.glob('*.parquet'))
                    if parquet_files:
                        main_file = parquet_files[0]
                        container_path = str(main_file)
                        loaded_datasets = [{'variable': 'df', 'format': 'parquet', 'path': container_path}]
            except Exception:
                pass
        if loaded_datasets:
            is_gcs = any((ds.get('gcs') for ds in loaded_datasets))
            if is_gcs:
                prompt_parts.append('AVAILABLE DATASETS (stored in GCS — use query_parquet() to load):')
                for ds in loaded_datasets:
                    prompt_parts.append(f"  - '{ds['path']}' → load as: {ds['variable']} = query_parquet('{ds['path']}')")
                prompt_parts.append('')
                prompt_parts.append('⚠️ CRITICAL: Files are in cloud storage, NOT on the local filesystem.')
                prompt_parts.append('DO NOT use pd.read_parquet() or pd.read_csv() — they will fail with FileNotFoundError.')
                prompt_parts.append("ALWAYS use query_parquet('filename.parquet') which reads from GCS via DuckDB.")
                prompt_parts.append("Example: df = query_parquet('bom.parquet')")
                prompt_parts.append('For SQL: df = query_parquet(\'bom.parquet\', "SELECT col1, col2 FROM {TABLE} WHERE col1 > 10")')
                prompt_parts.append('')
            else:
                prompt_parts.append('LOADED DATASETS (already loaded in kernel — do NOT reload unless needed):')
                for ds in loaded_datasets:
                    prompt_parts.append(f"  - Variable '{ds['variable']}' = {ds['format']} file at: {ds['path']}")
                prompt_parts.append('')
                prompt_parts.append('IMPORTANT: Data files are parquet format. NEVER use pd.read_csv().')
                prompt_parts.append("If you must reload data, use: pd.read_parquet(r'<exact path shown above>')")
                prompt_parts.append('')
        elif step_num == 1 and (not available_vars):
            from config.system_config import STORAGE_BACKEND as _sb_fallback
            if _sb_fallback == 'gcs':
                prompt_parts.append('IMPORTANT: You are in Step 1 and no data is loaded yet.')
                prompt_parts.append('You have NO variables defined.')
                prompt_parts.append("Use query_parquet('filename.parquet') to load data from GCS.")
                prompt_parts.append('DO NOT use pd.read_parquet() — files are in cloud storage.')
                prompt_parts.append('')
            else:
                suggested_file = '<absolute_dataset_path>.parquet'
                try:
                    client_data_dir = assets_datasets_dir(self.client_id, self.dataset_id)
                    if client_data_dir.exists():
                        parquet_files = list(client_data_dir.glob('*.parquet'))
                        if parquet_files:
                            suggested_file = str(parquet_files[0])
                except Exception:
                    pass
                prompt_parts.append('IMPORTANT: You are in Step 1 and no data is loaded yet.')
                prompt_parts.append('You have NO variables defined.')
                prompt_parts.append(f'You MUST start your code by loading the dataset:')
                prompt_parts.append(f"df = pd.read_parquet(r'{suggested_file}')")
                prompt_parts.append('')
        prompt_parts.append(f'CURRENT STEP ({step_num}): {description}')
        if details:
            prompt_parts.append('Details:')
            for detail in details:
                prompt_parts.append(f'  - {detail}')
            prompt_parts.append('')
        if completed_steps:
            prompt_parts.append('COMPLETED STEPS (already executed — do NOT repeat this code):')
            for prev_step in completed_steps:
                prompt_parts.append(f"  {prev_step['iteration']}. {prev_step['reasoning']}")
                if prev_step.get('code'):
                    code_preview = prev_step['code']
                    if len(code_preview) > 600:
                        code_preview = code_preview[:600] + '\n# ... (truncated)'
                    prompt_parts.append(f'     Code executed:\n{code_preview}')
                vars_info = prev_step.get('variables')
                if vars_info:
                    if isinstance(vars_info, list):
                        v_str = ', '.join(vars_info)
                    elif isinstance(vars_info, dict):
                        v_str = ', '.join(vars_info.keys())
                    else:
                        v_str = str(vars_info)
                    prompt_parts.append(f'     Variables created: {v_str}')
                if prev_step.get('output'):
                    condensed = prev_step['output'][:500]
                    prompt_parts.append(f'     Output: {condensed}')
            prompt_parts.append('')
        warnings = execution_context.get('warnings', [])
        if warnings:
            prompt_parts.append('⚠️ WARNINGS FROM PREVIOUS STEPS:')
            for w in warnings[-3:]:
                prompt_parts.append(f'  - {w}')
            prompt_parts.append('')
        data_profile = execution_context.get('data_profile', {})
        if data_profile:
            prompt_parts.append('DATASET PROFILE (exact column names, dtypes, null structure):')
            for df_name, prof in data_profile.items():
                prompt_parts.append(f"  {df_name}: shape={prof.get('shape', '?')}")
                if prof.get('columns'):
                    prompt_parts.append(f"    columns = {prof['columns']}")
                if prof.get('dtypes'):
                    prompt_parts.append(f"    dtypes  = {prof['dtypes']}")
                if prof.get('null_counts'):
                    prompt_parts.append(f"    nulls   = {prof['null_counts']}")
                if prof.get('sample_row'):
                    prompt_parts.append(f"    sample  = {str(prof['sample_row'][0])[:200]}")
            prompt_parts.append('USE ONLY THESE EXACT COLUMN NAMES — case-sensitive, no variation.')
            prompt_parts.append('')
        dataframes = []
        if isinstance(available_vars, list):
            vars_list = available_vars
            vars_formatted = ', '.join(vars_list)
            dataframes = [v for v in vars_list if 'df' in v or 'data' in v]
        elif isinstance(available_vars, dict):
            lines = []
            for name, info in available_vars.items():
                if isinstance(info, dict):
                    type_str = info.get('type', 'Unknown')
                    if type_str == 'DataFrame':
                        dataframes.append(name)
                    details_str = ''
                    if 'columns' in info:
                        cols = info['columns']
                        if len(cols) > 10:
                            cols_str = str(cols[:10])[:-1] + ', ...]'
                        else:
                            cols_str = str(cols)
                        details_str = f' columns={cols_str}'
                    if 'shape' in info:
                        details_str += f" shape={info['shape']}"
                    lines.append(f'- {name} ({type_str}){details_str}')
                    if 'dtypes' in info:
                        dtype_str = str(info['dtypes'])[:200]
                        lines.append(f'  dtypes: {dtype_str}')
                    if 'sample' in info:
                        sample_str = str(info['sample'])[:300]
                        lines.append(f'  sample rows: {sample_str}')
                    if 'value' in info and type_str != 'DataFrame':
                        lines.append(f"  value: {info['value']}")
                else:
                    lines.append(f'- {name} ({info})')
            vars_formatted = '\n'.join(lines)
        else:
            vars_formatted = str(available_vars)
        prompt_parts.append(f'AVAILABLE VARIABLES (Use ONLY these):\n{vars_formatted}')
        prompt_parts.append('')
        if dataframes and 'df' not in dataframes:
            if len(dataframes) == 1:
                prompt_parts.append(f"⚠️  CRITICAL: The dataframe is named '{dataframes[0]}'.")
                prompt_parts.append(f"DO NOT use 'df'. Use '{dataframes[0]}' instead.")
                prompt_parts.append('')
            else:
                prompt_parts.append(f"⚠️  CRITICAL: Available dataframes are: {', '.join(dataframes)}.")
                prompt_parts.append(f"DO NOT use 'df' unless it is defined. Use the specific names listed above.")
                prompt_parts.append('')
        if attempt > 0 and last_error:
            error_type, repair_hint, _ = self._classify_error(last_error)
            prompt_parts.append(f'PREVIOUS ATTEMPT FAILED [{error_type}]:')
            prompt_parts.append(f'Error: {last_error[:400]}')
            prompt_parts.append('')
            prompt_parts.append(f'REQUIRED FIX: {repair_hint}')
            prompt_parts.append("Identify the SPECIFIC error and fix it. If the approach was correct but a variable/column name was wrong, fix ONLY the name. Do NOT reload data that's already in memory. Do NOT start over from scratch. Use the EXACT variable names from AVAILABLE VARIABLES above.")
            prompt_parts.append('')
        prompt_parts.extend(['INSTRUCTIONS:', f'Generate Python code ONLY for step {step_num}. Do NOT include code for other steps.', '', 'CRITICAL — DO NOT REPEAT PRIOR WORK:', '- All code from COMPLETED STEPS has ALREADY been executed. Variables from those steps are alive in memory.', '- Do NOT re-import libraries that were imported in prior steps (they are already available).', '- Do NOT reload data that was loaded in prior steps (use the existing variable).', '- Do NOT recalculate values that already exist in AVAILABLE VARIABLES.', '- Write ONLY the NEW code needed for THIS step. Assume all prior state exists.', '', 'CRITICAL VARIABLE RULES:', f'- Use ONLY the variables listed in AVAILABLE VARIABLES above.', '- Look at the column names provided for each DataFrame to find the data you need.', "- Do NOT invent new variable names (like 'forecast_df' or 'sales_data') if they don't exist.", "- If you need 'predicted revenue', look for a dataframe with a 'predicted_revenue' column.", '- Do NOT generate code that checks if variables exist and raises errors — just use the available variables directly.', '', 'CODE RULES:', '- Build on existing variables from previous steps', "- Create clearly named new variables for this step's output", '- In live SQL mode, query with pd.read_sql(sql, conn); do not use file operations.', '- Add print statements to show results', '- Keep code focused on this single step', '', 'MODEL SELECTION (MANDATORY for any forecasting/prediction/ML step):', '- Use ONLY: RandomForest or XGBoost (in that order of preference)', '- NEVER use LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, statsmodels OLS/GLM, or TensorFlow/Keras', '- Default to RandomForestRegressor/Classifier for most problems', '- Use xgboost.XGBRegressor/XGBClassifier for complex non-linear patterns or high cardinality features', '', 'Return ONLY executable Python code, no explanations.'])
        return '\n'.join(prompt_parts)

    async def _generate_code_for_step(self, prompt: str, temperature_override: Optional[float]=None) -> str:
        try:
            temp = temperature_override if temperature_override is not None else self.temperature
            perf_hint = '  * ALWAYS use query_parquet(\'filename.parquet\') to load data from GCS\n  * For column selection: query_parquet(\'file.parquet\', "SELECT col1, col2 FROM {TABLE}")\n  * Do NOT use pd.read_parquet() — files are in cloud storage, not local filesystem' if self._is_gcs else "  * ALWAYS select only the columns you need: pd.read_parquet(path, columns=['col1','col2'])"
            system_prompt = (self.base_prompt or '') + f'\n\nCODE GENERATION RULES (always apply):\n- Return ONLY Python code — no markdown fences, no explanations, no apologies, no prose whatsoever\n- If you cannot complete the task, still return Python code (e.g., a comment + print statement)\n- PERFORMANCE CRITICAL — loading large files:\n{perf_hint}\n  * Do NOT add .sample() — always process the FULL dataset for accurate results\n  * Only sample if the user EXPLICITLY asks for it in their query (e.g. "use a 10% sample to save time")\n- Use pandas, numpy, sklearn, plotly as needed\n- CRITICAL: Only use variables that are listed in AVAILABLE VARIABLES in the prompt\n- Do NOT assume variable names — use exactly what is listed as available\n- Do NOT generate variable-existence checks that raise errors\n- Add print statements to show key results\n- Code must be ready to execute in a Jupyter cell with NO surrounding text\n- FINAL RESULT RULE: In the LAST step of the analysis, store your primary output in a variable\n  named FINAL_RESULT (e.g. FINAL_RESULT = result_df  or  FINAL_RESULT = {{"count": 123, ...}}).\n  This variable is used as the authoritative final output of the analysis.'
            response_dict = await self.llm_client.generate_completion(system_prompt=system_prompt, user_message=prompt, temperature=temp)
            self._update_usage(response_dict.get('usage'))
            llm_error = response_dict.get('error')
            if llm_error and (not response_dict.get('content')):
                raise ValueError(f'LLM API error: {llm_error}')
            code = (response_dict.get('content') or '').strip()
            if code.startswith('```python'):
                code = code[len('```python'):].strip()
            if code.startswith('```'):
                code = code[3:].strip()
            if code.endswith('```'):
                code = code[:-3].strip()
            if self._is_likely_prose(code):
                logger.warning(f'LLM returned prose instead of Python code: {code[:150]!r}')
                raise ValueError(f'LLM returned explanatory text instead of Python code. You MUST return ONLY executable Python. First 100 chars: {code[:100]!r}')
            logger.debug(f'Generated code:\n{code}')
            return code
        except ValueError:
            raise
        except Exception as e:
            logger.error(f'Failed to generate code: {e}')
            raise ValueError(f'Code generation failed: {e}')

    async def _generate_diagnostic_code(self, failed_code: str, error: str, available_vars: Dict[str, Any]) -> Optional[str]:
        _, _, diagnostic_code = self._classify_error(error)
        return diagnostic_code

    async def _summarize_completed_steps(self, steps: List[Dict]) -> str:
        try:
            steps_text = '\n'.join((f"Step {s['iteration']}: {s['reasoning']} | Output: {str(s.get('output', ''))[:100]}" for s in steps))
            prompt = f'Summarize what was accomplished in these data analysis steps in 2 sentences max. Focus on what data was loaded, transformed, or computed, and any key variables created.\n\n{steps_text}'
            response = await self.llm_client.generate_completion(system_prompt='You are a concise technical summarizer. Reply with 1-2 sentences only.', user_message=prompt, temperature=0.0)
            return (response.get('content') or '').strip() or f"Completed steps {[s['iteration'] for s in steps]}."
        except Exception as e:
            logger.warning(f'Context compaction failed: {e}')
            return f"Completed steps {[s['iteration'] for s in steps]}."

    def _classify_error(self, error_str: str) -> Tuple[str, str, str]:
        e = error_str.lower()
        if any((x in e for x in ['nameerror', 'is not defined'])):
            return ('UNDEFINED_VARIABLE', "The variable doesn't exist. Check AVAILABLE VARIABLES — use the EXACT name listed.", "print('Available vars:', [k for k in globals() if not k.startswith('_')][:20])")
        if any((x in e for x in ['keyerror', 'not in index'])):
            return ('MISSING_COLUMN', 'Column name is wrong. Use EXACT column name from DATASET PROFILE (case-sensitive).', "for _v,_o in globals().items():\n    if hasattr(_o,'columns'): print(f'{_v} columns:',_o.columns.tolist())")
        if any((x in e for x in ['arrowtypeerror', 'could not convert'])):
            return ('ARROW_TYPE_ERROR', "PyArrow mixed-type error. Use pd.to_numeric(col, errors='coerce') or .astype(str) to normalize dtypes before operations.", "for _v,_o in globals().items():\n    if hasattr(_o,'dtypes'): print(f'{_v} dtypes:',_o.dtypes.to_dict())")
        if any((x in e for x in ['arrowinvalid', 'no match for fieldref'])):
            return ('COLUMN_NOT_FOUND', 'Column name does not exist in the file. The error message shows the ACTUAL columns available — read them carefully and use THOSE exact names. Check FILE SCHEMAS for the correct column names. Do NOT guess or use plan column names.', self._build_parquet_schema_diagnostic(error_str))
        if 'pyarrow' in e:
            return ('ARROW_TYPE_ERROR', "PyArrow error. Use pd.to_numeric(col, errors='coerce') or .astype(str) to normalize dtypes before operations.", "for _v,_o in globals().items():\n    if hasattr(_o,'dtypes'): print(f'{_v} dtypes:',_o.dtypes.to_dict())")
        if any((x in e for x in ['typeerror', 'unsupported operand', 'cannot convert'])):
            return ('TYPE_MISMATCH', 'Type mismatch. Check dtypes and add explicit .astype() before the operation.', "for _v,_o in globals().items():\n    if not callable(_o) and not _v.startswith('_'):\n        print(f'{_v}: {type(_o).__name__}')")
        if any((x in e for x in ['memoryerror', 'cannot allocate'])):
            return ('MEMORY_ERROR', 'Dataset too large. Select fewer columns with columns= parameter, or split the operation into smaller chunks.', "for _v,_o in globals().items():\n    if hasattr(_o,'shape'): print(f'{_v}.shape:',_o.shape)")
        if any((x in e for x in ['timed out', 'timeout'])):
            return ('TIMEOUT', 'Operation too slow. Use vectorized operations, not loops. Sample the data first.', "for _v,_o in globals().items():\n    if hasattr(_o,'shape'): print(f'{_v}.shape:',_o.shape)")
        if any((x in e for x in ['modulenotfounderror', 'importerror'])):
            return ('IMPORT_ERROR', 'Only available: pandas, numpy, sklearn, scipy, plotly, matplotlib, re, json, datetime, collections. Do not import other packages.', 'import pkg_resources; print([p.project_name for p in pkg_resources.working_set][:20])')
        if any((x in e for x in ['filenotfounderror', 'no such file'])):
            return ('FILE_NOT_FOUND', 'Use ONLY the exact file path from LOADED DATASETS. Copy it character-for-character.', "import os; print(os.listdir('.')[:20])")
        if 'zero_row_result' in e:
            return ('ZERO_ROW_RESULT', 'A filter or join produced 0 rows. The filter column or values are WRONG. Check the DIAGNOSTIC OUTPUT for actual unique values in candidate columns. Try a DIFFERENT column (e.g., INV_ORG_ID instead of ORGANIZATION_ID). Print unique values in both tables to find the correct mapping. Do NOT repeat the same filter — use fundamentally different columns.', "import pandas as _pd_\nfor _v,_o in globals().items():\n    if _v.startswith('_'): continue\n    if isinstance(_o, _pd_.DataFrame) and _o.shape[0] > 0:\n        _ids = [c for c in _o.columns if any(c.upper().endswith(s) for s in ('_ID','_CODE','_KEY','_NAME'))]\n        if _ids: print(f'{_v}: {[(c, _o[c].nunique()) for c in _ids[:5]]}')")
        return ('GENERIC_ERROR', 'Fix the specific error shown below. Read carefully and correct that exact line.', "print('Kernel vars:', [k for k in globals() if not k.startswith('_')][:20])")

    def _validate_code_syntax(self, code: str, available_vars: Dict[str, Any]) -> Optional[str]:
        import ast as _ast
        try:
            tree = _ast.parse(code)
        except SyntaxError as e:
            return f'SyntaxError line {e.lineno}: {e.msg}'
        first_line = code.strip().split('\n')[0].lower()
        if any((first_line.startswith(p) for p in ['i cannot', "i'm sorry", 'unfortunately', 'to solve', 'here is the', 'as a'])):
            return f'LLM returned prose instead of code: {code[:80]!r}'
        if available_vars:
            loaded_names: set = set()
            assigned_names: set = set()
            for node in _ast.walk(tree):
                if isinstance(node, _ast.Name):
                    if isinstance(node.ctx, _ast.Load):
                        loaded_names.add(node.id)
                    elif isinstance(node.ctx, (_ast.Store, _ast.Del)):
                        assigned_names.add(node.id)
            safe = set(available_vars.keys()) | assigned_names
            safe |= {'pd', 'np', 'plt', 'px', 'json', 'os', 're', 'datetime', 'print', 'len', 'range', 'list', 'dict', 'str', 'int', 'float', 'type', 'True', 'False', 'None', 'min', 'max', 'sum', 'zip', 'enumerate', 'llm_query', 'requests', 'sklearn', 'scipy', 'math', 'collections', 'FINAL_RESULT'}
            suspicious_df = {n for n in loaded_names - safe if n == 'df' or n.startswith(('df_', 'data_', 'result_', 'filtered_'))}
            if suspicious_df:
                return f'Undefined variable(s): {suspicious_df}. Available: {list(available_vars.keys())[:8]}'
        banned_models = {'LinearRegression', 'LogisticRegression', 'Ridge', 'Lasso', 'ElasticNet'}
        banned_modules = {'tensorflow', 'keras', 'tf'}
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ImportFrom):
                module = node.module or ''
                if any((module == m or module.startswith(f'{m}.') for m in banned_modules)):
                    return f"Banned module '{module}' detected. Use RandomForestRegressor/Classifier or xgboost instead."
                for alias in node.names:
                    name = alias.name
                    if name in banned_models:
                        return f"Banned model '{name}' detected. Use RandomForestRegressor/Classifier or xgboost instead."
            if isinstance(node, _ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name in banned_modules or any((name.startswith(f'{m}.') for m in banned_modules)):
                        return f"Banned module '{name}' detected. Use RandomForestRegressor/Classifier or xgboost instead."
            if isinstance(node, _ast.Call):
                func = node.func
                func_name = None
                if isinstance(func, _ast.Name):
                    func_name = func.id
                elif isinstance(func, _ast.Attribute):
                    func_name = func.attr
                if func_name and func_name in banned_models:
                    return f"Banned model '{func_name}' detected. Use RandomForestRegressor/Classifier or xgboost instead."
        return None

    def _detect_doom_loop(self, current_code: str) -> bool:
        if len(self._recent_failed_codes) < self.doom_loop_threshold:
            return False
        last_n = self._recent_failed_codes[-self.doom_loop_threshold:]
        for prev in last_n:
            ratio = difflib.SequenceMatcher(None, current_code.strip(), prev.strip()).ratio()
            if ratio < 0.92:
                return False
        return True

    async def _probe_dataset_profile(self) -> Dict[str, Any]:
        code = '\nimport json as _j_, pandas as _p_\n_profile_ = {}\nfor _name_ in list(globals().keys()):\n    if _name_.startswith(\'_\'): continue\n    try:\n        _obj_ = eval(_name_)\n        if not isinstance(_obj_, _p_.DataFrame): continue\n        _pr_ = {\n            "shape": list(_obj_.shape),\n            "columns": _obj_.columns.tolist(),\n            "dtypes": {c: str(d) for c, d in _obj_.dtypes.items()},\n            "null_counts": {c: int(n) for c, n in _obj_.isnull().sum().items() if n > 0},\n            "sample_row": _obj_.head(1).to_dict(orient=\'records\'),\n        }\n        _num_cols_ = _obj_.select_dtypes(include=\'number\').columns.tolist()[:4]\n        if _num_cols_:\n            _pr_["numeric_ranges"] = {\n                c: {"min": float(_obj_[c].min()), "max": float(_obj_[c].max())}\n                for c in _num_cols_\n            }\n        _str_cols_ = [c for c in _obj_.columns if _obj_[c].dtype == \'object\'][:8]\n        if _str_cols_:\n            _sv_ = {}\n            for c in _str_cols_:\n                _vc_ = _obj_[c].dropna().value_counts().head(15)\n                _sv_[c] = {\n                    "unique_count": int(_obj_[c].nunique()),\n                    "top_values": [str(v) for v in _vc_.index.tolist()],\n                    "top_counts": [int(ct) for ct in _vc_.values.tolist()],\n                }\n            _pr_["string_values"] = _sv_\n        _profile_[_name_] = _pr_\n    except Exception:\n        pass\nprint(\'<PROFILE>\' + _j_.dumps(_profile_) + \'</PROFILE>\')\n'
        result = await self._execute_code(code)
        stdout = result.get('stdout', '')
        if '<PROFILE>' in stdout and '</PROFILE>' in stdout:
            try:
                start = stdout.find('<PROFILE>') + len('<PROFILE>')
                end = stdout.find('</PROFILE>')
                profile = json.loads(stdout[start:end])
                logger.info(f'Dataset profile captured: {list(profile.keys())}')
                return profile
            except Exception as e:
                logger.warning(f'Profile parsing failed: {e}')
        return {}

    async def _probe_parquet_schemas(self, loaded_datasets: List[Dict[str, str]], dataset_path: Optional[str]=None) -> Dict[str, Any]:
        from config.system_config import STORAGE_BACKEND
        schemas: Dict[str, Any] = {}
        if STORAGE_BACKEND == 'gcs':
            return await self._probe_parquet_schemas_gcs(loaded_datasets)
        file_paths: List[str] = []
        for ds in loaded_datasets or []:
            p = ds.get('path', '')
            if p and p != '<in-memory>':
                file_paths.append(p)
        if not file_paths:
            try:
                client_data_dir = assets_datasets_dir(self.client_id, self.dataset_id)
                if client_data_dir.exists():
                    file_paths.extend((str(f) for f in client_data_dir.glob('*.parquet')))
                    file_paths.extend((str(f) for f in client_data_dir.glob('*.csv')))
            except Exception as e:
                logger.warning(f'Schema discovery failed: {e}')
        if dataset_path and dataset_path not in file_paths:
            file_paths.append(dataset_path)
        for fpath in file_paths:
            try:
                fname = Path(fpath).name
                if self._planned_tables and (not self._matches_planned_tables(fname)):
                    continue
                if fpath.endswith('.parquet'):
                    import pyarrow.parquet as pq
                    pf = pq.ParquetFile(fpath)
                    schema = pf.schema_arrow
                    num_rows = pf.metadata.num_rows
                    schemas[fname] = {'path': fpath, 'columns': [f.name for f in schema], 'types': {f.name: str(f.type) for f in schema}, 'num_rows': num_rows}
                elif fpath.endswith('.csv'):
                    import pandas as _pd
                    header = _pd.read_csv(fpath, nrows=0)
                    schemas[fname] = {'path': fpath, 'columns': header.columns.tolist(), 'types': {c: 'unknown' for c in header.columns}}
                logger.info(f"Schema read: {fname} → {len(schemas.get(fname, {}).get('columns', []))} columns")
            except Exception as e:
                logger.warning(f'Failed to read schema for {fpath}: {e}')
        return schemas

    async def _probe_parquet_schemas_gcs(self, loaded_datasets: List[Dict[str, str]]) -> Dict[str, Any]:
        from config.system_config import GCS_BUCKET
        schemas: Dict[str, Any] = {}
        try:
            from util.storage.backend import get_storage_backend
            storage = get_storage_backend()
            data_prefix = storage_datasets_prefix(self.client_id, self.dataset_id)
            gcs_files = await storage.list_files(data_prefix)
            parquet_files = [f for f in gcs_files if f.endswith('.parquet')]
        except Exception as e:
            logger.warning(f'GCS schema probe: failed to list files: {e}')
            return schemas
        if not parquet_files:
            return schemas
        for gcs_path in parquet_files:
            fname = gcs_path.rsplit('/', 1)[-1] if '/' in gcs_path else gcs_path
            if self._planned_tables and (not self._matches_planned_tables(fname)):
                continue
            try:
                s3_uri = f's3://{GCS_BUCKET}/{gcs_path}'
                probe_code = f"""import json\n_schema_result = _coresight_conn.execute("DESCRIBE SELECT * FROM read_parquet('{s3_uri}') LIMIT 0").fetchdf()\n_row_count = _coresight_conn.execute("SELECT count(*) as cnt FROM read_parquet('{s3_uri}')").fetchone()[0]\nprint(json.dumps({{'columns': _schema_result['column_name'].tolist(),'types': dict(zip(_schema_result['column_name'], _schema_result['column_type'])),'num_rows': _row_count}}))"""
                result = await self._execute_code(probe_code)
                if result and (not result.get('error')):
                    output = result.get('output', '').strip()
                    if output:
                        import json
                        schema_data = json.loads(output.split('\n')[-1])
                        schemas[fname] = {'path': fname, 'columns': schema_data['columns'], 'types': schema_data['types'], 'num_rows': schema_data['num_rows']}
                        logger.info(f"GCS schema read: {fname} → {len(schema_data['columns'])} columns")
            except Exception as e:
                logger.warning(f'Failed to read GCS schema for {fname}: {e}')
        return schemas

    def _build_parquet_schema_diagnostic(self, error_str: str) -> str:
        import re
        path_match = re.search('[\'\\"]([^\'\\"]+\\.parquet)[\'\\"]', error_str)
        if path_match:
            fpath = path_match.group(1)
            return f"import pyarrow.parquet as pq\n_s_ = pq.read_schema(r'{fpath}')\nprint('ACTUAL COLUMNS:', [f.name for f in _s_])\nprint('ACTUAL TYPES:', {{f.name: str(f.type) for f in _s_}})"
        return "for _v,_o in globals().items():\n    if hasattr(_o,'columns'): print(f'{_v} columns:',_o.columns.tolist())"

    async def _validate_step_output(self, step: Dict, new_vars: Dict[str, Any], prev_vars: Dict[str, Any]) -> Tuple[bool, str]:
        issues = []
        for var_name, var_info in new_vars.items():
            if isinstance(var_info, dict) and var_info.get('type') == 'DataFrame':
                shape = var_info.get('shape', [1, 1])
                if shape[0] == 0:
                    prev_info = prev_vars.get(var_name, {})
                    prev_shape = prev_info.get('shape', [1, 1]) if isinstance(prev_info, dict) else [1, 1]
                    if prev_shape[0] > 0:
                        issues.append(f'{var_name} became empty (was {prev_shape[0]} rows, now 0)')
        if issues:
            return (False, '; '.join(issues))
        return (True, '')

    def _detect_row_explosion(self, new_vars: Dict[str, Any], prev_vars: Dict[str, Any]) -> List[str]:
        warnings = []
        max_prev_rows = 0
        for info in prev_vars.values():
            if isinstance(info, dict) and info.get('type') == 'DataFrame':
                rows = info.get('shape', [0, 0])[0]
                if rows > max_prev_rows:
                    max_prev_rows = rows
        if max_prev_rows == 0:
            return warnings
        for var_name, var_info in new_vars.items():
            if not isinstance(var_info, dict) or var_info.get('type') != 'DataFrame':
                continue
            if var_name in prev_vars:
                continue
            new_rows = var_info.get('shape', [0, 0])[0]
            if new_rows > max_prev_rows * 5:
                warnings.append(f"ROW_EXPLOSION: '{var_name}' has {new_rows:,} rows but the largest input DataFrame had only {max_prev_rows:,} rows ({new_rows / max_prev_rows:.1f}x). This may indicate a cartesian join from wrong merge keys. Verify the merge was correct before proceeding.")
        return warnings

    def _generate_zero_row_diagnostic(self, validation_issue: str, available_vars: Dict[str, Any]) -> Optional[str]:
        import re as _re
        match = _re.match('(\\w+) became empty', validation_issue)
        empty_var = match.group(1) if match else None
        code_lines = ['import pandas as _pd_diag_', '_diag_lines_ = []']
        if empty_var:
            code_lines.append(f"if '{empty_var}' in dir() and hasattr({empty_var}, 'columns'):\n    _diag_lines_.append(f'EMPTY DF: {empty_var} columns={{list({empty_var}.columns)}}')")
        code_lines.extend(['for _vn_ in sorted(globals().keys()):', "    if _vn_.startswith('_'): continue", '    _vobj_ = globals()[_vn_]', '    if not isinstance(_vobj_, _pd_diag_.DataFrame): continue', '    if _vobj_.shape[0] == 0: continue', "    _id_cols_ = [c for c in _vobj_.columns if any(c.upper().endswith(s) for s in ('_ID', '_CODE', '_KEY', '_NUM', '_NAME'))]", '    if _id_cols_:', "        _diag_lines_.append(f'\\n{_vn_} ({_vobj_.shape[0]} rows):')", '        for _c_ in _id_cols_[:8]:', '            _uvals_ = _vobj_[_c_].dropna().unique()', "            _diag_lines_.append(f'  {_c_}: {len(_uvals_)} unique, sample={list(_uvals_[:10])}')", "print('\\n'.join(_diag_lines_))"])
        return '\n'.join(code_lines)

    async def _decompose_failed_step(self, step: Dict, last_error: str, execution_context: Dict) -> List[Dict]:
        available_vars = list(execution_context.get('available_variables', {}).keys())
        prompt = f'''A data analysis step failed after all retries:\n\nSTEP: {step.get('description', '')}\nLAST ERROR: {last_error[:300]}\nAVAILABLE VARIABLES: {available_vars[:10]}\n\nDecompose into exactly 2 simpler sequential sub-steps that together accomplish the same goal.\nSub-step 1 should validate/prepare data. Sub-step 2 should do the main computation.\n\nReturn ONLY a valid JSON array, no markdown, no explanation:\n[\n  {{"step_num": "{step['step_num']}a", "description": "...", "details": ["..."]}},\n  {{"step_num": "{step['step_num']}b", "description": "...", "details": ["..."]}}\n]'''
        try:
            resp = await self.llm_client.generate_completion(system_prompt='You are a data science task decomposer. Return only valid JSON.', user_message=prompt, temperature=0.2, max_tokens=400)
            content = (resp.get('content') or '').strip()
            if content.startswith('```'):
                content = content[content.find('['):]
            if '[' in content and ']' in content:
                sub_steps = json.loads(content[content.find('['):content.rfind(']') + 1])
                if isinstance(sub_steps, list) and len(sub_steps) == 2:
                    logger.info(f"Decomposed step {step['step_num']} → {[s['step_num'] for s in sub_steps]}")
                    return sub_steps
        except Exception as e:
            logger.warning(f'Step decomposition failed: {e}')
        return []

    async def _inject_llm_query_helper(self) -> None:
        if isinstance(self.kernel_manager, DockerKernelManager):
            server_host = 'host.docker.internal'
        else:
            server_host = '127.0.0.1'
        server_port = int(os.environ.get('BACKEND_PORT', '8024'))
        injection_code = f"""\nimport requests as _coresight_requests_\n_CORESIGHT_LLM_URL_ = "http://{server_host}:{server_port}/api/agents/internal/llm-query"\n_CORESIGHT_CLIENT_ID_ = "{self.client_id}"\n\ndef llm_query(question, context="", model="fast"):\n    '''\n    Call the CoreSight LLM for semantic reasoning on data.\n    Use ONLY for tasks pandas cannot handle: text classification, entity extraction,\n    answer verification.\n\n    Args:\n        question (str): What you want to know\n        context (str): Data snippet to reason over (auto-truncated to 4000 chars)\n        model (str): "fast" (default) or "smart" (more capable, slower)\n\n    Returns:\n        str: The LLM answer\n\n    Example:\n        df["category"] = df["desc"].apply(\n            lambda x: llm_query("Classify as fast-moving or slow-moving:", x))\n        answer = llm_query("What is the main finding?", str(result_df.head()))\n    '''\n    try:\n        resp = _coresight_requests_.post(\n            _CORESIGHT_LLM_URL_,\n            json={{"question": str(question), "context": str(context)[:4000],\n                   "client_id": _CORESIGHT_CLIENT_ID_, "model": model}},\n            timeout=45\n        )\n        if resp.status_code == 200:\n            return resp.json().get("answer", "[no answer]")\n        return f"[llm_query error {{resp.status_code}}]"\n    except Exception as _e_:\n        return f"[llm_query unavailable: {{_e_}}]"\n\nprint("llm_query() ready")\n"""
        result = await self._execute_code(injection_code)
        if 'llm_query() ready' in result.get('stdout', ''):
            logger.info('llm_query() injected successfully into kernel')
        else:
            logger.warning(f"llm_query() injection may have failed: {result.get('stdout', '')[:100]}")

    async def _inject_sql_query_helpers(self) -> None:
        if not self._is_live_db:
            return
        import base64
        import json as _json
        creds_json = _json.dumps(self.db_credentials_env)
        creds_b64 = base64.b64encode(creds_json.encode('utf-8')).decode('utf-8')
        injection_code = f"""\nimport atexit\nimport json\nimport base64\nimport os\nimport tempfile\nimport pandas as pd\nimport numpy as np\n\ntry:\n    from sshtunnel import SSHTunnelForwarder\nexcept ImportError:\n    SSHTunnelForwarder = None\n\n_ssh_forwarder = None\n_ssh_key_path = None\n\ndef _cleanup():\n    global _ssh_forwarder, _ssh_key_path\n    if _ssh_forwarder is not None:\n        try:\n            _ssh_forwarder.stop()\n        except Exception:\n            pass\n        _ssh_forwarder = None\n    if _ssh_key_path and os.path.exists(_ssh_key_path):\n        try:\n            os.unlink(_ssh_key_path)\n        except OSError:\n            pass\n        _ssh_key_path = None\n\natexit.register(_cleanup)\n\ndef _connect_db():\n    global _ssh_forwarder, _ssh_key_path\n    _env = json.loads(base64.b64decode('{creds_b64}').decode('utf-8'))\n\n    db_type = _env.get('CS_DB_TYPE', 'postgres')\n    host = _env.get('CS_DB_HOST', '')\n    port = int(_env.get('CS_DB_PORT') or 5432)\n    db = _env.get('CS_DB_NAME', '')\n    user = _env.get('CS_DB_USER', '')\n    pwd = _env.get('CS_DB_PASSWORD', '')\n\n    # SSH tunnel setup\n    ssh_cfg = json.loads(_env.get('CS_SSH_TUNNEL', '{{}}') or '{{}}')\n    if ssh_cfg.get('enabled') and SSHTunnelForwarder is not None:\n        if _ssh_forwarder is not None:\n            try:\n                if _ssh_forwarder.is_active:\n                    host, port = '127.0.0.1', int(_ssh_forwarder.local_bind_port)\n                else:\n                    _ssh_forwarder = None\n            except Exception:\n                _ssh_forwarder = None\n\n        if _ssh_forwarder is None:\n            tunnel_kwargs = {{\n                'ssh_address_or_host': (ssh_cfg.get('host', ''), int(ssh_cfg.get('port') or 22)),\n                'ssh_username': ssh_cfg.get('username', ''),\n                'remote_bind_address': (host, port),\n                'local_bind_address': ('127.0.0.1', 0),\n                'set_keepalive': 30.0,\n            }}\n            if ssh_cfg.get('auth_method') == 'private_key':\n                key_content = ssh_cfg.get('private_key_content', '')\n                if not key_content:\n                    raise ValueError('SSH private key content is missing')\n                pem_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.pem', delete=False)\n                pem_file.write(key_content)\n                pem_file.flush()\n                pem_file.close()\n                _ssh_key_path = pem_file.name\n                tunnel_kwargs['ssh_pkey'] = _ssh_key_path\n                if ssh_cfg.get('private_key_passphrase'):\n                    tunnel_kwargs['ssh_private_key_password'] = ssh_cfg.get('private_key_passphrase')\n            else:\n                tunnel_kwargs['ssh_password'] = ssh_cfg.get('password', '')\n            _ssh_forwarder = SSHTunnelForwarder(**tunnel_kwargs)\n            _ssh_forwarder.start()\n            host, port = '127.0.0.1', int(_ssh_forwarder.local_bind_port)\n\n    # Connect with the appropriate raw DBAPI driver\n    if db_type in ('postgres', 'postgresql'):\n        import psycopg2\n        c = psycopg2.connect(\n            host=host, port=port, dbname=db, user=user, password=pwd,\n            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5,\n        )\n        c.autocommit = True\n    elif db_type in ('sqlserver', 'mssql'):\n        import pymssql\n        c = pymssql.connect(server=host, port=port, database=db, user=user, password=pwd)\n    elif db_type == 'mysql':\n        import pymysql\n        c = pymysql.connect(host=host, port=int(port), database=db, user=user, password=pwd)\n    else:\n        raise ValueError(f"Unsupported db_type: {{db_type}}")\n    return c\n\nconn = _connect_db()\n\nprint("conn ready")\n"""
        result = await self._execute_code(injection_code)
        stdout = result.get('stdout', '')
        if 'conn ready' in stdout:
            logger.info('DB connection injected into kernel (raw DBAPI, db_type=%s)', self.db_credentials_env.get('CS_DB_TYPE', ''))
            return
        detail = result.get('exception') or result.get('stderr') or stdout
        raise RuntimeError(f'DB connection injection failed: {detail}')

    async def _check_final_result_in_kernel(self) -> bool:
        try:
            result = await self._execute_code("print('__FR_YES__' if 'FINAL_RESULT' in dir() else '__FR_NO__')")
            return '__FR_YES__' in result.get('stdout', '')
        except Exception:
            return False

    async def _get_kernel_variables(self) -> Dict[str, Any]:
        try:
            code = '\nimport json as _json_\nimport pandas as _pd_\n_exclude_ = {\'_json_\', \'_pd_\', \'_vars_\', \'_name_\', \'_exclude_\', \'_obj_\', \'_info_\', \'In\', \'Out\', \'get_ipython\', \'exit\', \'quit\', \'open\', \'_intent_dict_\'}\n_vars_ = {}\n_intent_dict_ = globals().get(\'_VAR_INTENT_\', {})\nfor _name_ in list(globals().keys()):\n    if _name_.startswith(\'_\'):\n        continue\n    if _name_ in _exclude_:\n        continue\n    try:\n        _obj_ = eval(_name_)\n        _info_ = {"type": str(type(_obj_).__name__)}\n        if isinstance(_obj_, _pd_.DataFrame):\n            _info_["columns"] = _obj_.columns.tolist()\n            _info_["shape"] = list(_obj_.shape)\n            # RLM enhancement: include dtypes and 2-row sample for richer context\n            try:\n                _info_["dtypes"] = {col: str(dtype) for col, dtype in _obj_.dtypes.items()}\n            except Exception:\n                pass\n            try:\n                _info_["sample"] = _obj_.head(2).to_dict(orient=\'records\')\n            except Exception:\n                pass\n        elif not callable(_obj_):\n            # Capture scalar/string values for non-callable non-DataFrame vars.\n            # TASKS / COMPLETED_TASKS / _VAR_INTENT_ get a larger cap so the\n            # full todo list survives into the next prompt round-trip.\n            try:\n                if _name_ in (\'TASKS\', \'COMPLETED_TASKS\', \'_VAR_INTENT_\'):\n                    _info_["value"] = repr(_obj_)[:5000]\n                else:\n                    _info_["value"] = repr(_obj_)[:200]\n            except Exception:\n                pass\n        # Inject explicit LLM intent if it documented why it created this variable\n        if isinstance(_intent_dict_, dict) and _name_ in _intent_dict_:\n            try:\n                _info_["intent"] = str(_intent_dict_[_name_])\n            except Exception:\n                pass\n        if not callable(_obj_) or _name_[0].isupper():\n            _vars_[_name_] = _info_\n    except Exception:\n        pass\nprint(_json_.dumps(_vars_, default=str))\n'
            result = await self._execute_code(code)
            if result.get('stdout'):
                import json
                stdout = result['stdout'].strip()
                lines = stdout.split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict):
                            return parsed
                        continue
                    except (json.JSONDecodeError, ValueError):
                        try:
                            import ast
                            parsed = ast.literal_eval(line)
                            if isinstance(parsed, dict):
                                return parsed
                        except (ValueError, SyntaxError):
                            pass
                        continue
            return {}
        except Exception as e:
            logger.warning(f'Could not get kernel variables: {e}')
            return {}

    def _is_likely_prose(self, text: str) -> bool:
        if not text:
            return True
        first_line = text.strip().split('\n')[0].strip().lower()
        prose_starters = ('i am', "i'm", 'i cannot', "i can't", 'i will ', "i'll ", 'the environment', 'unfortunately', 'to complete', 'here is', "here's", 'please note', 'note that', 'as an ai', 'due to', 'based on', 'the previous', 'i need to', 'i will need', 'i understand', 'it seems', 'it appears', 'this step', 'the code')
        if any((first_line.startswith(p) for p in prose_starters)):
            return True
        python_indicators = ['import ', 'pd.', 'df', 'print(', ' = ', 'def ', 'for ', 'if ', '# ', '.read_', 'np.', 'sklearn']
        return not any((ind in text for ind in python_indicators))

    def _stdout_contains_error(self, stdout: str) -> bool:
        if not stdout:
            return False
        error_indicators = ['Traceback (most recent call last)', 'ArrowInvalid:', 'NameError:', 'KeyError:', 'TypeError:', 'ValueError:', 'FileNotFoundError:', 'ModuleNotFoundError:', 'ImportError:', 'MemoryError:', 'AttributeError:', 'IndexError:', 'ZeroDivisionError:']
        return any((indicator in stdout for indicator in error_indicators))

    def _extract_error_from_stdout(self, stdout: str) -> str:
        lines = stdout.strip().split('\n')
        for i in range(len(lines) - 1, -1, -1):
            if 'Traceback' in lines[i]:
                return '\n'.join(lines[i:])
        return '\n'.join(lines[-5:])

    @traceable(name='coder_generate_code')
    async def _generate_code(self, user_query: str, context: Dict) -> str:
        iteration = context.get('iteration', 1)
        context_prompt = self._build_context_prompt(context)
        prompt = f"\n                        {self.base_prompt}\n\n                        User Query: {user_query}\n\n                        Iteration: {iteration}/{self.max_iterations}\n\n                        {context_prompt}\n\n                        Generate Python code to advance toward answering the user's query.\n                        The code will execute in a Jupyter environment with pandas, numpy, matplotlib, and plotly available.\n\n                        Requirements:\n                        1. Use clean, commented code\n                        2. Handle errors gracefully\n                        3. Print clear output about what was done\n                        4. Store results in variables for next iteration\n                        5. If generating visualizations, save them and describe findings\n                        6. Be specific and focused on the query\n\n                        Return ONLY the Python code, no explanations:\n                 "
        response = await self.llm_client.generate_completion(system_prompt=self.base_prompt, user_message=prompt, temperature=self.temperature, max_tokens=2000)
        code = response.get('content', '').strip()
        if code.startswith('```python'):
            code = code[9:]
        if code.startswith('```'):
            code = code[3:]
        if code.endswith('```'):
            code = code[:-3]
        return code.strip()

    @traceable(name='coder_execute_code')
    async def _execute_code(self, code: str) -> Dict[str, Any]:
        result = {'stdout': '', 'stderr': '', 'exception': None, 'variables': {}}
        sql_calls = re.findall('pd\\.read_sql\\s*\\(\\s*[\'\\"]+(.*?)[\'\\"]+\\s*,', code, re.DOTALL)
        if sql_calls:
            try:
                from util.sql_validator import SQLValidatorFactory
                db_type = (self.db_credentials_env or {}).get('CS_DB_TYPE', '') or (self.db_credentials_env or {}).get('db_type', 'mssql')
                validator = SQLValidatorFactory.get_validator(db_type)
                for sql in sql_calls:
                    val_errors = validator.validate(sql)
                    if val_errors:
                        logger.warning(f'SQL Validation triggered: {val_errors}')
                        result['exception'] = 'SQL_SYNTAX_WARNING: Your code contains a potential SQL syntax error. ' + ' '.join(val_errors)
                        return result
            except Exception as _sql_val_err:
                logger.debug(f'SQL pre-validator skipped: {_sql_val_err}')
        if not self.mcp_client:
            result['exception'] = 'MCP client not initialized'
            logger.error('Cannot execute code: MCP client not initialized')
            return result
        try:
            if self.kernel_manager:
                self.kernel_manager.update_activity()
            logger.debug(f'Executing code via MCP (timeout={self.timeout_per_execution}s): {code[:100]}...')
            mcp_result = await self.mcp_client.call_tool(name='execute_code', arguments={'code': code, 'timeout': self.timeout_per_execution}, timeout_seconds=self.timeout_per_execution)
            if mcp_result is None:
                result['stdout'] = ''
            elif isinstance(mcp_result, dict):
                if 'result' in mcp_result:
                    val = mcp_result['result']
                    if isinstance(val, list):
                        result['stdout'] = '\n'.join((str(item) for item in val))
                    else:
                        result['stdout'] = str(val)
                elif 'output' in mcp_result:
                    result['stdout'] = str(mcp_result['output'])
                elif 'stdout' in mcp_result:
                    result['stdout'] = str(mcp_result['stdout'])
                else:
                    result['stdout'] = str(mcp_result)
            elif isinstance(mcp_result, list):
                result['stdout'] = '\n'.join((str(item) for item in mcp_result))
            else:
                result['stdout'] = str(mcp_result)
            logger.debug(f"Code execution successful, output length: {len(result['stdout'])}")
        except McpTimeoutError as e:
            result['exception'] = f'Code execution timed out: {e}'
            logger.error(f'MCP timeout: {e}')
            await self._cleanup_mcp_transport('execute_code timeout', force_process=True)
        except McpError as e:
            result['exception'] = f'MCP execution error: {e}'
            result['stderr'] = str(e)
            logger.error(f'MCP error: {e}')
        except Exception as e:
            result['exception'] = str(e)
            result['stderr'] = traceback.format_exc()
            logger.error(f'Code execution failed: {e}\n{traceback.format_exc()}')
        return result

    async def _analyze_results(self, execution_result: Dict, user_query: str) -> Dict[str, Any]:
        output = execution_result.get('stdout', '')
        prompt = f'\nAnalyze the following code execution output and the original user query.\n\nOriginal Query: {user_query}\n\nExecution Output:\n{output}\n\nProvide analysis in JSON format with:\n{{\n    "findings": "Brief summary of what was discovered",\n    "insights": "Key insights from the data",\n    "should_continue": boolean (true if more iterations needed),\n    "next_step": "What to do in next iteration (if should_continue is true)",\n    "concerns": "Any issues or edge cases to address"\n}}\n\nBe concise and data-focused.\n'
        response = await self.llm_client.generate_completion(system_prompt='You are a data analysis expert.', user_message=prompt, temperature=0.2, max_tokens=1000)
        self._update_usage(response.get('usage'))
        try:
            content = response.get('content', '')
            json_match = re.search('\\{.*\\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {'findings': output[:200], 'should_continue': False, 'next_step': None}

    async def _generate_final_result(self, context: Dict) -> Dict[str, Any]:
        completed = context.get('completed_iterations', [])
        if not completed:
            logger.warning('No completed iterations for final result generation')
            return {'prediction': 'Analysis could not be completed — no iterations executed successfully.', 'text_output': 'No results generated.', 'dataframe': None, 'iterations_completed': 0, 'timestamp': utcnow().isoformat()}
        history_summary = json.dumps([{'iteration': s.get('iteration', s.get('step_num', '?')), 'reasoning': s.get('reasoning', s.get('description', '')), 'output': s.get('output', '')[:1000]} for s in completed[-5:]], indent=2)
        prompt = f"\n                Based on the step-by-step data science analysis below, provide final insights and predictions.\n\n                User Query: {context['user_query']}\n\n                Completed Iterations:\n                {history_summary}\n\n                Generate a comprehensive final result in MARKDOWN format:\n                1. **Answer**: Direct answer to the user's query\n                2. **Key Findings**: Bullet points of key metrics and discovery\n                3. **Confidence**: Assessment of result reliability\n                4. **Recommendations**: Actionable next steps\n\n                FORMATTING RULES:\n                - Use `##` for section headers (e.g. `## 1. Answer`)\n                - Always add a blank line before headers\n                - Use `**bold**` for key numbers and terms\n                - Use `*` for bullet points\n                - Do NOT use plain text blocks without formatting\n                "
        response = await self.llm_client.generate_completion(system_prompt='You are a data science expert providing final analysis.', user_message=prompt, temperature=0.3, max_tokens=2000)
        self._update_usage(response.get('usage'))
        final_usage = self.usage_stats.copy()
        if 'models' in final_usage and isinstance(final_usage['models'], set):
            final_usage['models'] = list(final_usage['models'])
        return {'prediction': response.get('content', ''), 'text_output': response.get('content', ''), 'dataframe': None, 'iterations_completed': len(completed), 'timestamp': utcnow().isoformat(), '_agent_usage': final_usage}

    def _build_context_prompt(self, context: Dict) -> str:
        if context.get('iteration') == 1:
            return 'This is the first iteration. Focus on data exploration and understanding.'
        last_analysis = context.get('last_results', {})
        return f"\nPrevious findings: {last_analysis.get('findings', 'None yet')}\nNext step guidance: {last_analysis.get('next_step', 'Continue analysis')}\n"

    def _capture_stdio_process(self) -> None:
        self._stdio_process = None
        self._stdio_process_pid = None
        manager = self._stdio_context_manager
        try:
            gen = getattr(manager, 'gen', None)
            frame = getattr(gen, 'ag_frame', None)
            process = frame.f_locals.get('process') if frame else None
            if process is not None:
                self._stdio_process = process
                self._stdio_process_pid = getattr(process, 'pid', None)
                logger.debug('Captured MCP stdio subprocess pid=%s', self._stdio_process_pid)
        except Exception as exc:
            logger.debug('Could not capture MCP stdio subprocess: %s', exc)

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    async def _terminate_stdio_process(self, reason: str) -> None:
        process = self._stdio_process
        pid = self._stdio_process_pid
        self._stdio_process = None
        self._stdio_process_pid = None
        if process is not None:
            try:
                if getattr(process, 'returncode', None) is not None:
                    return
                logger.info('Terminating MCP stdio subprocess pid=%s (%s)', pid, reason)
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                    return
                except asyncio.TimeoutError:
                    logger.warning('MCP stdio subprocess pid=%s did not exit after SIGTERM; killing', pid)
                    process.kill()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        logger.warning('MCP stdio subprocess pid=%s survived SIGKILL', pid)
                return
            except ProcessLookupError:
                return
            except Exception as exc:
                logger.warning('Error terminating MCP stdio subprocess pid=%s: %s', pid, exc)
        if pid:
            try:
                if not self._pid_exists(pid):
                    return
                logger.info('Terminating MCP stdio subprocess pid=%s via pid fallback (%s)', pid, reason)
                os.kill(pid, signal.SIGTERM)
                await asyncio.sleep(0.5)
                if self._pid_exists(pid):
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            except Exception as exc:
                logger.warning('Error terminating MCP stdio pid fallback pid=%s: %s', pid, exc)

    async def _cleanup_mcp_transport(self, reason: str, force_process: bool=False) -> None:
        if force_process:
            await self._terminate_stdio_process(reason)
        if hasattr(self, '_mcp_context_manager') and self._mcp_context_manager:
            try:
                await self._mcp_context_manager.__aexit__(None, None, None)
                logger.info('MCP client cleaned up')
            except Exception as e:
                logger.warning(f'Error cleaning up MCP client: {e}')
            finally:
                self._mcp_context_manager = None
                self.mcp_client = None
        if hasattr(self, '_stdio_context_manager') and self._stdio_context_manager:
            try:
                await self._stdio_context_manager.__aexit__(None, None, None)
                logger.info('MCP stdio cleaned up')
            except Exception as e:
                logger.warning(f'Error cleaning up MCP stdio: {e}')
            finally:
                self._stdio_context_manager = None
        await self._terminate_stdio_process(reason)

    async def _cleanup_kernel(self) -> None:
        if self._session_owned and self.session_id:
            from util import session_kernel_store
            session_kernel_store.touch_session_kernel(self.session_id)
            logger.info('Session kernel kept alive for session=%s (not released to pool)', self.session_id)
            return
        try:
            await self._cleanup_mcp_transport('kernel cleanup')
            if self.kernel_manager:
                logger.info('Releasing kernel manager')
                await release_kernel_manager(self.kernel_manager, use_pool=getattr(self, '_use_pool', True))
                self.kernel_manager = None
        except Exception as e:
            logger.warning(f'Error cleaning up kernel resources: {e}')

    async def _fetch_generated_dataframe(self) -> Optional[List[Dict]]:
        try:
            code = '\nimport json as _json_\nimport pandas as _pd_\n_target_df_ = None\n_final_val_ = None\n\n# 1. Check FINAL_RESULT first (RLM pattern — LLM declares its output explicitly)\ntry:\n    if \'FINAL_RESULT\' in globals():\n        _fr_ = FINAL_RESULT\n        if isinstance(_fr_, _pd_.DataFrame):\n            _target_df_ = _fr_\n        elif isinstance(_fr_, list) and _fr_ and isinstance(_fr_[0], dict):\n            # List of records — convert directly\n            _target_df_ = _pd_.DataFrame(_fr_)\n        elif isinstance(_fr_, dict):\n            # Dict with list-of-dicts values → extract the tabular data\n            _list_cols_ = {k: v for k, v in _fr_.items()\n                           if isinstance(v, list) and v and isinstance(v[0], dict)}\n            if _list_cols_:\n                # Use the largest list (most records = most useful table)\n                _best_key_ = max(_list_cols_, key=lambda k: len(_list_cols_[k]))\n                _target_df_ = _pd_.DataFrame(_list_cols_[_best_key_])\n            else:\n                # Scalar dict — display as a single-row table\n                try:\n                    _target_df_ = _pd_.DataFrame([_fr_])\n                except Exception:\n                    _final_val_ = repr(_fr_)[:2000]\n        else:\n            _final_val_ = repr(_fr_)[:2000]\nexcept Exception:\n    pass\n\n# 2. Fall back to _generated_dataframe_\nif _target_df_ is None and _final_val_ is None:\n    try:\n        if \'_generated_dataframe_\' in globals() and isinstance(_generated_dataframe_, _pd_.DataFrame):\n            _target_df_ = _generated_dataframe_\n    except NameError:\n        pass\n\n# 3. Fall back to df\nif _target_df_ is None and _final_val_ is None:\n    try:\n        if \'df\' in globals() and isinstance(df, _pd_.DataFrame):\n            _target_df_ = df\n    except NameError:\n        pass\n\n# 4. Heuristic fallback: pick the most relevant DataFrame in globals()\n# This prevents empty executor responses when the model forgets FINAL_RESULT\n# but has already created tabular outputs under other variable names.\nif _target_df_ is None and _final_val_ is None:\n    try:\n        _candidates_ = []\n        for _name_, _obj_ in globals().items():\n            if not isinstance(_obj_, _pd_.DataFrame):\n                continue\n            if _name_.startswith("_"):\n                continue\n            _lname_ = _name_.lower()\n            _score_ = 0\n            if _lname_.endswith("_df") or _lname_.startswith("df_") or _lname_ == "df" or "dataframe" in _lname_:\n                _score_ += 4\n            if "result" in _lname_ or "summary" in _lname_ or "output" in _lname_ or "analysis" in _lname_:\n                _score_ += 3\n            if "preview" in _lname_ or "sample" in _lname_:\n                _score_ -= 2\n            _score_ += min(len(_obj_), 1000) / 1000.0\n            _candidates_.append((_score_, _name_, _obj_))\n        if _candidates_:\n            _candidates_.sort(key=lambda _x_: _x_[0], reverse=True)\n            _target_df_ = _candidates_[0][2]\n    except Exception:\n        pass\n\n# 5. Fallback: recover table-like Python objects from globals()\nif _target_df_ is None and _final_val_ is None:\n    try:\n        _obj_candidates_ = []\n        for _name_, _obj_ in globals().items():\n            if _name_.startswith("_"):\n                continue\n            if isinstance(_obj_, list) and _obj_:\n                if isinstance(_obj_[0], dict):\n                    _obj_candidates_.append((_name_, "list_dict", _obj_))\n                elif isinstance(_obj_[0], (int, float, str, bool)):\n                    _obj_candidates_.append((_name_, "list_scalar", _obj_))\n            elif isinstance(_obj_, dict) and _obj_:\n                _obj_candidates_.append((_name_, "dict", _obj_))\n        if _obj_candidates_:\n            _obj_candidates_.sort(\n                key=lambda _x_: (\n                    ("result" in _x_[0].lower())\n                    or ("summary" in _x_[0].lower())\n                    or ("analysis" in _x_[0].lower())\n                    or ("final" in _x_[0].lower()),\n                    len(_x_[0]),\n                ),\n                reverse=True,\n            )\n            _best_name_, _best_kind_, _best_obj_ = _obj_candidates_[0]\n            if _best_kind_ == "list_dict":\n                _target_df_ = _pd_.DataFrame(_best_obj_)\n            elif _best_kind_ == "dict":\n                try:\n                    _target_df_ = _pd_.DataFrame([_best_obj_])\n                except Exception:\n                    _final_val_ = f"{_best_name_}={repr(_best_obj_)[:1800]}"\n            else:\n                _target_df_ = _pd_.DataFrame({_best_name_: _best_obj_})\n    except Exception:\n        pass\n\n# 6. Last-resort scalar recovery from globals()\nif _target_df_ is None and _final_val_ is None:\n    try:\n        _scalar_candidates_ = []\n        for _name_, _obj_ in globals().items():\n            if _name_.startswith("_"):\n                continue\n            if isinstance(_obj_, (int, float, str, bool)):\n                _lname_ = _name_.lower()\n                _score_ = 0\n                if "corr" in _lname_ or "result" in _lname_ or "score" in _lname_ or "metric" in _lname_:\n                    _score_ += 3\n                if "tmp" in _lname_ or _lname_.startswith("i"):\n                    _score_ -= 1\n                _scalar_candidates_.append((_score_, _name_, _obj_))\n        if _scalar_candidates_:\n            _scalar_candidates_.sort(key=lambda _x_: _x_[0], reverse=True)\n            _best = _scalar_candidates_[0]\n            _final_val_ = f"{_best[1]}={repr(_best[2])[:1800]}"\n    except Exception:\n        pass\n\nif _target_df_ is not None:\n    _max_rows_ = __MAX_RESULT_ROWS_VAL__\n    if len(_target_df_) > _max_rows_:\n        print(f"RESULT_TRUNCATED: {len(_target_df_)} rows → {_max_rows_} rows")\n    _json_str_ = _target_df_.head(_max_rows_).to_json(orient=\'records\', date_format=\'iso\')\n    print(f"<JSON_START>{_json_str_}<JSON_END>")\nelif _final_val_ is not None:\n    print(f"<FINAL_VAL>{_final_val_}</FINAL_VAL>")\nelse:\n    print("NO_DATAFRAME_FOUND")\n    print(f"DEBUG_VARS: {list(globals().keys())}")\n'.replace('__MAX_RESULT_ROWS_VAL__', str(self.max_result_rows))
            result = await self._execute_code(code)
            output = result.get('stdout', '').strip()
            exec_exception = result.get('exception')
            if exec_exception:
                logger.warning(f'[DF_FETCH] Kernel exception during dataframe fetch: {exec_exception}')
            logger.info(f'[DF_FETCH] Kernel stdout length={len(output)}, first 300 chars: {output[:300]!r}')
            if '<JSON_START>' in output and '<JSON_END>' in output:
                try:
                    start_idx = output.find('<JSON_START>') + len('<JSON_START>')
                    end_idx = output.find('<JSON_END>')
                    json_str = output[start_idx:end_idx].strip()
                    if json_str:
                        return json.loads(json_str)
                except Exception as parse_err:
                    logger.warning(f'Failed to parse dataframe JSON: {parse_err}')
            if '<FINAL_VAL>' in output and '</FINAL_VAL>' in output:
                start_idx = output.find('<FINAL_VAL>') + len('<FINAL_VAL>')
                end_idx = output.find('</FINAL_VAL>')
                val_str = output[start_idx:end_idx].strip()
                logger.info(f'FINAL_RESULT scalar: {val_str[:100]}')
                return [{'FINAL_RESULT': val_str}]
            if 'NO_DATAFRAME_FOUND' in output:
                logger.warning(f'Dataframe fetch failed. Kernel output: {output}')
            return None
        except Exception as e:
            logger.warning(f'Failed to fetch dataframe: {e}')
            return None

    async def _fetch_generated_chart(self) -> Optional[Dict]:
        try:
            code = '\nimport json as _json_\n_chart_out_ = None\ntry:\n    if \'_generated_plotly_fig_\' in globals():\n        _chart_out_ = _generated_plotly_fig_.to_json()\n    elif \'fig\' in globals() and hasattr(fig, \'to_json\'):\n        _chart_out_ = fig.to_json()\n    elif \'FINAL_RESULT\' in globals() and isinstance(FINAL_RESULT, dict) and \'chart\' in FINAL_RESULT:\n        import plotly.io as _pio_\n        _chart_out_ = _pio_.to_json(FINAL_RESULT[\'chart\'])\nexcept Exception:\n    pass\nif _chart_out_:\n    print(f"<CHART_START>{_chart_out_}<CHART_END>")\nelse:\n    print("NO_CHART_FOUND")\n'
            result = await self._execute_code(code)
            output = result.get('stdout', '').strip()
            if '<CHART_START>' in output and '<CHART_END>' in output:
                try:
                    s = output.find('<CHART_START>') + len('<CHART_START>')
                    e = output.find('<CHART_END>')
                    return json.loads(output[s:e].strip())
                except Exception as exc:
                    logger.warning(f'Failed to parse chart JSON: {exc}')
            return None
        except Exception as e:
            logger.warning(f'Failed to fetch chart: {e}')
            return None

    async def _fetch_all_generated_charts(self) -> list:
        try:
            code = '\nimport json as _json_\n\n_seen_ids_ = set()\n_charts_out_ = []\n\ntry:\n    import plotly.graph_objects as _go_\n\n    for _name_, _obj_ in list(globals().items()):\n        if _name_.startswith(\'_\') and not _name_.startswith(\'_generated_plotly_fig_\'):\n            continue\n        if isinstance(_obj_, _go_.Figure):\n            _oid_ = id(_obj_)\n            if _oid_ in _seen_ids_:\n                continue\n            _seen_ids_.add(_oid_)\n            try:\n                _charts_out_.append((_name_, _obj_.to_json()))\n            except Exception:\n                pass\n\n    # Sort: regular names first (alphabetical), _generated_plotly_fig_* last\n    _charts_out_.sort(key=lambda x: (x[0].startswith(\'_generated_plotly_fig_\'), x[0]))\n\n    if _charts_out_:\n        for _cname_, _cjson_ in _charts_out_:\n            print(f"<MCHART>{_cname_}|||{_cjson_}<MCHART_END>")\n    else:\n        print("NO_CHARTS_FOUND")\nexcept ImportError:\n    print("NO_CHARTS_FOUND")\nexcept Exception as _e_:\n    print(f"NO_CHARTS_FOUND: {_e_}")\n'
            result = await self._execute_code(code)
            output = result.get('stdout', '')
            if 'NO_CHARTS_FOUND' in output:
                return []
            import re
            charts = []
            for m in re.finditer('<MCHART>(.*?)<MCHART_END>', output, re.DOTALL):
                raw = m.group(1)
                sep = raw.find('|||')
                if sep == -1:
                    continue
                name = raw[:sep].strip()
                json_str = raw[sep + 3:].strip()
                try:
                    figure = json.loads(json_str)
                    charts.append({'name': name, 'figure': figure})
                except Exception as exc:
                    logger.warning(f"Skipping chart '{name}': bad JSON ({exc})")
            if charts:
                logger.info('Fetched %d Plotly charts from kernel: %s', len(charts), [c['name'] for c in charts])
            return charts
        except Exception as e:
            logger.warning(f'Failed to fetch all charts: {e}')
            return []

    async def _stream_event(self, event_type: str, content: Dict) -> Dict:
        event = {'type': event_type, 'timestamp': utcnow().isoformat()}
        event.update(content)
        return event