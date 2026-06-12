import asyncio
import contextlib
import io
import json
import logging
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
from typing import Optional
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import pandas as pd
try:
    import plotly
    import plotly.io as pio
    import plotly.utils
except Exception:
    plotly = None
    pio = None
from dotenv import load_dotenv
from util.number_formatter import format_number_indian_system
from util.column_name_mapper import map_dataframe_columns
from util.graph import _round_kpi_values
from util.dataset_paths import assets_datasets_dir
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config.system_config import AGENT_CONFIG
logger = logging.getLogger(__name__)
import builtins as _builtins_module
_EXEC_ALLOWED_MODULES: frozenset = frozenset({'pandas', 'numpy', 'json', 'math', 'datetime', 're', 'pathlib', 'io', 'collections', 'itertools', 'functools', 'typing', 'copy', 'decimal', 'statistics', 'string', 'textwrap', 'hashlib', 'warnings', 'abc', 'enum', 'dataclasses', 'numbers', 'operator', 'struct', 'time', 'calendar', 'locale', 'pprint', 'urllib', 'plotly', 'matplotlib', 'seaborn', 'sklearn', 'scipy', 'statsmodels', 'xgboost', 'pyarrow', 'fastparquet', 'xlrd', 'openpyxl', 'duckdb', 'psycopg2', 'pymysql', 'pymongo', 'hdbcli', 'oracledb'})

def _make_restricted_importer(original_import):

    def _restricted_import(name, *args, **kwargs):
        top_level = name.split('.')[0]
        if top_level not in _EXEC_ALLOWED_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in the code execution sandbox. Only data-analysis modules are permitted.")
        return original_import(name, *args, **kwargs)
    return _restricted_import
_EXEC_RESTRICTED_BUILTINS: dict = {k: v for k, v in vars(_builtins_module).items() if k not in ('open', 'eval', 'compile', 'breakpoint', 'input', 'memoryview')}
_EXEC_RESTRICTED_BUILTINS['__import__'] = _make_restricted_importer(vars(_builtins_module).get('__import__', __import__))
ORACLE_CLIENT_INITIALIZED = False
try:
    import cx_Oracle
except Exception:
    cx_Oracle = None
    logger.info('cx_Oracle library not found. Oracle DB functionality will be disabled.')

class ExecutorAgent:

    def __init__(self, agent_name: str='executor_agent', provided_config: Optional[Dict]=None, client_id: str=None, db: Any=None, dataset_id: Optional[str]=None):
        if not client_id:
            raise ValueError('client_id is REQUIRED for multi-tenant operation. No default client exists. Every request must specify a valid client_id.')
        "\n        Original docstring continuation (preserving below):\n        \n        Args:\n            agent_name: Name of the agent (default: 'executor_agent')\n            provided_config: Optional configuration dictionary\n            client_id: Client identifier for multi-tenant support (default: 'default')\n            db: Database connection for potential future use\n        "
        self.agent_name = agent_name
        self.client_id = client_id
        self.dataset_id = dataset_id
        self.db = db
        self.config = provided_config or AGENT_CONFIG.get(self.agent_name, {})
        self.db_type = None
        self.execution_timeout = int(self.config.get('execution_timeout', 180))
        base_output_dir = Path(self.config.get('output_dir', f'assets/clients/{self.client_id}/output/'))
        self.data_objects_dir = base_output_dir / 'data_objects'
        self.data_objects_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_oracle_client(self):
        global ORACLE_CLIENT_INITIALIZED
        if ORACLE_CLIENT_INITIALIZED or not cx_Oracle:
            return
        try:
            logger.info('Attempting to initialize Oracle Instant Client...')
            load_dotenv()
            lib_dir = os.getenv('LD_LIBRARY_PATH')
            if lib_dir:
                cx_Oracle.init_oracle_client(lib_dir=lib_dir)
                ORACLE_CLIENT_INITIALIZED = True
                logger.info('Oracle Instant Client initialized successfully.')
            else:
                logger.warning('LD_LIBRARY_PATH environment variable not set. Skipping Oracle client initialization.')
        except Exception as e:
            if 'already initialized' in str(e).lower():
                ORACLE_CLIENT_INITIALIZED = True
                logger.info('Oracle client was already initialized by another instance.')
            else:
                logger.error(f'Failed to initialize Oracle Client: {e}', exc_info=True)

    async def process(self, **kwargs: Any) -> Dict[str, Any]:
        python_agent_task_id = kwargs.get('python_agent_task_id', 'unknown_task')
        logger.info(f'Executing code for task ID: {python_agent_task_id}')
        try:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            executor_agent_task_id = f'executor_{timestamp}'
            generated_response, error_text = await self._execute_code(kwargs.get('generated_code', ''))
            response = kwargs.copy()
            response.update({'executor_agent_task_id': executor_agent_task_id, 'generated_response': generated_response, 'error_text': error_text})
            if error_text:
                logger.error(f'Execution for task {python_agent_task_id} failed with error: {error_text}')
            else:
                logger.info(f'Execution completed successfully for task ID: {executor_agent_task_id}')
            return response
        except Exception as e:
            logger.error(f'Critical error in Executor Agent process: {e}', exc_info=True)
            return {'executor_agent_task_id': f"executor_critical_error_{datetime.now().strftime('%Y%m%d%H%M%S%f')}", 'generated_response': '', 'error_text': f'Executor agent critical error: {e}\n{traceback.format_exc()}'}

    async def _should_use_parquet(self) -> bool:
        if self.db is None:
            raise RuntimeError('store_in_local not configured in DB Configs')
        try:
            from services.db_credentials_service import DBCredentialsService
            service = DBCredentialsService(self.db)
            credentials = await service.get_credentials(client_id=self.client_id, db_type=None, decrypt_password=False)
            if credentials:
                from util.data_source import require_store_in_local
                return require_store_in_local(credentials)
            raise RuntimeError('store_in_local not configured in DB Configs')
        except Exception as e:
            logger.error(f"Failed to check store_in_local for client '{self.client_id}': {e}")
            raise

    async def _get_data_source(self):
        from util.data_source import get_data_source_for_client
        return await get_data_source_for_client(self.client_id, self.db, self.dataset_id)

    async def _inject_sql_methods(self, exec_namespace: Dict, data_source, credentials: Dict[str, Any]) -> None:
        from db_config.injectors.sql_injector import SQLInjector
        from db_config.connection_pool_manager import ConnectionPoolManager
        db_type = data_source.value
        try:
            pool_manager = ConnectionPoolManager()
            connector = await pool_manager.get_connection(self.client_id, self.db, dataset_id=self.dataset_id)
            injector = SQLInjector(db_type=db_type, connector=connector)
        except Exception as e:
            logger.warning(f'Failed to get connection from pool for client {self.client_id}: {e}. Falling back to direct connection.')
            injector = SQLInjector(db_type=db_type, db_credential=credentials)
            await injector.connect()
        exec_namespace['read_table'] = injector.read_table
        exec_namespace['query_sql'] = injector.query_sql
        exec_namespace['read_sql_query'] = injector.query_sql
        exec_namespace['test_connection'] = injector.test_connection
        exec_namespace['pd'] = pd
        if db_type in ('sap_oracle', 'sap_sybase'):
            from agents.db_helpers import format_date_for_sap
            exec_namespace['format_date_for_sap'] = format_date_for_sap
            logger.info('Injected format_date_for_sap() helper for SAP Oracle/Sybase')
        logger.info(f'Injected SQL methods for {db_type} database using pooled connection')

    async def _inject_mongodb_methods(self, exec_namespace: Dict, credentials: Dict[str, Any]) -> None:
        from db_config.injectors.mongo_injector import MongoInjector
        injector = MongoInjector(db_credential=credentials)
        await injector.connect()
        exec_namespace['find'] = injector.find
        exec_namespace['find_one'] = injector.find_one
        exec_namespace['aggregate'] = injector.aggregate
        exec_namespace['verify_db_connection'] = injector.verify_db_connection
        exec_namespace['pd'] = pd
        logger.info('Injected MongoDB methods using pre-fetched credentials')

    async def _execute_code(self, code: str) -> Tuple[str, str]:
        if '{CLIENT}' in code or '{client}' in code or '{Client}' in code:
            client_prefix = self.client_id.upper()
            original_code = code
            code = code.replace('{CLIENT}', client_prefix)
            code = code.replace('{client}', client_prefix)
            code = code.replace('{Client}', client_prefix)
            logger.warning(f'[MULTI-TENANT SAFETY NET] Replaced {{CLIENT}} placeholder in executor for client {self.client_id}. This should have been done earlier!')
            logger.debug(f'Original code snippet: {original_code[:200]}...')
            logger.debug(f'Modified code snippet: {code[:200]}...')
        from config.system_config import STORAGE_BACKEND
        if STORAGE_BACKEND == 'gcs':
            from util.dataset_paths import storage_datasets_prefix
            correct_datasets_dir = storage_datasets_prefix(self.client_id, self.dataset_id)
        else:
            correct_datasets_dir = str(assets_datasets_dir(self.client_id, self.dataset_id).resolve())
        import re as _re

        def _datasets_path_sub(m: '_re.Match') -> str:
            quote = m.group(1)
            rest = m.group(2)
            if self.dataset_id:
                ds_prefix = f'{self.dataset_id}/'
                if rest.startswith(ds_prefix):
                    rest = rest[len(ds_prefix):]
            return f'{quote}{correct_datasets_dir}/{rest}{quote}'
        _path_pattern = _re.compile('([\'\\"]).*?assets/clients/' + _re.escape(self.client_id) + '/datasets/([^\'\\"]+)\\1')
        code = _path_pattern.sub(_datasets_path_sub, code)
        exec_namespace = {}
        from util.data_source import DataSource, require_store_in_local
        from services.db_credentials_service import DBCredentialsService
        service = DBCredentialsService(self.db)
        credentials = await service.get_credentials(self.client_id, None, decrypt_password=True, dataset_id=self.dataset_id)
        if not credentials:
            raise RuntimeError('store_in_local not configured in DB Configs')
        db_type = credentials.get('db_type')
        store_in_local = require_store_in_local(credentials)
        data_source = DataSource.from_db_type(db_type, store_in_local)
        self.db_type = db_type
        logger.info(f"Executor optimization: Determined data source '{data_source.value}' for client {self.client_id} with single credential fetch")
        if data_source.is_live_db:
            try:
                if data_source.is_sql:
                    await self._inject_sql_methods(exec_namespace, data_source, credentials)
                elif data_source == DataSource.MONGODB:
                    await self._inject_mongodb_methods(exec_namespace, credentials)
                else:
                    raise RuntimeError(f'Unsupported live DB data source: {data_source}')
            except Exception as e:
                logger.error(f'Failed to inject DB methods for client {self.client_id}: {e}', exc_info=True)
                raise RuntimeError(f'Failed to establish database connection for live query execution: {str(e)}')
        else:
            exec_namespace['pd'] = pd
            exec_namespace['Path'] = Path
            from config.system_config import STORAGE_BACKEND as _exec_storage_backend
            if _exec_storage_backend == 'gcs':
                try:
                    import duckdb as _ddb
                    from config.system_config import S3_ENDPOINT_URL as _exec_s3_ep, S3_ACCESS_KEY as _exec_s3_ak, S3_SECRET_KEY as _exec_s3_sk, GCS_BUCKET as _exec_gcs_bucket
                    from util.dataset_paths import storage_datasets_prefix as _exec_ds_prefix
                    if not _exec_s3_ak or not _exec_s3_sk:
                        raise RuntimeError('GCS HMAC credentials (S3_ACCESS_KEY / S3_SECRET_KEY) are not configured. Set them in the environment before executing GCS queries.')
                    _exec_endpoint = _exec_s3_ep.replace('https://', '').replace('http://', '')
                    _exec_gcs_prefix = _exec_ds_prefix(self.client_id, self.dataset_id)
                    if _exec_gcs_prefix.startswith('assets/'):
                        _exec_gcs_prefix = _exec_gcs_prefix[len('assets/'):]
                    _exec_conn = _ddb.connect()
                    _exec_conn.execute('LOAD httpfs;')
                    _exec_conn.execute(f"SET s3_endpoint='{_exec_endpoint}';")
                    _exec_conn.execute(f"SET s3_access_key_id='{_exec_s3_ak}';")
                    _exec_conn.execute(f"SET s3_secret_access_key='{_exec_s3_sk}';")
                    _exec_conn.execute("SET s3_url_style='path';")
                    _exec_conn.execute('SET s3_use_ssl=true;')

                    def _query_parquet_impl(file_pattern: str, sql_query: str=None, _c=_exec_conn, _b=_exec_gcs_bucket, _p=_exec_gcs_prefix):
                        _gcs_path = f's3://{_b}/{_p}/{file_pattern}'
                        if sql_query:
                            return _c.execute(sql_query.replace('{TABLE}', f"read_parquet('{_gcs_path}')")).fetchdf()
                        return _c.execute(f"SELECT * FROM read_parquet('{_gcs_path}')").fetchdf()
                    exec_namespace['query_parquet'] = _query_parquet_impl
                    exec_namespace['_coresight_conn'] = _exec_conn
                    logger.info('ExecutorAgent: injected DuckDB GCS query_parquet() for client=%s prefix=%s', self.client_id, _exec_gcs_prefix)
                except RuntimeError:
                    raise
                except Exception as _exec_gcs_err:
                    logger.error('Failed to init DuckDB GCS for ExecutorAgent: %s', _exec_gcs_err, exc_info=True)
                    raise RuntimeError(f'DuckDB GCS initialisation failed: {_exec_gcs_err}. Ensure S3_ACCESS_KEY / S3_SECRET_KEY are set and the httpfs extension is installed in the kernel image.')
        execution_results = {'dataframes': [], 'plotly_charts': [], 'matplotlib_images': [], 'text_outputs': [], 'console_output': ''}
        error_text = ''
        exec_namespace['__builtins__'] = _EXEC_RESTRICTED_BUILTINS
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            try:

                def _run_exec():
                    exec(code, exec_namespace)
                await asyncio.wait_for(asyncio.to_thread(_run_exec), timeout=self.execution_timeout)
                self._process_outputs(exec_namespace, execution_results)
            except asyncio.TimeoutError:
                logger.error(f'Code execution timed out after {self.execution_timeout}s for client {self.client_id}')
                error_text = f'Execution timed out after {self.execution_timeout} seconds.\n\nThe code took too long to run. Common causes:\n1. Operating on a very large dataset without filtering first\n2. An infinite or very long loop\n3. A computationally expensive operation (e.g., large cross-join)\n\nSuggestions:\n1. Add filters (WHERE clause or df.query()) to reduce data size BEFORE aggregation\n2. Use .head(10000) to limit rows for exploration\n3. Sample the data: df.sample(n=10000) for large datasets\n4. Avoid nested loops over DataFrames — use vectorized operations\n5. For large joins, filter both tables first, then merge\n'
            except Exception as e:
                logger.error(f'Error during code execution: {e}', exc_info=True)
                error_msg = str(e)
                error_trace = traceback.format_exc()
                if 'XMLSyntaxError' in error_trace or 'Start tag expected' in error_msg:
                    error_text = f'XML Parsing Error: {e}\n\nThe data source returned invalid XML or non-XML data.\nSuggestions:\n1. Check if the data file exists and is actually in XML format\n2. Try using pd.read_csv() instead if the data is in CSV format\n3. Try using pd.read_json() if the data is in JSON format\n4. Try using pd.read_excel() if the data is in Excel format\n5. Check the file extension to determine the correct format\n6. Add error handling: try pd.read_csv() first, then fallback to other formats\n\nFull traceback:\n{error_trace}'
                elif 'FileNotFoundError' in error_trace or 'No such file' in error_msg:
                    error_text = f'File Not Found Error: {e}\n\nThe specified file path does not exist.\nSuggestions:\n1. Check the file path is correct\n2. Use Path.exists() to verify the file exists before reading\n3. List available files in the directory to find the correct filename\n\nFull traceback:\n{error_trace}'
                elif 'ORA-00904' in error_msg or 'invalid identifier' in error_msg.lower():
                    import re
                    col_match = re.search('["\\\']?(\\w+)["\\\']?:?\\s+invalid identifier', error_msg, re.IGNORECASE)
                    column_name = col_match.group(1) if col_match else 'the specified column'
                    error_text = f"""Database Column Error: {e}\n\nThe column '{column_name}' does not exist in the table(s) referenced in your SQL query.\nCommon causes:\n1. Column name is misspelled or doesn't exist in this table\n2. Column exists in a different table (e.g., RESB is a table, not a column in MARD)\n3. For Oracle: Column names are case-sensitive when quoted - ensure exact match\n\nSolutions:\n1. Check the RELEVANT_SCHEMA section to verify which columns exist in each table\n2. Verify '{column_name}' is a column name, not a table name\n3. For UNION/UNION ALL queries: Verify the column exists in EACH table\n4. Only use columns listed in the <column name="..."> tags for that specific table\n5. For Oracle: Use uppercase column names (MATNR, ERDAT, etc.)\n6. If unsure, use SELECT * FROM table_name to see available columns\n\nFull traceback:\n{error_trace}"""
                elif 'unhashable type' in error_msg.lower() and 'list' in error_msg.lower():
                    error_text = f"Type Error: {e}\n\nA list was used where a hashable type (string, number, tuple) is required.\nCommon causes:\n1. Using a list as a dictionary key: {{['col1', 'col2']: value}} ❌\n2. Using a list in a set: set([['a', 'b']]) ❌\n3. Incorrect column selection in DataFrame operations\n\nSolutions:\n1. For dictionary keys: Use a tuple instead: {{('col1', 'col2'): value}} ✅\n2. For sets: Convert list to tuple: set([tuple(['a', 'b'])]) ✅\n3. For DataFrame column selection: Use df[['col1', 'col2']] (this is correct) ✅\n4. For groupby: Use df.groupby(['col1', 'col2']) (this is correct) ✅\n5. Check if you're accidentally using a list where a string is expected\n\nFull traceback:\n{error_trace}"
                elif isinstance(e, KeyError) and 'merge' in error_trace.lower():
                    if hasattr(e, 'args') and len(e.args) > 0:
                        column_name = str(e.args[0]).strip('\'"')
                    else:
                        import re
                        col_match = re.search('[\'\\"](\\w+)[\'\\"]', error_msg)
                        column_name = col_match.group(1) if col_match else error_msg.strip('\'"')
                    error_text = f"Merge Error: {e}\n\nThe column '{column_name}' was not found in one of the DataFrames being merged.\nThis usually happens when column names don't match between DataFrames.\n\nSolutions:\n1. Check column names in both DataFrames:\n   print('DF1 columns:', df1.columns.tolist())\n   print('DF2 columns:', df2.columns.tolist())\n2. Normalize column names to same case before merging (for SAP Oracle ONLY):\n   df1.columns = [str(col).upper() for col in df1.columns]\n   df2.columns = [str(col).upper() for col in df2.columns]\n3. For Oracle: All columns are uppercase (MATNR, ERDAT, etc.)\n4. Verify both DataFrames have the merge key:\n   assert '{column_name}' in df1.columns\n   assert '{column_name}' in df2.columns\n5. Use explicit 'on' parameter with correct column name:\n   pd.merge(df1, df2, on='{column_name}')  # Use exact column name\n\nFull traceback:\n{error_trace}"
                elif isinstance(e, AttributeError) and '.str accessor' in error_msg.lower():
                    error_text = f"Column Normalization Error: {e}\n\nThe .str accessor was used on DataFrame columns that are not string type.\nThis happens when trying to normalize column names with df.columns.str.upper()\non a DataFrame where columns are not string-like (e.g., empty DataFrame or non-string column names).\n\nSolutions:\n1. Use a safer column normalization method:\n   # For SAP Oracle ONLY - use this safer approach:\n   df.columns = [str(col).upper() for col in df.columns]\n   # OR use rename:\n   df.rename(columns=str.upper, inplace=True)\n\n2. Check if DataFrame is empty before normalizing:\n   if not df.empty and len(df.columns) > 0:\n       df.columns = [str(col).upper() for col in df.columns]\n\n3. Verify column types:\n   print('Column types:', df.columns.dtype)\n   print('Columns:', df.columns.tolist())\n\n4. For SAP Oracle: Use the safer normalization method above\n5. For other data sources: Do NOT normalize column names\n\nFull traceback:\n{error_trace}"
                elif isinstance(e, KeyError) and ('pandas' in error_trace.lower() or 'get_loc' in error_trace.lower()):
                    if hasattr(e, 'args') and len(e.args) > 0:
                        column_name = str(e.args[0]).strip('\'"')
                    else:
                        import re
                        col_match = re.search('[\'\\"](\\w+)[\'\\"]', error_msg)
                        column_name = col_match.group(1) if col_match else error_msg.strip('\'"')
                    error_text = f"Column Not Found Error: {e}\n\nThe column '{column_name}' does not exist in the DataFrame.\nThis usually happens when:\n1. The column name is misspelled or doesn't exist in the loaded data\n2. The column name has different casing (e.g., 'audat' vs 'AUDAT')\n3. The column wasn't selected when reading the data\n4. The DataFrame is empty or has a different structure than expected\n\nSolutions:\n1. Check available columns in the DataFrame:\n   print('Available columns:', df.columns.tolist())\n2. Verify column name casing:\n   # For SAP Oracle data source ONLY: normalize to uppercase\n   # df.columns = [str(col).upper() for col in df.columns]  # ONLY if db_type is 'sap_oracle'\n   # For other data sources (postgres, mysql, sap_hana, parquet): preserve original casing\n   # Check if column exists with different casing:\n   if '{column_name.upper()}' in df.columns:\n       # Column exists with uppercase name (might be SAP/Oracle)\n   elif '{column_name.lower()}' in df.columns:\n       # Column exists with lowercase name\n3. When reading data, explicitly select the column:\n   df = pd.read_parquet(file_path, columns=['{column_name}', ...])\n   # OR for SQL:\n   df = read_sql_query('SELECT {column_name}, ... FROM table')\n4. Check the RELEVANT_SCHEMA section in the prompt to verify column names\n5. For SAP Oracle ONLY: All columns are uppercase (AUDAT, MATNR, ERDAT, etc.)\n   # Normalize ONLY for SAP Oracle: df.columns = [str(col).upper() for col in df.columns]\n6. If the DataFrame is from a join/merge, verify the column exists in the source table\n7. Check the db_type - normalization is ONLY needed for 'sap_oracle' (NOT sap_hana)\n\nFull traceback:\n{error_trace}"
                elif 'Cannot index by location index with a non-integer key' in error_msg:
                    error_text = f'Indexing Error: {e}\n\nYou are trying to use .iloc[] with a string or non-integer key.\n.iloc[] is strictly for integer-location based indexing.\n\nSolutions:\n1. Use .loc[] instead of .iloc[] if you want to index by string/label.\n2. Use integer index positions (0, 1, 2) if you want to use .iloc[].\n\nFull traceback:\n{error_trace}'
                elif 'unsupported format string passed to _iLocIndexer.__format__' in error_msg:
                    error_text = f"Formatting Error: {e}\n\nYou are trying to format a pandas .iloc object directly (e.g., f'{{df.iloc:.2f}}') instead of its value.\n\nSolutions:\n1. Access the specific value before formatting: e.g., f'{{df.iloc[0]:.2f}}'.\n2. Ensure you are actually extracting a scalar value from the DataFrame/Series.\n\nFull traceback:\n{error_trace}"
                else:
                    error_text = f'Execution error: {e}\n{error_trace}'
            finally:
                execution_results['console_output'] = buf.getvalue()
        try:
            sanitized_results = self._sanitize_nan_values(execution_results)
            if 'plotly' in globals() and plotly is not None and hasattr(plotly, 'utils'):
                response_json = json.dumps(sanitized_results, indent=2, cls=plotly.utils.PlotlyJSONEncoder)
            else:
                response_json = json.dumps(sanitized_results, indent=2)
        except Exception as e:
            logger.error(f'Error serializing execution results to JSON: {e}', exc_info=True)
            error_text += f'\nJSON Serialization Error: {e}'
            response_json = str(execution_results)
        return (response_json, error_text)

    @staticmethod
    def _sanitize_nan_values(obj):
        import math
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: ExecutorAgent._sanitize_nan_values(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ExecutorAgent._sanitize_nan_values(item) for item in obj]
        return obj

    def _process_outputs(self, namespace: Dict, results: Dict):
        self._process_dataframes(namespace, results['dataframes'])
        self._process_plotly_figures(namespace, results['plotly_charts'])
        self._process_matplotlib_figures(namespace, results['matplotlib_images'])
        self._process_text_outputs(namespace, results['text_outputs'])
        self._process_final_result(namespace, results)
    _DF_MEMORY_LIMIT = 50 * 1024 * 1024
    _DF_PREVIEW_ROWS = 10000

    def _process_dataframes(self, namespace: Dict, dataframes_list: list):
        for name, var in namespace.items():
            if re.match('_generated_dataframe_?(\\d+)?_', name) and isinstance(var, pd.DataFrame):
                df_display = var.copy()
                column_mapping = {}
                column_metadata = {}
                if self.db_type in ('sap_oracle', 'sap_sybase'):
                    table_name = None
                    column_mapping, column_metadata = map_dataframe_columns(df_display, table_name, self.db_type)
                    if column_mapping:
                        df_display.rename(columns=column_mapping, inplace=True)
                    final_column_metadata = {}
                    for orig_col, final_col in column_mapping.items():
                        if final_col in df_display.columns:
                            if final_col in column_metadata:
                                final_column_metadata[final_col] = column_metadata[final_col]
                        else:
                            matching_cols = [c for c in df_display.columns if c.startswith(final_col)]
                            if matching_cols:
                                actual_col = matching_cols[0]
                                if final_col in column_metadata:
                                    final_column_metadata[actual_col] = column_metadata[final_col]
                    for col in df_display.columns:
                        if col not in final_column_metadata and col in column_metadata:
                            final_column_metadata[col] = column_metadata[col]
                    column_metadata = final_column_metadata
                total_rows = len(df_display)
                mem_bytes = int(df_display.memory_usage(deep=True).sum())
                full_data_path: Optional[str] = None
                is_truncated = False
                if mem_bytes > self._DF_MEMORY_LIMIT:
                    try:
                        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
                        parquet_filename = f'{name}_{timestamp}.parquet'
                        parquet_path = self.data_objects_dir / parquet_filename
                        df_display.to_parquet(parquet_path, index=False)
                        full_data_path = str(parquet_path)
                        is_truncated = True
                        df_display = df_display.head(self._DF_PREVIEW_ROWS)
                        logging.getLogger(__name__).info("Large DataFrame '%s' (%d rows, %.1f MB) spilled to %s; preview truncated to %d rows.", name, total_rows, mem_bytes / (1024 * 1024), parquet_path, self._DF_PREVIEW_ROWS)
                    except Exception as spill_err:
                        logging.getLogger(__name__).warning("Failed to spill DataFrame '%s' to parquet: %s — serialising full DataFrame instead.", name, spill_err)
                dataframes_list.append({'name': name, 'json_data': df_display.to_json(orient='split', date_format='iso'), 'html_data': df_display.to_html(classes='table table-striped'), 'column_mapping': column_mapping, 'column_metadata': column_metadata, 'total_rows': total_rows, 'is_truncated': is_truncated, 'full_data_path': full_data_path})

    def _process_plotly_figures(self, namespace: Dict, charts_list: list):
        for name, var in namespace.items():
            if re.match('_generated_plotly_fig_?(\\d+)?_', name):
                fig_json = self._serialize_plotly_figure(var)
                if fig_json:
                    charts_list.append({'name': name, 'figure': fig_json})

    def _serialize_plotly_figure(self, fig: Any) -> Optional[str]:
        try:
            if hasattr(fig, 'to_json'):
                return fig.to_json()
            if hasattr(fig, 'to_dict'):
                return json.dumps(fig.to_dict())
            if 'plotly' in globals() and plotly is not None and hasattr(plotly, 'utils'):
                return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return json.dumps(fig)
        except Exception as e:
            logger.error(f'Could not serialize Plotly figure: {e}')
            return None

    def _process_matplotlib_figures(self, namespace: Dict, images_list: list):
        for name, var in namespace.items():
            if plt is not None and re.match('_generated_matplotlib_fig_?(\\d+)?_', name) and isinstance(var, plt.Figure):
                try:
                    img_path = self.data_objects_dir / f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
                    var.savefig(img_path)
                    plt.close(var)
                    images_list.append({'name': name, 'path': str(img_path)})
                    logger.info(f"Saved Matplotlib figure '{name}' to {img_path}")
                except Exception as e:
                    logger.error(f"Failed to save Matplotlib figure '{name}': {e}")

    def _process_text_outputs(self, namespace: Dict, text_list: list):
        custom_prompt_prefix = self._get_custom_prompt_prefix()
        for name, var in namespace.items():
            if re.match('_generated_text_output_?(\\d+)?_', name):
                formatted_value = format_number_indian_system(str(var))
                if custom_prompt_prefix:
                    formatted_value = f'{custom_prompt_prefix}{formatted_value}'
                text_list.append({'name': name, 'value': formatted_value})

    def _process_final_result(self, namespace: Dict, results: Dict):
        fr = namespace.get('FINAL_RESULT')
        if not isinstance(fr, dict):
            return
        if not results['plotly_charts']:
            chart_json = fr.get('chart')
            if chart_json:
                results['plotly_charts'].append({'name': '_analyst_chart_', 'figure': chart_json})
        if not results['dataframes']:
            table_data = fr.get('table')
            if table_data and isinstance(table_data, list) and (len(table_data) > 0):
                import json as _json
                import datetime as _dt
                import numpy as _np_local

                def _json_safe(v):
                    if isinstance(v, (_dt.datetime, _dt.date)):
                        return v.isoformat()
                    if hasattr(v, 'isoformat'):
                        return v.isoformat()
                    if isinstance(v, _np_local.integer):
                        return int(v)
                    if isinstance(v, _np_local.floating):
                        return float(v)
                    if isinstance(v, _np_local.bool_):
                        return bool(v)
                    return v
                columns = list(table_data[0].keys())
                rows = [[_json_safe(row.get(c)) for c in columns] for row in table_data]
                results['dataframes'].append({'name': '_analyst_table_', 'json_data': _json.dumps({'columns': columns, 'data': rows}), 'column_mapping': {}, 'column_metadata': {}})
        kpis = fr.get('kpis')
        if kpis and (not any((t['name'] == '_kpis_' for t in results['text_outputs']))):
            results['text_outputs'].append({'name': '_kpis_', 'value': str(_round_kpi_values(kpis))})
        summary = fr.get('summary', '')
        if summary and (not any((t['name'] == '_summary_' for t in results['text_outputs']))):
            results['text_outputs'].append({'name': '_summary_', 'value': summary})
        logger.info('Extracted FINAL_RESULT from executed code')

    def _get_custom_prompt_prefix(self) -> str:
        try:
            from util.xml_prompt_loader import load_custom_prompts
            custom_prompts_text = load_custom_prompts(self.client_id)
            if not custom_prompts_text:
                return ''
            if 'Say Hello' in custom_prompts_text or 'greet' in custom_prompts_text.lower():
                return 'Hello! '
            return ''
        except Exception as e:
            logger.debug(f'Could not load custom prompts for text output: {e}')
            return ''