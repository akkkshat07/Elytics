from __future__ import annotations
import difflib
import logging
import os
import re
from datetime import datetime
from util.time_utils import utcnow
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from agents.data_science_agent import DataScienceAgent
from util.xml_prompt_loader import load_client_prompt, BASE_PROMPTS_PATH
from util.dataset_paths import assets_datasets_dir, storage_datasets_prefix
from util.kernel_factory import create_kernel_manager
logger = logging.getLogger(__name__)

class DataAnalystAgent(DataScienceAgent):

    def __init__(self, agent_name: str='data_analyst_agent', provided_config: Optional[Dict]=None, client_id: str=None, db: Any=None, notebook_output_dir: str='test_outputs', llm_client: Any=None, resolved_prompt: Optional[str]=None, dataset_id: Optional[str]=None, session_id: Optional[str]=None):
        super().__init__(agent_name=agent_name, provided_config=provided_config, client_id=client_id, db=db, notebook_output_dir=notebook_output_dir, llm_client=llm_client, resolved_prompt=resolved_prompt, dataset_id=dataset_id, session_id=session_id)
        self.doom_loop_threshold: int = self.config.get('doom_loop_threshold', 3)
        self.always_generate_chart: bool = self.config.get('always_generate_chart', True)
        self.always_generate_table: bool = self.config.get('always_generate_table', True)
        self._recent_failed_codes: List[str] = []
        logger.info("DataAnalystAgent initialized for client '%s' | provider=%s, model=%s, max_iterations=%d, doom_loop_threshold=%d", client_id, self.llm_provider, self.model, self.max_iterations, self.doom_loop_threshold)

    def _load_system_prompt(self, resolved_prompt: Optional[str]=None) -> str:
        if resolved_prompt:
            return resolved_prompt
        import asyncio
        relative_path = f'agents/{self.agent_name}.xml'
        if self.db is not None and self.client_id:
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    return loop.run_until_complete(load_client_prompt(relative_path, self.client_id, self.db, use_formatting=False))
            except Exception as e:
                logger.warning('Client-aware prompt loading failed for %s (client=%s), falling back to base: %s', self.agent_name, self.client_id, e)
        try:
            prompt_path = Path(BASE_PROMPTS_PATH) / 'agents' / 'data_analyst_agent.xml'
            if prompt_path.exists():
                with open(prompt_path, 'r') as f:
                    return f.read()
        except Exception as e:
            logger.warning('Could not load data_analyst_agent.xml: %s', e)
        return 'You are an expert Data Analyst Agent specialised in descriptive and diagnostic analytics: trend analysis, KPI tracking, comparisons, root-cause analysis, distributions, and outlier detection.\nGenerate ONLY executable Python code. Always produce a Plotly chart and a summary DataFrame in FINAL_RESULT.'

    async def _probe_time_dimensions(self, loaded_datasets: List[Dict], execution_context: Dict) -> Dict[str, Any]:
        if not self.mcp_client or not loaded_datasets:
            return {}
        probe_code = '\nimport pandas as pd, json, traceback\n\ntime_info = {}\n_var_scope = {k: v for k, v in globals().items() if isinstance(v, pd.DataFrame)}\n\nfor _var_name, _df in _var_scope.items():\n    date_cols = []\n    date_range = {}\n    suggested_freq = "ME"   # default: monthly\n\n    for col in _df.columns:\n        _series = _df[col]\n        # Already datetime\n        if pd.api.types.is_datetime64_any_dtype(_series):\n            date_cols.append(col)\n        # String columns that look like dates\n        elif _series.dtype == object:\n            sample = _series.dropna().head(10)\n            try:\n                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors=\'coerce\')\n                if parsed.notna().sum() >= min(5, len(sample)):\n                    date_cols.append(col)\n            except Exception:\n                pass\n\n    if date_cols:\n        primary = date_cols[0]\n        try:\n            _ts = pd.to_datetime(_df[primary], errors=\'coerce\').dropna()\n            if len(_ts) > 0:\n                span_days = (_ts.max() - _ts.min()).days\n                date_range = {\n                    "min": _ts.min().isoformat(),\n                    "max": _ts.max().isoformat(),\n                    "span_days": span_days,\n                }\n                # Suggest frequency based on span\n                if span_days <= 14:\n                    suggested_freq = "D"\n                elif span_days <= 90:\n                    suggested_freq = "W"\n                elif span_days <= 730:\n                    suggested_freq = "ME"\n                else:\n                    suggested_freq = "QE"\n        except Exception as _e:\n            date_range = {"error": str(_e)}\n\n    time_info[_var_name] = {\n        "date_columns": date_cols,\n        "suggested_freq": suggested_freq,\n        "date_range": date_range,\n    }\n\nprint("__TIME_DIM__:" + json.dumps(time_info))\n'
        try:
            result = await self._execute_code(probe_code)
            stdout = result.get('stdout', '')
            for line in stdout.splitlines():
                if line.startswith('__TIME_DIM__:'):
                    import json as _json
                    return _json.loads(line[len('__TIME_DIM__:'):])
        except Exception as e:
            logger.warning('Time dimension probing failed: %s', e)
        return {}

    def _detect_doom_loop(self, current_code: str) -> bool:
        if len(self._recent_failed_codes) < self.doom_loop_threshold:
            return False
        last_n = self._recent_failed_codes[-self.doom_loop_threshold:]
        for prev in last_n:
            ratio = difflib.SequenceMatcher(None, current_code.strip(), prev.strip()).ratio()
            if ratio < 0.92:
                return False
        return True

    async def execute_analysis(self, user_query: str, plan: str, dataset_path: Optional[str]=None, dataset_dict: Optional[Dict]=None, context: Optional[Dict]=None) -> AsyncGenerator[Dict, None]:
        import traceback
        try:
            from util.notebook_builder import NotebookBuilder
            self.notebook_builder = NotebookBuilder(output_dir=self.notebook_output_dir, name_prefix='analysis')
            self.notebook_builder.add_markdown_cell(f"# Data Analysis\n\n**Query:** {user_query}\n\n**Generated:** {utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n---\n\n## Analyst Guidance\n\n{plan}")
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
                    from pathlib import Path as _Path
                    import sys as _sys
                    _root = _Path(__file__).resolve().parent.parent
                    from config.system_config import STORAGE_BACKEND
                    if STORAGE_BACKEND == 'gcs':
                        from agents.data_science_agent import _get_duckdb_bootstrap_code
                        data_prefix = storage_datasets_prefix(self.client_id, self.dataset_id)
                        bootstrap = _get_duckdb_bootstrap_code(self.client_id, self.dataset_id)
                        if bootstrap:
                            await self._execute_code(bootstrap)
                        try:
                            from util.storage.backend import get_storage_backend
                            storage = get_storage_backend()
                            gcs_files = await storage.list_files(data_prefix)
                            parquet_files = [f for f in gcs_files if f.endswith('.parquet')]
                            if parquet_files:
                                loaded_datasets = []
                                for pf in parquet_files:
                                    fname = pf.rsplit('/', 1)[-1] if '/' in pf else pf
                                    loaded_datasets.append({'path': fname, 'variable': 'df', 'format': 'parquet', 'gcs': True})
                            else:
                                loaded_datasets = [{'path': data_prefix, 'variable': 'df', 'format': 'parquet', 'gcs': True}]
                        except Exception:
                            loaded_datasets = [{'path': data_prefix, 'variable': 'df', 'format': 'parquet', 'gcs': True}]
                    else:
                        client_data_dir = assets_datasets_dir(self.client_id, self.dataset_id)
                        if client_data_dir.exists():
                            parquet_files = list(client_data_dir.glob('*.parquet'))
                            if parquet_files:
                                main_file = parquet_files[0]
                                force_code = f"import pandas as pd\ndf = pd.read_parquet(r'{main_file}')\nprint(f'Force-loaded: shape={{df.shape}}')\n"
                                await self._execute_code(force_code)
                                kernel_vars = await self._get_kernel_variables()
                                actual_vars = list(kernel_vars.keys())
                                loaded_datasets = [{'path': str(main_file), 'variable': 'df', 'format': 'parquet'}]
                except Exception as _fe:
                    logger.warning('Force-load fallback failed: %s', _fe)
            execution_context: Dict[str, Any] = {'user_query': user_query, 'plan_guidance': plan, 'dataset_path': dataset_path, 'available_variables': kernel_vars or {}, 'completed_iterations': [], 'execution_journal': [], 'context': context or {}, 'loaded_datasets': loaded_datasets, 'live_sql_mode': self._is_live_db, 'db_type': (self.db_credentials_env or {}).get('CS_DB_TYPE', '') if self._is_live_db else '', 'warnings': []}
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
                logger.info('Adhoc mode: skipping backend knowledge loading (DA)')
            else:
                try:
                    execution_context['knowledge_context'] = self._load_knowledge_for_coding()
                except Exception as e:
                    logger.warning('Knowledge loading failed (non-fatal): %s', e)
                    execution_context['knowledge_context'] = {}
            if not self._is_live_db:
                yield (await self._stream_event('status', {'message': 'Detecting time dimensions...'}))
                time_dims = await self._probe_time_dimensions(loaded_datasets, execution_context)
                if time_dims:
                    execution_context['time_dimensions'] = time_dims
                    logger.info('Time dimensions detected: %s', time_dims)
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
                    logger.error('Iteration %d: _decide_next_action failed: %s', iteration, e)
                    yield (await self._stream_event('error', {'message': f'Decision-making failed at iteration {iteration}: {e}', 'iteration': iteration}))
                    status = 'error'
                    break
                action = decision.get('action', 'code')
                reasoning = decision.get('reasoning', '')
                thinking = decision.get('thinking', '')
                code = decision.get('code', '')
                if action == 'done':
                    logger.info('Iteration %d: LLM declared DONE — %s', iteration, reasoning)
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
                logger.info('Iteration %d/%d: %s', iteration, self.max_iterations, reasoning)
                iteration_success = False
                last_error: Optional[str] = None
                _stashed_failed_code = None
                _stashed_error_type = None
                for attempt in range(self.max_retries_per_iteration):
                    try:
                        if attempt > 0:
                            code = await self._regenerate_code_after_error(user_query=user_query, plan_guidance=plan, execution_context=execution_context, iteration=iteration, failed_code=code, error=last_error, attempt=attempt)
                        if not code:
                            last_error = 'Failed to generate code'
                            continue
                        if self._detect_doom_loop(code):
                            doom_msg = f'Doom loop detected at iteration {iteration}: the last {self.doom_loop_threshold} failed attempts used nearly identical code. Aborting to prevent wasted compute.'
                            logger.warning(doom_msg)
                            yield (await self._stream_event('error', {'message': doom_msg, 'iteration': iteration, 'last_error': last_error or ''}))
                            status = 'error'
                            break
                        validation_err = self._validate_code_syntax(code, execution_context.get('available_variables', {}))
                        if validation_err:
                            logger.warning('Iteration %d: AST validation failed: %s', iteration, validation_err)
                            last_error = f'Code validation error: {validation_err}'
                            self._recent_failed_codes.append(code)
                            continue
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
                            logger.info('Iteration %d: detected error in stdout: %s', iteration, str(detected_error)[:150])
                        if detected_error:
                            last_error = detected_error
                            _stashed_failed_code = code
                            _stashed_error_type, _, _ = self._classify_error(detected_error)
                            logger.warning('Iteration %d failed (attempt %d): %s', iteration, attempt + 1, str(last_error)[:200])
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
                                logger.info('Re-profiled after iteration %d: new=%s', iteration, current_df_names - existing_profile_keys)
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
                            logger.warning('Iteration %d: zero-row self-correction triggered: %s', iteration, validation_issue)
                            iteration_success = False
                            yield (await self._stream_event('iteration_retry', {'iteration': iteration, 'attempt': attempt + 1, 'error': last_error, 'message': 'Zero-row result detected — retrying with diagnostic context...'}))
                            continue
                        elif not is_valid:
                            logger.warning('Iteration %d silent failure (no retries left): %s', iteration, validation_issue)
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
                                    logger.info('Context compacted: summarized iterations %s-%s', batch[0]['iteration'], batch[-1]['iteration'])
                                except Exception as compact_err:
                                    logger.warning('Context compaction skipped: %s', compact_err)
                        logger.info('Iteration %d completed successfully', iteration)
                        consecutive_failures = 0
                        logger.debug('Iteration %d: checking for FINAL_RESULT in new_vars. Keys: %s', iteration, list(new_vars.keys()) if new_vars else 'EMPTY')
                        if 'FINAL_RESULT' in new_vars:
                            logger.info('Iteration %d: FINAL_RESULT detected in kernel — auto-done', iteration)
                            if self.notebook_builder:
                                self.notebook_builder.add_markdown_cell(f'## Analysis Complete (iteration {iteration})\n\nFINAL_RESULT was set — stopping.')
                                self.notebook_builder.save()
                            status = 'done'
                        break
                    except Exception as exc:
                        last_error = str(exc)
                        logger.error('Error in iteration %d, attempt %d: %s', iteration, attempt + 1, exc)
                        if attempt < self.max_retries_per_iteration - 1:
                            yield (await self._stream_event('iteration_retry', {'iteration': iteration, 'attempt': attempt + 1, 'error': str(exc)}))
                if status == 'error':
                    break
                if not iteration_success:
                    consecutive_failures += 1
                    execution_context.setdefault('failed_iterations', []).append({'iteration': iteration, 'error': (last_error or 'unknown')[:500], 'code_snippet': (code or '')[:300]})
                    logger.warning('Iteration %d failed after %d attempts (consecutive: %d)', iteration, self.max_retries_per_iteration, consecutive_failures)
                    if self.notebook_builder:
                        self.notebook_builder.add_markdown_cell(f"❌ **Iteration {iteration} FAILED** after {self.max_retries_per_iteration} attempts.\n\nLast error: `{(str(last_error)[:300] if last_error else 'Unknown')}`")
                        self.notebook_builder.save()
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        yield (await self._stream_event('error', {'message': f'{consecutive_failures} consecutive iterations failed. Stopping to prevent wasted compute.', 'iteration': iteration, 'last_error': last_error}))
                        status = 'error'
                        break
            completed_count = len(execution_context['completed_iterations'])
            if status == 'error' or completed_count == 0:
                failure_msg = f'Analysis incomplete — {completed_count} iterations completed, stopped due to errors.'
                final_result = {'prediction': failure_msg, 'text_output': failure_msg, 'dataframe': None, 'plotly_charts': [], 'iterations_completed': completed_count, 'timestamp': utcnow().isoformat(), 'pipeline_failed': True, '_agent_usage': {k: list(v) if isinstance(v, set) else v for k, v in self.usage_stats.items()}}
            else:
                yield (await self._stream_event('status', {'message': 'Fetching result data...'}))
                final_df_records = await self._fetch_generated_dataframe()
                logger.info('DA finalize: _fetch_generated_dataframe returned %s rows', len(final_df_records) if isinstance(final_df_records, list) else type(final_df_records).__name__)
                yield (await self._stream_event('status', {'message': 'Generating final result...'}))
                final_result = await self._generate_final_result(execution_context)
                final_result['dataframe'] = final_df_records
                analyst_extras = await self._extract_analyst_final_result()
                logger.info('DA finalize: _extract_analyst_final_result keys=%s | chart=%s | table=%s', list(analyst_extras.keys()) if analyst_extras else 'empty', 'yes' if analyst_extras.get('chart') else 'no', len(analyst_extras['table']) if isinstance(analyst_extras.get('table'), list) else 'none')
                if analyst_extras:
                    final_result.update(analyst_extras)
                strict_ok, strict_reason = self._validate_query_aligned_final_result(user_query, final_result)
                logger.info('DA finalize: validation strict_ok=%s reason=%s | has_table=%s has_df=%s has_chart=%s has_kpis=%s', strict_ok, strict_reason or 'OK', bool(final_result.get('table')), bool(final_result.get('dataframe')), bool(final_result.get('chart')), bool(final_result.get('kpis')))
                debug_fallback_enabled = os.getenv('CORESIGHT_ENABLE_DEBUG_RESULT_FALLBACK', 'false').strip().lower() in {'1', 'true', 'yes', 'on'}
                if not strict_ok and debug_fallback_enabled:
                    if not final_result.get('dataframe'):
                        recovered_df = await self._recover_dataframe_from_completed_iterations(execution_context.get('completed_iterations', []))
                        if recovered_df:
                            final_result['dataframe'] = recovered_df
                    if not final_result.get('dataframe'):
                        table_payload = final_result.get('table')
                        if isinstance(table_payload, list) and table_payload:
                            if isinstance(table_payload[0], dict):
                                final_result['dataframe'] = table_payload
                            else:
                                final_result['dataframe'] = [{'value': str(v)} for v in table_payload[:500]]
                        elif isinstance(table_payload, dict):
                            final_result['dataframe'] = [table_payload]
                        elif final_result.get('kpis') and isinstance(final_result['kpis'], dict):
                            final_result['dataframe'] = [final_result['kpis']]
                        elif final_result.get('summary'):
                            final_result['dataframe'] = [{'summary': str(final_result['summary'])}]
                    strict_ok, strict_reason = self._validate_query_aligned_final_result(user_query, final_result)
                if not strict_ok:
                    _viz_check = final_result.get('viz_type', 'chart_and_table')
                    if _viz_check == 'kpi_card' and final_result.get('kpis'):
                        logger.info('DA finalize: kpi_card with valid kpis — bypassing row validation (reason: %s)', strict_reason)
                        strict_ok = True
                if not strict_ok:
                    msg = f'Result could not be finalized with query-aligned tabular output. Reason: {strict_reason}.'
                    logger.warning('DA finalize: strict_ok=False — %s', msg)
                    _has_table = isinstance(final_result.get('table'), list) and len(final_result.get('table', [])) > 0
                    _has_df = isinstance(final_result.get('dataframe'), list) and len(final_result.get('dataframe', [])) > 0
                    if _has_table or _has_df:
                        logger.info('DA finalize: strict_ok=False but rows present (table=%s df=%s) — keeping data, marking pipeline_failed', len(final_result.get('table') or []), len(final_result.get('dataframe') or []))
                        final_result['pipeline_failed'] = True
                    else:
                        final_result['prediction'] = msg
                        final_result['dataframe'] = None
                        final_result['table'] = []
                        final_result['pipeline_failed'] = True
                all_charts = await self._fetch_all_generated_charts()
                if len(all_charts) > 1:
                    final_result['charts'] = all_charts
                    if not final_result.get('chart'):
                        final_result['chart'] = all_charts[0]['figure']
                elif len(all_charts) == 1 and (not final_result.get('chart')):
                    final_result['chart'] = all_charts[0]['figure']
                    final_result['charts'] = all_charts
                if self.notebook_builder:
                    summary_text = final_result.get('text_output', final_result.get('prediction', ''))
                    self.notebook_builder.add_markdown_cell(f'---\n\n## Final Result\n\n{summary_text}')
                    nb_path = self.notebook_builder.save()
                    final_result['notebook_path'] = str(nb_path)
            logger.info('DA finalize: final_result keys=%s has_chart=%s has_table=%s', list(final_result.keys()) if isinstance(final_result, dict) else type(final_result).__name__, bool(final_result.get('chart')) if isinstance(final_result, dict) else False, bool(final_result.get('table')) if isinstance(final_result, dict) else False)
            yield (await self._stream_event('final_result', final_result))
            yield (await self._stream_event('status', {'message': f'Analysis complete ({completed_count} iterations)' if status != 'error' else 'Analysis incomplete due to errors', 'notebook_path': str(self.notebook_builder.filepath) if self.notebook_builder else None}))
        except Exception as e:
            logger.error('Error in execute_analysis: %s\n%s', e, traceback.format_exc())
            partial_usage = {k: list(v) if isinstance(v, set) else v for k, v in self.usage_stats.items()} if hasattr(self, 'usage_stats') and self.usage_stats else {}
            yield (await self._stream_event('error', {'message': str(e), 'traceback': traceback.format_exc(), '_agent_usage': partial_usage}))
        finally:
            await self._cleanup_kernel()

    async def _extract_analyst_final_result(self) -> Dict[str, Any]:
        extract_code = '\n                        import json as _json\n\n                        _out = {}\n                        if \'FINAL_RESULT\' in dir():\n                            _fr = FINAL_RESULT\n                            if isinstance(_fr, dict):\n                                # chart: may be a JSON string from fig.to_json() — parse to dict\n                                _chart_raw = _fr.get(\'chart\', None)\n                                if isinstance(_chart_raw, str):\n                                    try:\n                                        _out[\'chart\'] = _json.loads(_chart_raw)\n                                    except Exception:\n                                        _out[\'chart\'] = None  # unparseable chart, skip\n                                elif _chart_raw is not None:\n                                    _out[\'chart\'] = _chart_raw\n\n                                # table: cap to 500 rows to avoid stdout blowup\n                                _table_raw = _fr.get(\'table\', None)\n                                if isinstance(_table_raw, list):\n                                    _out[\'table\'] = _table_raw[:500]\n                                elif hasattr(_table_raw, \'to_dict\'):\n                                    try:\n                                        _out[\'table\'] = _table_raw.head(500).to_dict(orient=\'records\')\n                                    except Exception:\n                                        _out[\'table\'] = []\n                                # else: _table_raw is None — do NOT set _out[\'table\'] = None;\n                                # leaving it unset avoids overwriting a previously computed table\n                                # in final_result when final_result.update(analyst_extras) runs.\n\n                                _out[\'kpis\'] = _fr.get(\'kpis\', None)\n                                _out[\'summary\'] = _fr.get(\'summary\', \'\')\n                                _out[\'viz_type\'] = _fr.get(\'viz_type\', \'chart_and_table\')\n\n                                # If no table/kpis were extracted but FINAL_RESULT is a flat\n                                # scalar dict (e.g. {"avg_basket_size": 25.12, ...}), treat the\n                                # scalar fields as a single-row summary table so they reach the UI.\n                                _reserved_ = {\'chart\', \'table\', \'kpis\', \'summary\', \'viz_type\',\n                                              \'prediction\', \'text_output\', \'dataframe\'}\n                                if \'table\' not in _out and not _out.get(\'kpis\'):\n                                    _scalars_ = {k: v for k, v in _fr.items()\n                                                 if k not in _reserved_\n                                                 and not isinstance(v, (dict, list))}\n                                    if _scalars_:\n                                        _out[\'table\'] = [_scalars_]\n                            elif hasattr(_fr, \'to_dict\'):   # pandas DataFrame\n                                _out[\'table\'] = _fr.head(500).to_dict(orient=\'records\')\n                            else:\n                                _out[\'summary\'] = str(_fr)[:1000]\n\n                        print("__ANALYST_FINAL__:" + _json.dumps(_out, default=str))\n                '
        try:
            result = await self._execute_code(extract_code)
            stdout = result.get('stdout', '')
            exception = result.get('exception', '')
            if exception:
                logger.warning('FINAL_RESULT extraction kernel error: %s', exception[:500])
            import json as _json
            for line in stdout.splitlines():
                if line.startswith('__ANALYST_FINAL__:'):
                    parsed = _json.loads(line[len('__ANALYST_FINAL__:'):])
                    logger.info('Extracted FINAL_RESULT: stdout_length=%d keys=%s chart=%s table_rows=%s kpis=%s', len(stdout), list(parsed.keys()), 'yes' if parsed.get('chart') else 'no', len(parsed['table']) if isinstance(parsed.get('table'), list) else 'none', 'yes' if parsed.get('kpis') else 'no')
                    return parsed
            logger.warning('FINAL_RESULT extraction: __ANALYST_FINAL__ marker not found in stdout (%d chars). Possible stdout truncation.', len(stdout))
        except Exception as e:
            logger.warning('Could not extract FINAL_RESULT from kernel: %s', e)
        return {}

    async def _recover_dataframe_from_completed_iterations(self, completed_iterations: list[dict]) -> Optional[List[Dict[str, Any]]]:
        if not completed_iterations:
            return None
        preferred_tokens = ('result', 'final', 'summary', 'analysis', 'margin', 'comparison')
        candidate_names: list[str] = []
        for step in reversed(completed_iterations):
            vars_info = step.get('variables', {}) or {}
            step_candidates = []
            for name, info in vars_info.items():
                if not isinstance(info, dict) or info.get('type') != 'DataFrame':
                    continue
                if name.startswith('_'):
                    continue
                lname = name.lower()
                score = 0
                if any((tok in lname for tok in preferred_tokens)):
                    score += 4
                if lname.endswith('_df') or lname.startswith('df_') or lname == 'df':
                    score += 2
                step_candidates.append((score, name))
            if step_candidates:
                step_candidates.sort(key=lambda x: x[0], reverse=True)
                candidate_names.extend([name for _, name in step_candidates[:4]])
                break
        if not candidate_names:
            return None
        seen = set()
        ordered_names = []
        for name in candidate_names:
            if name in seen:
                continue
            seen.add(name)
            ordered_names.append(name)
        recover_code = f"""\n                        import json as _json_\n                        import pandas as _pd_\n                        _cands_ = {ordered_names!r}\n                        _rows_ = None\n                        for _nm_ in _cands_:\n                            try:\n                                _obj_ = globals().get(_nm_)\n                                if isinstance(_obj_, _pd_.DataFrame):\n                                    _rows_ = _obj_.head(500).to_dict(orient='records')\n                                    if _rows_:\n                                        break\n                            except Exception:\n                                pass\n                        if _rows_ is not None:\n                            print("__RECOVER_DF__:" + _json_.dumps(_rows_, default=str))\n                        """
        try:
            result = await self._execute_code(recover_code)
            stdout = result.get('stdout', '')
            import json as _json
            for line in stdout.splitlines():
                if line.startswith('__RECOVER_DF__:'):
                    payload = _json.loads(line[len('__RECOVER_DF__:'):])
                    if isinstance(payload, list):
                        return payload
        except Exception as exc:
            logger.warning('Failed to recover dataframe from completed iterations: %s', exc)
        return None

    def _validate_query_aligned_final_result(self, user_query: str, final_result: Dict[str, Any]) -> tuple[bool, str]:
        rows = None
        table_payload = final_result.get('table')
        if isinstance(table_payload, list) and table_payload:
            rows = table_payload
        elif isinstance(table_payload, dict):
            rows = [table_payload]
        if rows is None:
            df_payload = final_result.get('dataframe')
            if isinstance(df_payload, list):
                rows = df_payload
            elif isinstance(df_payload, dict):
                rows = [df_payload]
        if not rows:
            return (False, 'no non-empty table/dataframe present')
        q = (user_query or '').lower()
        compare_markers = ('compare', 'compared', 'vs', 'versus', 'between')
        if any((m in q for m in compare_markers)) and len(rows) < 2:
            return (False, 'comparison query produced fewer than 2 rows')
        quoted_entities = []
        for m in re.finditer('\'([^\']+)\'|\\"([^\\"]+)\\"', user_query or ''):
            entity = (m.group(1) or m.group(2) or '').strip().lower()
            if entity:
                quoted_entities.append(entity)
        quoted_entities = [e for e in quoted_entities if len(e) >= 3]
        if quoted_entities:
            serialized = str(rows[:30]).lower()
            if not any((e in serialized for e in quoted_entities)):
                return (False, 'query entities not reflected in tabular output')
        return (True, '')

    async def _decide_next_action(self, user_query: str, plan_guidance: str, execution_context: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        self._apply_da_decision_context(execution_context)
        return await super()._decide_next_action(user_query=user_query, plan_guidance=plan_guidance, execution_context=execution_context, iteration=iteration)

    async def _decide_next_action_streaming(self, user_query: str, plan_guidance: str, execution_context: Dict[str, Any], iteration: int):
        self._apply_da_decision_context(execution_context)
        async for ev in super()._decide_next_action_streaming(user_query=user_query, plan_guidance=plan_guidance, execution_context=execution_context, iteration=iteration):
            yield ev

    def _apply_da_decision_context(self, execution_context: Dict[str, Any]) -> None:
        time_dims = execution_context.get('time_dimensions', {})
        if time_dims:
            time_summary_lines = []
            for var_name, info in time_dims.items():
                date_cols = info.get('date_columns', [])
                freq = info.get('suggested_freq', 'ME')
                dr = info.get('date_range', {})
                if date_cols:
                    time_summary_lines.append(f"{var_name}: date cols={date_cols}, suggested_freq='{freq}', range={dr.get('min', '?')}→{dr.get('max', '?')} ({dr.get('span_days', '?')} days)")
            if time_summary_lines and (not any(('TIME DIMENSIONS' in w for w in execution_context.get('warnings', [])))):
                execution_context.setdefault('warnings', []).append('TIME DIMENSIONS DETECTED: ' + ' | '.join(time_summary_lines))
        completed = execution_context.get('completed_iterations', [])
        iteration_count = len(completed)
        if iteration_count >= 2 and (not any(('VISUALIZATION DECISION' in w for w in execution_context.get('warnings', [])))):
            viz_enforcement = 'VISUALIZATION DECISION (REQUIRED before any chart code):\nBefore creating any visualization, you MUST check and state in your thinking:\n  1. How many unique groups/categories exist in the result? (print nunique())\n  2. Is there a time dimension?\n  3. Based on count:\n     - 1 group → KPI card ONLY (NO chart — a single-bar chart is useless)\n     - 2-8 groups → Bar/pie chart + table\n     - Time series → Line chart + table\n     - 8+ groups → Table only or top-N chart\n     - Percentage/share → Stacked bar or pie\n  4. The chart MUST add insight beyond the raw numbers\n  5. You are presenting to a CXO — every chart must earn its place'
            execution_context.setdefault('warnings', []).append(viz_enforcement)