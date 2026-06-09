"""
Data Analyst Agent — Iterative self-healing agent for descriptive and diagnostic analytics.

Extends DataScienceAgent with:
- Analytical specialization (trends, KPIs, comparisons, root-cause, distributions)
- Time dimension probing (date column detection + resample frequency suggestion)
- Explicit doom loop detection (OpenCode-inspired: N consecutive identical failures → abort)
- Mandatory chart + table in FINAL_RESULT
- cell_* SSE event naming (unified with DS, set by graph.py coder_node)
- data_analyst_mode=True flag in final state

Usage in LangGraph:
    coder_node (graph.py) calls DataAnalystAgent.execute_analysis() and
    remaps the yielded events to cell_* for frontend consumption.
"""

import difflib
import logging
from datetime import datetime
from util.time_utils import utcnow
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from agents.data_science_agent import DataScienceAgent
from util.xml_prompt_loader import load_client_prompt, BASE_PROMPTS_PATH

logger = logging.getLogger(__name__)


class DataAnalystAgent(DataScienceAgent):
    """
    Self-healing iterative agent for descriptive and diagnostic analytics.

    Inherits all kernel/MCP/LLM infrastructure from DataScienceAgent and overrides:
    - System prompt → data_analyst_agent.xml
    - Config key → "data_analyst_agent"
    - execute_analysis → cell_* events, doom loop detection, time-dimension probing
    - New: _probe_time_dimensions, _detect_doom_loop
    """

    def __init__(
        self,
        agent_name: str = "data_analyst_agent",
        provided_config: Optional[Dict] = None,
        client_id: str = None,
        db: Any = None,
        notebook_output_dir: str = "test_outputs",
        llm_client: Any = None,
        datasource_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        # Delegate to DataScienceAgent.__init__ — it reads AGENT_CONFIG[agent_name]
        super().__init__(
            agent_name=agent_name,
            provided_config=provided_config,
            client_id=client_id,
            db=db,
            notebook_output_dir=notebook_output_dir,
            llm_client=llm_client,
            datasource_context=datasource_context,
            session_id=session_id,
            user_id=user_id,
        )

        # Extra config keys specific to the analyst agent
        self.doom_loop_threshold: int = self.config.get("doom_loop_threshold", 3)
        self.always_generate_chart: bool = self.config.get("always_generate_chart", True)
        self.always_generate_table: bool = self.config.get("always_generate_table", True)

        # Ring buffer of recent failed code snippets for doom-loop detection
        self._recent_failed_codes: List[str] = []

        logger.info(
            "DataAnalystAgent initialized for client '%s' | "
            "provider=%s, model=%s, max_iterations=%d, doom_loop_threshold=%d",
            client_id,
            self.llm_provider,
            self.model,
            self.max_iterations,
            self.doom_loop_threshold,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # System prompt override
    # ──────────────────────────────────────────────────────────────────────────

    def _load_system_prompt(self) -> str:
        """Load data_analyst_agent.xml with multi-tenant support.

        Uses load_client_prompt() so client-specific overrides, MongoDB
        section merges, and custom_prompts.xml are respected.
        """
        import asyncio

        relative_path = f"agents/{self.agent_name}.xml"

        # Try multi-tenant client-aware loading first
        if self.db is not None and self.client_id:
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    return loop.run_until_complete(
                        load_client_prompt(
                            relative_path, self.client_id, self.db,
                            use_formatting=False,
                            datasource_context=self.datasource_context,
                        )
                    )
            except Exception as e:
                logger.warning(
                    "Client-aware prompt loading failed for %s (client=%s), "
                    "falling back to base: %s",
                    self.agent_name, self.client_id, e,
                )

        # Fallback: direct base file read
        try:
            prompt_path = Path(BASE_PROMPTS_PATH) / "agents" / "data_analyst_agent.xml"
            if prompt_path.exists():
                with open(prompt_path, "r") as f:
                    return f.read()
        except Exception as e:
            logger.warning("Could not load data_analyst_agent.xml: %s", e)

        return (
            "You are an expert Data Analyst Agent specialised in descriptive and "
            "diagnostic analytics: trend analysis, KPI tracking, comparisons, "
            "root-cause analysis, distributions, and outlier detection.\n"
            "Generate ONLY executable Python code. Always produce a Plotly chart "
            "and a summary DataFrame in FINAL_RESULT."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # New methods: time-dimension probing & doom-loop detection
    # ──────────────────────────────────────────────────────────────────────────

    async def _probe_time_dimensions(
        self,
        loaded_datasets: List[Dict],
        execution_context: Dict,
    ) -> Dict[str, Any]:
        """
        Detect date/time columns in loaded datasets and suggest resample frequencies.

        Returns:
            {
                "date_columns": {var_name: [col1, col2, ...]},
                "suggested_freq": {var_name: "ME" | "W" | "D" | "QE"},
                "date_range": {var_name: {"min": ..., "max": ..., "span_days": ...}},
            }
        """
        if not self.mcp_client or not loaded_datasets:
            return {}

        probe_code = """
import pandas as pd, json, traceback

time_info = {}
_var_scope = {k: v for k, v in globals().items() if isinstance(v, pd.DataFrame)}

for _var_name, _df in _var_scope.items():
    date_cols = []
    date_range = {}
    suggested_freq = "ME"   # default: monthly

    for col in _df.columns:
        _series = _df[col]
        # Already datetime
        if pd.api.types.is_datetime64_any_dtype(_series):
            date_cols.append(col)
        # String columns that look like dates
        elif _series.dtype == object:
            sample = _series.dropna().head(10)
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors='coerce')
                if parsed.notna().sum() >= min(5, len(sample)):
                    date_cols.append(col)
            except Exception:
                pass

    if date_cols:
        primary = date_cols[0]
        try:
            _ts = pd.to_datetime(_df[primary], errors='coerce').dropna()
            if len(_ts) > 0:
                span_days = (_ts.max() - _ts.min()).days
                date_range = {
                    "min": _ts.min().isoformat(),
                    "max": _ts.max().isoformat(),
                    "span_days": span_days,
                }
                # Suggest frequency based on span
                if span_days <= 14:
                    suggested_freq = "D"
                elif span_days <= 90:
                    suggested_freq = "W"
                elif span_days <= 730:
                    suggested_freq = "ME"
                else:
                    suggested_freq = "QE"
        except Exception as _e:
            date_range = {"error": str(_e)}

    time_info[_var_name] = {
        "date_columns": date_cols,
        "suggested_freq": suggested_freq,
        "date_range": date_range,
    }

print("__TIME_DIM__:" + json.dumps(time_info))
"""
        try:
            result = await self._execute_code(probe_code)
            stdout = result.get("stdout", "")
            for line in stdout.splitlines():
                if line.startswith("__TIME_DIM__:"):
                    import json as _json
                    return _json.loads(line[len("__TIME_DIM__:"):])
        except Exception as e:
            logger.warning("Time dimension probing failed: %s", e)

        return {}

    def _detect_doom_loop(self, current_code: str) -> bool:
        """
        Return True if current_code is nearly identical to the last
        `doom_loop_threshold` failed codes (OpenCode-inspired pattern).

        Similarity threshold: ≥ 0.92 (Ratcliff-Obershelp)
        """
        if len(self._recent_failed_codes) < self.doom_loop_threshold:
            return False

        last_n = self._recent_failed_codes[-self.doom_loop_threshold :]
        for prev in last_n:
            ratio = difflib.SequenceMatcher(None, current_code.strip(), prev.strip()).ratio()
            if ratio < 0.92:
                return False  # At least one failed code was different → not a doom loop

        return True  # All N recent failures are nearly identical → doom loop

    # ──────────────────────────────────────────────────────────────────────────
    # Main analysis loop (overrides DataScienceAgent.execute_analysis)
    # ──────────────────────────────────────────────────────────────────────────

    async def execute_analysis(
        self,
        user_query: str,
        plan: str,
        dataset_path: Optional[str] = None,
        dataset_dict: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> AsyncGenerator[Dict, None]:
        """
        Execute iterative analytical workflow.

        Yields the SAME event types as DataScienceAgent.execute_analysis so the
        graph node (coder_node) can remap them to cell_* for the frontend.

        Event sequence:
            status            — operational messages
            iteration_start   → cell_start
            code_generated    → cell_code
            iteration_execution → cell_result
            iteration_retry   → cell_retry
            iteration_complete → cell_complete
            error             → cell_failed
            final_result      — parsed by graph node into executor_response
        """
        import traceback

        try:
            # ── Notebook init ──────────────────────────────────────────────
            from util.notebook_builder import NotebookBuilder

            self.notebook_builder = NotebookBuilder(
                output_dir=self.notebook_output_dir,
                name_prefix="analysis",
            )
            self.notebook_builder.add_markdown_cell(
                f"# Data Analysis\n\n"
                f"**Query:** {user_query}\n\n"
                f"**Generated:** {utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
                f"---\n\n## Analyst Guidance\n\n{plan}"
            )
            self.notebook_builder.save()

            yield await self._stream_event("status", {
                "message": (
                    f"Analysis started — working iteratively "
                    f"(max {self.max_iterations} iterations)"
                ),
                "max_iterations": self.max_iterations,
                "notebook_path": str(self.notebook_builder.filepath),
            })
            
            # ── Kernel init ────────────────────────────────────────────────

            # ── Kernel init ────────────────────────────────────────────────
            yield await self._stream_event("status", {"message": "Initializing Jupyter kernel..."})
            await self._initialize_kernel()

            # ── Fetch credentials so _is_live_db is evaluated correctly ──
            yield await self._stream_event("status", {"message": "Determine execution mode..."})
            await self._fetch_db_credentials()

            yield await self._stream_event("status", {"message": "Loading dataset..."})
            loaded_datasets = await self._load_dataset_to_kernel(dataset_path, dataset_dict)

            kernel_vars = await self._get_kernel_variables()
            actual_vars = list(kernel_vars.keys()) if kernel_vars else []

            # Force-load fallback (same as DS agent)
            if not actual_vars and not loaded_datasets:
                try:
                    from pathlib import Path as _Path
                    import sys as _sys
                    _root = _Path(__file__).resolve().parent.parent
                    client_data_dir = _root / "assets" / "clients" / self.client_id / "datasets"
                    if client_data_dir.exists():
                        parquet_files = list(client_data_dir.glob("*.parquet"))
                        if parquet_files:
                            main_file = parquet_files[0]
                            force_code = (
                                f"import pandas as pd\n"
                                f"df = pd.read_parquet(r'{main_file}')\n"
                                f"print(f'Force-loaded: shape={{df.shape}}')\n"
                            )
                            await self._execute_code(force_code)
                            kernel_vars = await self._get_kernel_variables()
                            actual_vars = list(kernel_vars.keys())
                            loaded_datasets = [{
                                "path": str(main_file),
                                "variable": "df",
                                "format": "parquet",
                            }]
                except Exception as _fe:
                    logger.warning("Force-load fallback failed: %s", _fe)

            execution_context: Dict[str, Any] = {
                "user_query": user_query,
                "plan_guidance": plan,
                "dataset_path": dataset_path,
                "available_variables": kernel_vars or {},
                "completed_iterations": [],
                "execution_journal": [],     # Compact 1-line-per-iteration journal (tiered prompts)
                "context": context or {},
                "loaded_datasets": loaded_datasets,
                "warnings": [],
            }
            planned_tables = ((context or {}).get("planned_tables") or [])
            self._planned_tables = [str(t).strip() for t in planned_tables if str(t).strip()]
            if self._planned_tables:
                logger.info(
                    "[PromptScope] data_analyst planned_tables=%s",
                    self._planned_tables,
                )

            # ── Inject llm_query helper ────────────────────────────────────
            yield await self._stream_event("status", {"message": "Injecting llm_query() helper..."})
            await self._inject_llm_query_helper()

            # ── Schema probing ─────────────────────────────────────────────
            yield await self._stream_event("status", {"message": "Reading schemas..."})
            file_schemas = await self._probe_parquet_schemas(
                loaded_datasets, dataset_path
            )
            execution_context["file_schemas"] = self._merge_live_db_schemas_from_plan(
                file_schemas,
                plan,
            )

            # ── Dataset profiling ──────────────────────────────────────────
            yield await self._stream_event("status", {"message": "Profiling dataset..."})
            execution_context["data_profile"] = await self._probe_dataset_profile()

            # ── Load business knowledge (once — filtered per-iteration) ────
            # In adhoc mode, skip backend table descriptions.
            is_adhoc = (context or {}).get("adhoc_mode", False)
            if is_adhoc:
                execution_context["knowledge_context"] = {}
                execution_context["adhoc_mode"] = True
                logger.info("Adhoc mode: skipping backend knowledge loading (DA)")
            else:
                try:
                    execution_context["knowledge_context"] = self._load_knowledge_for_coding()
                except Exception as e:
                    logger.warning("Knowledge loading failed (non-fatal): %s", e)
                    execution_context["knowledge_context"] = {}

            # ── Time dimension probing (NEW — analyst-specific) ────────────
            yield await self._stream_event("status", {"message": "Detecting time dimensions..."})
            time_dims = await self._probe_time_dimensions(loaded_datasets, execution_context)
            if time_dims:
                execution_context["time_dimensions"] = time_dims
                logger.info("Time dimensions detected: %s", time_dims)

            # ═══════════════════════════════════════════════════════════════
            # RECURSIVE EXECUTION LOOP
            # ═══════════════════════════════════════════════════════════════
            iteration = 0
            status = "continue"
            consecutive_failures = 0
            MAX_CONSECUTIVE_FAILURES = self.doom_loop_threshold
            self._recent_failed_codes = []

            while iteration < self.max_iterations and status == "continue":
                iteration += 1

                # ── Pre-iteration guard: stop if FINAL_RESULT already set ──
                if iteration > 1 and "FINAL_RESULT" in execution_context.get("available_variables", {}):
                    logger.info(
                        "Iteration %d: FINAL_RESULT already in kernel from "
                        "previous iteration — stopping immediately.",
                        iteration,
                    )
                    if self.notebook_builder:
                        self.notebook_builder.add_markdown_cell(
                            "## Analysis Complete\n\n"
                            "FINAL_RESULT was set in a previous iteration "
                            "— no further iterations needed."
                        )
                        self.notebook_builder.save()
                    status = "done"
                    break

                yield await self._stream_event("status", {
                    "message": (
                        f"Iteration {iteration}/{self.max_iterations} — "
                        "deciding next action..."
                    )
                })

                # ── Ask LLM: what next? ────────────────────────────────────
                try:
                    decision = await self._decide_next_action(
                        user_query=user_query,
                        plan_guidance=plan,
                        execution_context=execution_context,
                        iteration=iteration,
                    )
                except Exception as e:
                    logger.error("Iteration %d: _decide_next_action failed: %s", iteration, e)
                    yield await self._stream_event("error", {
                        "message": f"Decision-making failed at iteration {iteration}: {e}",
                        "iteration": iteration,
                    })
                    status = "error"
                    break

                action = decision.get("action", "code")
                reasoning = decision.get("reasoning", "")
                thinking = decision.get("thinking", "")
                code = decision.get("code", "")

                # ── Done? ──────────────────────────────────────────────────
                if action == "done":
                    logger.info("Iteration %d: LLM declared DONE — %s", iteration, reasoning)
                    yield await self._stream_event("iteration_complete", {
                        "iteration": iteration,
                        "action": "done",
                        "reasoning": reasoning,
                    })
                    if self.notebook_builder:
                        self.notebook_builder.add_markdown_cell(
                            f"## ✅ Analysis Complete (iteration {iteration})\n\n{reasoning}"
                        )
                        self.notebook_builder.save()
                    status = "done"
                    break

                # ── Stream iteration_start ─────────────────────────────────
                yield await self._stream_event("iteration_start", {
                    "iteration": iteration,
                    "max_iterations": self.max_iterations,
                    "reasoning": reasoning,
                })

                if self.notebook_builder:
                    self.notebook_builder.add_markdown_cell(
                        f"## Iteration {iteration}: {reasoning}"
                    )
                    self.notebook_builder.save()

                logger.info("Iteration %d/%d: %s", iteration, self.max_iterations, reasoning)

                # ── Retry inner loop ───────────────────────────────────────
                iteration_success = False
                last_error: Optional[str] = None
                _stashed_failed_code = None
                _stashed_error_type = None

                for attempt in range(self.max_retries_per_iteration):
                    try:
                        if attempt > 0:
                            code = await self._regenerate_code_after_error(
                                user_query=user_query,
                                plan_guidance=plan,
                                execution_context=execution_context,
                                iteration=iteration,
                                failed_code=code,
                                error=last_error,
                                attempt=attempt,
                            )

                        if not code:
                            last_error = "Failed to generate code"
                            continue

                        # ── Doom-loop check (before execution) ────────────
                        if self._detect_doom_loop(code):
                            doom_msg = (
                                f"Doom loop detected at iteration {iteration}: the last "
                                f"{self.doom_loop_threshold} failed attempts used nearly "
                                "identical code. Aborting to prevent wasted compute."
                            )
                            logger.warning(doom_msg)
                            yield await self._stream_event("error", {
                                "message": doom_msg,
                                "iteration": iteration,
                                "last_error": last_error or "",
                            })
                            status = "error"
                            break  # break inner loop

                        # ── AST validation ────────────────────────────────
                        validation_err = self._validate_code_syntax(
                            code, execution_context.get("available_variables", {})
                        )
                        if validation_err:
                            logger.warning(
                                "Iteration %d: AST validation failed: %s",
                                iteration, validation_err,
                            )
                            last_error = f"Code validation error: {validation_err}"
                            self._recent_failed_codes.append(code)
                            continue

                        yield await self._stream_event("code_generated", {
                            "iteration": iteration,
                            "code": code,
                            "attempt": attempt + 1,
                        })

                        if self.notebook_builder:
                            self.notebook_builder.add_code_cell(code)
                            self.notebook_builder.save()

                        # ── Execute ────────────────────────────────────────
                        execution_result = await self._execute_code(code)

                        yield await self._stream_event("iteration_execution", {
                            "iteration": iteration,
                            "attempt": attempt + 1,
                            "stdout": execution_result.get("stdout", ""),
                            "stderr": execution_result.get("stderr", ""),
                            "exception": execution_result.get("exception"),
                        })

                        if self.notebook_builder and execution_result.get("stdout"):
                            self.notebook_builder.add_output_to_last_cell(
                                execution_result.get("stdout", "")
                            )
                            self.notebook_builder.save()

                        # ── Error detection ────────────────────────────────
                        detected_error = execution_result.get("exception")
                        if not detected_error and self._stdout_contains_error(
                            execution_result.get("stdout", "")
                        ):
                            detected_error = self._extract_error_from_stdout(
                                execution_result.get("stdout", "")
                            )
                            execution_result["exception"] = detected_error
                            logger.info(
                                "Iteration %d: detected error in stdout: %s",
                                iteration, str(detected_error)[:150],
                            )

                        if detected_error:
                            last_error = detected_error
                            _stashed_failed_code = code
                            _stashed_error_type, _, _ = self._classify_error(detected_error)
                            logger.warning(
                                "Iteration %d failed (attempt %d): %s",
                                iteration, attempt + 1, str(last_error)[:200],
                            )

                            # Track for doom-loop detection
                            self._recent_failed_codes.append(code)
                            # Keep ring buffer bounded
                            if len(self._recent_failed_codes) > self.doom_loop_threshold * 2:
                                self._recent_failed_codes = self._recent_failed_codes[
                                    -self.doom_loop_threshold :
                                ]

                            # Diagnostic probe on first failure
                            if attempt == 0:
                                diag_code = await self._generate_diagnostic_code(
                                    code, last_error,
                                    execution_context.get("available_variables", {}),
                                )
                                if diag_code:
                                    diag_result = await self._execute_code(diag_code)
                                    diag_output = diag_result.get("stdout", "")
                                    if diag_output:
                                        last_error += (
                                            f"\n\nDIAGNOSTIC OUTPUT:\n{diag_output[:300]}"
                                        )

                            if attempt < self.max_retries_per_iteration - 1:
                                yield await self._stream_event("iteration_retry", {
                                    "iteration": iteration,
                                    "attempt": attempt + 1,
                                    "error": last_error,
                                    "message": "Retrying with error feedback...",
                                })
                            continue

                        # ── Success ────────────────────────────────────────
                        iteration_success = True
                        self._recent_failed_codes = []  # reset doom-loop buffer on success

                        # --- Lesson extraction hooks (fire-and-forget) ---
                        try:
                            raw_db = self._get_raw_db()
                            if raw_db:
                                from services.lesson_extractor import LessonExtractor
                                from services.agent_lesson_service import AgentLessonService
                                _lesson_svc = AgentLessonService(raw_db)

                                # Hook 1: Error recovery — extract lesson from diff
                                if attempt > 0 and _stashed_failed_code:
                                    lessons = LessonExtractor.extract_from_error_recovery(
                                        error_type=_stashed_error_type or "UNKNOWN",
                                        error_text=last_error or "",
                                        failed_code=_stashed_failed_code,
                                        fixed_code=code,
                                        file_schemas=execution_context.get("file_schemas", {}),
                                    )
                                    if lessons:
                                        logger.info("Lesson hook 1 (error recovery): extracted %d lesson(s)", len(lessons))
                                    for lsn in lessons:
                                        await _lesson_svc.save_lesson(self.client_id, lsn)

                                # Hook 2: Code pattern — scan successful code
                                pattern_lessons = LessonExtractor.extract_from_code_pattern(code)
                                if pattern_lessons:
                                    logger.info("Lesson hook 2 (code pattern): extracted %d lesson(s)", len(pattern_lessons))
                                for lsn in pattern_lessons:
                                    await _lesson_svc.save_lesson(self.client_id, lsn)
                        except Exception as _le:
                            logger.debug("Lesson extraction skipped: %s", _le)

                        # Lightweight FINAL_RESULT check — runs a tiny
                        # kernel snippet that can't fail, unlike the full
                        # _get_kernel_variables() which can blow up on
                        # large Plotly JSON in FINAL_RESULT.
                        fr_exists = await self._check_final_result_in_kernel()

                        prev_vars = execution_context.get("available_variables", {})
                        new_vars = await self._get_kernel_variables()
                        if new_vars:
                            execution_context["available_variables"] = new_vars
                        elif prev_vars:
                            # Introspection failed silently — keep previous vars
                            # so we don't lose FINAL_RESULT or other state
                            logger.warning(
                                "Iteration %d: _get_kernel_variables() returned "
                                "empty but previous vars existed (%d vars). "
                                "Keeping previous variables.",
                                iteration, len(prev_vars),
                            )
                            new_vars = prev_vars

                        # Ensure FINAL_RESULT is in available_variables if
                        # the lightweight check found it (even if
                        # _get_kernel_variables missed it)
                        if fr_exists and "FINAL_RESULT" not in new_vars:
                            logger.warning(
                                "Iteration %d: FINAL_RESULT exists in kernel "
                                "but _get_kernel_variables() missed it!",
                                iteration,
                            )
                            new_vars["FINAL_RESULT"] = {"type": "dict"}
                            execution_context["available_variables"] = new_vars

                        # Re-profile when new DataFrames appear in the kernel
                        existing_profile_keys = set(
                            execution_context.get("data_profile", {}).keys()
                        )
                        current_df_names = {
                            name for name, info in new_vars.items()
                            if isinstance(info, dict) and info.get("type") == "DataFrame"
                        }
                        if not existing_profile_keys or current_df_names != existing_profile_keys:
                            new_profile = await self._probe_dataset_profile()
                            if new_profile:
                                execution_context["data_profile"] = new_profile
                                logger.info(
                                    "Re-profiled after iteration %d: new=%s",
                                    iteration,
                                    current_df_names - existing_profile_keys,
                                )
                                # Hook 3: Data profile lessons
                                try:
                                    raw_db = self._get_raw_db()
                                    if raw_db:
                                        from services.lesson_extractor import LessonExtractor
                                        from services.agent_lesson_service import AgentLessonService
                                        profile_lessons = LessonExtractor.extract_from_data_profile(
                                            new_profile, execution_context.get("file_schemas", {})
                                        )
                                        if profile_lessons:
                                            logger.info("Lesson hook 3 (data profile): extracted %d lesson(s)", len(profile_lessons))
                                            _lsvc = AgentLessonService(raw_db)
                                            for lsn in profile_lessons:
                                                await _lsvc.save_lesson(self.client_id, lsn)
                                except Exception:
                                    pass  # Non-fatal

                        # Silent-failure check (empty DataFrames, etc.)
                        is_valid, validation_issue = await self._validate_step_output(
                            {"step_num": iteration, "description": reasoning},
                            new_vars,
                            prev_vars,
                        )
                        if not is_valid and attempt < self.max_retries_per_iteration - 1:
                            # Active self-correction: run diagnostic and retry
                            diag_code = self._generate_zero_row_diagnostic(
                                validation_issue, new_vars
                            )
                            diag_output = ""
                            if diag_code:
                                diag_result = await self._execute_code(diag_code)
                                diag_output = diag_result.get("stdout", "")[:500]

                            last_error = (
                                f"ZERO_ROW_RESULT: {validation_issue}. "
                                f"The filter/join produced an empty DataFrame. "
                                f"This likely means wrong column or wrong values "
                                f"were used for filtering. Re-check which column "
                                f"in the target table corresponds to the lookup "
                                f"value. Try alternative columns."
                            )
                            if diag_output:
                                last_error += (
                                    f"\n\nDIAGNOSTIC (unique values in "
                                    f"related columns):\n{diag_output}"
                                )

                            logger.warning(
                                "Iteration %d: zero-row self-correction "
                                "triggered: %s",
                                iteration, validation_issue,
                            )
                            iteration_success = False
                            yield await self._stream_event("iteration_retry", {
                                "iteration": iteration,
                                "attempt": attempt + 1,
                                "error": last_error,
                                "message": "Zero-row result detected — retrying "
                                           "with diagnostic context...",
                            })
                            continue  # Go back to retry loop
                        elif not is_valid:
                            logger.warning(
                                "Iteration %d silent failure (no retries left): %s",
                                iteration, validation_issue,
                            )
                            execution_context["warnings"].append(
                                f"Iteration {iteration}: ZERO_ROW_RESULT: "
                                f"{validation_issue}"
                            )

                        # Detect cartesian joins (row explosion)
                        explosion_warnings = self._detect_row_explosion(
                            new_vars, prev_vars
                        )
                        for w in explosion_warnings:
                            logger.warning(
                                "Iteration %d: %s", iteration, w
                            )
                            execution_context["warnings"].append(
                                f"Iteration {iteration}: {w}"
                            )

                        # Truncate long stdout before storing
                        raw_output = execution_result.get("stdout", "")
                        if len(raw_output) > self.output_storage_max_chars:
                            half = self.output_storage_max_chars // 2
                            raw_output = (
                                raw_output[:half]
                                + "\n...[truncated]...\n"
                                + raw_output[-half:]
                            )

                        execution_context["completed_iterations"].append({
                            "iteration": iteration,
                            "reasoning": reasoning,
                            "thinking": thinking,
                            "code": code,
                            "output": raw_output,
                            "variables": new_vars,
                        })

                        # Append compact journal entry + register artifact (tiered prompts)
                        execution_context.setdefault("execution_journal", []).append(
                            self._build_journal_entry(iteration, reasoning, new_vars, prev_vars)
                        )
                        self._register_artifact(iteration, reasoning, new_vars, prev_vars)

                        yield await self._stream_event("iteration_complete", {
                            "iteration": iteration,
                            "reasoning": reasoning,
                            "available_variables": list(new_vars.keys()),
                        })

                        # Context compaction (legacy path only — tiered prompts use journal instead)
                        from config.system_config import USE_TIERED_PROMPTS
                        if not USE_TIERED_PROMPTS:
                            completed = execution_context["completed_iterations"]
                            if (
                                len(completed) % self.context_compaction_interval == 0
                                and len(completed) >= self.context_compaction_interval
                            ):
                                try:
                                    n = self.context_compaction_interval
                                    batch = completed[-n:]
                                    summary_text = await self._summarize_completed_steps(batch)
                                    execution_context["completed_iterations"] = completed[:-n] + [{
                                        "iteration": (
                                            f"summary({batch[0]['iteration']}"
                                            f"-{batch[-1]['iteration']})"
                                        ),
                                        "reasoning": summary_text,
                                        "code": "",
                                        "output": "",
                                        "variables": new_vars,
                                    }]
                                    logger.info(
                                        "Context compacted: summarized iterations %s-%s",
                                        batch[0]["iteration"], batch[-1]["iteration"],
                                    )
                                except Exception as compact_err:
                                    logger.warning("Context compaction skipped: %s", compact_err)

                        logger.info("Iteration %d completed successfully", iteration)
                        consecutive_failures = 0

                        # Early stop: if FINAL_RESULT was set, auto-declare done
                        logger.debug(
                            "Iteration %d: checking for FINAL_RESULT in new_vars. "
                            "Keys: %s",
                            iteration,
                            list(new_vars.keys()) if new_vars else "EMPTY",
                        )
                        if "FINAL_RESULT" in new_vars:
                            logger.info(
                                "Iteration %d: FINAL_RESULT detected in kernel — auto-done",
                                iteration,
                            )
                            if self.notebook_builder:
                                self.notebook_builder.add_markdown_cell(
                                    f"## Analysis Complete (iteration {iteration})\n\n"
                                    "FINAL_RESULT was set — stopping."
                                )
                                self.notebook_builder.save()
                            status = "done"

                        break  # exit retry loop

                    except Exception as exc:
                        last_error = str(exc)
                        logger.error(
                            "Error in iteration %d, attempt %d: %s",
                            iteration, attempt + 1, exc,
                        )
                        if attempt < self.max_retries_per_iteration - 1:
                            yield await self._stream_event("iteration_retry", {
                                "iteration": iteration,
                                "attempt": attempt + 1,
                                "error": str(exc),
                            })

                # ── Doom loop broke us out of inner loop ───────────────────
                if status == "error":
                    break

                if not iteration_success:
                    consecutive_failures += 1
                    # Record the failure so the NEXT _decide_next_action knows what failed
                    execution_context.setdefault("failed_iterations", []).append({
                        "iteration": iteration,
                        "error": (last_error or "unknown")[:500],
                        "code_snippet": (code or "")[:300],
                    })
                    logger.warning(
                        "Iteration %d failed after %d attempts (consecutive: %d)",
                        iteration, self.max_retries_per_iteration, consecutive_failures,
                    )

                    if self.notebook_builder:
                        self.notebook_builder.add_markdown_cell(
                            f"❌ **Iteration {iteration} FAILED** after "
                            f"{self.max_retries_per_iteration} attempts.\n\n"
                            f"Last error: `{str(last_error)[:300] if last_error else 'Unknown'}`"
                        )
                        self.notebook_builder.save()

                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        yield await self._stream_event("error", {
                            "message": (
                                f"{consecutive_failures} consecutive iterations failed. "
                                "Stopping to prevent wasted compute."
                            ),
                            "iteration": iteration,
                            "last_error": last_error,
                        })
                        status = "error"
                        break

            # ═══════════════════════════════════════════════════════════════
            # POST-LOOP: Build final result
            # ═══════════════════════════════════════════════════════════════
            completed_count = len(execution_context["completed_iterations"])

            if status == "error" or completed_count == 0:
                failure_msg = (
                    f"Analysis incomplete — {completed_count} iterations completed, "
                    "stopped due to errors."
                )
                final_result = {
                    "prediction": failure_msg,
                    "text_output": failure_msg,
                    "dataframe": None,
                    "plotly_charts": [],
                    "iterations_completed": completed_count,
                    "timestamp": utcnow().isoformat(),
                    "pipeline_failed": True,
                    "_agent_usage": {
                        k: list(v) if isinstance(v, set) else v
                        for k, v in self.usage_stats.items()
                    },
                }
            else:
                yield await self._stream_event("status", {"message": "Fetching result data..."})
                final_df_records = await self._fetch_generated_dataframe()

                yield await self._stream_event("status", {"message": "Generating final result..."})
                # Build result, then enrich with chart/table from FINAL_RESULT kernel variable
                final_result = await self._generate_final_result(execution_context)
                final_result["dataframe"] = final_df_records

                # Extract chart + table from kernel's FINAL_RESULT if present
                analyst_extras = await self._extract_analyst_final_result()
                if analyst_extras:
                    final_result.update(analyst_extras)

                # Fetch ALL Plotly charts from the kernel namespace
                all_charts = await self._fetch_all_generated_charts()
                if len(all_charts) > 1:
                    # Multiple charts found — use the full array
                    final_result["charts"] = all_charts
                    if not final_result.get("chart"):
                        final_result["chart"] = all_charts[0]["figure"]
                elif len(all_charts) == 1 and not final_result.get("chart"):
                    # Single chart found but no chart from FINAL_RESULT — use it
                    final_result["chart"] = all_charts[0]["figure"]
                    final_result["charts"] = all_charts

                if self.notebook_builder:
                    summary_text = final_result.get(
                        "text_output", final_result.get("prediction", "")
                    )
                    self.notebook_builder.add_markdown_cell(
                        f"---\n\n## Final Result\n\n{summary_text}"
                    )
                    nb_path = self.notebook_builder.save()
                    final_result["notebook_path"] = str(nb_path)

            yield await self._stream_event("final_result", final_result)

            yield await self._stream_event("status", {
                "message": (
                    f"Analysis complete ({completed_count} iterations)"
                    if status != "error"
                    else "Analysis incomplete due to errors"
                ),
                "notebook_path": (
                    str(self.notebook_builder.filepath)
                    if self.notebook_builder
                    else None
                ),
            })

        except Exception as e:
            logger.error("Error in execute_analysis: %s\n%s", e, traceback.format_exc())
            # Preserve accumulated token usage even on error
            partial_usage = {
                k: list(v) if isinstance(v, set) else v
                for k, v in self.usage_stats.items()
            } if hasattr(self, 'usage_stats') and self.usage_stats else {}
            yield await self._stream_event("error", {
                "message": str(e),
                "traceback": traceback.format_exc(),
                "_agent_usage": partial_usage,
            })
        finally:
            await self._cleanup_kernel()

    # ──────────────────────────────────────────────────────────────────────────
    # Extract FINAL_RESULT from kernel (analyst variant)
    # ──────────────────────────────────────────────────────────────────────────

    async def _extract_analyst_final_result(self) -> Dict[str, Any]:
        """
        Read FINAL_RESULT from the kernel and extract chart JSON, table records,
        kpis, and summary text.  Returns an empty dict if not found.
        """
        extract_code = r"""
import json as _json

_out = {}
if 'FINAL_RESULT' in dir():
    _fr = FINAL_RESULT
    if isinstance(_fr, dict):
        _out['chart'] = _fr.get('chart', None)
        _out['table'] = _fr.get('table', None)
        _out['kpis'] = _fr.get('kpis', None)
        _out['summary'] = _fr.get('summary', '')
    elif hasattr(_fr, 'to_dict'):   # pandas DataFrame
        _out['table'] = _fr.to_dict(orient='records')
    else:
        _out['summary'] = str(_fr)[:1000]

print("__ANALYST_FINAL__:" + _json.dumps(_out, default=str))
"""
        try:
            result = await self._execute_code(extract_code)
            stdout = result.get("stdout", "")
            import json as _json
            for line in stdout.splitlines():
                if line.startswith("__ANALYST_FINAL__:"):
                    return _json.loads(line[len("__ANALYST_FINAL__:"):])
        except Exception as e:
            logger.warning("Could not extract FINAL_RESULT from kernel: %s", e)

        return {}

    # ──────────────────────────────────────────────────────────────────────────
    # Override _decide_next_action to inject time-dimension context
    # ──────────────────────────────────────────────────────────────────────────

    async def _decide_next_action(
        self,
        user_query: str,
        plan_guidance: str,
        execution_context: Dict[str, Any],
        iteration: int,
    ) -> Dict[str, Any]:
        """
        Extends DataScienceAgent._decide_next_action with:
        - Time-dimension probing context
        - Visualization decision enforcement (for later iterations)
        """
        # Inject time-dimension info into context so parent builds the right prompt
        time_dims = execution_context.get("time_dimensions", {})
        if time_dims:
            time_summary_lines = []
            for var_name, info in time_dims.items():
                date_cols = info.get("date_columns", [])
                freq = info.get("suggested_freq", "ME")
                dr = info.get("date_range", {})
                if date_cols:
                    time_summary_lines.append(
                        f"{var_name}: date cols={date_cols}, "
                        f"suggested_freq='{freq}', "
                        f"range={dr.get('min', '?')}→{dr.get('max', '?')} "
                        f"({dr.get('span_days', '?')} days)"
                    )
            if time_summary_lines and not any(
                "TIME DIMENSIONS" in w for w in execution_context.get("warnings", [])
            ):
                execution_context.setdefault("warnings", []).append(
                    "TIME DIMENSIONS DETECTED: " + " | ".join(time_summary_lines)
                )

        # Inject visualization decision enforcement for later iterations
        completed = execution_context.get("completed_iterations", [])
        iteration_count = len(completed)
        if iteration_count >= 2 and not any(
            "VISUALIZATION DECISION" in w
            for w in execution_context.get("warnings", [])
        ):
            viz_enforcement = (
                "VISUALIZATION DECISION (REQUIRED before any chart code):\n"
                "Before creating any visualization, you MUST check and state in your thinking:\n"
                "  1. How many unique groups/categories exist in the result? (print nunique())\n"
                "  2. Is there a time dimension?\n"
                "  3. Based on count:\n"
                "     - 1 group → KPI card ONLY (NO chart — a single-bar chart is useless)\n"
                "     - 2-8 groups → Bar/pie chart + table\n"
                "     - Time series → Line chart + table\n"
                "     - 8+ groups → Table only or top-N chart\n"
                "     - Percentage/share → Stacked bar or pie\n"
                "  4. The chart MUST add insight beyond the raw numbers\n"
                "  5. You are presenting to a CXO — every chart must earn its place"
            )
            execution_context.setdefault("warnings", []).append(viz_enforcement)

        return await super()._decide_next_action(
            user_query=user_query,
            plan_guidance=plan_guidance,
            execution_context=execution_context,
            iteration=iteration,
        )
