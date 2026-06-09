"""
Data Science Agent for Iterative Code Generation and Execution
Generates Python code, executes it in a Jupyter kernel via MCP server,
reads outputs, and iteratively refines to produce predictions/insights.
"""

import asyncio
import difflib
import hashlib
import json
import logging
import os
import re
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
from util.datasource_context import (
    get_client_dataset_storage_root,
    get_datasource_namespace_suffix,
    resolve_client_metadata_dir,
    resolve_client_metadata_path,
)
from util.xml_prompt_loader import load_xml_prompt_raw, load_client_prompt, load_client_data_descriptions, BASE_PROMPTS_PATH, CLIENTS_PROMPTS_PATH
from util.kernel_manager import get_kernel_manager, release_kernel_manager, DockerKernelManager, LocalKernelManager
from util.jupyter_env import ensure_jupyter_contents_dirs
from util.mcp.client import McpClient, McpError, McpTimeoutError
from util.notebook_builder import NotebookBuilder
from mcp.client.stdio import stdio_client, StdioServerParameters
import traceback
from util.sql_validator import SQLValidatorFactory
from services.session_memory import session_memory

logger = logging.getLogger(__name__)

# db_type values that represent a live database (not file uploads)
_LIVE_DB_TYPES = {"postgres", "mysql", "mongodb", "sqlserver", "sap_oracle", "sap_hana", "sap_sybase"}


def _check_jupyter_mcp_timeout_patch():
    """Warn once if the jupyter_mcp_server execute_code timeout patch is missing."""
    try:
        from patch_jupyter_mcp import find_server_py, check_patch
        if not check_patch(find_server_py()):
            logger.warning(
                "jupyter_mcp_server execute_code timeout is capped at 60s. "
                "ML/forecasting code will be killed early. "
                "Fix: conda activate coresight && python patch_jupyter_mcp.py"
            )
    except Exception:
        pass


_check_jupyter_mcp_timeout_patch()


class DataScienceAgent:
    """
    Iterative Data Science Agent that:
    1. Generates Python code based on user query
    2. Executes code in Jupyter kernel via MCP server
    3. Reads and analyzes outputs (dataframes, plots, metrics)
    4. Refines approach based on results
    5. Produces final predictions/insights
    """

    def __init__(
        self, 
        agent_name: str = "data_science_agent",
        provided_config: Optional[Dict] = None,
        client_id: str = None,
        db: Any = None,
        notebook_output_dir: str = "test_outputs",
        llm_client: Optional[LLMClient] = None,
        datasource_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Initialize the Data Science Agent.
        
        Args:
            agent_name: Name of the agent configuration
            provided_config: Optional config override
            client_id: The client ID for multi-tenant operation (REQUIRED)
            db: MongoDB database instance
            llm_client: Shared LLMClient instance from graph state (REQUIRED for graph usage)
        """
        if not client_id:
            raise ValueError(
                "client_id is REQUIRED for multi-tenant operation. "
                "No default client exists. Every request must specify a valid client_id."
            )
        
        self.agent_name = agent_name
        self.client_id = client_id
        self.db = db
        self.datasource_context = datasource_context
        self.session_id = session_id or ""
        self.user_id = user_id or ""
        self.config = provided_config or AGENT_CONFIG.get(self.agent_name, {})
        
        # --- Initialize LLMClient ---
        # Use shared LLMClient from graph state (REQUIRED for graph usage)
        if llm_client is None:
            raise ValueError(
                f"llm_client is REQUIRED for {self.agent_name}. "
                "When using agents in the graph, pass the shared LLMClient from state."
            )
        self.llm_client = llm_client
        
        # Load system prompts
        self.base_prompt = self._load_system_prompt()
        
        # MCP-based kernel communication setup
        self.kernel_manager: Optional[LocalKernelManager] = None
        self.mcp_client: Optional[McpClient] = None
        self._stdio_context_manager = None  # Store stdio context manager for cleanup
        self._mcp_context_manager = None  # Store MCP client context manager for cleanup
        self.execution_history: List[Dict] = []
        self.variables_state: Dict[str, Any] = {}
        self._data_dir: str = "/data"  # Will be updated after kernel init
        self._planned_tables: List[str] = []  # Tables identified by planner
        self._is_live_db: bool = False  # Set after _fetch_db_credentials
        self.db_credentials_env: Dict[str, str] = {}
        self._dataset_volume_mounted: bool = False

        # MCP notebook/kernel identity (for persistent state)
        self._mcp_notebook_name: Optional[str] = None
        self._mcp_notebook_path: Optional[str] = None
        self._mcp_kernel_id: Optional[str] = None
        self._session_kernel_reused: bool = False
        
        # Notebook generation
        self.notebook_output_dir = notebook_output_dir
        self.notebook_builder: Optional[NotebookBuilder] = None
        
        # Token usage tracking
        self.usage_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "models": set()
        }
        
        # Configuration — read from system_config.py AGENT_CONFIG["data_science_agent"]
        self.llm_provider = self.config.get("llm_provider", DEFAULT_LLM_PROVIDER)
        self.max_iterations = self.config.get("max_iterations", 15)
        self.temperature = self.config.get("temperature", 0.0)
        self.reasoning_effort = self.config.get("reasoning_effort", None)
        self.timeout_per_execution = self.config.get("timeout_per_execution", 180)
        self.idle_timeout_minutes = self.config.get("idle_timeout_minutes", 30.0)
        self.session_kernel_ttl_seconds = int(self.config.get("session_kernel_ttl_seconds", 1800))

        # Recursive loop configuration
        self.max_retries_per_iteration = self.config.get("max_retries_per_iteration", 3)
        self.retry_temperatures = self.config.get("retry_temperatures", [0.0, 0.25, 0.50])
        self.context_compaction_interval = self.config.get("context_compaction_interval", 3)
        self.code_preview_max_chars = self.config.get("code_preview_max_chars", 1200)
        self.output_preview_max_chars = self.config.get("output_preview_max_chars", 1500)
        self.output_storage_max_chars = self.config.get("output_storage_max_chars", 1000)
        self.string_values_top_n = self.config.get("string_values_top_n", 5)
        self.max_journal_detail_entries = self.config.get("max_journal_detail_entries", 1)

        # Doom loop detection (shared with DataAnalystAgent which inherits from this class)
        self.doom_loop_threshold: int = self.config.get("doom_loop_threshold", 3)
        self._recent_failed_codes: List[str] = []

        # --- Tiered Prompt Cache (built once per execute_analysis call) ---
        self._static_system_context: Optional[str] = None
        self._cached_lessons_text: Optional[str] = None
        self._cached_prefs_text: Optional[str] = None
        self._artifact_registry: List[Dict[str, Any]] = []
        self._profiled_datasets: set = set()
        self._previous_profile_shapes: Dict[str, Any] = {}

        # Resolve actual model name: use config's model_name, or fall back to
        # the provider's default_model defined in LLM_PROVIDERS.
        configured_model = self.config.get("model_name")
        if configured_model:
            self.model = configured_model
        else:
            provider_cfg = LLM_PROVIDERS.get(self.llm_provider, {})
            self.model = provider_cfg.get("default_model", "gpt-4")
        
        logger.info(
            f"DataScienceAgent initialized for client '{client_id}' | "
            f"provider={self.llm_provider}, model={self.model}, "
            f"max_iterations={self.max_iterations}, temperature={self.temperature}, "
            f"timeout={self.timeout_per_execution}s"
        )

    def _get_client_dataset_dir(self) -> Path:
        dataset_dir = get_client_dataset_storage_root(
            self.client_id,
            datasource_context=self.datasource_context,
        )
        return dataset_dir

    def _matches_planned_tables(self, filename: str) -> bool:
        """Check if a filename matches any of the planner's requested tables.

        Uses case-insensitive substring matching.
        E.g., planned table "PSTK" matches "20260222_161749_IFFCO_INV_AI_PSTK.parquet"
        """
        if not self._planned_tables:
            return True  # No filter — show all files
        fname_upper = filename.upper()
        return any(table.upper() in fname_upper for table in self._planned_tables)

    def _merge_live_db_schemas_from_plan(
        self,
        file_schemas: Optional[Dict[str, Any]],
        plan: str,
    ) -> Dict[str, Any]:
        """
        Reconstruct live DB table schemas from planner guidance when no parquet
        metadata is available.
        """
        merged_schemas: Dict[str, Any] = dict(file_schemas or {})
        if not getattr(self, "_is_live_db", False) or not plan:
            return merged_schemas

        table_names: List[str] = []
        if self._planned_tables:
            table_names.extend(self._planned_tables)

        table_names.extend(
            match.strip()
            for match in re.findall(
                r"<data_description[^>]*table_name=[\x27\x22]([^\x27\x22]+)[\x27\x22]",
                plan,
                re.IGNORECASE,
            )
            if match and match.strip()
        )
        table_names.extend(
            match.strip()
            for match in re.findall(
                r"TABLE\s+([A-Za-z0-9_.$#]+)\s*:",
                plan,
                re.IGNORECASE,
            )
            if match and match.strip()
        )

        seen_tables = set()
        recovered_tables = 0

        for raw_table_name in table_names:
            table_name = raw_table_name.strip()
            normalized_name = table_name.lower()
            if not table_name or normalized_name in seen_tables:
                continue
            seen_tables.add(normalized_name)

            columns: List[str] = []
            types: Dict[str, str] = {}

            csv_match = re.search(
                r"TABLE\s+" + re.escape(table_name) + r"\s*:\s*([^\n]+)",
                plan,
                re.IGNORECASE,
            )
            if csv_match and "<" not in csv_match.group(1):
                for col_def in csv_match.group(1).split(","):
                    col_def = col_def.strip()
                    if not col_def:
                        continue
                    col_match = re.match(r"([^\s(]+)\s*(?:\(([^)]+)\))?", col_def)
                    if not col_match:
                        continue
                    column_name = col_match.group(1).strip()
                    column_type = (col_match.group(2) or "unknown").strip()
                    if (
                        column_name
                        and column_name.lower() != "notes:"
                        and not column_name.startswith("<?")
                        and column_name not in columns
                    ):
                        columns.append(column_name)
                        types[column_name] = column_type or "unknown"

            xml_match = re.search(
                r"<data_description[^>]*table_name=[\x27\x22]"
                + re.escape(table_name)
                + r"[\x27\x22][^>]*>(.*?)</data_description>",
                plan,
                re.DOTALL | re.IGNORECASE,
            )
            if not xml_match:
                xml_match = re.search(
                    r"<table_info[^>]*name=[\x27\x22]"
                    + re.escape(table_name)
                    + r"[\x27\x22][^>]*>(.*?)($|</data_description>)",
                    plan,
                    re.DOTALL | re.IGNORECASE,
                )

            if xml_match:
                xml_content = xml_match.group(1)
                for column_match in re.finditer(
                    r"<column[^>]*name=[\x27\x22]([^\x27\x22]+)[\x27\x22]"
                    r"[^>]*data_type=[\x27\x22]([^\x27\x22]+)[\x27\x22]",
                    xml_content,
                    re.IGNORECASE,
                ):
                    column_name = column_match.group(1)
                    column_type = column_match.group(2)
                    if column_name not in columns:
                        columns.append(column_name)
                    types[column_name] = column_type or types.get(column_name, "unknown")

            if columns:
                merged_schemas[table_name] = {
                    "path": f"<live_database:{table_name}>",
                    "columns": columns,
                    "types": {column: types.get(column, "unknown") for column in columns},
                    "num_rows": None,
                }
                recovered_tables += 1

        if recovered_tables:
            logger.info(
                "Recovered %d live DB schema(s) from planner guidance for %s",
                recovered_tables,
                self.client_id,
            )
        return merged_schemas

    def _get_raw_db(self):
        """Get raw Motor database handle, unwrapping MongoDBManager if needed."""
        if self.db is None:
            return None
        return getattr(self.db, "db", self.db) if type(self.db).__name__ == "MongoDBManager" else self.db

    def _load_knowledge_for_coding(self) -> Dict[str, Any]:
        """Load table introductions, data descriptions, domain terminology,
        and client data profile.

        Called ONCE in ``execute_analysis``, stored in
        ``execution_context["knowledge_context"]``.  The raw XML is filtered
        and compressed per-iteration inside ``_decide_next_action``.
        """
        result: Dict[str, Any] = {
            "table_introductions_xml": "",
            "data_descriptions": {},
            "domain_terminology": "",
            "client_data_profile": "",
        }
        try:
            # 1. Table introductions (client → base fallback)
            intro_path = resolve_client_metadata_path(
                self.client_id,
                ("meta_information", "table_introductions.xml"),
                datasource_context=self.datasource_context,
            )
            if not intro_path or not intro_path.exists():
                intro_path = Path(BASE_PROMPTS_PATH) / "data_sources" / "meta_information" / "table_introductions.xml"
            if intro_path.exists():
                result["table_introductions_xml"] = load_xml_prompt_raw(intro_path)

            # 2. Data descriptions (datasource-specific dir → base fallback)
            desc_dir = resolve_client_metadata_dir(
                self.client_id,
                "data_descriptions",
                datasource_context=self.datasource_context,
            )
            if desc_dir and desc_dir.exists():
                result["data_descriptions"] = load_client_data_descriptions(
                    client_id=self.client_id,
                    base_descriptions_dir=Path(BASE_PROMPTS_PATH) / "data_sources" / "data_descriptions",
                    client_descriptions_dir=desc_dir,
                    datasource_context=self.datasource_context,
                )

            # 3. Domain terminology only (datasource → client → base fallback)
            term_path = None
            ds_key = get_datasource_namespace_suffix(self.datasource_context) if self.datasource_context else None
            if ds_key:
                ds_term = CLIENTS_PROMPTS_PATH / self.client_id / ds_key / "domain_knowledge" / "terminology.xml"
                if ds_term.exists():
                    term_path = ds_term
            if not term_path:
                client_term = CLIENTS_PROMPTS_PATH / self.client_id / "domain_knowledge" / "terminology.xml"
                if client_term.exists():
                    term_path = client_term
            if not term_path:
                term_path = Path(BASE_PROMPTS_PATH) / "domain_knowledge" / "terminology.xml"
            if term_path.exists():
                result["domain_terminology"] = load_xml_prompt_raw(term_path)

            # 4. Client data profile (geography, number format, industry)
            profile_path = resolve_client_metadata_path(
                self.client_id,
                ("meta_information", "client_data_profile.xml"),
                datasource_context=self.datasource_context,
            )
            if profile_path and profile_path.exists():
                result["client_data_profile"] = load_xml_prompt_raw(profile_path)

            logger.info(
                "Knowledge for coding | client=%s | intros=%d chars, descs=%d tables, terms=%d chars, profile=%d chars",
                self.client_id,
                len(result["table_introductions_xml"]),
                len(result["data_descriptions"]),
                len(result["domain_terminology"]),
                len(result.get("client_data_profile", "")),
            )
        except Exception as e:
            logger.warning("Failed to load knowledge for coding: %s", e)
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # TIERED PROMPT ARCHITECTURE — Token Optimization
    # ═══════════════════════════════════════════════════════════════════════════

    async def _build_static_system_context(
        self,
        user_query: str,
        plan_guidance: str,
        execution_context: Dict[str, Any],
    ) -> str:
        """Build the STATIC system prompt (Tier 1).

        Called ONCE per ``execute_analysis()`` run.  The result is identical
        across all iterations for the same query, which enables LLM-provider
        prompt caching (Gemini, Claude, OpenAI all auto-cache identical
        prefixes at 90 %+ discount).

        Contains:
            - XML agent prompt
            - Business knowledge (table intros, column descriptions, terminology, profile)
            - File schemas
            - Multi-table join context
            - Loaded dataset paths
            - Agent lessons (fetched once)
            - User preferences (fetched once)
            - All static instruction rules + response format
        """
        parts: List[str] = []

        # ── XML agent base prompt ──────────────────────────────────────────
        parts.append(self.base_prompt or "")
        parts.append(
            "\n\nYou are a recursive data science agent. "
            "You observe outputs, decide the next step, and iterate until the analysis is complete. "
            "You MUST respond with valid JSON only — no markdown fences, no explanations."
        )
        parts.append("")

        # ── File schemas (EXACT column names) ──────────────────────────────
        file_schemas = execution_context.get("file_schemas", {})
        if file_schemas:
            parts.append("FILE SCHEMAS (EXACT column names — use ONLY these, case-sensitive):")
            for fname, schema in file_schemas.items():
                rows_info = f", {schema['num_rows']:,} rows" if schema.get("num_rows") else ""
                parts.append(f"  {fname} ({schema.get('path', '')}{rows_info})")
                parts.append(f"    columns = {schema.get('columns', [])}")
                if schema.get("types"):
                    parts.append(f"    types   = {schema.get('types', {})}")
            parts.append("")
            parts.append(
                "⚠️ CRITICAL: The plan guidance may use WRONG column names (e.g. LAST_ISSUE_DATE). "
                "ALWAYS use the EXACT column names from FILE SCHEMAS above instead. "
                "When using pd.read_parquet(columns=[...]), use ONLY names from the schema."
            )
            parts.append("")

        # ── Business knowledge ─────────────────────────────────────────────
        knowledge_ctx = execution_context.get("knowledge_context", {})
        if knowledge_ctx and file_schemas:
            from util.knowledge_filter import (
                compress_table_introductions_for_coding,
                compress_data_descriptions_for_coding,
                compress_terminology_for_coding,
                _approx_token_count,
            )
            from config.system_config import MAX_CODING_KNOWLEDGE_TOKENS

            schema_tables = [Path(f).stem for f in file_schemas.keys()]
            knowledge_lines: list = []
            budget = MAX_CODING_KNOWLEDGE_TOKENS

            intros = compress_table_introductions_for_coding(
                knowledge_ctx.get("table_introductions_xml", ""), schema_tables,
            )
            if intros:
                cost = _approx_token_count(intros)
                if cost <= budget:
                    knowledge_lines.append("TABLE DESCRIPTIONS:")
                    knowledge_lines.append(intros)
                    budget -= cost

            descs = compress_data_descriptions_for_coding(
                knowledge_ctx.get("data_descriptions", {}), schema_tables,
            )
            if descs:
                cost = _approx_token_count(descs)
                if cost <= budget:
                    knowledge_lines.append("")
                    knowledge_lines.append("COLUMN DESCRIPTIONS (use to select correct columns):")
                    knowledge_lines.append(descs)
                    budget -= cost

            terms = compress_terminology_for_coding(
                knowledge_ctx.get("domain_terminology", ""),
            )
            if terms:
                cost = _approx_token_count(terms)
                if cost <= budget:
                    knowledge_lines.append("")
                    knowledge_lines.append("DOMAIN TERMINOLOGY:")
                    knowledge_lines.append(terms)

            if knowledge_lines:
                parts.append("BUSINESS KNOWLEDGE (understand what the data means):")
                parts.extend(knowledge_lines)
                parts.append("")

        # ── Client data profile ────────────────────────────────────────────
        client_profile = knowledge_ctx.get("client_data_profile", "")
        if client_profile:
            from config.system_config import MAX_DATA_PROFILE_TOKENS
            profile_cost = len(client_profile) // 4
            if profile_cost <= MAX_DATA_PROFILE_TOKENS:
                parts.append("CLIENT DATA PROFILE (formatting & locale guidance):")
                parts.append(client_profile)
                parts.append("")

        # ── Agent lessons (fetched ONCE, cached) ───────────────────────────
        try:
            raw_db = self._get_raw_db()
            if raw_db and not self._cached_lessons_text:
                from services.agent_lesson_service import AgentLessonService
                from config.system_config import MAX_LESSONS_TOKENS
                lesson_svc = AgentLessonService(raw_db)
                planned_tables = getattr(self, "_planned_tables", None)
                schema_tables = [Path(f).stem for f in file_schemas.keys()] if file_schemas else None
                filter_tables = planned_tables or schema_tables
                self._cached_lessons_text = await lesson_svc.format_lessons_for_prompt(
                    self.client_id, tables=filter_tables, max_tokens=MAX_LESSONS_TOKENS,
                )
        except Exception as le:
            logger.debug("Lesson injection skipped: %s", le)

        if self._cached_lessons_text:
            parts.append("LEARNED PATTERNS (from prior analyses — follow these strictly):")
            parts.append(self._cached_lessons_text)
            parts.append("")

        # ── User preferences (fetched ONCE, cached) ────────────────────────
        try:
            raw_db = self._get_raw_db()
            user_id = getattr(self, "_user_id", None)
            if raw_db and user_id and not self._cached_prefs_text:
                from services.user_preference_service import UserPreferenceService
                from services.preference_extractor import PreferenceExtractor
                from config.system_config import MAX_USER_PREFERENCES_TOKENS
                pref_svc = UserPreferenceService(raw_db)
                current_prefs = PreferenceExtractor.extract_as_dict(user_query) if user_query else {}
                self._cached_prefs_text = await pref_svc.format_for_prompt(
                    self.client_id, user_id,
                    current_query_prefs=current_prefs,
                    max_tokens=MAX_USER_PREFERENCES_TOKENS,
                )
        except Exception:
            pass

        if self._cached_prefs_text:
            parts.append("USER PREFERENCES (respect these for visualization and formatting):")
            parts.append(self._cached_prefs_text)
            parts.append("")

        # ── Multi-table join context ───────────────────────────────────────
        data_profile = execution_context.get("data_profile", {})
        if file_schemas and len(file_schemas) > 1:
            parts.append("MULTI-TABLE JOIN CONTEXT:")

            col_to_files: Dict[str, list] = {}
            file_row_counts: Dict[str, int] = {}
            for fname, schema in file_schemas.items():
                for col in schema.get("columns", []):
                    col_to_files.setdefault(col, []).append(fname)
                if schema.get("num_rows") is not None:
                    file_row_counts[fname] = schema["num_rows"]
                stem = Path(fname).stem
                stem_lower = stem.lower().replace("-", "_")
                for ds_name, prof in data_profile.items():
                    ds_lower = ds_name.lower().replace("-", "_")
                    if (ds_lower == stem_lower
                            or ds_lower.endswith(stem_lower)
                            or stem_lower.endswith(ds_lower)
                            or stem_lower in ds_lower
                            or ds_lower in stem_lower):
                        shape = prof.get("shape", [0])
                        file_row_counts[fname] = shape[0] if shape else 0

            shared_cols = {
                col: files for col, files in col_to_files.items()
                if len(files) > 1
            }
            if shared_cols:
                parts.append("  Shared columns (potential join keys):")
                for col, files in shared_cols.items():
                    parts.append(f"    {col}: appears in {', '.join(files)}")

            def _cols_near_match(a: str, b: str) -> bool:
                if a == b:
                    return False
                if a in b or b in a:
                    return True
                for suffix in ("_ID", "_NAME", "_CODE", "_KEY", "_NUM"):
                    if a.endswith(suffix) and b.endswith(suffix):
                        base_a = a[:-len(suffix)].rstrip("_")
                        base_b = b[:-len(suffix)].rstrip("_")
                        if base_a and base_b and (base_a in base_b or base_b in base_a):
                            return True
                return False

            all_cols_by_file = {
                fname: set(s.get("columns", []))
                for fname, s in file_schemas.items()
            }
            near_matches = []
            fnames_list = list(all_cols_by_file.keys())
            for i in range(len(fnames_list)):
                for j in range(i + 1, len(fnames_list)):
                    for col_a in all_cols_by_file[fnames_list[i]]:
                        for col_b in all_cols_by_file[fnames_list[j]]:
                            if _cols_near_match(col_a, col_b):
                                near_matches.append(
                                    (col_a, fnames_list[i], col_b, fnames_list[j])
                                )
            if near_matches:
                parts.append("  Near-match columns (VERIFY overlap before joining — names differ):")
                for col_a, f_a, col_b, f_b in near_matches[:10]:
                    parts.append(f"    {col_a} ({f_a}) ↔ {col_b} ({f_b})")

            for fname, rows in file_row_counts.items():
                if rows < 10000:
                    parts.append(
                        f"  ⚠️ CRITICAL: {fname} is a small table ({rows} rows) "
                        f"— IGNORE the plan's column selection for this file. "
                        f"Load ALL columns: pd.read_parquet(path) with NO columns= parameter."
                    )

            if not any(rows < 10000 for rows in file_row_counts.values()):
                for fname, schema in file_schemas.items():
                    n_cols = len(schema.get("columns", []))
                    if n_cols <= 6 and fname not in file_row_counts:
                        parts.append(
                            f"  ℹ️ {fname} has only {n_cols} columns — likely a small "
                            f"lookup table. Load ALL columns to avoid needing to reload."
                        )

            parts.append(
                "\n  ⚠️ MULTI-TABLE JOIN RULE: Before joining two tables, you MUST:\n"
                "    1. For small lookup/dimension tables: load ALL columns "
                "(do NOT use columns= parameter)\n"
                "    2. BEFORE joining, verify join key overlap:\n"
                "       overlap = set(df_a['col_a'].unique()) & set(df_b['col_b'].unique())\n"
                "       print(f'Overlap: {len(overlap)} common values')\n"
                "    3. If overlap is 0, try OTHER candidate columns — check near-matches above\n"
                "    4. Column names may differ (e.g., ORGANIZATION_ID ↔ INV_ORG_ID) "
                "— check VALUES, not just names\n"
                "    5. Use the column pair with the HIGHEST overlap for the join\n"
                "    6. FILTER-BY-VALUE (same rules apply): When using a value from "
                "table A to filter table B (e.g., .isin(), == comparison):\n"
                "       - Verify the looked-up value EXISTS in the target column "
                "BEFORE filtering\n"
                "       - If 0 rows result, the value maps to a DIFFERENT column "
                "in table B\n"
                "       - Print unique values in candidate columns to find the "
                "correct mapping"
            )
            parts.append("")

        # ── Loaded datasets (file paths) ───────────────────────────────────
        loaded_datasets = execution_context.get("loaded_datasets", [])
        if loaded_datasets:
            parts.append(
                "LOADED DATASETS "
                "(⚠️ CRITICAL: when calling pd.read_parquet/read_csv, "
                "use the EXACT absolute path below — NEVER a bare filename like 'data.parquet'):"
            )
            for ds in loaded_datasets:
                parts.append(
                    f"  - path='{ds.get('path', '?')}' variable={ds.get('variable', '?')} "
                    f"format={ds.get('format', '?')}"
                )
            parts.append("")

        # ── Static instructions + response format ──────────────────────────
        parts.extend([
            "RULES:",
            "- ITERATION 1: MANDATORY PLANNING-ONLY: In your very first execution step (Iteration 1), your `code` MUST ONLY contain the initialization of `TASKS`, `COMPLETED_TASKS`, and `_VAR_INTENT_`. You MUST NOT include any SQL, data loading, or analysis logic. Break down the user query into a detailed, tech-heavy `TASKS` list based on the PROVIDED ANALYST GUIDANCE. Every task MUST include the table names and specific columns/filters mentioned in the guidance (e.g., `[Step 1] Fetch TotalValue from NE_DMSSaleReportTable where OrderDate >= '2025-01-01'`). Start task execution in Iteration 2.",
            "- INTENT REGISTRY: Initialize `_VAR_INTENT_ = {}`. For every meaningful variable/DataFrame you create (starting from Iteration 2), add a 1-sentence entry describing its content (e.g., `_VAR_INTENT_['df_sales'] = 'Monthly 2025 sales aggregated'`).",
            "- STATE PROGRESSION: In every step after Iteration 1, append the task you just finished to `COMPLETED_TASKS.append(...)`.",
            "- T-SQL BEST PRACTICES (MSSQL):",
            "    - Do NOT use `RowCount` as a SQL alias (alias to `total_rows` instead).",
            "    - Use `TOP N` instead of `LIMIT N`.",
            "    - Always use `DATEFROMPARTS` or `CAST(col as date)` for grouping to maintain sargability.",
            "- Each iteration builds on previous ones. Variables from prior iterations are alive in memory.",
            "- Do NOT re-import libraries or re-load data that was already loaded.",
            "- NEVER re-compute DataFrames you already made; pass them continuously from task to task.",
            "- Do NOT set FINAL_RESULT until all your `TASKS` are in `COMPLETED_TASKS`. Premature completion ruins the analysis.",
            "- If this is the LAST step of the analysis, store your primary result in FINAL_RESULT.",
            '  Example: FINAL_RESULT = result_df  or  FINAL_RESULT = {"key": value}',
            "- Set FINAL_RESULT BEFORE declaring action='done'.",
            "",
            "WORKFLOW GUIDANCE (adapt based on query complexity):",
            "- For SIMPLE queries: plan → load/compute → FINAL_RESULT in 3-4 iterations.",
            "- For COMPLEX queries: plan → load → merge → compute → FINAL_RESULT in 5-6 iterations.",
            "- You MAY combine related operations in one cell to close multiple TASKS at once.",
            "- Before any groupby/filter, quickly check the relevant column: print(df['col'].nunique())",
            "- If a merge/join produces 0 rows, investigate immediately — don't proceed with empty data.",
            "- MERGE VERIFICATION: After any pd.merge/join, immediately print:",
            "  1. Result shape vs input shapes (row explosion = wrong keys)",
            "  2. Check for '_x'/'_y' column suffixes (= overlapping non-key columns, likely wrong join keys)",
            "  3. Sample 2-3 rows to sanity-check the joined data",
            "",
            "CODE QUALITY RULES:",
            "- Keep cells to 30-40 lines MAX.",
            "- NEVER re-import libraries or re-load data that's already in AVAILABLE VARIABLES.",
            "- NEVER reference variables that don't exist — check AVAILABLE VARIABLES above.",
            "- Use ONLY variables present in AVAILABLE VARIABLES. Do NOT try to parse or hardcode DataFrames from plain text outputs.",
            "- Every cell MUST end with print() showing what was produced (unless setting FINAL_RESULT).",
            "",
            "PERFORMANCE RULES:",
            "- For LARGE tables (>100K rows): select needed columns: pd.read_parquet(path, columns=['col1','col2'])",
            "- For SMALL lookup/dimension tables (<10K rows): load ALL columns to avoid repeated KeyErrors",
            "- When in doubt about which columns you'll need, load MORE columns — reloading wastes iterations",
            "- Do NOT add .sample() — always process the FULL dataset for accurate results",
            "- Only sample if the user EXPLICITLY asks for it in their query",
            "- Use vectorized operations, not loops.",
            "",
            "FILTER VERIFICATION RULES:",
            "- After EVERY filter (.isin(), .query(), boolean indexing, pd.merge), "
            "IMMEDIATELY check: print(f'Filtered: {result.shape[0]} rows')",
            "- If 0 rows: DO NOT proceed. Instead:",
            "  1. Print unique values in BOTH source and target filter columns",
            "  2. Check if you used the wrong column (e.g., ORG_ID vs INV_ORG_ID)",
            "  3. Try alternative columns ending in _ID, _CODE, _KEY, _NUM",
            "- When using a value from table A to filter table B:",
            "  1. Print the lookup value: print(f'Lookup: {value}')",
            "  2. Verify the value EXISTS in the target column before filtering",
            "  3. Column names often DIFFER between tables for the same entity",
            "  4. Values may also differ: ORG_ID=81 does NOT mean ORGANIZATION_ID=81",
            "- NEVER silently accept 0 rows and proceed to the next step",
            "",
            "RESPONSE FORMAT (return ONLY this JSON, no markdown fences, no extra text):",
            '{',
            '  "action": "code" or "done",',
            '  "thinking": "1-2 sentences: What data do I have, what do I still need, and what will this cell do?",',
            '  "reasoning": "≤6 words, creative step title (e.g. '
            '\'Pulling in the RFQ data\', '
            '\'Linking RFQs to their organization\', '
            '\'Spotting the top revenue drivers\', '
            '\'Assembling the final picture\'). '
            'No column/file/table names. No generic labels like Loading datasets or Merging datasets.",',
            '  "code": "python code to execute (empty string if action is done)"',
            '}',
        ])

        return "\n".join(parts)

    def _build_journal_entry(
        self,
        iteration: int,
        reasoning: str,
        new_vars: Dict[str, Any],
        prev_vars: Dict[str, Any],
    ) -> str:
        """Build a compact 1-line execution journal entry for a completed iteration.

        Example output:
            ``Step 2: Linking RFQs to organizations → merged_df (DataFrame, 4521×8)``
        """
        created = []
        for name, info in new_vars.items():
            if name in prev_vars:
                continue
            if isinstance(info, dict) and info.get("type") == "DataFrame":
                shape = info.get("shape", [])
                shape_str = f"{shape[0]}×{shape[1]}" if len(shape) >= 2 else str(shape)
                created.append(f"{name} (DataFrame, {shape_str})")
            elif isinstance(info, dict):
                created.append(f"{name} ({info.get('type', '?')})")
            else:
                created.append(f"{name} ({info})")
        vars_part = ", ".join(created[:5]) if created else "no new variables"
        return f"Step {iteration}: {reasoning} → {vars_part}"

    def _compute_profile_delta(
        self,
        current_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute the data profile DELTA since last iteration.

        Returns a dict of only NEW or CHANGED DataFrames:
        - New DataFrames get full profile (string_values capped at ``string_values_top_n``)
        - Changed DataFrames (shape differs) get shape + columns only
        - Unchanged DataFrames are omitted entirely

        On the first call (``_profiled_datasets`` is empty), ALL DataFrames
        are treated as new, ensuring the LLM always has full context on the
        first iteration.
        """
        delta: Dict[str, Any] = {}
        current_keys = set(current_profile.keys())
        top_n = self.string_values_top_n

        # New DataFrames: full profile
        for key in current_keys - self._profiled_datasets:
            prof = {}
            src = current_profile[key]
            for field in ("shape", "columns", "dtypes", "null_counts", "sample_row"):
                if src.get(field):
                    prof[field] = src[field]
            # Cap string_values to top_n
            if src.get("string_values"):
                capped = {}
                for col_name, sv in src["string_values"].items():
                    capped[col_name] = {
                        "unique_count": sv.get("unique_count", "?"),
                        "top_values": sv.get("top_values", [])[:top_n],
                    }
                prof["string_values"] = capped
            delta[key] = {"status": "new", "profile": prof}

        # Changed DataFrames: shape + columns only
        for key in current_keys & self._profiled_datasets:
            prev_shape = self._previous_profile_shapes.get(key)
            curr_shape = current_profile[key].get("shape")
            if prev_shape != curr_shape:
                delta[key] = {
                    "status": "changed",
                    "shape_before": prev_shape,
                    "shape_after": curr_shape,
                    "columns": current_profile[key].get("columns"),
                }

        # Update snapshots
        self._profiled_datasets = current_keys.copy()
        self._previous_profile_shapes = {
            k: v.get("shape") for k, v in current_profile.items()
        }

        return delta

    def _register_artifact(
        self,
        iteration: int,
        reasoning: str,
        new_vars: Dict[str, Any],
        prev_vars: Dict[str, Any],
    ) -> None:
        """Track intermediate artifacts for the narrator/business agent.

        Diffs ``new_vars`` against ``prev_vars`` to identify what was
        created vs modified in this iteration.
        """
        new_variables = {}
        modified_variables = {}
        for name, info in new_vars.items():
            if name not in prev_vars:
                new_variables[name] = info if isinstance(info, dict) else {"type": str(info)}
            elif isinstance(info, dict) and isinstance(prev_vars.get(name), dict):
                prev_shape = prev_vars[name].get("shape")
                curr_shape = info.get("shape")
                if prev_shape != curr_shape:
                    modified_variables[name] = {
                        "type": info.get("type", "?"),
                        "shape_before": prev_shape,
                        "shape_after": curr_shape,
                    }

        self._artifact_registry.append({
            "iteration": iteration,
            "reasoning": reasoning,
            "new_variables": new_variables,
            "modified_variables": modified_variables,
        })

    def _build_dynamic_user_message(
        self,
        user_query: str,
        plan_guidance: str,
        execution_context: Dict[str, Any],
        iteration: int,
    ) -> str:
        """Build the DYNAMIC user message (Tier 3).

        Contains ONLY what changes per iteration: query, plan, execution
        journal, last iteration detail, available variables, profile delta,
        warnings, and convergence info.  Typically ~1-2K tokens.
        """
        parts: List[str] = []

        # 0. Adhoc mode context
        if execution_context.get("adhoc_mode"):
            parts.append(
                "IMPORTANT: You are analyzing a USER-UPLOADED ad-hoc file. "
                "Focus on exploratory data analysis. "
                "Do NOT reference any other tables or data sources — only use the uploaded file(s). "
                "The file schema and sample data are available from the kernel."
            )
            parts.append("")

        # 1. User query + plan
        parts.append(f"USER QUERY: {user_query}")
        parts.append("")
        parts.append("PLAN GUIDANCE (use as direction, not a rigid checklist):")
        parts.append(plan_guidance)
        parts.append("")

        # 2. Execution journal (compact)
        journal = execution_context.get("execution_journal", [])
        if journal:
            parts.append("EXECUTION JOURNAL (DO NOT repeat any of this — all variables are alive in kernel):")
            for entry in journal:
                parts.append(f"  {entry}")
            parts.append("")

            # Last iteration detail (only the most recent N entries)
            completed = execution_context.get("completed_iterations", [])
            detail_count = self.max_journal_detail_entries
            if completed:
                recent = completed[-detail_count:]
                parts.append("LAST ITERATION DETAIL:")
                for item in recent:
                    iter_id = item.get("iteration", "?")
                    code_preview = item.get("code", "")
                    if code_preview and len(code_preview) > self.code_preview_max_chars:
                        code_preview = code_preview[:self.code_preview_max_chars] + "\n# ...[truncated]"
                    output_preview = item.get("output", "")
                    if output_preview and len(output_preview) > self.output_preview_max_chars:
                        output_preview = output_preview[:self.output_preview_max_chars] + "\n...[truncated]"
                    if code_preview:
                        parts.append(f"  Code (Step {iter_id}):\n{code_preview}")
                    if output_preview:
                        parts.append(f"  Output (Step {iter_id}):\n{output_preview}")
                parts.append("")
        else:
            parts.append("No iterations completed yet — this is the first iteration.")
            parts.append("")

        # 3. Available variables
        available_vars = execution_context.get("available_variables", {})
        if available_vars:
            vars_lines = []
            dataframes = []
            for name, info in available_vars.items():
                if isinstance(info, dict):
                    type_str = info.get("type", "Unknown")
                    if type_str == "DataFrame":
                        dataframes.append(name)
                    details = ""
                    if "columns" in info:
                        cols = info["columns"]
                        if len(cols) > 10:
                            cols_str = str(cols[:10])[:-1] + ", ...]"
                        else:
                            cols_str = str(cols)
                        details = f" columns={cols_str}"
                    if "shape" in info:
                        details += f" shape={info['shape']}"
                    
                    # Show intent if present
                    intent_str = f" intent=\"{info['intent']}\"" if "intent" in info else ""
                    # Show full value for orchestration state variables
                    value_str = ""
                    if name in ["TASKS", "COMPLETED_TASKS", "_VAR_INTENT_"] and "value" in info:
                        value_str = f" value={info['value']}"
                    
                    vars_lines.append(f"- {name} ({type_str}){details}{intent_str}{value_str}")
                else:
                    vars_lines.append(f"- {name} ({info})")

            parts.append(f"AVAILABLE VARIABLES:\n" + "\n".join(vars_lines))

            if dataframes and "df" not in dataframes:
                if len(dataframes) == 1:
                    parts.append(
                        f"\n⚠️ CRITICAL: The dataframe is named '{dataframes[0]}'. "
                        f"DO NOT use 'df'. Use '{dataframes[0]}' instead."
                    )
                else:
                    parts.append(
                        f"\n⚠️ CRITICAL: Available dataframes: {', '.join(dataframes)}. "
                        f"DO NOT use 'df' unless it is defined."
                    )
            parts.append(
                "⚠️ CRITICAL: These are the ONLY variables in memory. "
                "Data is NOT preloaded as table-name globals "
                "(e.g., IFFCO_INV_AI_CONS does NOT exist as a variable). "
                "Use ONLY the exact variable names listed above."
            )

            if "FINAL_RESULT" in available_vars:
                parts.append(
                    "\n🛑 STOP — FINAL_RESULT IS ALREADY SET IN THE KERNEL. "
                    "Your analysis is COMPLETE. You MUST return action: \"done\" immediately. "
                    "Do NOT generate more code. Do NOT re-compute or re-set FINAL_RESULT. "
                    "The answer has already been produced."
                )
            parts.append("")

        # 4. Dataset profile DELTA (only new/changed DataFrames)
        data_profile = execution_context.get("data_profile", {})
        if data_profile:
            delta = self._compute_profile_delta(data_profile)
            if delta:
                parts.append("DATASET PROFILE CHANGES:")
                for ds_name, info in delta.items():
                    if info["status"] == "new":
                        prof = info["profile"]
                        parts.append(f"  NEW: {ds_name}")
                        if prof.get("shape"):
                            parts.append(f"    shape   = {prof['shape']}")
                        if prof.get("columns"):
                            parts.append(f"    columns = {prof['columns']}")
                        if prof.get("dtypes"):
                            parts.append(f"    dtypes  = {prof['dtypes']}")
                        if prof.get("null_counts"):
                            parts.append(f"    nulls   = {prof['null_counts']}")
                        if prof.get("sample_row"):
                            parts.append(f"    sample  = {str(prof['sample_row'][0])[:200]}")
                        if prof.get("string_values"):
                            parts.append(f"    string_values:")
                            for col_name, sv in prof["string_values"].items():
                                unique_ct = sv.get("unique_count", "?")
                                top_vals = sv.get("top_values", [])
                                parts.append(f"      {col_name} ({unique_ct} unique): {top_vals}")
                    elif info["status"] == "changed":
                        parts.append(
                            f"  CHANGED: {ds_name} shape {info.get('shape_before')} → {info.get('shape_after')}"
                        )

                has_string_values = any(
                    info["status"] == "new" and info.get("profile", {}).get("string_values")
                    for info in delta.values()
                )
                if has_string_values:
                    parts.append(
                        "⚠️ STRING FILTER RULE: When filtering on a string column, "
                        "first EXPLORE with str.contains(), then REVIEW unique values, then FILTER."
                    )
                parts.append(
                    "USE ONLY THESE EXACT COLUMN NAMES — case-sensitive, no variation."
                )
                parts.append("")
            elif not journal:
                # First iteration with no delta means profile is in static context
                # Include full profile for first iteration
                parts.append("DATASET PROFILE (loaded DataFrames):")
                for ds_name, prof in data_profile.items():
                    parts.append(f"  {ds_name}:")
                    if prof.get("shape"):
                        parts.append(f"    shape   = {prof['shape']}")
                    if prof.get("columns"):
                        parts.append(f"    columns = {prof['columns']}")
                    if prof.get("dtypes"):
                        parts.append(f"    dtypes  = {prof['dtypes']}")
                    if prof.get("null_counts"):
                        parts.append(f"    nulls   = {prof['null_counts']}")
                    if prof.get("sample_row"):
                        parts.append(f"    sample  = {str(prof['sample_row'][0])[:200]}")
                    if prof.get("string_values"):
                        parts.append(f"    string_values:")
                        for col_name, sv in prof["string_values"].items():
                            unique_ct = sv.get("unique_count", "?")
                            top_vals = sv.get("top_values", [])[:self.string_values_top_n]
                            suffix = f" ... ({unique_ct} unique total)" if unique_ct and int(str(unique_ct)) > self.string_values_top_n else ""
                            parts.append(f"      {col_name} ({unique_ct} unique): {top_vals}{suffix}")
                parts.append(
                    "USE ONLY THESE EXACT COLUMN NAMES — case-sensitive, no variation."
                )
                has_sv = any(prof.get("string_values") for prof in data_profile.values())
                if has_sv:
                    parts.append(
                        "⚠️ STRING FILTER RULE: When filtering on a string column, "
                        "first EXPLORE with str.contains(), then REVIEW unique values, then FILTER."
                    )
                parts.append("")

        # 5. Failed iterations
        failed_iters = execution_context.get("failed_iterations", [])
        if failed_iters:
            parts.append("⚠️ FAILED ITERATIONS (these approaches ALREADY FAILED — learn from them):")
            for fi in failed_iters[-3:]:
                parts.append(f"  Iteration {fi['iteration']} FAILED:")
                parts.append(f"    Error: {fi['error'][:300]}")
                if fi.get("code_snippet"):
                    parts.append(f"    Attempted code (snippet): {fi['code_snippet']}")
            parts.append(
                "  → Do NOT repeat these failed approaches. Fix the SPECIFIC error "
                "and continue from AVAILABLE VARIABLES."
            )
            parts.append("")

        # 6. Warnings
        warnings = execution_context.get("warnings", [])
        if warnings:
            zero_row_warnings = [w for w in warnings if "ZERO_ROW" in w or "became empty" in w]
            other_warnings = [w for w in warnings if w not in zero_row_warnings]

            if zero_row_warnings:
                parts.append("⚠️ CRITICAL — ZERO-ROW RESULTS DETECTED:")
                for w in zero_row_warnings:
                    parts.append(f"  ❌ {w}")
                parts.append(
                    "  ACTION REQUIRED: Re-examine filter columns and values. "
                    "Try alternative columns. Print unique values to verify."
                )
                parts.append("")

            if other_warnings:
                parts.append("WARNINGS from prior iterations:")
                for w in other_warnings:
                    parts.append(f"  - {w}")
                parts.append("")

        # 7. Iteration counter + convergence
        remaining = self.max_iterations - iteration
        parts.append(f"ITERATION: {iteration} / {self.max_iterations}")
        if remaining <= 2:
            parts.append(
                f"⚠️ URGENT: Only {remaining} iteration(s) remaining. "
                "You MUST assemble FINAL_RESULT in this iteration. "
                "Use whatever data you have — a partial answer is better than no answer."
            )
        elif remaining <= self.max_iterations // 2:
            parts.append(
                f"Note: {remaining} iterations remaining. "
                "If you have enough data, proceed to computation and FINAL_RESULT."
            )

        # 8. Task directive
        parts.extend([
            "",
            "YOUR TASK:",
            "Based on the user query, plan guidance, execution journal and outputs,",
            "decide what to do NEXT.",
        ])

        return "\n".join(parts)

    async def _fetch_db_credentials(self) -> None:
        """Fetch client database credentials and determine operating mode.

        Sets ``self._is_live_db = True`` when the client has a real database,
        or ``False`` when the client uses file uploads (parquet).
        """
        try:
            from services.db_credentials_service import DBCredentialsService

            # Resolve raw mongo DB handle
            actual_db = getattr(self.db, "db", self.db) if type(self.db).__name__ == "MongoDBManager" else self.db

            service = DBCredentialsService(actual_db)
            credentials = await service.get_credentials(
                self.client_id,
                db_type=None,
                datasource_context=self.datasource_context,
                decrypt_password=True,
            )

            if credentials:
                db_type = credentials.get("db_type", "")
                if db_type in _LIVE_DB_TYPES:
                    db_host = credentials.get("db_host") or ""
                    db_password = credentials.get("db_password") or ""
                    db_user = credentials.get("db_username") or ""

                    if not db_host or not db_password or not db_user:
                        logger.error(
                            f"Incomplete DB credentials for client {self.client_id} "
                            f"(host={bool(db_host)}, user={bool(db_user)}, password={bool(db_password)}). "
                            f"Please re-save credentials via the DB configuration page."
                        )
                        self._is_live_db = False
                        return

                    self._is_live_db = True
                    self.db_credentials_env = {
                        "CS_DB_TYPE": db_type or "",
                        "CS_DB_HOST": db_host,
                        "CS_DB_PORT": str(credentials.get("db_port") or ""),
                        "CS_DB_NAME": credentials.get("db_name") or "",
                        "CS_DB_USER": db_user,
                        "CS_DB_PASSWORD": db_password,
                        "CS_SSH_TUNNEL": json.dumps(credentials.get("ssh_tunnel") or {"enabled": False}),
                    }
                    logger.info(f"Live DB mode for client {self.client_id} (type={db_type})")
                else:
                    self._is_live_db = False
                    logger.info(f"File-upload mode for client {self.client_id} (db_type={db_type})")
            else:
                self._is_live_db = False
                logger.warning(f"No credentials found for client {self.client_id} — defaulting to file-upload mode")
        except Exception as e:
            self._is_live_db = False
            logger.error(f"Error fetching DB credentials: {e} — defaulting to file-upload mode")

    def _load_system_prompt(self) -> str:
        """Load the data science agent system prompt from XML.

        Uses the multi-tenant load_client_prompt() pipeline so that
        client-specific overrides, MongoDB section merges, and
        custom_prompts.xml are respected — consistent with planner,
        python, and business agents.
        """
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
            prompt_path = Path(BASE_PROMPTS_PATH) / "agents" / f"{self.agent_name}.xml"
            if prompt_path.exists():
                with open(prompt_path, 'r') as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"Could not load XML prompt: {e}")

        # Last-resort fallback (base XML file should always exist)
        return (
            "You are an expert Data Science Agent. Generate clean, executable "
            "Python code for data analysis, predictions, and iterative refinement "
            "in a Jupyter environment."
        )

    def _update_usage(self, usage: Optional[Dict[str, Any]]) -> None:
        """Update token usage statistics."""
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

        prompt_tokens = _safe_int(
            usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("prompt_token_count")
        )
        completion_tokens = _safe_int(
            usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("candidates_token_count")
        )

        self.usage_stats["prompt_tokens"] = self.usage_stats.get("prompt_tokens", 0) + prompt_tokens
        self.usage_stats["completion_tokens"] = self.usage_stats.get("completion_tokens", 0) + completion_tokens
        # Canonical total is always derived from prompt+completion.
        self.usage_stats["total_tokens"] = self.usage_stats.get("total_tokens", 0) + (prompt_tokens + completion_tokens)

        # Preserve additional token counters when providers expose them.
        extra_token_keys = [
            "reasoning_tokens",
            "cached_input_tokens",
            "cache_creation_input_tokens",
            "audio_input_tokens",
            "audio_output_tokens",
            "image_input_tokens",
            "accepted_prediction_tokens",
            "rejected_prediction_tokens",
            "text_input_tokens",
            "text_output_tokens",
            "total_tokens_provider",
        ]
        for key in extra_token_keys:
            value = _safe_int(usage.get(key))
            if value:
                self.usage_stats[key] = self.usage_stats.get(key, 0) + value

        if usage.get("model"):
            self.usage_stats["models"].add(usage["model"])

    def _parse_plan_steps(self, plan: str) -> List[Dict[str, Any]]:
        """
        Parse a markdown plan into discrete executable steps.
        
        Expected format:
        1. Step description
           - Sub-bullet with details
        2. Next step description
        
        Args:
            plan: Markdown formatted plan string
        
        Returns:
            List of step dictionaries with:
            - step_num: int
            - description: str
            - details: List[str] (sub-bullets)
        """
        import re
        
        steps = []
        # Handle both list-of-steps (new planner output) and raw string (legacy)
        if isinstance(plan, list):
            plan = '\n'.join(str(s) for s in plan)
        lines = plan.strip().split('\n')
        current_step = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match numbered steps (e.g., "1. ", "2. ", etc.)
            step_match = re.match(r'^(\d+)\.\s+(.+)$', line)
            if step_match:
                # Save previous step if it exists
                if current_step:
                    steps.append(current_step)
                
                # Start new step
                step_num = int(step_match.group(1))
                description = step_match.group(2)
                current_step = {
                    "step_num": step_num,
                    "description": description,
                    "details": []
                }
            # Match sub-bullets (e.g., "- ", "* ")
            elif line.startswith(('-', '*', '•')) and current_step:
                detail = line.lstrip('-*• ').strip()
                if detail:
                    current_step["details"].append(detail)
        
        # Don't forget the last step
        if current_step:
            steps.append(current_step)
        
        logger.info(f"Parsed {len(steps)} steps from plan")
        return steps


    async def execute_analysis(
        self,
        user_query: str,
        plan: str,  # NEW: Accept the plan from planner agent
        dataset_path: Optional[str] = None,
        dataset_dict: Optional[Dict] = None,
        context: Optional[Dict] = None
    ) -> AsyncGenerator[Dict, None]:
        """
        Execute step-by-step data science analysis based on a structured plan.
        
        Args:
            user_query: The user's data science question/task
            plan: Markdown formatted plan with numbered steps
            dataset_path: Path to CSV/parquet file
            dataset_dict: Pre-loaded dataset as dict or pandas DataFrame
            context: Additional context (previous results, domain info, etc.)
        
        Yields:
            Dict with keys:
            - type: 'step_start', 'code_generated', 'step_execution', 'step_complete', 'step_retry', 'final_result', 'error'
            - content: Relevant data for each type
            - step_num: Current step number
        """
        try:
            # --- Create live notebook ---
            self.notebook_builder = NotebookBuilder(
                output_dir=self.notebook_output_dir,
                name_prefix="analysis",
            )
            self.notebook_builder.add_markdown_cell(
                f"# Data Science Analysis\n\n"
                f"**Query:** {user_query}\n\n"
                f"**Generated:** {utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
                f"---\n\n"
                f"## Guidance\n\n{plan}"
            )
            self.notebook_builder.save()
            
            yield await self._stream_event("status", {
                "message": f"Analysis started — working iteratively (max {self.max_iterations} iterations)",
                "max_iterations": self.max_iterations,
                "notebook_path": str(self.notebook_builder.filepath)
            })

            # ── Fetch credentials so _is_live_db is evaluated correctly ──
            yield await self._stream_event("status", {"message": "Determine execution mode..."})
            await self._fetch_db_credentials()
            
            
            # Setup kernel and data
            yield await self._stream_event("status", {"message": "Initializing Jupyter kernel..."})
            await self._initialize_kernel()
            
            yield await self._stream_event("status", {"message": "Loading dataset..."})
            loaded_datasets = await self._load_dataset_to_kernel(dataset_path, dataset_dict)
            
            # Verify what variables the kernel actually has after loading
            kernel_vars = await self._get_kernel_variables()
            actual_vars = list(kernel_vars.keys()) if kernel_vars else []
            
            # --- FORCE LOADING FIX ---
            # If no variables found, try to force-load the dataset immediately
            if not actual_vars and not loaded_datasets:
                try:
                    # Fallback discovery
                    client_data_dir = self._get_client_dataset_dir()
                    if client_data_dir.exists():
                        parquet_files = list(client_data_dir.glob("*.parquet"))
                        if parquet_files:
                            main_file = parquet_files[0]
                            container_path = f"/app/{main_file.name}"
                            
                            # Copy file if needed
                            if self.kernel_manager:
                                self.kernel_manager.copy_file_to_container(str(main_file), container_path)
                            
                            logger.info(f"Force-loading dataset: {container_path}")
                            force_code = f"""
import pandas as pd
try:
    df = pd.read_parquet(r'{container_path}')
    print(f"Force-loaded dataset from {container_path}")
    print(f"Shape: {{df.shape}}")
except Exception as e:
    print(f"Force-load failed: {{e}}")
"""
                            await self._execute_code(force_code)
                            
                            # Re-check variables
                            kernel_vars = await self._get_kernel_variables()
                            actual_vars = list(kernel_vars.keys())
                            
                            loaded_datasets = [{
                                'path': container_path,
                                'variable': 'df',
                                'format': 'parquet'
                            }]
                except Exception as e:
                    logger.warning(f"Force loading failed: {e}")

            if not actual_vars:
                # Loading may have failed silently — warn but continue
                logger.warning("No variables found in kernel after dataset loading")
            
            execution_context = {
                "user_query": user_query,
                "plan_guidance": plan,  # Guidance, not rigid steps
                "dataset_path": dataset_path,
                "available_variables": kernel_vars if kernel_vars else {},
                "completed_iterations": [],  # Renamed from completed_steps
                "execution_journal": [],     # Compact 1-line-per-iteration journal (tiered prompts)
                "context": context or {},
                "loaded_datasets": loaded_datasets,
                "warnings": [],
            }
            if self._session_kernel_reused:
                # Strong, deterministic hint: this is a dependent follow-up.
                # The kernel is already warm; do not repeat load/compute steps unless missing.
                execution_context["warnings"].append(
                    "SESSION_KERNEL_REUSE: This is a follow-up. Prior variables/results may already exist in the kernel. "
                    "Before loading data or recomputing, CHECK AVAILABLE VARIABLES and reuse existing DataFrames/FINAL_RESULT."
                )
                if kernel_vars:
                    if "FINAL_RESULT" in kernel_vars:
                        execution_context["warnings"].append(
                            "SESSION_KERNEL_REUSE_HINT: FINAL_RESULT already exists in the kernel from a previous question. "
                            "Prefer reusing it (e.g., render a bar chart from the existing table) instead of recomputing."
                        )
                    # If any df_* already exists, call it out explicitly (cheap and effective).
                    existing_dfs = [k for k, v in kernel_vars.items() if isinstance(v, dict) and v.get("type") == "DataFrame" and k.startswith("df")]
                    if existing_dfs:
                        execution_context["warnings"].append(
                            "SESSION_KERNEL_REUSE_HINT: Existing DataFrames detected: "
                            + ", ".join(sorted(existing_dfs)[:10])
                            + ("" if len(existing_dfs) <= 10 else " ...")
                        )
            planned_tables = ((context or {}).get("planned_tables") or [])
            self._planned_tables = [str(t).strip() for t in planned_tables if str(t).strip()]
            if self._planned_tables:
                logger.info(
                    "[PromptScope] data_science planned_tables=%s",
                    self._planned_tables,
                )

            # Inject llm_query() RLM primitive into kernel
            yield await self._stream_event("status", {"message": "Injecting llm_query() helper..."})
            await self._inject_llm_query_helper()

            # ── Pre-load parquet/CSV schemas (BEFORE any code runs) ───────
            # This reads file metadata only (~0ms, zero memory) so the LLM
            # always has exact column names and never hallucinates them.
            yield await self._stream_event("status", {"message": "Reading schemas..."})
            file_schemas = await self._probe_parquet_schemas(
                loaded_datasets, dataset_path
            )
            execution_context["file_schemas"] = self._merge_live_db_schemas_from_plan(
                file_schemas,
                plan,
            )

            # Profile all DataFrames currently in kernel
            yield await self._stream_event("status", {"message": "Profiling dataset..."})
            execution_context["data_profile"] = await self._probe_dataset_profile()

            # ── Load business knowledge (once — filtered per-iteration) ────
            # In adhoc mode, skip backend table descriptions (irrelevant for
            # user-uploaded files). File schemas + data profile are sufficient.
            is_adhoc = (context or {}).get("adhoc_mode", False)
            if is_adhoc:
                execution_context["knowledge_context"] = {}
                execution_context["adhoc_mode"] = True
                logger.info("Adhoc mode: skipping backend knowledge loading")
            else:
                try:
                    execution_context["knowledge_context"] = self._load_knowledge_for_coding()
                except Exception as e:
                    logger.warning("Knowledge loading failed (non-fatal): %s", e)
                    execution_context["knowledge_context"] = {}

            # ═══════════════════════════════════════════════════════════════════
            # RECURSIVE EXECUTION LOOP — LLM decides what to do next
            # ═══════════════════════════════════════════════════════════════════
            iteration = 0
            status = "continue"  # "continue" | "done" | "error"
            consecutive_failures = 0
            MAX_CONSECUTIVE_FAILURES = self.doom_loop_threshold  # abort if N iterations fail in a row
            self._recent_failed_codes = []  # reset doom loop buffer for this analysis run

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

                # ----- Ask LLM what to do next -----
                yield await self._stream_event("status", {
                    "message": f"Iteration {iteration}/{self.max_iterations} — deciding next action..."
                })

                try:
                    if iteration == 1:
                        # Dedicated high-fidelity planning step (User Feedback: "specific call to create a detailed plan")
                        decision = await self._generate_technical_plan(
                            user_query=user_query,
                            plan_guidance=plan,
                            execution_context=execution_context
                        )
                    else:
                        decision = await self._decide_next_action(
                            user_query=user_query,
                            plan_guidance=plan,
                            execution_context=execution_context,
                            iteration=iteration,
                        )
                except Exception as e:
                    logger.error(f"Iteration {iteration}: _decide_next_action failed: {e}")
                    yield await self._stream_event("error", {
                        "message": f"Decision-making failed at iteration {iteration}: {e}"
                    })
                    status = "error"
                    break

                action = decision.get("action", "code")
                reasoning = decision.get("reasoning", "")
                thinking = decision.get("thinking", "")
                code = decision.get("code", "")

                # ----- Check if LLM declared DONE -----
                if action == "done":
                    logger.info(f"Iteration {iteration}: LLM declared DONE — {reasoning}")
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

                # ----- Execute code -----
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

                logger.info(f"Iteration {iteration}/{self.max_iterations}: {reasoning}")

                # Try to execute this iteration with retries
                iteration_success = False
                last_error = None
                _stashed_failed_code = None
                _stashed_error_type = None

                for attempt in range(self.max_retries_per_iteration):
                    try:
                        if attempt > 0:
                            # Re-generate code with error feedback
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

                        # Pre-execution AST validation
                        validation_err = self._validate_code_syntax(
                            code, execution_context.get("available_variables", {})
                        )
                        if validation_err:
                            logger.warning(f"Iteration {iteration}: AST validation failed: {validation_err}")
                            last_error = f"Code validation error: {validation_err}"
                            self._recent_failed_codes.append(code)
                            continue

                        # Doom loop check — abort if LLM keeps generating nearly identical failing code
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
                            break

                        yield await self._stream_event("code_generated", {
                            "iteration": iteration,
                            "code": code,
                            "attempt": attempt + 1,
                        })

                        # Add code to notebook
                        if self.notebook_builder:
                            self.notebook_builder.add_code_cell(code)
                            self.notebook_builder.save()

                        # Execute
                        execution_result = await self._execute_code(code)

                        yield await self._stream_event("iteration_execution", {
                            "iteration": iteration,
                            "attempt": attempt + 1,
                            "stdout": execution_result.get("stdout", ""),
                            "stderr": execution_result.get("stderr", ""),
                            "exception": execution_result.get("exception"),
                        })

                        # Add output to notebook
                        if self.notebook_builder and execution_result.get("stdout"):
                            self.notebook_builder.add_output_to_last_cell(
                                execution_result.get("stdout", "")
                            )
                            self.notebook_builder.save()

                        # Check for errors — explicit exception OR error patterns in stdout
                        detected_error = execution_result.get("exception")
                        if not detected_error and self._stdout_contains_error(
                            execution_result.get("stdout", "")
                        ):
                            detected_error = self._extract_error_from_stdout(
                                execution_result.get("stdout", "")
                            )
                            execution_result["exception"] = detected_error
                            logger.info(
                                f"Iteration {iteration}: detected error in stdout: "
                                f"{detected_error[:150]}"
                            )

                        if detected_error:
                            last_error = detected_error
                            _stashed_failed_code = code
                            _stashed_error_type, _, _ = self._classify_error(detected_error)
                            logger.warning(
                                f"Iteration {iteration} failed (attempt {attempt + 1}): "
                                f"{last_error[:200]}"
                            )

                            # Track for doom loop detection
                            self._recent_failed_codes.append(code)
                            if len(self._recent_failed_codes) > self.doom_loop_threshold * 2:
                                self._recent_failed_codes = self._recent_failed_codes[
                                    -self.doom_loop_threshold:
                                ]

                            # Run diagnostic probe on first failure
                            if attempt == 0:
                                diag_code = await self._generate_diagnostic_code(
                                    code, last_error,
                                    execution_context.get("available_variables", {})
                                )
                                if diag_code:
                                    diag_result = await self._execute_code(diag_code)
                                    diag_output = diag_result.get("stdout", "")
                                    if diag_output:
                                        last_error += f"\n\nDIAGNOSTIC OUTPUT:\n{diag_output[:300]}"

                            if attempt < self.max_retries_per_iteration - 1:
                                yield await self._stream_event("iteration_retry", {
                                    "iteration": iteration,
                                    "attempt": attempt + 1,
                                    "error": last_error,
                                    "message": "Retrying with error feedback..."
                                })
                            continue

                        # Success! Update context
                        iteration_success = True
                        self._recent_failed_codes = []  # reset doom loop buffer on success

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

                        # Lightweight FINAL_RESULT check — see DA agent
                        # for why this is needed
                        fr_exists = await self._check_final_result_in_kernel()

                        # Extract new variables from kernel
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
                        # the lightweight check found it
                        if fr_exists and "FINAL_RESULT" not in new_vars:
                            logger.warning(
                                "Iteration %d: FINAL_RESULT exists in kernel "
                                "but _get_kernel_variables() missed it!",
                                iteration,
                            )
                            new_vars["FINAL_RESULT"] = {"type": "dict"}
                            execution_context["available_variables"] = new_vars

                        # Re-profile when new DataFrames appear in the kernel
                        # (so the LLM sees string values from newly loaded tables)
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
                                    f"Re-profiled after iteration {iteration}: "
                                    f"new={current_df_names - existing_profile_keys}"
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

                        # Detect silent failures (empty DataFrames, etc.)
                        is_valid, validation_issue = await self._validate_step_output(
                            {"step_num": iteration, "description": reasoning},
                            new_vars, prev_vars
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
                                f"Iteration {iteration}: zero-row self-correction "
                                f"triggered: {validation_issue}"
                            )
                            iteration_success = False
                            yield await self._stream_event("iteration_retry", {
                                "iteration": iteration,
                                "attempt": attempt + 1,
                                "error": last_error,
                                "message": "Zero-row result detected — retrying "
                                           "with diagnostic context..."
                            })
                            continue  # Go back to retry loop
                        elif not is_valid:
                            # Last attempt — can't retry, just warn
                            logger.warning(
                                f"Iteration {iteration} silent failure "
                                f"(no retries left): {validation_issue}"
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

                        # Context Window Optimization: Suppress raw stdout for successes
                        # to prevent prompt bloat and force the LLM to rely on 'available_variables'
                        # which now explicitly contains the dataframe samples and intent.
                        raw_output = execution_result.get("stdout", "").strip()
                        if raw_output:
                            raw_output = "[Execution Successful. Output suppressed to save context. Read variables from AVAILABLE VARIABLES section.]"

                        # Record completed iteration
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
                                        "iteration": f"summary({batch[0]['iteration']}-{batch[-1]['iteration']})",
                                        "reasoning": summary_text,
                                        "code": "",
                                        "output": "",
                                        "variables": new_vars,
                                    }]
                                    logger.info(
                                        f"Context compacted: summarized iterations "
                                        f"{batch[0]['iteration']}-{batch[-1]['iteration']}"
                                    )
                                except Exception as compact_err:
                                    logger.warning(f"Context compaction skipped: {compact_err}")

                        logger.info(f"Iteration {iteration} completed successfully")
                        consecutive_failures = 0

                        # Early stop: if FINAL_RESULT was set in the kernel,
                        # auto-declare done — don't waste iterations going backwards.
                        logger.debug(
                            "Iteration %d: checking for FINAL_RESULT in new_vars. "
                            "Keys: %s",
                            iteration,
                            list(new_vars.keys()) if new_vars else "EMPTY",
                        )
                        if "FINAL_RESULT" in new_vars:
                            logger.info(
                                f"Iteration {iteration}: FINAL_RESULT detected in kernel "
                                f"— auto-declaring done."
                            )
                            if self.notebook_builder:
                                self.notebook_builder.add_markdown_cell(
                                    f"## ✅ Analysis Complete (iteration {iteration})\n\n"
                                    f"FINAL_RESULT was set — stopping."
                                )
                                self.notebook_builder.save()
                            status = "done"

                        break  # Exit retry loop on success

                    except Exception as e:
                        last_error = str(e)
                        logger.error(
                            f"Error in iteration {iteration}, attempt {attempt + 1}: {e}"
                        )
                        if attempt < self.max_retries_per_iteration - 1:
                            yield await self._stream_event("iteration_retry", {
                                "iteration": iteration,
                                "attempt": attempt + 1,
                                "error": str(e),
                            })

                if not iteration_success:
                    consecutive_failures += 1
                    # Record the failure so the NEXT _decide_next_action knows what failed
                    execution_context.setdefault("failed_iterations", []).append({
                        "iteration": iteration,
                        "error": (last_error or "unknown")[:500],
                        "code_snippet": (code or "")[:300],
                    })
                    logger.warning(
                        f"Iteration {iteration} failed after {self.max_retries_per_iteration} "
                        f"attempts (consecutive failures: {consecutive_failures})"
                    )

                    if self.notebook_builder:
                        self.notebook_builder.add_markdown_cell(
                            f"❌ **Iteration {iteration} FAILED** after "
                            f"{self.max_retries_per_iteration} attempts.\n\n"
                            f"Last error: `{last_error[:300] if last_error else 'Unknown'}`"
                        )
                        self.notebook_builder.save()

                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        yield await self._stream_event("error", {
                            "message": (
                                f"{consecutive_failures} consecutive iterations failed. "
                                f"Stopping to prevent wasted compute."
                            ),
                            "iteration": iteration,
                            "last_error": last_error,
                        })
                        status = "error"
                        break
                    # Otherwise, continue — the LLM may recover in the next iteration

            # ═══════════════════════════════════════════════════════════════════
            # POST-LOOP: Generate final result
            # ═══════════════════════════════════════════════════════════════════
            completed_count = len(execution_context["completed_iterations"])

            if status == "error" or completed_count == 0:
                failure_msg = (
                    f"Analysis incomplete — {completed_count} iterations completed, "
                    f"stopped due to errors."
                )
                final_result = {
                    "prediction": failure_msg,
                    "text_output": failure_msg,
                    "dataframe": None,
                    "iterations_completed": completed_count,
                    "timestamp": utcnow().isoformat(),
                    "pipeline_failed": True,
                    "_agent_usage": {
                        k: list(v) if isinstance(v, set) else v
                        for k, v in self.usage_stats.items()
                    },
                }
                yield await self._stream_event("final_result", final_result)

                if self.notebook_builder:
                    self.notebook_builder.add_markdown_cell(
                        f"---\n\n## ❌ Analysis Aborted\n\n{failure_msg}"
                    )
                    self.notebook_builder.save()
            else:
                # All iterations succeeded or LLM declared DONE
                yield await self._stream_event("status", {"message": "Fetching result data..."})
                final_df_records = await self._fetch_generated_dataframe()

                # Fetch ALL Plotly charts, fallback to single chart extraction
                all_charts = await self._fetch_all_generated_charts()
                if not all_charts:
                    single = await self._fetch_generated_chart()
                    if single:
                        all_charts = [{"name": "_generated_plotly_fig_", "figure": single}]

                yield await self._stream_event("status", {"message": "Generating final result..."})
                final_result = await self._generate_final_result(execution_context)

                final_result["dataframe"] = final_df_records
                if all_charts:
                    final_result["charts"] = all_charts
                    final_result["chart"] = all_charts[0]["figure"]  # backward compat

                yield await self._stream_event("final_result", final_result)

                if self.notebook_builder:
                    summary_text = final_result.get(
                        "text_output", final_result.get("prediction", "")
                    )
                    self.notebook_builder.add_markdown_cell(
                        f"---\n\n## Final Result\n\n{summary_text}"
                    )
                    nb_path = self.notebook_builder.save()
                    final_result["notebook_path"] = str(nb_path)

            yield await self._stream_event("status", {
                "message": (
                    f"Analysis complete ({completed_count} iterations)"
                    if status != "error"
                    else "Analysis incomplete due to errors"
                ),
                "notebook_path": (
                    str(self.notebook_builder.filepath) if self.notebook_builder else None
                ),
            })

        except Exception as e:
            logger.error(f"Error in execute_analysis: {e}\n{traceback.format_exc()}")
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








    # ═══════════════════════════════════════════════════════════════════════════
    # RECURSIVE LOOP — Core Decision Methods
    # ═══════════════════════════════════════════════════════════════════════════

    async def _generate_technical_plan(
        self,
        user_query: str,
        plan_guidance: str,
        execution_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Specialized LLM call for Iteration 1 to create a high-fidelity technical plan.
        Translates 'Analyst Guidance' into a verbose, tech-heavy TASKS list.
        """
        system_prompt = (
            "You are a Senior Technical Data Architect. Your goal is to translate a business query and analyst guidance "
            "into a mandatory technical strategy for a recursive data science agent.\n\n"
            "RULES:\n"
            "1. Output ONLY valid JSON with keys: 'thinking' and 'code'.\n"
            "2. Your 'code' MUST ONLY initialize the following variables:\n"
            "   TASKS = [...]  # A list of 5-8 strings\n"
            "   COMPLETED_TASKS = []\n"
            "   _VAR_INTENT_ = {}\n"
            "3. Every string in 'TASKS' MUST be 'verbose' and tech-heavy. Include EXACT table names, column names, "
            "   and specific sargable filter values (e.g. 'Load TotalValue from NE_DMSSaleReportTable where OrderDate >= 2025-01-01') "
            "   from the PROVIDED guidance.\n"
            "4. Do NOT include any SQL loading or analysis logic in the code. This is a PLANNING-ONLY step.\n"
            "5. Do NOT include any explanation or markdown fences."
        )

        user_message = (
            f"USER QUERY: {user_query}\n\n"
            f"ANALYST GUIDANCE:\n{plan_guidance}\n\n"
            "Based on the guidance, create a 5-8 step verbose technical plan that implements the strategy exactly as described."
        )

        response = await self.llm_client.generate_completion(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.1,  # Low temperature for precise planning
        )
        self._update_usage(response.get("usage"))
        content = (response.get("content") or "").strip()
        
        # Parse JSON
        try:
            if content.startswith("```"):
                content = re.sub(r"```[a-z]*\n|```", "", content)
            parsed = json.loads(content)
            return {
                "action": "code",
                "thinking": parsed.get("thinking", "Initializing high-fidelity technical plan."),
                "reasoning": "Technical Planning Step",
                "code": parsed.get("code", "")
            }
        except Exception as e:
            logger.error(f"Planning step JSON parse failed: {e}")
            raise ValueError(f"Technical planning failed: {e}")


    @traceable(name="coder_decide_action")
    async def _decide_next_action(
        self,
        user_query: str,
        plan_guidance: str,
        execution_context: Dict[str, Any],
        iteration: int,
    ) -> Dict[str, Any]:
        """
        Ask the LLM to decide the next action: generate code or declare done.

        Dispatches to tiered prompting (static system + compact dynamic user
        message) when ``USE_TIERED_PROMPTS`` is enabled, or falls back to the
        legacy monolithic prompt builder.
        """
        from config.system_config import USE_TIERED_PROMPTS

        if USE_TIERED_PROMPTS:
            # Tier 1: Build static system context on first call (cached by provider)
            if self._static_system_context is None:
                self._static_system_context = await self._build_static_system_context(
                    user_query, plan_guidance, execution_context
                )

            system_prompt = self._static_system_context
            user_message = self._build_dynamic_user_message(
                user_query, plan_guidance, execution_context, iteration
            )

            response = await self.llm_client.generate_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=self.temperature,
            )
            self._update_usage(response.get("usage"))
            content = (response.get("content") or "").strip()
            return self._parse_decision_response(content)

        # Legacy path — full monolithic prompt
        return await self._decide_next_action_legacy(
            user_query, plan_guidance, execution_context, iteration
        )

    async def _decide_next_action_legacy(
        self,
        user_query: str,
        plan_guidance: str,
        execution_context: Dict[str, Any],
        iteration: int,
    ) -> Dict[str, Any]:
        """Legacy monolithic prompt builder — kept for backward compatibility.

        Used when ``USE_TIERED_PROMPTS=false``.
        """
        prompt_parts = []

        # 1. User query
        prompt_parts.append(f"USER QUERY: {user_query}")
        prompt_parts.append("")

        # 2. Plan guidance (directional, not rigid)
        prompt_parts.append("PLAN GUIDANCE (use as direction, not a rigid checklist):")
        prompt_parts.append(plan_guidance)
        prompt_parts.append("")

        # 2.5 File schemas (EXACT column names from file metadata — highest priority)
        file_schemas = execution_context.get("file_schemas", {})
        if file_schemas:
            prompt_parts.append("FILE SCHEMAS (EXACT column names — use ONLY these, case-sensitive):")
            for fname, schema in file_schemas.items():
                rows_info = f", {schema['num_rows']:,} rows" if schema.get("num_rows") else ""
                prompt_parts.append(f"  {fname} ({schema.get('path', '')}{rows_info})")
                prompt_parts.append(f"    columns = {schema.get('columns', [])}")
                if schema.get("types"):
                    prompt_parts.append(f"    types   = {schema.get('types', {})}")
            prompt_parts.append("")
            prompt_parts.append(
                "⚠️ CRITICAL: The plan guidance may use WRONG column names (e.g. LAST_ISSUE_DATE). "
                "ALWAYS use the EXACT column names from FILE SCHEMAS above instead. "
                "When using pd.read_parquet(columns=[...]), use ONLY names from the schema."
            )
            prompt_parts.append("")

        # 2.7 Business knowledge (table descriptions, column meanings, domain terms)
        knowledge_ctx = execution_context.get("knowledge_context", {})
        
        schema_tables = [Path(f).stem for f in file_schemas.keys()]
        if not schema_tables and self._is_live_db:
            schema_tables = self._planned_tables
            
        if knowledge_ctx and schema_tables:
            from util.knowledge_filter import (
                compress_table_introductions_for_coding,
                compress_data_descriptions_for_coding,
                compress_terminology_for_coding,
                _approx_token_count,
            )
            from config.system_config import MAX_CODING_KNOWLEDGE_TOKENS
            
            knowledge_lines: list = []
            budget = MAX_CODING_KNOWLEDGE_TOKENS

            # Priority 1: Table introductions (cheapest, highest-level)
            intros = compress_table_introductions_for_coding(
                knowledge_ctx.get("table_introductions_xml", ""), schema_tables,
            )
            if intros:
                cost = _approx_token_count(intros)
                if cost <= budget:
                    knowledge_lines.append("TABLE DESCRIPTIONS:")
                    knowledge_lines.append(intros)
                    budget -= cost

            # Priority 2: Column descriptions (most valuable for correct column selection)
            descs = compress_data_descriptions_for_coding(
                knowledge_ctx.get("data_descriptions", {}), schema_tables,
            )
            if descs:
                cost = _approx_token_count(descs)
                if cost <= budget:
                    knowledge_lines.append("")
                    knowledge_lines.append(
                        "COLUMN DESCRIPTIONS (use to select correct columns):"
                    )
                    knowledge_lines.append(descs)
                    budget -= cost

            # Priority 3: Domain terminology (if budget allows)
            terms = compress_terminology_for_coding(
                knowledge_ctx.get("domain_terminology", ""),
            )
            if terms:
                cost = _approx_token_count(terms)
                if cost <= budget:
                    knowledge_lines.append("")
                    knowledge_lines.append("DOMAIN TERMINOLOGY:")
                    knowledge_lines.append(terms)

            if knowledge_lines:
                prompt_parts.append(
                    "BUSINESS KNOWLEDGE (understand what the data means):"
                )
                prompt_parts.extend(knowledge_lines)
                prompt_parts.append("")

        # 2.8 Learned patterns from prior analyses (agent lessons)
        try:
            raw_db = self._get_raw_db()
            if raw_db:
                from services.agent_lesson_service import AgentLessonService
                from config.system_config import MAX_LESSONS_TOKENS
                lesson_svc = AgentLessonService(raw_db)
                # Filter lessons by planned tables for relevance
                planned_tables = getattr(self, "_planned_tables", None)
                schema_tables = [Path(f).stem for f in file_schemas.keys()] if file_schemas else None
                filter_tables = planned_tables or schema_tables
                lessons_text = await lesson_svc.format_lessons_for_prompt(
                    self.client_id,
                    tables=filter_tables,
                    max_tokens=MAX_LESSONS_TOKENS,
                )
                if lessons_text:
                    prompt_parts.append("LEARNED PATTERNS (from prior analyses — follow these strictly):")
                    prompt_parts.append(lessons_text)
                    prompt_parts.append("")
        except Exception as le:
            logger.debug("Lesson injection skipped: %s", le)

        # 2.9 Client data profile (geography, formatting, industry)
        client_profile = knowledge_ctx.get("client_data_profile", "")
        if client_profile:
            from config.system_config import MAX_DATA_PROFILE_TOKENS
            profile_cost = len(client_profile) // 4
            if profile_cost <= MAX_DATA_PROFILE_TOKENS:
                prompt_parts.append("CLIENT DATA PROFILE (formatting & locale guidance):")
                prompt_parts.append(client_profile)
                prompt_parts.append("")

        # 2.10 User preferences (chart type, detail level, time granularity)
        try:
            raw_db = self._get_raw_db()
            user_id = getattr(self, "_user_id", None)
            if raw_db and user_id:
                from services.user_preference_service import UserPreferenceService
                from services.preference_extractor import PreferenceExtractor
                from config.system_config import MAX_USER_PREFERENCES_TOKENS
                pref_svc = UserPreferenceService(raw_db)
                # Current query prefs override stored
                current_prefs = PreferenceExtractor.extract_as_dict(user_query) if user_query else {}
                prefs_text = await pref_svc.format_for_prompt(
                    self.client_id,
                    user_id,
                    current_query_prefs=current_prefs,
                    max_tokens=MAX_USER_PREFERENCES_TOKENS,
                )
                if prefs_text:
                    prompt_parts.append("USER PREFERENCES (respect these for visualization and formatting):")
                    prompt_parts.append(prefs_text)
                    prompt_parts.append("")
        except Exception:
            pass  # Non-fatal

        # 2.6 Multi-table join context (when multiple files are involved)
        data_profile = execution_context.get("data_profile", {})
        if file_schemas and len(file_schemas) > 1:
            prompt_parts.append("MULTI-TABLE JOIN CONTEXT:")

            # Build column→files index
            col_to_files: Dict[str, list] = {}
            file_row_counts: Dict[str, int] = {}
            for fname, schema in file_schemas.items():
                for col in schema.get("columns", []):
                    col_to_files.setdefault(col, []).append(fname)
                # Get row count: prefer data_profile, fallback to parquet metadata
                if schema.get("num_rows") is not None:
                    file_row_counts[fname] = schema["num_rows"]
                stem = Path(fname).stem
                stem_lower = stem.lower().replace("-", "_")
                for ds_name, prof in data_profile.items():
                    ds_lower = ds_name.lower().replace("-", "_")
                    if (ds_lower == stem_lower
                            or ds_lower.endswith(stem_lower)
                            or stem_lower.endswith(ds_lower)
                            or stem_lower in ds_lower
                            or ds_lower in stem_lower):
                        shape = prof.get("shape", [0])
                        file_row_counts[fname] = shape[0] if shape else 0

            # Shared columns (appear in 2+ files — strong join key candidates)
            shared_cols = {
                col: files for col, files in col_to_files.items()
                if len(files) > 1
            }
            if shared_cols:
                prompt_parts.append("  Shared columns (potential join keys):")
                for col, files in shared_cols.items():
                    prompt_parts.append(f"    {col}: appears in {', '.join(files)}")

            # Near-match columns (e.g., ORGANIZATION_ID ↔ INV_ORG_ID)
            # Detect: substring match OR shared suffix like _ID, _NAME, _CODE
            def _cols_near_match(a: str, b: str) -> bool:
                if a == b:
                    return False
                # Substring match
                if a in b or b in a:
                    return True
                # Shared meaningful suffix (e.g., both end with _ID, _NAME)
                for suffix in ("_ID", "_NAME", "_CODE", "_KEY", "_NUM"):
                    if a.endswith(suffix) and b.endswith(suffix):
                        # Strip suffix and check for overlap in the base
                        base_a = a[:-len(suffix)].rstrip("_")
                        base_b = b[:-len(suffix)].rstrip("_")
                        if base_a and base_b and (
                            base_a in base_b or base_b in base_a
                        ):
                            return True
                return False

            all_cols_by_file = {
                fname: set(s.get("columns", []))
                for fname, s in file_schemas.items()
            }
            near_matches = []
            fnames_list = list(all_cols_by_file.keys())
            for i in range(len(fnames_list)):
                for j in range(i + 1, len(fnames_list)):
                    for col_a in all_cols_by_file[fnames_list[i]]:
                        for col_b in all_cols_by_file[fnames_list[j]]:
                            if _cols_near_match(col_a, col_b):
                                near_matches.append(
                                    (col_a, fnames_list[i], col_b, fnames_list[j])
                                )
            if near_matches:
                prompt_parts.append(
                    "  Near-match columns (VERIFY overlap before joining — names differ):"
                )
                for col_a, f_a, col_b, f_b in near_matches[:10]:
                    prompt_parts.append(f"    {col_a} ({f_a}) ↔ {col_b} ({f_b})")

            # Identify small lookup tables — forceful override of plan's column selection
            small_tables = []
            for fname, rows in file_row_counts.items():
                if rows < 10000:
                    small_tables.append(fname)
                    prompt_parts.append(
                        f"  ⚠️ CRITICAL: {fname} is a small table ({rows} rows) "
                        f"— IGNORE the plan's column selection for this file. "
                        f"Load ALL columns: pd.read_parquet(path) with NO columns= parameter."
                    )

            # For small tables where we don't have row count yet, hint based on schema
            # (dimension tables often have few columns vs fact tables)
            if not small_tables:
                for fname, schema in file_schemas.items():
                    n_cols = len(schema.get("columns", []))
                    if n_cols <= 6 and fname not in file_row_counts:
                        prompt_parts.append(
                            f"  ℹ️ {fname} has only {n_cols} columns — likely a small "
                            f"lookup table. Load ALL columns to avoid needing to reload."
                        )

            prompt_parts.append(
                "\n  ⚠️ MULTI-TABLE JOIN RULE: Before joining two tables, you MUST:\n"
                "    1. For small lookup/dimension tables: load ALL columns "
                "(do NOT use columns= parameter)\n"
                "    2. BEFORE joining, verify join key overlap:\n"
                "       overlap = set(df_a['col_a'].unique()) & set(df_b['col_b'].unique())\n"
                "       print(f'Overlap: {len(overlap)} common values')\n"
                "    3. If overlap is 0, try OTHER candidate columns — check near-matches above\n"
                "    4. Column names may differ (e.g., ORGANIZATION_ID ↔ INV_ORG_ID) "
                "— check VALUES, not just names\n"
                "    5. Use the column pair with the HIGHEST overlap for the join\n"
                "    6. FILTER-BY-VALUE (same rules apply): When using a value from "
                "table A to filter table B (e.g., .isin(), == comparison):\n"
                "       - Verify the looked-up value EXISTS in the target column "
                "BEFORE filtering\n"
                "       - If 0 rows result, the value maps to a DIFFERENT column "
                "in table B\n"
                "       - Print unique values in candidate columns to find the "
                "correct mapping"
            )
            prompt_parts.append("")

        # 3. Dataset profile (runtime — only available after data is loaded)
        if not data_profile:
            data_profile = execution_context.get("data_profile", {})
        if data_profile:
            prompt_parts.append("DATASET PROFILE (loaded DataFrames):")
            for ds_name, prof in data_profile.items():
                prompt_parts.append(f"  {ds_name}:")
                if prof.get("shape"):
                    prompt_parts.append(f"    shape   = {prof['shape']}")
                if prof.get("columns"):
                    prompt_parts.append(f"    columns = {prof['columns']}")
                if prof.get("dtypes"):
                    prompt_parts.append(f"    dtypes  = {prof['dtypes']}")
                if prof.get("null_counts"):
                    prompt_parts.append(f"    nulls   = {prof['null_counts']}")
                if prof.get("sample_row"):
                    prompt_parts.append(
                        f"    sample  = {str(prof['sample_row'][0])[:200]}"
                    )
                if prof.get("string_values"):
                    prompt_parts.append(f"    string_values (actual values in string columns):")
                    for col_name, sv in prof["string_values"].items():
                        unique_ct = sv.get("unique_count", "?")
                        top_vals = sv.get("top_values", [])[:10]
                        suffix = f" ... ({unique_ct} unique total)" if unique_ct > 10 else ""
                        prompt_parts.append(
                            f"      {col_name} ({unique_ct} unique): {top_vals}{suffix}"
                        )
            prompt_parts.append(
                "USE ONLY THESE EXACT COLUMN NAMES — case-sensitive, no variation."
            )
            has_string_values = any(
                prof.get("string_values") for prof in data_profile.values()
            )
            if has_string_values:
                prompt_parts.append(
                    "⚠️ STRING FILTER RULE: When the user's query involves filtering on a string "
                    "column, you MUST follow this iterative discovery process:\n"
                    "  1. EXPLORE: Print unique values matching the user's keyword using "
                    "str.contains(r'keyword', case=False, na=False), then print the matching "
                    "unique values and their counts.\n"
                    "  2. REVIEW & DECIDE: In the NEXT iteration, look at the discovered values. "
                    "Decide which ones are relevant to the user's question. Not all matches may "
                    "be relevant (e.g., 'cement mixer' is NOT cement inventory).\n"
                    "  3. FILTER: Apply the final filter using the verified values.\n"
                    "Refer to the string_values above for initial awareness of what's in each column."
                )
            prompt_parts.append("")

        # 4. Available variables
        available_vars = execution_context.get("available_variables", {})
        if available_vars:
            vars_lines = []
            dataframes = []
            for name, info in available_vars.items():
                if isinstance(info, dict):
                    type_str = info.get("type", "Unknown")
                    if type_str == "DataFrame":
                        dataframes.append(name)
                    details = ""
                    if "columns" in info:
                        cols = info["columns"]
                        if len(cols) > 10:
                            cols_str = str(cols[:10])[:-1] + ", ...]"
                        else:
                            cols_str = str(cols)
                        details = f" columns={cols_str}"
                    if "shape" in info:
                        details += f" shape={info['shape']}"
                    vars_lines.append(f"- {name} ({type_str}){details}")
                else:
                    vars_lines.append(f"- {name} ({info})")

            prompt_parts.append(
                f"AVAILABLE VARIABLES:\n" + "\n".join(vars_lines)
            )

            # Smart aliasing check
            if dataframes and "df" not in dataframes:
                if len(dataframes) == 1:
                    prompt_parts.append(
                        f"\n⚠️ CRITICAL: The dataframe is named '{dataframes[0]}'. "
                        f"DO NOT use 'df'. Use '{dataframes[0]}' instead."
                    )
                else:
                    prompt_parts.append(
                        f"\n⚠️ CRITICAL: Available dataframes: {', '.join(dataframes)}. "
                        f"DO NOT use 'df' unless it is defined."
                    )
            prompt_parts.append(
                "⚠️ CRITICAL: These are the ONLY variables in memory. "
                "Data is NOT preloaded as table-name globals "
                "(e.g., IFFCO_INV_AI_CONS does NOT exist as a variable). "
                "Use ONLY the exact variable names listed above."
            )

            # If FINAL_RESULT already exists, tell the LLM to stop immediately
            if "FINAL_RESULT" in available_vars:
                prompt_parts.append(
                    "\n🛑 STOP — FINAL_RESULT IS ALREADY SET IN THE KERNEL. "
                    "Your analysis is COMPLETE. You MUST return action: \"done\" immediately. "
                    "Do NOT generate more code. Do NOT re-compute or re-set FINAL_RESULT. "
                    "The answer has already been produced."
                )

            prompt_parts.append("")

        # 5. Data Access / Dataset Context
        loaded_datasets = execution_context.get("loaded_datasets", [])
        if self._is_live_db:
            prompt_parts.append("DATA ACCESS (LIVE DATABASE):")
            prompt_parts.append("⚠️ CRITICAL: You are connected to a live SQL database.")
            prompt_parts.append("DO NOT use pd.read_parquet() or pd.read_csv().")
            prompt_parts.append("You MUST use the pre-defined python function `read_sql_query(query_string)` to fetch data.")
            prompt_parts.append("Example: df = read_sql_query('SELECT TOP 100 * FROM [table]')")
            prompt_parts.append("")
        elif loaded_datasets:
            prompt_parts.append(
                "LOADED DATASETS "
                "(⚠️ CRITICAL: when calling pd.read_parquet/read_csv, "
                "use the EXACT absolute path below — NEVER a bare filename like 'data.parquet'):"
            )
            for ds in loaded_datasets:
                prompt_parts.append(
                    f"  - path='{ds.get('path', '?')}' variable={ds.get('variable', '?')} "
                    f"format={ds.get('format', '?')}"
                )
            prompt_parts.append("")

        # 6a. Failed iterations — make previous failures visible to the LLM
        failed_iters = execution_context.get("failed_iterations", [])
        if failed_iters:
            prompt_parts.append("⚠️ FAILED ITERATIONS (these approaches ALREADY FAILED — learn from them):")
            for fi in failed_iters[-3:]:  # Cap at last 3 failures
                prompt_parts.append(f"  Iteration {fi['iteration']} FAILED:")
                prompt_parts.append(f"    Error: {fi['error'][:300]}")
                if fi.get("code_snippet"):
                    prompt_parts.append(f"    Attempted code (snippet): {fi['code_snippet']}")
            prompt_parts.append(
                "  → Do NOT repeat these failed approaches. Fix the SPECIFIC error "
                "(e.g., wrong variable name, missing merge step) and continue from "
                "AVAILABLE VARIABLES."
            )
            prompt_parts.append("")

        # 6b. Iteration history (with code and output previews)
        completed = execution_context.get("completed_iterations", [])
        if completed:
            # Build "accomplished state" summary — reinforces what's DONE
            accomplished_lines = [
                "ACCOMPLISHED SO FAR (DO NOT REPEAT ANY OF THIS — all variables are alive in kernel):"
            ]
            for item in completed:
                iter_id = item.get("iteration", "?")
                reasoning_str = item.get("reasoning", "")
                accomplished_lines.append(f"  - Iteration {iter_id}: {reasoning_str}")
            # List current DataFrames with shapes
            current_dfs = [
                f"  - {name} ({info.get('type','?')}, shape={info.get('shape','?')})"
                for name, info in available_vars.items()
                if isinstance(info, dict) and info.get("type") == "DataFrame"
            ]
            if current_dfs:
                accomplished_lines.append("  Current DataFrames in memory:")
                accomplished_lines.extend(current_dfs)
            accomplished_lines.append(
                "  ⚠️ Write ONLY new incremental code. Do NOT re-load, re-import, or re-compute anything above."
            )
            prompt_parts.extend(accomplished_lines)
            prompt_parts.append("")

            prompt_parts.append("COMPLETED ITERATIONS (detailed):")
            for item in completed:
                iter_id = item.get("iteration", "?")
                reasoning_str = item.get("reasoning", "")
                thinking_str = item.get("thinking", "")
                code_preview = item.get("code", "")
                if code_preview and len(code_preview) > self.code_preview_max_chars:
                    code_preview = code_preview[:self.code_preview_max_chars] + "\n# ...[truncated]"
                output_preview = item.get("output", "")
                if output_preview and len(output_preview) > self.output_preview_max_chars:
                    output_preview = output_preview[:self.output_preview_max_chars] + "\n...[truncated]"

                prompt_parts.append(f"  --- Iteration {iter_id}: {reasoning_str} ---")
                if thinking_str:
                    prompt_parts.append(f"  Thinking: {thinking_str}")
                if code_preview:
                    prompt_parts.append(f"  Code:\n{code_preview}")
                if output_preview:
                    prompt_parts.append(f"  Output:\n{output_preview}")
            prompt_parts.append("")
        else:
            prompt_parts.append("No iterations completed yet — this is the first iteration.")
            prompt_parts.append("")

        # 7. Warnings
        warnings = execution_context.get("warnings", [])
        if warnings:
            zero_row_warnings = [
                w for w in warnings if "ZERO_ROW" in w or "became empty" in w
            ]
            other_warnings = [w for w in warnings if w not in zero_row_warnings]

            if zero_row_warnings:
                prompt_parts.append(
                    "⚠️ CRITICAL — ZERO-ROW RESULTS DETECTED "
                    "(you MUST address these):"
                )
                for w in zero_row_warnings:
                    prompt_parts.append(f"  ❌ {w}")
                prompt_parts.append(
                    "  ACTION REQUIRED: Do NOT proceed with empty DataFrames. "
                    "Re-examine filter columns and values. "
                    "Try alternative columns (e.g., INV_ORG_ID instead of "
                    "ORGANIZATION_ID). Print unique values to verify."
                )
                prompt_parts.append("")

            if other_warnings:
                prompt_parts.append("WARNINGS from prior iterations:")
                for w in other_warnings:
                    prompt_parts.append(f"  - {w}")
                prompt_parts.append("")

        # 8. Instructions — the core recursive directive
        remaining = self.max_iterations - iteration
        convergence_note = ""
        if remaining <= 2:
            convergence_note = (
                f"⚠️ URGENT: Only {remaining} iteration(s) remaining. "
                "You MUST assemble FINAL_RESULT in this iteration. "
                "Use whatever data you have — a partial answer is better than no answer."
            )
        elif remaining <= self.max_iterations // 2:
            convergence_note = (
                f"Note: {remaining} iterations remaining. "
                "If you have enough data, proceed to computation and FINAL_RESULT."
            )

        prompt_parts.extend([
            f"ITERATION: {iteration} / {self.max_iterations}",
            *([convergence_note] if convergence_note else []),
            "",
            "YOUR TASK:",
            "Based on the user query, plan guidance, completed iterations and their outputs,",
            "decide what to do NEXT. You have two options:",
            "",
            '1. action="code" — Write Python code for the next logical step.',
            '2. action="done" — Declare the analysis complete (use ONLY when the user query is fully answered).',
            "",
            "RULES:",
            "- ITERATION 0 PLANNING: In your very first execution step (Iteration 0), you MUST initialize two python lists: `TASKS = [...]` (strings of the exact sub-tasks needed) and `COMPLETED_TASKS = []`.",
            "- INTENT REGISTRY: Also in Iteration 0, initialize `_VAR_INTENT_ = {}`. Whenever you create a meaningful DataFrame, add a 1-sentence summary of what it holds (e.g., `_VAR_INTENT_['df_sales'] = 'Filtered 2025 sales'`).",
            "- STATE PROGRESSION: In every step, append the task you just finished to `COMPLETED_TASKS.append(...)`.",
            "- Each iteration builds on previous ones. Variables from prior iterations are alive in memory.",
            "- Do NOT re-import libraries or re-load data that was already loaded.",
            "- NEVER re-compute DataFrames you already made; pass them continuously from task to task.",
            "- Do NOT set FINAL_RESULT until all your `TASKS` are in `COMPLETED_TASKS`. Premature completion ruins the analysis.",
            "- If this is the LAST step of the analysis, store your primary result in FINAL_RESULT.",
            '  Example: FINAL_RESULT = result_df  or  FINAL_RESULT = {"key": value}',
            "- Set FINAL_RESULT BEFORE declaring action='done'.",
            "",
            "WORKFLOW GUIDANCE (adapt based on query complexity):",
            "- For SIMPLE queries: plan → load/compute → FINAL_RESULT in 3-4 iterations.",
            "- For COMPLEX queries: plan → load → merge → compute → FINAL_RESULT in 5-6 iterations.",
            "- You MAY combine related operations in one cell to close multiple TASKS at once.",
            "- Before any groupby/filter, quickly check the relevant column: print(df['col'].nunique())",
            "- If a merge/join produces 0 rows, investigate immediately — don't proceed with empty data.",
            "- MERGE VERIFICATION: After any pd.merge/join, immediately print:",
            "  1. Result shape vs input shapes (row explosion = wrong keys)",
            "  2. Check for '_x'/'_y' column suffixes (= overlapping non-key columns, likely wrong join keys)",
            "  3. Sample 2-3 rows to sanity-check the joined data",
            "",
            "CODE QUALITY RULES:",
            "- Keep cells to 30-40 lines MAX.",
            "- NEVER re-import libraries or re-load data that's already in AVAILABLE VARIABLES.",
            "- NEVER reference variables that don't exist — check AVAILABLE VARIABLES above.",
            "- Use ONLY variables present in AVAILABLE VARIABLES. Do NOT try to parse or hardcode DataFrames from plain text outputs.",
            "- Every cell MUST end with print() showing what was produced (unless setting FINAL_RESULT).",
            "",
            "PERFORMANCE RULES:",
            "- For LARGE tables (>100K rows): select needed columns: pd.read_parquet(path, columns=['col1','col2'])",
            "- For SMALL lookup/dimension tables (<10K rows): load ALL columns to avoid repeated KeyErrors",
            "- When in doubt about which columns you'll need, load MORE columns — reloading wastes iterations",
            "- Do NOT add .sample() — always process the FULL dataset for accurate results",
            "- Only sample if the user EXPLICITLY asks for it in their query",
            "- Use vectorized operations, not loops.",
            "",
            "FILTER VERIFICATION RULES:",
            "- After EVERY filter (.isin(), .query(), boolean indexing, pd.merge), "
            "IMMEDIATELY check: print(f'Filtered: {result.shape[0]} rows')",
            "- If 0 rows: DO NOT proceed. Instead:",
            "  1. Print unique values in BOTH source and target filter columns",
            "  2. Check if you used the wrong column (e.g., ORG_ID vs INV_ORG_ID)",
            "  3. Try alternative columns ending in _ID, _CODE, _KEY, _NUM",
            "- When using a value from table A to filter table B:",
            "  1. Print the lookup value: print(f'Lookup: {value}')",
            "  2. Verify the value EXISTS in the target column before filtering",
            "  3. Column names often DIFFER between tables for the same entity",
            "  4. Values may also differ: ORG_ID=81 does NOT mean ORGANIZATION_ID=81",
            "- NEVER silently accept 0 rows and proceed to the next step",
            "",
            "RESPONSE FORMAT (return ONLY this JSON, no markdown fences, no extra text):",
            '{',
            '  "action": "code" or "done",',
            '  "thinking": "1-2 sentences: What data do I have, what do I still need, and what will this cell do?",',
            '  "reasoning": "≤6 words, creative step title (e.g. '
            '\'Pulling in the RFQ data\', '
            '\'Linking RFQs to their organization\', '
            '\'Spotting the top revenue drivers\', '
            '\'Assembling the final picture\'). '
            'No column/file/table names. No generic labels like Loading datasets or Merging datasets.",',
            '  "code": "python code to execute (empty string if action is done)"',
            '}',
        ])

        full_prompt = "\n".join(prompt_parts)

        # Use the XML-loaded system prompt so agent rules/constraints are applied
        system_prompt = (self.base_prompt or "") + (
            "\n\nYou are a recursive data science agent. "
            "You observe outputs, decide the next step, and iterate until the analysis is complete. "
            "You MUST respond with valid JSON only — no markdown fences, no explanations."
        )

        response = await self.llm_client.generate_completion(
            system_prompt=system_prompt,
            user_message=full_prompt,
            temperature=self.temperature,
        )
        self._update_usage(response.get("usage"))

        content = (response.get("content") or "").strip()

        # Parse the JSON response
        decision = self._parse_decision_response(content)
        return decision

    def _parse_decision_response(self, content: str) -> Dict[str, Any]:
        """
        Parse the LLM's decision response JSON.
        Falls back gracefully if the response isn't clean JSON.
        """
        # Strip markdown fences if present
        if content.startswith("```json"):
            content = content[len("```json"):].strip()
        if content.startswith("```"):
            content = content[3:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        try:
            parsed = json.loads(content)
            action = parsed.get("action", "code")
            reasoning = parsed.get("reasoning", "")
            thinking = parsed.get("thinking", "")
            code = parsed.get("code", "")

            if action not in ("code", "done"):
                logger.warning(f"Unknown action '{action}', defaulting to 'code'")
                action = "code"

            return {"action": action, "reasoning": reasoning, "thinking": thinking, "code": code}
        except json.JSONDecodeError:
            # If the LLM returned raw Python code instead of JSON, treat it as code
            logger.warning(
                f"Decision response was not valid JSON, treating as raw code: "
                f"{content[:100]!r}"
            )
            # Check if it looks like a "done" declaration
            lower = content.lower()
            if (
                "analysis complete" in lower
                or "final_result" in lower
                and "action" not in lower
            ):
                return {
                    "action": "code",
                    "reasoning": "LLM returned raw code (non-JSON response)",
                    "thinking": "",
                    "code": content,
                }
            return {
                "action": "code",
                "reasoning": "LLM returned raw code (non-JSON response)",
                "thinking": "",
                "code": content,
            }

    async def _regenerate_code_after_error(
        self,
        user_query: str,
        plan_guidance: str,
        execution_context: Dict[str, Any],
        iteration: int,
        failed_code: str,
        error: str,
        attempt: int,
    ) -> str:
        """
        Re-generate code for a failed iteration attempt, incorporating the error
        context and diagnostic output for targeted repairs.
        """
        prompt_parts = []

        prompt_parts.append(f"USER QUERY: {user_query}")
        prompt_parts.append("")

        # Available variables
        available_vars = execution_context.get("available_variables", {})
        if available_vars:
            vars_lines = []
            dataframes = []
            for name, info in available_vars.items():
                if isinstance(info, dict):
                    type_str = info.get("type", "Unknown")
                    if type_str == "DataFrame":
                        dataframes.append(name)
                    details = ""
                    if "columns" in info:
                        cols_str = str(info["columns"][:10]) if len(info["columns"]) > 10 else str(info["columns"])
                        details = f" columns={cols_str}"
                    if "shape" in info:
                        details += f" shape={info['shape']}"
                    
                    # Show intent/value logic identical to tiered prompt
                    intent_str = f" intent=\"{info['intent']}\"" if "intent" in info else ""
                    value_str = ""
                    if name in ["TASKS", "COMPLETED_TASKS", "_VAR_INTENT_"] and "value" in info:
                        value_str = f" value={info['value']}"
                        
                    vars_lines.append(f"- {name} ({type_str}){details}{intent_str}{value_str}")
                else:
                    vars_lines.append(f"- {name} ({info})")
            prompt_parts.append("AVAILABLE VARIABLES:\n" + "\n".join(vars_lines))

            if dataframes and "df" not in dataframes:
                prompt_parts.append(
                    f"⚠️ CRITICAL: Use '{dataframes[0]}' NOT 'df'."
                )
            prompt_parts.append(
                "⚠️ Data is NOT preloaded as table-name globals. "
                "Only variables listed in AVAILABLE VARIABLES exist."
            )
            prompt_parts.append("")

        # File schemas (EXACT column names — highest priority)
        file_schemas = execution_context.get("file_schemas", {})
        if file_schemas:
            prompt_parts.append("FILE SCHEMAS (EXACT column names — use ONLY these):")
            for fname, schema in file_schemas.items():
                prompt_parts.append(f"  {fname}: columns={schema.get('columns', [])}")
            prompt_parts.append("⚠️ Use ONLY these exact column names. Do NOT use names from the plan.")
            prompt_parts.append("")

        # Dataset file paths — CRITICAL: always use absolute paths when loading files
        loaded_datasets = execution_context.get("loaded_datasets", [])
        if self._is_live_db:
            prompt_parts.append("DATA ACCESS (LIVE DATABASE):")
            prompt_parts.append("⚠️ CRITICAL: You are connected to a live SQL database.")
            prompt_parts.append("DO NOT use pd.read_parquet() or pd.read_csv().")
            prompt_parts.append("You MUST use the pre-defined python function `read_sql_query(query_string)` to fetch data.")
            prompt_parts.append("Example: df = read_sql_query('SELECT TOP 100 * FROM [table]')")
            prompt_parts.append("")
        elif loaded_datasets:
            prompt_parts.append(
                "LOADED DATASETS (CRITICAL: if you reload a file, use these EXACT absolute paths — "
                "NEVER use a bare filename like 'data.parquet', it WILL raise FileNotFoundError):"
            )
            for ds in loaded_datasets:
                p = ds.get("path", "?")
                v = ds.get("variable", "?")
                fmt = ds.get("format", "?")
                prompt_parts.append(f"  - path='{p}'  variable={v}  format={fmt}")
            prompt_parts.append("")

        # Dataset profile
        data_profile = execution_context.get("data_profile", {})
        if data_profile:
            prompt_parts.append("DATASET PROFILE:")
            for ds_name, prof in data_profile.items():
                prompt_parts.append(f"  {ds_name}: columns={prof.get('columns', [])}, shape={prof.get('shape', [])}")
            prompt_parts.append("")

        # Error context — show MORE of the error for column-not-found errors
        error_type, repair_hint, _ = self._classify_error(error)
        prompt_parts.append(f"FAILED CODE (attempt {attempt}):")
        prompt_parts.append(failed_code[:800])
        prompt_parts.append("")
        # ArrowInvalid errors contain the correct schema in the traceback — show it all
        error_limit = 2000 if error_type == "COLUMN_NOT_FOUND" else 800
        prompt_parts.append(f"ERROR [{error_type}]:")
        prompt_parts.append(error[:error_limit])
        prompt_parts.append("")
        prompt_parts.append(f"REQUIRED FIX: {repair_hint}")
        prompt_parts.append(
            "Identify the SPECIFIC error and fix it. "
            "If the approach was correct but a variable/column name was wrong, fix ONLY the name. "
            "Do NOT reload data that's already in memory. Do NOT start over from scratch. "
            "Use the EXACT variable names from AVAILABLE VARIABLES above."
        )

        # Column reload hint: when KeyError for a column that exists in file schema
        if error_type == "MISSING_COLUMN" and file_schemas:
            import re as _re
            key_match = _re.search(r"KeyError:\s*['\"]([^'\"]+)['\"]", error)
            if key_match:
                missing_col = key_match.group(1)
                for fname, schema in file_schemas.items():
                    if missing_col in schema.get("columns", []):
                        ds_path = schema.get("path", fname)
                        prompt_parts.append(
                            f"\n⚠️ RECOVERY HINT: Column '{missing_col}' EXISTS in "
                            f"{fname} but was not loaded.\n"
                            f"RELOAD the file with the missing column included:\n"
                            f"  df = pd.read_parquet('{ds_path}', "
                            f"columns=[...existing..., '{missing_col}'])\n"
                            f"Or load ALL columns for small tables: "
                            f"pd.read_parquet('{ds_path}')"
                        )
                        break

        # Inject relevant lessons — use cached version if available (tiered prompts),
        # otherwise fall back to DB fetch for legacy path
        lessons_text = self._cached_lessons_text
        if not lessons_text:
            try:
                raw_db = self._get_raw_db()
                if raw_db:
                    from services.agent_lesson_service import AgentLessonService
                    lesson_svc = AgentLessonService(raw_db)
                    schema_tables = [Path(f).stem for f in file_schemas.keys()] if file_schemas else None
                    lessons_text = await lesson_svc.format_lessons_for_prompt(
                        self.client_id, tables=schema_tables, max_tokens=800,
                    )
            except Exception:
                pass
        if lessons_text:
            prompt_parts.append("LEARNED PATTERNS (from prior analyses — follow these):")
            prompt_parts.append(lessons_text)

        prompt_parts.append("")
        prompt_parts.append("Return ONLY corrected Python code. No explanations, no markdown fences.")

        full_prompt = "\n".join(prompt_parts)

        # Progressive temperature for retries from config
        temp = self.retry_temperatures[
            min(attempt, len(self.retry_temperatures) - 1)
        ]

        code = await self._generate_code_for_step(full_prompt, temperature_override=temp)
        return code

    async def _initialize_kernel(self) -> None:
        """Initialize Jupyter server + MCP bridge + activate a notebook-backed kernel."""
        try:
            # Step 1: Start local Jupyter server for this session
            self.kernel_manager = await get_kernel_manager(
                client_id=self.client_id,
                idle_timeout_minutes=self.idle_timeout_minutes,
                use_docker=False,  # Local subprocess — no Docker overhead or file copying
                environment=self.db_credentials_env if self._is_live_db else None,
            )

            success = await self.kernel_manager.start()
            if not success:
                raise RuntimeError("Failed to start local Jupyter kernel")

            # Step 2: Connect MCP client via stdio transport
            kernel_url = self.kernel_manager.get_connection_url()
            logger.info(f"Connecting MCP to local Jupyter kernel at {kernel_url}")
            
            # MCP_SERVER_COMMAND is "python -m jupyter_mcp_server"
            # Running as a module from the current env ensures the patched
            # copy (le=300) is used, not the uvx-cached unpatched copy.
            base_cmd = MCP_SERVER_COMMAND.split()

            # Arguments for the jupyter-mcp-server
            server_args = [
                "--jupyter-url", kernel_url,
                "--jupyter-token", "",
                "--jupyterlab", "false",
            ]

            full_args = base_cmd[1:] + server_args

            server_params = StdioServerParameters(
                command=base_cmd[0],  # python interpreter path
                args=full_args
            )
            
            # Properly enter the stdio_client context
            self._stdio_context_manager = stdio_client(server_params)
            read_stream, write_stream = await self._stdio_context_manager.__aenter__()
            
            # Step 3: Create McpClient from the streams
            self._mcp_context_manager = McpClient(read_stream, write_stream)
            self.mcp_client = await self._mcp_context_manager.__aenter__()
            
            logger.info("MCP kernel and client initialized successfully")

            # Step 4: Activate a notebook-backed kernel.
            # IMPORTANT: jupyter-mcp-server's `execute_code` tool is not intended
            # for stateful assignments. We use notebook tools so variables persist.
            await self._activate_or_reuse_session_notebook()
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP kernel: {e}")
            raise RuntimeError(f"MCP initialization failed: {e}") from e

    def _session_notebook_identity(self) -> tuple[str, str]:
        """Deterministic notebook name/path per session for reuse."""
        sid = (self.session_id or "").strip() or "no_session"
        # Keep names short and safe for tool routing
        safe_sid = re.sub(r"[^a-zA-Z0-9_-]+", "_", sid)[:64]
        safe_cid = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(self.client_id))[:64]
        nb_name = f"coresight_{safe_cid}_{safe_sid}"
        nb_path = f"coresight_sessions/{safe_cid}/{safe_sid}.ipynb"
        return nb_name, nb_path

    async def _activate_or_reuse_session_notebook(self) -> None:
        """
        Use `use_notebook` to ensure subsequent executions are stateful.
        Reuses prior notebook/kernel for dependent follow-ups when available.
        """
        if not self.mcp_client:
            raise RuntimeError("MCP client not initialized")

        import os
        nb_name, nb_path = self._session_notebook_identity()
        self._mcp_notebook_name = nb_name
        self._mcp_notebook_path = nb_path
        
        # Ensure parent directory exists (filesystem + Jupyter Contents API)
        parent_dir = os.path.dirname(nb_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        if self.kernel_manager:
            try:
                ensure_jupyter_contents_dirs(
                    self.kernel_manager.get_connection_url(),
                    nb_path,
                )
            except Exception as exc:
                logger.warning("Could not register notebook path on Jupyter server: %s", exc)

        # Reuse only when the orchestrator/router has flagged this session as follow-up.
        follow_ups = 0
        try:
            if self.session_id:
                follow_ups = session_memory.get_follow_up_count(self.session_id)
        except Exception:
            follow_ups = 0

        kernel_ctx = None
        try:
            if self.session_id:
                kernel_ctx = session_memory.get_kernel_context(self.session_id)
        except Exception:
            kernel_ctx = None

        mode = "create"
        kernel_id = None
        self._session_kernel_reused = False

        if follow_ups > 0 and kernel_ctx:
            # Strict scope validation to avoid cross-tenant reuse
            ctx_client = str(kernel_ctx.get("client_id", ""))
            ctx_user = str(kernel_ctx.get("user_id", ""))
            ctx_ds = kernel_ctx.get("datasource_context") or {}
            if (ctx_client == str(self.client_id) and ctx_user == str(self.user_id) and ctx_ds == (self.datasource_context or {})):
                mode = "connect"
                kernel_id = kernel_ctx.get("kernel_id")
                self._session_kernel_reused = True
                logger.info(
                    "Session notebook reuse enabled: session=%s follow_ups=%d kernel_id=%s",
                    self.session_id, follow_ups, (kernel_id or "")[:12],
                )
            else:
                logger.warning(
                    "Session notebook context scope mismatch; creating new notebook. "
                    "session=%s follow_ups=%d",
                    self.session_id, follow_ups,
                )

        args: Dict[str, Any] = {
            "notebook_name": nb_name,
            "notebook_path": nb_path,
            "mode": mode,
        }
        if kernel_id:
            args["kernel_id"] = kernel_id

        # Activate notebook (and thus a kernel)
        mcp_result = await self.mcp_client.call_tool(
            name="use_notebook",
            arguments=args,
            timeout_seconds=max(60, int(self.timeout_per_execution)),
        )
        
        # Check if the tool returned an error string instead of successfully activating
        if mcp_result and getattr(mcp_result, "content", None):
            result_text = "".join(c.text for c in mcp_result.content if c.type == "text")
            # If there's an error message that looks like failure, abort early
            if "not found in jupyter server" in result_text or "Failed to connect" in result_text or "Invalid configuration" in result_text:
                logger.error(f"use_notebook failed: {result_text}")
                raise RuntimeError(f"use_notebook failed: {result_text}")

        # Resolve the active kernel_id for this notebook for future reuse.
        # list_notebooks returns TSV; we parse by notebook name.
        resolved_kernel_id = await self._resolve_kernel_id_from_list_notebooks(nb_name)
        if resolved_kernel_id:
            self._mcp_kernel_id = resolved_kernel_id
            if self.session_id and self.user_id:
                try:
                    session_memory.set_kernel_context(
                        self.session_id,
                        client_id=str(self.client_id),
                        user_id=str(self.user_id),
                        datasource_context=self.datasource_context,
                        notebook_name=nb_name,
                        notebook_path=nb_path,
                        kernel_id=resolved_kernel_id,
                        ttl_seconds=self.session_kernel_ttl_seconds,
                    )
                except Exception:
                    pass

    async def _resolve_kernel_id_from_list_notebooks(self, notebook_name: str) -> Optional[str]:
        if not self.mcp_client:
            return None
        try:
            tsv = await self.mcp_client.call_tool(
                name="list_notebooks",
                arguments={},
                timeout_seconds=30,
            )
            if not isinstance(tsv, str):
                tsv = str(tsv)
            lines = [ln for ln in tsv.splitlines() if ln.strip()]
            if len(lines) < 2:
                return None
            header = [h.strip() for h in lines[0].split("\t")]
            idx_name = header.index("Name") if "Name" in header else 0
            idx_kid = header.index("Kernel_ID") if "Kernel_ID" in header else 2
            for ln in lines[1:]:
                cols = ln.split("\t")
                if len(cols) <= max(idx_name, idx_kid):
                    continue
                if cols[idx_name].strip() == notebook_name:
                    kid = cols[idx_kid].strip()
                    return kid or None
        except Exception:
            return None
        return None



    async def _load_dataset_to_kernel(
        self,
        dataset_path: Optional[str],
        dataset_dict: Optional[Dict]
    ) -> List[Dict[str, str]]:
        """Load dataset into kernel memory.
        
        Returns:
            List of dicts with 'path', 'variable', and 'format' for each loaded file.
            This info is passed into prompts so the LLM knows the correct paths/format.
        """
        code = ""
        loaded_datasets: List[Dict[str, str]] = []
        
        if dataset_path:
            # Explicit dataset path provided — expose it to the LLM; it generates the load code.
            # copy_file_to_container is a no-op for LocalKernelManager (host path works directly).
            p = Path(dataset_path)
            if self.kernel_manager:
                self.kernel_manager.copy_file_to_container(dataset_path, dataset_path)

            fmt = 'parquet' if p.suffix.lower() == '.parquet' else 'csv'
            loaded_datasets.append({'path': str(p), 'variable': 'df', 'format': fmt})
            # No pre-loading code — LLM generates step 1 load code with sampling
            
        elif dataset_dict is not None:
            # Load from dictionary/DataFrame
            import pandas as pd
            if isinstance(dataset_dict, pd.DataFrame):
                df_str = dataset_dict.to_json(orient='split')
            else:
                df_str = json.dumps(dataset_dict)
            
            code = f"""
import pandas as pd
import json
df = pd.read_json(json.loads({repr(df_str)}), orient='split')
print(f"Loaded dataset with shape: {{df.shape}}")
print(f"Columns: {{df.columns.tolist()}}")
"""
            loaded_datasets.append({'path': '<in-memory>', 'variable': 'df', 'format': 'dict'})
        elif self._is_live_db:
            # Tell the LLM that the database is connected
            loaded_datasets.append({
                'path': '<live_database>', 
                'variable': 'pd.DataFrame', 
                'format': f"SQL Database ({self.db_credentials_env.get('CS_DB_TYPE', 'unknown')})"
            })
            return loaded_datasets
        else:
            # Auto-discover client datasets.
            # IMPORTANT: We only expose file paths — we do NOT pre-load data.
            # Pre-loading causes 60s timeouts on large parquet files.
            # The LLM generates the load code in step 1, with column selection + sampling.
            try:
                client_data_dir = self._get_client_dataset_dir()

                if client_data_dir.exists():
                    all_files = (
                        list(client_data_dir.glob("*.parquet")) +
                        list(client_data_dir.glob("*.csv"))
                    )
                    if all_files:
                        for f in all_files:
                            fmt = 'parquet' if f.suffix.lower() == '.parquet' else 'csv'
                            safe_name = (
                                f.stem.replace(' ', '_').replace('-', '_').replace('.', '_')
                            )
                            loaded_datasets.append({
                                'path': str(f),    # actual host path — readable directly
                                'variable': safe_name,
                                'format': fmt,
                            })
                            logger.info(f"Staged dataset for LLM: {f.name} → {f}")
                    else:
                        logger.warning(f"No datasets found in {client_data_dir}")
                else:
                    logger.warning(f"Client dataset directory not found: {client_data_dir}")
            except Exception as e:
                logger.error(f"Error discovering client datasets: {e}")

            # No code executed — LLM generates load code in step 1
            return loaded_datasets

        if code:
            try:
                result = await self._execute_code(code)
                stdout = result.get("stdout", "")
                if "TIMEOUT ERROR" in stdout or "execution exceeded" in stdout.lower():
                    logger.error(f"Dataset loading TIMED OUT: {stdout[:200]}")
                elif result.get("exception"):
                    logger.error(f"Dataset loading failed: {result['exception']}")
                    logger.error(f"stderr: {result.get('stderr', '')}")
                else:
                    logger.info(f"Dataset loaded. Output: {stdout[:200]}")
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                raise

        return loaded_datasets

    def _build_step_code_prompt(
        self,
        step: Dict[str, Any],
        user_query: str,
        execution_context: Dict,
        attempt: int = 0,
        last_error: Optional[str] = None
    ) -> str:
        """
        Build a prompt for generating code for a specific step.
        
        Args:
            step: Step dictionary with step_num, description, details
            user_query: Original user query
            execution_context: Current execution state
            attempt: Which attempt this is (0-indexed)
            last_error: Error from previous attempt, if any
        
        Returns:
            Formatted prompt string for LLM
        """
        step_num = step["step_num"]
        description = step["description"]
        details = step.get("details", [])
        completed_steps = execution_context.get("completed_iterations", execution_context.get("completed_steps", []))
        available_vars = execution_context.get("available_variables", ["df"])
        
        # Include dataset path/format info so LLM won't guess wrong paths
        loaded_datasets = execution_context.get("loaded_datasets", [])
        
        prompt_parts = [
            f"USER QUERY: {user_query}",
            "",
        ]
        
        # Tell the LLM about loaded datasets (paths, formats, variable names)
        # Fix: If no datasets are loaded yet (Step 1), we must find them and tell the LLM
        if not loaded_datasets and step_num == 1:
            # Try to find what datasets *should* be loaded based on the plan or client context
            # We'll re-use the logic from _load_dataset_to_kernel partially here to get the path
            try:
                client_data_dir = self._get_client_dataset_dir()
                if client_data_dir.exists():
                     parquet_files = list(client_data_dir.glob("*.parquet"))
                     if parquet_files:
                         main_file = parquet_files[0]
                         container_path = f"/app/{main_file.name}"
                         loaded_datasets = [{
                             'variable': 'df',
                             'format': 'parquet',
                             'path': container_path
                         }]
            except Exception:
                pass

        if loaded_datasets:
            prompt_parts.append("LOADED DATASETS (already loaded in kernel — do NOT reload unless needed):")
            for ds in loaded_datasets:
                prompt_parts.append(f"  - Variable '{ds['variable']}' = {ds['format']} file at: {ds['path']}")
            prompt_parts.append("")
            
            if self._is_live_db:
                prompt_parts.append("IMPORTANT: You are connected to a LIVE SQL DATABASE.")
                prompt_parts.append("DO NOT use pd.read_parquet() or pd.read_csv().")
                prompt_parts.append("You MUST use the pre-defined helper function `read_sql_query(query: str) -> pd.DataFrame` to fetch data.")
                prompt_parts.append("Example: df = read_sql_query('SELECT TOP 100 * FROM [table_name]')")
                prompt_parts.append("")
            else:
                prompt_parts.append("IMPORTANT: Data files are parquet format. NEVER use pd.read_csv().")
                prompt_parts.append("If you must reload data, use: pd.read_parquet(r'<exact path shown above>')")
                prompt_parts.append("")
        elif step_num == 1 and not available_vars:
            # Fallback if no datasets loaded but we are in step 1
            if self._is_live_db:
                prompt_parts.append("IMPORTANT: You are in Step 1 and connected to a LIVE SQL DATABASE.")
                prompt_parts.append("You MUST start your code by fetching the initial dataset using the pre-defined helper function `read_sql_query`.")
                prompt_parts.append("Example: df = read_sql_query('SELECT TOP 100 * FROM [table_name]')")
                prompt_parts.append("")
            else:
                # Try to find a file to suggest
                suggested_file = "<filename>.parquet"
                try:
                    client_data_dir = self._get_client_dataset_dir()
                    if client_data_dir.exists():
                         parquet_files = list(client_data_dir.glob("*.parquet"))
                         if parquet_files:
                             suggested_file = parquet_files[0].name
                except Exception:
                    pass

                prompt_parts.append("IMPORTANT: You are in Step 1 and no data is loaded yet.")
                prompt_parts.append("You have NO variables defined.")
                prompt_parts.append(f"You MUST start your code by loading the dataset:")
                prompt_parts.append(f"df = pd.read_parquet(r'/app/{suggested_file}')")
                prompt_parts.append("")
        
        prompt_parts.append(f"CURRENT STEP ({step_num}): {description}")
        
        if details:
            prompt_parts.append("Details:")
            for detail in details:
                prompt_parts.append(f"  - {detail}")
            prompt_parts.append("")
        
        # Add context from previous steps — include actual code so LLM doesn't regenerate it
        if completed_steps:
            prompt_parts.append("COMPLETED STEPS (already executed — do NOT repeat this code):")
            for prev_step in completed_steps:
                prompt_parts.append(f"  {prev_step['iteration']}. {prev_step['reasoning']}")

                # Show actual code from recent steps so LLM sees what already ran
                if prev_step.get('code'):
                    code_preview = prev_step['code']
                    if len(code_preview) > 600:
                        code_preview = code_preview[:600] + "\n# ... (truncated)"
                    prompt_parts.append(f"     Code executed:\n{code_preview}")

                vars_info = prev_step.get('variables')
                if vars_info:
                    if isinstance(vars_info, list):
                        v_str = ", ".join(vars_info)
                    elif isinstance(vars_info, dict):
                        v_str = ", ".join(vars_info.keys())
                    else:
                        v_str = str(vars_info)
                    prompt_parts.append(f"     Variables created: {v_str}")

                if prev_step.get('output'):
                    condensed = prev_step['output'][:500]
                    prompt_parts.append(f"     Output: {condensed}")
            prompt_parts.append("")

        # P8: Warnings from previous steps (silent failures, empty DFs, etc.)
        warnings = execution_context.get("warnings", [])
        if warnings:
            prompt_parts.append("⚠️ WARNINGS FROM PREVIOUS STEPS:")
            for w in warnings[-3:]:
                prompt_parts.append(f"  - {w}")
            prompt_parts.append("")

        # P3: Dataset profile — exact column names, dtypes, null structure
        # Prevents ~60% of NameError/KeyError/TypeError failures
        data_profile = execution_context.get("data_profile", {})
        if data_profile:
            prompt_parts.append("DATASET PROFILE (exact column names, dtypes, null structure):")
            for df_name, prof in data_profile.items():
                prompt_parts.append(f"  {df_name}: shape={prof.get('shape', '?')}")
                if prof.get("columns"):
                    prompt_parts.append(f"    columns = {prof['columns']}")
                if prof.get("dtypes"):
                    prompt_parts.append(f"    dtypes  = {prof['dtypes']}")
                if prof.get("null_counts"):
                    prompt_parts.append(f"    nulls   = {prof['null_counts']}")
                if prof.get("sample_row"):
                    prompt_parts.append(
                        f"    sample  = {str(prof['sample_row'][0])[:200]}"
                    )
            prompt_parts.append("USE ONLY THESE EXACT COLUMN NAMES — case-sensitive, no variation.")
            prompt_parts.append("")
        
        # Format available variables nicely
        dataframes = []
        if isinstance(available_vars, list):
            # Fallback for old format
            vars_list = available_vars
            vars_formatted = ", ".join(vars_list)
            dataframes = [v for v in vars_list if 'df' in v or 'data' in v] # Guesswork
        elif isinstance(available_vars, dict):
            # New detailed format
            lines = []
            for name, info in available_vars.items():
                if isinstance(info, dict):
                    type_str = info.get("type", "Unknown")
                    if type_str == 'DataFrame':
                        dataframes.append(name)
                    details_str = ""
                    if "columns" in info:
                        cols = info["columns"]
                        # Truncate long column lists
                        if len(cols) > 10:
                            cols_str = str(cols[:10])[:-1] + ", ...]"
                        else:
                            cols_str = str(cols)
                        details_str = f" columns={cols_str}"
                    if "shape" in info:
                        details_str += f" shape={info['shape']}"
                    lines.append(f"- {name} ({type_str}){details_str}")
                    # RLM enhancement: include dtypes and sample rows for DataFrames
                    if "dtypes" in info:
                        dtype_str = str(info["dtypes"])[:200]
                        lines.append(f"  dtypes: {dtype_str}")
                    if "sample" in info:
                        sample_str = str(info["sample"])[:300]
                        lines.append(f"  sample rows: {sample_str}")
                    # Include scalar value preview for non-DataFrame variables
                    if "value" in info and type_str != 'DataFrame':
                        lines.append(f"  value: {info['value']}")
                else:
                    lines.append(f"- {name} ({info})")
            vars_formatted = "\n".join(lines)
        else:
            vars_formatted = str(available_vars)

        prompt_parts.append(f"AVAILABLE VARIABLES (Use ONLY these):\n{vars_formatted}")
        prompt_parts.append("")
        
        # SMART ALIASING CHECK
        if dataframes and 'df' not in dataframes:
            # If we have dataframes but none are named 'df', warn the user
            if len(dataframes) == 1:
                prompt_parts.append(f"⚠️  CRITICAL: The dataframe is named '{dataframes[0]}'.")
                prompt_parts.append(f"DO NOT use 'df'. Use '{dataframes[0]}' instead.")
                prompt_parts.append("")
            else:
                prompt_parts.append(f"⚠️  CRITICAL: Available dataframes are: {', '.join(dataframes)}.")
                prompt_parts.append(f"DO NOT use 'df' unless it is defined. Use the specific names listed above.")
                prompt_parts.append("")

        # P2: Typed error taxonomy — targeted repair hints instead of generic "fix this"
        if attempt > 0 and last_error:
            error_type, repair_hint, _ = self._classify_error(last_error)
            prompt_parts.append(f"PREVIOUS ATTEMPT FAILED [{error_type}]:")
            prompt_parts.append(f"Error: {last_error[:400]}")
            prompt_parts.append("")
            prompt_parts.append(f"REQUIRED FIX: {repair_hint}")
            prompt_parts.append(
            "Identify the SPECIFIC error and fix it. "
            "If the approach was correct but a variable/column name was wrong, fix ONLY the name. "
            "Do NOT reload data that's already in memory. Do NOT start over from scratch. "
            "Use the EXACT variable names from AVAILABLE VARIABLES above."
        )
            prompt_parts.append("")
        
        prompt_parts.extend([
            "INSTRUCTIONS:",
            f"Generate Python code ONLY for step {step_num}. Do NOT include code for other steps.",
            "",
            "CRITICAL — DO NOT REPEAT PRIOR WORK:",
            "- All code from COMPLETED STEPS has ALREADY been executed. Variables from those steps are alive in memory.",
            "- Do NOT re-import libraries that were imported in prior steps (they are already available).",
            "- Do NOT reload data that was loaded in prior steps (use the existing variable).",
            "- Do NOT recalculate values that already exist in AVAILABLE VARIABLES.",
            "- Write ONLY the NEW code needed for THIS step. Assume all prior state exists.",
            "",
            "CRITICAL VARIABLE RULES:",
            f"- Use ONLY the variables listed in AVAILABLE VARIABLES above.",
            "- Look at the column names provided for each DataFrame to find the data you need.",
            "- Do NOT invent new variable names (like 'forecast_df' or 'sales_data') if they don't exist.",
            "- If you need 'predicted revenue', look for a dataframe with a 'predicted_revenue' column.",
            "- Do NOT generate code that checks if variables exist and raises errors — just use the available variables directly.",
            "",
            "CODE RULES:",
            "- Build on existing variables from previous steps",
            "- Create clearly named new variables for this step's output",
            "- Add print statements to show results",
            "- Keep code focused on this single step",
            "",
            "MODEL SELECTION (MANDATORY for any forecasting/prediction/ML step):",
            "- Use ONLY: RandomForest or XGBoost (in that order of preference)",
            "- NEVER use LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, statsmodels OLS/GLM, or TensorFlow/Keras",
            "- Default to RandomForestRegressor/Classifier for most problems",
            "- Use xgboost.XGBRegressor/XGBClassifier for complex non-linear patterns or high cardinality features",
            "",
            "Return ONLY executable Python code, no explanations."
        ])
        
        return "\n".join(prompt_parts)

    async def _generate_code_for_step(
        self, prompt: str, temperature_override: Optional[float] = None
    ) -> str:
        """
        Generate Python code for a specific step using LLM.

        Args:
            prompt: The formatted prompt for this step
            temperature_override: If set, use this temperature instead of self.temperature.
                                  Used by progressive temperature (P7).

        Returns:
            Generated Python code
        """
        try:
            temp = temperature_override if temperature_override is not None else self.temperature

            # Use the XML-loaded system prompt so agent rules/constraints are applied.
            # Append essential code-generation instructions that complement the XML.
            system_prompt = (self.base_prompt or "") + """\n\nCODE GENERATION RULES (always apply):
- Return ONLY Python code — no markdown fences, no explanations, no apologies, no prose whatsoever
- If you cannot complete the task, still return Python code (e.g., a comment + print statement)
- PERFORMANCE CRITICAL — loading large files:
  * ALWAYS select only the columns you need: pd.read_parquet(path, columns=['col1','col2'])
  * Do NOT add .sample() — always process the FULL dataset for accurate results
  * Only sample if the user EXPLICITLY asks for it in their query (e.g. "use a 10% sample to save time")
- Use pandas, numpy, sklearn, plotly as needed
- CRITICAL: Only use variables that are listed in AVAILABLE VARIABLES in the prompt
- Do NOT assume variable names — use exactly what is listed as available
- Do NOT generate variable-existence checks that raise errors
- Add print statements to show key results
- Code must be ready to execute in a Jupyter cell with NO surrounding text
- FINAL RESULT RULE: In the LAST step of the analysis, store your primary output in a variable
  named FINAL_RESULT (e.g. FINAL_RESULT = result_df  or  FINAL_RESULT = {"count": 123, ...}).
  This variable is used as the authoritative final output of the analysis."""

            # Call LLM with correct method name and parameters
            response_dict = await self.llm_client.generate_completion(
                system_prompt=system_prompt,
                user_message=prompt,
                temperature=temp
            )
            
            # Update usage stats
            self._update_usage(response_dict.get("usage"))

            # Check for API-level errors (e.g. Gemini timeout, safety block, empty response)
            llm_error = response_dict.get("error")
            if llm_error and not response_dict.get("content"):
                raise ValueError(f"LLM API error: {llm_error}")

            # Extract code — use `or ""` to safely handle None content
            code = (response_dict.get("content") or "").strip()

            # Remove markdown code fences if present
            if code.startswith("```python"):
                code = code[len("```python"):].strip()
            if code.startswith("```"):
                code = code[3:].strip()
            if code.endswith("```"):
                code = code[:-3].strip()

            # Reject prose responses immediately — don't waste 60s timing out the kernel
            if self._is_likely_prose(code):
                logger.warning(f"LLM returned prose instead of Python code: {code[:150]!r}")
                raise ValueError(
                    f"LLM returned explanatory text instead of Python code. "
                    f"You MUST return ONLY executable Python. First 100 chars: {code[:100]!r}"
                )

            logger.debug(f"Generated code:\n{code}")
            return code

        except ValueError:
            raise  # Re-raise so the retry loop handles it with proper error context
        except Exception as e:
            logger.error(f"Failed to generate code: {e}")
            raise ValueError(f"Code generation failed: {e}")

    async def _generate_diagnostic_code(
        self,
        failed_code: str,
        error: str,
        available_vars: Dict[str, Any]
    ) -> Optional[str]:
        """
        Return targeted diagnostic code based on error type — instant, no LLM call.
        Uses _classify_error() to pick the right inspection code for the error type.
        """
        _, _, diagnostic_code = self._classify_error(error)
        return diagnostic_code

    async def _summarize_completed_steps(self, steps: List[Dict]) -> str:
        """
        Produce a compact 2-sentence summary of a batch of completed steps.
        Used for context compaction every 3 steps to prevent prompt bloat.
        """
        try:
            steps_text = "\n".join(
                f"Step {s['iteration']}: {s['reasoning']} | Output: {str(s.get('output',''))[:100]}"
                for s in steps
            )
            prompt = (
                f"Summarize what was accomplished in these data analysis steps in 2 sentences max. "
                f"Focus on what data was loaded, transformed, or computed, and any key variables created.\n\n"
                f"{steps_text}"
            )
            response = await self.llm_client.generate_completion(
                system_prompt="You are a concise technical summarizer. Reply with 1-2 sentences only.",
                user_message=prompt,
                temperature=0.0
            )
            return (response.get("content") or "").strip() or f"Completed steps {[s['iteration'] for s in steps]}."
        except Exception as e:
            logger.warning(f"Context compaction failed: {e}")
            return f"Completed steps {[s['iteration'] for s in steps]}."

    # -------------------------------------------------------------------------
    # P2: Typed Error Taxonomy
    # -------------------------------------------------------------------------

    def _classify_error(self, error_str: str) -> Tuple[str, str, str]:
        """
        Classify an error into a known type and return a targeted repair hint.

        Returns:
            Tuple of (error_type, repair_hint, diagnostic_code).
            diagnostic_code is pre-defined Python to run in the kernel for instant
            state inspection — no LLM call needed.
        """
        e = error_str.lower()

        if any(x in e for x in ["nameerror", "is not defined"]):
            return (
                "UNDEFINED_VARIABLE",
                "The variable doesn't exist. Check AVAILABLE VARIABLES — use the EXACT name listed.",
                "print('Available vars:', [k for k in globals() if not k.startswith('_')][:20])"
            )

        if any(x in e for x in ["keyerror", "not in index"]):
            return (
                "MISSING_COLUMN",
                "Column name is wrong. Use EXACT column name from DATASET PROFILE (case-sensitive).",
                "for _v,_o in globals().items():\n"
                "    if hasattr(_o,'columns'): print(f'{_v} columns:',_o.columns.tolist())"
            )

        # Check type-conversion ArrowInvalid BEFORE column-not-found ArrowInvalid
        # "could not convert" indicates a dtype mismatch, not a missing column
        if any(x in e for x in ["arrowtypeerror", "could not convert"]):
            return (
                "ARROW_TYPE_ERROR",
                "PyArrow mixed-type error. Use pd.to_numeric(col, errors='coerce') or .astype(str) "
                "to normalize dtypes before operations.",
                "for _v,_o in globals().items():\n"
                "    if hasattr(_o,'dtypes'): print(f'{_v} dtypes:',_o.dtypes.to_dict())"
            )

        if any(x in e for x in ["arrowinvalid", "no match for fieldref"]):
            return (
                "COLUMN_NOT_FOUND",
                "Column name does not exist in the file. The error message shows the ACTUAL "
                "columns available — read them carefully and use THOSE exact names. "
                "Check FILE SCHEMAS for the correct column names. Do NOT guess or use plan column names.",
                self._build_parquet_schema_diagnostic(error_str)
            )

        if "pyarrow" in e:
            return (
                "ARROW_TYPE_ERROR",
                "PyArrow error. Use pd.to_numeric(col, errors='coerce') or .astype(str) "
                "to normalize dtypes before operations.",
                "for _v,_o in globals().items():\n"
                "    if hasattr(_o,'dtypes'): print(f'{_v} dtypes:',_o.dtypes.to_dict())"
            )

        if any(x in e for x in ["typeerror", "unsupported operand", "cannot convert"]):
            return (
                "TYPE_MISMATCH",
                "Type mismatch. Check dtypes and add explicit .astype() before the operation.",
                "for _v,_o in globals().items():\n"
                "    if not callable(_o) and not _v.startswith('_'):\n"
                "        print(f'{_v}: {type(_o).__name__}')"
            )

        if any(x in e for x in ["memoryerror", "cannot allocate"]):
            return (
                "MEMORY_ERROR",
                "Dataset too large. Select fewer columns with columns= parameter, or split the operation into smaller chunks.",
                "for _v,_o in globals().items():\n"
                "    if hasattr(_o,'shape'): print(f'{_v}.shape:',_o.shape)"
            )

        if any(x in e for x in ["timed out", "timeout"]):
            return (
                "TIMEOUT",
                "Operation too slow. Use vectorized operations, not loops. Sample the data first.",
                "for _v,_o in globals().items():\n"
                "    if hasattr(_o,'shape'): print(f'{_v}.shape:',_o.shape)"
            )

        if any(x in e for x in ["modulenotfounderror", "importerror"]):
            return (
                "IMPORT_ERROR",
                "Only available: pandas, numpy, sklearn, scipy, plotly, matplotlib, re, json, "
                "datetime, collections. Do not import other packages.",
                "import pkg_resources; print([p.project_name for p in pkg_resources.working_set][:20])"
            )

        if any(x in e for x in ["filenotfounderror", "no such file"]):
            return (
                "FILE_NOT_FOUND",
                "Use ONLY the exact file path from LOADED DATASETS. Copy it character-for-character.",
                "import os; print(os.listdir('.')[:20])"
            )

        if "zero_row_result" in e:
            return (
                "ZERO_ROW_RESULT",
                "A filter or join produced 0 rows. The filter column or values are WRONG. "
                "Check the DIAGNOSTIC OUTPUT for actual unique values in candidate columns. "
                "Try a DIFFERENT column (e.g., INV_ORG_ID instead of ORGANIZATION_ID). "
                "Print unique values in both tables to find the correct mapping. "
                "Do NOT repeat the same filter — use fundamentally different columns.",
                "import pandas as _pd_\n"
                "for _v,_o in globals().items():\n"
                "    if _v.startswith('_'): continue\n"
                "    if isinstance(_o, _pd_.DataFrame) and _o.shape[0] > 0:\n"
                "        _ids = [c for c in _o.columns if any("
                "c.upper().endswith(s) for s in ('_ID','_CODE','_KEY','_NAME'))]\n"
                "        if _ids: print(f'{_v}: {[(c, _o[c].nunique()) for c in _ids[:5]]}')"
            )

        return (
            "GENERIC_ERROR",
            "Fix the specific error shown below. Read carefully and correct that exact line.",
            "print('Kernel vars:', [k for k in globals() if not k.startswith('_')][:20])"
        )

    # -------------------------------------------------------------------------
    # P4: Pre-execution AST Syntax Validation
    # -------------------------------------------------------------------------

    def _validate_code_syntax(self, code: str, available_vars: Dict[str, Any]) -> Optional[str]:
        """
        Fast pre-execution check using Python's ast module.
        Catches SyntaxErrors and obvious undefined variable references in microseconds.

        Returns:
            Error string if a problem is found, None if code looks OK.
        """
        import ast as _ast

        # 1. Syntax check
        try:
            tree = _ast.parse(code)
        except SyntaxError as e:
            return f"SyntaxError line {e.lineno}: {e.msg}"

        # 2. Obvious prose check (LLM returned explanation instead of code)
        first_line = code.strip().split('\n')[0].lower()
        if any(first_line.startswith(p) for p in
               ["i cannot", "i'm sorry", "unfortunately", "to solve", "here is the", "as a"]):
            return f"LLM returned prose instead of code: {code[:80]!r}"

        # 3. Undefined dataframe variable check (best-effort, not exhaustive)
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
            safe |= {
                "pd", "np", "plt", "px", "json", "os", "re", "datetime", "print",
                "len", "range", "list", "dict", "str", "int", "float", "type",
                "True", "False", "None", "min", "max", "sum", "zip", "enumerate",
                "llm_query", "requests", "sklearn", "scipy", "math", "collections",
                "FINAL_RESULT"
            }

            suspicious_df = {
                n for n in (loaded_names - safe)
                if n == "df" or n.startswith(("df_", "data_", "result_", "filtered_"))
            }
            if suspicious_df:
                return (
                    f"Undefined variable(s): {suspicious_df}. "
                    f"Available: {list(available_vars.keys())[:8]}"
                )

        # 4. Banned model check — reject LinearRegression, LogisticRegression, TensorFlow, etc.
        banned_models = {
            "LinearRegression", "LogisticRegression",
            "Ridge", "Lasso", "ElasticNet",
        }
        banned_modules = {"tensorflow", "keras", "tf"}
        for node in _ast.walk(tree):
            # Catch: from sklearn.linear_model import LinearRegression
            if isinstance(node, _ast.ImportFrom):
                module = node.module or ""
                if any(module == m or module.startswith(f"{m}.") for m in banned_modules):
                    return (
                        f"Banned module '{module}' detected. "
                        f"Use RandomForestRegressor/Classifier or xgboost instead."
                    )
                for alias in node.names:
                    name = alias.name
                    if name in banned_models:
                        return (
                            f"Banned model '{name}' detected. "
                            f"Use RandomForestRegressor/Classifier or xgboost instead."
                        )
            # Catch: import tensorflow
            if isinstance(node, _ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name in banned_modules or any(name.startswith(f"{m}.") for m in banned_modules):
                        return (
                            f"Banned module '{name}' detected. "
                            f"Use RandomForestRegressor/Classifier or xgboost instead."
                        )
            # Catch: sklearn.linear_model.LinearRegression() or LinearRegression()
            if isinstance(node, _ast.Call):
                func = node.func
                func_name = None
                if isinstance(func, _ast.Name):
                    func_name = func.id
                elif isinstance(func, _ast.Attribute):
                    func_name = func.attr
                if func_name and func_name in banned_models:
                    return (
                        f"Banned model '{func_name}' detected. "
                        f"Use RandomForestRegressor/Classifier or xgboost instead."
                    )

        return None

    # -------------------------------------------------------------------------
    # Doom loop detection (shared with DataAnalystAgent)
    # -------------------------------------------------------------------------

    def _detect_doom_loop(self, current_code: str) -> bool:
        """
        Return True if current_code is nearly identical to the last
        `doom_loop_threshold` failed codes (OpenCode-inspired pattern).

        Similarity threshold: >= 0.92 (Ratcliff-Obershelp via difflib).
        This prevents the LLM from burning tokens generating the same
        broken code repeatedly across iterations.
        """
        if len(self._recent_failed_codes) < self.doom_loop_threshold:
            return False

        last_n = self._recent_failed_codes[-self.doom_loop_threshold:]
        for prev in last_n:
            ratio = difflib.SequenceMatcher(None, current_code.strip(), prev.strip()).ratio()
            if ratio < 0.92:
                return False  # At least one failed code was different — not a doom loop

        return True  # All N recent failures are nearly identical — doom loop

    # -------------------------------------------------------------------------
    # P3: One-time Dataset Profile Probe
    # -------------------------------------------------------------------------

    async def _probe_dataset_profile(self) -> Dict[str, Any]:
        """
        Run a lightweight profiling probe across all DataFrames currently in the kernel.
        Called once after dataset load. Result is stored in execution_context["data_profile"]
        and injected into every step prompt so the LLM always has exact column names/dtypes.
        """
        code = """
import json as _j_, pandas as _p_
_profile_ = {}
for _name_ in list(globals().keys()):
    if _name_.startswith('_'): continue
    try:
        _obj_ = eval(_name_)
        if not isinstance(_obj_, _p_.DataFrame): continue
        _pr_ = {
            "shape": list(_obj_.shape),
            "columns": _obj_.columns.tolist(),
            "dtypes": {c: str(d) for c, d in _obj_.dtypes.items()},
            "null_counts": {c: int(n) for c, n in _obj_.isnull().sum().items() if n > 0},
            "sample_row": _obj_.head(1).to_dict(orient='records'),
        }
        _num_cols_ = _obj_.select_dtypes(include='number').columns.tolist()[:4]
        if _num_cols_:
            _pr_["numeric_ranges"] = {
                c: {"min": float(_obj_[c].min()), "max": float(_obj_[c].max())}
                for c in _num_cols_
            }
        _str_cols_ = [c for c in _obj_.columns if _obj_[c].dtype == 'object'][:8]
        if _str_cols_:
            _sv_ = {}
            for c in _str_cols_:
                _vc_ = _obj_[c].dropna().value_counts().head(15)
                _sv_[c] = {
                    "unique_count": int(_obj_[c].nunique()),
                    "top_values": [str(v) for v in _vc_.index.tolist()],
                    "top_counts": [int(ct) for ct in _vc_.values.tolist()],
                }
            _pr_["string_values"] = _sv_
        _profile_[_name_] = _pr_
    except Exception:
        pass
print('<PROFILE>' + _j_.dumps(_profile_) + '</PROFILE>')
"""
        result = await self._execute_code(code)
        stdout = result.get("stdout", "")
        if "<PROFILE>" in stdout and "</PROFILE>" in stdout:
            try:
                start = stdout.find("<PROFILE>") + len("<PROFILE>")
                end = stdout.find("</PROFILE>")
                profile = json.loads(stdout[start:end])
                logger.info(f"Dataset profile captured: {list(profile.keys())}")
                return profile
            except Exception as e:
                logger.warning(f"Profile parsing failed: {e}")
        return {}

    async def _probe_parquet_schemas(
        self,
        loaded_datasets: List[Dict[str, str]],
        dataset_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Read parquet/CSV file schemas WITHOUT loading data.

        This is instant (~0ms per file) and uses zero memory because
        pyarrow.parquet.read_schema() only reads file metadata.
        Called BEFORE the recursive loop so the LLM always has exact
        column names from the very first iteration.
        """
        schemas: Dict[str, Any] = {}

        # Collect file paths from loaded_datasets + fallback discovery
        file_paths: List[str] = []
        for ds in (loaded_datasets or []):
            p = ds.get("path", "")
            if p and p != "<in-memory>":
                file_paths.append(p)

        # If no explicit datasets, discover from client data dir
        if not file_paths:
            try:
                client_data_dir = self._get_client_dataset_dir()
                if client_data_dir.exists():
                    file_paths.extend(str(f) for f in client_data_dir.glob("*.parquet"))
                    file_paths.extend(str(f) for f in client_data_dir.glob("*.csv"))
            except Exception as e:
                logger.warning(f"Schema discovery failed: {e}")

        if dataset_path and dataset_path not in file_paths:
            file_paths.append(dataset_path)

        for fpath in file_paths:
            try:
                fname = Path(fpath).name
                if self._planned_tables and not self._matches_planned_tables(fname):
                    continue
                if fpath.endswith(".parquet"):
                    import pyarrow.parquet as pq
                    pf = pq.ParquetFile(fpath)
                    schema = pf.schema_arrow
                    num_rows = pf.metadata.num_rows
                    schemas[fname] = {
                        "path": fpath,
                        "columns": [f.name for f in schema],
                        "types": {f.name: str(f.type) for f in schema},
                        "num_rows": num_rows,
                    }
                elif fpath.endswith(".csv"):
                    import pandas as _pd
                    header = _pd.read_csv(fpath, nrows=0)
                    schemas[fname] = {
                        "path": fpath,
                        "columns": header.columns.tolist(),
                        "types": {c: "unknown" for c in header.columns},
                    }
                logger.info(f"Schema read: {fname} → {len(schemas.get(fname, {}).get('columns', []))} columns")
            except Exception as e:
                logger.warning(f"Failed to read schema for {fpath}: {e}")

        return schemas

    def _build_parquet_schema_diagnostic(self, error_str: str) -> str:
        """
        Build diagnostic code that reads the parquet schema for the file
        mentioned in the ArrowInvalid error. Extracts the file path from
        the error traceback and uses pyarrow.parquet.read_schema().
        """
        # Try to extract file path from error message
        import re
        path_match = re.search(r"['\"]([^'\"]+\.parquet)['\"]" , error_str)
        if path_match:
            fpath = path_match.group(1)
            return (
                f"import pyarrow.parquet as pq\n"
                f"_s_ = pq.read_schema(r'{fpath}')\n"
                f"print('ACTUAL COLUMNS:', [f.name for f in _s_])\n"
                f"print('ACTUAL TYPES:', {{f.name: str(f.type) for f in _s_}})"
            )
        # Fallback: list all DataFrames and their columns
        return (
            "for _v,_o in globals().items():\n"
            "    if hasattr(_o,'columns'): print(f'{_v} columns:',_o.columns.tolist())"
        )

    # -------------------------------------------------------------------------
    # P5: Step Output Validation (silent failure detection)
    # -------------------------------------------------------------------------

    async def _validate_step_output(
        self,
        step: Dict,
        new_vars: Dict[str, Any],
        prev_vars: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Check for silent failures after a step appears to have succeeded.
        Returns (is_valid, issue_description).
        """
        issues = []

        for var_name, var_info in new_vars.items():
            if isinstance(var_info, dict) and var_info.get("type") == "DataFrame":
                shape = var_info.get("shape", [1, 1])
                if shape[0] == 0:
                    prev_info = prev_vars.get(var_name, {})
                    prev_shape = prev_info.get("shape", [1, 1]) if isinstance(prev_info, dict) else [1, 1]
                    if prev_shape[0] > 0:
                        issues.append(
                            f"{var_name} became empty (was {prev_shape[0]} rows, now 0)"
                        )

        if issues:
            return False, "; ".join(issues)
        return True, ""

    def _detect_row_explosion(
        self,
        new_vars: Dict[str, Any],
        prev_vars: Dict[str, Any],
    ) -> List[str]:
        """
        Detect possible cartesian joins by checking if a newly created
        DataFrame has suspiciously more rows than any previous DataFrame.
        Returns a list of warning strings (empty if no issues).
        """
        warnings = []
        # Find the max row count among previous DataFrames
        max_prev_rows = 0
        for info in prev_vars.values():
            if isinstance(info, dict) and info.get("type") == "DataFrame":
                rows = info.get("shape", [0, 0])[0]
                if rows > max_prev_rows:
                    max_prev_rows = rows

        if max_prev_rows == 0:
            return warnings

        for var_name, var_info in new_vars.items():
            if not isinstance(var_info, dict) or var_info.get("type") != "DataFrame":
                continue
            # Only check NEW DataFrames (not previously existing ones)
            if var_name in prev_vars:
                continue
            new_rows = var_info.get("shape", [0, 0])[0]
            if new_rows > max_prev_rows * 5:
                warnings.append(
                    f"ROW_EXPLOSION: '{var_name}' has {new_rows:,} rows but "
                    f"the largest input DataFrame had only {max_prev_rows:,} rows "
                    f"({new_rows / max_prev_rows:.1f}x). This may indicate a "
                    f"cartesian join from wrong merge keys. Verify the merge "
                    f"was correct before proceeding."
                )
        return warnings

    def _generate_zero_row_diagnostic(
        self,
        validation_issue: str,
        available_vars: Dict[str, Any],
    ) -> Optional[str]:
        """
        Generate diagnostic code when a DataFrame becomes empty (0 rows).
        Inspects all DataFrames in the kernel to find candidate filter/join
        columns and their unique values, helping the LLM self-correct.
        """
        import re as _re
        match = _re.match(r"(\w+) became empty", validation_issue)
        empty_var = match.group(1) if match else None

        code_lines = [
            "import pandas as _pd_diag_",
            "_diag_lines_ = []",
        ]

        if empty_var:
            code_lines.append(
                f"if '{empty_var}' in dir() and hasattr({empty_var}, 'columns'):\n"
                f"    _diag_lines_.append(f'EMPTY DF: {empty_var} columns={{list({empty_var}.columns)}}')"
            )

        code_lines.extend([
            "for _vn_ in sorted(globals().keys()):",
            "    if _vn_.startswith('_'): continue",
            "    _vobj_ = globals()[_vn_]",
            "    if not isinstance(_vobj_, _pd_diag_.DataFrame): continue",
            "    if _vobj_.shape[0] == 0: continue",
            "    _id_cols_ = [c for c in _vobj_.columns if any("
            "c.upper().endswith(s) for s in ('_ID', '_CODE', '_KEY', '_NUM', '_NAME'))]",
            "    if _id_cols_:",
            "        _diag_lines_.append(f'\\n{_vn_} ({_vobj_.shape[0]} rows):')",
            "        for _c_ in _id_cols_[:8]:",
            "            _uvals_ = _vobj_[_c_].dropna().unique()",
            "            _diag_lines_.append("
            "f'  {_c_}: {len(_uvals_)} unique, sample={list(_uvals_[:10])}')",
            "print('\\n'.join(_diag_lines_))",
        ])

        return "\n".join(code_lines)

    # -------------------------------------------------------------------------
    # P6: Adaptive Step Decomposition
    # -------------------------------------------------------------------------

    async def _decompose_failed_step(
        self,
        step: Dict,
        last_error: str,
        execution_context: Dict
    ) -> List[Dict]:
        """
        When a step fails all retries, ask the LLM to decompose it into 2 simpler sub-steps.
        Returns [] if decomposition fails or produces invalid output.
        The caller inserts the sub-steps into plan_steps in place of the failed step.
        """
        available_vars = list(execution_context.get("available_variables", {}).keys())
        prompt = (
            f"A data analysis step failed after all retries:\n\n"
            f"STEP: {step.get('description', '')}\n"
            f"LAST ERROR: {last_error[:300]}\n"
            f"AVAILABLE VARIABLES: {available_vars[:10]}\n\n"
            f"Decompose into exactly 2 simpler sequential sub-steps that together accomplish the same goal.\n"
            f"Sub-step 1 should validate/prepare data. Sub-step 2 should do the main computation.\n\n"
            f"Return ONLY a valid JSON array, no markdown, no explanation:\n"
            f'[\n'
            f'  {{"step_num": "{step["step_num"]}a", "description": "...", "details": ["..."]}},\n'
            f'  {{"step_num": "{step["step_num"]}b", "description": "...", "details": ["..."]}}\n'
            f']'
        )
        try:
            resp = await self.llm_client.generate_completion(
                system_prompt="You are a data science task decomposer. Return only valid JSON.",
                user_message=prompt,
                temperature=0.2,
                max_tokens=400
            )
            content = (resp.get("content") or "").strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                content = content[content.find("["):]
            if "[" in content and "]" in content:
                sub_steps = json.loads(content[content.find("["):content.rfind("]") + 1])
                if isinstance(sub_steps, list) and len(sub_steps) == 2:
                    logger.info(
                        f"Decomposed step {step['step_num']} → "
                        f"{[s['step_num'] for s in sub_steps]}"
                    )
                    return sub_steps
        except Exception as e:
            logger.warning(f"Step decomposition failed: {e}")
        return []

    # -------------------------------------------------------------------------
    # P1: llm_query() Kernel Injection (RLM Core Primitive)
    # -------------------------------------------------------------------------

    async def _inject_llm_query_helper(self) -> None:
        """
        Inject the llm_query() helper function into the Jupyter kernel.
        This is the RLM paper's core primitive: generated code can call back to the LLM
        for semantic reasoning (text classification, entity extraction, answer verification)
        that pure pandas/numpy cannot handle.
        """
        if isinstance(self.kernel_manager, DockerKernelManager):
            server_host = "host.docker.internal"
        else:
            server_host = "127.0.0.1"
        server_port = int(os.environ.get("BACKEND_PORT", "8024"))

        injection_code = f"""
import requests as _coresight_requests_
_CORESIGHT_LLM_URL_ = "http://{server_host}:{server_port}/api/agents/internal/llm-query"
_CORESIGHT_CLIENT_ID_ = "{self.client_id}"

def llm_query(question, context="", model="fast"):
    '''
    Call the CoreSight LLM for semantic reasoning on data.
    Use ONLY for tasks pandas cannot handle: text classification, entity extraction,
    answer verification.

    Args:
        question (str): What you want to know
        context (str): Data snippet to reason over (auto-truncated to 4000 chars)
        model (str): "fast" (default) or "smart" (more capable, slower)
        
    Returns:
        str: LLM response
    '''
    try:
        resp = _coresight_requests_.post(
            _CORESIGHT_LLM_URL_,
            json={{"client_id": _CORESIGHT_CLIENT_ID_, "question": str(question), "context": str(context)[:4000], "model": model}},
            timeout=45
        )
        if resp.status_code == 200:
            return resp.json().get("answer", "[no answer]")
        return f"[llm_query error {{resp.status_code}}]"
    except Exception as _e_:
        return f"[llm_query unavailable: {{_e_}}]"
"""
        if self._is_live_db:
            import base64
            import json
            creds_json = json.dumps(self.db_credentials_env)
            creds_b64 = base64.b64encode(creds_json.encode('utf-8')).decode('utf-8')
            
            injection_code += f"""
import atexit
import json
import base64
import os
import tempfile
import urllib.parse

import pandas as pd
from sqlalchemy import create_engine, text
from sshtunnel import SSHTunnelForwarder

_SSH_FORWARDER = None
_SSH_KEY_PATH = None
_SSH_SIGNATURE = None

def _cleanup_ssh_tunnel():
    global _SSH_FORWARDER, _SSH_KEY_PATH, _SSH_SIGNATURE

    if _SSH_FORWARDER is not None:
        try:
            _SSH_FORWARDER.stop()
        except Exception:
            pass
        _SSH_FORWARDER = None

    if _SSH_KEY_PATH and os.path.exists(_SSH_KEY_PATH):
        try:
            os.unlink(_SSH_KEY_PATH)
        except OSError:
            pass

    _SSH_KEY_PATH = None
    _SSH_SIGNATURE = None

atexit.register(_cleanup_ssh_tunnel)

def _ensure_ssh_tunnel(env, target_host, target_port):
    global _SSH_FORWARDER, _SSH_KEY_PATH, _SSH_SIGNATURE

    ssh_cfg = json.loads(env.get('CS_SSH_TUNNEL', '{{}}') or '{{}}')
    if not ssh_cfg.get('enabled'):
        return target_host, int(target_port)

    signature = (
        ssh_cfg.get('host', ''),
        int(ssh_cfg.get('port') or 22),
        ssh_cfg.get('username', ''),
        ssh_cfg.get('auth_method', 'password'),
        target_host,
        int(target_port),
    )

    if _SSH_FORWARDER is not None and _SSH_SIGNATURE == signature:
        return '127.0.0.1', int(_SSH_FORWARDER.local_bind_port)

    _cleanup_ssh_tunnel()

    tunnel_kwargs = {{
        'ssh_address_or_host': (ssh_cfg.get('host', ''), int(ssh_cfg.get('port') or 22)),
        'ssh_username': ssh_cfg.get('username', ''),
        'remote_bind_address': (target_host, int(target_port)),
        'local_bind_address': ('127.0.0.1', 0),
        'set_keepalive': 30.0,
    }}

    if ssh_cfg.get('auth_method') == 'private_key':
        key_content = ssh_cfg.get('private_key_content', '')
        if not key_content:
            raise ValueError('SSH private key content is missing')
        pem_file = tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            suffix='.pem',
            delete=False,
        )
        pem_file.write(key_content)
        pem_file.flush()
        pem_file.close()
        _SSH_KEY_PATH = pem_file.name
        tunnel_kwargs['ssh_pkey'] = _SSH_KEY_PATH
        if ssh_cfg.get('private_key_passphrase'):
            tunnel_kwargs['ssh_private_key_password'] = ssh_cfg.get('private_key_passphrase')
    else:
        tunnel_kwargs['ssh_password'] = ssh_cfg.get('password', '')

    _SSH_FORWARDER = SSHTunnelForwarder(**tunnel_kwargs)
    _SSH_FORWARDER.start()
    _SSH_SIGNATURE = signature
    return '127.0.0.1', int(_SSH_FORWARDER.local_bind_port)

def _get_engine():
    # Safely load credentials injected from the parent agent
    env = json.loads(base64.b64decode('{creds_b64}').decode('utf-8'))
    
    db_type = env.get('CS_DB_TYPE', '')
    host = env.get('CS_DB_HOST', '')
    port = env.get('CS_DB_PORT', '')
    db = env.get('CS_DB_NAME', '')
    user = env.get('CS_DB_USER', '')
    pwd = urllib.parse.quote_plus(env.get('CS_DB_PASSWORD', ''))

    default_ports = {{
        'sqlserver': 1433,
        'mssql': 1433,
        'postgres': 5432,
        'postgresql': 5432,
        'mysql': 3306,
        'sap_oracle': 1521,
        'oracle': 1521,
        'sap_hana': 39015,
        'hana': 39015,
        'sap_sybase': 5000,
        'sybase': 5000,
    }}

    effective_port = int(port or default_ports.get(db_type, 5432))
    effective_host, effective_port = _ensure_ssh_tunnel(env, host, effective_port)

    if db_type == 'sqlserver' or db_type == 'mssql':
        conn_str = f"mssql+pymssql://{{user}}:{{pwd}}@{{effective_host}}:{{effective_port}}/{{db}}"
    elif db_type == 'postgres' or db_type == 'postgresql':
        conn_str = f"postgresql://{{user}}:{{pwd}}@{{effective_host}}:{{effective_port}}/{{db}}"
    elif db_type == 'mysql':
        conn_str = f"mysql+pymysql://{{user}}:{{pwd}}@{{effective_host}}:{{effective_port}}/{{db}}"
    elif db_type == 'sap_oracle' or db_type == 'oracle':
        conn_str = f"oracle+oracledb://{{user}}:{{pwd}}@{{effective_host}}:{{effective_port}}/{{db}}"
    elif db_type == 'sap_hana' or db_type == 'hana':
        conn_str = f"hana+hdbcli://{{user}}:{{pwd}}@{{effective_host}}:{{effective_port}}"
    elif db_type == 'sap_sybase' or db_type == 'sybase':
        if db:
            conn_str = f"sybase+pyodbc://{{user}}:{{pwd}}@{{effective_host}}:{{effective_port}}/{{db}}"
        else:
            conn_str = f"sybase+pyodbc://{{user}}:{{pwd}}@{{effective_host}}:{{effective_port}}"
    else:
        raise ValueError(f"Unsupported db_type: {{db_type}}")
        
    return create_engine(conn_str)

def read_sql_query(sql_query):
    '''
    Execute a T-SQL query against the live database and return a Pandas DataFrame.
    Use this to aggregate/filter massive tables BEFORE loading them into Python.
    Never use 'SELECT *' on large tables without aggregating or limiting.
    '''
    engine = _get_engine()
    with engine.begin() as conn:
        return pd.read_sql(text(sql_query), conn)

def query_sql(sql_query):
    '''Alias for read_sql_query'''
    return read_sql_query(sql_query)
"""

        injection_code += """
print("llm_query() ready")
"""
        result = await self._execute_code(injection_code)
        if "llm_query() ready" in result.get("stdout", ""):
            logger.info("llm_query() injected successfully into kernel")
        else:
            logger.warning(
                f"llm_query() injection may have failed: {result.get('stdout', '')[:100]}"
            )

    async def _check_final_result_in_kernel(self) -> bool:
        """
        Lightweight check: does FINAL_RESULT exist in the kernel?

        This is separate from _get_kernel_variables() because that method
        can fail silently when the kernel has large variables (e.g., Plotly
        JSON in FINAL_RESULT) that blow up the introspection output.
        This check is tiny and cannot fail.
        """
        try:
            result = await self._execute_code(
                "print('__FR_YES__' if 'FINAL_RESULT' in dir() else '__FR_NO__')"
            )
            return "__FR_YES__" in result.get("stdout", "")
        except Exception:
            return False

    async def _get_kernel_variables(self) -> Dict[str, Any]:
        """
        Get current variables from the Jupyter kernel with detailed info.

        Returns:
            Dictionary of variable names -> metadata (type, columns/info).
        """
        try:
            code = """
import json as _json_
import pandas as _pd_
_exclude_ = {'_json_', '_pd_', '_vars_', '_name_', '_exclude_', '_obj_', '_info_', 'In', 'Out', 'get_ipython', 'exit', 'quit', 'open', '_intent_dict_'}
_vars_ = {}
_intent_dict_ = globals().get('_VAR_INTENT_', {})

for _name_ in list(globals().keys()):
    if _name_.startswith('__') and _name_.endswith('__'):
        continue
    if _name_ in _exclude_:
        continue
    try:
        _obj_ = eval(_name_)
        _info_ = {"type": str(type(_obj_).__name__)}
        if isinstance(_obj_, _pd_.DataFrame):
            _info_["columns"] = _obj_.columns.tolist()
            _info_["shape"] = list(_obj_.shape)
            # Use 5-row sample for richer context without overwhelming the prompt
            try:
                _info_["dtypes"] = {col: str(dtype) for col, dtype in _obj_.dtypes.items()}
            except Exception:
                pass
            try:
                _info_["sample"] = _obj_.head(5).to_dict(orient='records')
            except Exception:
                pass
        elif not callable(_obj_):
            # Capture scalar/string values for non-callable non-DataFrame vars
            try:
                if _name_ in ['TASKS', 'COMPLETED_TASKS', '_VAR_INTENT_']:
                    _info_["value"] = repr(_obj_)[:5000]
                else:
                    _info_["value"] = repr(_obj_)[:500]
            except Exception:
                pass
                
        # Inject explicit LLM intent if it documented why it created this variable
        if _name_ in _intent_dict_ and isinstance(_intent_dict_, dict):
            _info_["intent"] = str(_intent_dict_[_name_])
            
        if not callable(_obj_) or _name_[0].isupper():
            _vars_[_name_] = _info_
    except Exception:
        pass
print(_json_.dumps(_vars_, default=str))
"""
            result = await self._execute_code(code)
            
            if result.get("stdout"):
                import json
                stdout = result["stdout"].strip()
                # Handle potential mixed output, take the last valid JSON line
                lines = stdout.split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict):
                            return parsed
                        # Skip non-dict JSON values (int, str, list, etc.)
                        continue
                    except (json.JSONDecodeError, ValueError):
                        # Fallback: try ast.literal_eval for Python repr strings
                        # (e.g. single-quoted dicts from MCP wrapper)
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
            logger.warning(f"Could not get kernel variables: {e}")
            return {}

    def _is_likely_prose(self, text: str) -> bool:
        """Return True if the text looks like natural-language prose rather than Python code.

        Used to detect when an LLM returns an explanation instead of code so we can
        reject it immediately (before wasting a 60s Jupyter kernel timeout).
        """
        if not text:
            return True
        first_line = text.strip().split('\n')[0].strip().lower()
        prose_starters = (
            "i am", "i'm", "i cannot", "i can't", "i will ", "i'll ",
            "the environment", "unfortunately", "to complete",
            "here is", "here's", "please note", "note that",
            "as an ai", "due to", "based on", "the previous",
            "i need to", "i will need", "i understand",
            "it seems", "it appears", "this step", "the code",
        )
        if any(first_line.startswith(p) for p in prose_starters):
            return True
        # If none of the common Python tokens appear anywhere in the text, treat as prose
        python_indicators = [
            'import ', 'pd.', 'df', 'print(', ' = ', 'def ',
            'for ', 'if ', '# ', '.read_', 'np.', 'sklearn',
        ]
        return not any(ind in text for ind in python_indicators)

    def _stdout_contains_error(self, stdout: str) -> bool:
        """Check if stdout contains Python error patterns (traceback in output)."""
        if not stdout:
            return False
        error_indicators = [
            "Traceback (most recent call last)",
            "ArrowInvalid:", "NameError:", "KeyError:",
            "TypeError:", "ValueError:", "FileNotFoundError:",
            "ModuleNotFoundError:", "ImportError:", "MemoryError:",
            "AttributeError:", "IndexError:", "ZeroDivisionError:",
        ]
        return any(indicator in stdout for indicator in error_indicators)

    def _extract_error_from_stdout(self, stdout: str) -> str:
        """Extract the last error block from stdout text."""
        lines = stdout.strip().split('\n')
        # Find last Traceback and return everything after it
        for i in range(len(lines) - 1, -1, -1):
            if "Traceback" in lines[i]:
                return '\n'.join(lines[i:])
        # Fallback: return last 5 lines containing the error
        return '\n'.join(lines[-5:])

    @traceable(name="coder_generate_code")
    async def _generate_code(
        self,
        user_query: str,
        context: Dict
    ) -> str:
        """Generate Python code for the next step."""
        iteration = context.get("iteration", 1)
        
        # Build context prompt
        context_prompt = self._build_context_prompt(context)
        
        prompt = f"""
                        {self.base_prompt}

                        User Query: {user_query}

                        Iteration: {iteration}/{self.max_iterations}

                        {context_prompt}

                        Generate Python code to advance toward answering the user's query.
                        The code will execute in a Jupyter environment with pandas, numpy, matplotlib, and plotly available.

                        Requirements:
                        1. Use clean, commented code
                        2. Handle errors gracefully
                        3. Print clear output about what was done
                        4. Store results in variables for next iteration
                        5. If generating visualizations, save them and describe findings
                        6. Be specific and focused on the query

                        Return ONLY the Python code, no explanations:
                 """
        
        response = await self.llm_client.generate_completion(
            system_prompt=self.base_prompt,
            user_message=prompt,
            temperature=self.temperature,
            max_tokens=2000
        )
        
        # Extract code from response (handle markdown code blocks)
        code = response.get("content", "").strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()

    @traceable(name="coder_execute_code")
    async def _execute_code(self, code: str) -> Dict[str, Any]:
        """Execute code in the active MCP notebook kernel.

        Returns:
            Dict with keys: stdout, stderr, exception, variables
        """
        result = {
            "stdout": "",
            "stderr": "",
            "exception": None,
            "variables": {}
        }
        
        # ── SQL Pre-validation (Custom Reliability Hack) ────────────────────
        # Extract SQL string if calling read_sql_query or query_sql
        # Brittle but effective for catching common syntax traps like RowCount
        sql_calls = re.findall(r"(?:read_sql_query|query_sql)\s*\(\s*['\"]+(.*?)['\"]+\s*\)", code, re.DOTALL)
        if sql_calls:
            db_type = self.datasource_context.get("db_type", "mssql") if self.datasource_context else "mssql"
            validator = SQLValidatorFactory.get_validator(db_type)
            for sql in sql_calls:
                val_errors = validator.validate(sql)
                if val_errors:
                    logger.warning(f"SQL Validation triggered: {val_errors}")
                    result["exception"] = (
                        "SQL_SYNTAX_WARNING: Your code contains a potential SQL syntax error. "
                        + " ".join(val_errors)
                    )
                    return result

        if not self.mcp_client:
            result["exception"] = "MCP client not initialized"
            logger.error("Cannot execute code: MCP client not initialized")
            return result
        
        try:
            # Update kernel manager activity timestamp
            if self.kernel_manager:
                self.kernel_manager.update_activity()
            
            # Execute code as a notebook cell so variables persist across iterations.
            # jupyter-mcp-server: insert_execute_code_cell(timeout default 90s).
            logger.debug(f"Executing notebook cell via MCP (timeout={self.timeout_per_execution}s): {code[:100]}...")
            # NOTE: Some jupyter-mcp-server versions do NOT accept -1 for insert_execute_code_cell.
            # Use 0 to insert at the top deterministically (still stateful, avoids tool errors).
            mcp_result = await self.mcp_client.call_tool(
                name="insert_execute_code_cell",
                arguments={"cell_index": 0, "cell_source": code, "timeout": int(self.timeout_per_execution)},
                timeout_seconds=float(self.timeout_per_execution) + 10,
            )
            
            # Parse MCP result — _extract_result returns different types:
            # - str: text output from the tool (most common for code execution)
            # - dict: structured content (if server returns structuredContent)
            # - list: multiple content items
            # - int/float: parsed simple values
            # - None: no output
            if mcp_result is None:
                result["stdout"] = ""
            elif isinstance(mcp_result, dict):
                # jupyter-mcp-server returns {'result': ['line1', 'line2', ...]}
                if "result" in mcp_result:
                    val = mcp_result["result"]
                    if isinstance(val, list):
                        result["stdout"] = "\n".join(str(item) for item in val)
                    else:
                        result["stdout"] = str(val)
                elif "output" in mcp_result:
                    result["stdout"] = str(mcp_result["output"])
                elif "stdout" in mcp_result:
                    result["stdout"] = str(mcp_result["stdout"])
                else:
                    result["stdout"] = str(mcp_result)
            elif isinstance(mcp_result, list):
                # Multiple content items
                result["stdout"] = "\n".join(str(item) for item in mcp_result)
            else:
                # String, int, float — convert to string
                result["stdout"] = str(mcp_result)
            
            logger.debug(f"Code execution successful, output length: {len(result['stdout'])}")
            
        except McpTimeoutError as e:
            result["exception"] = f"Code execution timed out: {e}"
            logger.error(f"MCP timeout: {e}")
        except McpError as e:
            result["exception"] = f"MCP execution error: {e}"
            result["stderr"] = str(e)
            logger.error(f"MCP error: {e}")
        except Exception as e:
            result["exception"] = str(e)
            result["stderr"] = traceback.format_exc()
            logger.error(f"Code execution failed: {e}\n{traceback.format_exc()}")
        
        return result

    async def _analyze_results(
        self,
        execution_result: Dict,
        user_query: str
    ) -> Dict[str, Any]:
        """Analyze execution results and decide next step."""
        output = execution_result.get("stdout", "")
        
        prompt = f"""
Analyze the following code execution output and the original user query.

Original Query: {user_query}

Execution Output:
{output}

Provide analysis in JSON format with:
{{
    "findings": "Brief summary of what was discovered",
    "insights": "Key insights from the data",
    "should_continue": boolean (true if more iterations needed),
    "next_step": "What to do in next iteration (if should_continue is true)",
    "concerns": "Any issues or edge cases to address"
}}

Be concise and data-focused.
"""
        
        response = await self.llm_client.generate_completion(
            system_prompt="You are a data analysis expert.",
            user_message=prompt,
            temperature=0.2,
            max_tokens=1000
        )
        
        # Update usage stats
        self._update_usage(response.get("usage"))
        
        try:
            # Extract JSON from response
            content = response.get("content", "")
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "findings": output[:200],
            "should_continue": False,
            "next_step": None
        }

    async def _generate_final_result(self, context: Dict) -> Dict[str, Any]:
        """Generate final prediction/insights from completed iterations."""
        # Use completed_iterations from execution context
        completed = context.get("completed_iterations", [])
        
        if not completed:
            logger.warning("No completed iterations for final result generation")
            return {
                "prediction": "Analysis could not be completed — no iterations executed successfully.",
                "text_output": "No results generated.",
                "dataframe": None,
                "iterations_completed": 0,
                "timestamp": utcnow().isoformat()
            }
        
        history_summary = json.dumps([
            {
                "iteration": s.get("iteration", s.get("step_num", "?")),
                "reasoning": s.get("reasoning", s.get("description", "")),
                "output": s.get("output", "")[:1000]
            }
            for s in completed[-5:]  # Last 5 iterations for context
        ], indent=2)
        
        prompt = f"""
Based on the step-by-step data science analysis below, provide final insights and predictions.

User Query: {context['user_query']}

Completed Iterations:
{history_summary}

Generate a comprehensive final result in MARKDOWN format:
1. **Answer**: Direct answer to the user's query
2. **Key Findings**: Bullet points of key metrics and discovery
3. **Confidence**: Assessment of result reliability
4. **Recommendations**: Actionable next steps

FORMATTING RULES:
- Use `##` for section headers (e.g. `## 1. Answer`)
- Always add a blank line before headers
- Use `**bold**` for key numbers and terms
- Use `*` for bullet points
- Do NOT use plain text blocks without formatting
"""
        
        response = await self.llm_client.generate_completion(
            system_prompt="You are a data science expert providing final analysis.",
            user_message=prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Update usage stats
        self._update_usage(response.get("usage"))
        
        # Format usage stats for final result (convert sets to list)
        final_usage = self.usage_stats.copy()
        if "models" in final_usage and isinstance(final_usage["models"], set):
            final_usage["models"] = list(final_usage["models"])
        
        return {
            "prediction": response.get("content", ""),
            "text_output": response.get("content", ""),
            "dataframe": None,  # Dataframes are captured from kernel variables at step level
            "iterations_completed": len(completed),
            "timestamp": utcnow().isoformat(),
            "_agent_usage": final_usage
        }

    def _build_context_prompt(self, context: Dict) -> str:
        """Build context from previous iterations."""
        if context.get("iteration") == 1:
            return "This is the first iteration. Focus on data exploration and understanding."
        
        last_analysis = context.get("last_results", {})
        return f"""
Previous findings: {last_analysis.get('findings', 'None yet')}
Next step guidance: {last_analysis.get('next_step', 'Continue analysis')}
"""

    async def _cleanup_kernel(self) -> None:
        """Cleanup MCP kernel and client resources."""
        try:
            # Cleanup MCP client first
            if hasattr(self, '_mcp_context_manager') and self._mcp_context_manager:
                try:
                    await self._mcp_context_manager.__aexit__(None, None, None)
                    logger.info("MCP client cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up MCP client: {e}")
                finally:
                    self._mcp_context_manager = None
                    self.mcp_client = None
            
            # Cleanup stdio context
            if hasattr(self, '_stdio_context_manager') and self._stdio_context_manager:
                try:
                    await self._stdio_context_manager.__aexit__(None, None, None)
                    logger.info("MCP stdio cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up MCP stdio: {e}")
                finally:
                    self._stdio_context_manager = None
            
            # Release kernel back to pool (or stop if no pooling)
            if self.kernel_manager:
                logger.info("Releasing kernel manager")
                await release_kernel_manager(self.kernel_manager, use_pool=getattr(self, '_use_pool', True))
                self.kernel_manager = None
                
        except Exception as e:
            logger.warning(f"Error cleaning up kernel resources: {e}")

    async def _fetch_generated_dataframe(self) -> Optional[List[Dict]]:
        """Fetch the final dataframe content from the kernel.

        Priority order:
        1. FINAL_RESULT variable (explicit LLM-declared output, RLM pattern)
        2. _generated_dataframe_ (legacy explicit output)
        3. df (generic fallback)
        """
        try:
            code = """
import json as _json_
import pandas as _pd_
_target_df_ = None
_final_val_ = None

# 1. Check FINAL_RESULT first (RLM pattern — LLM declares its output explicitly)
try:
    if 'FINAL_RESULT' in globals():
        _fr_ = FINAL_RESULT
        if isinstance(_fr_, _pd_.DataFrame):
            _target_df_ = _fr_
        elif isinstance(_fr_, list) and _fr_ and isinstance(_fr_[0], dict):
            # List of records — convert directly
            _target_df_ = _pd_.DataFrame(_fr_)
        elif isinstance(_fr_, dict):
            # Dict with list-of-dicts values → extract the tabular data
            _list_cols_ = {k: v for k, v in _fr_.items()
                           if isinstance(v, list) and v and isinstance(v[0], dict)}
            if _list_cols_:
                # Use the largest list (most records = most useful table)
                _best_key_ = max(_list_cols_, key=lambda k: len(_list_cols_[k]))
                _target_df_ = _pd_.DataFrame(_list_cols_[_best_key_])
            else:
                # Scalar dict — display as a single-row table
                try:
                    _target_df_ = _pd_.DataFrame([_fr_])
                except Exception:
                    _final_val_ = repr(_fr_)[:2000]
        else:
            _final_val_ = repr(_fr_)[:2000]
except Exception:
    pass

# 2. Fall back to _generated_dataframe_
if _target_df_ is None and _final_val_ is None:
    try:
        if '_generated_dataframe_' in globals() and isinstance(_generated_dataframe_, _pd_.DataFrame):
            _target_df_ = _generated_dataframe_
    except NameError:
        pass

# 3. Fall back to df
if _target_df_ is None and _final_val_ is None:
    try:
        if 'df' in globals() and isinstance(df, _pd_.DataFrame):
            _target_df_ = df
    except NameError:
        pass

if _target_df_ is not None:
    _json_str_ = _target_df_.head(500).to_json(orient='records', date_format='iso')
    print(f"<JSON_START>{_json_str_}<JSON_END>")
elif _final_val_ is not None:
    print(f"<FINAL_VAL>{_final_val_}</FINAL_VAL>")
else:
    print("NO_DATAFRAME_FOUND")
    print(f"DEBUG_VARS: {list(globals().keys())}")
"""
            result = await self._execute_code(code)
            output = result.get("stdout", "").strip()

            # 1. DataFrame result
            if "<JSON_START>" in output and "<JSON_END>" in output:
                try:
                    start_idx = output.find("<JSON_START>") + len("<JSON_START>")
                    end_idx = output.find("<JSON_END>")
                    json_str = output[start_idx:end_idx].strip()
                    if json_str:
                        return json.loads(json_str)
                except Exception as parse_err:
                    logger.warning(f"Failed to parse dataframe JSON: {parse_err}")

            # 2. Scalar FINAL_RESULT — return as single-row table for display
            if "<FINAL_VAL>" in output and "</FINAL_VAL>" in output:
                start_idx = output.find("<FINAL_VAL>") + len("<FINAL_VAL>")
                end_idx = output.find("</FINAL_VAL>")
                val_str = output[start_idx:end_idx].strip()
                logger.info(f"FINAL_RESULT scalar: {val_str[:100]}")
                return [{"FINAL_RESULT": val_str}]

            if "NO_DATAFRAME_FOUND" in output:
                logger.warning(f"Dataframe fetch failed. Kernel output: {output}")

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch dataframe: {e}")
            return None

    async def _fetch_generated_chart(self) -> Optional[Dict]:
        """Fetch the final Plotly chart from the Jupyter kernel.

        Priority:
        1. _generated_plotly_fig_ (explicit plotly output variable)
        2. fig (common fallback name)
        3. FINAL_RESULT['chart'] (dict-style output)
        """
        try:
            code = """
import json as _json_
_chart_out_ = None
try:
    if '_generated_plotly_fig_' in globals():
        _chart_out_ = _generated_plotly_fig_.to_json()
    elif 'fig' in globals() and hasattr(fig, 'to_json'):
        _chart_out_ = fig.to_json()
    elif 'FINAL_RESULT' in globals() and isinstance(FINAL_RESULT, dict) and 'chart' in FINAL_RESULT:
        import plotly.io as _pio_
        _chart_out_ = _pio_.to_json(FINAL_RESULT['chart'])
except Exception:
    pass
if _chart_out_:
    print(f"<CHART_START>{_chart_out_}<CHART_END>")
else:
    print("NO_CHART_FOUND")
"""
            result = await self._execute_code(code)
            output = result.get("stdout", "").strip()
            if "<CHART_START>" in output and "<CHART_END>" in output:
                try:
                    s = output.find("<CHART_START>") + len("<CHART_START>")
                    e = output.find("<CHART_END>")
                    return json.loads(output[s:e].strip())
                except Exception as exc:
                    logger.warning(f"Failed to parse chart JSON: {exc}")
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch chart: {e}")
            return None

    async def _fetch_all_generated_charts(self) -> list:
        """Fetch ALL Plotly figures from the Jupyter kernel namespace.

        Scans kernel globals() for every plotly.graph_objects.Figure instance,
        deduplicates by id(), and returns them as a list of
        ``{"name": str, "figure": dict}`` dicts.

        Uses ``<MCHART>name|||json<MCHART_END>`` markers so the Python-side
        parser can split multiple charts reliably.
        """
        try:
            code = """
import json as _json_

_seen_ids_ = set()
_charts_out_ = []

try:
    import plotly.graph_objects as _go_

    for _name_, _obj_ in list(globals().items()):
        if _name_.startswith('_') and not _name_.startswith('_generated_plotly_fig_'):
            continue
        if isinstance(_obj_, _go_.Figure):
            _oid_ = id(_obj_)
            if _oid_ in _seen_ids_:
                continue
            _seen_ids_.add(_oid_)
            try:
                _charts_out_.append((_name_, _obj_.to_json()))
            except Exception:
                pass

    # Sort: regular names first (alphabetical), _generated_plotly_fig_* last
    _charts_out_.sort(key=lambda x: (x[0].startswith('_generated_plotly_fig_'), x[0]))

    if _charts_out_:
        for _cname_, _cjson_ in _charts_out_:
            print(f"<MCHART>{_cname_}|||{_cjson_}<MCHART_END>")
    else:
        print("NO_CHARTS_FOUND")
except ImportError:
    print("NO_CHARTS_FOUND")
except Exception as _e_:
    print(f"NO_CHARTS_FOUND: {_e_}")
"""
            result = await self._execute_code(code)
            output = result.get("stdout", "")

            if "NO_CHARTS_FOUND" in output:
                return []

            # Parse all <MCHART>name|||json<MCHART_END> markers
            import re
            charts = []
            for m in re.finditer(r"<MCHART>(.*?)<MCHART_END>", output, re.DOTALL):
                raw = m.group(1)
                sep = raw.find("|||")
                if sep == -1:
                    continue
                name = raw[:sep].strip()
                json_str = raw[sep + 3:].strip()
                try:
                    figure = json.loads(json_str)
                    charts.append({"name": name, "figure": figure})
                except Exception as exc:
                    logger.warning(f"Skipping chart '{name}': bad JSON ({exc})")

            if charts:
                logger.info(
                    "Fetched %d Plotly charts from kernel: %s",
                    len(charts),
                    [c["name"] for c in charts],
                )
            return charts

        except Exception as e:
            logger.warning(f"Failed to fetch all charts: {e}")
            return []

    async def _stream_event(self, event_type: str, content: Dict) -> Dict:
        """Create a streaming event."""
        event = {
            "type": event_type,
            "timestamp": utcnow().isoformat()
        }
        # Merge content into top level for easier access
        event.update(content)
        return event
