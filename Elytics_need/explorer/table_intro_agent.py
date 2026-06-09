from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Iterable, Optional, Any
# Safe: escape() is used for XML/HTML escaping, not parsing untrusted input.
from xml.sax.saxutils import escape  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml

from explorer.models import TableMetadata
from util.llm_utils import LLMClient
from config.system_config import AGENT_CONFIG

logger = logging.getLogger(__name__)


INTRO_SYSTEM_PROMPT = (
    "You are an expert data documentation assistant. Given table metadata and sample rows, "
    "write a concise (2-3 sentence) description explaining the table's purpose, typical fields, "
    "and analytical use-cases. Avoid referencing specific company names. Write in plain English."
)

# Load configuration from centralized system config
_cfg = AGENT_CONFIG["explorer_table_intro"]
MAX_SAMPLE_ROWS = _cfg["max_sample_rows"]
MAX_COLUMN_VALUE_LENGTH = _cfg["max_column_value_length"]
MAX_TOTAL_PROMPT_LENGTH = _cfg["max_total_prompt_length"]
MAX_COLUMNS_TO_SHOW = _cfg["max_columns_to_show"]
MAX_OUTPUT_TOKENS = _cfg["max_output_tokens"]

# Max concurrent LLM calls to reduce total time while avoiding rate limits
MAX_CONCURRENT_LLM_CALLS = _cfg["max_concurrent_llm_calls"]


class TableIntroductionAgent:
    """Uses an LLM to draft table introduction summaries."""

    def __init__(self, client_id: str, agent_name: str = "explorer_table_intro", db: Optional[Any] = None):
        self.client_id = client_id
        self.llm = LLMClient(agent_name=agent_name, client_id=client_id, db=db)

    def _truncate_value(self, value: Any, max_length: int = MAX_COLUMN_VALUE_LENGTH) -> str:
        """Truncate a value to a maximum length for display."""
        if value is None:
            return "null"
        str_value = str(value)
        if len(str_value) > max_length:
            return str_value[:max_length] + "..."
        return str_value

    def _prepare_sample_rows(self, sample_rows: list[dict[str, Any]], max_rows: int = MAX_SAMPLE_ROWS) -> str:
        """Prepare sample rows with truncation to prevent excessive token usage."""
        if not sample_rows:
            return "[]"
        
        # Limit number of rows
        limited_rows = sample_rows[:max_rows]
        
        # Truncate long values in each row
        truncated_rows = []
        for row in limited_rows:
            truncated_row = {
                k: self._truncate_value(v) 
                for k, v in row.items()
            }
            truncated_rows.append(truncated_row)
        
        return json.dumps(truncated_rows, indent=2, default=str)

    def _prepare_columns_list(self, columns: list) -> str:
        """Prepare column list, potentially limiting if too many columns."""
        if MAX_COLUMNS_TO_SHOW is None or len(columns) <= MAX_COLUMNS_TO_SHOW:
            return ", ".join(col.name for col in columns)
        else:
            # Show first N columns and indicate there are more
            shown = ", ".join(col.name for col in columns[:MAX_COLUMNS_TO_SHOW])
            return f"{shown} ... ({len(columns) - MAX_COLUMNS_TO_SHOW} more columns)"

    def _build_user_prompt(self, table: TableMetadata) -> str:
        """Build user prompt with size management."""
        columns_str = self._prepare_columns_list(table.columns)
        sample_str = self._prepare_sample_rows(table.sample_rows)
        
        prompt = (
            f"Table name: {table.name}\n"
            f"Columns: {columns_str}\n"
            f"Sample rows:\n{sample_str}\n\n"
            "Return only the description text."
        )
        
        # Final safety check: truncate entire prompt if still too long
        if len(prompt) > MAX_TOTAL_PROMPT_LENGTH:
            # Try reducing sample rows further
            sample_str = self._prepare_sample_rows(table.sample_rows, max_rows=1)
            prompt = (
                f"Table name: {table.name}\n"
                f"Columns: {columns_str}\n"
                f"Sample rows:\n{sample_str}\n\n"
                "Return only the description text."
            )
            # If still too long, remove sample rows entirely
            if len(prompt) > MAX_TOTAL_PROMPT_LENGTH:
                prompt = (
                    f"Table name: {table.name}\n"
                    f"Columns: {columns_str}\n\n"
                    "Return only the description text."
                )
        
        return prompt

    def _build_fallback_description(self, table: TableMetadata) -> str:
        """Build a deterministic fallback when LLM generation fails."""
        col_names = [c.name for c in (table.columns or [])][:8]
        col_preview = ", ".join(col_names) if col_names else "its available columns"
        return (
            f"{table.name} stores business records used for analysis and reporting. "
            f"It includes fields such as {col_preview}. "
            "Use this table for filtering, aggregation, and trend exploration."
        )

    async def _describe_table(self, table: TableMetadata) -> tuple[str, dict | None]:
        """Generate table description with automatic input size management.
        
        Returns:
            Tuple of (description, usage_info) where usage_info contains token usage if available
        """
        user_prompt = self._build_user_prompt(table)
        usage_info = None

        result = await self.llm.generate_completion(
            system_prompt=INTRO_SYSTEM_PROMPT,
            user_message=user_prompt,
            temperature=0.2,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        if result["error"]:
            # Check if it's a length-related error and try fallback
            error_str = str(result["error"]).lower()
            if "reduce the length" in error_str or "too long" in error_str or "maximum context length" in error_str:
                # Fallback: try with minimal input (columns only, no samples)
                logger.warning(f"Input too long for {table.name}, trying fallback with columns only")
                fallback_prompt = (
                    f"Table name: {table.name}\n"
                    f"Columns: {self._prepare_columns_list(table.columns)}\n\n"
                    "Return only the description text."
                )
                result = await self.llm.generate_completion(
                    system_prompt=INTRO_SYSTEM_PROMPT,
                    user_message=fallback_prompt,
                    temperature=0.2,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
                if result["error"]:
                    raise RuntimeError(f"LLM error while documenting {table.name}: {result['error']}")
            else:
                raise RuntimeError(f"LLM error while documenting {table.name}: {result['error']}")
        
        # Capture usage if available
        usage_info = result.get("usage")
        return (result["content"] or "").strip(), usage_info

    async def generate(self, tables: Iterable[TableMetadata], output_path: Path) -> dict:
        """Generate table introductions for all tables, continuing even if some fail.
        Uses parallel LLM calls (limited by MAX_CONCURRENT_LLM_CALLS) to reduce total time.
        
        Returns:
            Dict with 'total_token_usage' aggregated across all table descriptions
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tables_list = list(tables) if not isinstance(tables, list) else tables
        total_tables = len(tables_list)
        
        # Track token usage across all tables
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        xml_lines = [
            '<?xml version="1.0" ?>',
            '<meta_information type="table_introductions">',
            "  <table_introductions>",
        ]

        logger.info(f"Starting table introduction generation for {total_tables} tables (parallel, max {MAX_CONCURRENT_LLM_CALLS} concurrent)")

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

        async def process_one(table: TableMetadata) -> tuple[str, str | None, dict | None]:
            """Process one table with semaphore. Returns (table_name, description, usage_info)."""
            async with semaphore:
                try:
                    description, usage_info = await self._describe_table(table)
                    return (table.name, description, usage_info)
                except Exception as e:
                    logger.error(
                        f"Failed to generate introduction for table {table.name}: {e}",
                        exc_info=True
                    )
                    fallback = self._build_fallback_description(table)
                    logger.warning(
                        "Using fallback introduction for table %s due to LLM failure",
                        table.name,
                    )
                    return (table.name, fallback, None)

        results = await asyncio.gather(
            *[process_one(table) for table in tables_list],
            return_exceptions=False
        )

        successful = 0
        failed = 0
        for table_name, description, usage_info in results:
            if description is not None:
                xml_lines.append(
                    f'    <table_introduction table_name="{escape(table_name)}">{escape(description)}</table_introduction>'
                )
                if usage_info:
                    total_prompt_tokens += usage_info.get("prompt_tokens", 0) or 0
                    total_completion_tokens += usage_info.get("completion_tokens", 0) or 0
                    total_tokens += (
                        (usage_info.get("prompt_tokens", 0) or 0)
                        + (usage_info.get("completion_tokens", 0) or 0)
                    )
                successful += 1
            else:
                failed += 1

        xml_lines.extend(["  </table_introductions>", "</meta_information>"])
        content = "\n".join(xml_lines)
        await asyncio.to_thread(output_path.write_text, content, "utf-8")
        
        logger.info(
            f"Table introduction generation complete: {successful} successful, {failed} failed out of {total_tables} total"
        )
        
        if failed > 0:
            logger.warning(
                f"{failed} table(s) failed to generate introductions. "
                f"Check logs above for details. Successful: {successful}/{total_tables}"
            )
        
        # Return token usage summary
        return {
            "total_token_usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "tables_processed": successful
            }
        }

