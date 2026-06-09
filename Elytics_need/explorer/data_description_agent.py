from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Optional, Any
# Safe: escape() is used for XML/HTML escaping, not parsing untrusted input.
from xml.sax.saxutils import escape  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml

from explorer.models import TableMetadata
from util.llm_utils import LLMClient
from config.system_config import AGENT_CONFIG
from services.subscription_service import get_explorer_limits

logger = logging.getLogger(__name__)

# Premium plans (column_limit >= 100 or unlimited) get high output token cap so all columns get described
PREMIUM_COLUMN_LIMIT_THRESHOLD = 100
UNLIMITED_OUTPUT_TOKENS = 16384

DATA_DESC_SYSTEM_PROMPT = (
    "You are a data documentation specialist. Given a table, its columns, and sample rows, "
    "describe each column's purpose in 1-2 sentences. Focus on business context, typical values, "
    "and how analysts should interpret it. "
    "You MUST return ONLY valid JSON in the format: {\"column_name\": \"description\", ...}. "
    "Do not include any markdown code fences, explanations, or additional text - only the JSON object."
)

# Load configuration from centralized system config
_cfg = AGENT_CONFIG["explorer_data_desc"]
MAX_SAMPLE_ROWS = _cfg["max_sample_rows"]
MAX_COLUMN_VALUE_LENGTH = _cfg["max_column_value_length"]
MAX_TOTAL_PROMPT_LENGTH = _cfg["max_total_prompt_length"]
MAX_COLUMNS_TO_SHOW = _cfg["max_columns_to_show"]
MAX_OUTPUT_TOKENS = _cfg["max_output_tokens"]

# Max concurrent LLM calls to reduce total time while avoiding rate limits
MAX_CONCURRENT_LLM_CALLS = _cfg["max_concurrent_llm_calls"]


class DataDescriptionAgent:
    """Generates per-table data_description XML files using LLM analysis."""

    def __init__(
        self,
        client_id: str,
        output_dir: Path,
        agent_name: str = "explorer_data_desc",
        db: Optional[Any] = None,
        data_sources_base: Optional[Path] = None,
    ):
        self.client_id = client_id
        self.output_dir = output_dir
        self.data_sources_base = data_sources_base or (output_dir / "data_sources")
        self.db = db
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
        if MAX_COLUMNS_TO_SHOW is None:
            return ", ".join(col.name for col in columns)
        if len(columns) <= MAX_COLUMNS_TO_SHOW:
            return ", ".join(col.name for col in columns)
        # Show first N columns and indicate there are more
        shown = ", ".join(col.name for col in columns[:MAX_COLUMNS_TO_SHOW])
        return f"{shown} ... ({len(columns) - MAX_COLUMNS_TO_SHOW} more columns)"

    def _build_user_prompt(self, table: TableMetadata) -> str:
        """Build user prompt with size management."""
        columns_str = self._prepare_columns_list(table.columns)
        json_instruction = "Return ONLY a valid JSON object in this exact format: {\"column_name\": \"description\", ...}. Do not use markdown code fences or add any other text."
        
        # Try with full samples first
        sample_str = self._prepare_sample_rows(table.sample_rows)
        prompt = f"Table: {table.name}\nColumns: {columns_str}\nSample rows:\n{sample_str}\n\n{json_instruction}"
        
        # If too long, try with fewer samples
        if len(prompt) > MAX_TOTAL_PROMPT_LENGTH:
            sample_str = self._prepare_sample_rows(table.sample_rows, max_rows=2)
            prompt = f"Table: {table.name}\nColumns: {columns_str}\nSample rows:\n{sample_str}\n\n{json_instruction}"
            
            # If still too long, remove samples entirely
            if len(prompt) > MAX_TOTAL_PROMPT_LENGTH:
                prompt = f"Table: {table.name}\nColumns: {columns_str}\n\n{json_instruction}"
        
        return prompt

    def _extract_json_payload(self, content: str) -> Optional[str]:
        """
        Best-effort extraction of JSON from LLM output that may include code fences
        or extra text. Returns None if nothing usable is found.
        """
        if not content:
            return None

        cleaned = content.strip()
        
        # Handle code fences - remove opening and closing fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines:
                cleaned = "\n".join(lines[1:]).rstrip()
            # Remove closing fence if present
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            elif cleaned.endswith("`"):
                cleaned = cleaned.rstrip("`")
            cleaned = cleaned.strip()
        
        # Find JSON object boundaries
        start = cleaned.find("{")
        if start == -1:
            return None
        
        # Find the last closing brace
        end = cleaned.rfind("}")
        if end != -1 and end > start:
            return cleaned[start:end + 1]
        
        # Incomplete JSON - extract from { to end and let parser try to fix it
        if ":" in cleaned[start:]:
            return cleaned[start:].rstrip()
        
        return None

    def _parse_json_response(self, table_name: str, content: str) -> dict[str, str]:
        payload = self._extract_json_payload(content)
        if not payload:
            logger.warning(
                "LLM returned no JSON payload for %s. Raw response (first 500 chars): %s",
                table_name,
                content[:500] if content else "(empty)"
            )
            if content:
                logger.debug("Full LLM response for %s (length: %d): %s", table_name, len(content), content)
            return {}

        # Try multiple parsing strategies
        parse_attempts = [
            payload,  # Try as-is first
            payload.rstrip().rstrip(",").rstrip() + "}",  # Remove trailing comma, add closing brace
            payload.rstrip() + "}",  # Just add closing brace
        ]
        
        for attempt in parse_attempts:
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    if attempt != payload:
                        logger.info("Successfully parsed JSON for %s after applying fix", table_name)
                    return {str(k): str(v) for k, v in parsed.items()}
            except json.JSONDecodeError:
                continue
        
        # All attempts failed
        logger.warning(
            "LLM returned invalid JSON for %s. Extracted payload (first 500 chars): %s",
            table_name,
            payload[:500]
        )
        logger.debug("Full extracted payload for %s (length: %d): %s", table_name, len(payload), payload)
        return {}

    def _is_premium_plan(self, limits: dict) -> bool:
        """True if client has premium (column_limit >= 100 or unlimited)."""
        cl = limits.get("column_limit")
        if cl is None:
            return True
        return isinstance(cl, int) and cl >= PREMIUM_COLUMN_LIMIT_THRESHOLD

    async def _describe_columns(
        self, table: TableMetadata, max_tokens_override: Optional[int] = None
    ) -> tuple[dict[str, str], dict | None]:
        """Generate column descriptions with automatic input size management.

        max_tokens_override: If set (e.g. for premium plans), use this instead of MAX_OUTPUT_TOKENS.

        Returns:
            Tuple of (descriptions_dict, usage_info) where usage_info contains token usage if available
        """
        max_tokens = max_tokens_override if max_tokens_override is not None else MAX_OUTPUT_TOKENS
        user_prompt = self._build_user_prompt(table)

        result = await self.llm.generate_completion(
            system_prompt=DATA_DESC_SYSTEM_PROMPT,
            user_message=user_prompt,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        if result["error"]:
            # Check if it's a length-related error and try fallback
            error_str = str(result["error"]).lower()
            if "reduce the length" in error_str or "too long" in error_str or "maximum context length" in error_str:
                logger.warning(f"Input too long for {table.name}, trying fallback with columns only")
                fallback_prompt = (
                    f"Table: {table.name}\n"
                    f"Columns: {self._prepare_columns_list(table.columns)}\n\n"
                    "Return ONLY a valid JSON object in this exact format: {\"column_name\": \"description\", ...}. "
                    "Do not use markdown code fences or add any other text."
                )
                result = await self.llm.generate_completion(
                    system_prompt=DATA_DESC_SYSTEM_PROMPT,
                    user_message=fallback_prompt,
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
            else:
                raise RuntimeError(f"LLM error while describing {table.name}: {result['error']}")

        descriptions = self._parse_json_response(table.name, result.get("content", ""))
        return descriptions, result.get("usage")

    async def _write_table_file(self, table: TableMetadata, descriptions: dict[str, str]) -> None:
        """
        Write the description XML file using the exact table name from TableMetadata,
        matching the name in table_introductions.xml.
        """
        table_dir = self.data_sources_base / "data_descriptions"
        table_dir.mkdir(parents=True, exist_ok=True)
        # Use table.name directly for filename (no prefix/suffix logic)
        file_path = table_dir / f"{table.name}_description.xml"

        xml_lines = [
            "<?xml version='1.0' encoding='utf-8'?>",
            f'<data_description table_name="{table.name}">',
            f'  <table_info name="{table.name}" total_columns="{len(table.columns)}" />',
            "  <columns>",
        ]

        for column in table.columns:
            desc = descriptions.get(column.name)
            if not desc:
                desc = "Column present in schema; description not generated."
            xml_lines.append(f'    <column name="{column.name}" data_type="{column.data_type}">')
            xml_lines.append(f"      <description>{escape(desc.strip())}</description>")
            xml_lines.append("    </column>")

        xml_lines.extend(["  </columns>", "</data_description>"])
        content = "\n".join(xml_lines)
        await asyncio.to_thread(file_path.write_text, content, "utf-8")

    async def generate(self, tables: Iterable[TableMetadata], on_table_done: Optional[Callable[[], Awaitable[None]]] = None) -> dict:
        """Generate data descriptions for all tables, continuing even if some fail.
        Uses parallel LLM calls (limited by MAX_CONCURRENT_LLM_CALLS) to reduce total time.
        Premium plans (column_limit >= 100 or unlimited) get a high output token cap so all
        columns are described and saved in xml_prompts/clients/<client_id>/data_sources/data_descriptions.
        Returns:
            Dict with 'total_token_usage' aggregated across all table descriptions
        """
        tables_list = list(tables) if not isinstance(tables, list) else tables
        total_tables = len(tables_list)
        # Premium plans: no effective output token cap so all columns get descriptions
        limits = await get_explorer_limits(self.client_id, db=self.db)
        effective_max_tokens = (
            UNLIMITED_OUTPUT_TOKENS if self._is_premium_plan(limits) else MAX_OUTPUT_TOKENS
        )
        if effective_max_tokens == UNLIMITED_OUTPUT_TOKENS:
            logger.info(
                f"Premium plan detected (column_limit={limits.get('column_limit')}); "
                f"using high output token cap ({UNLIMITED_OUTPUT_TOKENS}) for full column descriptions"
            )
        # Track token usage across all tables
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        
        logger.info(f"Starting data description generation for {total_tables} tables (parallel, max {MAX_CONCURRENT_LLM_CALLS} concurrent)")

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

        async def process_one(table: TableMetadata) -> tuple[TableMetadata, dict[str, str] | None, dict | None]:
            """Process one table with semaphore. Returns (table, descriptions, usage_info) or (table, None, None) on error."""
            async with semaphore:
                try:
                    descriptions, usage_info = await self._describe_columns(table, max_tokens_override=effective_max_tokens)
                    return (table, descriptions, usage_info)
                except Exception as e:
                    logger.error(
                        f"Failed to generate description for table {table.name}: {e}",
                        exc_info=True
                    )
                    return (table, None, None)
                finally:
                    if on_table_done is not None:
                        try:
                            await on_table_done()
                        except Exception:
                            pass

        results = await asyncio.gather(
            *[process_one(table) for table in tables_list],
            return_exceptions=False
        )

        successful = 0
        failed = 0
        for table, descriptions, usage_info in results:
            if descriptions is not None:
                await self._write_table_file(table, descriptions)
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
        
        logger.info(
            f"Data description generation complete: {successful} successful, {failed} failed out of {total_tables} total"
        )
        
        if failed > 0:
            logger.warning(
                f"{failed} table(s) failed to generate descriptions. "
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

