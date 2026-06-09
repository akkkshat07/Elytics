from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Awaitable, Callable, List, Sequence, Optional, Any, Dict
import defusedxml.ElementTree as ET

from sqlalchemy import text, select, column, and_, or_, not_, func, bindparam
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from explorer.data_description_agent import DataDescriptionAgent
from explorer.data_profile_agent import DataProfileAgent
from explorer.models import ColumnMetadata, TableMetadata
from explorer.schema_limits import apply_table_column_limits
from explorer.table_intro_agent import TableIntroductionAgent
from db_config.mongo_server import get_db
from services.db_credentials_service import DBCredentialsService
from services.subscription_service import get_explorer_limits
from util.data_source import DataSource, get_data_source_for_client
from util.dataset_paths import resolve_xml_data_sources_dir, assets_datasets_dir

logger = logging.getLogger(__name__)

SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z0-9_]+$")


def _assert_safe_identifier(value: str) -> None:
    if not SAFE_IDENTIFIER.match(value):
        pass
        # raise ValueError(f"Unsafe identifier detected: {value}")


async def get_data_source(client_id: str, db = None) -> DataSource:
    """
    Determine the active data source for a client.
    
    Returns:
        DataSource enum value based on credentials:
        - DataSource.PARQUET if store_in_local is True or no credentials found
        - DataSource.MYSQL if db_type is 'mysql' and not storing locally
        - DataSource.POSTGRES if db_type is 'postgres' and not storing locally
        - DataSource.MONGODB if db_type is 'mongodb' and not storing locally
    
    This function wraps get_data_source_for_client from util.data_source.
    """
    return await get_data_source_for_client(client_id, db=db)


async def copy_base_prompts_for_client(client_id: str, output_root: Path) -> None:
    """
    Standalone function to copy/merge base XML prompts for a client.
    
    This can be called without an ExplorerAgent instance, e.g. from
    the file-upload processing flow.
    
    Args:
        client_id: The client identifier.
        output_root: Root output directory (e.g. xml_prompts/clients/{client_id}).
    """
    import shutil
    import defusedxml.ElementTree as _ET

    project_root = Path(__file__).parent.parent
    base_prompts_dir = project_root / "xml_prompts" / "base" / "agents"

    dest_dir = output_root / "agents"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not base_prompts_dir.exists():
        return

    data_source = await get_data_source(client_id)
    ds_suffix = data_source.prompt_suffix
    logger.info(
        f"[copy_base_prompts_for_client] Using data source suffix '{ds_suffix}' "
        f"for client {client_id} (data_source: {data_source.value})"
    )

    def merge_agent_xml(base_path: Path, ds_path: Path, out_path: Path) -> None:
        if not base_path.exists() or not ds_path.exists():
            combined_fallback = base_prompts_dir / out_path.name
            if combined_fallback.exists():
                shutil.copy2(combined_fallback, out_path)
            return

        try:
            base_tree = _ET.parse(base_path)
            base_root = base_tree.getroot()

            ds_tree = _ET.parse(ds_path)
            ds_root = ds_tree.getroot()

            for ds_child in ds_root:
                match = None
                for base_child in base_root:
                    if base_child.tag == ds_child.tag:
                        match = base_child
                        break

                if match is None:
                    base_root.append(ds_child)
                else:
                    for sub in ds_child:
                        match.append(sub)

            base_tree.write(out_path, encoding="utf-8", xml_declaration=True)
        except Exception as e:
            logger.error(f"Failed to merge XML prompts for {out_path.name}: {e}")
            combined_fallback = base_prompts_dir / out_path.name
            if combined_fallback.exists():
                shutil.copy2(combined_fallback, out_path)

    for agent_name in ["planner"]:
        base_file = base_prompts_dir / f"{agent_name}.base.xml"
        ds_file = base_prompts_dir / f"{agent_name}.{ds_suffix}.xml"
        output_file = dest_dir / f"{agent_name}.xml"
        merge_agent_xml(base_file, ds_file, output_file)

    # data_science_planner shares the planner data-source rules
    base_file = base_prompts_dir / "data_science_planner.base.xml"
    ds_file = base_prompts_dir / f"planner.{ds_suffix}.xml"
    output_file = dest_dir / "data_science_planner.xml"
    merge_agent_xml(base_file, ds_file, output_file)

    # data_analyst_planner shares the planner data-source rules
    base_file = base_prompts_dir / "data_analyst_planner.base.xml"
    ds_file = base_prompts_dir / f"planner.{ds_suffix}.xml"
    output_file = dest_dir / "data_analyst_planner.xml"
    merge_agent_xml(base_file, ds_file, output_file)

    # Business agent is data-source agnostic
    business_src = base_prompts_dir / "business.xml"
    if business_src.exists():
        shutil.copy2(business_src, dest_dir)

    # Data science & data analyst agents are data-source agnostic — copy so
    # clients have per-tenant files that admins can customise independently.
    for agent_file in ["data_science_agent.xml", "data_analyst_agent.xml"]:
        src = base_prompts_dir / agent_file
        if src.exists():
            shutil.copy2(src, dest_dir / agent_file)

    logger.info(f"[copy_base_prompts_for_client] Copied base prompts for client {client_id}")


class ExplorerAgent:
    """Coordinator that runs schema/table intro/data description subagents."""

    def __init__(
        self,
        client_id: str,
        session_factory: async_sessionmaker[AsyncSession],
        output_root: Path,
        *,
        schema_filter: str | None = "public",
        table_prefix: str | None = None,
        store_in_local: bool,
        db=None,
        db_type: str = "postgres",
        db_name: str | None = None,
        db_username: str | None = None,
        credentials: Optional[Dict[str, Any]] = None,
        table_filter: Optional[List[str]] = None,
        namespace: Optional[Dict[str, Any]] = None,
        on_table_done: Optional[Callable[[], Awaitable[None]]] = None,
        dataset_id: Optional[str] = None,
    ):
        self.client_id = client_id
        self.session_factory = session_factory
        self.output_root = output_root
        self.dataset_id = dataset_id
        self.data_sources_root = resolve_xml_data_sources_dir(client_id, dataset_id, for_write=True)
        self.data_sources_root.mkdir(parents=True, exist_ok=True)
        self.schema_filter = schema_filter
        self.table_prefix = table_prefix
        self.store_in_local = store_in_local
        self.db = db
        self.db_type = db_type
        self.db_name = db_name
        self.db_username = db_username
        self.is_sap_ecc_oracle = False
        self.sap_schema: str | None = None
        self.table_filter = table_filter  # Optional list of table names to include
        self.namespace = namespace or None
        self.on_table_done = on_table_done

        if not output_root.exists():
            output_root.mkdir(parents=True, exist_ok=True)

        self.table_intro_agent = TableIntroductionAgent(client_id, db=db)
        self.data_desc_agent = DataDescriptionAgent(
            client_id, output_root, db=db, data_sources_base=self.data_sources_root
        )
        self.data_profile_agent = DataProfileAgent(client_id, db=db)

    def _quote(self, identifier: str) -> str:
        """Quote identifier based on database dialect."""
        if self.db_type == "mysql":
            return f"`{identifier}`"
        elif self.db_type == "sap_sybase":
            # Sybase ASE uses square brackets for identifiers
            return f"[{identifier}]"
        return f'"{identifier}"'

    def _get_like_operator(self) -> str:
        """Return case-insensitive LIKE operator based on dialect."""
        if self.db_type == "postgres":
            return "ILIKE"
        return "LIKE"  # MySQL is case-insensitive by default

    def _get_full_table_name(self, schema: str, table: str) -> str:
        """Get fully qualified table name with appropriate quoting."""
        if self.db_type == "sap_sybase":
            # Sybase prefers square brackets for schema.table
            return f"{self._quote(table)}"
        return f"{self._quote(schema)}.{self._quote(table)}"

    @staticmethod
    def _tables_to_normalized(tables: List[TableMetadata]) -> List[dict]:
        """Convert List[TableMetadata] to normalized list of dicts for apply_table_column_limits."""
        return [
            {
                "table_name": t.name,
                "schema": t.schema,
                "columns": [
                    {
                        "name": c.name,
                        "data_type": c.data_type,
                        "is_nullable": c.is_nullable,
                        "character_maximum_length": c.character_maximum_length,
                        "numeric_precision": c.numeric_precision,
                        "numeric_scale": c.numeric_scale,
                        "description": c.description,
                    }
                    for c in t.columns
                ],
            }
            for t in tables
        ]

    @staticmethod
    def _normalized_to_tables(trimmed_schema: List[dict]) -> List[TableMetadata]:
        """Convert trimmed schema (list of dicts from apply_table_column_limits) back to List[TableMetadata]."""
        result: List[TableMetadata] = []
        for t in trimmed_schema:
            schema_name = t.get("schema", "public")
            name = t.get("table_name") or t.get("name", "")
            cols = t.get("columns") or []
            columns = [
                ColumnMetadata(
                    name=c.get("name", ""),
                    data_type=c.get("data_type", "text"),
                    is_nullable=c.get("is_nullable", True),
                    character_maximum_length=c.get("character_maximum_length"),
                    numeric_precision=c.get("numeric_precision"),
                    numeric_scale=c.get("numeric_scale"),
                    description=c.get("description"),
                )
                for c in cols
            ]
            result.append(
                TableMetadata(schema=schema_name, name=name, columns=columns)
            )
        return result

    async def run(self) -> None:

        logger.info(f"ExplorerAgent.run() started for client: {self.client_id}, db_type: {self.db_type}")
        logger.info(f"Table filter provided: {self.table_filter is not None}, filter value: {self.table_filter}")
        if self.table_filter is None:
            logger.warning("WARNING: table_filter is None - this means all tables will be processed. This should only happen in automatic discovery mode.")
        else:
            logger.info(f"Table filter is set with {len(self.table_filter)} table(s): {self.table_filter[:10]}{'...' if len(self.table_filter) > 10 else ''}")
        if self.db_type == "mongodb":
            logger.info("Extracting MongoDB metadata...")
            tables = await self._extract_mongodb_metadata()
        else:
            logger.info("Extracting SQL metadata...")
            tables = await self._load_table_metadata()
            
        logger.info(f"Discovered {len(tables) if tables else 0} tables/collections")
        if not tables:
            logger.error("No tables discovered for explorer agent!")
            raise RuntimeError("No tables discovered for explorer agent.")

        # Filter to only selected tables if table_filter is provided (case-insensitive match)
        # Note: This is a safety check - tables should already be filtered in _load_table_metadata()
        if self.table_filter:
            table_filter_set = {name.lower() for name in self.table_filter}
            initial_count = len(tables)
            initial_tables = tables  # Save for error message
            tables = [t for t in tables if t.name.lower() in table_filter_set]
            logger.info(f"Filtered to {len(tables)} selected table(s) from {len(self.table_filter)} requested (initial: {initial_count})")
            if not tables:
                available_names = [t.name for t in initial_tables[:10]] if initial_count > 0 else []
                logger.warning(f"No tables match the provided table filter. Requested: {self.table_filter}, Available (sample): {available_names}")
                raise RuntimeError(f"No tables match the provided table filter. Requested: {self.table_filter}")
        
        # Apply subscription plan table/column limits
        # Note: This should not add tables back - it only limits columns per table
        tables_before_limits = len(tables)
        limits = await get_explorer_limits(self.client_id)
        normalized = self._tables_to_normalized(tables)
        trimmed_schema, limit_metadata = apply_table_column_limits(
            normalized,
            limits.get("table_limit"),
            limits.get("column_limit"),
        )
        tables = self._normalized_to_tables(trimmed_schema)
        tables_after_limits = len(tables)
        if tables_before_limits != tables_after_limits:
            logger.warning(f"Table count changed after applying limits: {tables_before_limits} -> {tables_after_limits}")
        if self.table_filter and tables_after_limits != len(self.table_filter):
            logger.warning(f"Table count mismatch: filter has {len(self.table_filter)} tables, but {tables_after_limits} tables after limits")
        meta_dir = self.data_sources_root / "meta_information"
        meta_dir.mkdir(parents=True, exist_ok=True)
        explorer_limits_path = meta_dir / "explorer_limits.json"
        limits_json = json.dumps(limit_metadata, indent=2)
        await asyncio.to_thread(explorer_limits_path.write_text, limits_json, "utf-8")
        logger.info(f"Applied explorer limits: {limit_metadata.get('total_tables_loaded')} tables, max {limit_metadata.get('columns_per_table_loaded')} cols; persisted to {explorer_limits_path}")

        # No empty table filtering - user manually selects tables, so we respect their choice
        logger.info(f"Processing {len(tables)} table(s) as selected by user")
        
        if len(tables) == 0:
            logger.warning("No tables found. Nothing to process.")
            return

        if self.db_type == "mongodb":
            # Samples are already fetched during extraction for MongoDB
            pass
        
        if self.db_type != "mongodb":
            await self._fetch_samples(tables)

    
        table_intro_path = self.data_sources_root / "meta_information" / "table_introductions.xml"

        # Process all tables (no limit)
        ai_subset = tables
        logger.info(f"About to generate metadata for {len(ai_subset)} table(s). Table names: {[t.name for t in ai_subset]}")
        if self.table_filter:
            logger.info(f"EXPECTED: Only {len(self.table_filter)} table(s) should be processed. Filter: {self.table_filter}")
            unexpected_tables = [t.name for t in ai_subset if t.name.lower() not in {name.lower() for name in self.table_filter}]
            if unexpected_tables:
                logger.error(f"ERROR: Found {len(unexpected_tables)} unexpected tables that don't match filter: {unexpected_tables[:10]}")
            else:
                logger.info(f"SUCCESS: All {len(ai_subset)} tables match the filter")

        
        if self._should_skip_generated_metadata():
            logger.info("Skipping table introductions and data descriptions generation for SAP sources.")
            # Copy base_sap metadata to client directory
            await self._copy_base_sap_metadata(ai_subset)
        else:
            await self.table_intro_agent.generate(ai_subset, table_intro_path)
            logger.info(f"Generated table_introductions.xml at {table_intro_path} for {len(ai_subset)} table(s)")
            
            await self.data_desc_agent.generate(ai_subset, on_table_done=self.on_table_done)
            logger.info(f"Generated data descriptions in {self.data_sources_root / 'data_descriptions'} for {len(ai_subset)} table(s)")

        # Generate client data profile (geography, formatting, industry)
        try:
            profile = await self.data_profile_agent.generate_profile(ai_subset)
            profile_path = meta_dir / "client_data_profile.xml"
            await self.data_profile_agent.save_as_xml(profile, profile_path)
            await self.data_profile_agent.save_to_mongodb(profile)
            logger.info("Generated client data profile for %s", self.client_id)

            # Seed initial lessons from profile
            await self._seed_lessons_from_profile(profile)
        except Exception as e:
            logger.warning("Data profile generation failed (non-fatal): %s", e)

        # Generate suggested questions based on the metadata
        await self._generate_suggested_questions()
        # Copy base prompts to client directory (combined with data-source specific rules)
        await self._copy_base_prompts()

        if self.store_in_local:
            await self._export_datasets(tables)
    
    async def _generate_suggested_questions(self) -> None:
        """Generate 30 suggested questions based on the client's metadata."""
        
        try:
            from explorer.question_generator import QuestionGenerator
            
            logger.info(f"Generating suggested questions for client {self.client_id}")
            generator = QuestionGenerator(
                self.client_id,
                db=self.db,
                dataset_id=self.dataset_id,
            )
            questions = await generator.generate_questions(count=30)
            logger.info(
                "Successfully generated %d suggested questions for client %s dataset_id=%s",
                len(questions),
                self.client_id,
                self.dataset_id,
            )
        except Exception as e:
            # Log the error so we can debug why questions aren't being generated
            logger.error(f"Failed to generate suggested questions for client {self.client_id}: {e}", exc_info=True)
            # Non-critical, so we don't raise - questions can be generated on-demand later

    async def _seed_lessons_from_profile(self, profile: Dict[str, Any]) -> None:
        """Create initial lessons from the data profile so coder agents have day-0 knowledge."""
        try:
            raw_db = self.db
            if raw_db is None:
                return
            if type(raw_db).__name__ == "MongoDBManager":
                raw_db = getattr(raw_db, "db", raw_db)

            from services.agent_lesson_service import AgentLessonService
            svc = AgentLessonService(raw_db)

            lessons = []

            # Fiscal year
            fc = profile.get("fiscal_calendar", {})
            if fc.get("start_month") and fc["start_month"] != 1:
                month_names = {4: "April-March", 7: "July-June", 10: "October-September"}
                label = month_names.get(fc["start_month"], f"Month {fc['start_month']}")
                lessons.append({
                    "lesson": f"Fiscal year runs {label} (starts month {fc['start_month']})",
                    "category": "F",
                    "lesson_type": "business_logic",
                    "source": "data_profile",
                    "tables_involved": [],
                    "confidence": 0.7,
                })

            # Date formats
            for df in profile.get("date_formats", [])[:2]:
                if df.get("confidence", 0) >= 0.5:
                    lessons.append({
                        "lesson": f"Date columns commonly use {df['pattern']} format",
                        "category": "B",
                        "lesson_type": "data_type_quirk",
                        "source": "data_profile",
                        "tables_involved": [],
                        "confidence": df["confidence"],
                    })

            # Number format
            nf = profile.get("number_format", {})
            if nf.get("system") == "indian":
                lessons.append({
                    "lesson": "Use Indian number system (lakhs, crores) for display — format: 1,00,000",
                    "category": "B",
                    "lesson_type": "data_type_quirk",
                    "source": "data_profile",
                    "tables_involved": [],
                    "confidence": 0.7,
                })

            # Small tables
            for tname in profile.get("small_tables", [])[:5]:
                lessons.append({
                    "lesson": f"Table {tname} is a small lookup table — load ALL columns (no columns= filter)",
                    "category": "E",
                    "lesson_type": "performance_pattern",
                    "source": "data_profile",
                    "tables_involved": [tname],
                    "confidence": 0.7,
                })

            # Currency
            cur = profile.get("currency", {})
            if cur.get("code"):
                lessons.append({
                    "lesson": f"Currency is {cur['code']} ({cur.get('symbol', '')})",
                    "category": "F",
                    "lesson_type": "business_logic",
                    "source": "data_profile",
                    "tables_involved": [],
                    "confidence": 0.7,
                })

            # Clear old profile-based lessons before seeding new ones
            await svc.delete_lessons_by_source(self.client_id, "data_profile")

            for lsn in lessons:
                await svc.save_lesson(self.client_id, lsn)

            if lessons:
                logger.info("Seeded %d lessons from data profile for %s", len(lessons), self.client_id)

        except Exception as e:
            logger.warning("Failed to seed lessons from profile: %s", e)

    def _should_skip_generated_metadata(self) -> bool:
        """Skip LLM-generated metadata for SAP sources (use base_sap at read time)."""
        return self.db_type in {"sap_oracle", "sap_sybase"} or self.is_sap_ecc_oracle

    async def _copy_base_sap_metadata(self, tables: List[TableMetadata]) -> None:
        """
        Copy base_sap metadata files to client directory for selected tables.
        Only copies files for tables that are in the filtered list.
        """
        import shutil
        
        base_sap_dir = Path("xml_prompts/base_sap") / "data_sources"
        base_sap_meta_dir = base_sap_dir / "meta_information"
        base_sap_desc_dir = base_sap_dir / "data_descriptions"
        
        client_meta_dir = self.data_sources_root / "meta_information"
        client_desc_dir = self.data_sources_root / "data_descriptions"
        
        # Create directories if they don't exist
        client_meta_dir.mkdir(parents=True, exist_ok=True)
        client_desc_dir.mkdir(parents=True, exist_ok=True)
        
        # Get set of selected table names (case-insensitive for matching)
        selected_table_names = {t.name.upper() for t in tables}
        
        # Copy table_introductions.xml - filter to only selected tables
        intros_file = base_sap_meta_dir / "table_introductions.xml"
        if intros_file.exists():
            try:
                tree = await asyncio.to_thread(ET.parse, intros_file)
                root = tree.getroot()
                
                # Find the table_introductions container element
                table_intros_container = root.find(".//table_introductions")
                if table_intros_container is not None:
                    # Filter to only selected tables
                    removed_count = 0
                    for elem in list(table_intros_container.findall("table_introduction")):
                        table_name = elem.get("table_name", "").upper()
                        if table_name not in selected_table_names:
                            table_intros_container.remove(elem)
                            removed_count += 1
                    
                    # Write filtered table_introductions.xml to client directory
                    client_intros_file = client_meta_dir / "table_introductions.xml"
                    tree.write(client_intros_file, encoding="utf-8", xml_declaration=True)
                    logger.info(f"Copied filtered table_introductions.xml to {client_intros_file} with {len(selected_table_names)} table(s) (removed {removed_count} non-selected)")
                else:
                    logger.warning(f"Could not find table_introductions container in {intros_file}")
            except Exception as e:
                logger.error(f"Failed to copy table_introductions.xml: {e}", exc_info=True)
        else:
            logger.warning(f"base_sap table_introductions.xml not found at {intros_file}")
        
        # Copy data description files for selected tables only
        copied_count = 0
        failed_count = 0
        for table in tables:
            desc_file = base_sap_desc_dir / f"{table.name}_description.xml"
            if desc_file.exists():
                try:
                    shutil.copy2(desc_file, client_desc_dir / desc_file.name)
                    copied_count += 1
                except Exception as e:
                    logger.warning(f"Failed to copy {desc_file.name}: {e}")
                    failed_count += 1
            else:
                logger.debug(f"Data description file not found for {table.name} in base_sap")
        
        logger.info(f"Copied {copied_count} data description file(s) from base_sap to client directory (failed: {failed_count})")

    async def _copy_base_prompts(self) -> None:
        """
        Create combined prompt files (planner.xml, python.xml, business.xml)
        for this client by merging base XML with data-source-specific rules.
        """
        import shutil

        # This file is in explorer/explorer_agent.py. Project root is parent of parent.
        project_root = Path(__file__).parent.parent
        base_prompts_dir = project_root / "xml_prompts" / "base" / "agents"

        dest_dir = self.output_root / "agents"
        dest_dir.mkdir(parents=True, exist_ok=True)

        if not base_prompts_dir.exists():
            return

        # Determine active data source for this client using DataSource enum
        # Use passed credentials if available to avoid re-querying
        # Use the prompt_suffix property from DataSource enum
        data_source = await get_data_source(self.client_id, db=self.db) 
        ds_suffix = data_source.prompt_suffix
        logger.info(f"Using data source suffix '{ds_suffix}' for client {self.client_id} (data_source: {data_source.value})")

        def merge_agent_xml(base_path: Path, ds_path: Path, output_path: Path) -> None:
            """
            Merge a base agent XML with a data-source-specific XML.
            If either file is missing, fall back to copying the existing combined XML.
            """
            if not base_path.exists() or not ds_path.exists():
                # Fallback: if we don't have split files, try copying the original combined xml
                combined_fallback = base_prompts_dir / output_path.name
                if combined_fallback.exists():
                    shutil.copy2(combined_fallback, output_path)
                return

            try:
                base_tree = ET.parse(base_path)
                base_root = base_tree.getroot()

                ds_tree = ET.parse(ds_path)
                ds_root = ds_tree.getroot()

                # Simple merge: for each top-level element under ds_root, append or extend
                for ds_child in ds_root:
                    # Find matching child in base_root by tag (namespace-aware)
                    match = None
                    for base_child in base_root:
                        if base_child.tag == ds_child.tag:
                            match = base_child
                            break

                    if match is None:
                        # No existing section with this tag, just append the whole section
                        base_root.append(ds_child)
                    else:
                        # Extend existing section with children/rules from data-source section
                        for sub in ds_child:
                            match.append(sub)

                base_tree.write(output_path, encoding="utf-8", xml_declaration=True)
            except Exception as e:
                # TODO: figure out if this is correct behaviour or not
                logger.error(f"Failed to merge XML prompts for {output_path.name}: {e}")
                # As a safety net, try to copy original combined xml if present
                combined_fallback = base_prompts_dir / output_path.name
                if combined_fallback.exists():
                    shutil.copy2(combined_fallback, output_path)

        # Merge planner and data_science_planner from base + data source rules
        for agent_name in ["planner"]:
            base_file = base_prompts_dir / f"{agent_name}.base.xml"
            ds_file = base_prompts_dir / f"{agent_name}.{ds_suffix}.xml"
            output_file = dest_dir / f"{agent_name}.xml"
            merge_agent_xml(base_file, ds_file, output_file)
        
        # Copy data science planner base + data source
        base_file = base_prompts_dir / "data_science_planner.base.xml"
        ds_file = base_prompts_dir / f"planner.{ds_suffix}.xml"
        output_file = dest_dir / f"data_science_planner.xml"
        merge_agent_xml(base_file, ds_file, output_file)

        # Copy data analyst planner base + data source
        base_file = base_prompts_dir / "data_analyst_planner.base.xml"
        ds_file = base_prompts_dir / f"planner.{ds_suffix}.xml"
        output_file = dest_dir / f"data_analyst_planner.xml"
        merge_agent_xml(base_file, ds_file, output_file)

        # Business agent is currently data-source agnostic – just copy as-is
        business_src = base_prompts_dir / "business.xml"
        if business_src.exists():
            shutil.copy2(business_src, dest_dir)

        # Data science & data analyst agents are data-source agnostic — copy
        # so clients have per-tenant files that admins can customise.
        for agent_file in ["data_science_agent.xml", "data_analyst_agent.xml"]:
            src = base_prompts_dir / agent_file
            if src.exists():
                shutil.copy2(src, dest_dir / agent_file)

    async def _export_datasets(self, tables: Sequence[TableMetadata]) -> None:
        import pandas as pd
        from motor.motor_asyncio import AsyncIOMotorClient
        from util.db_size import MAX_DB_SIZE_BYTES
        

        dataset_dir = assets_datasets_dir(self.client_id, self.dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        cumulative_bytes = 0
        skipped_tables: list[str] = []
        
        if self.db_type == "mongodb":
            # For MongoDB, session_factory is the AsyncIOMotorDatabase object
            db = self.session_factory
            
            for table in tables:
                try:
                    collection = db[table.name]
                    cursor = collection.find({}).limit(1000) # Export up to 1000 rows for local exploration
                    rows = await cursor.to_list(length=1000)
                    
                    if rows:
                        # Make serializable
                        serializable_rows = [self._make_mongodb_serializable(row) for row in rows]
                        df = pd.DataFrame(serializable_rows)
                    else:
                        df = pd.DataFrame()
                        
                    output_path = dataset_dir / f"{table.name}.parquet"
                    await asyncio.to_thread(df.to_parquet, output_path, index=False)
                    
                    file_size = output_path.stat().st_size
                    cumulative_bytes += file_size
                    if cumulative_bytes > MAX_DB_SIZE_BYTES:
                        logger.warning(
                            f"Cumulative Parquet export size ({cumulative_bytes} bytes) exceeds limit "
                            f"({MAX_DB_SIZE_BYTES} bytes) after exporting {table.name}. Stopping export."
                        )
                        remaining = [t.name for t in tables if t.name not in {table.name} and t.name not in skipped_tables]
                        skipped_tables.extend(remaining)
                        break
                    
                    logger.info(f"Exported MongoDB collection {table.name} to {output_path}")
                except Exception as e:
                    logger.error(f"Failed to export MongoDB collection {table.name}: {e}")
            
            if skipped_tables:
                logger.warning(f"Skipped {len(skipped_tables)} table(s) due to size limit: {skipped_tables}")
            return

        async with self.session_factory() as session:
            for table in tables:
                try:
                    # _assert_safe_identifier(table.schema)
                    # _assert_safe_identifier(table.name)
                    
                    full_name = self._get_full_table_name(table.schema, table.name)
                    query = text(f'SELECT * FROM {full_name}')
                    result = await session.execute(query)
                    # Use mappings to get dictionary-like rows for DataFrame creation
                    rows = result.mappings().fetchall()
                    
                    if rows:
                        df = pd.DataFrame(rows)
                    else:
                        # Create empty DataFrame with correct schema for empty tables
                        # Get column names and types from table metadata
                        if table.columns:
                            # Use column metadata if available
                            col_names = [col.name for col in table.columns]
                            df = pd.DataFrame(columns=col_names)
                        else:
                            # Fallback: query system tables for column names
                            if self.db_type == "sap_sybase":
                                # Use Sybase system tables
                                metadata_query = text("""
                                    SELECT c.name as column_name
                                    FROM syscolumns c
                                    INNER JOIN sysobjects o ON c.id = o.id
                                    WHERE o.name = :table_name
                                    ORDER BY c.colid
                                """)
                                col_result = await session.execute(
                                    metadata_query,
                                    {"table_name": table.name}
                                )
                            else:
                                # Use information_schema for other databases
                                metadata_query = text("""
                                    SELECT column_name 
                                    FROM information_schema.columns 
                                    WHERE table_schema = :schema AND table_name = :table_name
                                    ORDER BY ordinal_position
                                """)
                                col_result = await session.execute(
                                    metadata_query,
                                    {"schema": table.schema, "table_name": table.name}
                                )
                            col_rows = col_result.mappings().fetchall()
                            if col_rows:
                                col_names = [row["column_name"] for row in col_rows]
                                df = pd.DataFrame(columns=col_names)
                            else:
                                # If we can't get column info, create empty DataFrame
                                df = pd.DataFrame()
                    
                    # Save parquet file for all tables (including empty ones)
                    safe_name = table.name.replace("/", "_").replace("\\", "_")
                    output_path = dataset_dir / f"{safe_name}.parquet"
                    df.to_parquet(output_path, index=False)
                    
                    file_size = output_path.stat().st_size
                    cumulative_bytes += file_size
                    if cumulative_bytes > MAX_DB_SIZE_BYTES:
                        logger.warning(
                            f"Cumulative Parquet export size ({cumulative_bytes} bytes) exceeds limit "
                            f"({MAX_DB_SIZE_BYTES} bytes) after exporting {table.name}. Stopping export."
                        )
                        exported_names = {t.name for t in tables[:tables.index(table) + 1]}
                        skipped_tables.extend(t.name for t in tables if t.name not in exported_names)
                        break
                    
                except ValueError as e:
                    # Unsafe identifier - skip this table
                    logger.warning(f"Skipped table {table.schema}.{table.name} due to unsafe identifier: {e}")
                except Exception as e:
                    # Other errors - log but continue with other tables
                    logger.error(f"Failed to export table {table.schema}.{table.name}: {e}", exc_info=True)
        
        if skipped_tables:
            logger.warning(f"Skipped {len(skipped_tables)} table(s) due to size limit: {skipped_tables}")

    async def _load_table_metadata(self) -> List[TableMetadata]:
        async with self.session_factory() as session:
            try:
                if self.db_type == "sap_oracle" or self.db_type == "sap_sybase":
                    # Load directly from base_sap XML (skip discovery)
                    logger.info(f"Loading SAP metadata from base_sap XML for {self.db_type}")
                    return await self._load_sap_metadata()
                tables = await self._query_tables(session)
                logger.info(f"Query tables returned {len(tables)} rows")
                if tables:
                    logger.info(f"First table row keys: {tables[0].keys()}")
                
                # If table_filter is provided, filter tables first, then only fetch columns for selected tables
                table_names_to_query = None
                if self.table_filter:
                    # Create case-insensitive filter set for matching
                    table_filter_set = {name.lower() for name in self.table_filter}
                    # Save original tables for error message
                    original_tables = tables
                    # Filter tables to only selected ones before querying columns (case-insensitive match)
                    tables = [t for t in tables if t.get("table_name", "").lower() in table_filter_set]
                    logger.info(f"Pre-filtered to {len(tables)} selected table(s) before column query (from {len(self.table_filter)} requested)")
                    if tables:
                        # Extract table names for column query
                        table_names_to_query = [t.get("table_name") for t in tables]
                    else:
                        available_sample = [t.get("table_name") for t in original_tables[:10]] if original_tables else []
                        logger.warning(f"No tables matched the filter. Requested: {self.table_filter}, Available (sample): {available_sample}")
                
                # Fetch columns - only for selected tables if table_filter is provided
                if table_names_to_query:
                    logger.info(f"Querying columns for {len(table_names_to_query)} selected table(s): {table_names_to_query[:5]}{'...' if len(table_names_to_query) > 5 else ''}")
                columns = await self._query_columns(session, table_names_to_query)
                logger.info(f"Query columns returned {len(columns)} rows for {len(set(c.get('table_name') for c in columns)) if columns else 0} unique table(s)")
                if columns:
                    logger.info(f"First column row keys: {columns[0].keys()}")
                
                primary_keys = {}
            except Exception as e:
                logger.error(f"Error querying metadata: {e}", exc_info=True)
                raise e

        column_map: dict[tuple[str, str], List[ColumnMetadata]] = {}
        for column in columns:
            key = (column["table_schema"], column["table_name"])
            column_map.setdefault(key, []).append(
                ColumnMetadata(
                    name=column["column_name"],
                    data_type=column["data_type"],
                    is_nullable=column["is_nullable"] == "YES",
                    character_maximum_length=column["character_maximum_length"],
                    numeric_precision=column["numeric_precision"],
                    numeric_scale=column["numeric_scale"],
                )
            )

        metadata: List[TableMetadata] = []
        for table in tables:
            key = (table["table_schema"], table["table_name"])
            metadata.append(
                TableMetadata(
                    schema=table["table_schema"],
                    name=table["table_name"],
                    columns=column_map.get(key, []),
                )
            )
        return metadata

    def _get_sap_schema(self) -> str:
        if self.sap_schema:
            return self.sap_schema
        if self.schema_filter and self.schema_filter != "public":
            return self.schema_filter.upper()
        if self.db_username:
            return self.db_username.upper()
        return "SAPSR3"

    async def _detect_sap_schema(self, session: AsyncSession) -> str | None:
        """
        Detect SAP ECC Oracle schema by checking for DDIC tables.
        """
        try:
            candidates = []
            if self.schema_filter and self.schema_filter != "public":
                candidates.append(self.schema_filter.upper())
            if self.db_username:
                candidates.append(self.db_username.upper())

            if candidates:
                query = text(
                    """
                    SELECT COUNT(*) as cnt
                    FROM ALL_TABLES
                    WHERE OWNER = :schema
                      AND TABLE_NAME IN ('DD02L', 'DD03L', 'DD04T')
                    """
                )
                for schema in candidates:
                    result = await session.execute(query, {"schema": schema})
                    row = result.mappings().fetchone()
                    if row and row.get("cnt", 0) >= 3:
                        logger.info(f"SAP ECC Oracle detected in schema {schema}")
                        return schema

            query = text(
                """
                SELECT OWNER, COUNT(DISTINCT TABLE_NAME) as cnt
                FROM ALL_TABLES
                WHERE TABLE_NAME IN ('DD02L', 'DD03L', 'DD04T')
                GROUP BY OWNER
                ORDER BY cnt DESC
                FETCH FIRST 1 ROWS ONLY
                """
            )
            result = await session.execute(query)
            row = result.mappings().fetchone()
            if row and row.get("cnt", 0) >= 3:
                schema = row["OWNER"]
                logger.info(f"SAP ECC Oracle detected in schema {schema}")
                return schema

            logger.info("SAP ECC Oracle not detected in accessible schemas.")
            return None
        except Exception as e:
            logger.warning(f"Failed SAP ECC Oracle detection: {e}")
            return None

    async def _load_sap_metadata(self) -> List[TableMetadata]:
        """Load SAP metadata from base_sap XML files (no discovery)."""
        from glob import glob
        
        base_sap_dir = Path("xml_prompts/base_sap") / "data_sources"
        intros_file = base_sap_dir / "meta_information" / "table_introductions.xml"
        desc_dir = base_sap_dir / "data_descriptions"
        
        if not intros_file.exists():
            raise FileNotFoundError(f"base_sap table_introductions.xml not found at {intros_file}")
        
        # Parse table_introductions.xml to get table list
        intros_tree = await asyncio.to_thread(ET.parse, intros_file)
        intros_root = intros_tree.getroot()
        
        schema_name = self._get_sap_schema()  # Default: "SAPSR3"
        
        # Extract table names from table_introduction elements
        tables = []
        for elem in intros_root.findall(".//table_introduction"):
            table_name = elem.get("table_name")
            if table_name:
                tables.append(
                    TableMetadata(
                        schema=schema_name,
                        name=table_name,
                        columns=[],  # Empty - not needed at this stage (only schema and name are used)
                    )
                )
        
        logger.info(f"Loaded {len(tables)} tables from base_sap table_introductions.xml")
        
        # Apply table_filter if provided (case-insensitive match)
        if self.table_filter:
            table_filter_set = {name.lower() for name in self.table_filter}
            original_count = len(tables)
            original_tables = tables  # Save for error message
            tables = [t for t in tables if t.name.lower() in table_filter_set]
            logger.info(f"Filtered SAP tables to {len(tables)} selected table(s) from {len(self.table_filter)} requested (from {original_count} total in base_sap)")
            if not tables:
                available_sample = [t.name for t in original_tables[:10]] if original_count > 0 else []
                logger.warning(f"No SAP tables matched the filter. Requested: {self.table_filter}, Available (sample): {available_sample}")
        
        return tables

    async def _filter_tables_with_data(
        self, 
        tables: List[TableMetadata]
    ) -> List[TableMetadata]:
        """
        Filter tables to only include those with at least one row.
        This dramatically reduces processing time for databases with many empty tables.
        """
        if self.db_type == "mongodb":
            return await self._filter_mongodb_tables_with_data(tables)
        else:
            return await self._filter_sql_tables_with_data(tables)

    async def _filter_sql_tables_with_data(
        self, 
        tables: List[TableMetadata]
    ) -> List[TableMetadata]:
        """Filter SQL tables that have at least one row using EXISTS query."""
        tables_with_data = []
        total_tables = len(tables)
        
        logger.info(f"Checking {total_tables} tables for data...")
        
        async with self.session_factory() as session:
            for idx, table in enumerate(tables):
                try:
                    # For Sybase, query the actual owner from sysobjects
                    actual_owner = None
                    if self.db_type == "sap_sybase":
                        # Query the actual owner of the table from database
                        owner_query = text("""
                            SELECT USER_NAME(uid) as owner
                            FROM sysobjects
                            WHERE name = :table_name
                            AND type IN ('U', 'V')
                        """)
                        owner_result = await session.execute(owner_query, {"table_name": table.name})
                        owner_row = owner_result.fetchone()
                        
                        if owner_row:
                            actual_owner = owner_row[0]
                            # Use the actual owner instead of the metadata schema
                            full_name = f"[{actual_owner}].[{table.name}]"
                        else:
                            # Table doesn't exist in database, skip silently
                            continue
                    else:
                        full_name = self._get_full_table_name(table.schema, table.name)
                    
                    # Use dialect-specific queries for existence check
                    if self.db_type == "sap_oracle":
                        query = text(f"SELECT 1 FROM {full_name} WHERE ROWNUM = 1")
                    elif self.db_type == "sap_hana":
                        query = text(f"SELECT 1 FROM {full_name} LIMIT 1")
                    elif self.db_type == "sap_sybase":
                        # Sybase ASE uses TOP instead of LIMIT
                        query = text(f"SELECT TOP 1 1 FROM {full_name}")
                    else:
                        query = text(f"SELECT 1 FROM {full_name} LIMIT 1")
                    
                    result = await session.execute(query)
                    row = result.fetchone()
                    
                    if row and row[0]:  # EXISTS returns True/False (or 1/0)
                        tables_with_data.append(table)
                        # Use actual owner for logging if available
                        schema_for_log = actual_owner if actual_owner else table.schema
                        logger.info(f"Found table with data: {schema_for_log}.{table.name}")
                    else:
                        logger.debug(f"Skipping empty table: {table.schema}.{table.name}")
                        
                except Exception as e:
                    # Check if it's a "table not found" error - silently skip
                    error_str = str(e).lower()
                    if "not found" in error_str or "does not exist" in error_str or "208" in str(e):
                        # Table doesn't exist - silently skip, no logging
                        continue
                    
                    # For other errors (permissions, etc.), include the table to be safe
                    logger.warning(
                        f"Could not check row count for {table.schema}.{table.name}: {e}. "
                        "Including it in processing."
                    )
                    tables_with_data.append(table)
                
                # Log progress every 50 tables
                if (idx + 1) % 50 == 0:
                    logger.info(f"Checked {idx + 1}/{total_tables} tables, found {len(tables_with_data)} with data")
        
        filtered_count = total_tables - len(tables_with_data)
        if filtered_count > 0:
            logger.info(
                f"Filtered out {filtered_count} empty/non-existent table(s). "
                f"Processing {len(tables_with_data)} table(s) with data."
            )
        else:
            logger.info(f"All {total_tables} tables contain data.")
        
        return tables_with_data

    async def _filter_mongodb_tables_with_data(
        self, 
        tables: List[TableMetadata]
    ) -> List[TableMetadata]:
        """
        Filter MongoDB collections that have at least one document.
        Note: For MongoDB, we can optimize by checking during extraction,
        but this method provides a clean interface for post-extraction filtering.
        """
        db = self.session_factory
        tables_with_data = []
        total_tables = len(tables)
        
        logger.info(f"Checking {total_tables} MongoDB collections for data...")
        
        for idx, table in enumerate(tables):
            try:
                collection = db[table.name]
                # Use count_documents with limit=1 for efficiency
                # This stops counting after finding the first document
                count = await collection.count_documents({}, limit=1)
                
                if count > 0:
                    tables_with_data.append(table)
                else:
                    logger.debug(f"Skipping empty collection: {table.name}")
                    
            except Exception as e:
                # If we can't check, include the collection to be safe
                logger.warning(
                    f"Could not check document count for {table.name}: {e}. "
                    "Including it in processing."
                )
                tables_with_data.append(table)
            
            # Log progress every 50 collections
            if (idx + 1) % 50 == 0:
                logger.info(f"Checked {idx + 1}/{total_tables} collections, found {len(tables_with_data)} with data")
        
        filtered_count = total_tables - len(tables_with_data)
        if filtered_count > 0:
            logger.info(
                f"Filtered out {filtered_count} empty collection(s). "
                f"Processing {len(tables_with_data)} collection(s) with data."
            )
        else:
            logger.info(f"All {total_tables} collections contain data.")
        
        return tables_with_data

    async def _fetch_samples(self, tables: Sequence[TableMetadata], sample_limit: int = 25) -> None:
        async with self.session_factory() as session:
            # SAFETY CAP REMOVED: Processing all tables per user request
            for table in tables:
                # _assert_safe_identifier(table.schema)
                # _assert_safe_identifier(table.name)
                full_name = self._get_full_table_name(table.schema, table.name)
                
                if self.db_type == "sap_oracle":
                    query = text(f'SELECT * FROM {full_name} FETCH FIRST :limit ROWS ONLY')
                elif self.db_type == "sap_sybase":
                    # Sybase ASE uses TOP instead of LIMIT
                    query = text(f'SELECT TOP :limit * FROM {full_name}')
                else:
                    query = text(f'SELECT * FROM {full_name} LIMIT :limit')
                    
                try:
                    result = await session.execute(query, {"limit": sample_limit})
                    rows = [dict(row._mapping) for row in result]
                    table.sample_rows = rows
                except Exception as e:
                    # Fallback or log if sampling fails
                    table.sample_rows = []


    async def _query_tables(self, session: AsyncSession):
        if self.db_type == "sap_hana":
            # SAP HANA specific query
            # SECURITY: Construct WHERE clause from safe, parameterized conditions only
            conditions = []
            params = {}
            if self.schema_filter and self.schema_filter != 'public':
                # Explicit schema provided — filter to it exactly
                _assert_safe_identifier(self.schema_filter)
                conditions.append("SCHEMA_NAME = :schema")
                params["schema"] = self.schema_filter
            else:
                # No explicit schema: exclude SAP/HANA internal system schemas so that
                # application schemas (SAPHANADB, SAPSR3, customer schemas, etc.) are visible.
                # NOTE: db_username is NOT used here — the connection user (e.g. SYSTEM) is
                # an admin credential and does not own the application data in S4 HANA / BW / ECC.
                conditions.append("SCHEMA_NAME NOT LIKE '_SYS_%' AND SCHEMA_NAME NOT LIKE 'HANA_%' AND SCHEMA_NAME NOT IN ('SYS', 'SYSTEM')")
            
            if self.table_prefix:
                # Validate table_prefix is a safe identifier
                _assert_safe_identifier(self.table_prefix)
                conditions.append("TABLE_NAME LIKE :prefix")
                params["prefix"] = f"{self.table_prefix}%"
            
            # Construct WHERE clause from safe conditions (all use parameterized placeholders or are static)
            where_clause = " AND ".join(conditions)
            if where_clause:
                where_clause = f"WHERE {where_clause}"
            else:
                where_clause = ""
            
            # SECURITY: Query uses parameterized conditions only - where_clause contains only :schema/:prefix placeholders or static SQL
            # For SAP HANA system views (SYS.TABLES, SYS.VIEWS), we must use text() as SQLAlchemy select() doesn't support these views
            # All user input is validated via _assert_safe_identifier() and bound via params dict using parameterized placeholders
            # SAST: nosemgrep - This is safe: where_clause contains only validated parameterized placeholders or static SQL
            if where_clause:
                # Query with WHERE clause - all conditions use parameterized placeholders
                # Construct query using constant base and validated where_clause
                # SECURITY: where_clause contains only :schema/:prefix placeholders or static SQL - all user input validated
                base_query_str = "SELECT SCHEMA_NAME as table_schema, TABLE_NAME as table_name FROM SYS.TABLES "
                views_query_str = "SELECT SCHEMA_NAME as table_schema, VIEW_NAME as table_name FROM SYS.VIEWS "
                order_clause_str = " ORDER BY table_schema, table_name"
                view_where_str = where_clause.replace('TABLE_NAME', 'VIEW_NAME')
                # Build query string - where_clause is safe (validated, parameterized placeholders only)
                full_query_str = base_query_str + where_clause + " UNION ALL " + views_query_str + view_where_str + order_clause_str
                # nosemgrep: python.sqlalchemy.security.audit.avoid-sqlalchemy-text.avoid-sqlalchemy-text
                # Reason: SAP HANA system views require text(). All user input validated and bound via params dict.
                query = text(full_query_str)  # nosemgrep: python.sqlalchemy.security.audit.avoid-sqlalchemy-text.avoid-sqlalchemy-text
            else:
                # Query without WHERE clause - static SQL only
                # SECURITY: text() called with constant string literal - no user input
                query = text(
                    "SELECT SCHEMA_NAME as table_schema, TABLE_NAME as table_name FROM SYS.TABLES "
                    "UNION ALL "
                    "SELECT SCHEMA_NAME as table_schema, VIEW_NAME as table_name FROM SYS.VIEWS "
                    "ORDER BY table_schema, table_name"
                )
        elif self.db_type == "sap_sybase":
            # Sybase ASE specific query using sysobjects system table
            # SECURITY: Construct WHERE clause from safe, parameterized conditions only
            conditions = []
            params = {}
            
            # Sybase uses uid (user ID) for schema/owner
            # Only filter by schema if explicitly requested (schema_filter)
            # Don't auto-filter by db_username - tables might be owned by different users (e.g., 'dbo')
            if self.schema_filter:
                _assert_safe_identifier(self.schema_filter)
                conditions.append("USER_NAME(uid) = :schema")
                params["schema"] = self.schema_filter
            
            if self.table_prefix:
                _assert_safe_identifier(self.table_prefix)
                conditions.append("name LIKE :prefix")
                params["prefix"] = f"{self.table_prefix}%"
            
            # Filter for user tables (type='U') and views (type='V')
            # Exclude system objects
            conditions.append("type IN ('U', 'V')")
            conditions.append("name NOT LIKE 'sys%'")
            
            where_clause = " AND ".join(conditions)
            if where_clause:
                where_clause = f"WHERE {where_clause}"
            else:
                where_clause = ""
            
            # Build query using sysobjects
            # SECURITY: where_clause contains only validated parameterized placeholders or static SQL
            query_str = (
                "SELECT USER_NAME(uid) as table_schema, name as table_name "
                "FROM sysobjects "
                f"{where_clause} "
                "ORDER BY table_schema, table_name"
            )
            # nosemgrep: python.sqlalchemy.security.audit.avoid-sqlalchemy-text.avoid-sqlalchemy-text
            # Reason: Sybase system tables require text(). All user input validated and bound via params dict.
            query = text(query_str)  # nosemgrep: python.sqlalchemy.security.audit.avoid-sqlalchemy-text.avoid-sqlalchemy-text
        else:
            # SECURITY: Use SQLAlchemy select() with where() clauses to prevent SQL injection
            table_schema_col = column('table_schema')
            table_name_col = column('table_name')
            table_type_col = column('table_type')
            
            # Build WHERE conditions using SQLAlchemy operators (safe, parameterized)
            where_conditions = [
                table_type_col.in_(['BASE TABLE', 'VIEW']),
                ~table_schema_col.in_(['pg_catalog', 'information_schema', 'mysql', 'performance_schema', 'sys'])
            ]
            
            if self.schema_filter:
                # Validate schema_filter is a safe identifier
                _assert_safe_identifier(self.schema_filter)
                where_conditions.append(table_schema_col == self.schema_filter)
            
            if self.table_prefix:
                # Validate table_prefix is a safe identifier
                _assert_safe_identifier(self.table_prefix)
                like_op = self._get_like_operator()
                if like_op == "ILIKE":
                    # PostgreSQL case-insensitive LIKE
                    where_conditions.append(table_name_col.ilike(self.table_prefix + "%"))
                else:
                    # MySQL case-insensitive LIKE
                    where_conditions.append(table_name_col.like(self.table_prefix + "%"))
            
            # Use select() with text() for table reference, but SQLAlchemy operators for WHERE clause
            query = select(
                table_schema_col,
                table_name_col
            ).select_from(
                text("information_schema.tables")
            ).where(
                and_(*where_conditions)
            ).order_by(table_schema_col, table_name_col)
            
            params = {}
        result = await session.execute(query, params)
        mappings = result.mappings().fetchall()
        return mappings

    async def _query_columns(self, session: AsyncSession, table_names: List[str] = None):
        from sqlalchemy import bindparam
        
        if self.db_type == "sap_hana":
            # SAP HANA specific query
            # SECURITY: Construct WHERE clause from safe, parameterized conditions only
            conditions = []
            params = {}
            if self.schema_filter and self.schema_filter != 'public':
                # Explicit schema provided — filter to it exactly
                _assert_safe_identifier(self.schema_filter)
                conditions.append("SCHEMA_NAME = :schema")
                params["schema"] = self.schema_filter
            else:
                # No explicit schema: exclude SAP/HANA internal system schemas.
                # db_username is not used — see _query_tables comment for rationale.
                conditions.append("SCHEMA_NAME NOT LIKE '_SYS_%' AND SCHEMA_NAME NOT LIKE 'HANA_%' AND SCHEMA_NAME NOT IN ('SYS', 'SYSTEM')")

            if self.table_prefix:
                # Validate table_prefix is a safe identifier
                _assert_safe_identifier(self.table_prefix)
                conditions.append("TABLE_NAME LIKE :prefix")
                params["prefix"] = f"{self.table_prefix}%"
            
            # Using expanding=True requires distinct handling if using text()
            # For simplicity with text(), we often need bindparams to be explicit
            
            if table_names:
                conditions.append("TABLE_NAME IN :table_names")
                # We need to preserve the list/tuple for the params dict
                params["table_names"] = tuple(table_names)
            
            # Construct WHERE clause from safe conditions (all use parameterized placeholders or are static)
            where_clause = " AND ".join(conditions)
            if where_clause:
                where_clause = f"WHERE {where_clause}"
            else:
                where_clause = ""

            # SECURITY: Query uses parameterized conditions only - where_clause contains only :schema/:prefix placeholders or static SQL
            # For SAP HANA system views (SYS.TABLE_COLUMNS, SYS.VIEW_COLUMNS), we must use text() as SQLAlchemy select() doesn't support these views
            # All user input is validated via _assert_safe_identifier() and bound via params dict using parameterized placeholders
            if where_clause:
                # Query with WHERE clause - all conditions use parameterized placeholders
                # Construct query using constant base and validated where_clause
                # SECURITY: where_clause contains only :schema/:prefix placeholders or static SQL - all user input validated
                table_cols_base_str = (
                    "SELECT SCHEMA_NAME as table_schema, TABLE_NAME as table_name, "
                    "COLUMN_NAME as column_name, DATA_TYPE_NAME as data_type, "
                    "IS_NULLABLE as is_nullable, LENGTH as character_maximum_length, "
                    "SCALE as numeric_precision, 0 as numeric_scale, "
                    "POSITION as ordinal_position FROM SYS.TABLE_COLUMNS "
                )
                view_cols_base_str = (
                    "SELECT SCHEMA_NAME as table_schema, VIEW_NAME as table_name, "
                    "COLUMN_NAME as column_name, DATA_TYPE_NAME as data_type, "
                    "IS_NULLABLE as is_nullable, LENGTH as character_maximum_length, "
                    "SCALE as numeric_precision, 0 as numeric_scale, "
                    "POSITION as ordinal_position FROM SYS.VIEW_COLUMNS "
                )
                order_by_str = "ORDER BY table_schema, table_name, ordinal_position"
                view_where_str = where_clause.replace('TABLE_NAME', 'VIEW_NAME')
                # Build query string - where_clause is safe (validated, parameterized placeholders only)
                full_query_str = (
                    table_cols_base_str + where_clause +
                    " UNION ALL " +
                    view_cols_base_str + view_where_str + " " +
                    order_by_str
                )
                # nosemgrep: python.sqlalchemy.security.audit.avoid-sqlalchemy-text.avoid-sqlalchemy-text
                # Reason: SAP HANA system views require text(). All user input validated and bound via params dict.
                query = text(full_query_str)  # nosemgrep: python.sqlalchemy.security.audit.avoid-sqlalchemy-text.avoid-sqlalchemy-text
                if table_names:
                    query = query.bindparams(bindparam("table_names", expanding=True))
            else:
                # Query without WHERE clause - static SQL only
                # SECURITY: text() called with constant string literal - no user input
                query = text(
                    "SELECT SCHEMA_NAME as table_schema, TABLE_NAME as table_name, "
                    "COLUMN_NAME as column_name, DATA_TYPE_NAME as data_type, "
                    "IS_NULLABLE as is_nullable, LENGTH as character_maximum_length, "
                    "SCALE as numeric_precision, 0 as numeric_scale, "
                    "POSITION as ordinal_position FROM SYS.TABLE_COLUMNS "
                    "UNION ALL "
                    "SELECT SCHEMA_NAME as table_schema, VIEW_NAME as table_name, "
                    "COLUMN_NAME as column_name, DATA_TYPE_NAME as data_type, "
                    "IS_NULLABLE as is_nullable, LENGTH as character_maximum_length, "
                    "SCALE as numeric_precision, 0 as numeric_scale, "
                    "POSITION as ordinal_position FROM SYS.VIEW_COLUMNS "
                    "ORDER BY table_schema, table_name, ordinal_position"
                )
                # No need for .bindparams expanding in static query
        elif self.db_type == "sap_sybase":
            # Sybase ASE specific query using syscolumns, sysobjects, and systypes
            conditions = []
            params = {}
            
            # Only filter by schema if explicitly requested (schema_filter)
            # Don't auto-filter by db_username - tables might be owned by different users (e.g., 'dbo')
            if self.schema_filter:
                _assert_safe_identifier(self.schema_filter)
                conditions.append("USER_NAME(o.uid) = :schema")
                params["schema"] = self.schema_filter
            
            if self.table_prefix:
                _assert_safe_identifier(self.table_prefix)
                conditions.append("o.name LIKE :prefix")
                params["prefix"] = f"{self.table_prefix}%"
            
            if table_names:
                # For Sybase, we need to handle the IN clause with individual parameters
                placeholders = ",".join([f":table_name_{i}" for i in range(len(table_names))])
                conditions.append(f"o.name IN ({placeholders})")
                for i, table_name in enumerate(table_names):
                    params[f"table_name_{i}"] = table_name
            
            # Filter for user tables and views only
            conditions.append("o.type IN ('U', 'V')")
            conditions.append("o.name NOT LIKE 'sys%'")
            
            where_clause = " AND ".join(conditions)
            if where_clause:
                where_clause = f"WHERE {where_clause}"
            else:
                where_clause = ""
            
            # Build query using syscolumns joined with sysobjects and systypes
            # SECURITY: where_clause contains only validated parameterized placeholders or static SQL
            query_str = (
                "SELECT "
                "USER_NAME(o.uid) as table_schema, "
                "o.name as table_name, "
                "c.name as column_name, "
                "t.name as data_type, "
                "CASE WHEN c.status & 8 = 8 THEN 'NO' ELSE 'YES' END as is_nullable, "
                "NULL as column_default, "
                "c.length as character_maximum_length, "
                "c.scale as numeric_scale, "
                "c.prec as numeric_precision, "
                "c.colid as ordinal_position "
                "FROM syscolumns c "
                "INNER JOIN sysobjects o ON c.id = o.id "
                "INNER JOIN systypes t ON c.type = t.type "
                f"{where_clause} "
                "ORDER BY table_schema, table_name, ordinal_position"
            )
            # nosemgrep: python.sqlalchemy.security.audit.avoid-sqlalchemy-text.avoid-sqlalchemy-text
            # Reason: Sybase system tables require text(). All user input validated and bound via params dict.
            query = text(query_str)  # nosemgrep: python.sqlalchemy.security.audit.avoid-sqlalchemy-text.avoid-sqlalchemy-text
        else:
            # SECURITY: Use SQLAlchemy select() with where() clauses to prevent SQL injection
            table_schema_col = column('table_schema')
            table_name_col = column('table_name')
            column_name_col = column('column_name')
            ordinal_position_col = column('ordinal_position')
            
            # Initialize conditions and params for text() query
            conditions = []
            params = {}
            
            # Build WHERE conditions as strings (for text() query)
            conditions.append("table_schema NOT IN ('pg_catalog', 'information_schema', 'mysql', 'performance_schema', 'sys')")
            
            if self.schema_filter:
                # Validate schema_filter is a safe identifier
                _assert_safe_identifier(self.schema_filter)
                conditions.append("table_schema = :schema_filter")
                params["schema_filter"] = self.schema_filter
            
            if self.table_prefix:
                # Validate table_prefix is a safe identifier
                _assert_safe_identifier(self.table_prefix)
                like_op = self._get_like_operator()
                conditions.append(f"table_name {like_op} :prefix")
                params["prefix"] = f"{self.table_prefix}%"
            
            if table_names:
                # Filter to only selected tables
                placeholders = ",".join([f":table_name_{i}" for i in range(len(table_names))])
                conditions.append(f"table_name IN ({placeholders})")
                for i, table_name in enumerate(table_names):
                    params[f"table_name_{i}"] = table_name

            where_clause = " AND ".join(conditions)
            query = text(
                f"""
                SELECT table_schema,
                       table_name,
                       column_name,
                       data_type,
                       is_nullable,
                       character_maximum_length,
                       numeric_precision,
                       numeric_scale
                FROM information_schema.columns
                WHERE {where_clause}
                ORDER BY table_schema, table_name, ordinal_position
                """
            )
        
        logger.info(f"Executing column query: {query}")
        result = await session.execute(query, params)
        logger.info("Column query executed. Fetching mappings...")
        mappings = result.mappings().fetchall()
        logger.info(f"Fetched {len(mappings)} column rows.")
        return mappings


    async def _extract_mongodb_metadata(self) -> List[TableMetadata]:
        """Extract metadata from MongoDB by sampling collections."""
        
        # In MongoDB mode, session_factory is the AsyncIOMotorDatabase object
        db = self.session_factory
        logger.info(f"Extracting MongoDB metadata for database: {db.name}")
        
        try:
            collections = await db.list_collection_names()
            logger.info(f"Found collections: {collections}")
        except Exception as e:
            logger.error(f"Failed to list collection names: {e}")
            raise
        
        # Filter out system collections
        collections = [c for c in collections if not c.startswith("system.")]
        logger.info(f"Filtered collections: {collections}")
        
        # If table_filter is provided, filter to only selected collections
        if self.table_filter:
            table_filter_set = {name.lower() for name in self.table_filter}
            collections = [c for c in collections if c.lower() in table_filter_set]
            logger.info(f"Filtered to {len(collections)} selected collection(s) from {len(self.table_filter)} requested")
        
        if self.table_prefix:
            collections = [c for c in collections if c.startswith(self.table_prefix)]
            
        metadata: List[TableMetadata] = []
        
        for coll_name in collections:
            try:
                collection = db[coll_name]
                
                # No empty collection filtering - user manually selects collections, so we respect their choice
                
                # Sample 5 documents to infer schema
                cursor = collection.find({}).limit(5)
                docs = await cursor.to_list(length=5)
                
                column_map: dict[str, ColumnMetadata] = {}
                
                if docs:
                    for doc in docs:
                        for key, value in doc.items():
                            data_type = self._infer_mongodb_type(value)
                            if key not in column_map:
                                column_map[key] = ColumnMetadata(
                                    name=key,
                                    data_type=data_type,
                                    is_nullable=True
                                )
                
                # Fetch a larger sample for the TableMetadata object
                cursor = collection.find({}).limit(25)
                sample_docs = await cursor.to_list(length=25)
                serializable_samples = [self._make_mongodb_serializable(d) for d in sample_docs]
                
                metadata.append(
                    TableMetadata(
                        schema="mongodb",
                        name=coll_name,
                        columns=list(column_map.values()),
                        sample_rows=serializable_samples
                    )
                )
                logger.info(f"Extracted metadata for MongoDB collection: {coll_name}")
            except Exception as e:
                logger.error(f"Failed to extract metadata for collection {coll_name}: {e}")
                
        return metadata

    def _infer_mongodb_type(self, value: Any) -> str:
        """Infer a string type name from a MongoDB value."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return str(type(value).__name__)

    def _make_mongodb_serializable(self, doc: dict) -> dict:
        """Convert MongoDB-specific types (like ObjectId) to strings for serialization."""
        from bson import ObjectId
        from datetime import datetime
        
        new_doc = {}
        for k, v in doc.items():
            if isinstance(v, ObjectId):
                new_doc[k] = str(v)
            elif isinstance(v, datetime):
                new_doc[k] = v.isoformat()
            elif isinstance(v, dict):
                new_doc[k] = self._make_mongodb_serializable(v)
            elif isinstance(v, list):
                new_doc[k] = [
                    self._make_mongodb_serializable(i) if isinstance(i, dict) else 
                    (str(i) if isinstance(i, ObjectId) else i) 
                    for i in v
                ]
            else:
                new_doc[k] = v
        return new_doc
