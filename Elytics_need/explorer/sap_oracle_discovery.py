from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from explorer.models import ColumnMetadata, TableMetadata

logger = logging.getLogger(__name__)

EXCLUDE_TABCLASS = {"INTTAB", "APPEND", "POOL", "CLUSTER"}
EXCLUDE_CONTFLAG = {"L", "E", "S", "W"}

EXCLUDE_PREFIXES = (
    "/1",
    "/B",
    "/",
    "T0",
    "TEMP",
    "TMP",
    "DD",
    "TPARA",
    "INDX",
    "PA0",
    "PB",
    "PC",
    "Z",
    "Y",
)

EXCLUDE_SUFFIXES = (
    "_TMP",
    "_SHADOW",
    "_DELTA",
    "_HIST",
)

INCLUDE_CONTFLAG = {"A", "C"}
BUSINESS_COMPONENTS = {
    "FI",
    "CO",
    "MM",
    "SD",
    "PP",
    "QM",
    "PM",
    "PS",
    "WM",
    "LE",
}

BUSINESS_TABLE_PREFIXES = {
    "BKPF",
    "BSEG",
    "SKA1",
    "LFA1",
    "KNA1",
    "MARA",
    "MARC",
    "MAKT",
    "EKKO",
    "EKPO",
    "VBAK",
    "VBAP",
    "LIKP",
    "LIPS",
    "AFKO",
    "AFPO",
    "PLKO",
    "COEP",
    "COBK",
    "TSTC",
    "TSTCT",
}

ALLOWLIST_CONFIG = {
    "auto_approve_patterns": [
        r"^BSEG$",
        r"^MARA$",
        r"^KNA1$",
        r"^LFA1$",
        r"^BKPF$",
        r"^EKPO$",
        r"^VBAK$",
        r"^VBAP$",
    ],
    "require_manual_approval": [
        r"^T0.*",
        r"^USR.*",
    ],
    "blocklist": [
        r"^USR02$",
        r"^DDLOG$",
        r"^D010.*",
        r"^PA0.*",
        r"^PB.*",
        r"^PC.*",
    ],
}

_ALLOWLIST_REGEX = {
    "auto": [re.compile(p) for p in ALLOWLIST_CONFIG["auto_approve_patterns"]],
    "manual": [re.compile(p) for p in ALLOWLIST_CONFIG["require_manual_approval"]],
    "block": [re.compile(p) for p in ALLOWLIST_CONFIG["blocklist"]],
}

BASE_SAP_INTRO_PATH = (
    Path(__file__).resolve().parents[1]
    / "xml_prompts"
    / "base_sap"
    / "data_sources"
    / "meta_information"
    / "table_introductions.xml"
)
BASE_SAP_DESCRIPTIONS_DIR = (
    Path(__file__).resolve().parents[1]
    / "xml_prompts"
    / "base_sap"
    / "data_sources"
    / "data_descriptions"
)


def _load_test_tabnames() -> Optional[set[str]]:
    try:
        tree = ET.parse(BASE_SAP_INTRO_PATH)
    except FileNotFoundError:
        logger.warning(
            "SAP table introductions not found at %s; disabling test-only filter.",
            BASE_SAP_INTRO_PATH,
        )
        return None
    except ET.ParseError as exc:
        logger.warning(
            "Failed to parse SAP table introductions at %s: %s; disabling test-only filter.",
            BASE_SAP_INTRO_PATH,
            exc,
        )
        return None

    tables = {
        (e.attrib.get("table_name") or "").strip().upper()
        for e in tree.findall(".//table_introduction")
    }
    tables.discard("")
    if not tables:
        logger.warning(
            "No tables found in %s; disabling test-only filter.",
            BASE_SAP_INTRO_PATH,
        )
        return None

    logger.info("Loaded %s SAP tables from %s for discovery filter.", len(tables), BASE_SAP_INTRO_PATH)
    return tables


# Restrict SAP ECC Oracle discovery to tables in base_sap introductions.
TEST_ONLY_TABNAMES = _load_test_tabnames()


def write_data_descriptions_xml(
    tables: Iterable[TableMetadata],
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Write per-table data description XML files for SAP tables.
    All columns are included; descriptions are left empty if unknown.
    """
    out_dir = output_dir or BASE_SAP_DESCRIPTIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for table in tables:
        columns = list(table.columns)

        xml_lines = [
            "<?xml version='1.0' encoding='utf-8'?>",
            f'<data_description table_name="{table.name}">',
            f'  <table_info name="{table.name}" total_columns="{len(columns)}" />',
            "  <columns>",
        ]

        for col in columns:
            desc = (col.description or "").strip()
            xml_lines.append(f'    <column name="{col.name}" data_type="{col.data_type}">')
            xml_lines.append(f"      <description>{escape(desc)}</description>")
            xml_lines.append("    </column>")

        xml_lines.extend(["  </columns>", "</data_description>"])
        (out_dir / f"{table.name}_description.xml").write_text(
            "\n".join(xml_lines),
            encoding="utf-8",
        )

    return out_dir

class AllowlistDatabase:
    """
    Placeholder allowlist store.
    Replace with a persistent backend when ready.
    """

    def __init__(self, approved_tables: Optional[Iterable[str]] = None):
        self._approved = {t.upper() for t in (approved_tables or [])}

    def get_approval_status(self, table_name: str) -> str:
        return "APPROVED" if table_name.upper() in self._approved else "PENDING"


class SAPOracleTableDiscoveryPipeline:
    """
    Safe, read-only SAP ECC table discovery for Oracle using DDIC metadata.
    """

    def __init__(self, sap_schema: str = "SAPSR3"):
        self.sap_schema = sap_schema.upper()
        self.allowlist_db = AllowlistDatabase()

    async def discover_tables(self, session: AsyncSession) -> List[TableMetadata]:
        tables = await self._query_ddic_tables(session)
        if not tables:
            return []

        columns = await self._query_ddic_columns(session, [t["TABNAME"] for t in tables])
        if not columns:
            logger.warning("No DD03L column metadata found for SAP DDIC tables.")

        rollname_map = self._collect_rollnames(columns)
        element_texts = await self._query_data_element_texts(session, rollname_map.keys())

        table_map = self._build_table_metadata(tables, columns, element_texts)
        if TEST_ONLY_TABNAMES is not None:
            discovered_tabnames = {t["TABNAME"] for t in table_map.values()}
            missing = sorted(TEST_ONLY_TABNAMES - discovered_tabnames)
            if missing:
                preview = ", ".join(missing[:20])
                logger.warning(
                    "SAP discovery skipped %s table(s) from base_sap introductions because they were "
                    "not found in DD02L/DD02T for schema %s. First 20: %s",
                    len(missing),
                    self.sap_schema,
                    preview,
                )
            explicit_tables = [
                t for t in table_map.values() if t["TABNAME"] in TEST_ONLY_TABNAMES
            ]
            # for table in explicit_tables:
            #     logger.info(
            #         "SAP discovery matched base_sap table: %s.%s",
            #         self.sap_schema,
            #         table["TABNAME"],
            #     )
            stats = await self._get_oracle_stats(
                session, [t["TABNAME"] for t in explicit_tables]
            )
            for table in explicit_tables:
                table["oracle_stats"] = stats.get(table["TABNAME"], {})
            return [self._to_table_metadata(t) for t in explicit_tables]

        tadir_data = await self._get_tadir_components(session, list(table_map.keys()))

        tier1_filtered = [
            t for t in table_map.values()
            if self._tier1_filter(t)
        ]
        tier2_filtered = [
            t for t in tier1_filtered
            if self._tier2_filter(t, tadir_data)
        ]

        stats = await self._get_oracle_stats(session, [t["TABNAME"] for t in tier2_filtered])
        for table in tier2_filtered:
            table["oracle_stats"] = stats.get(table["TABNAME"], {})

        approved = [
            t for t in tier2_filtered
            if self._allowlist_filter(t)
        ]

        return [self._to_table_metadata(t) for t in approved]

    async def _query_ddic_tables(self, session: AsyncSession) -> List[dict]:
        def _build_query(schema_prefix: str, where_clause: str) -> str:
            prefix = f"{schema_prefix}." if schema_prefix else ""
            return f"""
                SELECT
                    l.TABNAME,
                    l.TABCLASS,
                    l.CONTFLAG,
                    t.DDTEXT,
                    l.MAINFLAG,
                    l.ACTFLAG
                FROM {prefix}DD02L l
                LEFT JOIN {prefix}DD02T t
                  ON t.TABNAME = l.TABNAME
                 AND t.DDLANGUAGE = 'E'
                 AND t.AS4LOCAL = 'A'
                WHERE {where_clause}
                ORDER BY l.TABNAME
            """

        where_clauses = [
            (
                "AS4LOCAL='A', ACTFLAG='A'",
                "l.AS4LOCAL = 'A' AND l.TABCLASS IN ('TRANSP') AND l.ACTFLAG = 'A'",
            ),
            (
                "AS4LOCAL='A', ACTFLAG in ('A',' ')",
                "l.AS4LOCAL = 'A' AND l.TABCLASS IN ('TRANSP') "
                "AND NVL(l.ACTFLAG, ' ') IN ('A', ' ')",
            ),
            (
                "AS4LOCAL='A' (no ACTFLAG)",
                "l.AS4LOCAL = 'A' AND l.TABCLASS IN ('TRANSP')",
            ),
            (
                "no AS4LOCAL/ACTFLAG",
                "l.TABCLASS IN ('TRANSP')",
            ),
        ]

        for label, clause in where_clauses:
            try:
                result = await session.execute(text(_build_query(self.sap_schema, clause)))
                rows = [self._normalize_row(row) for row in result.mappings().fetchall()]
                if TEST_ONLY_TABNAMES is not None:
                    rows = [row for row in rows if row.get("TABNAME") in TEST_ONLY_TABNAMES]
                if rows:
                    logger.info(
                        f"DD02L discovery returned {len(rows)} tables in schema {self.sap_schema} "
                        f"using filter: {label}"
                    )
                    return rows
            except Exception as e:
                logger.warning(f"DD02L query failed for schema {self.sap_schema} ({label}): {e}")

        for label, clause in where_clauses:
            try:
                result = await session.execute(text(_build_query("", clause)))
                rows = [self._normalize_row(row) for row in result.mappings().fetchall()]
                if TEST_ONLY_TABNAMES is not None:
                    rows = [row for row in rows if row.get("TABNAME") in TEST_ONLY_TABNAMES]
                if rows:
                    logger.info(
                        f"DD02L discovery returned {len(rows)} tables via unqualified DD02L/ DD02T "
                        f"using filter: {label}"
                    )
                    return rows
            except Exception as e:
                logger.warning(f"DD02L query failed for unqualified DD02L ({label}): {e}")

        logger.warning(
            "DD02L discovery returned 0 rows. Check SELECT privileges on DD02L/DD02T "
            f"for schema {self.sap_schema}."
        )
        await self._probe_ddic_access(session)
        return []

    async def _query_ddic_columns(self, session: AsyncSession, table_names: List[str]) -> List[dict]:
        if not table_names:
            return []

        all_rows: List[dict] = []
        for chunk in self._chunked(table_names):
            query = text(
                f"""
                SELECT
                    l.TABNAME,
                    l.FIELDNAME,
                    l.POSITION,
                    l.KEYFLAG,
                    l.DATATYPE,
                    l.LENG,
                    l.DECIMALS,
                    l.ROLLNAME,
                    l.CHECKTABLE,
                    t.DDTEXT AS FIELD_TEXT
                FROM {self.sap_schema}.DD03L l
                LEFT JOIN {self.sap_schema}.DD03T t
                  ON t.TABNAME = l.TABNAME
                 AND t.FIELDNAME = l.FIELDNAME
                 AND t.DDLANGUAGE = 'E'
                WHERE l.AS4LOCAL = 'A'
                  AND l.TABNAME IN :table_names
                ORDER BY l.TABNAME, l.POSITION
                """
            ).bindparams(bindparam("table_names", expanding=True))

            result = await session.execute(query, {"table_names": chunk})
            all_rows.extend([self._normalize_row(row) for row in result.mappings().fetchall()])

        return all_rows

    async def _query_data_element_texts(
        self,
        session: AsyncSession,
        rollnames: Iterable[str],
    ) -> Dict[str, str]:
        rollnames = [r for r in rollnames if r]
        if not rollnames:
            return {}

        element_texts: Dict[str, str] = {}
        for chunk in self._chunked(rollnames):
            query = text(
                f"""
                SELECT
                    ROLLNAME,
                    DDTEXT
                FROM {self.sap_schema}.DD04T
                WHERE DDLANGUAGE = 'E'
                  AND ROLLNAME IN :rollnames
                """
            ).bindparams(bindparam("rollnames", expanding=True))

            result = await session.execute(query, {"rollnames": chunk})
            element_texts.update(
                {
                    self._normalize_row(row)["ROLLNAME"]: self._normalize_row(row)["DDTEXT"]
                    for row in result.mappings().fetchall()
                }
            )

        return element_texts

    async def _get_tadir_components(
        self,
        session: AsyncSession,
        table_names: List[str],
    ) -> Dict[str, Dict[str, str]]:
        if not table_names:
            return {}

        component_map: Dict[str, Dict[str, str]] = {}
        for chunk in self._chunked(table_names):
            query = text(
                f"""
                SELECT
                    t.OBJ_NAME as TABNAME,
                    d.COMPONENT,
                    d.DEVCLASS as PACKAGE
                FROM {self.sap_schema}.TADIR t
                JOIN {self.sap_schema}.TDEVC d ON t.DEVCLASS = d.DEVCLASS
                WHERE t.PGMID = 'R3TR'
                  AND t.OBJECT = 'TABL'
                  AND t.OBJ_NAME IN :table_names
                """
            ).bindparams(bindparam("table_names", expanding=True))

            try:
                result = await session.execute(query, {"table_names": chunk})
                rows = result.mappings().fetchall()
            except Exception as e:
                logger.warning(f"Failed to query TADIR/TDEVC component data: {e}")
                return {}

            for row in rows:
                normalized = self._normalize_row(row)
                component_map[normalized["TABNAME"]] = {
                    "component": normalized.get("COMPONENT"),
                    "package": normalized.get("PACKAGE"),
                }

        return component_map

    async def _get_oracle_stats(
        self,
        session: AsyncSession,
        table_names: List[str],
    ) -> Dict[str, dict]:
        if not table_names:
            return {}

        stats: Dict[str, dict] = {}
        for chunk in self._chunked(table_names):
            query = text(
                """
                SELECT
                    TABLE_NAME,
                    NUM_ROWS,
                    BLOCKS,
                    LAST_ANALYZED
                FROM ALL_TABLES
                WHERE OWNER = :schema
                  AND TABLE_NAME IN :table_names
                """
            ).bindparams(bindparam("table_names", expanding=True))

            try:
                result = await session.execute(
                    query,
                    {"schema": self.sap_schema, "table_names": chunk},
                )
                rows = result.mappings().fetchall()
            except Exception as e:
                logger.warning(f"Failed to fetch Oracle stats: {e}")
                return {}

            for row in rows:
                normalized = self._normalize_row(row)
                num_rows = normalized.get("NUM_ROWS")
                stats[normalized["TABLE_NAME"]] = {
                    "num_rows": num_rows,
                    "blocks": normalized.get("BLOCKS"),
                    "last_analyzed": normalized.get("LAST_ANALYZED"),
                    "has_stats": num_rows is not None,
                    "is_empty": num_rows == 0 if num_rows is not None else None,
                    "is_large": num_rows > 100_000_000 if num_rows is not None else None,
                }

        return stats

    def _collect_rollnames(self, columns: List[dict]) -> Dict[str, None]:
        rollnames = {}
        for col in columns:
            rollname = col.get("ROLLNAME")
            if rollname:
                rollnames[rollname] = None
        return rollnames

    def _build_table_metadata(
        self,
        tables: List[dict],
        columns: List[dict],
        element_texts: Dict[str, str],
    ) -> Dict[str, dict]:
        column_map: Dict[str, List[dict]] = {}
        for col in columns:
            column_map.setdefault(col["TABNAME"], []).append(col)

        table_map: Dict[str, dict] = {}
        for table in tables:
            name = table["TABNAME"]
            table_columns = column_map.get(name, [])
            for col in table_columns:
                rollname = col.get("ROLLNAME")
                field_text = col.get("FIELD_TEXT")
                col["DDTEXT"] = field_text or element_texts.get(rollname)

            table["columns"] = table_columns
            table_map[name] = table

        return table_map

    def _tier1_filter(self, table_metadata: dict) -> bool:
        if table_metadata["TABCLASS"] in EXCLUDE_TABCLASS:
            return False
        if table_metadata["CONTFLAG"] in EXCLUDE_CONTFLAG:
            return False

        table_name = table_metadata["TABNAME"]
        if "/" in table_name:
            return False
        for prefix in EXCLUDE_PREFIXES:
            if table_name.startswith(prefix):
                return False
        for suffix in EXCLUDE_SUFFIXES:
            if table_name.endswith(suffix):
                return False

        columns = [c["FIELDNAME"] for c in table_metadata.get("columns", [])]
        if "MANDT" not in columns:
            return False

        return True

    def _tier2_filter(self, table_metadata: dict, tadir_data: Dict[str, Dict[str, str]]) -> bool:
        if table_metadata["CONTFLAG"] not in INCLUDE_CONTFLAG:
            return False

        table_name = table_metadata["TABNAME"]
        if tadir_data:
            component = tadir_data.get(table_name, {}).get("component")
            if component:
                top_component = component.split("-")[0]
                if top_component in BUSINESS_COMPONENTS:
                    return True
        return any(table_name.startswith(prefix) for prefix in BUSINESS_TABLE_PREFIXES)

    def _allowlist_filter(self, table_metadata: dict) -> bool:
        table_name = table_metadata["TABNAME"]
        table_class = table_metadata["TABCLASS"]

        for pattern in _ALLOWLIST_REGEX["block"]:
            if pattern.match(table_name):
                return False

        for pattern in _ALLOWLIST_REGEX["manual"]:
            if pattern.match(table_name):
                approval_status = self.allowlist_db.get_approval_status(table_name)
                return approval_status == "APPROVED"

        # if table_class == "VIEW":
        #     approval_status = self.allowlist_db.get_approval_status(table_name)
        #     return approval_status == "APPROVED"

        for pattern in _ALLOWLIST_REGEX["auto"]:
            if pattern.match(table_name):
                return True

        approval_status = self.allowlist_db.get_approval_status(table_name)
        return approval_status == "APPROVED"

    def _to_table_metadata(self, table: dict) -> TableMetadata:
        columns = []
        for col in table.get("columns", []):
            columns.append(
                ColumnMetadata(
                    name=col["FIELDNAME"],
                    data_type=self._map_sap_datatype(col.get("DATATYPE")),
                    is_nullable=col.get("KEYFLAG") != "X",
                    character_maximum_length=col.get("LENG"),
                    numeric_precision=None,
                    numeric_scale=col.get("DECIMALS"),
                    description=col.get("DDTEXT"),
                )
            )

        return TableMetadata(
            schema=self.sap_schema,
            name=table["TABNAME"],
            columns=columns,
        )

    def _map_sap_datatype(self, sap_type: Optional[str]) -> str:
        type_mapping = {
            "CHAR": "VARCHAR",
            "NUMC": "VARCHAR",
            "DATS": "DATE",
            "TIMS": "TIME",
            "DEC": "DECIMAL",
            "QUAN": "DECIMAL",
            "CURR": "DECIMAL",
            "INT1": "INTEGER",
            "INT2": "INTEGER",
            "INT4": "INTEGER",
            "FLTP": "FLOAT",
            "CLNT": "VARCHAR",
            "LANG": "VARCHAR",
            "CUKY": "VARCHAR",
            "UNIT": "VARCHAR",
        }
        if not sap_type:
            return "VARCHAR"
        return type_mapping.get(sap_type.upper(), "VARCHAR")

    def _chunked(self, values: Iterable[str], size: int = 1000) -> Iterable[List[str]]:
        batch: List[str] = []
        for value in values:
            batch.append(value)
            if len(batch) >= size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _normalize_row(self, row: dict) -> dict:
        return {str(key).upper(): value for key, value in dict(row).items()}

    async def _probe_ddic_access(self, session: AsyncSession) -> None:
        """
        Probe DDIC tables to surface privilege issues in logs.
        """
        tables = ["DD02L", "DD02T", "DD03L", "DD04T", "TADIR", "TDEVC"]
        for table_name in tables:
            qualified = f"{self.sap_schema}.{table_name}"
            try:
                await session.execute(text(f"SELECT 1 FROM {qualified} WHERE ROWNUM = 1"))
                logger.info(f"DDIC access OK: {qualified}")
                continue
            except Exception as e:
                logger.warning(f"DDIC access failed: {qualified}: {e}")

            try:
                await session.execute(text(f"SELECT 1 FROM {table_name} WHERE ROWNUM = 1"))
                logger.info(f"DDIC access OK via synonym: {table_name}")
            except Exception as e:
                logger.warning(f"DDIC access failed via synonym: {table_name}: {e}")
