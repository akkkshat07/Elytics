"""
Data Profile Agent — Heuristic + LLM detection of client data characteristics.

Runs as part of the Explorer pipeline after table introductions and data
descriptions are generated.  Detects:
  - Geography / locale
  - Number format (Indian / US / EU)
  - Date formats
  - Currency
  - Industry vertical
  - Fiscal calendar
  - ID / relationship patterns

Outputs:
  1. XML file at  xml_prompts/clients/{client_id}/data_sources/meta_information/client_data_profile.xml
  2. Flat dict cached in MongoDB  client_configs.data_profile
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from xml.sax.saxutils import escape  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml

from explorer.models import TableMetadata
from util.llm_utils import LLMClient
from config.system_config import AGENT_CONFIG

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Heuristic patterns
# ──────────────────────────────────────────────────────────────────────

_INDIA_COL_PATTERNS = re.compile(
    r"(pincode|pin_code|district|taluka|tehsil|mandal|"
    r"gst_?no|gstin|pan_?no|aadhaar|ifsc|"
    r"state_?code.*india|india)",
    re.IGNORECASE,
)
_US_COL_PATTERNS = re.compile(
    r"(zip_?code|zip5|state_?code|ssn|ein|fips|county_?code)",
    re.IGNORECASE,
)
_EU_COL_PATTERNS = re.compile(
    r"(postcode|post_?code|vat_?number|iban|bic|nuts_?code)",
    re.IGNORECASE,
)

_HEALTHCARE_PATTERNS = re.compile(
    r"(patient|diagnosis|icd|ndc|prescription|clinical|hospital|"
    r"doctor|physician|symptom|treatment|lab_?result)",
    re.IGNORECASE,
)
_MANUFACTURING_PATTERNS = re.compile(
    r"(material|plant|bom|bill_?of_?material|work_?order|"
    r"production|batch|quality|inspection|mara|marc|"
    r"machine|assembly|warehouse)",
    re.IGNORECASE,
)
_RETAIL_PATTERNS = re.compile(
    r"(product|sku|cart|checkout|customer|order_?line|"
    r"catalog|category|price|discount|coupon|store)",
    re.IGNORECASE,
)
_FINANCE_PATTERNS = re.compile(
    r"(account|ledger|journal|gl_?code|debit|credit|"
    r"transaction|balance|invoice|receivable|payable)",
    re.IGNORECASE,
)
_SUPPLY_CHAIN_PATTERNS = re.compile(
    r"(inventory|shipment|freight|logistics|supplier|vendor|"
    r"procurement|purchase_?order|delivery|dispatch|"
    r"stock|reorder|lead_?time)",
    re.IGNORECASE,
)

# Indian number format: 1,00,000 or 12,34,567
_INDIAN_NUMBER_RE = re.compile(r"\d{1,2}(?:,\d{2})*,\d{3}(?:\.\d+)?$")
# US number format: 100,000 or 1,234,567
_US_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
# EU number format: 100.000,00 (period as thousand sep, comma as decimal)
_EU_NUMBER_RE = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?$")

_DATE_FORMATS = [
    ("%d/%m/%Y", "DD/MM/YYYY"),
    ("%m/%d/%Y", "MM/DD/YYYY"),
    ("%Y-%m-%d", "YYYY-MM-DD"),
    ("%d-%m-%Y", "DD-MM-YYYY"),
    ("%Y/%m/%d", "YYYY/MM/DD"),
    ("%d.%m.%Y", "DD.MM.YYYY"),
    ("%Y%m%d", "YYYYMMDD"),
]

_CURRENCY_COL_PATTERNS = {
    "INR": re.compile(r"(inr|rupee|₹)", re.IGNORECASE),
    "USD": re.compile(r"(usd|dollar|\$)", re.IGNORECASE),
    "EUR": re.compile(r"(eur|euro|€)", re.IGNORECASE),
    "GBP": re.compile(r"(gbp|pound|£)", re.IGNORECASE),
}

_CURRENCY_SYMBOLS = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP"}


class DataProfileAgent:
    """Detects client data characteristics from table metadata + sample rows."""

    def __init__(
        self,
        client_id: str,
        db: Optional[Any] = None,
        agent_name: str = "explorer_data_profile",
    ):
        self.client_id = client_id
        self.db = db
        self.llm = LLMClient(
            agent_name=agent_name, client_id=client_id, db=db
        )

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    async def generate_profile(
        self, tables: Sequence[TableMetadata]
    ) -> Dict[str, Any]:
        """Run all heuristic detectors, optionally validate with LLM, return profile dict."""
        all_col_names = []
        all_table_names = []
        all_sample_values: Dict[str, List[str]] = {}  # col_name -> sample values

        for t in tables:
            all_table_names.append(t.name)
            for c in t.columns:
                all_col_names.append(c.name)
                col_key = f"{t.name}.{c.name}"
                # Collect sample string values for this column
                vals = []
                for row in t.sample_rows[:10]:
                    v = row.get(c.name)
                    if v is not None:
                        vals.append(str(v))
                if vals:
                    all_sample_values[col_key] = vals

        profile: Dict[str, Any] = {}

        # Run all detectors
        profile["geography"] = self._detect_geography(all_col_names, all_sample_values)
        profile["number_format"] = self._detect_number_format(all_sample_values)
        profile["date_formats"] = self._detect_date_formats(all_sample_values)
        profile["currency"] = self._detect_currency(all_col_names, all_sample_values)
        profile["industry"] = self._detect_industry(all_table_names, all_col_names)
        profile["fiscal_calendar"] = self._detect_fiscal_calendar(all_sample_values)
        profile["id_patterns"] = self._detect_id_patterns(tables)
        profile["small_tables"] = self._detect_small_tables(tables)

        # LLM validation/refinement (one small call)
        try:
            profile = await self._llm_validate(profile, all_table_names, all_col_names)
        except Exception as e:
            logger.warning("LLM validation of data profile failed (using heuristics only): %s", e)

        profile["generated_at"] = datetime.utcnow().isoformat() + "Z"
        profile["client_id"] = self.client_id

        return profile

    async def save_as_xml(self, profile: Dict[str, Any], output_path: Path) -> None:
        """Write the profile dict as a structured XML file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            f'<client_data_profile client_id="{escape(self.client_id)}" '
            f'generated_at="{escape(profile.get("generated_at", ""))}">',
        ]

        # Company profile
        geo = profile.get("geography", {})
        ind = profile.get("industry", {})
        lines.append("  <company_profile>")
        if ind.get("industry"):
            lines.append(
                f'    <industry confidence="{ind.get("confidence", 0)}">'
                f"{escape(ind['industry'])}</industry>"
            )
        if geo.get("region"):
            lines.append(
                f'    <geography confidence="{geo.get("confidence", 0)}">'
                f"{escape(geo['region'])}</geography>"
            )
            locale_map = {"india": "en_IN", "us": "en_US", "eu": "en_EU"}
            locale = locale_map.get(geo["region"], "en_US")
            lines.append(f"    <locale>{locale}</locale>")
        lines.append("  </company_profile>")

        # Formatting
        nf = profile.get("number_format", {})
        cur = profile.get("currency", {})
        df = profile.get("date_formats", [])
        fc = profile.get("fiscal_calendar", {})
        lines.append("  <formatting>")
        if nf.get("system"):
            example_map = {"indian": "1,00,000.00", "us": "100,000.00", "eu": "100.000,00"}
            lines.append(
                f'    <number_format system="{escape(nf["system"])}" '
                f'example="{example_map.get(nf["system"], "")}"/>'
            )
        if cur.get("code"):
            lines.append(
                f'    <currency symbol="{escape(cur.get("symbol", ""))}" '
                f'code="{escape(cur["code"])}"/>'
            )
        if df:
            lines.append("    <date_formats>")
            for d in df[:5]:
                lines.append(
                    f'      <format pattern="{escape(d["pattern"])}" '
                    f'confidence="{d.get("confidence", 0)}"/>'
                )
            lines.append("    </date_formats>")
        if fc.get("start_month"):
            month_names = {
                1: "January", 4: "April", 7: "July", 10: "October",
            }
            sm = fc["start_month"]
            label = f"{month_names.get(sm, f'Month {sm}')}-{month_names.get((sm - 2) % 12 + 1, '')}"
            lines.append(
                f'    <fiscal_year start_month="{sm}" label="{escape(label)}"/>'
            )
        lines.append("  </formatting>")

        # Data characteristics
        id_pats = profile.get("id_patterns", {})
        small = profile.get("small_tables", [])
        lines.append("  <data_characteristics>")
        if id_pats.get("relationships"):
            lines.append("    <table_relationships>")
            for rel in id_pats["relationships"][:20]:
                lines.append(
                    f'      <relationship from_table="{escape(rel["from_table"])}" '
                    f'from_col="{escape(rel["from_col"])}" '
                    f'to_table="{escape(rel["to_table"])}" '
                    f'to_col="{escape(rel["to_col"])}"/>'
                )
            lines.append("    </table_relationships>")
        if small:
            lines.append(
                f"    <small_lookup_tables>{escape(', '.join(small))}</small_lookup_tables>"
            )
        lines.append("  </data_characteristics>")

        lines.append("</client_data_profile>")

        content = "\n".join(lines)
        await asyncio.to_thread(output_path.write_text, content, "utf-8")
        logger.info("Saved client data profile XML to %s", output_path)

    async def save_to_mongodb(self, profile: Dict[str, Any]) -> None:
        """Cache a flat version of the profile in MongoDB client_configs."""
        try:
            raw_db = self.db
            if raw_db is None:
                return
            # Unwrap MongoDBManager if needed
            if type(raw_db).__name__ == "MongoDBManager":
                raw_db = getattr(raw_db, "db", raw_db)

            flat = {
                "geography": profile.get("geography", {}).get("region", ""),
                "locale": {
                    "india": "en_IN", "us": "en_US", "eu": "en_EU"
                }.get(profile.get("geography", {}).get("region", ""), "en_US"),
                "number_format": profile.get("number_format", {}).get("system", ""),
                "currency_code": profile.get("currency", {}).get("code", ""),
                "currency_symbol": profile.get("currency", {}).get("symbol", ""),
                "industry": profile.get("industry", {}).get("industry", ""),
                "fiscal_start_month": profile.get("fiscal_calendar", {}).get("start_month"),
                "date_formats": [
                    d["pattern"] for d in profile.get("date_formats", [])[:5]
                ],
                "small_tables": profile.get("small_tables", []),
                "generated_at": profile.get("generated_at", ""),
            }

            await raw_db["client_configs"].update_one(
                {"client_id": self.client_id},
                {"$set": {"data_profile": flat}},
                upsert=False,
            )
            logger.info("Cached data profile in MongoDB for client %s", self.client_id)
        except Exception as e:
            logger.warning("Failed to cache data profile in MongoDB: %s", e)

    # ──────────────────────────────────────────────────────────────────
    # Heuristic detectors
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_geography(
        col_names: List[str], sample_values: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Detect geography from column name patterns and sample value patterns."""
        scores: Counter = Counter()

        for col in col_names:
            if _INDIA_COL_PATTERNS.search(col):
                scores["india"] += 2
            if _US_COL_PATTERNS.search(col):
                scores["us"] += 2
            if _EU_COL_PATTERNS.search(col):
                scores["eu"] += 2

        # Check sample values for geography-specific patterns
        for col_key, vals in sample_values.items():
            for v in vals[:5]:
                # 6-digit PIN codes (India)
                if re.match(r"^\d{6}$", v.strip()):
                    col_lower = col_key.lower()
                    if "pin" in col_lower or "postal" in col_lower or "zip" in col_lower:
                        scores["india"] += 1
                # 5-digit ZIP codes (US)
                if re.match(r"^\d{5}$", v.strip()):
                    col_lower = col_key.lower()
                    if "zip" in col_lower or "postal" in col_lower:
                        scores["us"] += 1
                # GST numbers (India): 15-char alphanumeric
                if re.match(r"^\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z\d]{2}$", v.strip()):
                    scores["india"] += 3

        if not scores:
            return {}

        top = scores.most_common(1)[0]
        total = sum(scores.values())
        confidence = round(min(top[1] / max(total, 1), 1.0), 2)
        return {"region": top[0], "confidence": confidence}

    @staticmethod
    def _detect_number_format(
        sample_values: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Detect number formatting system from sample values."""
        scores: Counter = Counter()

        for col_key, vals in sample_values.items():
            col_lower = col_key.lower()
            # Only check columns likely to have formatted numbers
            if not any(
                kw in col_lower
                for kw in ("amount", "price", "cost", "value", "total", "revenue",
                           "sales", "qty", "quantity", "budget", "salary")
            ):
                continue

            for v in vals[:5]:
                v = v.strip()
                if _INDIAN_NUMBER_RE.match(v):
                    scores["indian"] += 1
                elif _US_NUMBER_RE.match(v):
                    scores["us"] += 1
                elif _EU_NUMBER_RE.match(v):
                    scores["eu"] += 1

        if not scores:
            return {}

        top = scores.most_common(1)[0]
        return {"system": top[0]}

    @staticmethod
    def _detect_date_formats(
        sample_values: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Detect date formats from sample values in date-like columns."""
        format_hits: Counter = Counter()

        for col_key, vals in sample_values.items():
            col_lower = col_key.lower()
            # Only look at columns with date-like names
            if not any(
                kw in col_lower
                for kw in ("date", "time", "created", "updated", "modified",
                           "erdat", "aedat", "budat", "timestamp", "_at", "_on")
            ):
                continue

            for v in vals[:5]:
                v = v.strip()
                for fmt, label in _DATE_FORMATS:
                    try:
                        datetime.strptime(v, fmt)
                        format_hits[label] += 1
                    except (ValueError, TypeError):
                        continue

        if not format_hits:
            return []

        total = sum(format_hits.values())
        result = []
        for label, count in format_hits.most_common(5):
            result.append({
                "pattern": label,
                "confidence": round(count / max(total, 1), 2),
            })
        return result

    @staticmethod
    def _detect_currency(
        col_names: List[str], sample_values: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Detect currency from column names and value prefixes."""
        scores: Counter = Counter()

        for col in col_names:
            for code, pattern in _CURRENCY_COL_PATTERNS.items():
                if pattern.search(col):
                    scores[code] += 2

        # Check sample value prefixes
        for vals in sample_values.values():
            for v in vals[:3]:
                v = v.strip()
                for symbol, code in _CURRENCY_SYMBOLS.items():
                    if v.startswith(symbol):
                        scores[code] += 3

        if not scores:
            return {}

        top = scores.most_common(1)[0]
        code = top[0]
        symbol_map = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}
        return {"code": code, "symbol": symbol_map.get(code, "")}

    @staticmethod
    def _detect_industry(
        table_names: List[str], col_names: List[str]
    ) -> Dict[str, Any]:
        """Detect industry vertical from table/column name patterns."""
        scores: Counter = Counter()
        all_names = " ".join(table_names + col_names)

        pattern_map = {
            "healthcare": _HEALTHCARE_PATTERNS,
            "manufacturing": _MANUFACTURING_PATTERNS,
            "retail": _RETAIL_PATTERNS,
            "finance": _FINANCE_PATTERNS,
            "supply_chain": _SUPPLY_CHAIN_PATTERNS,
        }

        for industry, pattern in pattern_map.items():
            matches = pattern.findall(all_names)
            scores[industry] += len(matches)

        if not scores:
            return {}

        top = scores.most_common(1)[0]
        total = sum(scores.values())
        confidence = round(min(top[1] / max(total, 1), 1.0), 2)
        return {"industry": top[0], "confidence": confidence}

    @staticmethod
    def _detect_fiscal_calendar(
        sample_values: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Detect fiscal calendar from date ranges in sample data."""
        months_seen: List[int] = []

        for col_key, vals in sample_values.items():
            col_lower = col_key.lower()
            if not any(
                kw in col_lower
                for kw in ("date", "erdat", "budat", "created", "timestamp")
            ):
                continue

            for v in vals:
                v = v.strip()
                for fmt, _ in _DATE_FORMATS:
                    try:
                        dt = datetime.strptime(v, fmt)
                        months_seen.append(dt.month)
                        break
                    except (ValueError, TypeError):
                        continue

        if len(months_seen) < 5:
            return {}

        month_counts = Counter(months_seen)
        # If April is common and January is not the most common start,
        # likely April-March fiscal year
        if month_counts.get(4, 0) > 0 and month_counts.get(1, 0) < month_counts.get(4, 0):
            return {"start_month": 4}

        # Default: calendar year
        return {"start_month": 1}

    @staticmethod
    def _detect_id_patterns(
        tables: Sequence[TableMetadata],
    ) -> Dict[str, Any]:
        """Detect primary key columns and potential foreign key relationships."""
        table_cols: Dict[str, List[str]] = {}
        for t in tables:
            table_cols[t.name] = [c.name for c in t.columns]

        # Find columns ending in _ID or _id across tables
        id_cols_by_name: Dict[str, List[str]] = {}  # col_name -> [table_names]
        for tname, cols in table_cols.items():
            for col in cols:
                if col.upper().endswith("_ID") or col.upper() == "ID":
                    id_cols_by_name.setdefault(col.upper(), []).append(tname)

        relationships = []
        for col_name, table_list in id_cols_by_name.items():
            if len(table_list) >= 2:
                # Same column name in multiple tables → potential join
                for i, t1 in enumerate(table_list):
                    for t2 in table_list[i + 1:]:
                        # Find the actual column name (preserving case) in each table
                        t1_col = next(
                            (c for c in table_cols[t1] if c.upper() == col_name),
                            col_name,
                        )
                        t2_col = next(
                            (c for c in table_cols[t2] if c.upper() == col_name),
                            col_name,
                        )
                        relationships.append({
                            "from_table": t1,
                            "from_col": t1_col,
                            "to_table": t2,
                            "to_col": t2_col,
                        })

        return {"relationships": relationships[:30]}

    @staticmethod
    def _detect_small_tables(
        tables: Sequence[TableMetadata],
    ) -> List[str]:
        """Identify small lookup/dimension tables (few sample rows, few columns)."""
        small = []
        for t in tables:
            # Heuristic: tables with ≤ 10 columns and name suggesting dimension/lookup
            name_lower = t.name.lower()
            is_lookup = any(
                kw in name_lower
                for kw in ("dim_", "lookup", "ref_", "master", "type", "status",
                           "config", "category", "region", "country", "currency")
            )
            is_small = len(t.columns) <= 10
            if is_lookup and is_small:
                small.append(t.name)
        return small

    # ──────────────────────────────────────────────────────────────────
    # LLM validation
    # ──────────────────────────────────────────────────────────────────

    async def _llm_validate(
        self,
        profile: Dict[str, Any],
        table_names: List[str],
        col_names: List[str],
    ) -> Dict[str, Any]:
        """One small LLM call to validate/refine heuristic results."""
        # Build a concise summary for the LLM
        summary_parts = [
            f"Tables ({len(table_names)}): {', '.join(table_names[:30])}",
            f"Sample columns: {', '.join(col_names[:50])}",
            "",
            "Heuristic detection results:",
        ]

        geo = profile.get("geography", {})
        if geo:
            summary_parts.append(f"  Geography: {geo.get('region', 'unknown')} (confidence {geo.get('confidence', 0)})")
        ind = profile.get("industry", {})
        if ind:
            summary_parts.append(f"  Industry: {ind.get('industry', 'unknown')} (confidence {ind.get('confidence', 0)})")
        nf = profile.get("number_format", {})
        if nf:
            summary_parts.append(f"  Number format: {nf.get('system', 'unknown')}")
        cur = profile.get("currency", {})
        if cur:
            summary_parts.append(f"  Currency: {cur.get('code', 'unknown')}")
        df = profile.get("date_formats", [])
        if df:
            summary_parts.append(f"  Date formats: {[d['pattern'] for d in df[:3]]}")

        system_prompt = (
            "You are a data analyst. Given table/column names and heuristic detection results, "
            "validate and refine the profile. Return ONLY valid JSON with these fields:\n"
            '{"geography": "india|us|eu|other", "industry": "string", '
            '"number_format": "indian|us|eu", "currency": "INR|USD|EUR|GBP|other", '
            '"notes": "any refinements or corrections"}\n'
            "If the heuristics seem correct, return the same values. "
            "If they seem wrong, provide corrections."
        )

        response = await self.llm.generate_completion(
            system_prompt=system_prompt,
            user_prompt="\n".join(summary_parts),
            temperature=0.0,
            max_output_tokens=500,
        )

        if not response:
            return profile

        # Try to parse JSON from response
        import json
        try:
            # Strip markdown fences if present
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            refinement = json.loads(text)

            # Apply refinements only if they change values with higher confidence
            if refinement.get("geography") and not geo:
                profile["geography"] = {
                    "region": refinement["geography"],
                    "confidence": 0.6,
                }
            if refinement.get("industry") and not ind:
                profile["industry"] = {
                    "industry": refinement["industry"],
                    "confidence": 0.6,
                }
            if refinement.get("number_format") and not nf:
                profile["number_format"] = {"system": refinement["number_format"]}
            if refinement.get("currency") and not cur:
                code = refinement["currency"]
                symbol_map = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}
                profile["currency"] = {
                    "code": code,
                    "symbol": symbol_map.get(code, ""),
                }

            logger.info("LLM validation notes: %s", refinement.get("notes", ""))

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Could not parse LLM validation response: %s", e)

        return profile
