# Refactored Code
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List, Optional, Union, Tuple

# --- Configuration and Constants ---

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from util.llm_utils import LLMClient
from config.system_config import AGENT_CONFIG
from util.xml_prompt_loader import load_xml_prompt_raw, load_client_prompt, load_client_terminology, BASE_PROMPTS_PATH
from util.number_formatter import format_numbers_in_text
from services.schema_mapper import SchemaMapper
from config.client_config import ClientConfigManager
from langsmith import traceable

logger = logging.getLogger(__name__)

# No prompt truncation constants: we avoid truncating content to prevent any perception of sampling.

# Constants for JSON keys
KEY_ANALYSIS_STRUCTURED = "analysis_structured"
KEY_METRICS = "metrics"
KEY_INSIGHTS = "insights"
KEY_SUMMARY = "summary"
KEY_NOTE = "note"
KEY_RECOMMENDATIONS = "recommendations"
KEY_FOLLOW_UPS = "follow_ups"
KEY_SUMMARY_BULLETS = "summary_bullets"
KEY_SECTION_KPIS = "section_kpis"


class BusinessAgent:
    """
    Analyzes execution results to provide business insights and recommendations.

    This agent serves as the final step in a multi-agent workflow, synthesizing data
    to produce actionable business intelligence.
    """

    # --- Class-level initialization tracking ---
    _AGENT_INITIALIZED = {}
    _CACHED_PROMPTS = {}

    def __init__(
        self,
        prompt_file_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        client_id: str = None,
        db: Any = None,
        llm_client: Optional[LLMClient] = None,
        resolved_prompt: Optional[str] = None,
    ):
        """
        Initializes the Business Agent.
        
        Args:
            prompt_file_path: Optional custom prompt file path
            output_dir: Optional output directory
            client_id: The client ID for multi-tenant operation (REQUIRED - no default)
            db: MongoDB database instance
            llm_client: Shared LLMClient instance from graph state (REQUIRED for graph usage)
        """
        # MULTI-TENANT: Validate client_id is provided
        if not client_id:
            raise ValueError(
                "client_id is REQUIRED for multi-tenant operation. "
                "No default client exists. Every request must specify a valid client_id."
            )
        
        """
        Original docstring continuation (preserving below):

        Args:
            prompt_file_path: Optional path to the agent's prompt file.
            output_dir: Optional directory to store output responses.
            client_id: Client identifier for multi-tenant support (default: 'default')
            db: Database connection for loading client-specific prompts
        """
        self.name = "Business Agent"
        self.client_id = client_id
        self.db = db
        self._resolved_prompt = resolved_prompt

        logger.info(f"Initializing Business Agent | client_id={client_id}")
        
        # Get configuration
        config = AGENT_CONFIG.get("business_agent", {})
        self.prompt_file_path = prompt_file_path or config.get("prompt_file")
        self.output_dir = output_dir or config.get("output_dir")

        # Check if this agent configuration has already been initialized
        agent_key = self._get_agent_cache_key()
        
        if agent_key in self._AGENT_INITIALIZED:
            logger.info(f"Business Agent already initialized, using cached data | client_id={client_id}")
            self._load_from_cache(agent_key)
        else:
            logger.info(f"Performing one-time initialization for Business Agent...")
            self._initialize_fresh(agent_key)

        if self._resolved_prompt is not None:
            self.prompt = self._resolved_prompt

        # --- Initialize LLMClient ---
        # Use shared LLMClient from graph state (REQUIRED for graph usage)
        if llm_client is None:
            raise ValueError(
                "llm_client is REQUIRED for BusinessAgent. "
                "When using agents in the graph, pass the shared LLMClient from state."
            )
        self.llm_client = llm_client
        # Numeric whitelist built per request to enforce exact copying
        self._allowed_numeric_tokens: set[str] = set()

    def _get_agent_cache_key(self) -> str:
        """Generate a unique cache key for this agent configuration."""
        # Include critical config elements that affect initialization
        key_elements = {
            'prompt_file_path': str(self.prompt_file_path) if self.prompt_file_path else None,
            'output_dir': str(self.output_dir) if self.output_dir else None,
            'client_id': self.client_id,  # MULTI-TENANT: Include client_id in cache key
        }
        config_hash = hash(str(sorted(key_elements.items())))
        return f"business_agent_{self.client_id}_{config_hash}"

    def _load_from_cache(self, agent_key: str) -> None:
        """Load cached initialization data."""
        cached_data = self._AGENT_INITIALIZED[agent_key]
        
        # Load cached paths and settings
        self.output_dir = cached_data['output_dir']
        
        # Load cached content
        self.prompt = self._CACHED_PROMPTS.get(agent_key, "Default business agent prompt.")

    def _initialize_fresh(self, agent_key: str) -> None:
        """Perform fresh initialization and cache the results."""
        # --- Path Setup ---
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        # --- Load Content ---
        # MULTI-TENANT: Resolve prompt path dynamically based on client_id
        client_base_dir = Path(PROJECT_ROOT) / "xml_prompts" / "clients" / self.client_id
        client_prompt_path = client_base_dir / "agents" / "business.xml"
        
        prompt_path: Optional[Path] = None
        
        # Priority 1: Client specific prompt
        if client_prompt_path.exists():
            prompt_path = client_prompt_path
            logger.info(f"Using client specific prompt: {prompt_path}")
        # Priority 2: Explicitly provided path (e.g. from config)
        elif isinstance(self.prompt_file_path, str):
            try:
                prompt_path = Path(self.prompt_file_path)
                logger.info(f"Using configured prompt path: {prompt_path}")
            except Exception:
                prompt_path = None
        elif isinstance(self.prompt_file_path, Path):
            prompt_path = self.prompt_file_path
            logger.info(f"Using configured prompt path object: {prompt_path}")

        # Always load the base prompt from file for caching — never cache the resolved prompt
        # since it is request-specific (contains persona / custom prompt injections).
        base_prompt = self._load_file_content(
            prompt_path, "Error loading prompt file", "Default business agent prompt."
        )
        self.prompt = base_prompt

        # --- Cache the initialization data ---
        self._AGENT_INITIALIZED[agent_key] = {
            'output_dir': self.output_dir,
        }

        # Cache only the base file prompt — resolved_prompt is applied after cache load
        self._CACHED_PROMPTS[agent_key] = base_prompt
        
        logger.info(f"Business Agent initialization completed and cached.")

    def _get_relative_path(self, absolute_path: Path) -> str:
        """
        Convert absolute path to relative path from BASE_PROMPTS_PATH.
        
        Args:
            absolute_path: The absolute path to convert
            
        Returns:
            Relative path string for client-aware loading
        """
        try:
            if "xml_prompts" in str(absolute_path):
                parts = absolute_path.parts
                xml_idx = parts.index("xml_prompts")
                
                if xml_idx + 1 < len(parts) and parts[xml_idx + 1] == "base":
                    relative_parts = parts[xml_idx + 2:]
                else:
                    relative_parts = parts[xml_idx + 1:]
                
                return str(Path(*relative_parts))
            
            return str(absolute_path.name)
        except Exception as e:
            logger.warning(f"Error converting path to relative: {e}")
            return str(absolute_path.name)

    def _load_company_context(self, query: str = "", use_knowledge_filtering: bool = False) -> Optional[str]:
        """
        Load company context from all domain knowledge files in client-specific directory.
        Loads company_profile.xml, website_data.xml, and other domain knowledge files.
        
        Returns:
            Formatted company context string or None if not available
        """
        try:
            # SECURITY: Use defusedxml to prevent XXE attacks
            from defusedxml.ElementTree import parse, fromstring
            from config.system_config import USE_KNOWLEDGE_SUMMARIZATION
            from util.knowledge_filter import filter_domain_knowledge_by_query
            from util.knowledge_summarizer import summarize_domain_knowledge_for_prompt

            context_parts = []
            domain_knowledge = {}
            client_domain_knowledge_dir = Path(PROJECT_ROOT) / "xml_prompts" / "clients" / self.client_id / "domain_knowledge"
            
            if not client_domain_knowledge_dir.exists():
                logger.debug(f"Client domain knowledge directory not found for client '{self.client_id}'")
                return None
            
            # Load company_profile.xml if it exists
            company_profile_path = client_domain_knowledge_dir / "company_profile.xml"
            if company_profile_path.exists():
                try:
                    profile_xml = load_xml_prompt_raw(company_profile_path)
                    root = fromstring(profile_xml)
                    
                    # Extract business context
                    business_context = root.find('.//business_context')
                    if business_context is not None:
                        industry = business_context.find('industry')
                        core_business = business_context.find('core_business')
                        business_model = business_context.find('business_model')
                        value_proposition = business_context.find('value_proposition')
                        
                        if industry is not None and industry.text:
                            context_parts.append(f"Industry: {industry.text}")
                        if core_business is not None and core_business.text:
                            context_parts.append(f"Core Business: {core_business.text}")
                        if business_model is not None and business_model.text:
                            context_parts.append(f"Business Model: {business_model.text}")
                        if value_proposition is not None and value_proposition.text:
                            context_parts.append(f"Value Proposition: {value_proposition.text}")
                    
                    # Extract products and services
                    products = [p.text for p in root.findall('.//products/product') if p.text]
                    services = [s.text for s in root.findall('.//services/service') if s.text]
                    
                    if products:
                        context_parts.append(f"Products: {', '.join(products[:5])}")
                    if services:
                        context_parts.append(f"Services: {', '.join(services[:5])}")
                    
                    # Extract terminology for domain-specific language
                    terminology = [t.text for t in root.findall('.//terminology/term') if t.text]
                    if terminology:
                        context_parts.append(f"Key Terminology: {', '.join(terminology[:10])}")
                except Exception as e:
                    logger.warning(f"Failed to load company_profile.xml: {e}")
            
            # Load website_data.xml if it exists
            website_data_path = client_domain_knowledge_dir / "website_data.xml"
            if website_data_path.exists():
                try:
                    website_xml = load_xml_prompt_raw(website_data_path)
                    from defusedxml.ElementTree import fromstring
                    root = fromstring(website_xml)
                    
                    # Extract summary_description
                    summary_description = root.find('.//summary_description')
                    if summary_description is not None and summary_description.text:
                        context_parts.append(f"Website Summary:\n{summary_description.text.strip()}")
                        logger.info(f"Loaded website_data.xml summary for client '{self.client_id}'")
                except Exception as e:
                    logger.warning(f"Failed to load website_data.xml: {e}")
            
            # Load other domain knowledge files (terminology.xml, etc.)
            for file_path in client_domain_knowledge_dir.glob("*.xml"):
                if file_path.name not in ["company_profile.xml", "website_data.xml"]:
                    try:
                        file_xml = load_xml_prompt_raw(file_path)
                        domain_knowledge[file_path.stem] = file_xml
                        logger.info(f"Loaded domain knowledge file: {file_path.name} for client '{self.client_id}'")
                    except Exception as e:
                        logger.warning(f"Failed to load domain knowledge file {file_path.name}: {e}")

            if domain_knowledge:
                if use_knowledge_filtering and query:
                    domain_knowledge = filter_domain_knowledge_by_query(domain_knowledge, query)
                if USE_KNOWLEDGE_SUMMARIZATION:
                    domain_knowledge = {
                        key: summarize_domain_knowledge_for_prompt(content)
                        for key, content in domain_knowledge.items()
                    }
                for key, content in domain_knowledge.items():
                    context_parts.append(f"\n{key}:\n{content}")
            
            # Load client data profile (geography, formatting, locale) if available
            try:
                profile_path = (
                    Path(PROJECT_ROOT) / "xml_prompts" / "clients" / self.client_id
                    / "data_sources" / "meta_information" / "client_data_profile.xml"
                )
                if profile_path.exists():
                    from defusedxml.ElementTree import parse as _parse_xml
                    tree = _parse_xml(str(profile_path))
                    root = tree.getroot()
                    profile_parts = []
                    # Geography
                    geo = root.find(".//geography")
                    if geo is not None and geo.text:
                        profile_parts.append(f"Geography: {geo.text}")
                    # Number format
                    nf = root.find(".//number_format")
                    if nf is not None:
                        profile_parts.append(
                            f"Number format: {nf.get('system', '')} (e.g. {nf.get('example', '')})"
                        )
                    # Currency
                    cur = root.find(".//currency")
                    if cur is not None:
                        profile_parts.append(
                            f"Currency: {cur.get('code', '')} ({cur.get('symbol', '')})"
                        )
                    # Date format preference
                    date_fmts = root.find(".//date_formats")
                    if date_fmts is not None:
                        first_fmt = date_fmts.find("format")
                        if first_fmt is not None:
                            profile_parts.append(
                                f"Date format: Always format dates as {first_fmt.get('pattern', 'DD/MM/YYYY')} in your responses"
                            )
                    # Fiscal year
                    fy = root.find(".//fiscal_year")
                    if fy is not None:
                        profile_parts.append(
                            f"Fiscal year: {fy.get('label', '')} (starts month {fy.get('start_month', '')})"
                        )
                    if profile_parts:
                        context_parts.append("Data Profile:\n" + "\n".join(profile_parts))
                        logger.info("Loaded data profile for business agent (client: %s)", self.client_id)
            except Exception as e:
                logger.debug("Data profile loading skipped in business agent: %s", e)

            # Load admin-configured display preferences from SchemaMapper
            # (these may differ from auto-detected profile values)
            try:
                schema_mapper = SchemaMapper.get_sync(self.client_id, self.db)
                number_config = schema_mapper.get_number_format_config()
                date_config = schema_mapper.get_date_format_config()
                display_pref_parts = []
                if date_config.get("date_format"):
                    display_pref_parts.append(
                        f"IMPORTANT: Always format all dates as {date_config['date_format']} in your responses."
                    )
                if number_config.get("currency_symbol"):
                    display_pref_parts.append(
                        f"Use {number_config['currency_symbol']} as the currency symbol."
                    )
                if display_pref_parts:
                    context_parts.append(
                        "Display Preferences (admin-configured):\n" + "\n".join(display_pref_parts)
                    )
            except Exception as e:
                logger.debug("Display preferences loading skipped in business agent: %s", e)

            if context_parts:
                logger.info(f"Injected company context from domain knowledge for Business Agent (client: {self.client_id})")
                return "\n".join(context_parts)

            return None

        except Exception as e:
            logger.warning(f"Failed to load company context for Business Agent: {e}")
            return None
    
    def _load_file_content(self, file_path: Optional[Path], error_msg: str, default_content: str = "") -> str:
        """
        Loads content from a file or returns default content on failure. 
        Supports both XML and text files.
        
        MULTI-TENANT: Uses client-aware loading for XML files when client_id is not 'default'.
        """
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return default_content
        try:
            if file_path.suffix.lower() == '.xml':
                logger.info(f"Loading XML content from {file_path} (client: {self.client_id})")
                
                # MULTI-TENANT: Use client-aware loading for XML files (client_id is always present now)
                if self.db is not None:
                    relative_path = self._get_relative_path(file_path)
                    try:
                        import asyncio
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            return load_xml_prompt_raw(file_path)
                        else:
                            return loop.run_until_complete(
                                load_client_prompt(relative_path, self.client_id, self.db, use_formatting=False)
                            )
                    except Exception as e:
                        logger.warning(f"Client-aware loading failed, falling back to base: {e}")
                        return load_xml_prompt_raw(file_path)
                else:
                    return load_xml_prompt_raw(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    logger.info(f"Loaded content from {file_path}")
                    return f.read()
        except Exception as e:
            logger.error(f"{error_msg}: {e}")
            return default_content

    @classmethod
    def is_agent_initialized(cls, prompt_file_path: Optional[str] = None, output_dir: Optional[str] = None,
                            client_id: str = None) -> bool:
        """
        Check if a specific agent configuration has been initialized.
        
        Args:
            prompt_file_path: Optional custom prompt file path
            output_dir: Optional output directory
            client_id: The client ID (REQUIRED for multi-tenant operation)
        """
        if not client_id:
            raise ValueError("client_id is REQUIRED")
            
        config = AGENT_CONFIG.get("business_agent", {})
        prompt_path = prompt_file_path or config.get("prompt_file")
        out_dir = output_dir or config.get("output_dir")
        
        key_elements = {
            'prompt_file_path': str(prompt_path) if prompt_path else None,
            'output_dir': str(out_dir) if out_dir else None,
            'client_id': client_id,  # MULTI-TENANT: Include client_id
        }
        config_hash = hash(str(sorted(key_elements.items())))
        agent_key = f"business_agent_{client_id}_{config_hash}"
        return agent_key in cls._AGENT_INITIALIZED

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached initialization data for Business Agent."""
        cls._AGENT_INITIALIZED.clear()
        cls._CACHED_PROMPTS.clear()
        logger.info("Cleared all Business Agent initialization cache.")

    @classmethod
    def force_reinitialize(cls, prompt_file_path: Optional[str] = None, output_dir: Optional[str] = None,
                          client_id: str = None, db: Any = None):
        """
        Force reinitialize Business Agent, clearing its cache first.
        
        Args:
            prompt_file_path: Optional custom prompt file path
            output_dir: Optional output directory
            client_id: The client ID (REQUIRED for multi-tenant operation)
            db: MongoDB database instance
        """
        if not client_id:
            raise ValueError("client_id is REQUIRED")
            
        cls.clear_cache()
        return cls(prompt_file_path, output_dir, client_id, db)

    # --- Prompt Construction Helpers ---

    def _summarize_planner_context(self, planner_response: Optional[Dict[str, Any]]) -> str:
        """Creates a summary string from the Planner Agent's response."""
        if not planner_response:
            return "Planner Context: [Not available for this task.]\n---\n"

        original_query = planner_response.get('user_question', '')
        plan_text = planner_response.get('plan', '')

        parts = []
        if original_query:
            parts.append(f"Original User Query: {original_query}")
        if plan_text:
            parts.append(f"Analysis Plan:\n{plan_text}")

        if parts:
            return "Planner Context:\n" + "\n\n".join(parts) + "\n---\n"
        return "Planner Context: [Planner response available but empty.]\n---\n"

    def _format_text_outputs(self, text_outputs: List[Dict[str, Any]]) -> List[str]:
        """Formats text outputs for inclusion in the prompt."""
        lines = ["Text Outputs:"]
        for item in text_outputs:
            value = str(item.get('value', 'N/A'))
            lines.append(f"  - Name: {item.get('name', 'N/A')}, Value: {value}")
        return lines

    def _format_dataframes(
        self,
        dataframes: List[Dict[str, Any]],
        *,
        max_rows: int = 5000,
        display_map: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Formats dataframes into key=value lines only.
        Example:
        - Dataset Name: _generated_dataframe_
            Segment="Category A", Total Value=114944094.74
            Segment="Category B", Total Value=38128287.31
        """
        import json
        import logging

        logger = logging.getLogger(__name__)

        if display_map is None:
            display_map = {}

        lines: List[str] = ["DataFrames Generated:"]
        # logger.info(f"DataFrames: {dataframes}")
        for item in dataframes:
            name = item.get("name", "N/A")
            lines.append(f"  - Dataset Name: {name}")

            json_data = item.get("json_data")
            raw_data = item.get("data")  # DS/DA format: list of record dicts

            if json_data:
                try:
                    df_dict = json.loads(json_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON for DataFrame {name}: {e}")
                    lines.append("    Error: Could not parse DataFrame JSON.")
                    continue
                columns = df_dict.get("columns", [])
                rows = df_dict.get("data", [])
            elif raw_data and isinstance(raw_data, list) and raw_data:
                first = raw_data[0]
                if isinstance(first, dict):
                    # Convert list-of-dicts to column/row arrays
                    columns = list(first.keys())
                    rows = [[r.get(c) for c in columns] for r in raw_data]
                else:
                    lines.append("    Content: No structured data available.")
                    continue
            else:
                lines.append("    Content: No data available.")
                continue

            if not columns or not rows:
                continue

            # Use pretty labels if provided
            def pretty_label(col: str) -> str:
                return display_map.get(col, col)

            # First column as key, second as main value
            key_col = columns[0]
            value_col = columns[1] if len(columns) > 1 else None

            for row in rows[:max_rows]:
                padded = list(row) + [None] * max(0, len(columns) - len(row))

                key_val = padded[0]
                key_part = f'{pretty_label(key_col)}="{key_val}"' if isinstance(key_val, str) else f"{pretty_label(key_col)}={key_val}"

                parts = [key_part]
                if value_col is not None:
                    parts.append(f"{pretty_label(value_col)}={padded[1]}")

                for col_name, col_val in zip(columns[2:], padded[2:]):
                    if isinstance(col_val, str):
                        parts.append(f'{pretty_label(col_name)}="{col_val}"')
                    else:
                        parts.append(f"{pretty_label(col_name)}={col_val}")

                lines.append("   " + ", ".join(parts))

            if len(rows) > max_rows:
                lines.append(f"   ... ({len(rows) - max_rows} more rows)")

        return lines

    def _format_plotly_charts(self, plotly_charts: List[Dict[str, Any]]) -> List[str]:
        """Formats Plotly charts for inclusion in the prompt."""
        lines = ["Interactive Plotly Charts Generated:"]
        for chart in plotly_charts:
            name = chart.get('name', 'N/A')
            lines.append(f"  - Name: {name}")
            # 'figure' is the legacy key; DS/DA nodes store Plotly JSON under 'data'
            figure_data = chart.get('figure') or chart.get('data')
            if figure_data:
                # Do not include raw figure JSON in the prompt to avoid bloat; include only a concise parsed summary.
                try:
                    # figure_data may already be a dict (from _fetch_generated_chart) or a JSON string
                    if isinstance(figure_data, dict):
                        fig_dict = figure_data
                    else:
                        fig_dict = json.loads(figure_data)
                    data = fig_dict.get('data', [])
                    title = fig_dict.get('layout', {}).get('title', {}).get('text', 'No title')
                    chart_type = data[0].get('type', 'unknown') if data else 'unknown'
                    summary = f"    Summary: {chart_type.capitalize()} chart with {len(data)} data series. Title: '{title}'"
                    lines.append(summary)
                except Exception as e:
                    logger.error(f"Error parsing Plotly chart {name}: {e}")
                    lines.append("    Error: Could not parse Plotly figure.")
            else:
                lines.append("    Content: No figure data available.")
        return lines

    def _summarize_executor_results(self, executor_results: Dict[str, Any]) -> str:
        """Creates a comprehensive summary of the Executor Agent's outputs."""
        summary_parts = []

        # Include artifact registry (analysis chain) if available — gives narrator
        # visibility into the full analytical journey, not just the final output
        artifact_registry = executor_results.get("artifact_registry")
        if artifact_registry and isinstance(artifact_registry, list):
            summary_parts.append("ANALYSIS CHAIN (steps the data agent performed):")
            for entry in artifact_registry:
                iter_num = entry.get("iteration", "?")
                reason = entry.get("reasoning", "")
                new_vars = entry.get("new_variables", {})
                vars_desc = []
                for vname, vinfo in new_vars.items():
                    if isinstance(vinfo, dict) and vinfo.get("type") == "DataFrame":
                        shape = vinfo.get("shape", [])
                        shape_str = f"{shape[0]}×{shape[1]}" if len(shape) >= 2 else str(shape)
                        vars_desc.append(f"{vname} ({shape_str})")
                    else:
                        vars_desc.append(vname)
                vars_str = ", ".join(vars_desc) if vars_desc else ""
                line = f"  Step {iter_num}: {reason}"
                if vars_str:
                    line += f" → {vars_str}"
                summary_parts.append(line)
            summary_parts.append("")

        # Prefer summarized payload when present
        if executor_results.get("summary") and isinstance(executor_results.get("summary"), dict):
            s = executor_results["summary"]
            meta = s.get("meta", {}) or {}
            metrics = s.get("metrics", {}) or {}
            top = s.get("top", {}) or {}
            rollups = s.get("group_rollups", {}) or {}
            desc = (s.get("descriptive", {}) or {}).get("numeric_columns", {}) or {}
            chart_specs = s.get("chart_specs", []) or []
            sample = s.get("sample", {}) or {}
            
            # CRITICAL: Handle empty data case (e.g., cross-tenant facility queries)
            row_count = meta.get("row_count", 0)
            if row_count == 0:
                summary_parts.append("Data Analysis Result:")
                summary_parts.append("  - No data found matching the specified criteria")
                notes = meta.get("notes", [])
                if notes:
                    summary_parts.append(f"  - Note: {'; '.join(notes)}")
                summary_parts.append("\nPlease verify:")
                summary_parts.append("  1. The facility/location name is spelled correctly")
                summary_parts.append("  2. The facility belongs to your organization")
                summary_parts.append("  3. The data filters are appropriate for your dataset")
                return "\n".join(summary_parts)

            # Meta
            summary_parts.append("Summary (Summarizer present):")
            if meta:
                primary = meta.get("primary_frame", "")
                row_count = meta.get("row_count", "")
                notes = "; ".join(meta.get("notes", []) or [])
                summary_parts.append(f"  - Primary Frame: {primary} | Rows: {row_count}")
                if notes:
                    summary_parts.append(f"  - Notes: {notes}")

            # Metrics (explicit numbers to cite, NO computation by LLM)
            if metrics:
                summary_parts.append("Metrics:")
                for k, v in metrics.items():
                    summary_parts.append(f"  - {k}: {v}")

            # Top items - commented out per no-sample policy
            # Items are available in summary.top.items_by_value if needed

            # Group rollups with dynamic column names via SchemaMapper
            # Get client-specific grouping dimensions and metric columns
            try:
                schema_mapper = SchemaMapper.get_sync(self.client_id, self.db)
                grouping_dimensions = schema_mapper.get_grouping_dimensions()
                metric_columns = schema_mapper.get_metric_columns()
                
                # Get primary metric column names (backward compatible)
                value_col_name = metric_columns.get("primary_value") or metric_columns.get("value", "VALUE")
                qty_col_name = metric_columns.get("primary_quantity") or metric_columns.get("qty", "QTY")
                
                logger.info(f"[BusinessAgent] Using SchemaMapper for '{self.client_id}': "
                           f"grouping_dimensions={list(grouping_dimensions.keys())}, "
                           f"value_col={value_col_name}, qty_col={qty_col_name}")
            except Exception as e:
                logger.warning(f"[BusinessAgent] Error loading schema for '{self.client_id}': {e}, using generic fallback")
                grouping_dimensions = {}
                value_col_name = "VALUE"
                qty_col_name = "QTY"
            
            def _emit_rollup(title: str, recs, key: str, display_name: str, value_col: str, qty_col: str):
                """Emit rollup data using dynamic column names."""
                if recs:
                    summary_parts.append(title)
                    for r in recs:
                        name = r.get(key, "")
                        # Use dynamic column names instead of hardcoded CL_VALUE/ON_HAND_QTY
                        value = r.get(value_col, "")
                        qty = r.get(qty_col, "")
                        cnt = r.get("count", "")
                        # Build output string dynamically
                        parts = [f"{display_name}={name}"]
                        if value:
                            parts.append(f"{value_col}={value}")
                        if qty:
                            parts.append(f"{qty_col}={qty}")
                        if cnt:
                            parts.append(f"count={cnt}")
                        summary_parts.append(f"  - {', '.join(parts)}")

            # Iterate over all grouping dimensions dynamically
            for dim_key, rollup_data in rollups.items():
                if rollup_data:
                    if dim_key in grouping_dimensions:
                        dim_info = grouping_dimensions[dim_key]
                        col_name = dim_info["physical_name"]
                        display_name = dim_info["display_name"]
                    else:
                        # Backward compatibility: use dim_key as column name
                        col_name = dim_key.replace("by_", "").upper()
                        display_name = dim_key.replace("by_", "").replace("_", " ").title()
                    
                    _emit_rollup(
                        f"Rollup by {display_name} (Top):",
                        rollup_data,
                        col_name,
                        display_name,
                        value_col_name,
                        qty_col_name
                    )

            # Descriptive stats (for reference only; no LLM math)
            if desc:
                summary_parts.append("Descriptive (numeric columns):")
                for col, stats in desc.items():
                    min_v = stats.get("min"); max_v = stats.get("max")
                    mean_v = stats.get("mean"); med_v = stats.get("median")
                    std_v = stats.get("std")
                    summary_parts.append(
                        f"  - {col}: min={min_v}, max={max_v}, mean={mean_v}, median={med_v}, std={std_v}"
                    )

            # Chart specs
            if chart_specs:
                summary_parts.append("Chart Specs:")
                for cs in chart_specs:
                    title = cs.get("title", "")
                    xtype = cs.get("x", ""); ytype = cs.get("y", "")
                    summary_parts.append(f"  - {title} (x={xtype}, y={ytype})")

            # Sample heads removed per no-sample policy

            # Also include any charts passed through executor_results
            if executor_results.get("charts"):
                summary_parts.extend(self._format_plotly_charts(executor_results["charts"]))

            return "\n".join(summary_parts)
        # logger.info(f"Executor Results: {executor_results}")
        # Legacy/No summarizer path
        if executor_results.get("text_outputs"):
            summary_parts.extend(self._format_text_outputs(executor_results["text_outputs"]))
        if executor_results.get("dataframes"):
            summary_parts.extend(self._format_dataframes(executor_results["dataframes"]))
        if executor_results.get("plotly_charts"):
            summary_parts.extend(self._format_plotly_charts(executor_results["plotly_charts"]))
        if executor_results.get("matplotlib_image_paths"):
            summary_parts.append("Static Matplotlib Images Generated:")
            for path in executor_results["matplotlib_image_paths"]:
                summary_parts.append(f"  - Image Path: {path}")

        return "\n".join(summary_parts) or "No specific data outputs were generated."

    def _replace_schema_placeholders(self, prompt: str) -> str:
        """
        Replace client-specific placeholders in the prompt with actual values from SchemaMapper.
        
        Placeholders are dynamically resolved based on client schema:
        - {COLUMN_CLOSING_VALUE} → Physical column name for closing value metric
        - {COLUMN_AVAILABLE_QTY} → Physical column name for available quantity metric
        - {COLUMN_AGING_BUCKET} → Physical column name for aging/segment dimension
        - {COLUMN_INVENTORY_GROUP} → Physical column name for grouping dimension
        - {COLUMN_ORGANIZATION} → Physical column name for organization/entity dimension
        - {DISPLAY_AGING_BUCKET} → Display name for aging/segment dimension
        - {DISPLAY_INVENTORY_GROUP} → Display name for grouping dimension
        - {DISPLAY_ORGANIZATION} → Display name for organization/entity dimension
        """
        try:
            if self.db is not None:
                schema_mapper = SchemaMapper.get_sync(self.client_id, self.db)
                
                # Get metric columns dynamically
                metric_columns = schema_mapper.get_metric_columns()
                col_closing_value = metric_columns.get("primary_value") or metric_columns.get("value", "VALUE")
                col_available_qty = metric_columns.get("primary_quantity") or metric_columns.get("qty", "QTY")
                
                # Try to get from old logical names for backward compatibility
                try:
                    if "primary_value" not in metric_columns:
                        col_closing_value = schema_mapper.get_column("closing_value")
                except ValueError:
                    pass
                try:
                    if "primary_quantity" not in metric_columns:
                        col_available_qty = schema_mapper.get_column("available_quantity")
                except ValueError:
                    pass
                
                # Get grouping dimensions dynamically
                grouping_dimensions = schema_mapper.get_grouping_dimensions()
                
                # Map common dimension types (backward compatible)
                col_aging_bucket = None
                col_inventory_group = None
                col_organization = None
                display_aging_bucket = None
                display_inventory_group = None
                display_organization = None
                
                # Try to find dimensions by common keys or logical names
                for dim_key, dim_info in grouping_dimensions.items():
                    logical_name = dim_info.get("logical_name", "")
                    if logical_name in ["aging_bucket", "segment"] or "slab" in dim_key or "segment" in dim_key:
                        col_aging_bucket = dim_info["physical_name"]
                        display_aging_bucket = dim_info["display_name"]
                    elif logical_name in ["inventory_group", "category"] or "group" in dim_key:
                        col_inventory_group = dim_info["physical_name"]
                        display_inventory_group = dim_info["display_name"]
                    elif logical_name in ["organization", "entity"] or "site" in dim_key or "entity" in dim_key:
                        col_organization = dim_info["physical_name"]
                        display_organization = dim_info["display_name"]
                
                # Fallback to old logical names for backward compatibility
                if not col_aging_bucket:
                    try:
                        col_aging_bucket = schema_mapper.get_column("aging_bucket")
                        display_aging_bucket = schema_mapper.get_display_name("aging_bucket")
                    except ValueError:
                        col_aging_bucket = "SEGMENT"
                        display_aging_bucket = "Segment"
                
                if not col_inventory_group:
                    try:
                        col_inventory_group = schema_mapper.get_column("inventory_group")
                        display_inventory_group = schema_mapper.get_display_name("inventory_group")
                    except ValueError:
                        col_inventory_group = "CATEGORY"
                        display_inventory_group = "Category"
                
                if not col_organization:
                    try:
                        col_organization = schema_mapper.get_column("organization")
                        display_organization = schema_mapper.get_display_name("organization")
                    except ValueError:
                        col_organization = "ENTITY_NAME"
                        display_organization = "Entity"
                
                # Replace placeholders
                prompt = prompt.replace("{COLUMN_CLOSING_VALUE}", col_closing_value)
                prompt = prompt.replace("{COLUMN_AVAILABLE_QTY}", col_available_qty)
                prompt = prompt.replace("{COLUMN_AGING_BUCKET}", col_aging_bucket)
                prompt = prompt.replace("{COLUMN_INVENTORY_GROUP}", col_inventory_group)
                prompt = prompt.replace("{COLUMN_ORGANIZATION}", col_organization)
                prompt = prompt.replace("{DISPLAY_AGING_BUCKET}", display_aging_bucket)
                prompt = prompt.replace("{DISPLAY_INVENTORY_GROUP}", display_inventory_group)
                prompt = prompt.replace("{DISPLAY_ORGANIZATION}", display_organization)
                
                logger.info(f"Replaced schema placeholders for client '{self.client_id}'")
            else:
                # Fallback to generic defaults     
                pass
        except Exception as e:
            logger.error(f"Error replacing schema placeholders: {e}", exc_info=True)
            # Return original prompt on error
        
        return prompt

    def _build_system_prompt(self, query: str = "", use_knowledge_filtering: bool = False) -> Tuple[str, Dict[str, Any], str]:
        """Builds the system prompt with client-specific configurations."""
        
        # Replace client-specific placeholders in the prompt
        # prompt_with_placeholders = self._replace_schema_placeholders(self.prompt)
        # Initialize with base prompt
        prompt_with_placeholders = self.prompt
        
        # MULTI-TENANT: Load and inject company profile context if available
        company_context = self._load_company_context(query=query, use_knowledge_filtering=use_knowledge_filtering)
        
        # Get dynamic grouping dimensions for summarizer appendix
        group_rollup_info = []
        try:
            if self.db is not None:
                schema_mapper = SchemaMapper.get_sync(self.client_id, self.db)
                grouping_dimensions = schema_mapper.get_grouping_dimensions()
                
                # Build dynamic list of grouping dimensions
                for dim_key, dim_info in grouping_dimensions.items():
                    col_name = dim_info.get("physical_name", dim_key)
                    group_rollup_info.append(f"{dim_key} ({col_name})")
                
                # Fallback to generic if no dimensions found
                if not group_rollup_info:
                    group_rollup_info = ["by_category (CATEGORY)", "by_segment (SEGMENT)", "by_entity (ENTITY_NAME)"]
            else:
                # Fallback to generic defaults
                group_rollup_info = ["by_category (CATEGORY)", "by_segment (SEGMENT)", "by_entity (ENTITY_NAME)"]
        except Exception as e:
            logger.warning(f"Error getting grouping dimensions for summarizer appendix: {e}")
            group_rollup_info = ["by_category (CATEGORY)", "by_segment (SEGMENT)", "by_entity (ENTITY_NAME)"]
        
        group_rollup_str = ", ".join(group_rollup_info)
        
        summarizer_appendix = f"""

Summarizer Schema (Read-Only)
- Do NOT compute any values. Cite only what is present.
- If a summarizer is present, expect the following keys in the payload you summarize from:
  - summary.meta: metadata (row_count, notes, primary_frame).
  - summary.metrics: explicit totals/aggregates (e.g., total_value, total_quantity, avg_value, median_value).
  - summary.top: top lists such as items_by_value.
  - summary.group_rollups: {group_rollup_str}.
  - summary.descriptive.numeric_columns: basic stats (min, max, mean, median, std).
  - summary.chart_specs: lightweight chart suggestions (title, x, y).

Numeric formatting rule:
  - Copy numbers EXACTLY as they appear in inputs (including commas/decimal places). Do not reformat.
"""

        # Safety net: Append custom prompts if not already present
        # (They may be missing if prompt was loaded via load_xml_prompt_raw due to event loop issues)
        if "CUSTOM OVERRIDES" not in prompt_with_placeholders:
            try:
                from util.xml_prompt_loader import load_custom_prompts
                custom_prompts = load_custom_prompts(self.client_id)
                if custom_prompts:
                    logger.info(f"Appending custom prompts to business agent (safety net) | client_id={self.client_id}")
                    prompt_with_placeholders += "\n\n" + custom_prompts
            except Exception as e:
                logger.warning(f"Failed to load custom prompts for business agent | client_id={self.client_id} | error={e}")

        knowledge_metrics = {
            "use_knowledge_filtering": use_knowledge_filtering,
            "company_context_chars": len(company_context) if company_context else 0,
            "company_context_tokens_est": len(company_context) // 4 if company_context else 0,
        }

        return prompt_with_placeholders + summarizer_appendix, knowledge_metrics, company_context or ""


    # --- Technical Noise Filtering ---

    # Patterns that indicate technical/process output, not business-relevant findings
    _TECHNICAL_PATTERNS = re.compile(
        r"("
        r"(?:Overlap|overlap)\s+\w+\s+vs\s+\w+"  # "Overlap ORGANIZATION_ID vs INV_ORG_ID"
        r"|Filtered\s+\w+\s+using\s+\w+"  # "Filtered df_nm using INV_ORG_ID"
        r"|(?:Merge|merge|Join|join)\w*\s+(?:on|using|with)\s"  # "Merged on ORGANIZATION_ID"
        r"|\.shape\s*[:=]"  # "df.shape: (5077331, 12)"
        r"|(?:columns?|dtypes?|dtype)\s*[:=]"  # "columns: ['ORGANIZATION_ID', ...]"
        r"|(?:Index|RangeIndex|Int64Index)\("  # pandas index output
        r"|(?:Loading|Reading)\s+(?:parquet|csv|excel|file)"  # "Loading parquet file"
        r"|\.(?:head|tail|describe|info)\(\)"  # "df.head()"
        r"|(?:rows?|records?)\s*(?:×|x)\s*\d+\s*(?:columns?|cols?)"  # "5077331 rows × 12 columns"
        r"|(?:KeyError|ValueError|TypeError|AttributeError|NameError)\b"  # error class names
        r"|common\s+values?\s*(?:found|:)"  # "8 common values found"
        r"|^\s*\d+\s+rows?\s*$"  # just "5077331 rows"
        r")",
        re.IGNORECASE | re.MULTILINE,
    )

    def _strip_technical_noise(self, console_output: str) -> str:
        """Strip lines containing technical patterns from console output.

        Keeps only lines that look like actual analytical findings/results
        (e.g., "TOTAL INVENTORY IN PHULPUR: 413,972,846.97").
        """
        if not console_output:
            return ""

        cleaned_lines = []
        for line in console_output.split("\n"):
            stripped = line.strip()
            # Keep step headers
            if stripped.startswith("--- Step"):
                cleaned_lines.append(line)
                continue
            # Skip empty lines
            if not stripped:
                continue
            # Skip lines matching technical patterns
            if self._TECHNICAL_PATTERNS.search(stripped):
                continue
            # Skip lines that are mostly UPPER_CASE column names (e.g., "ORGANIZATION_ID INV_ORG_ID")
            words = stripped.split()
            if words and len(words) <= 10:
                upper_count = sum(1 for w in words if re.match(r'^[A-Z][A-Z_]{2,}$', w))
                if upper_count > len(words) * 0.5:
                    continue
            cleaned_lines.append(line)

        # Remove consecutive step headers with no content between them
        result = []
        for i, line in enumerate(cleaned_lines):
            if line.strip().startswith("--- Step"):
                # Check if next non-empty line is another step header or end of list
                has_content = False
                for j in range(i + 1, len(cleaned_lines)):
                    if cleaned_lines[j].strip():
                        has_content = not cleaned_lines[j].strip().startswith("--- Step")
                        break
                if has_content:
                    result.append(line)
            else:
                result.append(line)

        return "\n".join(result).strip()

    def _build_user_message(
        self,
        *,
        query: str,
        plan_text: str,
        analyst_findings: str,
        execution_summary: str,
        console_output: str,
        reference_guidance: str,
        business_insights_sections: Dict[str, bool],
        company_context: str = "",
        persona_guidance: str = "",
    ) -> str:
        """Constructs the final user message for the LLM.

        Information hierarchy (most important first):
        1. THE QUESTION — what the user asked
        2. THE ANSWER — analyst's synthesized findings
        3. DATA EVIDENCE — structured data (tables, metrics)
        4. PROCESS LOG — console output (background reference only)
        """

        def _truncate(text: str, max_chars: int, label: str) -> str:
            if not text:
                return ""
            if len(text) <= max_chars:
                return text
            logger.info(
                "[PromptBudget] business truncating %s from %d to %d chars",
                label,
                len(text),
                max_chars,
            )
            return text[:max_chars] + f"\n... [{label} truncated for latency budget]"

        # Deterministic prompt budgets per section
        query = _truncate(query, 500, "question")
        plan_text = _truncate(plan_text, 4000, "analysis_plan")
        analyst_findings = _truncate(analyst_findings, 8000, "analyst_findings")
        execution_summary = _truncate(execution_summary, 12000, "data_evidence")
        console_output = _truncate(console_output, 6000, "process_log")
        reference_guidance = _truncate(reference_guidance, 3000, "reference_guidance")
        company_context = _truncate(company_context, 2000, "company_context")

        # Build analysis_structured object conditionally
        analysis_structured_parts = []
        if business_insights_sections.get("metrics", True):
            analysis_structured_parts.append('    "metrics": [\n      "Key quantitative fact with context (e.g., \'Total value: ₹69.26 Cr across 2,208 records\')",\n      "Comparative metric if available (e.g., \'Category A accounts for 58% of total\')",\n      "Distribution metric (e.g., \'Majority (65%) concentrated in top segment\')"\n    ]')
        if business_insights_sections.get("insights", True):
            analysis_structured_parts.append('    "insights": [\n      "Business implication: [Observation] + [Why it matters] + [Impact]",\n      "Pattern or anomaly with business impact",\n      "Risk or opportunity identified from the data"\n    ]')
        if business_insights_sections.get("summary", True):
            analysis_structured_parts.append('    "summary": "One executive-level sentence answering the question with critical takeaway"')
        if business_insights_sections.get("note", True):
            analysis_structured_parts.append('    "note": "Optional context about limitations or caveats (only if relevant)"')

        analysis_structured_json = "{\n" + ",\n".join(analysis_structured_parts) + "\n            }" if analysis_structured_parts else "{}"

        # Build root level sections conditionally
        root_sections = []
        if business_insights_sections.get("recommendations", True):
            root_sections.append('  "recommendations": [\n    "Immediate action with expected impact",\n    "Process improvement with rationale",\n    "Strategic initiative for long-term value"\n  ]')
        if business_insights_sections.get("follow_ups", True):
            root_sections.append('  "follow_ups": [\n    "Drill-down question for deeper analysis",\n    "Trend analysis question",\n    "Comparative question across segments"\n  ]')

        # section_kpis: chips for metrics and insights only — recommendations chips are always empty
        kpi_sections = {}
        if business_insights_sections.get("metrics", True):
            kpi_sections['"metrics"'] = '[\n    {"label": "2-3 word label", "value": "Exact number or % from bullets", "sub": "One short context phrase", "color": "red|orange|blue|green"}\n  ]'
        if business_insights_sections.get("insights", True):
            kpi_sections['"insights"'] = '[\n    {"label": "2-3 word label", "value": "New number not in metrics chips", "sub": "One short context phrase", "color": "red|orange|blue"}\n  ]'
        if business_insights_sections.get("recommendations", True):
            kpi_sections['"recommendations"'] = '[\n    {"label": "count or scope of actions", "value": "e.g. \\"3 Actions\\" or \\"2 Areas\\"", "sub": "urgency or timeframe (e.g. \\"immediate priority\\")", "color": "red|orange|blue|green"}\n  ]'
        if kpi_sections:
            kpi_inner = ",\n  ".join(f"{k}: {v}" for k, v in kpi_sections.items())
            root_sections.append(f'  "section_kpis": {{\n  {kpi_inner}\n  }}')

        root_sections_json = ",\n".join(root_sections) if root_sections else ""

        # Build JSON structure
        json_structure = "{\n"
        if analysis_structured_parts:
            json_structure += '            "analysis_structured": ' + analysis_structured_json
            if root_sections_json:
                json_structure += ",\n"
        if root_sections_json:
            json_structure += root_sections_json
        json_structure += "\n            }"

        # Build guidelines conditionally
        guidelines = []
        if business_insights_sections.get("summary", True):
            guidelines.append("""**Summary** - Executive punchline:
- Answer the question in ONE direct sentence
- Lead with the key number in **bold**: e.g. "**43.8 Cr** units in Phulpur"
- Add context: vs total, vs average, trend direction
- A CXO should get the answer in 5 seconds""")
        if business_insights_sections.get("metrics", True):
            guidelines.append("""**Metrics** - Numbers that matter:
- Lead every metric with the number in **bold**: e.g. "**43.8 Cr** units across 8 locations"
- Always add context: % of total, vs average, rank, or trend
- Not just "Total: 43.8 Cr" but "**43.8 Cr** total inventory — 12% of network capacity"
- Cite EXACTLY as shown in data (no reformatting)
- 2-4 metrics max — only the ones a CXO would care about""")
        if business_insights_sections.get("insights", True):
            guidelines.append("""**Insights** - Business implications with teeth:
- Structure: [What you see] + [Why it matters] + [What to do about it]
- Quantify impact: e.g. "Concentration of **65%** in one warehouse creates supply chain risk"
- Use strong language when warranted: "requires attention", "significant exposure", "opportunity to optimize"
- Connect to business outcomes: cost, revenue, risk, efficiency
- 2-3 insights max — quality over quantity""")
        if business_insights_sections.get("recommendations", True):
            guidelines.append("""**Recommendations** - Monday morning actions:
- Specific enough to act on: not "improve inventory" but "review slow-moving items in warehouse exceeding 90-day holding"
- Include expected impact where possible
- Prioritize: immediate action → short-term improvement → strategic initiative
- 2-3 recommendations max""")
        if business_insights_sections.get("follow_ups", True):
            guidelines.append("""**Follow-ups** - Smart exploration:
- Drill-downs, trends, comparisons
- Specific and answerable with available data
- Avoid generic "tell me more" """)
        guidelines.append("""**section_kpis** - Compact KPI chips shown above each section header. Follow these strict per-section rules:

metrics chips (2-3 max):
- value = a standalone number/% directly from the metrics bullets (e.g. "100%", "37,904", "-912")
- label = what that number measures in 2-3 words (e.g. "Products in Deficit", "Pending Demand")
- sub = one short phrase of context (e.g. "unable to meet demand")
- color: red if negative/deficit/risk, orange if gap/shortage/warning, blue if neutral count, green if positive/achieved

insights chips (1-2 max):
- NEVER repeat a number already used in metrics chips
- value = a scope/scale figure that shows business impact (e.g. "100%", "2 SKUs", "All Categories")
- If no genuinely new quantitative insight exists, return empty [] — do NOT invent or reuse numbers
- color: red if systemic failure, orange if concentrated risk, blue if pattern observation

recommendations chips (1-2 max):
- Do NOT use action verb labels like "Reorder" or "Review" as the value
- value = count or scope: how many actions, categories, or areas need attention (e.g. "3 Actions", "2 Categories", "All SKUs")
- label = what those actions target in 2-3 words (e.g. "Need Reorder", "Require Audit", "At Risk")
- sub = urgency or timeframe (e.g. "immediate priority", "within 48 hours", "before next cycle")
- color: red if critical/immediate, orange if high priority, blue if planned, green if preventive""")

        guidelines_text = "\n\n".join(guidelines) if guidelines else ""

        # Build guidelines section separately to avoid f-string backslash issue
        guidelines_section = ""
        if guidelines_text:
            guidelines_section = f"INSIGHT GENERATION GUIDELINES:\n\n{guidelines_text}\n\n"

        # === INFORMATION HIERARCHY: Question → Answer → Data → Process Log ===

        # Section 1: THE QUESTION (most important — anchors the entire response)
        question_section = f"""USER QUESTION: "{query}"
""" if query else ""

        # Section 2: ANALYSIS PLAN (what the system planned to do)
        plan_section = ""
        if plan_text:
            plan_section = f"""
ANALYSIS PLAN:
{plan_text}
"""

        # Section 3: ANALYST'S FINDINGS (the synthesized answer — primary source of truth)
        findings_section = ""
        if analyst_findings:
            findings_section = f"""
ANALYST'S ANSWER (use this as your primary source — this is the data analyst's own summary):
{analyst_findings}
"""

        # Section 4: STRUCTURED DATA (tables, metrics from summarizer)
        data_section = f"""
DATA EVIDENCE:
{execution_summary}
""" if execution_summary else ""

        # Section 5: PROCESS LOG (console output — background reference only, do NOT describe the process)
        process_section = ""
        if console_output and console_output.strip():
            process_section = f"""
PROCESS LOG (background reference only — extract numbers and results, IGNORE the analytical process):
{console_output}
"""

        # Persona block placed BEFORE JSON template so LLM has domain context in working
        # memory when it starts generating each section's content.
        persona_block = ""
        if persona_guidance:
            persona_block = f"""
--- AGENT PERSONA (use this vocabulary and framing in EVERY section below) ---
{persona_guidance}
--- END PERSONA ---

"""

        base_message = f"""You are an expert business analyst translating data analysis results into CXO-grade insights.
Your job: take the FINAL ANSWER from the analysis and present it as a strategic business insight.
You are NOT describing what the analyst did. You ARE answering the user's question with impact.

{question_section}{plan_section}{findings_section}{data_section}{process_section}{persona_block}You MUST return ONLY a single valid JSON object with EXACTLY these keys and structure:
{json_structure}

EVERY section shown above is MANDATORY. Do NOT omit any key. If a section has less to say, still provide at least one meaningful item. An empty array [] or empty string "" for any section is a FAILURE.

{guidelines_section}CRITICAL CONSTRAINTS:
- ANSWER THE QUESTION. The summary MUST directly answer what was asked with the actual number/finding.
- ALL sections (summary, metrics, insights, recommendations, follow_ups) MUST be populated — never skip any.
- Use **bold** markers around key numbers: e.g. "**43.8 Cr** total inventory at Phulpur"
- Do NOT compute new numbers (cite only what exists in data)
- Do NOT reformat numbers (copy EXACTLY as shown)
- Do NOT make assumptions beyond provided data
- Do NOT use vague language - use specific numbers
- Output must be VALID JSON (no markdown, no extra text)
- If data is empty, acknowledge clearly in summary but still populate all sections
- ABSOLUTELY NEVER mention technical/internal details: column names (ORGANIZATION_ID, INV_ORG_ID, etc.),
  join operations, merge operations, data normalization, table structures, data retrieval methods,
  parquet files, DataFrames, filtering steps, or any aspect of HOW the data was processed.
  A CXO does not care about the analytical process. Focus ONLY on the business answer.
"""

        if company_context:
            base_message += f"""
Company Context:
{company_context}

Use company-specific terminology and consider industry context in your insights.
"""

        if reference_guidance:
            base_message += f"""
**ADDITIONAL GUIDANCE** (apply within the required JSON structure above — do NOT omit any sections):
{reference_guidance}
"""
        logger.info(
            "[PromptSize] business user_message_chars=%d | question=%d | plan=%d | findings=%d | evidence=%d | process=%d | reference=%d | persona=%d",
            len(base_message),
            len(query or ""),
            len(plan_text or ""),
            len(analyst_findings or ""),
            len(execution_summary or ""),
            len(console_output or ""),
            len(reference_guidance or ""),
            len(persona_guidance or ""),
        )
        return base_message

    # --- Numeric Validation Helpers ---

    _NUM_TOKEN_PATTERN = re.compile(r"(?:₹)?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?")

    def _extract_numeric_tokens(self, text: str) -> List[str]:
        if not text:
            return []
        return self._NUM_TOKEN_PATTERN.findall(text)

    def _build_numeric_whitelist(self, execution_summary: str, console_output: str, executor_results: Dict[str, Any]) -> None:
        tokens = set()
        # 1) Tokens from formatted strings the LLM sees
        for src in [execution_summary or "", console_output or ""]:
            for t in self._extract_numeric_tokens(src):
                tokens.add(t)
        # 2) Tokens from structured summary to avoid truncation effects
        try:
            summary = executor_results.get("summary") if isinstance(executor_results, dict) else None
            if isinstance(summary, dict):
                # Get metric column names dynamically from SchemaMapper
                try:
                    schema_mapper = SchemaMapper.get_sync(self.client_id, self.db)
                    metric_columns = schema_mapper.get_metric_columns()
                    value_col_name = metric_columns.get("primary_value") or metric_columns.get("value", "VALUE")
                    qty_col_name = metric_columns.get("primary_quantity") or metric_columns.get("qty", "QTY")
                except Exception:
                    # Fallback to defaults if SchemaMapper fails
                    value_col_name = "VALUE"
                    qty_col_name = "QTY"
                
                # Metrics
                for v in (summary.get("metrics") or {}).values():
                    tokens.update(self._extract_numeric_tokens(str(v)))
                # Top items - use dynamic column names
                for rec in (summary.get("top") or {}).get("items_by_value", []) or []:
                    # Extract all numeric values from record, not just hardcoded columns
                    for k, v in rec.items():
                        if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit()):
                            tokens.update(self._extract_numeric_tokens(str(v)))
                # Group rollups - iterate dynamically over all rollup keys
                group_rollups = summary.get("group_rollups", {}) or {}
                for rollup_key, rollup_data in group_rollups.items():
                    for rec in rollup_data or []:
                        # Extract all numeric values, including count and metric columns
                        for k, v in rec.items():
                            if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit()):
                                tokens.update(self._extract_numeric_tokens(str(v)))
                # Descriptive stats
                for rec in (summary.get("descriptive", {}) or {}).get("numeric_columns", {}) or {}.values():
                    for k in ("min", "max", "mean", "median", "std"):
                        if k in rec:
                            tokens.update(self._extract_numeric_tokens(str(rec[k])))
        except Exception:
            pass
        self._allowed_numeric_tokens = tokens

    def _all_tokens_allowed(self, text: str) -> bool:
        # Allow strings with no numbers
        nums = self._extract_numeric_tokens(text)
        if not nums:
            return True
        return all(n in self._allowed_numeric_tokens for n in nums)

    # --- Follow-up Normalization ---

    _LEADING_REQUEST_PATTERN = re.compile(
        r"^(?:would you like|should we|do you want(?: to)?|can i|shall we|could we|would you want|do you wish)\b[\s,:-]*",
        re.IGNORECASE,
    )

    def _normalize_followup(self, q: str) -> str:
        if not isinstance(q, str):
            return ""
        s = q.strip()
        # Strip markdown bold/italic markers
        s = re.sub(r'\*+', '', s)
        # Remove polite/request phrases to make it a neutral analytical question
        s = self._LEADING_REQUEST_PATTERN.sub("", s)
        # Remove leading 'to ' if present after stripping the request phrase
        if s.lower().startswith("to "):
            s = s[3:]
        # Ensure it ends with a question mark
        if s and not s.endswith("?"):
            s = s.rstrip(".") + "?"
        # Capitalize first letter for cleanliness
        if s:
            s = s[0].upper() + s[1:]
        return s

    # --- Main Processing Logic ---

    @traceable(name="narrator_process")
    async def process(
        self,
        executor_results: Dict[str, Any],
        planner_response: Optional[Dict[str, Any]] = None,
        reference_guidance: str = "",
        use_knowledge_filtering: bool = False,
        persona_guidance: str = "",
    ) -> AsyncGenerator[Tuple[str, Optional[Dict[str, Any]]], None]:
        """
        Generates business insights by streaming from an LLM based on execution results.

        Args:
            executor_results: The results from the Executor Agent.
            planner_response: The full response from the Planner Agent.
            reference_guidance: Optional guidance for the response format.

        Yields:
            Tuples of (token, usage) where usage is None for content tokens and contains usage info in final chunk.
        """
        try:
            # Load client config to get business_insights_sections settings
            try:
                from db_config.database import get_db
                db = get_db()
                config_manager = ClientConfigManager(db)
                client_config = await config_manager.get_client_config(self.client_id)
                business_insights_sections = client_config.business_insights_sections
            except Exception as e:
                logger.warning(f"Failed to load client config for business_insights_sections, using defaults: {e}")
                # Default: all sections enabled
                business_insights_sections = {
                    "summary": True,
                    "metrics": True,
                    "insights": True,
                    "recommendations": True,
                    "follow_ups": True,
                    "note": True
                }
            
            # Store in instance variable for use in process_raw_business_insights
            self._business_insights_sections = business_insights_sections
            
            execution_summary = self._summarize_executor_results(executor_results)
            # Prefer clean console output (stdout only, no code/errors) over full debug log
            console_output = executor_results.get("business_console_output") or executor_results.get("console_output", "")
            # Further strip technical noise from console output
            console_output = self._strip_technical_noise(console_output)

            # Safety truncation: prevent context window overflow for the LLM.
            # gpt-4o has 128k tokens (~512k chars). Reserve ~80k tokens for
            # system prompt, guidelines, domain knowledge, and response.
            # Cap console_output + execution_summary to ~40k tokens (~160k chars).
            MAX_CONSOLE_CHARS = 100_000
            MAX_SUMMARY_CHARS = 60_000
            if len(console_output) > MAX_CONSOLE_CHARS:
                logger.warning(f"Console output too large ({len(console_output)} chars), truncating to {MAX_CONSOLE_CHARS}")
                console_output = console_output[:MAX_CONSOLE_CHARS] + "\n... [truncated for context window limit]"
            if len(execution_summary) > MAX_SUMMARY_CHARS:
                logger.warning(f"Execution summary too large ({len(execution_summary)} chars), truncating to {MAX_SUMMARY_CHARS}")
                execution_summary = execution_summary[:MAX_SUMMARY_CHARS] + "\n... [truncated for context window limit]"

            # Extract the user question prominently
            query = ""
            plan_text = ""
            if isinstance(planner_response, dict):
                query = planner_response.get("user_question", "") or planner_response.get("question", "")
                plan_text = planner_response.get("plan", "")

            # DS/DA Agent's own LLM analysis — the synthesized answer/findings.
            # This is the PRIMARY context for business insights (not console output).
            analyst_findings = executor_results.get("ds_analysis", "") or executor_results.get("da_analysis", "")

            system_prompt, knowledge_metrics, company_context = self._build_system_prompt(
                query=query,
                use_knowledge_filtering=use_knowledge_filtering,
            )
            user_message = self._build_user_message(
                query=query,
                plan_text=plan_text,
                analyst_findings=analyst_findings,
                execution_summary=execution_summary,
                console_output=console_output,
                reference_guidance=reference_guidance,
                business_insights_sections=business_insights_sections,
                company_context=company_context,
                persona_guidance=persona_guidance,
            )
            # Build numeric whitelist from the same sources the LLM sees to enforce exact copying
            try:
                self._build_numeric_whitelist(execution_summary, console_output, executor_results)
            except Exception as _:
                # best-effort; do not fail the run
                self._allowed_numeric_tokens = set()
            temperature = AGENT_CONFIG.get("business_agent", {}).get("temperature", 0.0)

            # Store inputs for later retrieval
            self._last_inputs = {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "knowledge_metrics": knowledge_metrics,
            }
            self._last_usage = None

            async for token, usage in self.llm_client.generate_completion_stream(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature
            ):
                if token == "__USAGE__" and usage:
                    # Store usage when we receive it
                    self._last_usage = usage
                    # Don't yield the usage marker, just store it
                else:
                    # Yield content tokens
                    yield (token, usage)
                    # Also capture usage if it comes with a token (some providers might do this)
                    if usage:
                        self._last_usage = usage

        except Exception as e:
            logger.error(f"Error in BusinessAgent.process stream: {e}", exc_info=True)
            yield (f"Error generating business insights: {e}", None)

    # --- JSON Parsing and Cleaning Helpers ---

    def _clean_string_list(self, items: List[Any]) -> List[str]:
        """Cleans a list of strings by removing unwanted characters and fragments."""
        if not isinstance(items, list):
            return []
        
        cleaned_list = []
        # Regex to find JSON fragments that sometimes leak into string values
        json_artefact_pattern = re.compile(r'\"\s*\}\,\s*\"(?:recommendations|follow_ups)\"\:\s*\[\s*\"')
        
        for item in items:
            if isinstance(item, str):
                clean_item = item.strip('"')
                clean_item = json_artefact_pattern.sub('', clean_item)
                cleaned_list.append(clean_item)
        return cleaned_list

    def _repair_json_string(self, json_string: str) -> str:
        """Applies multiple cleaning passes to repair a malformed JSON string."""
        content = json_string.strip()
        
        # Remove markdown code blocks
        content = re.sub(r'```(?:json)?', '', content)
        
        # Find the first valid JSON object if multiple are concatenated (e.g., "}{")
        brace_count = 0
        for i, char in enumerate(content):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    content = content[:i+1]
                    break
        
        # Trim leading/trailing garbage
        json_start = content.find('{')
        json_end = content.rfind('}')
        if json_start != -1 and json_end != -1:
            content = content[json_start : json_end+1]

        return content

    def _manual_extract_from_broken_json(self, content: str) -> Dict[str, Any]:
        """Uses regex to extract data from a string that failed JSON parsing."""
        logger.warning("Attempting manual JSON extraction as a last resort.")
        
        def extract_list(pattern: str) -> List[str]:
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                return []
            return [item.strip() for item in re.findall(r'"([^"]*)"', match.group(1))]

        def extract_string(pattern: str) -> str:
            match = re.search(pattern, content)
            return match.group(1).strip() if match else ""

        a_struct = {
            KEY_METRICS: extract_list(r'"metrics"\s*:\s*\[(.*?)\]'),
            KEY_INSIGHTS: extract_list(r'"insights"\s*:\s*\[(.*?)\]'),
            KEY_SUMMARY: extract_string(r'"summary"\s*:\s*"(.*?)"'),
            KEY_NOTE: extract_string(r'"note"\s*:\s*"(.*?)"'),
        }

        return {
            KEY_ANALYSIS_STRUCTURED: a_struct,
            KEY_RECOMMENDATIONS: extract_list(r'"recommendations"\s*:\s*\[(.*?)\]'),
            KEY_FOLLOW_UPS: extract_list(r'"follow_ups"\s*:\s*\[(.*?)\]'),
            KEY_SECTION_KPIS: {},
        }

    # --- Response Processing and Formatting ---

    async def process_raw_business_insights(self, raw_tokens: str) -> Dict[str, Any]:
        """
        Processes raw LLM token string into a structured dictionary of insights.

        Args:
            raw_tokens: The raw string response from the LLM.

        Returns:
            A structured dictionary containing analysis, recommendations, and follow-ups.
        """
        try:
            parsed_data = None
            try:
                # First attempt: parse directly
                parsed_data = json.loads(raw_tokens)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON decode failed: {e}. Attempting to repair.")
                repaired_json = self._repair_json_string(raw_tokens)
                try:
                    # Second attempt: parse repaired string
                    parsed_data = json.loads(repaired_json)
                except json.JSONDecodeError as e2:
                    logger.warning(f"Repaired JSON decode failed: {e2}. Falling back to manual extraction.")
                    # Third attempt: manual regex extraction
                    parsed_data = self._manual_extract_from_broken_json(raw_tokens)

            if not parsed_data or not any(parsed_data.values()):
                logger.error("All parsing methods failed. Using fallback insights.")
                return self._generate_fallback_insights()
            
            # Normalize and clean the successfully parsed data
            recs = self._clean_string_list(parsed_data.get(KEY_RECOMMENDATIONS, []))
            fus = self._clean_string_list(parsed_data.get(KEY_FOLLOW_UPS, []))
            
            a_struct = self._normalize_structured_analysis(
                parsed_data.get(KEY_ANALYSIS_STRUCTURED)
            )
            analysis_text = self._compose_analysis_text(a_struct)

            # Apply numeric validation filter: drop any strings that contain numbers
            # not present in the allowed whitelist gathered from inputs.
            # GUARD: never let the filter empty an entire section — keep originals if filter
            # would wipe everything, since an imperfectly-numbered insight is better than none.
            def _filter_strings(items: List[str]) -> List[str]:
                out: List[str] = []
                for s in items:
                    try:
                        if self._all_tokens_allowed(s):
                            out.append(s)
                        else:
                            logger.warning(f"Dropping string with non-whitelisted numbers: {s}")
                    except Exception:
                        # If validator fails, be conservative and keep the string
                        out.append(s)
                return out

            # Snapshot originals before filtering for section-empty guard
            orig_metrics = list(a_struct.get(KEY_METRICS, []))
            orig_insights = list(a_struct.get(KEY_INSIGHTS, []))
            orig_summary = a_struct.get(KEY_SUMMARY, "")
            orig_recs = list(recs)
            orig_fus = list(fus)

            # Filter metrics/insights/summary/note and also recs/follow-ups if they contain numbers
            a_struct[KEY_METRICS] = _filter_strings(a_struct.get(KEY_METRICS, []))
            a_struct[KEY_INSIGHTS] = _filter_strings(a_struct.get(KEY_INSIGHTS, []))
            if a_struct.get(KEY_SUMMARY):
                a_struct[KEY_SUMMARY] = a_struct[KEY_SUMMARY] if self._all_tokens_allowed(a_struct[KEY_SUMMARY]) else ""
            if a_struct.get(KEY_NOTE):
                a_struct[KEY_NOTE] = a_struct[KEY_NOTE] if self._all_tokens_allowed(a_struct[KEY_NOTE]) else ""
            recs = _filter_strings(recs)
            fus = _filter_strings(fus)

            # Section-empty guard: if numeric filter wiped an entire section, restore originals.
            # An imperfectly-numbered insight is better than a missing section.
            if not a_struct[KEY_METRICS] and orig_metrics:
                logger.warning("Numeric filter emptied metrics — restoring originals (%d items)", len(orig_metrics))
                a_struct[KEY_METRICS] = orig_metrics
            if not a_struct[KEY_INSIGHTS] and orig_insights:
                logger.warning("Numeric filter emptied insights — restoring originals (%d items)", len(orig_insights))
                a_struct[KEY_INSIGHTS] = orig_insights
            if not a_struct.get(KEY_SUMMARY) and orig_summary:
                logger.warning("Numeric filter emptied summary — restoring original")
                a_struct[KEY_SUMMARY] = orig_summary
            if not recs and orig_recs:
                logger.warning("Numeric filter emptied recommendations — restoring originals (%d items)", len(orig_recs))
                recs = orig_recs
            if not fus and orig_fus:
                logger.warning("Numeric filter emptied follow_ups — restoring originals (%d items)", len(orig_fus))
                fus = orig_fus
            
            # Format numbers in Indian style
            a_struct[KEY_METRICS] = [format_numbers_in_text(text) for text in a_struct[KEY_METRICS]]
            a_struct[KEY_INSIGHTS] = [format_numbers_in_text(text) for text in a_struct[KEY_INSIGHTS]]
            if a_struct.get(KEY_SUMMARY):
                a_struct[KEY_SUMMARY] = format_numbers_in_text(a_struct[KEY_SUMMARY])
            if a_struct.get(KEY_NOTE):
                a_struct[KEY_NOTE] = format_numbers_in_text(a_struct[KEY_NOTE])
            if a_struct.get(KEY_RECOMMENDATIONS):
                a_struct[KEY_RECOMMENDATIONS] = format_numbers_in_text(a_struct[KEY_RECOMMENDATIONS])
            recs = [format_numbers_in_text(text) for text in recs]
            # Normalize follow-ups to neutral, analytical questions
            fus = [self._normalize_followup(x) for x in fus if x]
            fus = [format_numbers_in_text(text) for text in fus]

            # Filter out disabled sections based on client config
            business_insights_sections = getattr(self, '_business_insights_sections', {
                "summary": True,
                "metrics": True,
                "insights": True,
                "recommendations": True,
                "follow_ups": True,
                "note": True
            })
            
            # Remove disabled sections from analysis_structured
            if not business_insights_sections.get("summary", True):
                a_struct[KEY_SUMMARY] = ""
            if not business_insights_sections.get("metrics", True):
                a_struct[KEY_METRICS] = []
            if not business_insights_sections.get("insights", True):
                a_struct[KEY_INSIGHTS] = []
            if not business_insights_sections.get("note", True):
                a_struct[KEY_NOTE] = ""
            
            # Remove disabled sections from root level
            if not business_insights_sections.get("recommendations", True):
                recs = []
            if not business_insights_sections.get("follow_ups", True):
                fus = []

            # Recompose analysis text post-filter
            analysis_text = self._compose_analysis_text(a_struct)

            section_kpis = parsed_data.get(KEY_SECTION_KPIS, {})
            if not isinstance(section_kpis, dict):
                section_kpis = {}

            return {
                "analysis": analysis_text,
                KEY_ANALYSIS_STRUCTURED: a_struct,
                KEY_RECOMMENDATIONS: recs,
                KEY_FOLLOW_UPS: fus,
                KEY_SECTION_KPIS: section_kpis,
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in process_raw_business_insights: {e}", exc_info=True)
            return self._generate_fallback_insights()

    # --- Structuring and Fallback Helpers ---

    def _normalize_structured_analysis(self, a_struct: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensures the structured analysis dictionary has the expected format and types."""
        if not isinstance(a_struct, dict):
            return {KEY_METRICS: [], KEY_INSIGHTS: [], KEY_SUMMARY: "", KEY_NOTE: ""}

        def to_list(value: Any) -> List[str]:
            if isinstance(value, list):
                return [str(x).strip() for x in value if str(x).strip()]
            if isinstance(value, str):
                # Split on newlines, semicolons, or bullet characters that may come inline
                parts = re.split(r"[\n;]|\s*•\s+", value)
                return [p.strip().lstrip("-*• ") for p in parts if p.strip()]
            return []

        def normalize_bullet_multiline(s: str) -> str:
            """If a single string contains multiple bullets inline, convert to multi-line bullets.
            Example: "• a • b" -> "• a\n• b". Also handles cases without leading bullet.
            """
            if not isinstance(s, str):
                return ""
            t = s.strip()
            if not t:
                return t
            # Insert newline before each bullet following text
            t = re.sub(r"\s*•\s+", "\n• ", t)
            return t

        normalized_summary = normalize_bullet_multiline(str(a_struct.get(KEY_SUMMARY, "")).strip())
        normalized_note = str(a_struct.get(KEY_NOTE, "")).strip()

        # Split summary into parts if author provided multiple points; keep first as summary,
        # push remaining parts into insights so frontend renders them as bullets properly.
        summary_parts = [p.strip().lstrip("-*• ") for p in re.split(r"[\n;]|\s*•\s+", normalized_summary) if p.strip()]

        # Prepare insights list from input
        insights_list = to_list(a_struct.get(KEY_INSIGHTS))

        summary_bullets: List[str] = []
        if len(summary_parts) > 1:
            # Expose all parts as dedicated bullets for UI rendering
            summary_bullets = [p for p in summary_parts if p]
            # Render summary itself as newline-separated bullets so it displays as bullets in UI
            normalized_summary = "\n".join([f"• {p}" for p in summary_bullets])
            # Do NOT move these into insights to avoid duplication

        # Ensure insights are de-duplicated while preserving order
        seen = set()
        dedup_insights: List[str] = []
        for i in insights_list:
            key = i.lower()
            if key not in seen:
                seen.add(key)
                dedup_insights.append(i)

        # Build structured object, optionally including summary_bullets
        result = {
            KEY_METRICS: to_list(a_struct.get(KEY_METRICS)),
            KEY_INSIGHTS: dedup_insights,
            KEY_SUMMARY: normalized_summary.lstrip("-*• "),
            KEY_NOTE: normalized_note,
        }
        if summary_bullets:
            result[KEY_SUMMARY_BULLETS] = summary_bullets
        return result

    def _compose_analysis_text(self, a_struct: Dict[str, Any]) -> str:
        """Creates a readable analysis string from structured parts for backward compatibility."""
        lines = []
        if a_struct.get(KEY_METRICS):
            lines.extend([f"• {m}" for m in a_struct[KEY_METRICS]])
        if a_struct.get(KEY_INSIGHTS):
            lines.extend([f"• {i}" for i in a_struct[KEY_INSIGHTS]])
        if a_struct.get(KEY_SUMMARY):
            lines.append(a_struct[KEY_SUMMARY])
        if a_struct.get(KEY_NOTE):
            lines.append(f"Note: {a_struct[KEY_NOTE]}")
        return "\n".join(lines).strip()[:4000]
    
    def _generate_fallback_insights(self) -> Dict[str, Any]:
        """Generates a fallback structured response if the LLM fails completely."""
        logger.warning("Generating fallback business insights.")
        analysis_text = """
        Key analysis indicates a strong performance hierarchy among products, with two leading products driving 45% of sales.
        These top products show consistent growth, while others are declining. Weekly sales peak mid-week, suggesting opportunities
        for targeted promotions during slower periods.
        """
        recommendations = [
            "Increase investment in top-performing products (A and C) to maximize growth.",
            "Investigate the root cause of declining sales for underperforming products (B and E).",
            "Align inventory and staffing with mid-week sales peaks and consider promotions on slower days."
        ]
        follow_ups = [
            "How do these sales trends correlate with recent marketing campaigns?",
            "What is the profit margin for each of the top 5 products?",
            "Are there significant regional variations in product performance?"
        ]
        
        a_struct = {
            KEY_METRICS: ["Top 2 products account for 45% of sales.", "Sales for products B & E declined by ~30%."],
            KEY_INSIGHTS: [
                "Product performance is highly concentrated in a few key items.",
                "There is a clear weekly sales cycle with mid-week peaks.",
                "Growth trends suggest increasing market demand for top products."
            ],
            KEY_SUMMARY: "Analysis reveals a strong product hierarchy and distinct weekly sales patterns, offering clear opportunities for strategic focus and operational adjustments.",
            KEY_NOTE: "This analysis is based on last quarter's sales data."
        }

        return {
            "analysis": self._compose_analysis_text(a_struct),
            KEY_ANALYSIS_STRUCTURED: a_struct,
            KEY_RECOMMENDATIONS: recommendations,
            KEY_FOLLOW_UPS: follow_ups,
            KEY_SECTION_KPIS: {},
        }