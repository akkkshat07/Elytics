from __future__ import annotations
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List, Optional, Union, Tuple
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
KEY_ANALYSIS_STRUCTURED = 'analysis_structured'
KEY_METRICS = 'metrics'
KEY_INSIGHTS = 'insights'
KEY_SUMMARY = 'summary'
KEY_NOTE = 'note'
KEY_RECOMMENDATIONS = 'recommendations'
KEY_FOLLOW_UPS = 'follow_ups'
KEY_SUMMARY_BULLETS = 'summary_bullets'
KEY_SECTION_KPIS = 'section_kpis'

class BusinessAgent:
    _AGENT_INITIALIZED = {}
    _CACHED_PROMPTS = {}

    def __init__(self, prompt_file_path: Optional[str]=None, output_dir: Optional[str]=None, client_id: str=None, db: Any=None, llm_client: Optional[LLMClient]=None, resolved_prompt: Optional[str]=None):
        if not client_id:
            raise ValueError('client_id is REQUIRED for multi-tenant operation. No default client exists. Every request must specify a valid client_id.')
        "\n        Original docstring continuation (preserving below):\n\n        Args:\n            prompt_file_path: Optional path to the agent's prompt file.\n            output_dir: Optional directory to store output responses.\n            client_id: Client identifier for multi-tenant support (default: 'default')\n            db: Database connection for loading client-specific prompts\n        "
        self.name = 'Business Agent'
        self.client_id = client_id
        self.db = db
        self._resolved_prompt = resolved_prompt
        logger.info(f'Initializing Business Agent | client_id={client_id}')
        config = AGENT_CONFIG.get('business_agent', {})
        self.prompt_file_path = prompt_file_path or config.get('prompt_file')
        self.output_dir = output_dir or config.get('output_dir')
        agent_key = self._get_agent_cache_key()
        if agent_key in self._AGENT_INITIALIZED:
            logger.info(f'Business Agent already initialized, using cached data | client_id={client_id}')
            self._load_from_cache(agent_key)
        else:
            logger.info(f'Performing one-time initialization for Business Agent...')
            self._initialize_fresh(agent_key)
        if self._resolved_prompt is not None:
            self.prompt = self._resolved_prompt
        if llm_client is None:
            raise ValueError('llm_client is REQUIRED for BusinessAgent. When using agents in the graph, pass the shared LLMClient from state.')
        self.llm_client = llm_client
        self._allowed_numeric_tokens: set[str] = set()

    def _get_agent_cache_key(self) -> str:
        key_elements = {'prompt_file_path': str(self.prompt_file_path) if self.prompt_file_path else None, 'output_dir': str(self.output_dir) if self.output_dir else None, 'client_id': self.client_id}
        config_hash = hash(str(sorted(key_elements.items())))
        return f'business_agent_{self.client_id}_{config_hash}'

    def _load_from_cache(self, agent_key: str) -> None:
        cached_data = self._AGENT_INITIALIZED[agent_key]
        self.output_dir = cached_data['output_dir']
        self.prompt = self._CACHED_PROMPTS.get(agent_key, 'Default business agent prompt.')

    def _initialize_fresh(self, agent_key: str) -> None:
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        client_base_dir = Path(PROJECT_ROOT) / 'xml_prompts' / 'clients' / self.client_id
        client_prompt_path = client_base_dir / 'agents' / 'business.xml'
        prompt_path: Optional[Path] = None
        if client_prompt_path.exists():
            prompt_path = client_prompt_path
            logger.info(f'Using client specific prompt: {prompt_path}')
        elif isinstance(self.prompt_file_path, str):
            try:
                prompt_path = Path(self.prompt_file_path)
                logger.info(f'Using configured prompt path: {prompt_path}')
            except Exception:
                prompt_path = None
        elif isinstance(self.prompt_file_path, Path):
            prompt_path = self.prompt_file_path
            logger.info(f'Using configured prompt path object: {prompt_path}')
        base_prompt = self._load_file_content(prompt_path, 'Error loading prompt file', 'Default business agent prompt.')
        self.prompt = base_prompt
        self._AGENT_INITIALIZED[agent_key] = {'output_dir': self.output_dir}
        self._CACHED_PROMPTS[agent_key] = base_prompt
        logger.info(f'Business Agent initialization completed and cached.')

    def _get_relative_path(self, absolute_path: Path) -> str:
        try:
            if 'xml_prompts' in str(absolute_path):
                parts = absolute_path.parts
                xml_idx = parts.index('xml_prompts')
                if xml_idx + 1 < len(parts) and parts[xml_idx + 1] == 'base':
                    relative_parts = parts[xml_idx + 2:]
                else:
                    relative_parts = parts[xml_idx + 1:]
                return str(Path(*relative_parts))
            return str(absolute_path.name)
        except Exception as e:
            logger.warning(f'Error converting path to relative: {e}')
            return str(absolute_path.name)

    def _load_company_context(self, query: str='', use_knowledge_filtering: bool=False) -> Optional[str]:
        try:
            from defusedxml.ElementTree import parse, fromstring
            from config.system_config import USE_KNOWLEDGE_SUMMARIZATION
            from util.knowledge_filter import filter_domain_knowledge_by_query
            from util.knowledge_summarizer import summarize_domain_knowledge_for_prompt
            context_parts = []
            domain_knowledge = {}
            client_domain_knowledge_dir = Path(PROJECT_ROOT) / 'xml_prompts' / 'clients' / self.client_id / 'domain_knowledge'
            if not client_domain_knowledge_dir.exists():
                logger.debug(f"Client domain knowledge directory not found for client '{self.client_id}'")
                return None
            company_profile_path = client_domain_knowledge_dir / 'company_profile.xml'
            if company_profile_path.exists():
                try:
                    profile_xml = load_xml_prompt_raw(company_profile_path)
                    root = fromstring(profile_xml)
                    business_context = root.find('.//business_context')
                    if business_context is not None:
                        industry = business_context.find('industry')
                        core_business = business_context.find('core_business')
                        business_model = business_context.find('business_model')
                        value_proposition = business_context.find('value_proposition')
                        if industry is not None and industry.text:
                            context_parts.append(f'Industry: {industry.text}')
                        if core_business is not None and core_business.text:
                            context_parts.append(f'Core Business: {core_business.text}')
                        if business_model is not None and business_model.text:
                            context_parts.append(f'Business Model: {business_model.text}')
                        if value_proposition is not None and value_proposition.text:
                            context_parts.append(f'Value Proposition: {value_proposition.text}')
                    products = [p.text for p in root.findall('.//products/product') if p.text]
                    services = [s.text for s in root.findall('.//services/service') if s.text]
                    if products:
                        context_parts.append(f"Products: {', '.join(products[:5])}")
                    if services:
                        context_parts.append(f"Services: {', '.join(services[:5])}")
                    terminology = [t.text for t in root.findall('.//terminology/term') if t.text]
                    if terminology:
                        context_parts.append(f"Key Terminology: {', '.join(terminology[:10])}")
                except Exception as e:
                    logger.warning(f'Failed to load company_profile.xml: {e}')
            website_data_path = client_domain_knowledge_dir / 'website_data.xml'
            if website_data_path.exists():
                try:
                    website_xml = load_xml_prompt_raw(website_data_path)
                    from defusedxml.ElementTree import fromstring
                    root = fromstring(website_xml)
                    summary_description = root.find('.//summary_description')
                    if summary_description is not None and summary_description.text:
                        context_parts.append(f'Website Summary:\n{summary_description.text.strip()}')
                        logger.info(f"Loaded website_data.xml summary for client '{self.client_id}'")
                except Exception as e:
                    logger.warning(f'Failed to load website_data.xml: {e}')
            for file_path in client_domain_knowledge_dir.glob('*.xml'):
                if file_path.name not in ['company_profile.xml', 'website_data.xml']:
                    try:
                        file_xml = load_xml_prompt_raw(file_path)
                        domain_knowledge[file_path.stem] = file_xml
                        logger.info(f"Loaded domain knowledge file: {file_path.name} for client '{self.client_id}'")
                    except Exception as e:
                        logger.warning(f'Failed to load domain knowledge file {file_path.name}: {e}')
            if domain_knowledge:
                if use_knowledge_filtering and query:
                    domain_knowledge = filter_domain_knowledge_by_query(domain_knowledge, query)
                if USE_KNOWLEDGE_SUMMARIZATION:
                    domain_knowledge = {key: summarize_domain_knowledge_for_prompt(content) for key, content in domain_knowledge.items()}
                for key, content in domain_knowledge.items():
                    context_parts.append(f'\n{key}:\n{content}')
            try:
                profile_path = Path(PROJECT_ROOT) / 'xml_prompts' / 'clients' / self.client_id / 'data_sources' / 'meta_information' / 'client_data_profile.xml'
                if profile_path.exists():
                    from defusedxml.ElementTree import parse as _parse_xml
                    tree = _parse_xml(str(profile_path))
                    root = tree.getroot()
                    profile_parts = []
                    geo = root.find('.//geography')
                    if geo is not None and geo.text:
                        profile_parts.append(f'Geography: {geo.text}')
                    nf = root.find('.//number_format')
                    if nf is not None:
                        profile_parts.append(f"Number format: {nf.get('system', '')} (e.g. {nf.get('example', '')})")
                    cur = root.find('.//currency')
                    if cur is not None:
                        profile_parts.append(f"Currency: {cur.get('code', '')} ({cur.get('symbol', '')})")
                    date_fmts = root.find('.//date_formats')
                    if date_fmts is not None:
                        first_fmt = date_fmts.find('format')
                        if first_fmt is not None:
                            profile_parts.append(f"Date format: Always format dates as {first_fmt.get('pattern', 'DD/MM/YYYY')} in your responses")
                    fy = root.find('.//fiscal_year')
                    if fy is not None:
                        profile_parts.append(f"Fiscal year: {fy.get('label', '')} (starts month {fy.get('start_month', '')})")
                    if profile_parts:
                        context_parts.append('Data Profile:\n' + '\n'.join(profile_parts))
                        logger.info('Loaded data profile for business agent (client: %s)', self.client_id)
            except Exception as e:
                logger.debug('Data profile loading skipped in business agent: %s', e)
            try:
                schema_mapper = SchemaMapper.get_sync(self.client_id, self.db)
                number_config = schema_mapper.get_number_format_config()
                date_config = schema_mapper.get_date_format_config()
                display_pref_parts = []
                if date_config.get('date_format'):
                    display_pref_parts.append(f"IMPORTANT: Always format all dates as {date_config['date_format']} in your responses.")
                if number_config.get('currency_symbol'):
                    display_pref_parts.append(f"Use {number_config['currency_symbol']} as the currency symbol.")
                if display_pref_parts:
                    context_parts.append('Display Preferences (admin-configured):\n' + '\n'.join(display_pref_parts))
            except Exception as e:
                logger.debug('Display preferences loading skipped in business agent: %s', e)
            if context_parts:
                logger.info(f'Injected company context from domain knowledge for Business Agent (client: {self.client_id})')
                return '\n'.join(context_parts)
            return None
        except Exception as e:
            logger.warning(f'Failed to load company context for Business Agent: {e}')
            return None

    def _load_file_content(self, file_path: Optional[Path], error_msg: str, default_content: str='') -> str:
        if not file_path or not os.path.exists(file_path):
            logger.warning(f'File not found: {file_path}')
            return default_content
        try:
            if file_path.suffix.lower() == '.xml':
                logger.info(f'Loading XML content from {file_path} (client: {self.client_id})')
                if self.db is not None:
                    relative_path = self._get_relative_path(file_path)
                    try:
                        import asyncio
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            return load_xml_prompt_raw(file_path)
                        else:
                            return loop.run_until_complete(load_client_prompt(relative_path, self.client_id, self.db, use_formatting=False))
                    except Exception as e:
                        logger.warning(f'Client-aware loading failed, falling back to base: {e}')
                        return load_xml_prompt_raw(file_path)
                else:
                    return load_xml_prompt_raw(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    logger.info(f'Loaded content from {file_path}')
                    return f.read()
        except Exception as e:
            logger.error(f'{error_msg}: {e}')
            return default_content

    @classmethod
    def is_agent_initialized(cls, prompt_file_path: Optional[str]=None, output_dir: Optional[str]=None, client_id: str=None) -> bool:
        if not client_id:
            raise ValueError('client_id is REQUIRED')
        config = AGENT_CONFIG.get('business_agent', {})
        prompt_path = prompt_file_path or config.get('prompt_file')
        out_dir = output_dir or config.get('output_dir')
        key_elements = {'prompt_file_path': str(prompt_path) if prompt_path else None, 'output_dir': str(out_dir) if out_dir else None, 'client_id': client_id}
        config_hash = hash(str(sorted(key_elements.items())))
        agent_key = f'business_agent_{client_id}_{config_hash}'
        return agent_key in cls._AGENT_INITIALIZED

    @classmethod
    def clear_cache(cls) -> None:
        cls._AGENT_INITIALIZED.clear()
        cls._CACHED_PROMPTS.clear()
        logger.info('Cleared all Business Agent initialization cache.')

    @classmethod
    def force_reinitialize(cls, prompt_file_path: Optional[str]=None, output_dir: Optional[str]=None, client_id: str=None, db: Any=None):
        if not client_id:
            raise ValueError('client_id is REQUIRED')
        cls.clear_cache()
        return cls(prompt_file_path, output_dir, client_id, db)

    def _summarize_planner_context(self, planner_response: Optional[Dict[str, Any]]) -> str:
        if not planner_response:
            return 'Planner Context: [Not available for this task.]\n---\n'
        original_query = planner_response.get('user_question', '')
        plan_text = planner_response.get('plan', '')
        parts = []
        if original_query:
            parts.append(f'Original User Query: {original_query}')
        if plan_text:
            parts.append(f'Analysis Plan:\n{plan_text}')
        if parts:
            return 'Planner Context:\n' + '\n\n'.join(parts) + '\n---\n'
        return 'Planner Context: [Planner response available but empty.]\n---\n'

    def _format_text_outputs(self, text_outputs: List[Dict[str, Any]]) -> List[str]:
        lines = ['Text Outputs:']
        for item in text_outputs:
            value = str(item.get('value', 'N/A'))
            lines.append(f"  - Name: {item.get('name', 'N/A')}, Value: {value}")
        return lines

    def _format_dataframes(self, dataframes: List[Dict[str, Any]], *, max_rows: int=5000, display_map: Optional[Dict[str, str]]=None) -> List[str]:
        import json
        import logging
        logger = logging.getLogger(__name__)
        if display_map is None:
            display_map = {}
        lines: List[str] = ['DataFrames Generated:']
        for item in dataframes:
            name = item.get('name', 'N/A')
            lines.append(f'  - Dataset Name: {name}')
            json_data = item.get('json_data')
            raw_data = item.get('data')
            if json_data:
                try:
                    df_dict = json.loads(json_data)
                except json.JSONDecodeError as e:
                    logger.error(f'Error parsing JSON for DataFrame {name}: {e}')
                    lines.append('    Error: Could not parse DataFrame JSON.')
                    continue
                columns = df_dict.get('columns', [])
                rows = df_dict.get('data', [])
            elif raw_data and isinstance(raw_data, list) and raw_data:
                first = raw_data[0]
                if isinstance(first, dict):
                    columns = list(first.keys())
                    rows = [[r.get(c) for c in columns] for r in raw_data]
                else:
                    lines.append('    Content: No structured data available.')
                    continue
            else:
                lines.append('    Content: No data available.')
                continue
            if not columns or not rows:
                continue

            def pretty_label(col: str) -> str:
                return display_map.get(col, col)
            key_col = columns[0]
            value_col = columns[1] if len(columns) > 1 else None
            for row in rows[:max_rows]:
                padded = list(row) + [None] * max(0, len(columns) - len(row))
                key_val = padded[0]
                key_part = f'{pretty_label(key_col)}="{key_val}"' if isinstance(key_val, str) else f'{pretty_label(key_col)}={key_val}'
                parts = [key_part]
                if value_col is not None:
                    parts.append(f'{pretty_label(value_col)}={padded[1]}')
                for col_name, col_val in zip(columns[2:], padded[2:]):
                    if isinstance(col_val, str):
                        parts.append(f'{pretty_label(col_name)}="{col_val}"')
                    else:
                        parts.append(f'{pretty_label(col_name)}={col_val}')
                lines.append('   ' + ', '.join(parts))
            if len(rows) > max_rows:
                lines.append(f'   ... ({len(rows) - max_rows} more rows)')
        return lines

    def _format_plotly_charts(self, plotly_charts: List[Dict[str, Any]]) -> List[str]:
        lines = ['Interactive Plotly Charts Generated:']
        for chart in plotly_charts:
            name = chart.get('name', 'N/A')
            lines.append(f'  - Name: {name}')
            figure_data = chart.get('figure') or chart.get('data')
            if figure_data:
                try:
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
                    logger.error(f'Error parsing Plotly chart {name}: {e}')
                    lines.append('    Error: Could not parse Plotly figure.')
            else:
                lines.append('    Content: No figure data available.')
        return lines

    def _summarize_executor_results(self, executor_results: Dict[str, Any]) -> str:
        summary_parts = []
        artifact_registry = executor_results.get('artifact_registry')
        if artifact_registry and isinstance(artifact_registry, list):
            summary_parts.append('ANALYSIS CHAIN (steps the data agent performed):')
            for entry in artifact_registry:
                iter_num = entry.get('iteration', '?')
                reason = entry.get('reasoning', '')
                new_vars = entry.get('new_variables', {})
                vars_desc = []
                for vname, vinfo in new_vars.items():
                    if isinstance(vinfo, dict) and vinfo.get('type') == 'DataFrame':
                        shape = vinfo.get('shape', [])
                        shape_str = f'{shape[0]}×{shape[1]}' if len(shape) >= 2 else str(shape)
                        vars_desc.append(f'{vname} ({shape_str})')
                    else:
                        vars_desc.append(vname)
                vars_str = ', '.join(vars_desc) if vars_desc else ''
                line = f'  Step {iter_num}: {reason}'
                if vars_str:
                    line += f' → {vars_str}'
                summary_parts.append(line)
            summary_parts.append('')
        if executor_results.get('summary') and isinstance(executor_results.get('summary'), dict):
            s = executor_results['summary']
            meta = s.get('meta', {}) or {}
            metrics = s.get('metrics', {}) or {}
            top = s.get('top', {}) or {}
            rollups = s.get('group_rollups', {}) or {}
            desc = (s.get('descriptive', {}) or {}).get('numeric_columns', {}) or {}
            chart_specs = s.get('chart_specs', []) or []
            sample = s.get('sample', {}) or {}
            row_count = meta.get('row_count', 0)
            if row_count == 0:
                summary_parts.append('Data Analysis Result:')
                summary_parts.append('  - No data found matching the specified criteria')
                notes = meta.get('notes', [])
                if notes:
                    summary_parts.append(f"  - Note: {'; '.join(notes)}")
                summary_parts.append('\nPlease verify:')
                summary_parts.append('  1. The facility/location name is spelled correctly')
                summary_parts.append('  2. The facility belongs to your organization')
                summary_parts.append('  3. The data filters are appropriate for your dataset')
                return '\n'.join(summary_parts)
            summary_parts.append('Summary (Summarizer present):')
            if meta:
                primary = meta.get('primary_frame', '')
                row_count = meta.get('row_count', '')
                notes = '; '.join(meta.get('notes', []) or [])
                summary_parts.append(f'  - Primary Frame: {primary} | Rows: {row_count}')
                if notes:
                    summary_parts.append(f'  - Notes: {notes}')
            if metrics:
                summary_parts.append('Metrics:')
                for k, v in metrics.items():
                    summary_parts.append(f'  - {k}: {v}')
            try:
                schema_mapper = SchemaMapper.get_sync(self.client_id, self.db)
                grouping_dimensions = schema_mapper.get_grouping_dimensions()
                metric_columns = schema_mapper.get_metric_columns()
                value_col_name = metric_columns.get('primary_value') or metric_columns.get('value', 'VALUE')
                qty_col_name = metric_columns.get('primary_quantity') or metric_columns.get('qty', 'QTY')
                logger.info(f"[BusinessAgent] Using SchemaMapper for '{self.client_id}': grouping_dimensions={list(grouping_dimensions.keys())}, value_col={value_col_name}, qty_col={qty_col_name}")
            except Exception as e:
                logger.warning(f"[BusinessAgent] Error loading schema for '{self.client_id}': {e}, using generic fallback")
                grouping_dimensions = {}
                value_col_name = 'VALUE'
                qty_col_name = 'QTY'

            def _emit_rollup(title: str, recs, key: str, display_name: str, value_col: str, qty_col: str):
                if recs:
                    summary_parts.append(title)
                    for r in recs:
                        name = r.get(key, '')
                        value = r.get(value_col, '')
                        qty = r.get(qty_col, '')
                        cnt = r.get('count', '')
                        parts = [f'{display_name}={name}']
                        if value:
                            parts.append(f'{value_col}={value}')
                        if qty:
                            parts.append(f'{qty_col}={qty}')
                        if cnt:
                            parts.append(f'count={cnt}')
                        summary_parts.append(f"  - {', '.join(parts)}")
            for dim_key, rollup_data in rollups.items():
                if rollup_data:
                    if dim_key in grouping_dimensions:
                        dim_info = grouping_dimensions[dim_key]
                        col_name = dim_info['physical_name']
                        display_name = dim_info['display_name']
                    else:
                        col_name = dim_key.replace('by_', '').upper()
                        display_name = dim_key.replace('by_', '').replace('_', ' ').title()
                    _emit_rollup(f'Rollup by {display_name} (Top):', rollup_data, col_name, display_name, value_col_name, qty_col_name)
            if desc:
                summary_parts.append('Descriptive (numeric columns):')
                for col, stats in desc.items():
                    min_v = stats.get('min')
                    max_v = stats.get('max')
                    mean_v = stats.get('mean')
                    med_v = stats.get('median')
                    std_v = stats.get('std')
                    summary_parts.append(f'  - {col}: min={min_v}, max={max_v}, mean={mean_v}, median={med_v}, std={std_v}')
            if chart_specs:
                summary_parts.append('Chart Specs:')
                for cs in chart_specs:
                    title = cs.get('title', '')
                    xtype = cs.get('x', '')
                    ytype = cs.get('y', '')
                    summary_parts.append(f'  - {title} (x={xtype}, y={ytype})')
            if executor_results.get('charts'):
                summary_parts.extend(self._format_plotly_charts(executor_results['charts']))
            return '\n'.join(summary_parts)
        if executor_results.get('text_outputs'):
            summary_parts.extend(self._format_text_outputs(executor_results['text_outputs']))
        if executor_results.get('dataframes'):
            summary_parts.extend(self._format_dataframes(executor_results['dataframes']))
        if executor_results.get('plotly_charts'):
            summary_parts.extend(self._format_plotly_charts(executor_results['plotly_charts']))
        if executor_results.get('matplotlib_image_paths'):
            summary_parts.append('Static Matplotlib Images Generated:')
            for path in executor_results['matplotlib_image_paths']:
                summary_parts.append(f'  - Image Path: {path}')
        return '\n'.join(summary_parts) or 'No specific data outputs were generated.'

    def _replace_schema_placeholders(self, prompt: str) -> str:
        try:
            if self.db is not None:
                schema_mapper = SchemaMapper.get_sync(self.client_id, self.db)
                metric_columns = schema_mapper.get_metric_columns()
                col_closing_value = metric_columns.get('primary_value') or metric_columns.get('value', 'VALUE')
                col_available_qty = metric_columns.get('primary_quantity') or metric_columns.get('qty', 'QTY')
                try:
                    if 'primary_value' not in metric_columns:
                        col_closing_value = schema_mapper.get_column('closing_value')
                except ValueError:
                    pass
                try:
                    if 'primary_quantity' not in metric_columns:
                        col_available_qty = schema_mapper.get_column('available_quantity')
                except ValueError:
                    pass
                grouping_dimensions = schema_mapper.get_grouping_dimensions()
                col_aging_bucket = None
                col_inventory_group = None
                col_organization = None
                display_aging_bucket = None
                display_inventory_group = None
                display_organization = None
                for dim_key, dim_info in grouping_dimensions.items():
                    logical_name = dim_info.get('logical_name', '')
                    if logical_name in ['aging_bucket', 'segment'] or 'slab' in dim_key or 'segment' in dim_key:
                        col_aging_bucket = dim_info['physical_name']
                        display_aging_bucket = dim_info['display_name']
                    elif logical_name in ['inventory_group', 'category'] or 'group' in dim_key:
                        col_inventory_group = dim_info['physical_name']
                        display_inventory_group = dim_info['display_name']
                    elif logical_name in ['organization', 'entity'] or 'site' in dim_key or 'entity' in dim_key:
                        col_organization = dim_info['physical_name']
                        display_organization = dim_info['display_name']
                if not col_aging_bucket:
                    try:
                        col_aging_bucket = schema_mapper.get_column('aging_bucket')
                        display_aging_bucket = schema_mapper.get_display_name('aging_bucket')
                    except ValueError:
                        col_aging_bucket = 'SEGMENT'
                        display_aging_bucket = 'Segment'
                if not col_inventory_group:
                    try:
                        col_inventory_group = schema_mapper.get_column('inventory_group')
                        display_inventory_group = schema_mapper.get_display_name('inventory_group')
                    except ValueError:
                        col_inventory_group = 'CATEGORY'
                        display_inventory_group = 'Category'
                if not col_organization:
                    try:
                        col_organization = schema_mapper.get_column('organization')
                        display_organization = schema_mapper.get_display_name('organization')
                    except ValueError:
                        col_organization = 'ENTITY_NAME'
                        display_organization = 'Entity'
                prompt = prompt.replace('{COLUMN_CLOSING_VALUE}', col_closing_value)
                prompt = prompt.replace('{COLUMN_AVAILABLE_QTY}', col_available_qty)
                prompt = prompt.replace('{COLUMN_AGING_BUCKET}', col_aging_bucket)
                prompt = prompt.replace('{COLUMN_INVENTORY_GROUP}', col_inventory_group)
                prompt = prompt.replace('{COLUMN_ORGANIZATION}', col_organization)
                prompt = prompt.replace('{DISPLAY_AGING_BUCKET}', display_aging_bucket)
                prompt = prompt.replace('{DISPLAY_INVENTORY_GROUP}', display_inventory_group)
                prompt = prompt.replace('{DISPLAY_ORGANIZATION}', display_organization)
                logger.info(f"Replaced schema placeholders for client '{self.client_id}'")
            else:
                pass
        except Exception as e:
            logger.error(f'Error replacing schema placeholders: {e}', exc_info=True)
        return prompt

    def _build_system_prompt(self, query: str='', use_knowledge_filtering: bool=False) -> Tuple[str, Dict[str, Any], str]:
        prompt_with_placeholders = self.prompt
        company_context = self._load_company_context(query=query, use_knowledge_filtering=use_knowledge_filtering)
        group_rollup_info = []
        try:
            if self.db is not None:
                schema_mapper = SchemaMapper.get_sync(self.client_id, self.db)
                grouping_dimensions = schema_mapper.get_grouping_dimensions()
                for dim_key, dim_info in grouping_dimensions.items():
                    col_name = dim_info.get('physical_name', dim_key)
                    group_rollup_info.append(f'{dim_key} ({col_name})')
                if not group_rollup_info:
                    group_rollup_info = ['by_category (CATEGORY)', 'by_segment (SEGMENT)', 'by_entity (ENTITY_NAME)']
            else:
                group_rollup_info = ['by_category (CATEGORY)', 'by_segment (SEGMENT)', 'by_entity (ENTITY_NAME)']
        except Exception as e:
            logger.warning(f'Error getting grouping dimensions for summarizer appendix: {e}')
            group_rollup_info = ['by_category (CATEGORY)', 'by_segment (SEGMENT)', 'by_entity (ENTITY_NAME)']
        group_rollup_str = ', '.join(group_rollup_info)
        summarizer_appendix = f'\n\nSummarizer Schema (Read-Only)\n- Do NOT compute any values. Cite only what is present.\n- If a summarizer is present, expect the following keys in the payload you summarize from:\n  - summary.meta: metadata (row_count, notes, primary_frame).\n  - summary.metrics: explicit totals/aggregates (e.g., total_value, total_quantity, avg_value, median_value).\n  - summary.top: top lists such as items_by_value.\n  - summary.group_rollups: {group_rollup_str}.\n  - summary.descriptive.numeric_columns: basic stats (min, max, mean, median, std).\n  - summary.chart_specs: lightweight chart suggestions (title, x, y).\n\nNumeric formatting rule:\n  - Copy numbers EXACTLY as they appear in inputs (including commas/decimal places). Do not reformat.\n'
        if 'CUSTOM OVERRIDES' not in prompt_with_placeholders:
            try:
                from util.xml_prompt_loader import load_custom_prompts
                custom_prompts = load_custom_prompts(self.client_id)
                if custom_prompts:
                    logger.info(f'Appending custom prompts to business agent (safety net) | client_id={self.client_id}')
                    prompt_with_placeholders += '\n\n' + custom_prompts
            except Exception as e:
                logger.warning(f'Failed to load custom prompts for business agent | client_id={self.client_id} | error={e}')
        knowledge_metrics = {'use_knowledge_filtering': use_knowledge_filtering, 'company_context_chars': len(company_context) if company_context else 0, 'company_context_tokens_est': len(company_context) // 4 if company_context else 0}
        return (prompt_with_placeholders + summarizer_appendix, knowledge_metrics, company_context or '')
    _TECHNICAL_PATTERNS = re.compile('((?:Overlap|overlap)\\s+\\w+\\s+vs\\s+\\w+|Filtered\\s+\\w+\\s+using\\s+\\w+|(?:Merge|merge|Join|join)\\w*\\s+(?:on|using|with)\\s|\\.shape\\s*[:=]|(?:columns?|dtypes?|dtype)\\s*[:=]|(?:Index|RangeIndex|Int64Index)\\(|(?:Loading|Reading)\\s+(?:parquet|csv|excel|file)|\\.(?:head|tail|describe|info)\\(\\)|(?:rows?|records?)\\s*(?:×|x)\\s*\\d+\\s*(?:columns?|cols?)|(?:KeyError|ValueError|TypeError|AttributeError|NameError)\\b|common\\s+values?\\s*(?:found|:)|^\\s*\\d+\\s+rows?\\s*$)', re.IGNORECASE | re.MULTILINE)

    def _strip_technical_noise(self, console_output: str) -> str:
        if not console_output:
            return ''
        cleaned_lines = []
        for line in console_output.split('\n'):
            stripped = line.strip()
            if stripped.startswith('--- Step'):
                cleaned_lines.append(line)
                continue
            if not stripped:
                continue
            if self._TECHNICAL_PATTERNS.search(stripped):
                continue
            words = stripped.split()
            if words and len(words) <= 10:
                upper_count = sum((1 for w in words if re.match('^[A-Z][A-Z_]{2,}$', w)))
                if upper_count > len(words) * 0.5:
                    continue
            cleaned_lines.append(line)
        result = []
        for i, line in enumerate(cleaned_lines):
            if line.strip().startswith('--- Step'):
                has_content = False
                for j in range(i + 1, len(cleaned_lines)):
                    if cleaned_lines[j].strip():
                        has_content = not cleaned_lines[j].strip().startswith('--- Step')
                        break
                if has_content:
                    result.append(line)
            else:
                result.append(line)
        return '\n'.join(result).strip()

    def _build_user_message(self, *, query: str, plan_text: str, analyst_findings: str, execution_summary: str, console_output: str, reference_guidance: str, business_insights_sections: Dict[str, bool], company_context: str='', persona_guidance: str='') -> str:

        def _truncate(text: str, max_chars: int, label: str) -> str:
            if not text:
                return ''
            if len(text) <= max_chars:
                return text
            logger.info('[PromptBudget] business truncating %s from %d to %d chars', label, len(text), max_chars)
            return text[:max_chars] + f'\n... [{label} truncated for latency budget]'
        query = _truncate(query, 500, 'question')
        plan_text = _truncate(plan_text, 4000, 'analysis_plan')
        analyst_findings = _truncate(analyst_findings, 8000, 'analyst_findings')
        execution_summary = _truncate(execution_summary, 12000, 'data_evidence')
        console_output = _truncate(console_output, 6000, 'process_log')
        reference_guidance = _truncate(reference_guidance, 3000, 'reference_guidance')
        company_context = _truncate(company_context, 2000, 'company_context')
        analysis_structured_parts = []
        if business_insights_sections.get('metrics', True):
            analysis_structured_parts.append('    "metrics": [\n      "Key quantitative fact with context (e.g., \'Total value: ₹69.26 Cr across 2,208 records\')",\n      "Comparative metric if available (e.g., \'Category A accounts for 58% of total\')",\n      "Distribution metric (e.g., \'Majority (65%) concentrated in top segment\')"\n    ]')
        if business_insights_sections.get('insights', True):
            analysis_structured_parts.append('    "insights": [\n      "Business implication: [Observation] + [Why it matters] + [Impact]",\n      "Pattern or anomaly with business impact",\n      "Risk or opportunity identified from the data"\n    ]')
        if business_insights_sections.get('summary', True):
            analysis_structured_parts.append('    "summary": "One executive-level sentence answering the question with critical takeaway"')
        if business_insights_sections.get('note', True):
            analysis_structured_parts.append('    "note": "Optional context about limitations or caveats (only if relevant)"')
        analysis_structured_json = '{\n' + ',\n'.join(analysis_structured_parts) + '\n            }' if analysis_structured_parts else '{}'
        root_sections = []
        if business_insights_sections.get('recommendations', True):
            root_sections.append('  "recommendations": [\n    "Immediate action with expected impact",\n    "Process improvement with rationale",\n    "Strategic initiative for long-term value"\n  ]')
        if business_insights_sections.get('follow_ups', True):
            root_sections.append('  "follow_ups": [\n    "Drill-down question for deeper analysis",\n    "Trend analysis question",\n    "Comparative question across segments"\n  ]')
        kpi_sections = {}
        if business_insights_sections.get('metrics', True):
            kpi_sections['"metrics"'] = '[\n    {"label": "2-3 word label", "value": "Exact number or % from bullets", "sub": "One short context phrase", "color": "red|orange|blue|green"}\n  ]'
        if business_insights_sections.get('insights', True):
            kpi_sections['"insights"'] = '[\n    {"label": "2-3 word label", "value": "New number not in metrics chips", "sub": "One short context phrase", "color": "red|orange|blue"}\n  ]'
        if business_insights_sections.get('recommendations', True):
            kpi_sections['"recommendations"'] = '[\n    {"label": "count or scope of actions", "value": "e.g. \\"3 Actions\\" or \\"2 Areas\\"", "sub": "urgency or timeframe (e.g. \\"immediate priority\\")", "color": "red|orange|blue|green"}\n  ]'
        if kpi_sections:
            kpi_inner = ',\n  '.join((f'{k}: {v}' for k, v in kpi_sections.items()))
            root_sections.append(f'  "section_kpis": {{\n  {kpi_inner}\n  }}')
        root_sections_json = ',\n'.join(root_sections) if root_sections else ''
        json_structure = '{\n'
        if analysis_structured_parts:
            json_structure += '            "analysis_structured": ' + analysis_structured_json
            if root_sections_json:
                json_structure += ',\n'
        if root_sections_json:
            json_structure += root_sections_json
        json_structure += '\n            }'
        guidelines = []
        if business_insights_sections.get('summary', True):
            guidelines.append('**Summary** - Executive punchline:\n- Answer the question in ONE direct sentence\n- Lead with the key number in **bold**: e.g. "**43.8 Cr** units in Phulpur"\n- Add context: vs total, vs average, trend direction\n- A CXO should get the answer in 5 seconds')
        if business_insights_sections.get('metrics', True):
            guidelines.append('**Metrics** - Numbers that matter:\n- Lead every metric with the number in **bold**: e.g. "**43.8 Cr** units across 8 locations"\n- Always add context: % of total, vs average, rank, or trend\n- Not just "Total: 43.8 Cr" but "**43.8 Cr** total inventory — 12% of network capacity"\n- Cite EXACTLY as shown in data (no reformatting)\n- 2-4 metrics max — only the ones a CXO would care about')
        if business_insights_sections.get('insights', True):
            guidelines.append('**Insights** - Business implications with teeth:\n- Structure: [What you see] + [Why it matters] + [What to do about it]\n- Quantify impact: e.g. "Concentration of **65%** in one warehouse creates supply chain risk"\n- Use strong language when warranted: "requires attention", "significant exposure", "opportunity to optimize"\n- Connect to business outcomes: cost, revenue, risk, efficiency\n- 2-3 insights max — quality over quantity')
        if business_insights_sections.get('recommendations', True):
            guidelines.append('**Recommendations** - Monday morning actions:\n- Specific enough to act on: not "improve inventory" but "review slow-moving items in warehouse exceeding 90-day holding"\n- Include expected impact where possible\n- Prioritize: immediate action → short-term improvement → strategic initiative\n- 2-3 recommendations max')
        if business_insights_sections.get('follow_ups', True):
            guidelines.append('**Follow-ups** - Smart exploration:\n- Drill-downs, trends, comparisons\n- Specific and answerable with available data\n- Avoid generic "tell me more" ')
        guidelines.append('**section_kpis** - Compact KPI chips shown above each section header. Follow these strict per-section rules:\n\nmetrics chips (2-3 max):\n- value = a standalone number/% directly from the metrics bullets (e.g. "100%", "37,904", "-912")\n- label = what that number measures in 2-3 words (e.g. "Products in Deficit", "Pending Demand")\n- sub = one short phrase of context (e.g. "unable to meet demand")\n- color: red if negative/deficit/risk, orange if gap/shortage/warning, blue if neutral count, green if positive/achieved\n\ninsights chips (1-2 max):\n- NEVER repeat a number already used in metrics chips\n- value = a scope/scale figure that shows business impact (e.g. "100%", "2 SKUs", "All Categories")\n- If no genuinely new quantitative insight exists, return empty [] — do NOT invent or reuse numbers\n- color: red if systemic failure, orange if concentrated risk, blue if pattern observation\n\nrecommendations chips (1-2 max):\n- Do NOT use action verb labels like "Reorder" or "Review" as the value\n- value = count or scope: how many actions, categories, or areas need attention (e.g. "3 Actions", "2 Categories", "All SKUs")\n- label = what those actions target in 2-3 words (e.g. "Need Reorder", "Require Audit", "At Risk")\n- sub = urgency or timeframe (e.g. "immediate priority", "within 48 hours", "before next cycle")\n- color: red if critical/immediate, orange if high priority, blue if planned, green if preventive')
        guidelines_text = '\n\n'.join(guidelines) if guidelines else ''
        guidelines_section = ''
        if guidelines_text:
            guidelines_section = f'INSIGHT GENERATION GUIDELINES:\n\n{guidelines_text}\n\n'
        question_section = f'USER QUESTION: "{query}"\n' if query else ''
        plan_section = ''
        if plan_text:
            plan_section = f'\nANALYSIS PLAN:\n{plan_text}\n'
        findings_section = ''
        if analyst_findings:
            findings_section = f"\nANALYST'S ANSWER (use this as your primary source — this is the data analyst's own summary):\n{analyst_findings}\n"
        data_section = f'\nDATA EVIDENCE:\n{execution_summary}\n' if execution_summary else ''
        process_section = ''
        if console_output and console_output.strip():
            process_section = f'\nPROCESS LOG (background reference only — extract numbers and results, IGNORE the analytical process):\n{console_output}\n'
        persona_block = ''
        if persona_guidance:
            persona_block = f'\n--- AGENT PERSONA (use this vocabulary and framing in EVERY section below) ---\n{persona_guidance}\n--- END PERSONA ---\n\n'
        base_message = f"""You are an expert business analyst translating data analysis results into CXO-grade insights.\nYour job: take the FINAL ANSWER from the analysis and present it as a strategic business insight.\nYou are NOT describing what the analyst did. You ARE answering the user's question with impact.\n\n{question_section}{plan_section}{findings_section}{data_section}{process_section}{persona_block}Produce your response in TWO parts, in this EXACT order:\n\nPART 1 — NARRATIVE (plain markdown prose, shown live to the user as you write it):\nWrite a concise, CXO-grade narrative (2-4 short paragraphs, or a few bullet points) that DIRECTLY answers the user's question using the actual numbers and the key insights. Use **bold** around key numbers. This part is readable markdown prose — NOT JSON, no key names, no braces.\n\nPART 2 — STRUCTURED DATA:\nOn its own new line, output this EXACT sentinel and nothing else on that line:\n===STRUCTURED===\nImmediately after the sentinel line, output ONLY a single valid JSON object with EXACTLY these keys and structure:\n{json_structure}\n\nEVERY section shown above is MANDATORY. Do NOT omit any key. If a section has less to say, still provide at least one meaningful item. An empty array [] or empty string "" for any section is a FAILURE.\n\n{guidelines_section}CRITICAL CONSTRAINTS:\n- ANSWER THE QUESTION. The summary MUST directly answer what was asked with the actual number/finding.\n- ALL sections (summary, metrics, insights, recommendations, follow_ups) MUST be populated — never skip any.\n- Use **bold** markers around key numbers: e.g. "**43.8 Cr** total inventory at Phulpur"\n- Do NOT compute new numbers (cite only what exists in data)\n- Do NOT reformat numbers (copy EXACTLY as shown)\n- Do NOT make assumptions beyond provided data\n- Do NOT use vague language - use specific numbers\n- PART 1 is markdown prose; the narrative and the JSON must AGREE (same numbers, same findings)\n- After the ===STRUCTURED=== sentinel, PART 2 must be VALID JSON only (no markdown fences, no extra text)\n- If data is empty, acknowledge clearly in summary but still populate all sections\n- ABSOLUTELY NEVER mention technical/internal details: column names (ORGANIZATION_ID, INV_ORG_ID, etc.),\n  join operations, merge operations, data normalization, table structures, data retrieval methods,\n  parquet files, DataFrames, filtering steps, or any aspect of HOW the data was processed.\n  A CXO does not care about the analytical process. Focus ONLY on the business answer.\n"""
        if company_context:
            base_message += f'\nCompany Context:\n{company_context}\n\nUse company-specific terminology and consider industry context in your insights.\n'
        if reference_guidance:
            base_message += f'\n**ADDITIONAL GUIDANCE** (apply within the required JSON structure above — do NOT omit any sections):\n{reference_guidance}\n'
        logger.info('[PromptSize] business user_message_chars=%d | question=%d | plan=%d | findings=%d | evidence=%d | process=%d | reference=%d | persona=%d', len(base_message), len(query or ''), len(plan_text or ''), len(analyst_findings or ''), len(execution_summary or ''), len(console_output or ''), len(reference_guidance or ''), len(persona_guidance or ''))
        return base_message
    _NUM_TOKEN_PATTERN = re.compile('(?:₹)?\\d{1,3}(?:,\\d{2,3})*(?:\\.\\d+)?|\\d+(?:\\.\\d+)?')

    def _extract_numeric_tokens(self, text: str) -> List[str]:
        if not text:
            return []
        return self._NUM_TOKEN_PATTERN.findall(text)

    def _build_numeric_whitelist(self, execution_summary: str, console_output: str, executor_results: Dict[str, Any]) -> None:
        tokens = set()
        for src in [execution_summary or '', console_output or '']:
            for t in self._extract_numeric_tokens(src):
                tokens.add(t)
        try:
            summary = executor_results.get('summary') if isinstance(executor_results, dict) else None
            if isinstance(summary, dict):
                try:
                    schema_mapper = SchemaMapper.get_sync(self.client_id, self.db)
                    metric_columns = schema_mapper.get_metric_columns()
                    value_col_name = metric_columns.get('primary_value') or metric_columns.get('value', 'VALUE')
                    qty_col_name = metric_columns.get('primary_quantity') or metric_columns.get('qty', 'QTY')
                except Exception:
                    value_col_name = 'VALUE'
                    qty_col_name = 'QTY'
                for v in (summary.get('metrics') or {}).values():
                    tokens.update(self._extract_numeric_tokens(str(v)))
                for rec in (summary.get('top') or {}).get('items_by_value', []) or []:
                    for k, v in rec.items():
                        if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit()):
                            tokens.update(self._extract_numeric_tokens(str(v)))
                group_rollups = summary.get('group_rollups', {}) or {}
                for rollup_key, rollup_data in group_rollups.items():
                    for rec in rollup_data or []:
                        for k, v in rec.items():
                            if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit()):
                                tokens.update(self._extract_numeric_tokens(str(v)))
                for rec in (summary.get('descriptive', {}) or {}).get('numeric_columns', {}) or {}.values():
                    for k in ('min', 'max', 'mean', 'median', 'std'):
                        if k in rec:
                            tokens.update(self._extract_numeric_tokens(str(rec[k])))
        except Exception:
            pass
        self._allowed_numeric_tokens = tokens

    def _all_tokens_allowed(self, text: str) -> bool:
        nums = self._extract_numeric_tokens(text)
        if not nums:
            return True
        return all((n in self._allowed_numeric_tokens for n in nums))
    _LEADING_REQUEST_PATTERN = re.compile('^(?:would you like|should we|do you want(?: to)?|can i|shall we|could we|would you want|do you wish)\\b[\\s,:-]*', re.IGNORECASE)

    def _normalize_followup(self, q: str) -> str:
        if not isinstance(q, str):
            return ''
        s = q.strip()
        s = re.sub('\\*+', '', s)
        s = self._LEADING_REQUEST_PATTERN.sub('', s)
        if s.lower().startswith('to '):
            s = s[3:]
        if s and (not s.endswith('?')):
            s = s.rstrip('.') + '?'
        if s:
            s = s[0].upper() + s[1:]
        return s

    @traceable(name='narrator_process')
    async def process(self, executor_results: Dict[str, Any], planner_response: Optional[Dict[str, Any]]=None, reference_guidance: str='', use_knowledge_filtering: bool=False, persona_guidance: str='') -> AsyncGenerator[Tuple[str, Optional[Dict[str, Any]]], None]:
        try:
            try:
                from db_config.database import get_db
                db = get_db()
                config_manager = ClientConfigManager(db)
                client_config = await config_manager.get_client_config(self.client_id)
                business_insights_sections = client_config.business_insights_sections
            except Exception as e:
                logger.warning(f'Failed to load client config for business_insights_sections, using defaults: {e}')
                business_insights_sections = {'summary': True, 'metrics': True, 'insights': True, 'recommendations': True, 'follow_ups': True, 'note': True}
            self._business_insights_sections = business_insights_sections
            execution_summary = self._summarize_executor_results(executor_results)
            console_output = executor_results.get('business_console_output') or executor_results.get('console_output', '')
            console_output = self._strip_technical_noise(console_output)
            MAX_CONSOLE_CHARS = 100000
            MAX_SUMMARY_CHARS = 60000
            if len(console_output) > MAX_CONSOLE_CHARS:
                logger.warning(f'Console output too large ({len(console_output)} chars), truncating to {MAX_CONSOLE_CHARS}')
                console_output = console_output[:MAX_CONSOLE_CHARS] + '\n... [truncated for context window limit]'
            if len(execution_summary) > MAX_SUMMARY_CHARS:
                logger.warning(f'Execution summary too large ({len(execution_summary)} chars), truncating to {MAX_SUMMARY_CHARS}')
                execution_summary = execution_summary[:MAX_SUMMARY_CHARS] + '\n... [truncated for context window limit]'
            query = ''
            plan_text = ''
            if isinstance(planner_response, dict):
                query = planner_response.get('user_question', '') or planner_response.get('question', '')
                plan_text = planner_response.get('plan', '')
            analyst_findings = executor_results.get('ds_analysis', '') or executor_results.get('da_analysis', '')
            system_prompt, knowledge_metrics, company_context = self._build_system_prompt(query=query, use_knowledge_filtering=use_knowledge_filtering)
            user_message = self._build_user_message(query=query, plan_text=plan_text, analyst_findings=analyst_findings, execution_summary=execution_summary, console_output=console_output, reference_guidance=reference_guidance, business_insights_sections=business_insights_sections, company_context=company_context, persona_guidance=persona_guidance)
            try:
                self._build_numeric_whitelist(execution_summary, console_output, executor_results)
            except Exception as _:
                self._allowed_numeric_tokens = set()
            temperature = AGENT_CONFIG.get('business_agent', {}).get('temperature', 0.0)
            self._last_inputs = {'system_prompt': system_prompt, 'user_message': user_message, 'knowledge_metrics': knowledge_metrics}
            self._last_usage = None
            async for token, usage in self.llm_client.generate_completion_stream(system_prompt=system_prompt, user_message=user_message, temperature=temperature):
                if token == '__USAGE__' and usage:
                    self._last_usage = usage
                else:
                    yield (token, usage)
                    if usage:
                        self._last_usage = usage
        except Exception as e:
            logger.error(f'Error in BusinessAgent.process stream: {e}', exc_info=True)
            yield (f'Error generating business insights: {e}', None)

    def _clean_string_list(self, items: List[Any]) -> List[str]:
        if not isinstance(items, list):
            return []
        cleaned_list = []
        json_artefact_pattern = re.compile('\\"\\s*\\}\\,\\s*\\"(?:recommendations|follow_ups)\\"\\:\\s*\\[\\s*\\"')
        for item in items:
            if isinstance(item, str):
                clean_item = item.strip('"')
                clean_item = json_artefact_pattern.sub('', clean_item)
                cleaned_list.append(clean_item)
        return cleaned_list

    def _repair_json_string(self, json_string: str) -> str:
        content = json_string.strip()
        content = re.sub('```(?:json)?', '', content)
        brace_count = 0
        for i, char in enumerate(content):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    content = content[:i + 1]
                    break
        json_start = content.find('{')
        json_end = content.rfind('}')
        if json_start != -1 and json_end != -1:
            content = content[json_start:json_end + 1]
        return content

    def _manual_extract_from_broken_json(self, content: str) -> Dict[str, Any]:
        logger.warning('Attempting manual JSON extraction as a last resort.')

        def extract_list(pattern: str) -> List[str]:
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                return []
            return [item.strip() for item in re.findall('"([^"]*)"', match.group(1))]

        def extract_string(pattern: str) -> str:
            match = re.search(pattern, content)
            return match.group(1).strip() if match else ''
        a_struct = {KEY_METRICS: extract_list('"metrics"\\s*:\\s*\\[(.*?)\\]'), KEY_INSIGHTS: extract_list('"insights"\\s*:\\s*\\[(.*?)\\]'), KEY_SUMMARY: extract_string('"summary"\\s*:\\s*"(.*?)"'), KEY_NOTE: extract_string('"note"\\s*:\\s*"(.*?)"')}
        return {KEY_ANALYSIS_STRUCTURED: a_struct, KEY_RECOMMENDATIONS: extract_list('"recommendations"\\s*:\\s*\\[(.*?)\\]'), KEY_FOLLOW_UPS: extract_list('"follow_ups"\\s*:\\s*\\[(.*?)\\]'), KEY_SECTION_KPIS: {}}

    async def process_raw_business_insights(self, raw_tokens: str) -> Dict[str, Any]:
        try:
            if raw_tokens and '===STRUCTURED===' in raw_tokens:
                raw_tokens = raw_tokens.split('===STRUCTURED===', 1)[1]
            raw_tokens = (raw_tokens or '').strip()
            if raw_tokens.startswith('```'):
                raw_tokens = re.sub('^```[a-zA-Z]*\\n?|```$', '', raw_tokens).strip()
            parsed_data = None
            try:
                parsed_data = json.loads(raw_tokens)
            except json.JSONDecodeError as e:
                logger.warning(f'Initial JSON decode failed: {e}. Attempting to repair.')
                repaired_json = self._repair_json_string(raw_tokens)
                try:
                    parsed_data = json.loads(repaired_json)
                except json.JSONDecodeError as e2:
                    logger.warning(f'Repaired JSON decode failed: {e2}. Falling back to manual extraction.')
                    parsed_data = self._manual_extract_from_broken_json(raw_tokens)
            if not parsed_data or not any(parsed_data.values()):
                logger.error('All parsing methods failed. Using fallback insights.')
                return self._generate_fallback_insights()
            recs = self._clean_string_list(parsed_data.get(KEY_RECOMMENDATIONS, []))
            fus = self._clean_string_list(parsed_data.get(KEY_FOLLOW_UPS, []))
            a_struct = self._normalize_structured_analysis(parsed_data.get(KEY_ANALYSIS_STRUCTURED))
            analysis_text = self._compose_analysis_text(a_struct)

            def _filter_strings(items: List[str]) -> List[str]:
                out: List[str] = []
                for s in items:
                    try:
                        if self._all_tokens_allowed(s):
                            out.append(s)
                        else:
                            logger.warning(f'Dropping string with non-whitelisted numbers: {s}')
                    except Exception:
                        out.append(s)
                return out
            orig_metrics = list(a_struct.get(KEY_METRICS, []))
            orig_insights = list(a_struct.get(KEY_INSIGHTS, []))
            orig_summary = a_struct.get(KEY_SUMMARY, '')
            orig_recs = list(recs)
            orig_fus = list(fus)
            a_struct[KEY_METRICS] = _filter_strings(a_struct.get(KEY_METRICS, []))
            a_struct[KEY_INSIGHTS] = _filter_strings(a_struct.get(KEY_INSIGHTS, []))
            if a_struct.get(KEY_SUMMARY):
                a_struct[KEY_SUMMARY] = a_struct[KEY_SUMMARY] if self._all_tokens_allowed(a_struct[KEY_SUMMARY]) else ''
            if a_struct.get(KEY_NOTE):
                a_struct[KEY_NOTE] = a_struct[KEY_NOTE] if self._all_tokens_allowed(a_struct[KEY_NOTE]) else ''
            recs = _filter_strings(recs)
            fus = _filter_strings(fus)
            if not a_struct[KEY_METRICS] and orig_metrics:
                logger.warning('Numeric filter emptied metrics — restoring originals (%d items)', len(orig_metrics))
                a_struct[KEY_METRICS] = orig_metrics
            if not a_struct[KEY_INSIGHTS] and orig_insights:
                logger.warning('Numeric filter emptied insights — restoring originals (%d items)', len(orig_insights))
                a_struct[KEY_INSIGHTS] = orig_insights
            if not a_struct.get(KEY_SUMMARY) and orig_summary:
                logger.warning('Numeric filter emptied summary — restoring original')
                a_struct[KEY_SUMMARY] = orig_summary
            if not recs and orig_recs:
                logger.warning('Numeric filter emptied recommendations — restoring originals (%d items)', len(orig_recs))
                recs = orig_recs
            if not fus and orig_fus:
                logger.warning('Numeric filter emptied follow_ups — restoring originals (%d items)', len(orig_fus))
                fus = orig_fus
            a_struct[KEY_METRICS] = [format_numbers_in_text(text) for text in a_struct[KEY_METRICS]]
            a_struct[KEY_INSIGHTS] = [format_numbers_in_text(text) for text in a_struct[KEY_INSIGHTS]]
            if a_struct.get(KEY_SUMMARY):
                a_struct[KEY_SUMMARY] = format_numbers_in_text(a_struct[KEY_SUMMARY])
            if a_struct.get(KEY_NOTE):
                a_struct[KEY_NOTE] = format_numbers_in_text(a_struct[KEY_NOTE])
            if a_struct.get(KEY_RECOMMENDATIONS):
                a_struct[KEY_RECOMMENDATIONS] = format_numbers_in_text(a_struct[KEY_RECOMMENDATIONS])
            recs = [format_numbers_in_text(text) for text in recs]
            fus = [self._normalize_followup(x) for x in fus if x]
            fus = [format_numbers_in_text(text) for text in fus]
            business_insights_sections = getattr(self, '_business_insights_sections', {'summary': True, 'metrics': True, 'insights': True, 'recommendations': True, 'follow_ups': True, 'note': True})
            if not business_insights_sections.get('summary', True):
                a_struct[KEY_SUMMARY] = ''
            if not business_insights_sections.get('metrics', True):
                a_struct[KEY_METRICS] = []
            if not business_insights_sections.get('insights', True):
                a_struct[KEY_INSIGHTS] = []
            if not business_insights_sections.get('note', True):
                a_struct[KEY_NOTE] = ''
            if not business_insights_sections.get('recommendations', True):
                recs = []
            if not business_insights_sections.get('follow_ups', True):
                fus = []
            analysis_text = self._compose_analysis_text(a_struct)
            section_kpis = parsed_data.get(KEY_SECTION_KPIS, {})
            if not isinstance(section_kpis, dict):
                section_kpis = {}
            return {'analysis': analysis_text, KEY_ANALYSIS_STRUCTURED: a_struct, KEY_RECOMMENDATIONS: recs, KEY_FOLLOW_UPS: fus, KEY_SECTION_KPIS: section_kpis}
        except Exception as e:
            logger.error(f'Unexpected error in process_raw_business_insights: {e}', exc_info=True)
            return self._generate_fallback_insights()

    def _normalize_structured_analysis(self, a_struct: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(a_struct, dict):
            return {KEY_METRICS: [], KEY_INSIGHTS: [], KEY_SUMMARY: '', KEY_NOTE: ''}

        def to_list(value: Any) -> List[str]:
            if isinstance(value, list):
                return [str(x).strip() for x in value if str(x).strip()]
            if isinstance(value, str):
                parts = re.split('[\\n;]|\\s*•\\s+', value)
                return [p.strip().lstrip('-*• ') for p in parts if p.strip()]
            return []

        def normalize_bullet_multiline(s: str) -> str:
            if not isinstance(s, str):
                return ''
            t = s.strip()
            if not t:
                return t
            t = re.sub('\\s*•\\s+', '\n• ', t)
            return t
        normalized_summary = normalize_bullet_multiline(str(a_struct.get(KEY_SUMMARY, '')).strip())
        normalized_note = str(a_struct.get(KEY_NOTE, '')).strip()
        summary_parts = [p.strip().lstrip('-*• ') for p in re.split('[\\n;]|\\s*•\\s+', normalized_summary) if p.strip()]
        insights_list = to_list(a_struct.get(KEY_INSIGHTS))
        summary_bullets: List[str] = []
        if len(summary_parts) > 1:
            summary_bullets = [p for p in summary_parts if p]
            normalized_summary = '\n'.join([f'• {p}' for p in summary_bullets])
        seen = set()
        dedup_insights: List[str] = []
        for i in insights_list:
            key = i.lower()
            if key not in seen:
                seen.add(key)
                dedup_insights.append(i)
        result = {KEY_METRICS: to_list(a_struct.get(KEY_METRICS)), KEY_INSIGHTS: dedup_insights, KEY_SUMMARY: normalized_summary.lstrip('-*• '), KEY_NOTE: normalized_note}
        if summary_bullets:
            result[KEY_SUMMARY_BULLETS] = summary_bullets
        return result

    def _compose_analysis_text(self, a_struct: Dict[str, Any]) -> str:
        lines = []
        if a_struct.get(KEY_METRICS):
            lines.extend([f'• {m}' for m in a_struct[KEY_METRICS]])
        if a_struct.get(KEY_INSIGHTS):
            lines.extend([f'• {i}' for i in a_struct[KEY_INSIGHTS]])
        if a_struct.get(KEY_SUMMARY):
            lines.append(a_struct[KEY_SUMMARY])
        if a_struct.get(KEY_NOTE):
            lines.append(f'Note: {a_struct[KEY_NOTE]}')
        return '\n'.join(lines).strip()[:4000]

    def _generate_fallback_insights(self) -> Dict[str, Any]:
        logger.warning('Generating fallback business insights.')
        analysis_text = '\n        Key analysis indicates a strong performance hierarchy among products, with two leading products driving 45% of sales.\n        These top products show consistent growth, while others are declining. Weekly sales peak mid-week, suggesting opportunities\n        for targeted promotions during slower periods.\n        '
        recommendations = ['Increase investment in top-performing products (A and C) to maximize growth.', 'Investigate the root cause of declining sales for underperforming products (B and E).', 'Align inventory and staffing with mid-week sales peaks and consider promotions on slower days.']
        follow_ups = ['How do these sales trends correlate with recent marketing campaigns?', 'What is the profit margin for each of the top 5 products?', 'Are there significant regional variations in product performance?']
        a_struct = {KEY_METRICS: ['Top 2 products account for 45% of sales.', 'Sales for products B & E declined by ~30%.'], KEY_INSIGHTS: ['Product performance is highly concentrated in a few key items.', 'There is a clear weekly sales cycle with mid-week peaks.', 'Growth trends suggest increasing market demand for top products.'], KEY_SUMMARY: 'Analysis reveals a strong product hierarchy and distinct weekly sales patterns, offering clear opportunities for strategic focus and operational adjustments.', KEY_NOTE: "This analysis is based on last quarter's sales data."}
        return {'analysis': self._compose_analysis_text(a_struct), KEY_ANALYSIS_STRUCTURED: a_struct, KEY_RECOMMENDATIONS: recommendations, KEY_FOLLOW_UPS: follow_ups, KEY_SECTION_KPIS: {}}