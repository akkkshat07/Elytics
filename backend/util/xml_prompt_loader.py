from __future__ import annotations
from defusedxml.ElementTree import parse, fromstring
from xml.etree.ElementTree import Element, SubElement, ElementTree, tostring
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import copy
import time
from util.metrics import record_prompt_load
logger = logging.getLogger(__name__)
BASE_PROMPTS_PATH = Path('xml_prompts/base')
CLIENTS_PROMPTS_PATH = Path('xml_prompts/clients')

def _read_client_prompt_content(client_id: str, relative_path: str) -> Optional[str]:
    from config.system_config import STORAGE_BACKEND
    if STORAGE_BACKEND == 'gcs':
        try:
            from util.storage.gcs_client import GCSClient, get_gcs_client
            gcs = get_gcs_client()
            storage_path = f'clients/{client_id}/xml_prompts/{relative_path}'
            blob = gcs._bucket.blob(storage_path)
            if not blob.exists():
                return None
            return blob.download_as_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f'Failed to read client prompt from GCS: clients/{client_id}/xml_prompts/{relative_path}: {e}')
            return None
    else:
        local_path = CLIENTS_PROMPTS_PATH / client_id / relative_path
        if local_path.exists():
            return local_path.read_text(encoding='utf-8')
        return None

async def _read_client_prompt_content_async(client_id: str, relative_path: str) -> Optional[str]:
    from config.system_config import STORAGE_BACKEND
    if STORAGE_BACKEND == 'gcs':
        try:
            from util.storage.backend import get_storage_backend
            storage = get_storage_backend()
            storage_path = f'clients/{client_id}/xml_prompts/{relative_path}'
            if not await storage.exists(storage_path):
                logger.debug(f'Client prompt not found in GCS (expected): {storage_path}')
                return None
            return await storage.read_text(storage_path)
        except Exception as e:
            logger.warning(f'Failed to read client prompt from GCS: clients/{client_id}/xml_prompts/{relative_path}: {e}')
            return None
    else:
        local_path = CLIENTS_PROMPTS_PATH / client_id / relative_path
        if local_path.exists():
            return local_path.read_text(encoding='utf-8')
        return None

def _client_prompt_exists(client_id: str, relative_path: str) -> bool:
    from config.system_config import STORAGE_BACKEND
    if STORAGE_BACKEND == 'gcs':
        try:
            from util.storage.gcs_client import get_gcs_client
            gcs = get_gcs_client()
            storage_path = f'clients/{client_id}/xml_prompts/{relative_path}'
            blob = gcs._bucket.blob(storage_path)
            return blob.exists()
        except Exception as e:
            logger.warning(f'Failed to check existence in GCS: {e}')
            return False
    else:
        local_path = CLIENTS_PROMPTS_PATH / client_id / relative_path
        return local_path.exists()

async def _client_prompt_exists_async(client_id: str, relative_path: str) -> bool:
    from config.system_config import STORAGE_BACKEND
    if STORAGE_BACKEND == 'gcs':
        try:
            from util.storage.backend import get_storage_backend
            storage = get_storage_backend()
            storage_path = f'clients/{client_id}/xml_prompts/{relative_path}'
            return await storage.exists(storage_path)
        except Exception as e:
            logger.warning(f'Failed to check existence in GCS: {e}')
            return False
    else:
        local_path = CLIENTS_PROMPTS_PATH / client_id / relative_path
        return local_path.exists()

async def _list_client_prompt_files_async(client_id: str, relative_dir: str, suffix: str='.xml') -> List[str]:
    from config.system_config import STORAGE_BACKEND
    if STORAGE_BACKEND == 'gcs':
        try:
            from util.storage.backend import get_storage_backend
            storage = get_storage_backend()
            prefix = f'clients/{client_id}/xml_prompts/{relative_dir}'
            if not prefix.endswith('/'):
                prefix += '/'
            files = await storage.list_files(prefix)
            result = []
            for f in files:
                name = f.rsplit('/', 1)[-1] if '/' in f else f
                if name.endswith(suffix):
                    result.append(name)
            return result
        except Exception as e:
            logger.warning(f'Failed to list client prompt files from GCS: {e}')
            return []
    else:
        local_dir = CLIENTS_PROMPTS_PATH / client_id / relative_dir
        if local_dir.exists():
            return [f.name for f in local_dir.glob(f'*{suffix}')]
        return []

def load_xml_prompt_raw(xml_file_path: Path) -> str:
    try:
        if not xml_file_path.exists():
            logger.error(f'XML prompt file not found: {xml_file_path}')
            return 'You are an AI agent. Please analyze the input and provide appropriate output.'
        raw_content = xml_file_path.read_text(encoding='utf-8')
        logger.info(f'Successfully loaded raw XML prompt from {xml_file_path}')
        return raw_content
    except Exception as e:
        logger.error(f'Error loading raw XML prompt from {xml_file_path}: {e}')
        return 'You are an AI agent. Please analyze the input and provide appropriate output.'

def _format_xml_root_to_prompt(root: Element, source_label: str='unknown') -> str:
    agent_name = root.get('name', 'agent')
    priority = root.get('priority', '1')
    prompt_parts = []
    role_elem = root.find('{http://coresight.generic.com/prompts}role')
    if role_elem is not None and role_elem.text:
        prompt_parts.append(f'# Role\n{role_elem.text.strip()}')
    core_instructions = root.find('{http://coresight.generic.com/prompts}core_instructions')
    if core_instructions is not None:
        prompt_parts.append('## Core Instructions')
        for instruction in core_instructions:
            if instruction.text and instruction.text.strip():
                tag_name = instruction.tag.replace('{http://coresight.generic.com/prompts}', '').replace('_', ' ').title()
                prompt_parts.append(f'**{tag_name}**: {instruction.text.strip()}')
    rules = root.find('{http://coresight.generic.com/prompts}rules')
    if rules is not None:
        prompt_parts.append('## Rules')
        for rule in rules.findall('{http://coresight.generic.com/prompts}rule'):
            if rule.text and rule.text.strip():
                rule_type = rule.get('type', 'general')
                priority = rule.get('priority', '1')
                mandatory = rule.get('mandatory', 'true')
                rule_text = rule.text.strip()
                mandatory_text = 'MANDATORY' if mandatory.lower() == 'true' else 'OPTIONAL'
                prompt_parts.append(f'- **{rule_type.upper()}** ({mandatory_text}): {rule_text}')
    data_access = root.find('{http://coresight.generic.com/prompts}data_access')
    if data_access is not None:
        prompt_parts.append('## Data Access')
        primary_table = data_access.find('{http://coresight.generic.com/prompts}primary_table')
        if primary_table is not None and primary_table.text:
            prompt_parts.append(f'**Primary Table**: {primary_table.text.strip()}')
        available_tables = data_access.find('{http://coresight.generic.com/prompts}available_tables')
        if available_tables is not None:
            prompt_parts.append('**Available Tables**:')
            for table in available_tables.findall('{http://coresight.generic.com/prompts}table'):
                table_name = table.get('name', '')
                description = table.get('description', '')
                prompt_parts.append(f'- {table_name}: {description}')
        join_patterns = data_access.find('{http://coresight.generic.com/prompts}join_patterns')
        if join_patterns is not None:
            prompt_parts.append('**Join Patterns**:')
            for pattern in join_patterns.findall('{http://coresight.generic.com/prompts}pattern'):
                pattern_name = pattern.get('name', '')
                tables = pattern.get('tables', '')
                join_column = pattern.get('join_column', '')
                prompt_parts.append(f'- {pattern_name}: {tables} on {join_column}')
        filters = data_access.find('{http://coresight.generic.com/prompts}filters')
        if filters is not None and filters.text:
            prompt_parts.append(f'**Supported Filters**: {filters.text.strip()}')
    output_schema = root.find('{http://coresight.generic.com/prompts}output_schema')
    if output_schema is not None:
        prompt_parts.append('## Output Schema')
        format_elem = output_schema.find('{http://coresight.generic.com/prompts}format')
        if format_elem is not None and format_elem.text:
            prompt_parts.append(f'**Format**: {format_elem.text.strip()}')
        required_fields = output_schema.find('{http://coresight.generic.com/prompts}required_fields')
        if required_fields is not None and required_fields.text:
            prompt_parts.append(f'**Required Fields**: {required_fields.text.strip()}')
        validation_rules = output_schema.find('{http://coresight.generic.com/prompts}validation_rules')
        if validation_rules is not None and validation_rules.text:
            prompt_parts.append(f'**Validation Rules**: {validation_rules.text.strip()}')
    for section_name in ['year_month_resolution', 'special_handling', 'special_rules']:
        section = root.find(f'{{http://coresight.generic.com/prompts}}{section_name}')
        if section is not None:
            section_title = section_name.replace('_', ' ').title()
            prompt_parts.append(f'## {section_title}')
            for item in section:
                if item.text and item.text.strip():
                    item_name = item.tag.replace('{http://coresight.generic.com/prompts}', '').replace('_', ' ').title()
                    prompt_parts.append(f'**{item_name}**: {item.text.strip()}')
    domain_terminology = root.find('{http://coresight.generic.com/prompts}domain_terminology')
    if domain_terminology is not None:
        prompt_parts.append('## Domain Terminology')
        for term_group in domain_terminology:
            if term_group.text and term_group.text.strip():
                group_name = term_group.tag.replace('{http://coresight.generic.com/prompts}', '').replace('_', ' ').title()
                prompt_parts.append(f'**{group_name}**: {term_group.text.strip()}')
    examples = root.find('{http://coresight.generic.com/prompts}examples')
    if examples is not None:
        prompt_parts.append('## Examples')
        for i, example in enumerate(examples.findall('{http://coresight.generic.com/prompts}example'), 1):
            example_type = example.get('type', f'example_{i}')
            input_elem = example.find('{http://coresight.generic.com/prompts}input')
            output_elem = example.find('{http://coresight.generic.com/prompts}output')
            prompt_parts.append(f'### Example {i}: {example_type}')
            if input_elem is not None and input_elem.text:
                prompt_parts.append(f'**Input**: {input_elem.text.strip()}')
            if output_elem is not None and output_elem.text:
                prompt_parts.append(f'**Output**: {output_elem.text.strip()}')
    formatted_prompt = '\n\n'.join(prompt_parts)
    logger.info(f'Successfully formatted XML prompt from {source_label}')
    return formatted_prompt

def load_xml_prompt(xml_file_path: Path) -> str:
    try:
        if not xml_file_path.exists():
            logger.error(f'XML prompt file not found: {xml_file_path}')
            return 'You are an AI agent. Please analyze the input and provide appropriate output.'
        tree = parse(xml_file_path)
        root = tree.getroot()
        return _format_xml_root_to_prompt(root, str(xml_file_path))
    except Exception as e:
        logger.error(f'XML parsing error in {xml_file_path}: {e}')
        return 'You are an AI agent. Please analyze the input and provide appropriate output.'
    except Exception as e:
        logger.error(f'Error loading XML prompt from {xml_file_path}: {e}')
        return 'You are an AI agent. Please analyze the input and provide appropriate output.'

def load_xml_data_descriptions(data_descriptions_dir: Path) -> Dict[str, str]:
    descriptions = {}
    try:
        if not data_descriptions_dir.exists():
            logger.warning(f'Data descriptions directory not found: {data_descriptions_dir}')
            return descriptions
        for xml_file in data_descriptions_dir.glob('*.xml'):
            try:
                tree = parse(xml_file)
                root = tree.getroot()
                table_name = root.get('table_name', xml_file.stem)
                table_info = root.find('table_info')
                total_columns = table_info.get('total_columns', 'unknown') if table_info is not None else 'unknown'
                description_parts = [f'Table: {table_name} ({total_columns} columns)']
                columns = root.find('columns')
                if columns is not None:
                    description_parts.append('Columns:')
                    for column in columns.findall('column'):
                        col_name = column.get('name', '')
                        col_type = column.get('data_type', '')
                        desc_elem = column.find('description')
                        desc_text = desc_elem.text.strip() if desc_elem is not None else 'No description'
                        sample_values = column.find('sample_values')
                        sample_text = ''
                        if sample_values is not None and sample_values.text:
                            sample_text = f' (Sample: {sample_values.text.strip()})'
                        description_parts.append(f'- {col_name} ({col_type}): {desc_text}{sample_text}')
                descriptions[table_name] = '\n'.join(description_parts)
                logger.info(f'Loaded data description for table: {table_name}')
            except Exception as e:
                logger.error(f'Error loading data description from {xml_file}: {e}')
                continue
    except Exception as e:
        logger.error(f'Error loading data descriptions from {data_descriptions_dir}: {e}')
    return descriptions

async def get_client_prompt_override(client_id: str, prompt_path: str, db) -> Optional[Dict[str, Any]]:
    try:
        override = await db.client_prompts.find_one({'client_id': client_id, 'prompt_path': prompt_path, 'active': True})
        if override:
            logger.debug(f"Found override for client '{client_id}', prompt '{prompt_path}'")
            return override
        logger.debug(f"No override found for client '{client_id}', prompt '{prompt_path}'")
        return None
    except Exception as e:
        logger.error(f'Error getting client prompt override: {e}')
        return None

def merge_xml_sections(base_xml: str, override_sections: Dict[str, Dict[str, str]], prompt_path: str) -> str:
    try:
        if not override_sections:
            logger.debug('No override sections to merge')
            return base_xml
        base_root = fromstring(base_xml)
        for section_path, override_spec in override_sections.items():
            try:
                content = override_spec.get('content', '')
                strategy = override_spec.get('merge_strategy', 'replace')
                logger.debug(f"Applying {strategy} strategy for section '{section_path}'")
                override_elem = fromstring(content)
                if strategy == 'replace':
                    _replace_section(base_root, section_path, override_elem)
                elif strategy == 'merge':
                    _merge_section(base_root, section_path, override_elem)
                elif strategy == 'append':
                    _append_section(base_root, section_path, override_elem)
                else:
                    logger.warning(f"Unknown merge strategy '{strategy}', using replace")
                    _replace_section(base_root, section_path, override_elem)
            except Exception as e:
                logger.error(f"XML parse error in override for section '{section_path}': {e}")
                continue
            except Exception as e:
                logger.error(f"Error merging section '{section_path}': {e}")
                continue
        merged_xml = tostring(base_root, encoding='unicode')
        logger.info(f"Successfully merged {len(override_sections)} sections for '{prompt_path}'")
        return merged_xml
    except Exception as e:
        logger.error(f'XML parse error in base prompt: {e}')
        return base_xml
    except Exception as e:
        logger.error(f'Error merging XML sections: {e}')
        return base_xml

def _replace_section(root: Element, section_path: str, new_elem: Element) -> None:
    parent = root
    path_parts = section_path.split('.')
    for i, part in enumerate(path_parts[:-1]):
        clean_part = part.split('[')[0].replace('{http://coresight.generic.com/prompts}', '')
        found = parent.find(f'{{http://coresight.generic.com/prompts}}{clean_part}')
        if found is None:
            logger.warning(f"Section path '{section_path}' not found")
            return
        parent = found
    final_part = path_parts[-1].split('[')[0].replace('{http://coresight.generic.com/prompts}', '')
    for i, child in enumerate(parent):
        child_tag = child.tag.replace('{http://coresight.generic.com/prompts}', '')
        if child_tag == final_part:
            parent[i] = new_elem
            logger.debug(f"Replaced section '{section_path}'")
            return
    logger.warning(f"Section '{section_path}' not found for replacement")

def _merge_section(root: Element, section_path: str, merge_elem: Element) -> None:
    parent = root
    path_parts = section_path.split('.')
    for part in path_parts:
        clean_part = part.split('[')[0].replace('{http://coresight.generic.com/prompts}', '')
        found = parent.find(f'{{http://coresight.generic.com/prompts}}{clean_part}')
        if found is None:
            logger.warning(f"Section path '{section_path}' not found for merge")
            return
        parent = found
    for child in merge_elem:
        parent.append(copy.deepcopy(child))
    logger.debug(f"Merged section '{section_path}'")

def _append_section(root: Element, section_path: str, append_elem: Element) -> None:
    parent = root
    path_parts = section_path.split('.')
    for part in path_parts[:-1]:
        clean_part = part.split('[')[0].replace('{http://coresight.generic.com/prompts}', '')
        found = parent.find(f'{{http://coresight.generic.com/prompts}}{clean_part}')
        if found is None:
            logger.warning(f"Section path '{section_path}' not found for append")
            return
        parent = found
    final_part = path_parts[-1].split('[')[0].replace('{http://coresight.generic.com/prompts}', '')
    target_index = -1
    for i, child in enumerate(parent):
        child_tag = child.tag.replace('{http://coresight.generic.com/prompts}', '')
        if child_tag == final_part:
            target_index = i
            break
    if target_index >= 0:
        parent.insert(target_index + 1, copy.deepcopy(append_elem))
        logger.debug(f"Appended section after '{section_path}'")
    else:
        logger.warning(f"Section '{section_path}' not found for append")

def load_custom_prompts(client_id: str) -> str:
    try:
        custom_prompt_content = _read_client_prompt_content(client_id, 'agents/custom_prompts.xml')
        if custom_prompt_content is None:
            logger.debug(f'No custom_prompts.xml found for client: {client_id}')
            return ''
        root = fromstring(custom_prompt_content)
        prompts_container = root.find('.//{http://coresight.generic.com/prompts}prompts')
        if prompts_container is None:
            return ''
        prompt_data_list = []
        for prompt_elem in prompts_container.findall('.//{http://coresight.generic.com/prompts}prompt'):
            is_active = prompt_elem.get('active', 'true').lower() == 'true'
            if not is_active:
                continue
            title = prompt_elem.findtext('.//{http://coresight.generic.com/prompts}title', '')
            content = prompt_elem.findtext('.//{http://coresight.generic.com/prompts}content', '')
            category = prompt_elem.get('category', 'general')
            if content:
                content = content.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
                prompt_data_list.append({'title': title, 'content': content, 'category': category})
        prompt_data_list.sort(key=lambda x: x['title'])
        custom_sections = []
        for prompt_data in prompt_data_list:
            override_marker = '⚠️ HIGH PRIORITY OVERRIDE: '
            section = f"### {prompt_data['title']}\n\n{override_marker}{prompt_data['content']}"
            custom_sections.append(section)
        if custom_sections:
            result = '\n\n' + '=' * 80 + '\n'
            result += '##  CRITICAL: CUSTOM OVERRIDES (HIGHEST PRIORITY - MUST FOLLOW)\n'
            result += '=' * 80 + '\n'
            result += '⚠️ WARNING: These instructions OVERRIDE all other prompts.\n'
            result += '⚠️ You MUST follow these rules for ALL outputs.\n'
            result += '⚠️ Failure to follow these will result in incorrect results.\n'
            result += '=' * 80 + '\n\n'
            result += '\n\n'.join(custom_sections)
            result += '\n\n' + '=' * 80 + '\n'
            logger.info(f'Loaded {len(custom_sections)} custom prompts for client: {client_id}')
            logger.debug(f'Custom prompts content:\n{result}')
            return result
        return ''
    except Exception as e:
        logger.error(f'Error loading custom prompts for client {client_id}: {e}', exc_info=True)
        return ''

async def load_client_prompt(prompt_path: str, client_id: str, db=None, use_formatting: bool=True) -> str:
    start_time = time.time()
    cache_hit = False
    prompt_type = Path(prompt_path).stem
    try:
        logger.info(f'Loading prompt | client_id={client_id} | prompt={prompt_path} | format={use_formatting}')
        client_xml_content = await _read_client_prompt_content_async(client_id, prompt_path)
        base_prompt_path = BASE_PROMPTS_PATH / prompt_path
        if client_xml_content is not None:
            base_xml = client_xml_content
            is_client_prompt = True
            selected_source = f'client:{client_id}/{prompt_path}'
            logger.info(f'Using client-specific prompt | client_id={client_id} | path={prompt_path}')
        elif base_prompt_path.exists():
            base_xml = base_prompt_path.read_text(encoding='utf-8')
            is_client_prompt = False
            selected_source = str(base_prompt_path)
            logger.info(f'Using base prompt (no client override) | client_id={client_id} | path={base_prompt_path}')
        else:
            logger.error(f'Prompt not found in client or base directories | client_id={client_id} | prompt={prompt_path}')
            return 'You are an AI agent. Please analyze the input and provide appropriate output.'
        logger.debug(f'Prompt loaded | client_id={client_id} | source={selected_source} | size={len(base_xml)} bytes')
        if db is None:
            logger.debug(f'No database provided, skipping DB overrides | client_id={client_id}')
            cache_hit = not is_client_prompt
            if use_formatting:
                base_prompt = _format_xml_root_to_prompt(fromstring(base_xml), selected_source)
            else:
                base_prompt = base_xml
            if 'agents/' in prompt_path:
                logger.debug(f'Attempting to load custom prompts | client_id={client_id} | prompt_path={prompt_path} | use_formatting={use_formatting}')
                custom_prompts = load_custom_prompts(client_id)
                if custom_prompts:
                    logger.info(f'Appending custom prompts to base prompt | client_id={client_id} | custom_prompts_length={len(custom_prompts)}')
                    base_prompt += '\n\n' + custom_prompts
                else:
                    logger.warning(f'No custom prompts loaded | client_id={client_id}')
            return base_prompt
        override = await get_client_prompt_override(client_id, prompt_path, db)
        if not override:
            cache_hit = not is_client_prompt
            logger.info(f'No DB override found | client_id={client_id} | prompt={prompt_path}')
            if use_formatting:
                base_prompt = _format_xml_root_to_prompt(fromstring(base_xml), selected_source)
            else:
                base_prompt = base_xml
            if 'agents/' in prompt_path:
                logger.debug(f'Attempting to load custom prompts | client_id={client_id} | prompt_path={prompt_path} | use_formatting={use_formatting}')
                custom_prompts = load_custom_prompts(client_id)
                if custom_prompts:
                    logger.info(f'Appending custom prompts to base prompt | client_id={client_id} | custom_prompts_length={len(custom_prompts)}')
                    base_prompt += '\n\n' + custom_prompts
                else:
                    logger.warning(f'No custom prompts loaded | client_id={client_id}')
            return base_prompt
        cache_hit = False
        logger.info(f'Applying DB prompt overrides | client_id={client_id} | prompt={prompt_path}')
        override_sections = override.get('override_sections', {})
        merged_xml = merge_xml_sections(base_xml, override_sections, prompt_path)
        logger.debug(f'Prompt merge complete | client_id={client_id} | merged_size={len(merged_xml)} bytes')
        if use_formatting:
            try:
                formatted = _format_xml_root_to_prompt(fromstring(merged_xml), f'merged:{selected_source}')
                if 'agents/' in prompt_path:
                    logger.debug(f'Attempting to load custom prompts | client_id={client_id} | prompt_path={prompt_path}')
                    custom_prompts = load_custom_prompts(client_id)
                    if custom_prompts:
                        logger.info(f'Appending custom prompts to formatted prompt | client_id={client_id} | custom_prompts_length={len(custom_prompts)}')
                        formatted += '\n\n' + custom_prompts
                    else:
                        logger.warning(f'No custom prompts loaded | client_id={client_id}')
                logger.info(f'Prompt loaded and formatted successfully | client_id={client_id} | prompt={prompt_path}')
                return formatted
            except Exception as e:
                logger.error(f'Error formatting merged prompt | client_id={client_id} | error={e}', exc_info=True)
                return merged_xml
        logger.info(f'Prompt loaded successfully (unformatted) | client_id={client_id} | prompt={prompt_path}')
        return merged_xml
    except Exception as e:
        logger.error(f'Error loading client prompt | client_id={client_id} | prompt={prompt_path} | error={e}', exc_info=True)
        return 'You are an AI agent. Please analyze the input and provide appropriate output.'
    finally:
        duration_ms = (time.time() - start_time) * 1000
        record_prompt_load(client_id=client_id, prompt_type=prompt_type, duration_ms=duration_ms, cache_hit=cache_hit)

async def load_client_terminology(client_id: str, db=None) -> Dict[str, Any]:
    try:
        terminology = {}
        if db is None:
            return terminology
        override = await get_client_prompt_override(client_id, 'domain_knowledge/terminology.xml', db)
        if override:
            override_sections = override.get('override_sections', {})
            for section_path, spec in override_sections.items():
                content = spec.get('content', '')
                try:
                    term_elem = fromstring(content)
                    for category in term_elem:
                        category_name = category.get('name', 'general')
                        if category_name not in terminology:
                            terminology[category_name] = {}
                        for term in category:
                            term_key = term.get('key', '')
                            term_value = term.text.strip() if term.text else ''
                            if term_key and term_value:
                                terminology[category_name][term_key] = term_value
                except Exception as e:
                    logger.error(f'Error parsing terminology override: {e}')
                    continue
        logger.info(f"Loaded {len(terminology)} terminology categories for client '{client_id}'")
        return terminology
    except Exception as e:
        logger.error(f'Error loading client terminology: {e}')
        return {}

def get_base_prompt_path(relative_path: str) -> Path:
    return BASE_PROMPTS_PATH / relative_path

async def load_base_prompt(prompt_type: str, db=None) -> str:
    try:
        prompt_paths = {'planner': 'agents/planner.xml', 'python': 'agents/python.xml', 'business': 'agents/business.xml', 'executor': 'agents/executor.xml'}
        relative_path = prompt_paths.get(prompt_type.lower())
        if not relative_path:
            logger.error(f'Unknown prompt type: {prompt_type}')
            return ''
        base_path = get_base_prompt_path(relative_path)
        if not base_path.exists():
            logger.error(f'Base prompt file not found: {base_path}')
            return ''
        content = base_path.read_text(encoding='utf-8')
        logger.info(f"Loaded base prompt for type '{prompt_type}' from {base_path}")
        return content
    except Exception as e:
        logger.error(f"Error loading base prompt for type '{prompt_type}': {e}")
        return ''

def merge_data_descriptions(base_xml: str, client_xml: str, table_name: str) -> str:
    try:
        base_root = fromstring(base_xml)
        client_root = fromstring(client_xml)
        base_columns = base_root.find('columns')
        client_columns = client_root.find('columns')
        if base_columns is None:
            logger.warning(f'No columns found in base data description for {table_name}')
            return base_xml
        if client_columns is None:
            logger.debug(f'No client columns found for {table_name}, using base only')
            return base_xml
        for base_col in base_columns.findall('column'):
            col_name = base_col.get('name')
            client_col = None
            for c_col in client_columns.findall('column'):
                if c_col.get('name') == col_name:
                    client_col = c_col
                    break
            if client_col is not None:
                client_sample = client_col.find('sample_values')
                if client_sample is not None:
                    base_sample = base_col.find('sample_values')
                    if base_sample is not None:
                        base_col.remove(base_sample)
                    base_col.append(copy.deepcopy(client_sample))
        merged_xml = tostring(base_root, encoding='unicode')
        logger.debug(f'Merged data description for {table_name}')
        return merged_xml
    except Exception as e:
        logger.error(f'Error merging data description for {table_name}: {e}')
        return base_xml

def load_client_data_descriptions(client_id: str, base_descriptions_dir: Path=None, client_descriptions_dir: Path=None, dataset_id: Optional[str]=None, datasource_context: Optional[Dict[str, Any]]=None) -> Dict[str, str]:
    descriptions: Dict[str, str] = {}
    try:
        if base_descriptions_dir is None:
            base_descriptions_dir = BASE_PROMPTS_PATH / 'data_sources' / 'data_descriptions'
        use_storage_backend = client_descriptions_dir is None
        if dataset_id:
            desc_storage_rel = f'data_sources/{dataset_id}/data_descriptions'
        else:
            desc_storage_rel = 'data_sources/data_descriptions'
        if client_descriptions_dir is None:
            from util.dataset_paths import resolve_xml_data_sources_dir
            client_descriptions_dir = resolve_xml_data_sources_dir(client_id, dataset_id) / 'data_descriptions'
        logger.info('Loading data descriptions | client_id=%s | base=%s | client=%s', client_id, base_descriptions_dir, client_descriptions_dir)
        xml_filenames: set[str] = set()
        if client_descriptions_dir and client_descriptions_dir.exists():
            xml_filenames.update((f.name for f in client_descriptions_dir.glob('*.xml')))
        if base_descriptions_dir and base_descriptions_dir.exists():
            xml_filenames.update((f.name for f in base_descriptions_dir.glob('*.xml')))
        if not xml_filenames:
            logger.warning("No data description files found for client '%s' in base or client directories", client_id)
            return descriptions
        for filename in sorted(xml_filenames):
            selected_file: Optional[Path] = None
            is_client_file = False
            if client_descriptions_dir and client_descriptions_dir.exists():
                candidate = client_descriptions_dir / filename
                if candidate.exists():
                    selected_file = candidate
                    is_client_file = True
            if not selected_file and base_descriptions_dir and base_descriptions_dir.exists():
                candidate = base_descriptions_dir / filename
                if candidate.exists():
                    selected_file = candidate
            if selected_file:
                xml_content = selected_file.read_text(encoding='utf-8')
            elif use_storage_backend:
                xml_content = _read_client_prompt_content(client_id, f'{desc_storage_rel}/{filename}')
                if xml_content is None:
                    continue
                is_client_file = True
            else:
                continue
            table_name = Path(filename).stem
            try:
                root = fromstring(xml_content)
                table_name = root.get('table_name', table_name)
            except Exception as parse_err:
                logger.debug('Could not parse XML table_name for %s, using stem: %s', filename, parse_err)
            descriptions[table_name] = xml_content
            logger.debug('Loaded data description: %s (%s)', table_name, 'client' if is_client_file else 'base')
        logger.info("Loaded %d data descriptions for client '%s'", len(descriptions), client_id)
        return descriptions
    except Exception as e:
        logger.error('Error loading client data descriptions: %s', e)
        return descriptions
__all__ = ['load_xml_prompt', 'load_xml_prompt_raw', 'load_xml_data_descriptions', 'load_client_prompt', 'get_client_prompt_override', 'merge_xml_sections', 'load_client_terminology', 'get_base_prompt_path', 'load_base_prompt', 'merge_data_descriptions', 'load_client_data_descriptions', 'load_custom_prompts', 'BASE_PROMPTS_PATH']