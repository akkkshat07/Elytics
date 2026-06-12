from __future__ import annotations
import logging
import re
from typing import Dict, Iterable, List, Tuple
logger = logging.getLogger(__name__)

def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4) if text else 0

def normalize_table_name(name: str) -> str:
    if not name:
        return ''
    normalized = re.sub('[^a-z0-9]+', '', name.lower())
    return normalized

def extract_table_names_from_text(text: str, available_tables: Iterable[str], max_tables: int=12) -> List[str]:
    if not text:
        return []
    normalized_text = normalize_table_name(text)
    matches: List[str] = []
    for table in available_tables:
        normalized_table = normalize_table_name(table)
        if not normalized_table:
            continue
        if normalized_table in normalized_text:
            matches.append(table)
            if len(matches) >= max_tables:
                break
    return matches

def _extract_table_descriptions(xml_content: str) -> List[Tuple[str, str]]:
    if not xml_content:
        return []
    try:
        from defusedxml.ElementTree import fromstring
        root = fromstring(xml_content)
        if root.tag != 'meta_information':
            return []
        table_introductions = root.find('table_introductions')
        if table_introductions is None:
            return []
        results: List[Tuple[str, str]] = []
        for entry in table_introductions.findall('table_introduction'):
            table_name = entry.attrib.get('table_name', '')
            description = (entry.text or '').strip()
            if table_name:
                results.append((table_name, description))
        return results
    except Exception:
        return []

def _keyword_match_tables_from_descriptions(xml_content: str, query: str, max_tables: int) -> List[str]:
    if not xml_content or not query:
        return []
    query_tokens = set(re.findall('[a-zA-Z0-9_]+', query.lower()))
    if not query_tokens:
        return []
    scored: List[Tuple[str, int]] = []
    for table_name, description in _extract_table_descriptions(xml_content):
        if not description:
            continue
        desc_tokens = set(re.findall('[a-zA-Z0-9_]+', description.lower()))
        score = len(query_tokens & desc_tokens)
        if score > 0:
            scored.append((table_name, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [table for table, _ in scored[:max_tables]]

def _semantic_match_tables_from_descriptions(xml_content: str, query: str, max_tables: int, similarity_threshold: float) -> List[str]:
    if not xml_content or not query:
        return []
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as exc:
        logger.info('Semantic table matching unavailable (%s).', exc)
        return []
    descriptions = _extract_table_descriptions(xml_content)
    if not descriptions:
        return []
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    scored: List[Tuple[str, float]] = []
    for table_name, description in descriptions:
        if not description:
            continue
        content_embedding = model.encode(description[:500])
        similarity = float(cosine_similarity([query_embedding], [content_embedding])[0][0])
        if similarity >= similarity_threshold:
            scored.append((table_name, similarity))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [table for table, _ in scored[:max_tables]]

def select_relevant_tables(query: str, table_introductions_xml: str, available_tables: Iterable[str], max_tables: int=12, semantic_top_k: int=5, semantic_threshold: float=0.7) -> Tuple[List[str], str]:
    direct = extract_table_names_from_text(query, available_tables, max_tables=max_tables)
    if direct:
        return (direct, 'name_match')
    keyword_tables = _keyword_match_tables_from_descriptions(table_introductions_xml, query, max_tables=max_tables)
    if keyword_tables:
        return (keyword_tables, 'description_keyword')
    semantic_tables = _semantic_match_tables_from_descriptions(table_introductions_xml, query, max_tables=semantic_top_k, similarity_threshold=semantic_threshold)
    if semantic_tables:
        return (semantic_tables, 'description_semantic')
    return ([], 'none')

def extract_table_names_from_table_introductions(xml_content: str) -> List[str]:
    if not xml_content:
        return []
    try:
        from defusedxml.ElementTree import fromstring
        root = fromstring(xml_content)
        if root.tag != 'meta_information':
            return []
        table_introductions = root.find('table_introductions')
        if table_introductions is None:
            return []
        return [entry.attrib.get('table_name', '') for entry in table_introductions.findall('table_introduction') if entry.attrib.get('table_name')]
    except Exception:
        return []

def _parse_data_description_table_name(xml_content: str) -> str:
    try:
        from defusedxml.ElementTree import fromstring
        root = fromstring(xml_content)
        if root.tag == 'data_description':
            return root.attrib.get('table_name', '')
    except Exception:
        return ''
    return ''

def filter_data_descriptions_by_tables(knowledge_content: Dict[str, str], table_names: Iterable[str], fallback_to_all: bool=False) -> Dict[str, str]:
    table_set = {normalize_table_name(t) for t in table_names if t}
    if not table_set and (not fallback_to_all):
        return {}
    filtered: Dict[str, str] = {}
    for key, content in knowledge_content.items():
        if not content:
            continue
        if key == 'table_introductions':
            continue
        if key in ('company_profile', 'website_data', 'terminology'):
            continue
        if not content.lstrip().startswith('<?xml'):
            continue
        table_name = ''
        if key.endswith('_description'):
            table_name = key[:-len('_description')]
        if not table_name:
            table_name = _parse_data_description_table_name(content)
        normalized_table = normalize_table_name(table_name)
        if normalized_table and (normalized_table in table_set or (fallback_to_all and (not table_set))):
            filtered[key] = content
    return filtered

def filter_table_introductions_xml(xml_content: str, table_names: Iterable[str]) -> str:
    if not xml_content or not table_names:
        return xml_content
    table_set = {normalize_table_name(t) for t in table_names if t}
    if not table_set:
        return xml_content
    try:
        from defusedxml.ElementTree import fromstring, tostring
        from xml.etree.ElementTree import Element, SubElement
        root = fromstring(xml_content)
        if root.tag != 'meta_information':
            return xml_content
        table_introductions = root.find('table_introductions')
        if table_introductions is None:
            return xml_content
        filtered_root = Element('meta_information', root.attrib)
        filtered_intro = SubElement(filtered_root, 'table_introductions')
        matched = 0
        for entry in table_introductions.findall('table_introduction'):
            table_name = entry.attrib.get('table_name', '')
            if normalize_table_name(table_name) in table_set:
                filtered_intro.append(entry)
                matched += 1
        if matched == 0:
            return xml_content
        return tostring(filtered_root, encoding='unicode')
    except Exception as exc:
        logger.warning('Failed to filter table introductions XML: %s', exc)
        return xml_content

def split_knowledge_sections(knowledge_content: Dict[str, str]) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    table_introductions = knowledge_content.get('table_introductions', '')
    data_descriptions: Dict[str, str] = {}
    domain_knowledge: Dict[str, str] = {}
    for key, content in knowledge_content.items():
        if key == 'table_introductions':
            continue
        if not content:
            continue
        if key in ('company_profile', 'website_data', 'terminology'):
            domain_knowledge[key] = content
            continue
        if content.lstrip().startswith('<?xml'):
            data_descriptions[key] = content
        else:
            domain_knowledge[key] = content
    return (table_introductions, data_descriptions, domain_knowledge)

def filter_domain_knowledge_by_query(knowledge: Dict[str, str], query: str, max_items: int=3, similarity_threshold: float=0.7) -> Dict[str, str]:
    if not knowledge or not query:
        return {}
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as exc:
        logger.info('Semantic filtering unavailable (%s). Falling back to keyword match.', exc)
        return _keyword_filter_domain_knowledge(knowledge, query, max_items=max_items)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    scored: List[Tuple[str, float]] = []
    for key, content in knowledge.items():
        if not content:
            continue
        sample = content[:500]
        content_embedding = model.encode(sample)
        similarity = float(cosine_similarity([query_embedding], [content_embedding])[0][0])
        if similarity >= similarity_threshold:
            scored.append((key, similarity))
    scored.sort(key=lambda item: item[1], reverse=True)
    return {key: knowledge[key] for key, _ in scored[:max_items]}

def _keyword_filter_domain_knowledge(knowledge: Dict[str, str], query: str, max_items: int=3) -> Dict[str, str]:
    query_tokens = set(re.findall('[a-zA-Z0-9_]+', query.lower()))
    scored: List[Tuple[str, int]] = []
    for key, content in knowledge.items():
        tokens = set(re.findall('[a-zA-Z0-9_]+', content.lower()))
        score = len(query_tokens & tokens)
        scored.append((key, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return {key: knowledge[key] for key, _ in scored[:max_items] if _ > 0}

def reduce_knowledge_for_prompt(knowledge_sections: Dict[str, str], max_tokens: int, preserve_keys: Iterable[str]) -> Dict[str, str]:
    if max_tokens <= 0:
        return knowledge_sections
    preserve_set = set(preserve_keys)
    total_tokens = sum((_approx_token_count(content) for content in knowledge_sections.values()))
    if total_tokens <= max_tokens:
        return knowledge_sections
    reduced = dict(knowledge_sections)
    removable = [(key, _approx_token_count(content)) for key, content in knowledge_sections.items() if key not in preserve_set]
    removable.sort(key=lambda item: item[1], reverse=True)
    for key, _ in removable:
        reduced.pop(key, None)
        total_tokens = sum((_approx_token_count(content) for content in reduced.values()))
        if total_tokens <= max_tokens:
            return reduced
    logger.warning('Knowledge content exceeds max_tokens even after removing optional sections. Preserved keys: %s', sorted(preserve_set))
    return {key: value for key, value in reduced.items() if key in preserve_set}

def build_knowledge_context(knowledge_sections: Dict[str, str]) -> str:
    parts: List[str] = []
    for key, content in knowledge_sections.items():
        if content:
            parts.append(f'{key.upper()}:\n{content}')
    return '\n\n'.join(parts)

def _table_name_matches_any(table_name: str, schema_tables: List[str]) -> bool:
    upper = table_name.upper()
    for st in schema_tables:
        st_upper = st.upper()
        if upper in st_upper or st_upper in upper:
            return True
    return False

def compress_table_introductions_for_coding(xml_content: str, table_names: List[str], max_chars_per_intro: int=250) -> str:
    if not xml_content or not table_names:
        return ''
    try:
        from defusedxml.ElementTree import fromstring
        root = fromstring(xml_content)
        if root.tag != 'meta_information':
            return ''
        table_introductions = root.find('table_introductions')
        if table_introductions is None:
            return ''
        lines: List[str] = []
        for entry in table_introductions.findall('table_introduction'):
            tname = entry.attrib.get('table_name', '')
            if not _table_name_matches_any(tname, table_names):
                continue
            text = (entry.text or '').strip()
            if len(text) > max_chars_per_intro:
                text = text[:max_chars_per_intro].rstrip() + '...'
            lines.append(f'{tname}: {text}')
        return '\n'.join(lines)
    except Exception as exc:
        logger.warning('Failed to compress table introductions for coding: %s', exc)
        return ''

def compress_data_descriptions_for_coding(data_descriptions: Dict[str, str], table_names: List[str], max_description_chars: int=100, max_columns_per_table: int=20) -> str:
    if not data_descriptions or not table_names:
        return ''
    parts: List[str] = []
    for key, xml_content in data_descriptions.items():
        if not xml_content:
            continue
        table_name = key
        if key.endswith('_description'):
            table_name = key[:-len('_description')]
        if not _table_name_matches_any(table_name, table_names):
            continue
        try:
            from defusedxml.ElementTree import fromstring
            root = fromstring(xml_content)
            if root.tag != 'data_description':
                continue
            table_info = root.find('table_info')
            total_cols = table_info.get('total_columns', '?') if table_info is not None else '?'
            columns_elem = root.find('columns')
            if columns_elem is None:
                continue
            col_lines: List[str] = []
            for idx, col in enumerate(columns_elem.findall('column')):
                if idx >= max_columns_per_table:
                    remaining = int(total_cols) - max_columns_per_table if total_cols != '?' else 0
                    if remaining > 0:
                        col_lines.append(f'  ... ({remaining} more columns)')
                    break
                col_name = col.get('name', '')
                col_type = col.get('data_type', '')
                desc_elem = col.find('description')
                desc_text = ''
                if desc_elem is not None and desc_elem.text:
                    desc_text = desc_elem.text.strip()
                    if len(desc_text) > max_description_chars:
                        desc_text = desc_text[:max_description_chars].rstrip() + '...'
                col_lines.append(f'  {col_name} ({col_type}): {desc_text}')
            if col_lines:
                parts.append(f'{table_name} ({total_cols} columns):\n' + '\n'.join(col_lines))
        except Exception as exc:
            logger.warning('Failed to compress data description for %s: %s', key, exc)
            continue
    return '\n\n'.join(parts)

def compress_terminology_for_coding(xml_content: str, max_terms: int=20) -> str:
    if not xml_content:
        return ''
    try:
        from defusedxml.ElementTree import fromstring
        root = fromstring(xml_content)
        terms: List[str] = []
        for term in root.iter('term'):
            key = term.get('key', '')
            definition_elem = term.find('definition')
            if key and definition_elem is not None and definition_elem.text:
                defn = definition_elem.text.strip()
                first_sentence = defn.split('. ')[0].rstrip('.')
                terms.append(f'{key}: {first_sentence}')
            if len(terms) >= max_terms:
                break
        return '\n'.join(terms)
    except Exception as exc:
        logger.warning('Failed to compress terminology for coding: %s', exc)
        return ''