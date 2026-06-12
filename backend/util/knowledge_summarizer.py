from __future__ import annotations
import logging
from typing import List
logger = logging.getLogger(__name__)

def summarize_table_introductions_for_prompt(loaded_xml_content: str, relevant_tables: List[str], max_columns: int=10, max_description_chars: int=200) -> str:
    if not loaded_xml_content or not relevant_tables:
        return loaded_xml_content
    try:
        from defusedxml.ElementTree import fromstring, tostring, Element, SubElement
        root = fromstring(loaded_xml_content)
        if root.tag != 'meta_information':
            return loaded_xml_content
        table_introductions = root.find('table_introductions')
        if table_introductions is None:
            return loaded_xml_content
        relevant_set = {t.lower() for t in relevant_tables}
        summary_root = Element('meta_information', root.attrib)
        summary_container = SubElement(summary_root, 'table_introductions')
        for entry in table_introductions.findall('table_introduction'):
            table_name = entry.attrib.get('table_name', '')
            if table_name.lower() not in relevant_set:
                continue
            text = (entry.text or '').strip()
            if len(text) > max_description_chars:
                text = text[:max_description_chars].rstrip() + '...'
            summary_entry = SubElement(summary_container, 'table_introduction', entry.attrib)
            summary_entry.text = text
        return tostring(summary_root, encoding='unicode')
    except Exception as exc:
        logger.warning('Failed to summarize table introductions: %s', exc)
        return loaded_xml_content

def summarize_data_description_for_prompt(loaded_xml_content: str, max_columns: int=15, max_description_chars: int=160) -> str:
    if not loaded_xml_content:
        return loaded_xml_content
    try:
        from defusedxml.ElementTree import fromstring, tostring, Element, SubElement
        root = fromstring(loaded_xml_content)
        if root.tag != 'data_description':
            return loaded_xml_content
        summary_root = Element('data_description', root.attrib)
        table_info = root.find('table_info')
        if table_info is not None:
            summary_root.append(table_info)
        summary_columns = SubElement(summary_root, 'columns')
        columns = root.find('columns')
        if columns is None:
            return loaded_xml_content
        for idx, column in enumerate(columns.findall('column')):
            if idx >= max_columns:
                break
            name = column.attrib.get('name', '')
            data_type = column.attrib.get('data_type', '')
            summary_column = SubElement(summary_columns, 'column', {'name': name, 'data_type': data_type})
            description = column.find('description')
            if description is not None and description.text:
                text = description.text.strip()
                if len(text) > max_description_chars:
                    text = text[:max_description_chars].rstrip() + '...'
                desc_elem = SubElement(summary_column, 'description')
                desc_elem.text = text
        return tostring(summary_root, encoding='unicode')
    except Exception as exc:
        logger.warning('Failed to summarize data description: %s', exc)
        return loaded_xml_content

def summarize_domain_knowledge_for_prompt(content: str, max_chars: int=1000) -> str:
    if not content:
        return content
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + '...'

def summarize_company_profile_xml(content: str) -> str:
    if not content:
        return content
    try:
        from defusedxml.ElementTree import fromstring
        root = fromstring(content)
        business_context = root.find('.//business_context')
        company_info = []
        if business_context is not None:
            industry = business_context.find('industry')
            core_business = business_context.find('core_business')
            business_model = business_context.find('business_model')
            if industry is not None and industry.text:
                company_info.append(f'Industry: {industry.text}')
            if core_business is not None and core_business.text:
                company_info.append(f'Core Business: {core_business.text}')
            if business_model is not None and business_model.text:
                company_info.append(f'Business Model: {business_model.text}')
        products = [p.text for p in root.findall('.//products/product') if p.text]
        services = [s.text for s in root.findall('.//services/service') if s.text]
        if products:
            company_info.append(f"Products: {', '.join(products[:5])}")
        if services:
            company_info.append(f"Services: {', '.join(services[:5])}")
        terminology = [t.text for t in root.findall('.//terminology/term') if t.text]
        if terminology:
            company_info.append(f"Key Terminology: {', '.join(terminology[:10])}")
        return '\n'.join(company_info) if company_info else content
    except Exception as exc:
        logger.warning('Failed to summarize company profile XML: %s', exc)
        return content