import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
logger = logging.getLogger(__name__)
_COLUMN_NAMES_CACHE: Optional[Dict[str, Dict[str, str]]] = None

def load_column_names() -> Dict[str, Dict[str, str]]:
    global _COLUMN_NAMES_CACHE
    if _COLUMN_NAMES_CACHE is not None:
        return _COLUMN_NAMES_CACHE
    try:
        project_root = Path(__file__).resolve().parent.parent
        column_names_file = project_root / 'xml_prompts' / 'base_sap' / 'column_names.json'
        if not column_names_file.exists():
            logger.warning(f'column_names.json not found at {column_names_file}')
            _COLUMN_NAMES_CACHE = {}
            return {}
        with open(column_names_file, 'r', encoding='utf-8') as f:
            _COLUMN_NAMES_CACHE = json.load(f)
        logger.info(f'Loaded {len(_COLUMN_NAMES_CACHE)} column name mappings for SAP Oracle/Sybase')
        return _COLUMN_NAMES_CACHE
    except Exception as e:
        logger.error(f'Error loading column_names.json: {e}', exc_info=True)
        _COLUMN_NAMES_CACHE = {}
        return {}

def get_short_name(table_name: str, column_name: str) -> Optional[str]:
    column_mappings = load_column_names()
    key = f'{table_name}-{column_name}'
    if key in column_mappings:
        return column_mappings[key].get('short_name')
    key_upper = f'{table_name.upper()}-{column_name.upper()}'
    if key_upper in column_mappings:
        return column_mappings[key_upper].get('short_name')
    return None

def get_column_display_name(table_name: str, column_name: str, db_type: Optional[str]=None) -> str:
    if db_type not in ('sap_oracle', 'sap_sybase'):
        return column_name
    short_name = get_short_name(table_name, column_name)
    return short_name if short_name else column_name

def map_dataframe_columns(df, table_name: Optional[str]=None, db_type: Optional[str]=None) -> Tuple[Dict[str, str], Dict[str, str]]:
    if db_type not in ('sap_oracle', 'sap_sybase'):
        return ({}, {})
    column_mappings = load_column_names()
    column_mapping = {}
    column_metadata = {}
    for col in df.columns:
        col_str = str(col).strip()
        col_upper = col_str.upper()
        short_name = None
        full_name = None
        matched_key = None
        if table_name:
            key = f'{table_name}-{col_str}'
            key_upper = f'{table_name.upper()}-{col_upper}'
            if key in column_mappings:
                short_name = column_mappings[key].get('short_name')
                full_name = column_mappings[key].get('full_name')
                matched_key = key
            elif key_upper in column_mappings:
                short_name = column_mappings[key_upper].get('short_name')
                full_name = column_mappings[key_upper].get('full_name')
                matched_key = key_upper
        if not short_name:
            for key, value in column_mappings.items():
                key_upper_check = key.upper()
                if key_upper_check.endswith(f'-{col_upper}'):
                    short_name = value.get('short_name')
                    full_name = value.get('full_name')
                    matched_key = key
                    break
        if not short_name:
            col_normalized = col_str.replace('_', ' ').replace('-', ' ').strip()
            col_normalized_lower = col_normalized.lower()
            for key, value in column_mappings.items():
                value_short = value.get('short_name', '')
                if value_short:
                    short_normalized = value_short.replace(' ', '').replace('-', '').lower()
                    col_normalized_no_space = col_normalized_lower.replace(' ', '')
                    if col_normalized_lower == value_short.lower() or col_normalized_no_space == short_normalized or col_str.lower() == value_short.lower():
                        short_name = value_short
                        full_name = value.get('full_name')
                        matched_key = key
                        break
        if not short_name:
            for key, value in column_mappings.items():
                value_short = value.get('short_name', '')
                if value_short and value_short.lower() == col_str.lower():
                    short_name = value_short
                    full_name = value.get('full_name')
                    matched_key = key
                    break
        if short_name and col_str != short_name:
            final_name = short_name
            column_mapping[col_str] = short_name
        else:
            final_name = col_str
        if full_name:
            column_metadata[final_name] = full_name
        elif not short_name:
            logger.debug(f"Column '{col_str}' not found in column_names.json")
    return (column_mapping, column_metadata)

async def get_db_type_for_client(client_id: str, db=None) -> Optional[str]:
    try:
        if db is None:
            from db_config.mongo_server import get_db
            db = await get_db()
        from services.db_credentials_service import DBCredentialsService
        service = DBCredentialsService(db)
        credentials = await service.get_credentials(client_id=client_id, db_type=None, decrypt_password=False)
        if credentials:
            return credentials.get('db_type')
        return None
    except Exception as e:
        logger.warning(f'Error getting db_type for client {client_id}: {e}')
        return None