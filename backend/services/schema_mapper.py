import asyncio
import logging
import os
from typing import Dict, List, Optional, Any
from functools import lru_cache
logger = logging.getLogger(__name__)

class SchemaMapper:
    _schema_cache: Dict[str, Dict] = {}
    _schema_cache_lock: Optional['asyncio.Lock'] = None

    @classmethod
    def _get_lock(cls) -> 'asyncio.Lock':
        if cls._schema_cache_lock is None:
            cls._schema_cache_lock = asyncio.Lock()
        return cls._schema_cache_lock

    def __init__(self, client_id: str, db):
        self.client_id = client_id
        self.db = db
        self.schema: Dict = {}

    @classmethod
    async def create(cls, client_id: str, db) -> 'SchemaMapper':
        instance = cls(client_id, db)
        await instance.initialize()
        return instance

    @classmethod
    def get_sync(cls, client_id: str, db) -> 'SchemaMapper':
        instance = cls.__new__(cls)
        instance.client_id = client_id
        instance.db = db
        if client_id in cls._schema_cache:
            instance.schema = cls._schema_cache[client_id]
        else:
            instance.schema = instance._load_client_schema(client_id, db)
            cls._schema_cache[client_id] = instance.schema
        return instance

    async def initialize(self) -> None:
        client_id = self.client_id
        async with self._get_lock():
            if client_id not in self._schema_cache:
                self._schema_cache[client_id] = self._load_client_schema(client_id, self.db)
                logger.info(f"[SchemaMapper] Loaded schema for '{client_id}' from database")
            else:
                logger.debug(f"[SchemaMapper] Loaded schema for '{client_id}' from cache")
        self.schema = self._schema_cache[client_id]

    def _load_client_schema(self, client_id: str, db) -> Dict:
        try:
            from pymongo import MongoClient
            from dotenv import load_dotenv
            load_dotenv()
            mongo_url = os.getenv('DATABASE_URL', 'mongodb://localhost:27017')
            db_name = os.getenv('DATABASE_NAME', 'core-sight')
            sync_client = MongoClient(mongo_url, serverSelectionTimeoutMS=2000)
            sync_db = sync_client[db_name]
            schema_doc = sync_db.client_schemas.find_one({'client_id': client_id})
            sync_client.close()
            if not schema_doc:
                logger.debug(f"[SchemaMapper] No schema found for '{client_id}', using fallback")
                return self._get_fallback_schema(client_id)
            required_fields = ['tables', 'display_config']
            for field in required_fields:
                if field not in schema_doc:
                    logger.error(f"[SchemaMapper] Schema for '{client_id}' missing required field: {field}")
                    return self._get_fallback_schema(client_id)
            return schema_doc
        except Exception as e:
            logger.error(f"[SchemaMapper] Error loading schema for '{client_id}': {e}")
            return self._get_fallback_schema(client_id)

    def _get_fallback_schema(self, client_id: str) -> Dict:
        logger.debug(f"[SchemaMapper] Using generic fallback schema for '{client_id}'")
        prefix = client_id.upper()
        return {'client_id': client_id, 'tables': [{'logical_name': 'main_data', 'physical_name': f'{prefix}_DATA', 'columns': [{'logical_name': 'category', 'physical_name': 'CATEGORY', 'display_name': 'Category'}, {'logical_name': 'segment', 'physical_name': 'SEGMENT', 'display_name': 'Segment'}, {'logical_name': 'entity', 'physical_name': 'ENTITY_NAME', 'display_name': 'Entity'}, {'logical_name': 'metric_value', 'physical_name': 'VALUE', 'display_name': 'Value'}, {'logical_name': 'metric_quantity', 'physical_name': 'QTY', 'display_name': 'Quantity'}, {'logical_name': 'date', 'physical_name': 'DATE', 'display_name': 'Date'}]}], 'display_config': {'currency_symbol': '₹', 'number_format': 'indian', 'domain_terminology': {}}, 'guardrails': {'domain_keywords': [], 'facility_names': [], 'product_terms': []}, 'grouping_dimensions': {}, 'metrics': {}}

    def get_table(self, logical_name: str) -> str:
        for table in self.schema.get('tables', []):
            if table.get('logical_name') == logical_name:
                physical_name = table.get('physical_name')
                logger.debug(f"[SchemaMapper] '{self.client_id}': {logical_name} → {physical_name}")
                return physical_name
        available = [t.get('logical_name') for t in self.schema.get('tables', [])]
        raise ValueError(f"[SchemaMapper] Logical table '{logical_name}' not found for client '{self.client_id}'. Available tables: {available}")

    def get_column(self, logical_name: str, table_logical_name: Optional[str]=None) -> str:
        tables_to_search = self.schema.get('tables', [])
        if table_logical_name:
            tables_to_search = [t for t in tables_to_search if t.get('logical_name') == table_logical_name]
        for table in tables_to_search:
            for col in table.get('columns', []):
                if col.get('logical_name') == logical_name:
                    physical_name = col.get('physical_name')
                    logger.debug(f"[SchemaMapper] '{self.client_id}': {logical_name} → {physical_name}")
                    return physical_name
        raise ValueError(f"[SchemaMapper] Logical column '{logical_name}' not found for client '{self.client_id}'")

    def get_display_name(self, logical_name: str, default: Optional[str]=None) -> str:
        terminology = self.schema.get('display_config', {}).get('domain_terminology', {})
        if logical_name in terminology:
            return terminology[logical_name]
        for table in self.schema.get('tables', []):
            for col in table.get('columns', []):
                if col.get('logical_name') == logical_name:
                    display = col.get('display_name')
                    if display:
                        return display
        if default:
            return default
        return logical_name.replace('_', ' ').title()

    def get_grouping_dimensions(self) -> Dict[str, Dict[str, str]]:
        grouping_config = self.schema.get('grouping_dimensions', {})
        if grouping_config:
            result = {}
            for dim_key, dim_config in grouping_config.items():
                try:
                    logical_name = dim_config.get('logical_name')
                    if logical_name:
                        physical_name = self.get_column(logical_name)
                        display_name = dim_config.get('display_name') or self.get_display_name(logical_name)
                        result[dim_key] = {'logical_name': logical_name, 'physical_name': physical_name, 'display_name': display_name}
                except ValueError:
                    logger.warning(f"[SchemaMapper] Skipping invalid grouping dimension '{dim_key}'")
                    continue
            return result
        backward_compat = {}
        try:
            logical_mappings = [('category', 'by_category'), ('segment', 'by_segment'), ('entity', 'by_entity'), ('inventory_group', 'by_group'), ('aging_bucket', 'by_slab'), ('organization', 'by_site')]
            for logical_name, default_key in logical_mappings:
                try:
                    physical_name = self.get_column(logical_name)
                    display_name = self.get_display_name(logical_name)
                    backward_compat[default_key] = {'logical_name': logical_name, 'physical_name': physical_name, 'display_name': display_name}
                except ValueError:
                    continue
            if backward_compat:
                logger.info(f"[SchemaMapper] Using backward-compatible grouping dimensions for '{self.client_id}'")
                return backward_compat
        except Exception as e:
            logger.debug(f'[SchemaMapper] Backward compatibility check failed: {e}')
        return {}

    def get_grouping_columns(self) -> Dict[str, str]:
        dimensions = self.get_grouping_dimensions()
        return {dim_key: dim_info['physical_name'] for dim_key, dim_info in dimensions.items()}

    def get_metric_columns(self) -> Dict[str, str]:
        metrics_config = self.schema.get('metrics', {})
        if metrics_config:
            result = {}
            for metric_key, metric_config in metrics_config.items():
                try:
                    logical_name = metric_config.get('logical_name')
                    if logical_name:
                        physical_name = self.get_column(logical_name)
                        result[metric_key] = physical_name
                except ValueError:
                    logger.warning(f"[SchemaMapper] Skipping invalid metric '{metric_key}'")
                    continue
            return result
        backward_compat = {}
        try:
            metric_mappings = [('metric_value', 'primary_value'), ('metric_quantity', 'primary_quantity'), ('closing_value', 'primary_value'), ('available_quantity', 'primary_quantity')]
            for logical_name, default_key in metric_mappings:
                try:
                    physical_name = self.get_column(logical_name)
                    backward_compat[default_key] = physical_name
                except ValueError:
                    continue
            if backward_compat:
                logger.info(f"[SchemaMapper] Using backward-compatible metric columns for '{self.client_id}'")
                return backward_compat
        except Exception as e:
            logger.debug(f'[SchemaMapper] Backward compatibility check for metrics failed: {e}')
        return {}

    def get_guardrails_config(self) -> Dict[str, List[str]]:
        return self.schema.get('guardrails', {'domain_keywords': [], 'facility_names': [], 'product_terms': []})

    def get_number_format_config(self) -> Dict[str, str]:
        display_config = self.schema.get('display_config', {})
        return {'currency_symbol': display_config.get('currency_symbol', '₹'), 'number_format': display_config.get('number_format', 'indian')}

    def get_date_format_config(self) -> Dict[str, str]:
        display_config = self.schema.get('display_config', {})
        return {'date_format': display_config.get('date_format', 'DD/MM/YYYY')}

    @classmethod
    async def clear_cache(cls) -> None:
        async with cls._get_lock():
            cls._schema_cache.clear()
        logger.info('[SchemaMapper] Schema cache cleared')

    def __repr__(self):
        return f"SchemaMapper(client_id='{self.client_id}', tables={len(self.schema.get('tables', []))})"