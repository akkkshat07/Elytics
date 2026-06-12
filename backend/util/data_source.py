from enum import Enum
from typing import Optional, Dict, Any
import logging
logger = logging.getLogger(__name__)
STORE_IN_LOCAL_MISSING_ERROR = 'store_in_local not configured in DB Configs'

def require_store_in_local(credentials: Dict[str, Any]) -> bool:
    if 'store_in_local' not in credentials:
        raise RuntimeError(STORE_IN_LOCAL_MISSING_ERROR)
    value = credentials['store_in_local']
    if not isinstance(value, bool):
        raise RuntimeError('store_in_local must be configured as a boolean in DB Configs')
    return value

class DataSource(Enum):
    PARQUET = 'parquet'
    MYSQL = 'mysql'
    POSTGRES = 'postgres'
    MONGODB = 'mongodb'
    SAP_ORACLE = 'sap_oracle'
    SAP_HANA = 'sap_hana'
    SAP_SYBASE = 'sap_sybase'
    FILE_UPLOAD = 'file_upload'

    @property
    def prompt_suffix(self) -> str:
        if self in (DataSource.MYSQL, DataSource.POSTGRES, DataSource.SAP_ORACLE, DataSource.SAP_HANA, DataSource.SAP_SYBASE):
            return 'sql_db'
        elif self == DataSource.MONGODB:
            return 'mongodb'
        elif self in (DataSource.PARQUET, DataSource.FILE_UPLOAD):
            return 'parquet'
        else:
            return 'parquet'

    @property
    def is_live_db(self) -> bool:
        return self in (DataSource.MYSQL, DataSource.POSTGRES, DataSource.MONGODB, DataSource.SAP_ORACLE, DataSource.SAP_HANA, DataSource.SAP_SYBASE)

    @property
    def is_sql(self) -> bool:
        return self in (DataSource.MYSQL, DataSource.POSTGRES, DataSource.SAP_ORACLE, DataSource.SAP_HANA, DataSource.SAP_SYBASE)

    @classmethod
    def from_db_type(cls, db_type: Optional[str], store_in_local: bool) -> 'DataSource':
        if store_in_local:
            return cls.PARQUET
        if not db_type:
            raise RuntimeError('db_type not configured in DB Configs')
        db_type_lower = db_type.lower()
        if db_type_lower == 'file_upload':
            return cls.PARQUET
        if db_type_lower == 'mongodb':
            return cls.MONGODB
        if db_type_lower == 'mysql':
            return cls.MYSQL
        if db_type_lower in ('postgres', 'postgresql'):
            return cls.POSTGRES
        if db_type_lower == 'sap_oracle':
            return cls.SAP_ORACLE
        if db_type_lower == 'sap_hana':
            return cls.SAP_HANA
        if db_type_lower == 'sap_sybase':
            return cls.SAP_SYBASE
        raise RuntimeError(f'Unsupported db_type configured in DB Configs: {db_type}')

async def get_data_source_for_client(client_id: str, db=None, dataset_id: Optional[str]=None) -> DataSource:
    try:
        if db is None:
            from db_config.mongo_server import get_db
            db = await get_db()
        from services.db_credentials_service import DBCredentialsService
        service = DBCredentialsService(db)
        credentials = await service.get_credentials(client_id=client_id, db_type=None, decrypt_password=False, dataset_id=dataset_id)
        if not credentials:
            raise RuntimeError('No database credentials configured for this client.')
        db_type = credentials.get('db_type')
        store_in_local = require_store_in_local(credentials)
        ssh_cfg = (credentials.get('additional_params') or {}).get('ssh') or {}
        ssh_enabled = bool(ssh_cfg.get('enabled'))
        if (db_type or '').lower() in ('postgres', 'postgresql') and ssh_enabled and store_in_local:
            raise RuntimeError('SSH Postgres dataset is configured with store_in_local=true. Disable local storage to run live SQL mode.')
        logger.info('data_source_resolution client_id=%s dataset_id=%s db_type=%s store_in_local=%s ssh_enabled=%s', client_id, dataset_id, db_type, store_in_local, ssh_enabled)
        if db_type == 'sap_oracle':
            return DataSource.SAP_ORACLE
        if db_type == 'sap_sybase':
            return DataSource.SAP_SYBASE
        return DataSource.from_db_type(db_type, store_in_local)
    except Exception as e:
        logger.error(f'get_data_source_for_client: failed to load credentials for client {client_id}: {e}')
        raise