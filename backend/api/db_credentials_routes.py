from __future__ import annotations
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import logging
import asyncio
import shutil
from pathlib import Path
from db_config.mongo_server import get_db
from middleware.auth_middleware import require_admin
from services.db_credentials_service import DBCredentialsService
from util.audit_logger import audit_logger, AuditEventType, AuditSeverity
from util.client_data_cleanup import cleanup_client_data
from util.dataset_paths import assets_datasets_dir
from admin.client_onboarding import create_xml_directory_structure
from explorer.explorer_agent import ExplorerAgent
from explorer.file_metadata_generator import FileMetadataGenerator
from db_config.connectors.postgres_connector import PostgresConnector
from db_config.connectors.mysql_connector import MySQLConnector
from db_config.connectors.mongo_connector import MongoConnector
from db_config.connectors.sap_hana_connector import SAPHANAConnector
from db_config.connectors.oracle_connector import OracleConnector
from db_config.connectors.sybase_connector import SybaseConnector
logger = logging.getLogger(__name__)
router = APIRouter(prefix='/db-credentials', tags=['Database Credentials'])

class SaveCredentialsRequest(BaseModel):
    dataset_name: Optional[str] = Field(default=None, description='Human-readable name for a new dataset (required to create a new dataset)')
    dataset_id: Optional[str] = Field(default=None, description='Deprecated: editing existing datasets is not supported via this endpoint')
    db_type: str = Field(default='postgres', description='Database type (postgres, mysql, mongodb, sqlserver, sap_oracle, sap_hana, sap_sybase, file_upload, etc.)')
    db_host: str = Field(default='', description='Database host (optional for file_upload)')
    db_port: int = Field(default=0, ge=0, le=65535, description='Database port (optional for file_upload)')
    db_name: str = Field(default='', description='Database name (optional for file_upload)')
    db_username: str = Field(default='', description='Database username (optional for file_upload)')
    db_password: str = Field(default='', description='Database password (optional for file_upload)')
    db_url: str = Field(default='', description='Full database URL (optional for file_upload)')
    additional_params: Optional[Dict[str, Any]] = Field(default=None, description='Additional connection parameters')
    store_in_local: bool = Field(..., description='Whether to store data locally as parquet files')
    skip_metadata_refresh: bool = Field(default=False, description='Skip automatic metadata refresh (use when metadata will be generated manually)')

    @validator('db_type')
    def validate_db_type(cls, v):
        allowed_types = ['postgres', 'mysql', 'mongodb', 'sqlserver', 'sap_oracle', 'sap_hana', 'sap_sybase', 'file_upload']
        if v not in allowed_types:
            raise ValueError(f"db_type must be one of: {', '.join(allowed_types)}")
        return v

    @validator('db_host', 'db_username', 'db_password', always=True)
    def validate_not_empty(cls, v, values):
        db_type = values.get('db_type', 'postgres')
        if db_type == 'file_upload':
            return v if v else ''
        if not v or (isinstance(v, str) and (not v.strip())):
            raise ValueError('Field cannot be empty')
        return v.strip() if isinstance(v, str) else v

    @validator('db_name', always=True)
    def validate_db_name(cls, v, values):
        db_type = values.get('db_type', 'postgres')
        if db_type in ('file_upload', 'sap_sybase'):
            return v.strip() if isinstance(v, str) and v.strip() else ''
        if not v or (isinstance(v, str) and (not v.strip())):
            raise ValueError('Field cannot be empty')
        return v.strip() if isinstance(v, str) else v

class TestConnectionRequest(BaseModel):
    db_type: str = Field(default='postgres', description='Database type')
    db_host: str = Field(..., description='Database host')
    db_port: int = Field(..., ge=1, le=65535, description='Database port')
    db_name: str = Field(..., description='Database name')
    db_username: str = Field(..., description='Database username')
    db_password: str = Field(..., description='Database password')
    db_url: Optional[str] = Field(None, description='Database URL (optional)')

@router.post('/save')
async def save_credentials(request: SaveCredentialsRequest, admin_user: Dict=Depends(require_admin), db=Depends(get_db)):
    try:
        admin_client_id = admin_user.get('client_id')
        admin_email = admin_user.get('email', 'unknown')
        if not admin_client_id:
            raise HTTPException(status_code=403, detail='Admin user missing client_id. Please contact system administrator.')
        logger.info(f'Admin {admin_email} saving DB credentials | client_id={admin_client_id} | db_type={request.db_type}')
        service = DBCredentialsService(db)
        if request.dataset_id:
            raise HTTPException(status_code=400, detail='Editing existing datasets is not supported. Create a new dataset instead.')
        if request.db_type != 'file_upload':
            is_super_admin = admin_user.get('role') == 'super_admin'
            if not is_super_admin:
                from util.db_size import get_database_size_bytes, format_size_mb, MAX_DB_SIZE_BYTES
                db_size_bytes = await get_database_size_bytes(db_type=request.db_type, db_host=request.db_host, db_port=request.db_port, db_name=request.db_name, db_username=request.db_username, db_password=request.db_password, db_url=request.db_url)
                if db_size_bytes > 0 and db_size_bytes > MAX_DB_SIZE_BYTES:
                    raise HTTPException(status_code=400, detail=f'Database size ({format_size_mb(db_size_bytes)} MB) exceeds the maximum allowed ({format_size_mb(MAX_DB_SIZE_BYTES)} MB). Please upgrade your plan or reduce the database size.')
        pre_count = await service.collection.count_documents({'client_id': admin_client_id})
        is_first_save = pre_count == 0
        try:
            result = await service.save_credentials(client_id=admin_client_id, db_type=request.db_type, db_host=request.db_host, db_port=request.db_port, db_name=request.db_name, db_username=request.db_username, db_password=request.db_password, db_url=request.db_url, additional_params=request.additional_params, store_in_local=request.store_in_local, created_by=admin_email, dataset_name=request.dataset_name)
        except ValueError as ve:
            if str(ve) == 'DUPLICATE_DATASET_NAME':
                raise HTTPException(status_code=409, detail='A dataset with this name already exists.')
            raise
        if is_first_save:
            try:
                client_config = await db.client_configs.find_one({'client_id': admin_client_id})
                if client_config:
                    client_name = client_config.get('name', admin_client_id)
                    database_prefix = client_config.get('database_prefix', admin_client_id.upper())
                    facilities_list = []
                    orgs_cursor = db.organizations.find({'client_id': admin_client_id})
                    async for org in orgs_cursor:
                        facilities_list.append({'name': org.get('facility_name', ''), 'org_id': org.get('organization_id', 0)})
                    if not facilities_list:
                        facilities = client_config.get('metadata', {}).get('manufacturing_units', [])
                        facilities_list = [{'name': f, 'org_id': i + 1} for i, f in enumerate(facilities)] if facilities else []
                    logger.info(f'Creating XML directory structure for client {admin_client_id} after DB config save')
                    xml_success = await create_xml_directory_structure(admin_client_id, client_name, facilities_list, database_prefix, include_base_metadata=False, db_type=request.db_type)
                    if xml_success:
                        logger.info(f'Successfully created XML directory structure for client {admin_client_id}')
                    else:
                        logger.warning(f'Failed to create XML directory structure for client {admin_client_id}')
                else:
                    logger.warning(f'Client config not found for {admin_client_id}, skipping XML directory creation')
            except Exception as e:
                logger.error(f'Error creating XML directory structure after DB config save: {e}', exc_info=True)
        await audit_logger.log_event(event_type=AuditEventType.DB_CREDENTIALS_SAVED, severity=AuditSeverity.CRITICAL, user_id=admin_email, client_id=admin_client_id, details={'db_type': request.db_type, 'db_host': request.db_host, 'db_name': request.db_name, 'db_username': request.db_username, 'action': 'save_credentials'})
        logger.info(f'Successfully saved DB credentials for client {admin_client_id}')
        if not request.skip_metadata_refresh:
            asyncio.create_task(_refresh_metadata_after_save(client_id=admin_client_id, db_type=request.db_type, store_in_local=request.store_in_local, db_host=request.db_host, db_port=request.db_port, db_name=request.db_name, db_username=request.db_username, db_password=request.db_password, db_url=request.db_url, dataset_id=result.get('dataset_id'), additional_params=request.additional_params))
        else:
            logger.info(f'Skipping automatic metadata refresh for client {admin_client_id} (will be generated manually)')
        return {'success': True, 'message': 'Database credentials saved successfully', 'data': result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to save credentials: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'Failed to save credentials: {str(e)}')

class ToggleDatasetBody(BaseModel):
    dataset_id: str
    is_enabled: bool

class RenameDatasetBody(BaseModel):
    dataset_id: str
    dataset_name: str

@router.get('/load')
async def load_credentials(dataset_id: str=Query(..., description='Dataset id to load'), db_type: str=Query(default=None, description='Optional db_type filter'), admin_user: Dict=Depends(require_admin), db=Depends(get_db)):
    try:
        admin_client_id = admin_user.get('client_id')
        admin_email = admin_user.get('email', 'unknown')
        if not admin_client_id:
            raise HTTPException(status_code=403, detail='Admin user missing client_id. Please contact system administrator.')
        logger.info(f"Admin {admin_email} loading DB credentials | client_id={admin_client_id} | db_type={db_type or 'any'}")
        service = DBCredentialsService(db)
        credentials = await service.get_credentials(client_id=admin_client_id, db_type=db_type, decrypt_password=True, dataset_id=dataset_id)
        if not credentials:
            return {'success': True, 'message': 'No credentials found', 'data': None}
        logger.info(f"Successfully loaded DB credentials for client {admin_client_id} (type: {credentials.get('db_type')})")
        return {'success': True, 'message': 'Credentials loaded successfully', 'data': credentials}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to load credentials: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'Failed to load credentials: {str(e)}')

@router.get('/list')
async def list_credentials(include_password: bool=Query(default=False, description='Include decrypted passwords'), enabled_only: bool=Query(default=False, description='Only return enabled datasets'), admin_user: Dict=Depends(require_admin), db=Depends(get_db)):
    try:
        admin_client_id = admin_user.get('client_id')
        admin_email = admin_user.get('email', 'unknown')
        if not admin_client_id:
            raise HTTPException(status_code=403, detail='Admin user missing client_id. Please contact system administrator.')
        logger.info(f'Admin {admin_email} listing DB credentials | client_id={admin_client_id} | include_password={include_password}')
        service = DBCredentialsService(db)
        credentials = await service.list_credentials(client_id=admin_client_id, include_password=include_password, enabled_only=enabled_only)
        logger.info(f'Found {len(credentials)} credential sets for client {admin_client_id}')
        return {'success': True, 'message': f'Found {len(credentials)} credential set(s)', 'data': credentials}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to list credentials: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'Failed to list credentials: {str(e)}')

@router.delete('/delete')
async def delete_credentials(dataset_id: str=Query(..., description='Dataset id to delete'), admin_user: Dict=Depends(require_admin), db=Depends(get_db)):
    try:
        admin_client_id = admin_user.get('client_id')
        admin_email = admin_user.get('email', 'unknown')
        if not admin_client_id:
            raise HTTPException(status_code=403, detail='Admin user missing client_id. Please contact system administrator.')
        logger.info(f'Admin {admin_email} deleting DB credentials | client_id={admin_client_id} | dataset_id={dataset_id}')
        service = DBCredentialsService(db)
        try:
            deleted = await service.delete_credentials(client_id=admin_client_id, dataset_id=dataset_id)
        except ValueError as ve:
            if str(ve) == 'LAST_ENABLED_DATASET':
                raise HTTPException(status_code=400, detail='Cannot delete the last enabled dataset.')
            raise
        if not deleted:
            raise HTTPException(status_code=404, detail='No credentials found for this dataset_id')
        await audit_logger.log_event(event_type=AuditEventType.DB_CREDENTIALS_DELETED, severity=AuditSeverity.CRITICAL, user_id=admin_email, client_id=admin_client_id, details={'dataset_id': dataset_id, 'action': 'delete_credentials'})
        logger.info(f'Successfully deleted DB credentials for client {admin_client_id}')
        return {'success': True, 'message': f'Credentials deleted successfully', 'data': {'client_id': admin_client_id, 'dataset_id': dataset_id}}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to delete credentials: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'Failed to delete credentials: {str(e)}')

@router.patch('/toggle')
async def toggle_dataset_endpoint(body: ToggleDatasetBody, admin_user: Dict=Depends(require_admin), db=Depends(get_db)):
    admin_client_id = admin_user.get('client_id')
    if not admin_client_id:
        raise HTTPException(status_code=403, detail='Admin user missing client_id.')
    service = DBCredentialsService(db)
    try:
        ok = await service.toggle_dataset(admin_client_id, body.dataset_id, body.is_enabled)
    except ValueError as ve:
        if str(ve) == 'LAST_ENABLED_DATASET':
            raise HTTPException(status_code=400, detail='Cannot disable the last enabled dataset.')
        raise
    if not ok:
        raise HTTPException(status_code=404, detail='Dataset not found.')
    return {'success': True, 'message': 'Dataset updated'}

@router.patch('/rename')
async def rename_dataset_endpoint(body: RenameDatasetBody, admin_user: Dict=Depends(require_admin), db=Depends(get_db)):
    admin_client_id = admin_user.get('client_id')
    if not admin_client_id:
        raise HTTPException(status_code=403, detail='Admin user missing client_id.')
    service = DBCredentialsService(db)
    try:
        ok = await service.update_dataset_name(admin_client_id, body.dataset_id, body.dataset_name)
    except ValueError as ve:
        if str(ve) == 'DUPLICATE_DATASET_NAME':
            raise HTTPException(status_code=409, detail='A dataset with this name already exists.')
        if 'cannot be empty' in str(ve).lower():
            raise HTTPException(status_code=400, detail=str(ve))
        raise
    if not ok:
        raise HTTPException(status_code=404, detail='Dataset not found.')
    return {'success': True, 'message': 'Dataset renamed'}

@router.post('/reset')
async def reset_db_config(admin_user: Dict=Depends(require_admin), db=Depends(get_db)):
    try:
        admin_client_id = admin_user.get('client_id')
        admin_email = admin_user.get('email', 'unknown')
        if not admin_client_id:
            raise HTTPException(status_code=403, detail='Admin user missing client_id. Please contact system administrator.')
        service = DBCredentialsService(db)
        deleted_count = await service.delete_credentials_for_client(admin_client_id)
        deleted = deleted_count > 0
        cleanup_client_data(admin_client_id, preserve_uploads=False)
        await audit_logger.log_event(event_type=AuditEventType.DB_CREDENTIALS_DELETED, severity=AuditSeverity.CRITICAL, user_id=admin_email, client_id=admin_client_id, details={'action': 'reset_db_config', 'credentials_deleted': deleted})
        logger.info('DB config reset for client %s by %s (credentials_deleted=%s)', admin_client_id, admin_email, deleted)
        return {'success': True, 'message': 'DB config reset successfully. All credentials and associated data have been removed.', 'data': {'client_id': admin_client_id, 'credentials_deleted': deleted}}
    except HTTPException:
        raise
    except Exception as e:
        logger.error('Failed to reset DB config: %s', e, exc_info=True)
        raise HTTPException(status_code=500, detail=f'Failed to reset DB config: {str(e)}')

@router.post('/test-connection')
async def test_connection(request: TestConnectionRequest, admin_user: Dict=Depends(require_admin), db=Depends(get_db)):
    try:
        admin_email = admin_user.get('email', 'unknown')
        logger.info(f'Admin {admin_email} testing DB connection | db_type={request.db_type} | db_host={request.db_host}')
        service = DBCredentialsService(db)
        ssh_config = (request.additional_params or {}).get('ssh') if request.additional_params else None
        result = await service.test_connection(db_type=request.db_type, db_host=request.db_host, db_port=request.db_port, db_name=request.db_name, db_username=request.db_username, db_password=request.db_password, db_url=request.db_url, ssh_config=ssh_config)
        if result['success']:
            logger.info(f'Connection test successful for admin {admin_email}')
            from util.db_size import get_database_size_bytes, format_size_mb, MAX_DB_SIZE_BYTES
            db_size_bytes = await get_database_size_bytes(db_type=request.db_type, db_host=request.db_host, db_port=request.db_port, db_name=request.db_name, db_username=request.db_username, db_password=request.db_password, db_url=request.db_url)
            is_super_admin = admin_user.get('role') == 'super_admin'
            size_exceeded = db_size_bytes > MAX_DB_SIZE_BYTES and (not is_super_admin) and (db_size_bytes > 0)
            if size_exceeded:
                return {'success': False, 'message': f'Database size ({format_size_mb(db_size_bytes)} MB) exceeds the maximum allowed ({format_size_mb(MAX_DB_SIZE_BYTES)} MB). Please upgrade your plan or reduce the database size.', 'data': {'db_type': result['db_type'], 'db_host': result.get('db_host'), 'db_name': result.get('db_name'), 'db_size_bytes': db_size_bytes, 'max_db_size_bytes': MAX_DB_SIZE_BYTES}}
            return {'success': True, 'message': result['message'], 'data': {'db_type': result['db_type'], 'db_host': result.get('db_host'), 'db_name': result.get('db_name'), 'db_size_bytes': db_size_bytes, 'max_db_size_bytes': MAX_DB_SIZE_BYTES}}
        else:
            logger.warning(f"Connection test failed for admin {admin_email}: {result['message']}")
            return {'success': result['success'], 'message': result['message'], 'data': {'db_type': result['db_type'], 'db_host': result.get('db_host'), 'db_name': result.get('db_name')}}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to test connection: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'Failed to test connection: {str(e)}')

@router.get('/health')
async def health_check(admin_user: Dict=Depends(require_admin)):
    return {'success': True, 'message': 'DB Credentials service is healthy', 'admin_user': admin_user.get('email', 'unknown')}

def _ensure_asyncpg_dsn(dsn: str) -> str:
    if dsn.startswith('postgresql://') or dsn.startswith('postgres://'):
        return dsn.replace('postgresql://', 'postgresql+asyncpg://', 1).replace('postgres://', 'postgresql+asyncpg://', 1)
    return dsn

def _build_mongodb_dsn(host: str, port: int, db_name: str, user: str, password: str, provided_url: Optional[str]=None) -> str:
    if provided_url and ('mongodb+srv' in provided_url or 'mongodb.net' in provided_url):
        return provided_url
    return f'mongodb+srv://{user}:{password}@{host}/{db_name}'

def _cleanup_metadata_dirs(client_id: str, preserve_uploads: bool=False, preserve_datasets: bool=False, dataset_id: Optional[str]=None) -> None:
    try:
        if dataset_id:
            cleanup_client_data(client_id, preserve_uploads=preserve_uploads, dataset_id=dataset_id)
            return
        xml_data_sources_dir = Path('xml_prompts/clients') / client_id / 'data_sources'
        if xml_data_sources_dir.exists():
            shutil.rmtree(xml_data_sources_dir)
            logger.info(f'Removed XML data sources for client {client_id}: {xml_data_sources_dir}')
        datasets_dir = Path(f'assets/clients/{client_id}/datasets')
        if datasets_dir.exists() and (not preserve_datasets):
            shutil.rmtree(datasets_dir)
            logger.info(f'Removed datasets for client {client_id}: {datasets_dir}')
        uploads_dir = Path(f'assets/clients/{client_id}/uploads')
        if uploads_dir.exists() and (not preserve_uploads):
            shutil.rmtree(uploads_dir)
            logger.info(f'Removed uploads for client {client_id}: {uploads_dir}')
    except Exception as e:
        logger.warning(f'Failed to clean metadata directories for client {client_id}: {e}', exc_info=True)

async def _refresh_metadata_after_save(client_id: str, db_type: str, store_in_local: bool, db_host: str, db_port: int, db_name: str, db_username: str, db_password: str, db_url: str, dataset_id: Optional[str]=None, additional_params: Optional[dict]=None) -> None:
    try:
        if db_type == 'file_upload':
            await _generate_file_upload_metadata(client_id, dataset_id=dataset_id)
        elif db_type == 'postgres':
            await _generate_sql_metadata(client_id=client_id, db_type='postgres', store_in_local=store_in_local, db_host=db_host, db_port=db_port, db_name=db_name, db_username=db_username, db_password=db_password, db_url=db_url, dataset_id=dataset_id, additional_params=additional_params)
        elif db_type == 'mysql':
            await _generate_sql_metadata(client_id=client_id, db_type='mysql', store_in_local=store_in_local, db_host=db_host, db_port=db_port, db_name=db_name, db_username=db_username, db_password=db_password, db_url=db_url, dataset_id=dataset_id, additional_params=additional_params)
        elif db_type == 'sap_hana':
            await _generate_sql_metadata(client_id=client_id, db_type='sap_hana', store_in_local=store_in_local, db_host=db_host, db_port=db_port, db_name=db_name, db_username=db_username, db_password=db_password, db_url=db_url, dataset_id=dataset_id, additional_params=additional_params)
        elif db_type == 'mongodb':
            await _generate_mongodb_metadata(client_id=client_id, db_host=db_host, db_port=db_port, db_name=db_name, db_username=db_username, db_password=db_password, db_url=db_url, store_in_local=store_in_local, dataset_id=dataset_id, additional_params=additional_params)
        elif db_type == 'sap_oracle':
            logger.info(f'Metadata refresh skipped for sap_oracle (uses predefined base_sap metadata) for client {client_id}')
        elif db_type == 'sap_sybase':
            logger.info(f'Metadata refresh skipped for sap_sybase (uses predefined base_sap metadata) for client {client_id}')
        else:
            logger.info(f'Metadata refresh skipped: unsupported db_type {db_type} for client {client_id}')
    except Exception as e:
        logger.error(f'Metadata refresh failed for client {client_id}: {e}', exc_info=True)

async def _generate_sql_metadata(client_id: str, db_type: str, store_in_local: bool, db_host: str, db_port: int, db_name: str, db_username: str, db_password: str, db_url: str, dataset_id: Optional[str]=None, additional_params: Optional[dict]=None) -> None:
    dsn = db_url.strip() if db_url else ''
    if not dsn:
        if db_type == 'postgres':
            dsn = f'postgresql+asyncpg://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
        elif db_type == 'mysql':
            dsn = f'mysql+aiomysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
        elif db_type == 'sap_hana':
            dsn = f'hana://{db_username}:{db_password}@{db_host}:{db_port}'
        elif db_type == 'sap_sybase':
            from db_config.connectors.sybase_connector import build_sybase_url
            dsn = build_sybase_url(db_host, db_port, db_username, db_password, db_name)
    if db_type == 'postgres':
        from explorer.explorer_routes import _ensure_asyncpg_dsn
        dsn = _ensure_asyncpg_dsn(dsn)
    _cleanup_metadata_dirs(client_id, dataset_id=dataset_id)
    ssh_config = additional_params.get('ssh') if additional_params else None
    if db_type == 'mysql':
        connector = MySQLConnector(dsn, ssh_config=ssh_config)
    elif db_type == 'sap_hana':
        connector = SAPHANAConnector(dsn)
    elif db_type == 'sap_oracle':
        connector = OracleConnector(dsn, ssh_config=ssh_config)
    elif db_type == 'sap_sybase':
        connector = SybaseConnector(dsn)
    else:
        connector = PostgresConnector(dsn, ssh_config=ssh_config)
    try:
        await connector.connect()
        session_factory = connector.get_db()
        output_root = Path('xml_prompts/clients') / client_id
        db_instance = await get_db()
        agent = ExplorerAgent(client_id=client_id, session_factory=session_factory, output_root=output_root, store_in_local=store_in_local, db=db_instance, db_type=db_type, db_name=db_name, db_username=db_username, dataset_id=dataset_id)
        await agent.run()
        logger.info(f'Explorer metadata regenerated for client {client_id} ({db_type})')
    except Exception as e:
        logger.error(f'Failed to regenerate {db_type} metadata for client {client_id}: {e}', exc_info=True)
    finally:
        try:
            await connector.disconnect()
        except Exception:
            pass

async def _generate_mongodb_metadata(client_id: str, db_host: str, db_port: int, db_name: str, db_username: str, db_password: str, db_url: str, store_in_local: bool, dataset_id: Optional[str]=None, additional_params: Optional[dict]=None) -> None:
    dsn = _build_mongodb_dsn(db_host, db_port, db_name, db_username, db_password, db_url)
    _cleanup_metadata_dirs(client_id, dataset_id=dataset_id)
    ssh_config = additional_params.get('ssh') if additional_params else None
    connector = MongoConnector(dsn, db_name, ssh_config=ssh_config)
    try:
        await connector.connect()
        session_factory = connector.get_db()
        output_root = Path('xml_prompts/clients') / client_id
        from db_config.mongo_server import get_db
        db_instance = await get_db()
        agent = ExplorerAgent(client_id=client_id, session_factory=session_factory, output_root=output_root, store_in_local=store_in_local, db=db_instance, db_type='mongodb', db_name=db_name, dataset_id=dataset_id)
        await agent.run()
        logger.info(f'Explorer metadata regenerated for client {client_id} (mongodb)')
    except Exception as e:
        logger.error(f'Failed to regenerate MongoDB metadata for client {client_id}: {e}', exc_info=True)
    finally:
        try:
            await connector.disconnect()
        except Exception:
            pass

async def _generate_file_upload_metadata(client_id: str, dataset_id: Optional[str]=None) -> None:
    _cleanup_metadata_dirs(client_id, preserve_uploads=True, preserve_datasets=True, dataset_id=dataset_id)
    parquet_dir = assets_datasets_dir(client_id, dataset_id)
    if not parquet_dir.exists():
        logger.info(f'Skipped file upload metadata refresh for {client_id}: no datasets directory')
        return
    parquet_files = list(parquet_dir.glob('*.parquet'))
    if not parquet_files:
        logger.info(f'Skipped file upload metadata refresh for {client_id}: no parquet files found')
        return
    output_root = Path('xml_prompts/clients') / client_id
    metadata_generator = FileMetadataGenerator(client_id=client_id, output_root=output_root, dataset_id=dataset_id)
    try:
        await metadata_generator.generate_metadata(parquet_dir, max_sample_rows=100)
        logger.info(f'File upload metadata regenerated for client {client_id}')
    except Exception as e:
        logger.error(f'Failed to regenerate file upload metadata for client {client_id}: {e}', exc_info=True)