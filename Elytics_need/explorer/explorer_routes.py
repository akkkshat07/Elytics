from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends, UploadFile, File
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
import os
import asyncio
from pathlib import Path
import json
import logging
# SECURITY: Use defusedxml for parsing (prevents XXE attacks)
import defusedxml.ElementTree as ET
from defusedxml.ElementTree import parse, fromstring
# Import Element creation classes from standard library (safe for creation, not parsing)
# Safe: Only used for element creation, not parsing untrusted input. All parsing uses defusedxml.
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml
from glob import glob
import shutil
from datetime import datetime
from util.time_utils import utcnow
import aiofiles

from explorer.explorer_agent import ExplorerAgent, copy_base_prompts_for_client
from explorer.file_schema_extractor import FileSchemaExtractor
from explorer.file_metadata_generator import FileMetadataGenerator
from explorer.schema_limits import apply_table_column_limits
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import MetaData, text
from middleware.auth_middleware import require_admin
from db_config.connectors.postgres_connector import PostgresConnector
from db_config.connectors.mysql_connector import MySQLConnector
from db_config.connectors.mongo_connector import MongoConnector
from db_config.connectors.sap_hana_connector import SAPHANAConnector
from db_config.connectors.oracle_connector import OracleConnector
from db_config.connectors.sybase_connector import SybaseConnector
from util.audit_logger import audit_logger, AuditEventType, AuditSeverity
from util.file_security import sanitize_filename, validate_path_within_directory
from util.client_data_cleanup import cleanup_client_data
from util.dataset_paths import assets_uploads_dir, assets_datasets_dir, resolve_xml_data_sources_dir
from util.dataset_paths import storage_datasets_prefix, storage_uploads_prefix, storage_xml_data_sources_prefix
from util.data_source import require_store_in_local
from db_config.mongo_server import get_db
from services.db_credentials_service import DBCredentialsService
from services.subscription_service import get_explorer_limits
from config.system_config import DEFAULT_LLM_PROVIDER, LLM_PROVIDERS, STORAGE_BACKEND
import hdbcli.dbapi

logger = logging.getLogger(__name__)


def _cleanup_existing_client_data(
    client_id: str, preserve_uploads: bool = False, dataset_id: Optional[str] = None
) -> None:
    """Thin wrapper for backward compatibility; delegates to util.client_data_cleanup."""
    cleanup_client_data(client_id, preserve_uploads=preserve_uploads, dataset_id=dataset_id)


def _cleanup_for_file_upload(client_id: str, dataset_id: Optional[str] = None) -> None:
    """Clean up for file upload processing; preserves uploads directory."""
    cleanup_client_data(client_id, preserve_uploads=True, dataset_id=dataset_id)


def _empty_upload_folder(client_id: str, dataset_id: Optional[str] = None) -> None:
    """
    Empty the upload folder for a client (remove all contents).
    Used before adding new uploads so each upload replaces previous content.
    """
    uploads_dir = assets_uploads_dir(client_id, dataset_id)
    if not uploads_dir.exists():
        pass
    else:
        try:
            for item in uploads_dir.iterdir():
                if item.is_file():
                    item.unlink()
                else:
                    shutil.rmtree(item)
            logger.info(f"Emptied upload folder for client {client_id}: {uploads_dir}")
        except Exception as e:
            logger.warning(f"Error emptying upload folder for client {client_id}: {e}", exc_info=True)

    # Also clean GCS if enabled
    if STORAGE_BACKEND == "gcs":
        try:
            import asyncio
            from util.storage.backend import get_storage_backend
            storage = get_storage_backend()
            prefix = storage_uploads_prefix(client_id, dataset_id)
            asyncio.get_event_loop().run_until_complete(storage.delete_prefix(prefix))
        except Exception as e:
            logger.warning(f"Error emptying GCS upload folder for client {client_id}: {e}")


async def _sync_file_to_gcs(local_path: Path, client_id: str, dataset_id: Optional[str], subfolder: str = "datasets") -> None:
    """Upload a locally-saved file to GCS if STORAGE_BACKEND=gcs."""
    if STORAGE_BACKEND != "gcs":
        return
    try:
        from util.storage.backend import get_storage_backend
        storage = get_storage_backend()
        if subfolder == "datasets":
            prefix = storage_datasets_prefix(client_id, dataset_id)
        elif subfolder == "uploads":
            prefix = storage_uploads_prefix(client_id, dataset_id)
        else:
            prefix = f"clients/{client_id}/{subfolder}"
        remote_key = f"{prefix}/{local_path.name}"
        await storage.upload_file(str(local_path), remote_key)
        logger.debug(f"Synced {local_path.name} to GCS: {remote_key}")
    except Exception as e:
        logger.warning(f"Failed to sync {local_path} to GCS: {e}")

router = APIRouter()


async def _validate_llm_config_available(client_id: str, db: Any) -> tuple[bool, Optional[str]]:
    """
    Validate that LLM configuration is available for the explorer agent.
    
    Checks in order:
    1. Client-specific default LLM config in MongoDB
    2. System defaults from system_config.py (checks if API key env var exists)
    
    Args:
        client_id: Client identifier
        db: MongoDB database instance
        
    Returns:
        Tuple of (is_available: bool, error_message: Optional[str])
    """
    try:
        # Check for client-specific LLM config in MongoDB
        collection = db.llm_configurations
        config_doc = await collection.find_one({"client_id": client_id})
        
        if config_doc and config_doc.get("configurations"):
            # Look for default+active config
            for config in config_doc.get("configurations", []):
                if config.get("is_default") and config.get("is_active"):
                    api_key = config.get("api_key")
                    if api_key:
                        logger.info(f"Found client-specific LLM config for client {client_id}")
                        return True, None
        
        # No client config found, check system defaults
        logger.info(f"No client-specific LLM config found for {client_id}, checking system defaults")
        
        # Check if system default provider has API key configured
        default_provider = DEFAULT_LLM_PROVIDER
        provider_config = LLM_PROVIDERS.get(default_provider, {})
        api_key_env_var = provider_config.get("api_key_env_var")
        
        if api_key_env_var:
            api_key = os.getenv(api_key_env_var)
            if api_key:
                logger.info(f"Found system default LLM config: provider={default_provider}")
                return True, None
        
        # No LLM config available
        error_msg = (
            "No LLM configuration available. Please configure an LLM in the Admin Dashboard "
            "(LLM Configuration tab) or ensure system defaults are set in system_config.py. "
            "The explorer agent requires an LLM to generate table introductions and column descriptions."
        )
        logger.error(f"LLM config validation failed for client {client_id}: {error_msg}")
        return False, error_msg
        
    except Exception as e:
        logger.error(f"Error validating LLM config for client {client_id}: {e}", exc_info=True)
        error_msg = (
            "Error checking LLM configuration. Please ensure an LLM is configured in the Admin Dashboard "
            "or system defaults are available."
        )
        return False, error_msg

class ExplorerRequest(BaseModel):
    client_id: str
    schema_filter: Optional[str] = None
    table_prefix: Optional[str] = None
    store_in_local: bool
    dataset_id: Optional[str] = None

class TableListRequest(BaseModel):
    db_host: str
    db_port: int
    db_name: str
    db_username: str
    db_password: str
    db_url: Optional[str] = ""
    db_type: str = "postgres"
    schema_name: Optional[str] = None  # Renamed from 'schema' to avoid conflict with BaseModel.schema
    namespace: Optional[Dict[str, Any]] = None
    additional_params: Optional[Dict[str, Any]] = None
    
    @validator('db_host', 'db_name', 'db_username', 'db_password', 'db_url', pre=True)
    def validate_and_strip(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return v.strip()
        return v
    
    @validator('db_port')
    def validate_port(cls, v):
        if not isinstance(v, int) or v < 1 or v > 65535:
            raise ValueError('Port must be a number between 1 and 65535')
        return v

class ExplorerAdhocRequest(BaseModel):
    """Request to run explorer agent with explicit DB credentials."""
    client_id: str
    db_host: str
    db_port: int
    db_name: str
    db_username: str
    db_password: str
    db_url: Optional[str] = ""
    db_type: str = "postgres"  # New field
    schema_filter: Optional[str] = None  # Defaults to None, ExplorerAgent will handle dialector-specific defaults
    table_prefix: Optional[str] = None
    store_in_local: bool
    table_filter: Optional[List[str]] = None  # Optional list of table names to include (only generate metadata for these)
    namespace: Optional[Dict[str, Any]] = None
    dataset_id: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None

    @validator('db_host', 'db_name', 'db_username', 'db_password', 'db_url', pre=True)
    def validate_and_strip(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            v = v.strip()
            # Special fix for cases where whitespace might be inside the URL or host
            # but usually it's leading/trailing
            return v
        return v
    
    @validator('db_port')
    def validate_port(cls, v):
        if not isinstance(v, int) or v < 1 or v > 65535:
            raise ValueError('Port must be a number between 1 and 65535')
        return v

def _dsn(host: str, port: int, db: str, user: str, password: str) -> str:
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"

def _oracle_dsn(host: str, port: int, service: str, user: str, password: str) -> str:
    return f"oracle+oracledb://{user}:{password}@{host}:{port}/?service_name={service}"

def _ensure_asyncpg_dsn(dsn: str) -> str:
    """Ensure DSN uses asyncpg driver for async operations."""
    if dsn.startswith('postgresql://') or dsn.startswith('postgres://'):
        return dsn.replace('postgresql://', 'postgresql+asyncpg://', 1).replace('postgres://', 'postgresql+asyncpg://', 1)
    return dsn

def _build_mongodb_dsn(host: str, port: int, db_name: str, user: str, password: str, provided_url: Optional[str] = None) -> str:
    """Build MongoDB connection string using Atlas SRV format."""
    if provided_url and ("mongodb+srv" in provided_url or "mongodb.net" in provided_url):
        return provided_url
    
    # Always return SRV format for Atlas
    return f"mongodb+srv://{user}:{password}@{host}/{db_name}"


SYSTEM_SCHEMAS = {
    "information_schema",
    "pg_catalog",
    "mysql",
    "performance_schema",
    "sys",
    "sysibm",
    "syscat",
    "system",
}


def _namespace_name(namespace: Optional[Dict[str, Any]]) -> Optional[str]:
    if not namespace:
        return None
    value = namespace.get("name") or namespace.get("namespace_id")
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _namespace_type(namespace: Optional[Dict[str, Any]]) -> Optional[str]:
    if not namespace:
        return None
    value = namespace.get("namespace_type")
    return str(value).strip().lower() if value else None


def _namespace_payload(name: str, namespace_type: str, table_count: int = 0, is_default: bool = False) -> Dict[str, Any]:
    return {
        "namespace_id": name,
        "namespace_type": namespace_type,
        "name": name,
        "display_name": name,
        "table_count": int(table_count or 0),
        "is_default": bool(is_default),
    }


def _table_identity(table: Dict[str, Any], schema: Optional[str]) -> Dict[str, Any]:
    name = str(table.get("name") or table.get("table_name") or "")
    namespace_id = table.get("namespace_id") or schema or table.get("schema")
    qualified = f"{namespace_id}.{name}" if namespace_id else name
    table["schema"] = table.get("schema") or schema
    table["namespace_id"] = namespace_id
    table["qualified_name"] = table.get("qualified_name") or qualified
    table["table_id"] = table.get("table_id") or qualified
    return table


async def _build_table_list_dsn(request: TableListRequest) -> str:
    if request.db_type == "mysql":
        db_name = _namespace_name(request.namespace) or request.db_name
        if request.db_url and "mysql" in request.db_url and not _namespace_name(request.namespace):
            return request.db_url
        return f"mysql+aiomysql://{request.db_username}:{request.db_password}@{request.db_host}:{request.db_port}/{db_name}"
    if request.db_type == "mongodb":
        db_name = _namespace_name(request.namespace) or request.db_name
        return _build_mongodb_dsn(
            request.db_host,
            request.db_port,
            db_name,
            request.db_username,
            request.db_password,
            request.db_url if not _namespace_name(request.namespace) else None,
        )
    if request.db_type == "sap_hana":
        if request.db_url and "hana" in request.db_url:
            return request.db_url
        return f"hana://{request.db_username}:{request.db_password}@{request.db_host}:{request.db_port}"
    if request.db_type == "sap_oracle":
        if request.db_url and ("oracle" in request.db_url or "oracledb" in request.db_url):
            return request.db_url
        return _oracle_dsn(
            request.db_host,
            request.db_port,
            request.db_name,
            request.db_username,
            request.db_password,
        )
    if request.db_type == "sap_sybase":
        if request.db_url and "sybase" in request.db_url:
            return request.db_url
        from db_config.connectors.sybase_connector import build_sybase_url
        return build_sybase_url(request.db_host, request.db_port, request.db_username, request.db_password, request.db_name)
    if request.db_url and ("postgres" in request.db_url or "postgresql" in request.db_url):
        if _namespace_type(request.namespace) == "database" and _namespace_name(request.namespace):
            return _dsn(request.db_host, request.db_port, _namespace_name(request.namespace), request.db_username, request.db_password)
        return _ensure_asyncpg_dsn(request.db_url)
    postgres_db_name = _namespace_name(request.namespace) if _namespace_type(request.namespace) == "database" else request.db_name
    return _dsn(request.db_host, request.db_port, postgres_db_name or request.db_name, request.db_username, request.db_password)

async def run_explorer_task(request: ExplorerRequest):
    """
    Background task to run the explorer agent.
    """
    client_id = request.client_id
    logger.info(f"Starting Explorer Agent for client: {client_id}")
    
    # Clean up existing data before generating new data
    _cleanup_existing_client_data(client_id, dataset_id=request.dataset_id)
    
    # Get database credentials from db_credentials collection for this client
    db = await get_db()
    
    # Validate LLM configuration is available
    is_available, error_msg = await _validate_llm_config_available(client_id, db)
    if not is_available:
        raise RuntimeError(error_msg)
    
    service = DBCredentialsService(db)
    credentials = await service.get_credentials(
        client_id=client_id,
        db_type=None,
        decrypt_password=True,
        dataset_id=request.dataset_id,
    )
    logger.info(f"Fetched credentials: {credentials}")
    
    if not credentials:
        raise RuntimeError(
            f"No database credentials found for client {client_id}. "
            "Please configure database credentials first via /api/db-credentials/save endpoint."
        )
    
    # Determine db_type and use db_url if available, otherwise construct it
    db_type = credentials.get("db_type", "postgres")
    db_url = credentials.get("db_url")
    
    # If it's MongoDB, only use db_url if it's an Atlas URL
    if db_type == "mongodb" and db_url:
        if not ("mongodb+srv" in db_url or "mongodb.net" in db_url):
            logger.info("Non-Atlas MongoDB URL found in credentials, will reconstruct from fields")
            db_url = None

            
    db_username = credentials.get("db_username")
    db_password = credentials.get("db_password") 
    if not db_url:
        db_host = credentials.get("db_host")
        db_port = credentials.get("db_port")
        db_name = credentials.get("db_name")
    
        
        if not all([db_host, db_port, db_name, db_username, db_password]):
            raise RuntimeError(
                f"Incomplete database credentials for client {client_id}. "
                "Missing required fields."
            )
        
        if db_type == "postgres":
            db_url = f"postgresql+asyncpg://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
        elif db_type == "mysql":
            db_url = f"mysql+aiomysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
        elif db_type == "mongodb":
            db_url = _build_mongodb_dsn(db_host, db_port, db_name, db_username, db_password)
        elif db_type == "sap_hana":
            db_url = f"hana://{db_username}:{db_password}@{db_host}:{db_port}"
            if int(db_port) == 443:
                db_url += "?encrypt=true&sslValidateCertificate=false"
        elif db_type == "sap_oracle":
            if db_url and ("oracle" in db_url or "oracledb" in db_url):
                # Use provided Oracle DSN
                pass
            else:
                db_url = _oracle_dsn(db_host, db_port, db_name, db_username, db_password)
        else:
            raise RuntimeError(f"Unsupported database type: {db_type}")

    ssh_config = None
    additional_params_creds = credentials.get("additional_params") or {}
    ssh_config = additional_params_creds.get("ssh")

    if db_type == "mysql":
         connector = MySQLConnector(db_url, ssh_config=ssh_config)
    elif db_type == "mongodb":
         connector = MongoConnector(db_url, credentials.get("db_name"), ssh_config=ssh_config)
    elif db_type == "sap_hana":
         connector = SAPHANAConnector(db_url)
    elif db_type == "sap_oracle":
         connector = OracleConnector(db_url, ssh_config=ssh_config)
    else:
         dsn = _ensure_asyncpg_dsn(db_url)
         connector = PostgresConnector(dsn, ssh_config=ssh_config)
    
    try:
        await connector.connect()
        
        # Enforce database size limit (skip for file_upload)
        if db_type not in ("file_upload",):
            from util.db_size import get_database_size_bytes, format_size_mb, MAX_DB_SIZE_BYTES
            db_size_bytes = await get_database_size_bytes(
                db_type=db_type,
                db_host=credentials.get("db_host", ""),
                db_port=credentials.get("db_port", 0),
                db_name=credentials.get("db_name", ""),
                db_username=credentials.get("db_username", ""),
                db_password=credentials.get("db_password", ""),
                db_url=db_url,
            )
            if db_size_bytes > 0 and db_size_bytes > MAX_DB_SIZE_BYTES:
                raise RuntimeError(
                    f"Database size ({format_size_mb(db_size_bytes)} MB) exceeds the maximum allowed "
                    f"({format_size_mb(MAX_DB_SIZE_BYTES)} MB). Please upgrade your plan."
                )
        
        session_factory = connector.get_db()
        
        # Define output path relative to project root
        # Assuming running from project root, output to xml_prompts/clients/{client_id}
        output_root = Path("xml_prompts/clients") / client_id
        
        # Adjust schema filter for MySQL
        # If schema_filter is "public" (default) and we are on MySQL, use db_name or None
        effective_schema_filter = request.schema_filter
        if db_type == "mysql" and request.schema_filter == "public":
            # Try to use db_name from credentials if available
            effective_schema_filter = credentials.get("db_name")
            # If db_name is not available (e.g. only db_url provided), use None (fetch all schemas)
            # or try to extract from db_url (too complex for now, None is safe)
        elif db_type == "sap_hana" and request.schema_filter == "public":
            # SAP HANA should default to None (all schemas) or current user schema
            effective_schema_filter = None
        
        agent = ExplorerAgent(
            client_id=client_id,
            session_factory=session_factory,
            output_root=output_root,
            schema_filter=effective_schema_filter,
            table_prefix=request.table_prefix,
            store_in_local=request.store_in_local,
            db=db,
            db_type=db_type,
            db_name=credentials.get("db_name"),
            db_username=db_username,
            dataset_id=credentials.get("dataset_id"),
        )
        
        await agent.run()
        logger.info(f"Explorer Agent completed for client: {client_id}")
        
        # Persist store_in_local to db_credentials
        try:
            service = DBCredentialsService(db)
            await service.update_store_in_local(
                client_id=client_id,
                db_type=credentials.get("db_type", "postgres"),
                store_in_local=request.store_in_local,
                dataset_id=credentials.get("dataset_id"),
            )
            logger.info(f"Persisted store_in_local={request.store_in_local} to db_credentials for client {client_id}")
        except Exception as e:
            logger.warning(f"Failed to persist store_in_local to db_credentials for client {client_id}: {e}")
            # Don't fail the explorer task if this fails
        
    except Exception as e:
        logger.error(f"Explorer Agent failed for client {client_id}: {str(e)}", exc_info=True)
    finally:
        await connector.disconnect()

@router.post("/run")
async def run_explorer(request: ExplorerRequest, background_tasks: BackgroundTasks, admin_user: Dict = Depends(require_admin)):
    """
    Trigger the Explorer Agent to analyze the database and generate metadata.
    This runs as a background task.
    """
    if admin_user.get("role") != "super_admin" and request.client_id != admin_user.get("client_id"):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        # Validate LLM config before starting background task
        db = await get_db()
        is_available, error_msg = await _validate_llm_config_available(request.client_id, db)
        if not is_available:
            raise HTTPException(status_code=400, detail=error_msg)
        
        background_tasks.add_task(run_explorer_task, request)
        return {
            "status": "accepted",
            "message": f"Explorer Agent started for client {request.client_id}",
            "details": {
                "client_id": request.client_id,
                "output_path": f"xml_prompts/clients/{request.client_id}",
                "export_datasets": request.store_in_local
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start Explorer Agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-adhoc")
async def run_explorer_adhoc(request: ExplorerAdhocRequest, background_tasks: BackgroundTasks, admin_user: Dict = Depends(require_admin)):
    """
    Run Explorer Agent with explicit DB credentials (no persistent client config needed).
    Generates XML metadata + LLM descriptions in xml_prompts/clients/{client_id}/.
    """
    if admin_user.get("role") != "super_admin" and request.client_id != admin_user.get("client_id"):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        # Validate LLM config before starting background task
        db = await get_db()
        is_available, error_msg = await _validate_llm_config_available(request.client_id, db)
        if not is_available:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Build DSN based on db_type and provided fields
        if request.db_type == "mysql":
            mysql_db_name = _namespace_name(request.namespace) or request.db_name
            if request.db_url and "mysql" in request.db_url:
                dsn = request.db_url
            else:
                dsn = f"mysql+aiomysql://{request.db_username}:{request.db_password}@{request.db_host}:{request.db_port}/{mysql_db_name}"
            logger.info(f"Using MySQL DSN for explorer agent")
        elif request.db_type == "mongodb":
            mongo_db_name = _namespace_name(request.namespace) or request.db_name
            dsn = _build_mongodb_dsn(
                request.db_host, 
                request.db_port, 
                mongo_db_name,
                request.db_username, 
                request.db_password, 
                request.db_url if not _namespace_name(request.namespace) else None,
            )
            logger.info(f"Using MongoDB DSN for explorer agent")
        elif request.db_type == "sap_hana":
            if request.db_url and request.db_url.startswith("hana://"):
                dsn = request.db_url
            else:
                dsn = f"hana://{request.db_username}:{request.db_password}@{request.db_host}:{request.db_port}"
                if int(request.db_port) == 443:
                    dsn += "?encrypt=true&sslValidateCertificate=false"
            logger.info(f"Using SAP HANA DSN for explorer agent: {dsn}")
        elif request.db_type == "sap_oracle":
            if request.db_url and ("oracle" in request.db_url or "oracledb" in request.db_url):
                dsn = request.db_url
            else:
                dsn = _oracle_dsn(
                    request.db_host,
                    request.db_port,
                    request.db_name,
                    request.db_username,
                    request.db_password,
                )
            logger.info("Using Oracle DSN for explorer agent")
        elif request.db_type == "sap_sybase":
            if request.db_url and "sybase" in request.db_url:
                dsn = request.db_url
            else:
                # Sybase connection string using ODBC connection string format
                from db_config.connectors.sybase_connector import build_sybase_url
                dsn = build_sybase_url(request.db_host, request.db_port, request.db_username, request.db_password, request.db_name)
            logger.info("Using Sybase DSN for explorer agent")
        else:
            postgres_db_name = _namespace_name(request.namespace) if _namespace_type(request.namespace) == "database" else request.db_name
            if request.db_url and ("postgres" in request.db_url or "postgresql" in request.db_url) and _namespace_type(request.namespace) != "database":
                dsn = _ensure_asyncpg_dsn(request.db_url)
            else:
                dsn = _dsn(request.db_host, request.db_port, postgres_db_name or request.db_name, request.db_username, request.db_password)
            logger.info(f"Using PostgreSQL DSN for explorer agent")
        
        _adhoc_ssh_config = None
        if request.additional_params:
            _adhoc_ssh_config = request.additional_params.get("ssh")

        if request.db_type == "mysql":
             connector = MySQLConnector(dsn, ssh_config=_adhoc_ssh_config)
        elif request.db_type == "mongodb":
             connector = MongoConnector(dsn, _namespace_name(request.namespace) or request.db_name, ssh_config=_adhoc_ssh_config)
        elif request.db_type == "sap_hana":
             connector = SAPHANAConnector(dsn)
        elif request.db_type == "sap_oracle":
             connector = OracleConnector(dsn, ssh_config=_adhoc_ssh_config)
        elif request.db_type == "sap_sybase":
             connector = SybaseConnector(dsn)
        else:
             connector = PostgresConnector(dsn, ssh_config=_adhoc_ssh_config)
        
        # Set processing_status to "processing" BEFORE background task starts
        try:
            db_instance_early = await get_db()
            service_early = DBCredentialsService(db_instance_early)
            existing_creds = await service_early.get_credentials(
                client_id=request.client_id, db_type=None, dataset_id=request.dataset_id
            )
            if existing_creds:
                existing_params = existing_creds.get("additional_params") or {}
                existing_params["processing_status"] = "processing"
                existing_params["total_tables"] = len(request.table_filter or [])
                existing_params["processed_tables"] = 0
                configured_store_in_local = require_store_in_local(existing_creds)
                await service_early.save_credentials(
                    client_id=request.client_id,
                    db_type=request.db_type,
                    db_host=request.db_host or "",
                    db_port=request.db_port or 0,
                    db_name=request.db_name or "",
                    db_username=request.db_username or "",
                    db_password=request.db_password or "",
                    db_url=request.db_url or "",
                    store_in_local=configured_store_in_local,
                    additional_params=existing_params,
                    dataset_id=existing_creds.get("dataset_id"),
                )
                logger.info(f"Set processing_status='processing' for client {request.client_id}")
        except Exception as e:
            logger.warning(f"Failed to set processing_status for client {request.client_id}: {e}")

        async def run_task():
            logger.info(f"Starting adhoc Explorer Agent for client: {request.client_id}")
            try:
                # Clean up existing data before generating new data
                _cleanup_existing_client_data(request.client_id, dataset_id=request.dataset_id)
                
                await connector.connect()
                
                # Enforce database size limit
                from util.db_size import get_database_size_bytes, format_size_mb, MAX_DB_SIZE_BYTES
                db_size_bytes = await get_database_size_bytes(
                    db_type=request.db_type,
                    db_host=request.db_host,
                    db_port=request.db_port,
                    db_name=request.db_name,
                    db_username=request.db_username,
                    db_password=request.db_password,
                    db_url=request.db_url,
                )
                if db_size_bytes > 0 and db_size_bytes > MAX_DB_SIZE_BYTES:
                    raise RuntimeError(
                        f"Database size ({format_size_mb(db_size_bytes)} MB) exceeds the maximum allowed "
                        f"({format_size_mb(MAX_DB_SIZE_BYTES)} MB). Please upgrade your plan."
                    )
                
                session_factory = connector.get_db()
                output_root = Path("xml_prompts/clients") / request.client_id
                
                # Get db instance for passing to ExplorerAgent
                db_instance = await get_db()
                
                # Adjust schema filter for MySQL and SAP HANA
                effective_schema_filter = request.schema_filter
                effective_table_filter = request.table_filter
                if request.db_type == "postgres" and _namespace_type(request.namespace) == "database":
                    effective_schema_filter = None if request.schema_filter == _namespace_name(request.namespace) else request.schema_filter
                elif request.db_type == "mysql" and (request.schema_filter == "public" or request.schema_filter is None):
                    effective_schema_filter = _namespace_name(request.namespace) or request.db_name
                elif request.db_type == "sap_hana":
                    effective_schema_filter = _namespace_name(request.namespace) or effective_schema_filter
                    if request.schema_filter == "public":
                        effective_schema_filter = None
                    # SAP HANA table filters may arrive as schema-qualified names
                    # (e.g. "SAPHANADB.VBAK"). Strip the schema prefix so downstream
                    # matching works correctly against bare TABLE_NAME values, and use
                    # the extracted schema as the effective_schema_filter when all
                    # selected tables share the same schema (avoids scanning 300K+ rows).
                    if request.table_filter:
                        bare_names = []
                        schemas_seen: set = set()
                        for entry in request.table_filter:
                            if "." in entry:
                                schema_part, table_part = entry.split(".", 1)
                                schemas_seen.add(schema_part.strip().upper())
                                bare_names.append(table_part.strip())
                            else:
                                bare_names.append(entry.strip())
                        effective_table_filter = bare_names
                        # If all selected tables are in one schema and no explicit
                        # schema_filter was provided, use that schema to narrow the query.
                        if len(schemas_seen) == 1 and not effective_schema_filter:
                            effective_schema_filter = next(iter(schemas_seen))
                            logger.info(
                                f"SAP HANA: inferred schema_filter='{effective_schema_filter}' "
                                f"from schema-qualified table_filter entries"
                            )
                elif request.namespace:
                    effective_schema_filter = _namespace_name(request.namespace) or request.schema_filter

                _adhoc_processed_count = 0

                async def _on_adhoc_table_done() -> None:
                    nonlocal _adhoc_processed_count
                    _adhoc_processed_count += 1
                    try:
                        _prog_db = await get_db()
                        _prog_svc = DBCredentialsService(_prog_db)
                        await _prog_svc.update_processing_progress(
                            request.client_id, _adhoc_processed_count, dataset_id=request.dataset_id
                        )
                    except Exception as _prog_err:
                        logger.warning(f"Failed to update processing progress for client {request.client_id}: {_prog_err}")

                agent = ExplorerAgent(
                    client_id=request.client_id,
                    session_factory=session_factory,
                    output_root=output_root,
                    schema_filter=effective_schema_filter,
                    table_prefix=request.table_prefix,
                    store_in_local=request.store_in_local,
                    db=db_instance,
                    db_type=request.db_type,
                    db_name=request.db_name,
                    db_username=request.db_username,
                    table_filter=effective_table_filter,
                    namespace=request.namespace,
                    on_table_done=_on_adhoc_table_done,
                    dataset_id=request.dataset_id,
                )

                await agent.run()
                logger.info(f"Adhoc Explorer Agent completed for client: {request.client_id}")
                
                # Persist store_in_local to db_credentials
                try:
                    service = DBCredentialsService(db_instance)
                    await service.update_store_in_local(
                        client_id=request.client_id,
                        db_type=request.db_type,
                        store_in_local=request.store_in_local,
                        dataset_id=request.dataset_id,
                    )
                    logger.info(f"Persisted store_in_local={request.store_in_local} to db_credentials for client {request.client_id}")
                except Exception as e:
                    logger.warning(f"Failed to persist store_in_local to db_credentials for client {request.client_id}: {e}")
                    # Don't fail the explorer task if this fails

                # Update processing_status to "complete"
                try:
                    creds_now = await service.get_credentials(
                        client_id=request.client_id, db_type=None, dataset_id=request.dataset_id
                    )
                    if creds_now:
                        params_now = creds_now.get("additional_params") or {}
                        params_now["processing_status"] = "complete"
                        configured_store_in_local = require_store_in_local(creds_now)
                        await service.save_credentials(
                            client_id=request.client_id,
                            db_type=request.db_type,
                            db_host=request.db_host or "",
                            db_port=request.db_port or 0,
                            db_name=request.db_name or "",
                            db_username=request.db_username or "",
                            db_password=request.db_password or "",
                            db_url=request.db_url or "",
                            store_in_local=configured_store_in_local,
                            additional_params=params_now,
                            dataset_id=creds_now.get("dataset_id"),
                        )
                        logger.info(f"Set processing_status='complete' for client {request.client_id}")
                        # Notify admin that dataset configuration is complete and ready for chat
                        try:
                            from notifications.notification_service import create_notification
                            from notifications.notification_model import Notification
                            _notif_uid = str(admin_user.get("user_id") or admin_user.get("_id") or "")
                            _dataset_name = creds_now.get("dataset_name") or request.db_name or request.db_type or "Dataset"
                            if _notif_uid and request.client_id:
                                await create_notification(Notification(
                                    client_id=request.client_id,
                                    user_id=_notif_uid,
                                    type="db_config_completed",
                                    title="Dataset Ready",
                                    message=f'"{_dataset_name}" has been configured and is ready for analysis.',
                                    metadata={"dataset_id": str(request.dataset_id or ""), "db_type": request.db_type},
                                    target_role="admin",
                                ))
                        except Exception as _ne:
                            logger.warning(f"Failed to send db_config_completed notification: {_ne}")
                except Exception as e:
                    logger.warning(f"Failed to set processing_status='complete' for client {request.client_id}: {e}")
                    
            except Exception as e:
                logger.error(f"Adhoc Explorer Agent failed for client {request.client_id}: {str(e)}", exc_info=True)
                # Reset processing_status to "error" so frontend stops polling
                try:
                    db_err = await get_db()
                    svc_err = DBCredentialsService(db_err)
                    creds_err = await svc_err.get_credentials(
                        client_id=request.client_id, db_type=None, dataset_id=request.dataset_id
                    )
                    if creds_err:
                        params_err = creds_err.get("additional_params") or {}
                        params_err["processing_status"] = "error"
                        params_err["processing_error"] = str(e)
                        configured_store_in_local = require_store_in_local(creds_err)
                        await svc_err.save_credentials(
                            client_id=request.client_id,
                            db_type=request.db_type,
                            db_host=request.db_host or "",
                            db_port=request.db_port or 0,
                            db_name=request.db_name or "",
                            db_username=request.db_username or "",
                            db_password=request.db_password or "",
                            db_url=request.db_url or "",
                            store_in_local=configured_store_in_local,
                            additional_params=params_err,
                            dataset_id=creds_err.get("dataset_id"),
                        )
                except Exception as cleanup_err:
                    logger.warning(f"Failed to set processing_status='error' for client {request.client_id}: {cleanup_err}")
            finally:
                await connector.disconnect()
        
        background_tasks.add_task(run_task)
        
        return {
            "status": "accepted",
            "message": f"Explorer Agent started for client {request.client_id}",
            "details": {
                "client_id": request.client_id,
                "output_path": f"xml_prompts/clients/{request.client_id}",
                "export_datasets": request.store_in_local
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start adhoc Explorer Agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _get_base_sap_metadata(client_id: str) -> dict:
    """
    Read metadata directly from base_sap (all tables) for table selection.
    Used by /list-tables endpoint to show all available tables.
    """
    base_sap_dir = Path("xml_prompts/base_sap") / "data_sources"
    base_sap_meta_dir = base_sap_dir / "meta_information"
    base_sap_desc_dir = base_sap_dir / "data_descriptions"
    
    if not base_sap_desc_dir.exists() and not base_sap_meta_dir.exists():
        raise HTTPException(status_code=404, detail="Base SAP metadata not found.")
    
    # Parse table introductions from base_sap
    intros_file = base_sap_meta_dir / "table_introductions.xml"
    intros_map = {}
    if intros_file.exists():
        try:
            intros_tree = parse(intros_file)
            intros_root = intros_tree.getroot()
            intros_map = {
                elem.get("table_name"): (elem.text or "").strip()
                for elem in intros_root.findall(".//table_introduction")
            }
        except ET.ParseError as e:
            logger.warning(f"Failed to parse base_sap table_introductions.xml: {e}")
    
    # Parse column descriptions from base_sap
    column_desc_map: dict[str, dict[str, str]] = {}
    column_type_map: dict[str, dict[str, str]] = {}
    if base_sap_desc_dir.exists():
        for file_path in glob(str(base_sap_desc_dir / "*_description.xml")):
            table_name = Path(file_path).name.replace("_description.xml", "")
            try:
                tree = parse(file_path)
                root = tree.getroot()
                table_descs = {}
                table_types = {}
                for col in root.findall(".//column"):
                    col_name = col.get("name")
                    data_type = col.get("data_type", "")
                    desc_node = col.find("description")
                    if col_name:
                        if desc_node is not None:
                            table_descs[col_name] = (desc_node.text or "").strip()
                        if data_type:
                            table_types[col_name] = data_type
                column_desc_map[table_name] = table_descs
                column_type_map[table_name] = table_types
            except ET.ParseError as e:
                logger.warning(f"Failed parsing {file_path}: {e}")
    
    # Build tables payload (same logic as get_metadata)
    tables_dict = {}
    for table_name, table_types in column_type_map.items():
        table_descs = column_desc_map.get(table_name, {})
        columns_payload = []
        for col_name, data_type in table_types.items():
            data_type_lower = data_type.lower()
            sdtype = "text"
            if "int" in data_type_lower or "integer" in data_type_lower or "numeric" in data_type_lower or "decimal" in data_type_lower:
                sdtype = "numerical"
            elif "bool" in data_type_lower:
                sdtype = "categorical"
            elif "date" in data_type_lower or "time" in data_type_lower or "timestamp" in data_type_lower:
                sdtype = "datetime"
            elif "double" in data_type_lower or "float" in data_type_lower or "real" in data_type_lower:
                sdtype = "numerical"
            elif "char" in data_type_lower or "varchar" in data_type_lower or "text" in data_type_lower:
                sdtype = "text"
            
            description = table_descs.get(col_name, "")
            columns_payload.append({
                "name": col_name,
                "sdtype": sdtype,
                "description": description,
            })
        tables_dict[table_name] = {
            "name": table_name,
            "primary_key": "id",
            "introduction": intros_map.get(table_name),
            "column_count": len(columns_payload),
            "columns": columns_payload,
            "row_count": None,  # Row count not available in base_sap metadata
        }
    
    # Include tables from table_introductions that don't have description files
    for table_name, introduction in intros_map.items():
        if table_name not in tables_dict:
            tables_dict[table_name] = {
                "name": table_name,
                "primary_key": "id",
                "introduction": introduction,
                "column_count": 0,
                "columns": [],
                "row_count": None,  # Row count not available in base_sap metadata
            }
    
    tables_payload = list(tables_dict.values())
    
    # Get explorer limits
    limits = await get_explorer_limits(client_id)
    total_loaded = len(tables_payload)
    max_cols = max((t.get("column_count", len(t.get("columns", []))) for t in tables_payload), default=0)
    explorer_limits = {
        "table_limit": limits.get("table_limit"),
        "column_limit": limits.get("column_limit"),
        "total_tables_available": total_loaded,
        "total_tables_loaded": total_loaded,
        "columns_per_table_loaded": max_cols,
        "is_table_limit_reached": False,
        "is_column_limit_reached": False,
        "upgrade_required": False,
    }
    
    return {
        "client_id": client_id,
        "total_tables": len(tables_payload),
        "tables": tables_payload,
        "explorer_limits": explorer_limits,
    }


@router.get("/metadata")
async def get_metadata(
    client_id: str = Query(..., description="Client identifier (matches xml_prompts/clients/{client_id})"),
    db_type: Optional[str] = Query(None, description="Optional db_type hint (e.g., 'sap_oracle', 'sap_sybase') to use base_sap metadata"),
    dataset_id: Optional[str] = Query(None, description="Dataset scope for metadata XML paths"),
    admin_user: Dict = Depends(require_admin),
):
    """Return explorer metadata parsed from generated XML files.

    Primary sources:
    - xml_prompts/clients/{client_id}/data_sources/meta_information/table_introductions.xml (table descriptions)
    - xml_prompts/clients/{client_id}/data_sources/data_descriptions/*_description.xml (column descriptions)

    For sap_oracle and sap_sybase clients, uses base_sap metadata.
    """
    if admin_user.get("role") != "super_admin" and client_id != admin_user.get("client_id"):
        raise HTTPException(status_code=403, detail="Access denied")
    base_dir = resolve_xml_data_sources_dir(client_id, dataset_id)
    meta_dir = base_dir / "meta_information"
    desc_dir = base_dir / "data_descriptions"
    base_sap_dir = Path("xml_prompts/base_sap") / "data_sources"
    base_sap_meta_dir = base_sap_dir / "meta_information"
    base_sap_desc_dir = base_sap_dir / "data_descriptions"

    # Determine if we should use base_sap (for sap_oracle and sap_sybase clients)
    use_base_sap = False
    determined_db_type = db_type  # Use provided db_type if available
    
    # If db_type was provided as parameter, use it directly
    if determined_db_type in ("sap_oracle", "sap_sybase"):
        use_base_sap = True
    else:
        # Try to get db_type from credentials if not provided
        try:
            db = await get_db()
            service = DBCredentialsService(db)
            credentials = await service.get_credentials(
                client_id=client_id, db_type=None, decrypt_password=False, dataset_id=dataset_id
            )
            if credentials:
                determined_db_type = credentials.get("db_type")
                if determined_db_type in ("sap_oracle", "sap_sybase"):
                    use_base_sap = True
        except Exception as e:
            logger.warning(f"Failed to determine db_type for get_metadata: {e}")

    # For sap_oracle and sap_sybase, check base_sap first; for others, check client-specific first
    if use_base_sap:
        if not base_sap_desc_dir.exists() and not base_sap_meta_dir.exists():
            raise HTTPException(status_code=404, detail="Base SAP metadata not found. Run explorer first.")
    elif not desc_dir.exists() and not meta_dir.exists():
        raise HTTPException(status_code=404, detail="Metadata not found. Run explorer first.")

    # Parse table introductions
    # For sap_oracle and sap_sybase, check client directory first (has selected tables),
    # then fallback to base_sap (has all tables) if client directory doesn't exist
    # For other clients, use client-specific only
    if use_base_sap:
        # Check client directory first (has filtered/selected tables from explorer run)
        # Only fallback to base_sap if client directory doesn't exist (initial state)
        client_intros_file = meta_dir / "table_introductions.xml"
        if client_intros_file.exists():
            intros_file = client_intros_file
            logger.info(f"Using client-specific table_introductions.xml for SAP client {client_id}")
        else:
            intros_file = base_sap_meta_dir / "table_introductions.xml" if base_sap_meta_dir.exists() else None
            logger.info(f"Using base_sap table_introductions.xml for SAP client {client_id} (client directory not found)")
    else:
        intros_file = meta_dir / "table_introductions.xml" if meta_dir.exists() else None
        # DO NOT fallback to base_sap for non-SAP clients
    intros_map = {}
    if intros_file and intros_file.exists():
        try:
            intros_tree = parse(intros_file)
            intros_root = intros_tree.getroot()
            intros_map = {
                elem.get("table_name"): (elem.text or "").strip()
                for elem in intros_root.findall(".//table_introduction")
            }
        except ET.ParseError as e:
            logger.warning(f"Failed to parse table_introductions.xml: {e}")

    # Parse column descriptions per table from data_descriptions directory
    column_desc_map: dict[str, dict[str, str]] = {}
    column_type_map: dict[str, dict[str, str]] = {}
    # For SAP: check client directory first (has selected tables), then fallback to base_sap
    if use_base_sap:
        # Prefer client directory (has filtered/selected tables from explorer run)
        # Only fallback to base_sap if client directory doesn't exist (initial state)
        if desc_dir.exists():
            desc_search_dir = desc_dir
            logger.info(f"Using client-specific data_descriptions for SAP client {client_id}")
        elif base_sap_desc_dir.exists():
            desc_search_dir = base_sap_desc_dir
            logger.info(f"Using base_sap data_descriptions for SAP client {client_id} (client directory not found)")
        else:
            desc_search_dir = None
    else:
        desc_search_dir = desc_dir if desc_dir.exists() else None
    if desc_search_dir and desc_search_dir.exists():
        for file_path in glob(str(desc_search_dir / "*_description.xml")):
            table_name = Path(file_path).name.replace("_description.xml", "")
            try:
                tree = parse(file_path)
                root = tree.getroot()
                table_descs = {}
                table_types = {}
                for col in root.findall(".//column"):
                    col_name = col.get("name")
                    data_type = col.get("data_type", "")
                    desc_node = col.find("description")
                    if col_name:
                        if desc_node is not None:
                            table_descs[col_name] = (desc_node.text or "").strip()
                        if data_type:
                            table_types[col_name] = data_type
                column_desc_map[table_name] = table_descs
                column_type_map[table_name] = table_types
            except ET.ParseError as e:
                logger.warning(f"Failed parsing {file_path}: {e}")

    # Build tables payload from description files (primary source)
    tables_dict = {}
    for table_name, table_descs in column_desc_map.items():
        table_types = column_type_map.get(table_name, {})
        columns_payload = []
        for col_name, description in table_descs.items():
            # Map data_type to sdtype
            data_type = table_types.get(col_name, "").lower()
            sdtype = "text"  # Default
            if "int" in data_type or "integer" in data_type or "numeric" in data_type or "decimal" in data_type:
                sdtype = "numerical"
            elif "bool" in data_type:
                sdtype = "categorical"
            elif "date" in data_type or "time" in data_type or "timestamp" in data_type:
                sdtype = "datetime"
            elif "double" in data_type or "float" in data_type or "real" in data_type:
                sdtype = "numerical"
            elif "char" in data_type or "varchar" in data_type or "text" in data_type:
                sdtype = "text"
            
            columns_payload.append({
                "name": col_name,
                "sdtype": sdtype,
                "description": description,
            })
        tables_dict[table_name] = {
            "name": table_name,
            "primary_key": "id",  # Default
            "introduction": intros_map.get(table_name),
            "column_count": len(columns_payload),
            "columns": columns_payload,
        }

    # Also include tables from table_introductions that don't have description files
    for table_name, introduction in intros_map.items():
        if table_name not in tables_dict:
            tables_dict[table_name] = {
                "name": table_name,
                "primary_key": "id",
                "introduction": introduction,
                "column_count": 0,
                "columns": [],
            }

    # Schema and primary_key are now set from defaults or data_description files
    # All info comes from description files
    # Default values are used: primary_key="id", schema from db_type or default

    tables_payload = list(tables_dict.values())

    if not tables_payload:
        raise HTTPException(status_code=404, detail="No metadata found. Run explorer first to generate table descriptions.")

    # Include explorer_limits: from persisted file or best-effort from plan + current counts
    client_meta_dir = base_dir / "meta_information"
    explorer_limits_path = client_meta_dir / "explorer_limits.json"
    explorer_limits = None
    if explorer_limits_path.exists():
        try:
            explorer_limits = await asyncio.to_thread(
                lambda: json.loads(explorer_limits_path.read_text("utf-8"))
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read explorer_limits.json: {e}")
    if explorer_limits is None:
        limits = await get_explorer_limits(client_id)
        total_loaded = len(tables_payload)
        max_cols = max((t.get("column_count", len(t.get("columns", []))) for t in tables_payload), default=0)
        explorer_limits = {
            "table_limit": limits.get("table_limit"),
            "column_limit": limits.get("column_limit"),
            "total_tables_available": total_loaded,
            "total_tables_loaded": total_loaded,
            "columns_per_table_loaded": max_cols,
            "is_table_limit_reached": False,
            "is_column_limit_reached": False,
            "upgrade_required": False,
        }

    return {
        "client_id": client_id,
        "total_tables": len(tables_payload),
        "tables": tables_payload,
        "explorer_limits": explorer_limits,
    }



@router.get("/sample-questions")
async def get_sample_questions(
    client_id: str = Query(..., description="Client identifier"),
    regenerate: bool = Query(False, description="Force regenerate questions"),
    db = Depends(get_db),
    admin_user: Dict = Depends(require_admin),
):
    """
    Get 30 contextual sample questions for a client.
    
    Questions are cached in XML file at:
    xml_prompts/clients/{CLIENT_ID}/data_sources/suggested_questions.xml
    
    Uses LLM to create relevant questions from schema, table introductions,
    and column descriptions. Generates 30 questions optimized for UI display
    (max 60 characters each to prevent UI breaking).
    
    Args:
        client_id: Client identifier
        regenerate: If True, regenerate questions even if cached
    
    Returns:
        {
            "client_id": str,
            "questions": List[str],  # All 30 questions
            "generated_at": str,
            "from_cache": bool
        }
    """
    if admin_user.get("role") != "super_admin" and client_id != admin_user.get("client_id"):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        from explorer.question_generator import QuestionGenerator
        
        generator = QuestionGenerator(client_id, db=db)
        
        # Try to load from cache first
        cached_questions = None
        if not regenerate:
            cached_questions = generator.load_questions_from_xml()
        
        if cached_questions and len(cached_questions) == 30:
            # Return cached questions
            return {
                "client_id": client_id,
                "questions": cached_questions,
                "generated_at": utcnow().isoformat(),
                "from_cache": True
            }
        
        # Generate new questions (30 total)
        questions = await generator.generate_questions(count=30)
        
        return {
            "client_id": client_id,
            "questions": questions,
            "generated_at": utcnow().isoformat(),
            "from_cache": False
        }
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Explorer metadata not found for client {client_id}. Run explorer first."
        )
    except Exception as e:
        logger.error(f"Failed to generate sample questions: {e}", exc_info=True)
        # Return fallback questions on error
        fallback_questions = [
            "What's the total count of records?",
            "Show me the top 10 by value",
            "What's the average across all entries?",
            "Show me the distribution by category",
            "Which items have the highest count?",
            "What's the trend over time?",
            "Show me a breakdown by group",
            "What's the total value?",
            "Which categories are most common?",
            "Show me recent activity summary",
            "What are the key metrics?",
            "Compare performance across segments",
            "Show monthly trends",
            "What's the growth rate?",
            "Identify top performers",
            "Show bottom 10 entries",
            "What's the distribution?",
            "Analyze by region",
            "Show year-over-year comparison",
            "What are the outliers?",
            "Calculate total revenue",
            "Show customer breakdown",
            "What's the conversion rate?",
            "Analyze seasonal patterns",
            "Show correlation analysis",
            "What's the retention rate?",
            "Compare quarter performance",
            "Show engagement metrics",
            "What's the churn rate?",
            "Analyze product performance"
        ]
        return {
            "client_id": client_id,
            "questions": fallback_questions,
            "generated_at": utcnow().isoformat(),
            "fallback": True,
            "from_cache": False
        }


class SuggestedQuestionRequest(BaseModel):
    """Request model for adding/updating suggested questions"""
    question: str
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class AddSuggestedQuestionRequest(BaseModel):
    """Request model for adding a new suggested question"""
    question: str
    position: Optional[int] = None
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()
    
    @validator('position')
    def validate_position(cls, v):
        if v is not None and (v < 1):
            raise ValueError('Position must be a positive integer')
        return v


@router.get("/suggested-questions")
async def get_suggested_questions(
    dataset_id: Optional[str] = Query(None, description="Dataset scope for suggested questions"),
    admin_user: Dict = Depends(require_admin),
):
    """
    Get all suggested questions for the admin's client.
    
    **Permissions**: Admin only (client-scoped)
    
    **Returns**: List of questions with IDs, metadata, and count
    """
    try:
        admin_client_id = admin_user.get("client_id")
        admin_email = admin_user.get("email", "unknown")
        
        if not admin_client_id:
            raise HTTPException(
                status_code=403,
                detail="Admin user missing client_id. Please contact system administrator."
            )
        
        logger.info(f"Admin {admin_email} loading suggested questions for client {admin_client_id}")
        
        from explorer.question_generator import QuestionGenerator
        generator = QuestionGenerator(admin_client_id, dataset_id=dataset_id)
        
        # Load questions with IDs
        questions_with_ids = generator.load_questions_with_ids()
        metadata = generator.get_xml_metadata()
        
        if questions_with_ids is None:
            # No questions file exists yet
            return {
                "client_id": admin_client_id,
                "dataset_id": dataset_id,
                "questions": [],
                "generated_at": None,
                "count": 0
            }
        
        return {
            "client_id": admin_client_id,
            "dataset_id": dataset_id,
            "questions": questions_with_ids,
            "generated_at": metadata.get("generated_at") if metadata else None,
            "count": len(questions_with_ids)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get suggested questions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load suggested questions: {str(e)}")


@router.post("/suggested-questions")
async def add_suggested_question(
    request: AddSuggestedQuestionRequest,
    dataset_id: Optional[str] = Query(None, description="Dataset scope for suggested questions"),
    admin_user: Dict = Depends(require_admin),
):
    """
    Add a new suggested question for the admin's client.
    
    **Permissions**: Admin only (client-scoped)
    
    **Request Body**: AddSuggestedQuestionRequest with question text and optional position
    
    **Returns**: Updated list of questions
    """
    try:
        admin_client_id = admin_user.get("client_id")
        admin_email = admin_user.get("email", "unknown")
        
        if not admin_client_id:
            raise HTTPException(
                status_code=403,
                detail="Admin user missing client_id. Please contact system administrator."
            )
        
        logger.info(
            f"Admin {admin_email} adding suggested question for client {admin_client_id} | "
            f"position={request.position}"
        )
        
        from explorer.question_generator import QuestionGenerator
        generator = QuestionGenerator(admin_client_id, dataset_id=dataset_id)
        
        # Load existing questions
        questions_with_ids = generator.load_questions_with_ids() or []
        questions_list = [q["text"] for q in questions_with_ids]
        
        # Add new question at position or append
        if request.position is not None and 1 <= request.position <= len(questions_list):
            questions_list.insert(request.position - 1, request.question)
        else:
            questions_list.append(request.question)
        
        # Save to XML (will create backup in the endpoint)
        questions_file = generator.questions_file
        
        # Create backup before modification
        backup_file = questions_file.with_suffix(f".xml.backup.{utcnow().strftime('%Y%m%d_%H%M%S')}")
        if questions_file.exists():
            try:
                shutil.copy2(questions_file, backup_file)
                logger.info(f"Created backup: {backup_file}")
            except Exception as backup_err:
                logger.error(f"Failed to create backup: {backup_err}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create backup before modification. Operation aborted for safety."
                )
        
        try:
            # Save updated questions
            generator._save_questions_to_xml(questions_list)
            
            # Build updated questions list with re-sequenced IDs (no need to reload from disk)
            updated_questions = [
                {"id": idx + 1, "text": text}
                for idx, text in enumerate(questions_list)
            ]
            
            # Audit log (non-blocking - run in background)
            import asyncio
            asyncio.create_task(
                audit_logger.log_event(
                    event_type=AuditEventType.METADATA_TABLE_INTRODUCTION_UPDATED,
                    severity=AuditSeverity.INFO,
                    user_id=admin_email,
                    client_id=admin_client_id,
                    details={
                        "action": "suggested_question_added",
                        "question": request.question[:200],
                        "position": request.position,
                        "total_questions": len(updated_questions),
                        "backup_file": str(backup_file) if questions_file.exists() else None
                    }
                )
            )
            
            logger.info(f"Successfully added suggested question for client {admin_client_id}")
            
            return {
                "success": True,
                "message": "Question added successfully",
                "client_id": admin_client_id,
                "dataset_id": dataset_id,
                "questions": updated_questions,
                "count": len(updated_questions)
            }
        except Exception as write_err:
            # Restore from backup if operation failed
            if backup_file.exists() and questions_file.exists():
                try:
                    backup_file.replace(questions_file)
                    logger.info(f"Restored from backup after error")
                except Exception as restore_err:
                    logger.error(f"Failed to restore from backup: {restore_err}")
            raise write_err
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add suggested question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add question: {str(e)}")


@router.put("/suggested-questions/{question_id}")
async def update_suggested_question(
    question_id: int,
    request: SuggestedQuestionRequest,
    dataset_id: Optional[str] = Query(None, description="Dataset scope for suggested questions"),
    admin_user: Dict = Depends(require_admin),
):
    """
    Update an existing suggested question for the admin's client.
    
    **Permissions**: Admin only (client-scoped)
    
    **Path Parameters**: question_id - ID of the question to update
    
    **Request Body**: SuggestedQuestionRequest with updated question text
    
    **Returns**: Updated list of questions
    """
    try:
        admin_client_id = admin_user.get("client_id")
        admin_email = admin_user.get("email", "unknown")
        
        if not admin_client_id:
            raise HTTPException(
                status_code=403,
                detail="Admin user missing client_id. Please contact system administrator."
            )
        
        logger.info(
            f"Admin {admin_email} updating suggested question {question_id} for client {admin_client_id}"
        )
        
        from explorer.question_generator import QuestionGenerator
        generator = QuestionGenerator(admin_client_id, dataset_id=dataset_id)
        
        # Load existing questions
        questions_with_ids = generator.load_questions_with_ids()
        if not questions_with_ids:
            raise HTTPException(
                status_code=404,
                detail="No suggested questions found. Add questions first."
            )
        
        # Find question by ID
        question_found = False
        old_question_text = None
        questions_list = []
        for q in questions_with_ids:
            if q["id"] == question_id:
                questions_list.append(request.question)
                old_question_text = q["text"]
                question_found = True
            else:
                questions_list.append(q["text"])
        
        if not question_found:
            raise HTTPException(
                status_code=404,
                detail=f"Question with ID {question_id} not found"
            )
        
        # Create backup before modification
        questions_file = generator.questions_file
        backup_file = questions_file.with_suffix(f".xml.backup.{utcnow().strftime('%Y%m%d_%H%M%S')}")
        if questions_file.exists():
            try:
                shutil.copy2(questions_file, backup_file)
                logger.info(f"Created backup: {backup_file}")
            except Exception as backup_err:
                logger.error(f"Failed to create backup: {backup_err}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create backup before modification. Operation aborted for safety."
                )
        
        try:
            # Save updated questions
            generator._save_questions_to_xml(questions_list)
            
            # Build updated questions list with re-sequenced IDs (no need to reload from disk)
            updated_questions = [
                {"id": idx + 1, "text": text}
                for idx, text in enumerate(questions_list)
            ]
            
            # Audit log (non-blocking - run in background)
            import asyncio
            asyncio.create_task(
                audit_logger.log_event(
                    event_type=AuditEventType.METADATA_TABLE_INTRODUCTION_UPDATED,
                    severity=AuditSeverity.INFO,
                    user_id=admin_email,
                    client_id=admin_client_id,
                    details={
                        "action": "suggested_question_updated",
                        "question_id": question_id,
                        "old_question": old_question_text[:200] if old_question_text else None,
                        "new_question": request.question[:200],
                        "backup_file": str(backup_file)
                    }
                )
            )
            
            logger.info(f"Successfully updated suggested question {question_id} for client {admin_client_id}")
            
            return {
                "success": True,
                "message": f"Question {question_id} updated successfully",
                "client_id": admin_client_id,
                "dataset_id": dataset_id,
                "questions": updated_questions,
                "count": len(updated_questions)
            }
        except Exception as write_err:
            # Restore from backup if operation failed
            if backup_file.exists() and questions_file.exists():
                try:
                    backup_file.replace(questions_file)
                    logger.info(f"Restored from backup after error")
                except Exception as restore_err:
                    logger.error(f"Failed to restore from backup: {restore_err}")
            raise write_err
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update suggested question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update question: {str(e)}")


@router.delete("/suggested-questions/{question_id}")
async def delete_suggested_question(
    question_id: int,
    dataset_id: Optional[str] = Query(None, description="Dataset scope for suggested questions"),
    admin_user: Dict = Depends(require_admin),
):
    """
    Delete a suggested question for the admin's client.
    
    **Permissions**: Admin only (client-scoped)
    
    **Path Parameters**: question_id - ID of the question to delete
    
    **Returns**: Updated list of questions (with re-sequenced IDs)
    """
    try:
        admin_client_id = admin_user.get("client_id")
        admin_email = admin_user.get("email", "unknown")
        
        if not admin_client_id:
            raise HTTPException(
                status_code=403,
                detail="Admin user missing client_id. Please contact system administrator."
            )
        
        logger.info(
            f"Admin {admin_email} deleting suggested question {question_id} (type: {type(question_id)}) for client {admin_client_id}"
        )
        
        from explorer.question_generator import QuestionGenerator
        generator = QuestionGenerator(admin_client_id, dataset_id=dataset_id)
        
        # Load existing questions
        questions_with_ids = generator.load_questions_with_ids()
        if not questions_with_ids:
            logger.warning(f"No questions found for client {admin_client_id}")
            raise HTTPException(
                status_code=404,
                detail="No suggested questions found."
            )
        
        logger.debug(f"Loaded {len(questions_with_ids)} questions. Looking for ID: {question_id}")
        logger.debug(f"Available question IDs: {[q['id'] for q in questions_with_ids]}")
        
        # Ensure question_id is an integer for comparison
        try:
            question_id_int = int(question_id)
        except (ValueError, TypeError):
            logger.error(f"Invalid question_id type: {type(question_id)}, value: {question_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid question ID: {question_id}. Must be an integer."
            )
        
        # Find and remove question by ID
        question_found = False
        deleted_question_text = None
        questions_list = []
        for q in questions_with_ids:
            # Compare as integers to handle any type mismatches
            if int(q["id"]) == question_id_int:
                deleted_question_text = q["text"]
                question_found = True
                logger.info(f"Found question to delete: ID {question_id_int}, text: {deleted_question_text[:50]}...")
                # Skip this question (don't add to list)
            else:
                questions_list.append(q["text"])
        
        if not question_found:
            logger.warning(f"Question with ID {question_id_int} not found. Available IDs: {[q['id'] for q in questions_with_ids]}")
            raise HTTPException(
                status_code=404,
                detail=f"Question with ID {question_id} not found"
            )
        
        # Create backup before modification
        questions_file = generator.questions_file
        backup_file = questions_file.with_suffix(f".xml.backup.{utcnow().strftime('%Y%m%d_%H%M%S')}")
        if questions_file.exists():
            try:
                shutil.copy2(questions_file, backup_file)
                logger.info(f"Created backup: {backup_file}")
            except Exception as backup_err:
                logger.error(f"Failed to create backup: {backup_err}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create backup before modification. Operation aborted for safety."
                )
        
        try:
            # Save updated questions (IDs will be re-sequenced automatically by _save_questions_to_xml)
            generator._save_questions_to_xml(questions_list)
            
            # Build updated questions list with re-sequenced IDs (no need to reload from disk)
            updated_questions = [
                {"id": idx + 1, "text": text}
                for idx, text in enumerate(questions_list)
            ]
            
            # Audit log (non-blocking - run in background)
            import asyncio
            asyncio.create_task(
                audit_logger.log_event(
                    event_type=AuditEventType.METADATA_TABLE_INTRODUCTION_UPDATED,
                    severity=AuditSeverity.INFO,
                    user_id=admin_email,
                    client_id=admin_client_id,
                    details={
                        "action": "suggested_question_deleted",
                        "question_id": question_id,
                        "deleted_question": deleted_question_text[:200] if deleted_question_text else None,
                        "remaining_count": len(updated_questions),
                        "backup_file": str(backup_file)
                    }
                )
            )
            
            logger.info(f"Successfully deleted suggested question {question_id} for client {admin_client_id}")
            
            return {
                "success": True,
                "message": f"Question {question_id} deleted successfully",
                "client_id": admin_client_id,
                "dataset_id": dataset_id,
                "questions": updated_questions,
                "count": len(updated_questions)
            }
        except Exception as write_err:
            # Restore from backup if operation failed
            if backup_file.exists() and questions_file.exists():
                try:
                    backup_file.replace(questions_file)
                    logger.info(f"Restored from backup after error")
                except Exception as restore_err:
                    logger.error(f"Failed to restore from backup: {restore_err}")
            raise write_err
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete suggested question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete question: {str(e)}")


@router.post("/suggested-questions/auto-generate")
async def auto_generate_suggested_questions(
    dataset_id: Optional[str] = Query(None, description="Dataset scope for suggested questions"),
    admin_user: Dict = Depends(require_admin),
    db=Depends(get_db),
):
    """
    Auto-generate suggested questions using LLM based on the client's dataset schema.

    **Permissions**: Admin only (client-scoped)

    Generates 30 contextual questions from the dataset's table introductions and
    column descriptions, then saves them to the XML cache (overwriting any existing
    manually-curated questions after creating a timestamped backup).

    **Returns**: Updated list of questions with IDs and generation timestamp
    """
    try:
        admin_client_id = admin_user.get("client_id")
        admin_email = admin_user.get("email", "unknown")

        if not admin_client_id:
            raise HTTPException(
                status_code=403,
                detail="Admin user missing client_id. Please contact system administrator.",
            )

        logger.info(
            f"Admin {admin_email} auto-generating suggested questions for client "
            f"{admin_client_id} | dataset_id={dataset_id}"
        )

        from explorer.question_generator import QuestionGenerator

        generator = QuestionGenerator(admin_client_id, db=db, dataset_id=dataset_id)

        # Back up existing questions file before overwriting
        questions_file = generator.questions_file
        if questions_file.exists():
            backup_file = questions_file.with_suffix(
                f".xml.backup.{utcnow().strftime('%Y%m%d_%H%M%S')}"
            )
            try:
                shutil.copy2(questions_file, backup_file)
                logger.info(f"Created backup before auto-generate: {backup_file}")
            except Exception as backup_err:
                logger.error(f"Failed to create backup before auto-generate: {backup_err}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create backup before auto-generation. Operation aborted for safety.",
                )

        # Generate fresh questions (always regenerate)
        questions = await generator.generate_questions(count=30)

        # Reload with IDs for the response
        questions_with_ids = generator.load_questions_with_ids() or [
            {"id": idx + 1, "text": q} for idx, q in enumerate(questions)
        ]
        metadata = generator.get_xml_metadata()

        asyncio.create_task(
            audit_logger.log_event(
                event_type=AuditEventType.METADATA_TABLE_INTRODUCTION_UPDATED,
                severity=AuditSeverity.INFO,
                user_id=admin_email,
                client_id=admin_client_id,
                details={
                    "action": "suggested_questions_auto_generated",
                    "dataset_id": dataset_id,
                    "count": len(questions_with_ids),
                },
            )
        )

        return {
            "success": True,
            "client_id": admin_client_id,
            "dataset_id": dataset_id,
            "questions": questions_with_ids,
            "generated_at": metadata.get("generated_at") if metadata else utcnow().isoformat(),
            "count": len(questions_with_ids),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to auto-generate suggested questions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to auto-generate questions: {str(e)}"
        )


class UpdateIntroductionRequest(BaseModel):
    introduction: str

    @validator('introduction')
    def validate_introduction(cls, v):
        if not v or not v.strip():
            raise ValueError('Introduction cannot be empty')
        if len(v.strip()) > 10000:
            raise ValueError('Introduction is too long (max 10000 characters)')
        return v.strip()


class UpdateColumnDescriptionRequest(BaseModel):
    description: str
    
    @validator('description')
    def validate_description(cls, v):
        if not v or not v.strip():
            raise ValueError('Description cannot be empty')
        if len(v.strip()) > 5000:
            raise ValueError('Description is too long (max 5000 characters)')
        return v.strip()


class UpdatePrimaryKeyRequest(BaseModel):
    primary_key: str
    
    @validator('primary_key')
    def validate_primary_key(cls, v):
        if not v or not v.strip():
            raise ValueError('Primary key cannot be empty')
        if len(v.strip()) > 255:
            raise ValueError('Primary key name is too long (max 255 characters)')
        return v.strip()


@router.put("/metadata/{client_id}/table/{table_name}/introduction")
async def update_table_introduction(
    client_id: str,
    table_name: str,
    request: UpdateIntroductionRequest,
    admin_user: Dict = Depends(require_admin),
):
    """
    Update table introduction in table_introductions.xml
    
    **Security**: 
    - Only admins can update metadata
    - Validates client_id matches admin's client
    - Creates backup before modification
    - Atomic write operation (writes to temp file first)
    
    **Location**: xml_prompts/clients/{client_id}/data_sources/meta_information/table_introductions.xml
    """
    try:
        # Validate client access - admin can only modify their own client's data
        admin_client_id = admin_user.get("client_id")
        if admin_client_id and admin_client_id != client_id:
            logger.warning(
                f"Admin {admin_user.get('email')} attempted to modify client {client_id} "
                f"(their client: {admin_client_id})"
            )
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. You can only modify metadata for your own client ({admin_client_id})"
            )
        
        intros_file = Path("xml_prompts/clients") / client_id / "data_sources" / "meta_information" / "table_introductions.xml"
        
        if not intros_file.exists():
            raise HTTPException(
                status_code=404, 
                detail="table_introductions.xml not found. Run explorer first to generate metadata."
            )
        
        # Create backup before modification (critical for sensitive data)
        backup_file = intros_file.with_suffix(f".xml.backup.{utcnow().strftime('%Y%m%d_%H%M%S')}")
        try:
            shutil.copy2(intros_file, backup_file)
            logger.info(f"Created backup: {backup_file}")
        except Exception as backup_err:
            logger.error(f"Failed to create backup: {backup_err}")
            raise HTTPException(
                status_code=500,
                detail="Failed to create backup before modification. Operation aborted for safety."
            )
        
        try:
            # Parse existing XML
            tree = parse(intros_file)
            root = tree.getroot()
            
            # Validate XML structure
            if root is None:
                raise HTTPException(status_code=500, detail="Invalid XML: root element is None")
            
            # Find or create the table_introduction element
            intro_elem = None
            old_introduction = None
            for elem in root.findall(".//table_introduction"):
                if elem.get("table_name") == table_name:
                    intro_elem = elem
                    old_introduction = elem.text
                    break
            
            if intro_elem is None:
                # Create new element if not found
                table_intros_node = root.find(".//table_introductions")
                if table_intros_node is None:
                    raise HTTPException(
                        status_code=500, 
                        detail="Invalid XML structure: missing table_introductions node"
                    )
                intro_elem = SubElement(table_intros_node, "table_introduction")
                intro_elem.set("table_name", table_name)
            
            # Update text content
            intro_elem.text = request.introduction
            
            # Atomic write: write to temp file first, then rename
            temp_file = intros_file.with_suffix(".xml.tmp")
            try:
                indent(tree, space="  ")
                tree.write(temp_file, encoding="utf-8", xml_declaration=True)
                
                # Validate the written XML before replacing original
                try:
                    parse(temp_file)
                except Exception as parse_err:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Generated XML is invalid: {parse_err}. Original file preserved."
                    )
                
                # Atomic replace
                temp_file.replace(intros_file)
                logger.info(f"Updated table introduction for {table_name} in client {client_id}")
                
                # Audit log the change (sensitive data modification)
                await audit_logger.log_event(
                    event_type=AuditEventType.METADATA_TABLE_INTRODUCTION_UPDATED,
                    severity=AuditSeverity.INFO,
                    user_id=admin_user.get("email", "unknown"),
                    client_id=client_id,
                    details={
                        "table_name": table_name,
                        "old_introduction": old_introduction[:200] if old_introduction else None,
                        "new_introduction": request.introduction[:200],
                        "backup_file": str(backup_file)
                    }
                )
                
                return {
                    "success": True,
                    "message": f"Table introduction updated for {table_name}",
                    "table_name": table_name,
                    "client_id": client_id,
                    "backup_created": str(backup_file)
                }
            except Exception as write_err:
                # Clean up temp file on error
                if temp_file.exists():
                    temp_file.unlink()
                raise write_err
        except HTTPException:
            # Restore from backup if operation failed
            if backup_file.exists() and intros_file.exists():
                try:
                    backup_file.replace(intros_file)
                    logger.info(f"Restored from backup after error")
                except Exception as restore_err:
                    logger.error(f"Failed to restore from backup: {restore_err}")
            raise
    except HTTPException:
        raise
    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse XML: {e}")
    except Exception as e:
        logger.error(f"Failed to update table introduction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.put("/metadata/{client_id}/table/{table_name}/column/{column_name}/description")
async def update_column_description(
    client_id: str,
    table_name: str,
    column_name: str,
    request: UpdateColumnDescriptionRequest,
    admin_user: Dict = Depends(require_admin),
):
    """
    Update column description in table_name_description.xml
    
    **Security**: 
    - Only admins can update metadata
    - Validates client_id matches admin's client
    - Creates backup before modification
    - Atomic write operation (writes to temp file first)
    
    **Location**: xml_prompts/clients/{client_id}/data_sources/data_descriptions/{table_name}_description.xml
    """
    try:
        # Validate client access - admin can only modify their own client's data
        admin_client_id = admin_user.get("client_id")
        if admin_client_id and admin_client_id != client_id:
            logger.warning(
                f"Admin {admin_user.get('email')} attempted to modify client {client_id} "
                f"(their client: {admin_client_id})"
            )
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. You can only modify metadata for your own client ({admin_client_id})"
            )
        
        desc_file = Path("xml_prompts/clients") / client_id / "data_sources" / "data_descriptions" / f"{table_name}_description.xml"
        
        if not desc_file.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"{table_name}_description.xml not found. Run explorer first to generate metadata."
            )
        
        # Create backup before modification (critical for sensitive data)
        backup_file = desc_file.with_suffix(f".xml.backup.{utcnow().strftime('%Y%m%d_%H%M%S')}")
        try:
            shutil.copy2(desc_file, backup_file)
            logger.info(f"Created backup: {backup_file}")
        except Exception as backup_err:
            logger.error(f"Failed to create backup: {backup_err}")
            raise HTTPException(
                status_code=500,
                detail="Failed to create backup before modification. Operation aborted for safety."
            )
        
        try:
            # Parse existing XML
            tree = parse(desc_file)
            root = tree.getroot()
            
            # Validate XML structure
            if root is None:
                raise HTTPException(status_code=500, detail="Invalid XML: root element is None")
            
            # Find the column element
            col_elem = None
            old_description = None
            for elem in root.findall(".//column"):
                if elem.get("name") == column_name:
                    col_elem = elem
                    desc_elem_old = elem.find("description")
                    old_description = desc_elem_old.text if desc_elem_old is not None else None
                    break
            
            if col_elem is None:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Column '{column_name}' not found in {table_name}_description.xml"
                )
            
            # Find or create description element
            desc_elem = col_elem.find("description")
            if desc_elem is None:
                desc_elem = SubElement(col_elem, "description")
            
            # Update text content
            desc_elem.text = request.description
            
            # Atomic write: write to temp file first, then rename
            temp_file = desc_file.with_suffix(".xml.tmp")
            try:
                indent(tree, space="  ")
                tree.write(temp_file, encoding="utf-8", xml_declaration=True)
                
                # Validate the written XML before replacing original
                try:
                    parse(temp_file)
                except Exception as parse_err:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Generated XML is invalid: {parse_err}. Original file preserved."
                    )
                
                # Atomic replace
                temp_file.replace(desc_file)
                logger.info(f"Updated column description for {table_name}.{column_name} in client {client_id}")
                
                # Audit log the change (sensitive data modification)
                await audit_logger.log_event(
                    event_type=AuditEventType.METADATA_COLUMN_DESCRIPTION_UPDATED,
                    severity=AuditSeverity.INFO,
                    user_id=admin_user.get("email", "unknown"),
                    client_id=client_id,
                    details={
                        "table_name": table_name,
                        "column_name": column_name,
                        "old_description": old_description[:200] if old_description else None,
                        "new_description": request.description[:200],
                        "backup_file": str(backup_file)
                    }
                )
                
                return {
                    "success": True,
                    "message": f"Column description updated for {table_name}.{column_name}",
                    "table_name": table_name,
                    "column_name": column_name,
                    "client_id": client_id,
                    "backup_created": str(backup_file)
                }
            except Exception as write_err:
                # Clean up temp file on error
                if temp_file.exists():
                    temp_file.unlink()
                raise write_err
        except HTTPException:
            # Restore from backup if operation failed
            if backup_file.exists() and desc_file.exists():
                try:
                    backup_file.replace(desc_file)
                    logger.info(f"Restored from backup after error")
                except Exception as restore_err:
                    logger.error(f"Failed to restore from backup: {restore_err}")
            raise
    except HTTPException:
        raise
    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse XML: {e}")
    except Exception as e:
        logger.error(f"Failed to update column description: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.put("/metadata/{client_id}/table/{table_name}/primary-key")
async def update_primary_key(
    client_id: str,
    table_name: str,
    request: UpdatePrimaryKeyRequest,
    admin_user: Dict = Depends(require_admin),
):
    """
    Update primary key for a table in table_metadata.xml
    
    **Security**: 
    - Only admins can update metadata
    - Validates client_id matches admin's client
    - Atomic write operation (writes to temp file first)
    
    **Location**: xml_prompts/clients/{client_id}/data_sources/meta_information/table_metadata.xml
    """
    try:
        # Validate client access - admin can only modify their own client's data
        admin_client_id = admin_user.get("client_id")
        if admin_client_id and admin_client_id != client_id:
            logger.warning(
                f"Admin {admin_user.get('email')} attempted to modify client {client_id} "
                f"(their client: {admin_client_id})"
            )
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. You can only modify metadata for your own client ({admin_client_id})"
            )
        
        # Validate that the column exists in the table
        # Get metadata to check column existence
        base_dir = Path("xml_prompts/clients") / client_id / "data_sources"
        desc_dir = base_dir / "data_descriptions"
        desc_file = desc_dir / f"{table_name}_description.xml"
        
        # Use the actual column name (preserve original case from file)
        primary_key_to_use = request.primary_key
        
        if desc_file.exists():
            try:
                tree = parse(desc_file)
                root = tree.getroot()
                column_names = [col.get("name") for col in root.findall(".//column") if col.get("name")]
                # Case-insensitive validation
                column_names_lower = [name.lower() for name in column_names]
                if request.primary_key.lower() not in column_names_lower:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Column '{request.primary_key}' not found in table '{table_name}'. Available columns: {', '.join(column_names[:10])}"
                    )
                # Use the actual column name from the file (preserve original case)
                primary_key_to_use = next((name for name in column_names if name.lower() == request.primary_key.lower()), request.primary_key)
            except ET.ParseError:
                logger.warning(f"Could not parse {desc_file} to validate column, proceeding anyway")
        else:
            logger.warning(f"Description file not found for {table_name}, cannot validate column existence")
        
        metadata_file = Path("xml_prompts/clients") / client_id / "data_sources" / "meta_information" / "table_metadata.xml"
        metadata_dir = metadata_file.parent
        
        # Create directory if it doesn't exist
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Parse existing XML or create new structure
            if metadata_file.exists():
                tree = parse(metadata_file)
                root = tree.getroot()
            else:
                # Create new table_metadata.xml structure
                root = Element("meta_information")
                root.set("type", "table_metadata")
                tables_elem = SubElement(root, "tables")
                tree = ElementTree(root)
            
            # Find or create tables element
            tables_elem = root.find("tables")
            if tables_elem is None:
                tables_elem = SubElement(root, "tables")
            
            # Find table element
            table_elem = None
            old_primary_key = None
            for elem in tables_elem.findall("table"):
                if elem.get("name") == table_name:
                    table_elem = elem
                    old_primary_key = elem.get("primary_key")
                    break
            
            if table_elem is None:
                # Create new table element
                table_elem = SubElement(tables_elem, "table")
                table_elem.set("name", table_name)
            
            # Update primary_key attribute
            table_elem.set("primary_key", primary_key_to_use)
            
            # Atomic write: write to temp file first, then rename
            temp_file = metadata_file.with_suffix(".xml.tmp")
            try:
                indent(tree, space="  ")
                tree.write(temp_file, encoding="utf-8", xml_declaration=True)
                
                # Validate the written XML before replacing original
                try:
                    parse(temp_file)
                except Exception as parse_err:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Generated XML is invalid: {parse_err}. Original file preserved."
                    )
                
                # Atomic replace
                temp_file.replace(metadata_file)
                logger.info(f"Updated primary key for {table_name} to {primary_key_to_use} in client {client_id}")
                
                # Audit log the change
                await audit_logger.log_event(
                    event_type=AuditEventType.METADATA_TABLE_INTRODUCTION_UPDATED,  # Reuse existing event type
                    severity=AuditSeverity.INFO,
                    user_id=admin_user.get("email", "unknown"),
                    client_id=client_id,
                    details={
                        "table_name": table_name,
                        "old_primary_key": old_primary_key,
                        "new_primary_key": primary_key_to_use
                    }
                )
                
                return {
                    "success": True,
                    "message": f"Primary key updated for {table_name}",
                    "table_name": table_name,
                    "primary_key": primary_key_to_use,
                    "client_id": client_id
                }
            except Exception as write_err:
                # Clean up temp file on error
                if temp_file.exists():
                    temp_file.unlink()
                raise write_err
        except HTTPException:
            raise
    except HTTPException:
        raise
    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse XML: {e}")
    except Exception as e:
        logger.error(f"Failed to update primary key: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/list-namespaces")
async def list_namespaces(request: TableListRequest, admin_user: Dict = Depends(require_admin)):
    """Return connector-specific namespaces available from provided DB credentials."""
    try:
        allowed_db_types = {"postgres", "mysql", "mongodb", "sap_hana", "sap_oracle", "sap_sybase"}
        if request.db_type not in allowed_db_types:
            raise HTTPException(status_code=400, detail=f"Unsupported db_type '{request.db_type}'.")

        dsn = await _build_table_list_dsn(request)
        ssh_config = request.additional_params.get("ssh") if request.additional_params else None
        namespaces: List[Dict[str, Any]] = []

        if request.db_type == "mongodb":
            connector = MongoConnector(dsn, request.db_name, ssh_config=ssh_config)
            await connector.connect()
            try:
                db_names = await connector._client.list_database_names() if connector._client else [request.db_name]
                for db_name in db_names:
                    if db_name.lower() in {"admin", "local", "config"}:
                        continue
                    try:
                        table_count = len(await connector._client[db_name].list_collection_names()) if connector._client else 0
                    except Exception:
                        table_count = 0
                    namespaces.append(_namespace_payload(db_name, "database", table_count, db_name == request.db_name))
            finally:
                await connector.disconnect()
        elif request.db_type == "sap_hana":
            def _get_hana_namespaces():
                password_to_use = request.db_password
                try:
                    if password_to_use.startswith("gAAAA"):
                        password_to_use = DBCredentialsService(None)._decrypt_password(password_to_use)
                except Exception:
                    pass
                connect_kwargs = {
                    "address": request.db_host,
                    "port": int(request.db_port),
                    "user": request.db_username,
                    "password": password_to_use,
                }
                if int(request.db_port) == 443:
                    connect_kwargs["encrypt"] = "true"
                    connect_kwargs["sslValidateCertificate"] = "false"
                conn = hdbcli.dbapi.connect(**connect_kwargs)
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT SCHEMA_NAME, SUM(TABLE_COUNT) AS TABLE_COUNT
                        FROM (
                            SELECT SCHEMA_NAME, COUNT(*) AS TABLE_COUNT FROM SYS.TABLES GROUP BY SCHEMA_NAME
                            UNION ALL
                            SELECT SCHEMA_NAME, COUNT(*) AS TABLE_COUNT FROM SYS.VIEWS GROUP BY SCHEMA_NAME
                        )
                        WHERE SCHEMA_NAME NOT LIKE '_SYS_%'
                          AND SCHEMA_NAME NOT LIKE 'HANA_%'
                          AND SCHEMA_NAME NOT IN ('SYS', 'SYSTEM')
                        GROUP BY SCHEMA_NAME
                        ORDER BY SCHEMA_NAME
                        """
                    )
                    return cursor.fetchall()
                finally:
                    conn.close()
            rows = await asyncio.to_thread(_get_hana_namespaces)
            namespaces = [_namespace_payload(str(row[0]), "schema", int(row[1] or 0), False) for row in rows]
        elif request.db_type == "postgres":
            connector = PostgresConnector(dsn, ssh_config=ssh_config)
            await connector.connect()
            try:
                session_factory = connector.get_db()
                async with session_factory() as session:
                    result = await session.execute(
                        text(
                            """
                            SELECT datname AS namespace_name
                            FROM pg_database
                            WHERE datallowconn = true
                              AND datistemplate = false
                              AND datname NOT IN ('postgres', 'template0', 'template1')
                            ORDER BY datname
                            """
                        )
                    )
                    db_names = [str(row["namespace_name"]) for row in result.mappings().fetchall()]
            finally:
                await connector.disconnect()

            for db_name in db_names:
                table_count = 0
                try:
                    db_dsn = _dsn(request.db_host, request.db_port, db_name, request.db_username, request.db_password)
                    db_connector = PostgresConnector(db_dsn, ssh_config=ssh_config)
                    await db_connector.connect()
                    try:
                        session_factory = db_connector.get_db()
                        async with session_factory() as session:
                            count_result = await session.execute(
                                text(
                                    """
                                    SELECT COUNT(*) AS table_count
                                    FROM information_schema.tables
                                    WHERE table_type IN ('BASE TABLE', 'VIEW')
                                      AND table_schema NOT IN ('pg_catalog', 'information_schema')
                                    """
                                )
                            )
                            table_count = int(count_result.scalar() or 0)
                    finally:
                        await db_connector.disconnect()
                except Exception as count_err:
                    logger.warning("Failed to count tables for Postgres database %s: %s", db_name, count_err)
                namespaces.append(_namespace_payload(db_name, "database", table_count, db_name == request.db_name))
        else:
            if request.db_type == "mysql":
                connector = MySQLConnector(dsn, ssh_config=ssh_config)
                namespace_type = "database"
                query = text(
                    """
                    SELECT s.SCHEMA_NAME AS namespace_name, COUNT(t.TABLE_NAME) AS table_count
                    FROM information_schema.SCHEMATA s
                    LEFT JOIN information_schema.TABLES t ON t.TABLE_SCHEMA = s.SCHEMA_NAME
                    WHERE s.SCHEMA_NAME NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
                    GROUP BY s.SCHEMA_NAME
                    ORDER BY s.SCHEMA_NAME
                    """
                )
            elif request.db_type == "sap_oracle":
                connector = OracleConnector(dsn, ssh_config=ssh_config)
                namespace_type = "schema"
                query = text(
                    """
                    SELECT OWNER AS namespace_name, COUNT(*) AS table_count
                    FROM ALL_TABLES
                    WHERE OWNER NOT IN ('SYS', 'SYSTEM')
                    GROUP BY OWNER
                    ORDER BY OWNER
                    """
                )
            elif request.db_type == "sap_sybase":
                connector = SybaseConnector(dsn)
                namespace_type = "schema"
                query = text(
                    """
                    SELECT USER_NAME(uid) AS namespace_name, COUNT(*) AS table_count
                    FROM sysobjects
                    WHERE type IN ('U', 'V') AND name NOT LIKE 'sys%'
                    GROUP BY USER_NAME(uid)
                    ORDER BY namespace_name
                    """
                )
            await connector.connect()
            try:
                session_factory = connector.get_db()
                async with session_factory() as session:
                    result = await session.execute(query)
                    for row in result.mappings().fetchall():
                        name = str(row["namespace_name"])
                        namespaces.append(
                            _namespace_payload(
                                name,
                                namespace_type,
                                int(row.get("table_count") or 0),
                                name == (request.schema_name or request.db_name or "public"),
                            )
                        )
            finally:
                await connector.disconnect()

        if not namespaces:
            fallback = request.schema_name or request.db_name or request.db_username or "default"
            namespaces = [_namespace_payload(fallback, "schema", 0, True)]

        return {
            "database": request.db_name,
            "db_type": request.db_type,
            "total_namespaces": len(namespaces),
            "namespaces": namespaces,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list namespaces for admin {admin_user.get('email')}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to list namespaces: {str(e)}")


@router.post("/list-tables")
async def list_tables(request: TableListRequest, admin_user: Dict = Depends(require_admin)):
    """Return basic table + column info directly from provided DB credentials.

    Does NOT persist client config or generate XML/LLM metadata.
    For quick ad-hoc exploration using logged-in admin context.
    """
    try:
        allowed_db_types = {"postgres", "mysql", "mongodb", "sap_hana", "sap_oracle", "sap_sybase"}
        if request.db_type not in allowed_db_types:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported db_type '{request.db_type}'. "
                    f"Supported: {', '.join(sorted(allowed_db_types))}."
                ),
            )

        # Log the connection attempt for debugging
        logger.info(
            f"Admin {admin_user.get('email')} attempting to list tables from "
            f"database '{request.db_name}' at {request.db_host}:{request.db_port} "
            f"(db_type={request.db_type})"
        )
        logger.info(
            "list_tables request details: db_type=%s schema_name=%s db_username=%s client_id=%s",
            request.db_type,
            request.schema_name,
            request.db_username,
            admin_user.get("client_id"),
        )
        
        # Build DSN based on db_type and provided fields
        if request.db_type == "mysql":
            mysql_db_name = _namespace_name(request.namespace) or request.db_name
            if request.db_url and "mysql" in request.db_url:
                dsn = request.db_url
            else:
                dsn = f"mysql+aiomysql://{request.db_username}:{request.db_password}@{request.db_host}:{request.db_port}/{mysql_db_name}"
            logger.info(f"Using MySQL DSN for listing tables")
        elif request.db_type == "mongodb":
            mongo_db_name = _namespace_name(request.namespace) or request.db_name
            dsn = _build_mongodb_dsn(
                request.db_host, 
                request.db_port, 
                mongo_db_name,
                request.db_username, 
                request.db_password, 
                request.db_url if not _namespace_name(request.namespace) else None,
            )
            logger.info(f"Using MongoDB DSN for listing tables")
            
            try:
                _mongo_ssh = request.additional_params.get("ssh") if request.additional_params else None
                connector = MongoConnector(dsn, mongo_db_name, ssh_config=_mongo_ssh)
                await connector.connect()
                db = connector.get_db()
                collections = await db.list_collection_names()
                # Filter out system collections
                collections = [c for c in collections if not c.startswith("system.")]
                
                tables = []
                for coll_name in collections:
                    try:
                        collection = db[coll_name]
                        # Get document count (this is fast in MongoDB)
                        doc_count = await collection.count_documents({})
                        tables.append({
                            "name": coll_name,
                            "schema": "mongodb",
                            "namespace_id": mongo_db_name,
                            "qualified_name": f"{mongo_db_name}.{coll_name}",
                            "table_id": f"{mongo_db_name}.{coll_name}",
                            "column_count": 0, # We'll do full metadata later
                            "columns": [],
                            "row_count": doc_count
                        })
                    except Exception as e:
                        logger.warning(f"Failed to get document count for {coll_name}: {e}")
                        tables.append({
                            "name": coll_name,
                            "schema": "mongodb",
                            "namespace_id": mongo_db_name,
                            "qualified_name": f"{mongo_db_name}.{coll_name}",
                            "table_id": f"{mongo_db_name}.{coll_name}",
                            "column_count": 0,
                            "columns": [],
                            "row_count": None
                        })
                
                await connector.disconnect()
                
                client_id = admin_user.get("client_id") or "session"
                client_name = admin_user.get("client_name") or admin_user.get("client_id") or "Session Client"
                payload = {
                    "client_id": client_id,
                    "client_name": client_name,
                    "schema": "mongodb",
                    "database": mongo_db_name,
                    "total_tables": len(tables),
                    "tables": tables,
                }
                if client_id and client_id != "session":
                    limits = await get_explorer_limits(client_id)
                    # Get limit metadata but don't trim tables - show all tables for selection
                    # The frontend will enforce the limit when selecting tables
                    _, limit_metadata = apply_table_column_limits(
                        tables, limits.get("table_limit"), limits.get("column_limit")
                    )
                    # Update metadata to reflect that all tables are available (not trimmed)
                    limit_metadata["total_tables_available"] = len(tables)
                    limit_metadata["total_tables_loaded"] = len(tables)
                    payload["explorer_limits"] = limit_metadata
                return payload
            except Exception as e:
                error_msg = f"MongoDB connection error: {str(e)}"
                logger.error(f"Failed to list collections for admin {admin_user.get('email')}: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
        elif request.db_type == "sap_hana":
            if request.db_url and "hana" in request.db_url:
                dsn = request.db_url
            else:
                dsn = f"hana://{request.db_username}:{request.db_password}@{request.db_host}:{request.db_port}"
            logger.info(f"Using SAP HANA DSN for listing tables")
          

        elif request.db_type == "sap_oracle":
            admin_client_id = admin_user.get("client_id")
            if not admin_client_id:
                raise HTTPException(
                    status_code=403,
                    detail="Admin user missing client_id. Please contact system administrator."
                )
            logger.info("Oracle list-tables reading directly from base_sap for client %s", admin_client_id)
            try:
                # Read directly from base_sap (all tables) for selection
                metadata_payload = await _get_base_sap_metadata(admin_client_id)
                client_name = admin_user.get("client_name") or admin_client_id
                sap_schema = _namespace_name(request.namespace) or request.schema_name
                tables = [
                    _table_identity(dict(table), sap_schema)
                    for table in metadata_payload.get("tables", [])
                ]
                return {
                    "client_id": admin_client_id,
                    "client_name": client_name,
                    "schema": sap_schema,
                    "database": request.db_name,
                    "total_tables": len(tables),
                    "tables": tables,
                    "explorer_limits": metadata_payload.get("explorer_limits"),
                }
            except HTTPException as e:
                # Re-raise HTTPException to preserve status code
                raise
            except Exception as e:
                logger.error(f"Failed to get metadata for client {admin_client_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to retrieve metadata: {str(e)}")
        elif request.db_type == "sap_sybase":
            admin_client_id = admin_user.get("client_id")
            if not admin_client_id:
                raise HTTPException(
                    status_code=403,
                    detail="Admin user missing client_id. Please contact system administrator."
                )
            logger.info("Sybase list-tables reading directly from base_sap for client %s", admin_client_id)
            try:
                # Read directly from base_sap (all tables) for selection
                metadata_payload = await _get_base_sap_metadata(admin_client_id)
                client_name = admin_user.get("client_name") or admin_client_id
                sap_schema = _namespace_name(request.namespace) or request.schema_name
                tables = [
                    _table_identity(dict(table), sap_schema)
                    for table in metadata_payload.get("tables", [])
                ]
                return {
                    "client_id": admin_client_id,
                    "client_name": client_name,
                    "schema": sap_schema,
                    "database": request.db_name or None,
                    "total_tables": len(tables),
                    "tables": tables,
                }
            except HTTPException as e:
                # Re-raise HTTPException to preserve status code
                raise
            except Exception as e:
                logger.error(f"Failed to get metadata for client {admin_client_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to retrieve metadata: {str(e)}")
        else:
            postgres_db_name = _namespace_name(request.namespace) if _namespace_type(request.namespace) == "database" else request.db_name
            if request.db_url and ("postgres" in request.db_url or "postgresql" in request.db_url) and _namespace_type(request.namespace) != "database":
                dsn = _ensure_asyncpg_dsn(request.db_url)
            else:
                dsn = _dsn(request.db_host, request.db_port, postgres_db_name or request.db_name, request.db_username, request.db_password)
            logger.info(f"Using PostgreSQL DSN for listing tables")
        
        # Determine effective schema for filtering
        if request.db_type == "postgres" and _namespace_type(request.namespace) == "database":
            target_schema = request.schema_name
            if target_schema == _namespace_name(request.namespace):
                target_schema = None
        else:
            target_schema = _namespace_name(request.namespace) or request.schema_name

            # # Default to DBADMIN if schema is not provided or is 'public' (frontend default)
            # if not target_schema or target_schema == 'public':
            #     target_schema = 'COREOPS_USER'

        if request.db_type == "sap_hana":
            # Use direct hdbcli connection to avoid SQLAlchemy async/sync conflicts
            def _get_hana_tables():
                # Decrypt password if needed
                from services.db_credentials_service import DBCredentialsService
                # Pass None for db since we only need _decrypt_password which doesn't use the db connection
                creds_service = DBCredentialsService(None)
                
                # Simple check if password looks encrypted (starts with 'gAAAA') - standard Fernet prefix
                # or just try to decrypt regardless, if it fails assume it's plain text (user testing)
                password_to_use = request.db_password
                try:
                    if password_to_use.startswith("gAAAA"):
                        password_to_use = creds_service._decrypt_password(password_to_use)
                except Exception:
                    # Not encrypted or decryption failed, try using as is
                    pass

                connect_kwargs = {
                    'address': request.db_host,
                    'port': int(request.db_port),
                    'user': request.db_username,
                    'password': password_to_use
                }
                
                # If connecting to HANA Cloud (usually port 443), enable SSL
                if int(request.db_port) == 443:
                    connect_kwargs['encrypt'] = 'true'
                    connect_kwargs['sslValidateCertificate'] = 'false'
                
                conn = hdbcli.dbapi.connect(**connect_kwargs)
                try:
                    cursor = conn.cursor()
                    
                    target_schema_to_use = target_schema if target_schema else request.db_username
                    where_clause = f"WHERE SCHEMA_NAME = '{target_schema_to_use}'"

                    logger.info(f"Filtering by schema: {target_schema_to_use}")
                    query = f"SELECT SCHEMA_NAME, TABLE_NAME FROM SYS.TABLES {where_clause} UNION ALL SELECT SCHEMA_NAME, VIEW_NAME FROM SYS.VIEWS {where_clause}"
                    
                    logger.info(f"Executing HANA query: {query}")
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    logger.info(f"HANA query returned {len(rows)} rows")
                    if len(rows) > 0:
                        logger.info(f"First 5 rows: {rows[:5]}")
                    
                    # Manual metadata construction
                    tables_metadata = MetaData()
                    for schema, table in rows:
                        # We just need the name for the list
                        # Construct a dummy table object to match the expected output structure
                        from sqlalchemy import Table, Column, String
                        Table(table, tables_metadata, schema=schema)
                    return tables_metadata
                finally:
                    conn.close()

            metadata = await asyncio.to_thread(_get_hana_tables)
        else:
            from db_config.connectors.postgres_connector import PostgresConnector
            ssh_config = None
            if hasattr(request, "additional_params") and request.additional_params:
                ssh_config = request.additional_params.get("ssh")

            if request.db_type == "mysql":
                connector = MySQLConnector(dsn, ssh_config=ssh_config)
            elif request.db_type == "sap_oracle":
                connector = OracleConnector(dsn, ssh_config=ssh_config)
            else:
                connector = PostgresConnector(dsn, ssh_config=ssh_config)

            await connector.connect()

            metadata = MetaData()
            try:
                async with connector._engine.begin() as conn:
                    def _reflect(sync_conn):
                        metadata.reflect(bind=sync_conn, schema=target_schema, views=True)
                    await conn.run_sync(_reflect)
            except Exception as e:
                await connector.disconnect()
                error_msg = str(e)
                # Provide more descriptive error messages
                if "does not exist" in error_msg:
                    error_msg = (
                        f"Database '{request.db_name}' does not exist on the PostgreSQL server. "
                        f"Please check the database name. You may need to create the database first using: "
                        f"`CREATE DATABASE {request.db_name};`"
                    )
                elif "authentication failed" in error_msg.lower() or "password" in error_msg.lower():
                    error_msg = f"Authentication failed. Please check your database username and password."
                elif "could not connect" in error_msg.lower() or "connection refused" in error_msg.lower():
                    error_msg = f"Could not connect to database at {request.db_host}:{request.db_port}. Please check the host and port."
                else:
                    error_msg = f"Database connection error: {error_msg}"
                logger.error(
                    f"Failed to list tables for admin {admin_user.get('email')}: {error_msg} "
                    f"(Attempted connection to: {request.db_host}:{request.db_port}/{request.db_name})"
                )
                raise HTTPException(status_code=400, detail=error_msg)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in list_tables for admin {admin_user.get('email')}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    tables: List[Dict] = []
    # Use target_schema if defined (for HANA), otherwise request.schema_name
    filter_schema = target_schema
    
    for table_name, table in metadata.tables.items():
        # Extract schema and table name from qualified names (e.g., "public.users" -> ("public", "users"))
        # This is needed for PostgreSQL and MySQL which may return qualified names from metadata.tables
        if '.' in table_name:
            schema_part, bare_name = table_name.split('.', 1)
            # Filter by schema if filter_schema is provided
            if filter_schema and schema_part != filter_schema:
                continue
            name = bare_name
            extracted_schema = schema_part
        else:
            name = table_name
            extracted_schema = None
        
        cols = []
        for col in table.columns:
            cols.append({
                "name": col.name,
                "type": str(col.type),
                "nullable": bool(col.nullable),
                "primary_key": bool(col.primary_key)
            })
        tables.append({
            "name": name,
            "schema": table.schema or extracted_schema or filter_schema or "public",
            "column_count": len(cols),
            "columns": cols,
            "row_count": None  # Will be populated below
        })
        if request.db_type == "postgres" and _namespace_type(request.namespace) == "database":
            tables[-1]["namespace_id"] = _namespace_name(request.namespace)
            tables[-1]["qualified_name"] = f"{_namespace_name(request.namespace)}.{name}"
            tables[-1]["table_id"] = f"{_namespace_name(request.namespace)}.{name}"
        else:
            _table_identity(tables[-1], tables[-1].get("schema"))

    # Add row counts for tables
    if request.db_type == "postgres":
        try:
            def _quote_pg_identifier(identifier: str) -> str:
                return '"' + str(identifier).replace('"', '""') + '"'

            async with connector._engine.begin() as conn:
                count_result = await conn.execute(
                    text(
                        """
                        SELECT ns.nspname AS schema_name,
                               cls.relname AS table_name,
                               GREATEST(cls.reltuples::bigint, 0) AS row_count
                        FROM pg_class cls
                        JOIN pg_namespace ns ON ns.oid = cls.relnamespace
                        WHERE cls.relkind IN ('r', 'p', 'v', 'm')
                          AND ns.nspname NOT IN ('pg_catalog', 'information_schema')
                        """
                    )
                )
                row_count_map = {
                    (row["schema_name"], row["table_name"]): int(row["row_count"] or 0)
                    for row in count_result.mappings().fetchall()
                }
                for table in tables:
                    try:
                        schema = table.get("schema") or "public" or filter_schema
                        table_name = table["name"]
                        estimated_count = row_count_map.get((schema, table_name), 0)
                        table["row_count"] = estimated_count

                        # Fresh or never-analyzed PostgreSQL tables often report reltuples=0.
                        # Use an exact count only in that ambiguous case so non-empty tables
                        # show useful numbers on the selection screen.
                        if estimated_count == 0:
                           
                            try:
                                exact_query = text(
                                    f"SELECT COUNT(*) FROM {_quote_pg_identifier(schema)}.{_quote_pg_identifier(table_name)}"
                                )
                                exact_result = await conn.execute(exact_query)
                                table["row_count"] = int(exact_result.scalar() or 0)
                            except Exception as exact_err:
                                logger.debug(
                                    "Exact row count fallback failed for %s.%s: %s",
                                    schema,
                                    table_name,
                                    exact_err,
                                )
                    except Exception as e:
                        logger.warning(f"Failed to get row count for {table_name}: {e}")
                        table["row_count"] = None
        except Exception as e:
            logger.warning(f"Failed to get row counts for PostgreSQL tables: {e}")
    
    elif request.db_type == "mysql" and 'connector' in locals() and getattr(connector, "_engine", None):
        try:
            async with connector._engine.begin() as conn:
                for table in tables:
                    try:
                        table_name = table["name"]
                        # Use the table's schema (consistent with PostgreSQL logic)
                        schema = request.db_name or filter_schema or table.get("schema")
                        # Use information_schema for MySQL (approximate but fast)
                        count_query = text(
                            """
                            SELECT TABLE_ROWS 
                            FROM information_schema.TABLES 
                            WHERE TABLE_SCHEMA = :schema
                            AND TABLE_NAME = :table_name
                            """
                        )
                        result = await conn.execute(
                            count_query,
                            {
                                "table_name": table_name,
                                "schema": schema,
                            },
                        )
                        row = result.fetchone()
                        if row and row[0] is not None:
                            row_count = int(row[0])
                            # Clamp negative values to 0 (TABLE_ROWS can sometimes be negative)
                            table["row_count"] = max(0, row_count)
                        else:
                            table["row_count"] = 0
                    except Exception as e:
                        logger.warning(f"Failed to get row count for {table['name']}: {e}")
                        table["row_count"] = None
        except Exception as e:
            logger.warning(f"Failed to get row counts for MySQL tables: {e}")
    
    elif request.db_type == "mongodb":
        # MongoDB row counts are already handled in the MongoDB section above
        # The collections loop already sets row_count
        pass
    
    elif request.db_type == "sap_oracle":
        # For SAP Oracle, row counts should come from metadata if available
        # They may already be in the table metadata from ALL_TABLES
        pass
    
    elif request.db_type == "sap_hana":
        # For SAP HANA, retrieve row counts from M_TABLES system view
        try:
            def _get_hana_row_counts():
                from services.db_credentials_service import DBCredentialsService
                creds_service = DBCredentialsService(None)
                
                password_to_use = request.db_password
                try:
                    if password_to_use.startswith("gAAAA"):
                        password_to_use = creds_service._decrypt_password(password_to_use)
                except Exception:
                    pass

                connect_kwargs = {
                    'address': request.db_host,
                    'port': int(request.db_port),
                    'user': request.db_username,
                    'password': password_to_use
                }
                
                if int(request.db_port) == 443:
                    connect_kwargs['encrypt'] = 'true'
                    connect_kwargs['sslValidateCertificate'] = 'false'
                
                conn = hdbcli.dbapi.connect(**connect_kwargs)
                try:
                    cursor = conn.cursor()
                    target_schema_to_use = filter_schema if filter_schema else request.db_username
                    
                    query = f"""
                    SELECT SCHEMA_NAME, TABLE_NAME, RECORD_COUNT 
                    FROM SYS.M_TABLES 
                    WHERE SCHEMA_NAME = '{target_schema_to_use}'
                    """
                    
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    
                    row_count_map = {
                        (row[0], row[1]): int(row[2] or 0)
                        for row in rows
                    }
                    return row_count_map
                finally:
                    conn.close()

            row_count_map = await asyncio.to_thread(_get_hana_row_counts)
            for table in tables:
                try:
                    schema = table.get("schema") or request.db_username or filter_schema 
                    table_name = table["name"]
                    table["row_count"] = row_count_map.get((schema, table_name), 0)
                except Exception as e:
                    logger.warning(f"Failed to get row count for {table_name}: {e}")
                    table["row_count"] = None
        except Exception as e:
            logger.warning(f"Failed to get row counts for SAP HANA tables: {e}")
    
    # Dispose engine after getting row counts (only for PostgreSQL and MySQL)
    if request.db_type == "postgres":
        try:
            await connector.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting: {e}")
    elif request.db_type =="mysql" and 'connector' in locals():
        try:
            await connector.disconnect()
        except Exception as e:
            logger.warning(f"Error disposing engine: {e}")

    client_id = admin_user.get("client_id") or "session"
    client_name = admin_user.get("client_name") or admin_user.get("client_id") or "Session Client"
    payload = {
        "client_id": client_id,
        "client_name": client_name,
        "schema": filter_schema or request.schema_name,
        "database": _namespace_name(request.namespace) if _namespace_type(request.namespace) == "database" else request.db_name,
        "total_tables": len(tables),
        "tables": tables,
    }
    if client_id and client_id != "session":
        limits = await get_explorer_limits(client_id)
        # Get limit metadata but don't trim tables - show all tables for selection
        # The frontend will enforce the limit when selecting tables
        _, limit_metadata = apply_table_column_limits(
            tables, limits.get("table_limit"), limits.get("column_limit")
        )
        # Update metadata to reflect that all tables are available (not trimmed)
        limit_metadata["total_tables_available"] = len(tables)
        limit_metadata["total_tables_loaded"] = len(tables)
        payload["explorer_limits"] = limit_metadata
    return payload


@router.post("/erd")
async def get_erd(
    request: TableListRequest,
    admin_user: dict = Depends(require_admin)
):
    """Extract Entity Relationship Diagram data from database schema.
    
    Returns nodes (tables with columns) and edges (foreign key relationships).
    """
    try:
        # Use db_url if provided, otherwise build DSN from individual fields
        if request.db_url and request.db_url.strip():
            if request.db_type == "mysql":
                 dsn = request.db_url
            elif request.db_type == "mongodb":
                 dsn = request.db_url
            elif request.db_type == "sap_hana":
                 dsn = request.db_url
            else:
                 dsn = _ensure_asyncpg_dsn(request.db_url)
            logger.info(f"Using provided database URL for ERD extraction")
        elif request.db_type == "mongodb":
            dsn = _build_mongodb_dsn(
                request.db_host, 
                request.db_port, 
                request.db_name, 
                request.db_username, 
                request.db_password, 
                request.db_url
            )
            logger.info(f"Using MongoDB DSN for ERD extraction")
            
            try:
                _mongo_ssh = request.additional_params.get("ssh") if hasattr(request, "additional_params") and request.additional_params else None
                connector = MongoConnector(dsn, request.db_name, ssh_config=_mongo_ssh)
                await connector.connect()
                db = connector.get_db()
                collections = await db.list_collection_names()
                collections = [c for c in collections if not c.startswith("system.")]
                
                nodes = []
                for coll_name in collections:
                    nodes.append({
                        "id": coll_name,
                        "label": coll_name,
                        "schema": "mongodb",
                        "columns": []
                    })
                
                await connector.disconnect()
                return {
                    "nodes": nodes,
                    "edges": []
                }
            except Exception as e:
                error_msg = f"MongoDB connection error: {str(e)}"
                logger.error(f"Failed to extract ERD for MongoDB: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
        else:
            if request.db_type == "mysql":
                dsn = f"mysql+aiomysql://{request.db_username}:{request.db_password}@{request.db_host}:{request.db_port}/{request.db_name}"
            elif request.db_type == "sap_hana":
                dsn = f"hana://{request.db_username}:{request.db_password}@{request.db_host}:{request.db_port}"
                if int(request.db_port) == 443:
                    dsn += "?encrypt=true&sslValidateCertificate=false"
            else:
                dsn = _dsn(request.db_host, request.db_port, request.db_name, request.db_username, request.db_password)
            logger.info(f"Built DSN from individual fields for ERD extraction")
            import defusedxml.ElementTree as ET
            import io


        if request.db_type == "sap_hana":
            from sqlalchemy import create_engine
            # Use a completely synchronous approach for HANA to avoid async/sync conflicts
            def _get_hana_metadata():
                engine = create_engine(dsn)
                try:
                    metadata = MetaData()
                    metadata.reflect(bind=engine, schema=request.schema_name, views=True)
                    return metadata
                finally:
                    engine.dispose()

            metadata = await asyncio.to_thread(_get_hana_metadata)
        else:
            engine = create_async_engine(dsn, echo=False)
            metadata = MetaData()
            async with engine.begin() as conn:
                await conn.run_sync(lambda c: metadata.reflect(c, schema=request.schema_name, views=True))
        
        # Build nodes (tables)
        nodes = []
        for table_name, table in metadata.tables.items():
            # table_name format is "schema.table" or just "table"
            if "." in table_name:
                schema, name = table_name.split(".", 1)
            else:
                name = table_name
                
            columns = []
            for col in table.columns:
                columns.append({
                    "name": col.name,
                    "type": str(col.type),
                    "primary_key": bool(col.primary_key),
                    "nullable": bool(col.nullable)
                })
            
            nodes.append({
                "id": name,
                "label": name,
                "schema": request.schema_name,
                "columns": columns
            })
        
        # Build edges (foreign key relationships)
        edges = []
        for table_name, table in metadata.tables.items():
            if "." in table_name:
                _, source_table = table_name.split(".", 1)
            else:
                source_table = table_name
                
            for fk in table.foreign_keys:
                # fk.column is the source column (in this table)
                # fk.target_fullname is like "schema.table.column"
                target_parts = fk.target_fullname.split(".")
                if len(target_parts) >= 2:
                    target_table = target_parts[-2]
                    target_column = target_parts[-1]
                else:
                    # Fallback if format unexpected
                    target_table = str(fk.column.table.name)
                    target_column = fk.column.name
                
                edges.append({
                    "id": f"{source_table}.{fk.parent.name}->{target_table}.{target_column}",
                    "source": source_table,
                    "target": target_table,
                    "sourceColumn": fk.parent.name,
                    "targetColumn": target_column,
                    "type": "foreign_key"
                })
        
        if request.db_type != "sap_hana" and request.db_type != "sap_ecc" and request.db_type != "sap_oracle":
            await engine.dispose()
        
        return {
            "nodes": nodes,
            "edges": edges,
            "database": request.db_name,
            "schema": request.schema_name
        }
        
    except Exception as e:
        logger.error(f"Failed to extract ERD: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _process_file_background(
    file_path: Path,
    client_id: str,
    file_ext: str,
    schema_data: dict,
    user_email: str,
    on_table_done=None,
    dataset_id: Optional[str] = None,
):
    """Background task to convert uploaded files to Parquet and generate AI metadata."""
    try:
        # Validate LLM config before processing
        db = await get_db()
        
        # Apply subscription plan table/column limits before processing
        limits = await get_explorer_limits(client_id)
        trimmed_tables, limit_metadata = apply_table_column_limits(
            schema_data.get("tables") or [],
            limits.get("table_limit"),
            limits.get("column_limit"),
        )
        schema_data = {"tables": trimmed_tables, "total_tables": len(trimmed_tables)}
        if not trimmed_tables:
            raise HTTPException(
                status_code=400,
                detail="No tables available to process for this client under current plan limits. Please upgrade your plan."
            )
        
        # Clean up existing data before processing new files
        # Preserve uploads directory since we need the uploaded file for processing
        _cleanup_for_file_upload(client_id, dataset_id=dataset_id)

        parquet_dir = assets_datasets_dir(client_id, dataset_id)
        parquet_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified parquet directory: {parquet_dir}")
        
        import pandas as pd
        import zipfile
        import io
        import asyncio

        from explorer.models import TableMetadata

        # For Excel: read all sheets once to avoid repeated file opens
        excel_all = None
        if file_ext in ['.xlsx', '.xls']:
            engine = 'openpyxl' if file_ext == '.xlsx' else 'xlrd'
            excel_all = await asyncio.to_thread(
                pd.read_excel, file_path, sheet_name=None, engine=engine
            )
            if isinstance(excel_all, pd.DataFrame):
                excel_all = {None: excel_all}
            logger.info(f"Read Excel file {file_path.name} once: {len(excel_all)} sheet(s)")
        
        saved_files = []
        tables_metadata: List[TableMetadata] = []
        for table in schema_data['tables']:
            table_name = table['name']
            df = None
            
            try:
                # Read the source data based on file type
                # Get sheet_name from table metadata if it's an Excel file
                sheet_name = table.get('sheet_name')
                
                if file_ext == '.csv':
                    try:
                        df = await asyncio.to_thread(pd.read_csv, file_path)
                    except UnicodeDecodeError:
                        logger.warning(f"UTF-8 decode failed for {file_path.name}, retrying with latin-1 encoding")
                        df = await asyncio.to_thread(pd.read_csv, file_path, encoding="latin-1")
                    logger.info(f"Read CSV file {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
                elif file_ext in ['.xlsx', '.xls']:
                    # Use pre-loaded excel_all (read once above)
                    df = excel_all.get(sheet_name) if sheet_name is not None else excel_all.get(None)
                    if df is None and excel_all:
                        first_key = next(iter(excel_all.keys()))
                        df = excel_all[first_key]
                    if df is not None:
                        logger.info(f"Using Excel sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")
                elif file_ext == '.parquet':
                    df = await asyncio.to_thread(pd.read_parquet, file_path)
                    logger.info(f"Read Parquet file {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
                elif file_ext == '.zip':
                    file_source = table.get('file_source')
                    if not file_source:
                        logger.warning(f"No file_source for table {table_name}, skipping")
                        continue

                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        with zip_ref.open(file_source) as zf:
                            file_bytes = io.BytesIO(zf.read())
                            source_ext = Path(file_source).suffix.lower()

                            if source_ext == '.csv':
                                try:
                                    df = await asyncio.to_thread(pd.read_csv, file_bytes)
                                except UnicodeDecodeError:
                                    logger.warning(f"UTF-8 decode failed for {file_source} in ZIP, retrying with latin-1 encoding")
                                    file_bytes.seek(0)
                                    df = await asyncio.to_thread(pd.read_csv, file_bytes, encoding="latin-1")
                            elif source_ext in ['.xlsx', '.xls']:
                                engine = 'openpyxl' if source_ext == '.xlsx' else 'xlrd'
                                # Read specific sheet if sheet_name is specified
                                if sheet_name is not None:
                                    df = await asyncio.to_thread(
                                        pd.read_excel, file_bytes, sheet_name=sheet_name, engine=engine
                                    )
                                    logger.info(f"Read Excel file {file_source} sheet '{sheet_name}' from ZIP {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
                                else:
                                    df = await asyncio.to_thread(
                                        pd.read_excel, file_bytes, engine=engine
                                    )
                                    logger.info(f"Read Excel file {file_source} from ZIP {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
                            elif source_ext == '.parquet':
                                df = await asyncio.to_thread(pd.read_parquet, file_bytes)
                            else:
                                logger.warning(f"Unsupported file in ZIP: {file_source}")
                                continue
                    if source_ext not in ['.xlsx', '.xls']:
                        logger.info(f"Read file {file_source} from ZIP {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
                else:
                    logger.warning(f"Unknown file extension: {file_ext}")
                    continue
                
                if df is None or df.empty:
                    logger.warning(f"DataFrame is empty for table {table_name}, skipping")
                    continue
                
                # Build TableMetadata from df (avoids re-reading Parquet later)
                table_meta = FileMetadataGenerator._build_table_metadata_from_df(df, table_name, max_sample_rows=100)
                tables_metadata.append(table_meta)
                
                # Ensure directory exists before saving
                if not parquet_dir.exists():
                    parquet_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Recreated parquet directory: {parquet_dir}")
                
                # Save as parquet
                parquet_path = parquet_dir / f"{table_name}.parquet"
                FileSchemaExtractor.save_as_parquet(df, parquet_path)
                
                # Verify file was actually saved
                if parquet_path.exists() and parquet_path.stat().st_size > 0:
                    logger.info(f"Successfully saved Parquet file: {parquet_path} ({parquet_path.stat().st_size} bytes)")
                    saved_files.append(str(parquet_path))
                    await _sync_file_to_gcs(parquet_path, client_id, dataset_id, subfolder="datasets")
                else:
                    logger.error(f"Failed to save Parquet file: {parquet_path} (file does not exist or is empty)")
                    raise Exception(f"Failed to save parquet file for table {table_name}")
            except Exception as e:
                logger.error(f"Error processing table {table_name} from file {file_path.name}: {e}", exc_info=True)
                # Continue with other tables instead of failing completely
                continue
        
        logger.info(f"Successfully processed {len(saved_files)} parquet files from {file_path.name} to {parquet_dir}")
        if not saved_files:
            raise HTTPException(
                status_code=400,
                detail="No parquet files were generated from the uploaded file. Please verify file contents and plan limits."
            )
        
        # Generate AI metadata in background (use pre-built tables_metadata, no Parquet re-read)
        try:
            logger.info(f"Generating AI metadata for client {client_id}")
            output_root = Path("xml_prompts/clients") / client_id
            metadata_generator = FileMetadataGenerator(
                client_id=client_id, output_root=output_root, db=db, dataset_id=dataset_id
            )
            metadata_result = await metadata_generator.generate_metadata_from_tables(tables_metadata, on_table_done=on_table_done)
            logger.info(f"AI metadata generation complete: {metadata_result}")
            
            # Persist explorer_limits so GET /metadata can return it
            meta_dir = metadata_generator.data_sources_root / "meta_information"
            meta_dir.mkdir(parents=True, exist_ok=True)
            explorer_limits_path = meta_dir / "explorer_limits.json"
            limits_json = json.dumps(limit_metadata, indent=2)
            await asyncio.to_thread(explorer_limits_path.write_text, limits_json, "utf-8")

            # Generate suggested questions after metadata is created
            try:
                logger.info(f"Generating suggested questions for client {client_id}")
                from explorer.question_generator import QuestionGenerator
                question_generator = QuestionGenerator(client_id=client_id, db=db, dataset_id=dataset_id)
                questions = await question_generator.generate_questions(count=30)
                logger.info(f"Generated {len(questions)} suggested questions for client {client_id}")
            except Exception as qe:
                logger.warning(f"Failed to generate suggested questions: {str(qe)}", exc_info=True)
                # Don't fail the whole process if question generation fails
            
            # Copy/merge base prompts (planner.xml, python.xml, etc.) for this client
            try:
                await copy_base_prompts_for_client(client_id, output_root)
                logger.info(f"Copied base prompts for client {client_id} (file upload)")
            except Exception as bp_err:
                logger.warning(f"Failed to copy base prompts for client {client_id}: {bp_err}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to generate AI metadata: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to generate AI metadata: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Background file processing failed: {str(e)}", exc_info=True)
        raise


@router.get("/upload-config")
async def get_upload_config(admin_user: Dict = Depends(require_admin)):
    """Return file-upload limits so the frontend stays in sync with the backend."""
    max_bytes = int(os.getenv("MAX_UPLOAD_SIZE", 200 * 1024 * 1024))
    return {"max_upload_size_bytes": max_bytes}


@router.post("/upload")
async def upload_file(
    client_id: str = Query(..., description="Client ID for file storage"),
    dataset_id: Optional[str] = Query(None, description="Dataset scope for uploads (subfolder under uploads/)"),
    file: UploadFile = File(...),
    admin_user: dict = Depends(require_admin)
):
    """
    Upload CSV, Excel, Parquet, or ZIP file for schema exploration.
    Only saves file and extracts basic schema - no heavy processing.
    
    Heavy processing (Parquet conversion + AI metadata) happens when
    user clicks 'Explore Uploaded Data' button.
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in FileSchemaExtractor.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Supported: {', '.join(FileSchemaExtractor.SUPPORTED_FORMATS)}"
            )
        
        MAX_FILE_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 200 * 1024 * 1024))
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

        if admin_user.get("role") != "super_admin" and file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB. Your file: {file_size / (1024*1024):.2f}MB"
            )

        _empty_upload_folder(client_id, dataset_id=dataset_id)
        upload_dir = assets_uploads_dir(client_id, dataset_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file - stream in chunks to avoid memory issues
        # SECURITY: Sanitize filename to prevent path traversal
        sanitized_name = sanitize_filename(file.filename)
        timestamp = utcnow().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{sanitized_name}"
        file_path = upload_dir / safe_filename
        
        # SECURITY: Validate path is within upload directory (prevent path traversal)
        is_valid, error_msg = validate_path_within_directory(file_path, upload_dir)
        if not is_valid:
            logger.error(f"Path traversal attempt detected: {error_msg}")
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Stream file to disk in 1MB chunks
        CHUNK_SIZE = 1024 * 1024  # 1MB
        async with aiofiles.open(file_path, 'wb') as f:
            while chunk := await file.read(CHUNK_SIZE):
                await f.write(chunk)
        
        logger.info(f"Saved uploaded file: {file_path} ({file_size} bytes)")
        
        # Sync to GCS if enabled
        await _sync_file_to_gcs(file_path, client_id, dataset_id, subfolder="uploads")
        
        # Small delay to ensure file is completely flushed to disk
        import asyncio
        await asyncio.sleep(0.1)
        
        # Verify file was written correctly
        if not file_path.exists() or file_path.stat().st_size != file_size:
            raise HTTPException(
                status_code=500,
                detail="File upload incomplete. Please try again."
            )
        
        # Quick schema extraction (just metadata, no data reading)
        try:
            extractor = FileSchemaExtractor(file_path)
            # Schema extraction may parse Excel; offload to keep event loop responsive.
            schema_data = await asyncio.to_thread(extractor.extract_schema)
        except ValueError as e:
            # User-friendly error for invalid files
            logger.error(f"Schema extraction failed for {file.filename}: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error during schema extraction: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process file '{file.filename}': {str(e)}"
            )
        
        # Apply subscription plan table/column limits
        limits = await get_explorer_limits(client_id, role=admin_user.get("role"))
        trimmed_tables, limit_metadata = apply_table_column_limits(
            schema_data.get("tables") or [],
            limits.get("table_limit"),
            limits.get("column_limit"),
        )
        schema_data = {
            "tables": trimmed_tables,
            "total_tables": len(trimmed_tables),
            "source_type": schema_data.get("source_type", "file"),
            "extraction_time": schema_data.get("extraction_time"),
        }
        
        # Audit log
        await audit_logger.log_event(
            event_type=AuditEventType.FILE_UPLOAD,
            severity=AuditSeverity.INFO,
            user_id=admin_user.get('email', 'unknown'),
            client_id=client_id,
            details={
                'filename': file.filename,
                'file_size': file_size,
                'tables_extracted': schema_data['total_tables']
            }
        )
        
        # Return immediately with schema info - no background processing
        # Processing happens when user clicks "Explore Uploaded Data" button
        return {
            "success": True,
            "client_id": client_id,
            "filename": file.filename,
            "file_size": file_size,
            "file_path": str(file_path),
            "schema": schema_data,
            "explorer_limits": limit_metadata,
            "message": f"File uploaded successfully. Found {schema_data['total_tables']} table(s)."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


class ProcessUploadRequest(BaseModel):
    file_path: str
    schema_info: dict


class ProcessFileItem(BaseModel):
    file_path: str
    schema_info: dict


class ProcessUploadsRequest(BaseModel):
    files: List[ProcessFileItem]


class RemoveUploadRequest(BaseModel):
    file_path: str


@router.post("/process-upload")
async def process_uploaded_file(
    request: ProcessUploadRequest,
    dataset_id: Optional[str] = Query(None, description="Target dataset (required if multiple datasets exist)"),
    admin_user: dict = Depends(require_admin),
):
    """
    Process uploaded file: Convert to Parquet and generate AI metadata.
    This is called when user clicks 'Explore Uploaded Data' button.
    """
    client_id: str = admin_user.get("client_id")
    try:
        file_path = Path(request.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        schema_data = request.schema_info
        
        # Apply limits for response (background task will apply again and persist)
        limits = await get_explorer_limits(client_id, role=admin_user.get("role"))
        trimmed_tables, limit_metadata = apply_table_column_limits(
            schema_data.get("tables") or [],
            limits.get("table_limit"),
            limits.get("column_limit"),
        )
        schema_data = {"tables": trimmed_tables, "total_tables": len(trimmed_tables)}

        # Save file upload config to db_credentials BEFORE heavy processing starts.
        # This ensures the UI shows "processing in progress" on page refresh instead of
        # showing a blank upload form while processing is still running.
        credential_warning = None
        try:
            cred_db = await get_db()
            cred_service = DBCredentialsService(cred_db)
            # Preserve existing additional_params (e.g. file_size set by frontend)
            existing_cred = await cred_service.get_credentials(
                client_id=client_id, db_type=None, dataset_id=dataset_id, decrypt_password=False
            )
            if not existing_cred:
                raise RuntimeError("store_in_local not configured in DB Configs")
            existing_params = existing_cred.get("additional_params") or {}
            configured_store_in_local = require_store_in_local(existing_cred)
            merged_params = {
                **existing_params,
                "file_count": 1,
                "total_tables": len(schema_data.get("tables", [])),
                "processed_tables": 0,
                "processing_status": "processing",
            }
            await cred_service.save_credentials(
                client_id=client_id,
                db_type="file_upload",
                db_host="",
                db_port=0,
                db_name="",
                db_username="",
                db_password="",
                db_url="",
                additional_params=merged_params,
                store_in_local=configured_store_in_local,
                created_by=admin_user.get('email', 'unknown'),
                dataset_id=dataset_id,
            )
            logger.info(f"Saved file upload config (processing) in db_credentials for client {client_id}")
        except Exception as e:
            logger.error(f"Failed to save file upload config to db_credentials: {e}", exc_info=True)
            credential_warning = "File processed successfully but configuration could not be saved. Data may not persist after refresh."

        _single_processed_count = 0

        async def _on_single_table_done() -> None:
            nonlocal _single_processed_count
            _single_processed_count += 1
            try:
                _prog_db = await get_db()
                _prog_svc = DBCredentialsService(_prog_db)
                await _prog_svc.update_processing_progress(
                    client_id, _single_processed_count, dataset_id=dataset_id
                )
            except Exception as _prog_err:
                logger.warning(f"Failed to update processing progress for client {client_id}: {_prog_err}")

        # Use the background processing function synchronously here
        # since user is waiting for it (with progress indicator)
        await _process_file_background(
            file_path=file_path,
            client_id=client_id,
            file_ext=file_ext,
            schema_data=schema_data,
            user_email=admin_user.get("email", "unknown"),
            on_table_done=_on_single_table_done,
            dataset_id=dataset_id,
        )

        logger.info(f"File processing complete for client {client_id}")

        # Update processing_status to "complete" after successful processing
        try:
            cred_db = await get_db()
            cred_service = DBCredentialsService(cred_db)
            # Preserve existing additional_params (e.g. file_size set by frontend)
            existing_cred = await cred_service.get_credentials(
                client_id=client_id, db_type=None, dataset_id=dataset_id, decrypt_password=False
            )
            if not existing_cred:
                raise RuntimeError("store_in_local not configured in DB Configs")
            existing_params = existing_cred.get("additional_params") or {}
            configured_store_in_local = require_store_in_local(existing_cred)
            merged_params = {
                **existing_params,
                "file_count": 1,
                "total_tables": len(schema_data.get("tables", [])),
                "processing_status": "complete",
            }
            # Remove transient keys
            merged_params.pop("processed_tables", None)
            await cred_service.save_credentials(
                client_id=client_id,
                db_type="file_upload",
                db_host="",
                db_port=0,
                db_name="",
                db_username="",
                db_password="",
                db_url="",
                additional_params=merged_params,
                store_in_local=configured_store_in_local,
                created_by=admin_user.get('email', 'unknown'),
                dataset_id=dataset_id,
            )
            logger.info(f"Updated file upload config to complete in db_credentials for client {client_id}")
            # Notify admin that the uploaded dataset is ready for analysis
            try:
                from notifications.notification_service import create_notification
                from notifications.notification_model import Notification
                _notif_uid = str(admin_user.get("user_id") or admin_user.get("_id") or "")
                if _notif_uid and client_id:
                    _notif_creds = await cred_service.get_credentials(client_id=client_id, db_type=None, dataset_id=dataset_id)
                    _dataset_name = (_notif_creds.get("dataset_name") or dataset_id or "Dataset") if _notif_creds else (dataset_id or "Dataset")
                    await create_notification(Notification(
                        client_id=client_id,
                        user_id=_notif_uid,
                        type="db_config_completed",
                        title="Dataset Ready",
                        message=f'"{_dataset_name}" has been configured and is ready for analysis.',
                        metadata={"dataset_id": str(dataset_id or ""), "db_type": "file_upload"},
                        target_role="admin",
                    ))
            except Exception as _ne:
                logger.warning(f"Failed to send db_config_completed notification: {_ne}")
        except Exception as e:
            logger.error(f"Failed to update processing_status to complete: {e}", exc_info=True)
            if not credential_warning:
                credential_warning = "File processed but status could not be updated. Data may not persist after refresh."

        response = {
            "success": True,
            "client_id": client_id,
            "schema": schema_data,
            "explorer_limits": limit_metadata,
            "message": "File processed successfully. Parquet files created and AI metadata generated.",
        }
        if credential_warning:
            response["warning"] = credential_warning
        return response
        
    except HTTPException:
        # Reset processing_status to "error" so frontend stops polling
        try:
            db_err = await get_db()
            svc_err = DBCredentialsService(db_err)
            creds_err = await svc_err.get_credentials(
                client_id=client_id, db_type=None, dataset_id=dataset_id
            )
            if creds_err:
                params_err = creds_err.get("additional_params") or {}
                params_err["processing_status"] = "error"
                await svc_err.save_credentials(
                    client_id=client_id, db_type="file_upload",
                    db_host="", db_port=0, db_name="", db_username="",
                    db_password="", db_url="", additional_params=params_err,
                    dataset_id=creds_err.get("dataset_id"),
                )
        except Exception:
            pass
        raise
    except Exception as e:
        logger.error(f"Failed to process uploaded file: {str(e)}")
        # Reset processing_status to "error" so frontend stops polling
        try:
            db_err = await get_db()
            svc_err = DBCredentialsService(db_err)
            creds_err = await svc_err.get_credentials(
                client_id=client_id, db_type=None, dataset_id=dataset_id
            )
            if creds_err:
                params_err = creds_err.get("additional_params") or {}
                params_err["processing_status"] = "error"
                params_err["processing_error"] = str(e)
                await svc_err.save_credentials(
                    client_id=client_id, db_type="file_upload",
                    db_host="", db_port=0, db_name="", db_username="",
                    db_password="", db_url="", additional_params=params_err,
                    dataset_id=creds_err.get("dataset_id"),
                )
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


@router.post("/upload-multiple")
async def upload_files(
    client_id: str = Query(..., description="Client ID for file storage"),
    dataset_id: Optional[str] = Query(None, description="Dataset scope for uploads"),
    files: List[UploadFile] = File(...),
    admin_user: dict = Depends(require_admin),
):
    """
    Upload multiple CSV, Excel, Parquet, or ZIP files for schema exploration.
    Saves files and extracts basic schemas for each.
    """
    try:
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")

        # Empty upload folder before adding new files, then ensure directory exists
        _empty_upload_folder(client_id, dataset_id=dataset_id)
        upload_dir = assets_uploads_dir(client_id, dataset_id)
        upload_dir.mkdir(parents=True, exist_ok=True)

        is_super_admin = admin_user.get("role") == "super_admin"
        MAX_FILE_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 200 * 1024 * 1024))
        results = []

        for file in files:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in FileSchemaExtractor.SUPPORTED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {file_ext}. Supported: {', '.join(FileSchemaExtractor.SUPPORTED_FORMATS)}"
                )

            # Determine size
            file.file.seek(0, 2)
            file_size = file.file.tell()
            file.file.seek(0)

            # Enforce size limit (super_admin bypasses)
            if not is_super_admin and file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large: {file.filename}. Max {MAX_FILE_SIZE / (1024*1024):.0f}MB"
                )

            # Save streamed
            timestamp = utcnow().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}_{file.filename}"
            file_path = upload_dir / safe_filename

            CHUNK_SIZE = 1024 * 1024
            async with aiofiles.open(file_path, 'wb') as f:
                while chunk := await file.read(CHUNK_SIZE):
                    await f.write(chunk)

            # Verify write
            if not file_path.exists():
                raise HTTPException(status_code=500, detail=f"File upload incomplete: {file.filename}")

            # Sync to GCS if enabled
            await _sync_file_to_gcs(file_path, client_id, dataset_id, subfolder="uploads")

            # Extract schema
            try:
                extractor = FileSchemaExtractor(file_path)
                schema_data = await asyncio.to_thread(extractor.extract_schema)
            except ValueError as e:
                logger.error(f"Schema extraction failed for {file.filename}: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(
                    f"Unexpected error during schema extraction for {file.filename}: {str(e)}",
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process file '{file.filename}': {str(e)}",
                )

            results.append({
                "filename": file.filename,
                "file_size": file_size,
                "file_path": str(file_path),
                "schema": schema_data,
            })

        # Apply subscription plan limits to combined table list (one application across all files)
        all_tables = []
        for r in results:
            all_tables.extend(r["schema"].get("tables") or [])
        limits = await get_explorer_limits(client_id, role=admin_user.get("role"))
        trimmed_tables, limit_metadata = apply_table_column_limits(
            all_tables,
            limits.get("table_limit"),
            limits.get("column_limit"),
        )
        if not trimmed_tables:
            raise HTTPException(
                status_code=400,
                detail="No tables available to process under current plan limits. Please upgrade your plan."
            )
        aggregated_schema = {
            "tables": trimmed_tables,
            "total_tables": len(trimmed_tables),
            "source_type": "file",
            "extraction_time": utcnow().isoformat(),
        }

        # Audit one event summarizing
        await audit_logger.log_event(
            event_type=AuditEventType.FILE_UPLOAD,
            severity=AuditSeverity.INFO,
            user_id=admin_user.get('email', 'unknown'),
            client_id=client_id,
            details={
                'batch_count': len(results),
                'files': [r['filename'] for r in results],
                'total_tables': sum(r['schema'].get('total_tables', 0) for r in results)
            }
        )

        return {
            "success": True,
            "client_id": client_id,
            "files": results,
            "schema": aggregated_schema,
            "explorer_limits": limit_metadata,
            "message": f"Uploaded {len(results)} file(s) successfully."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")


@router.post("/remove-upload")
async def remove_uploaded_file(
    request: RemoveUploadRequest,
    client_id: str = Query(..., description="Client ID for file storage"),
    dataset_id: Optional[str] = Query(None, description="Dataset scope for uploads (must match upload path)"),
    admin_user: dict = Depends(require_admin)
):
    """
    Remove a previously uploaded file for a client.
    """
    try:
        uploads_dir = assets_uploads_dir(client_id, dataset_id)
        file_path = Path(request.file_path)

        is_valid, error_msg = validate_path_within_directory(file_path, uploads_dir)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        if not file_path.exists():
            return {
                "success": True,
                "message": "File already removed"
            }

        file_path.unlink()
        logger.info(f"Removed uploaded file for client {client_id}: {file_path}")
        return {
            "success": True,
            "message": "Uploaded file removed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove uploaded file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to remove uploaded file: {str(e)}")


@router.post("/process-uploads")
async def process_uploaded_files(
    request: ProcessUploadsRequest,
    dataset_id: Optional[str] = Query(None, description="Target dataset (required if multiple datasets exist)"),
    admin_user: dict = Depends(require_admin),
):
    """
    Process multiple uploaded files: Convert to Parquet and generate AI metadata once.
    Cleans previous datasets/XML once before processing the batch.
    """
    client_id: str = admin_user.get("client_id")
    try:
        items = request.files or []
        if not items:
            raise HTTPException(status_code=400, detail="No files provided to process")

        # Validate LLM config before processing
        db = await get_db()

        # Clean up once for the whole batch (preserving raw uploads)
        _cleanup_for_file_upload(client_id, dataset_id=dataset_id)

        # Apply subscription plan limits to combined table list before processing
        all_tables = []
        item_table_pairs = []  # (item, table) for each table
        for item in items:
            for table in (item.schema_info or {}).get("tables", []):
                all_tables.append(table)
                item_table_pairs.append((item, table))
        limits = await get_explorer_limits(client_id, role=admin_user.get("role"))
        trimmed_tables, limit_metadata = apply_table_column_limits(
            all_tables,
            limits.get("table_limit"),
            limits.get("column_limit"),
        )
        if not trimmed_tables:
            raise HTTPException(
                status_code=400,
                detail="No tables available to process under current plan limits. Please upgrade your plan."
            )

        # Save file upload config to db_credentials BEFORE heavy processing starts.
        # This ensures the UI shows "processing in progress" on page refresh instead of
        # showing a blank upload form while processing is still running.
        credential_warning = None
        try:
            cred_db = await get_db()
            cred_service = DBCredentialsService(cred_db)
            existing_cred = await cred_service.get_credentials(
                client_id=client_id, db_type=None, dataset_id=dataset_id, decrypt_password=False
            )
            if not existing_cred:
                raise RuntimeError("store_in_local not configured in DB Configs")
            configured_store_in_local = require_store_in_local(existing_cred)
            await cred_service.save_credentials(
                client_id=client_id,
                db_type="file_upload",
                db_host="",
                db_port=0,
                db_name="",
                db_username="",
                db_password="",
                db_url="",
                additional_params={
                    "file_count": len(items),
                    "total_tables": len(trimmed_tables),
                    "processed_tables": 0,
                    "processing_status": "processing"
                },
                store_in_local=configured_store_in_local,
                created_by=admin_user.get('email', 'unknown'),
                dataset_id=dataset_id,
            )
            logger.info(f"Saved file upload config (processing) in db_credentials for client {client_id}")
        except Exception as e:
            logger.error(f"Failed to save file upload config to db_credentials: {e}", exc_info=True)
            credential_warning = "Files processed successfully but configuration could not be saved. Data may not persist after refresh."

        # Build (item, table_trimmed) list preserving order of trimmed_tables
        tables_to_process = []  # list of (item, table_trimmed)
        seen = set()
        for t in trimmed_tables:
            key = (t.get("schema", ""), t.get("name", ""))
            if key in seen:
                continue
            seen.add(key)
            # Find (item, original_table) that matches this table
            for (item, orig) in item_table_pairs:
                if (orig.get("schema", ""), orig.get("name", "")) == key:
                    tables_to_process.append((item, t))
                    break

        parquet_dir = assets_datasets_dir(client_id, dataset_id)
        parquet_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified parquet directory: {parquet_dir}")

        import pandas as pd
        import zipfile
        import io
        import asyncio

        from explorer.models import TableMetadata

        # Excel read-once per file: cache {file_path: {sheet_name: df}}
        excel_cache: Dict[Path, Dict] = {}
        total_tables = 0
        aggregated_tables = []
        saved_files = []  # Track successfully saved files
        tables_metadata: List[TableMetadata] = []

        for (item, table) in tables_to_process:
            file_path = Path(item.file_path)
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            file_ext = file_path.suffix.lower()
            table_name = table["name"]
            df = None

            try:
                # Get sheet_name from table metadata if it's an Excel file
                sheet_name = table.get("sheet_name")

                if file_ext == ".csv":
                    try:
                        df = await asyncio.to_thread(pd.read_csv, file_path)
                    except UnicodeDecodeError:
                        logger.warning(f"UTF-8 decode failed for {file_path.name}, retrying with latin-1 encoding")
                        df = await asyncio.to_thread(pd.read_csv, file_path, encoding="latin-1")
                    logger.info(f"Read CSV file {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
                elif file_ext in [".xlsx", ".xls"]:
                    # Excel read-once per file: use cache
                    if file_path not in excel_cache:
                        engine = "openpyxl" if file_ext == ".xlsx" else "xlrd"
                        excel_all = await asyncio.to_thread(
                            pd.read_excel, file_path, sheet_name=None, engine=engine
                        )
                        excel_cache[file_path] = (
                            {None: excel_all} if isinstance(excel_all, pd.DataFrame) else excel_all
                        )
                        logger.info(f"Read Excel file {file_path.name} once: {len(excel_cache[file_path])} sheet(s)")
                    excel_sheets = excel_cache[file_path]
                    df = excel_sheets.get(sheet_name) if sheet_name is not None else excel_sheets.get(None)
                    if df is None and excel_sheets:
                        df = next(iter(excel_sheets.values()))
                    if df is not None:
                        logger.info(f"Using Excel sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")
                elif file_ext == ".parquet":
                    df = await asyncio.to_thread(pd.read_parquet, file_path)
                    logger.info(f"Read Parquet file {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
                elif file_ext == ".zip":
                    file_source = table.get("file_source")
                    if not file_source:
                        logger.warning(f"No file_source for table {table_name} in ZIP, skipping")
                        continue
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        with zip_ref.open(file_source) as zf:
                            file_bytes = io.BytesIO(zf.read())
                            source_ext = Path(file_source).suffix.lower()
                            if source_ext == ".csv":
                                try:
                                    df = await asyncio.to_thread(pd.read_csv, file_bytes)
                                except UnicodeDecodeError:
                                    logger.warning(f"UTF-8 decode failed for {file_source} in ZIP, retrying with latin-1 encoding")
                                    file_bytes.seek(0)
                                    df = await asyncio.to_thread(pd.read_csv, file_bytes, encoding="latin-1")
                            elif source_ext in [".xlsx", ".xls"]:
                                engine = "openpyxl" if source_ext == ".xlsx" else "xlrd"
                                if sheet_name is not None:
                                    df = await asyncio.to_thread(
                                        pd.read_excel, file_bytes, sheet_name=sheet_name, engine=engine
                                    )
                                    logger.info(f"Read Excel file {file_source} sheet '{sheet_name}' from ZIP {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
                                else:
                                    df = await asyncio.to_thread(
                                        pd.read_excel, file_bytes, engine=engine
                                    )
                                    logger.info(f"Read Excel file {file_source} from ZIP {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
                            elif source_ext == ".parquet":
                                df = await asyncio.to_thread(pd.read_parquet, file_bytes)
                            else:
                                logger.warning(f"Unsupported file in ZIP: {file_source}")
                                continue
                    if source_ext not in [".xlsx", ".xls"]:
                        logger.info(f"Read file {file_source} from ZIP {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
                else:
                    logger.warning(f"Unknown file extension: {file_ext}")
                    continue

                if df is None or df.empty:
                    logger.warning(f"DataFrame is empty for table {table_name}, skipping")
                    continue

                # Keep only columns that are in trimmed table schema
                col_names = [c.get("name") for c in (table.get("columns") or []) if c.get("name")]
                if col_names and set(col_names).issubset(set(df.columns)):
                    df = df[col_names]

                # Build TableMetadata from df (avoids re-reading Parquet later)
                table_meta = FileMetadataGenerator._build_table_metadata_from_df(df, table_name, max_sample_rows=100)
                tables_metadata.append(table_meta)

                if not parquet_dir.exists():
                    parquet_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Recreated parquet directory: {parquet_dir}")

                parquet_path = parquet_dir / f"{table_name}.parquet"
                FileSchemaExtractor.save_as_parquet(df, parquet_path)

                if parquet_path.exists() and parquet_path.stat().st_size > 0:
                    logger.info(f"Successfully saved Parquet file: {parquet_path} ({parquet_path.stat().st_size} bytes)")
                    saved_files.append(str(parquet_path))
                    await _sync_file_to_gcs(parquet_path, client_id, dataset_id, subfolder="datasets")
                else:
                    logger.error(f"Failed to save Parquet file: {parquet_path} (file does not exist or is empty)")
                    raise Exception(f"Failed to save parquet file for table {table_name}")

                aggregated_tables.append(table)
                total_tables += 1
            except Exception as e:
                logger.error(f"Error processing table {table_name} from file {file_path.name}: {e}", exc_info=True)
                continue

        logger.info(f"Successfully processed {total_tables} tables. Saved {len(saved_files)} parquet files to {parquet_dir}")
        if not saved_files:
            raise HTTPException(
                status_code=400,
                detail="No parquet files were generated from uploaded files. Please verify file contents and plan limits."
            )

        # Generate AI metadata once for the batch (use pre-built tables_metadata, no Parquet re-read)
        try:
            logger.info(f"Generating AI metadata for client {client_id} (batch)")
            output_root = Path("xml_prompts/clients") / client_id
            metadata_generator = FileMetadataGenerator(
                client_id=client_id, output_root=output_root, db=db, dataset_id=dataset_id
            )

            _processed_count = 0

            async def _on_table_done_uploads() -> None:
                nonlocal _processed_count
                _processed_count += 1
                try:
                    _prog_db = await get_db()
                    _prog_svc = DBCredentialsService(_prog_db)
                    await _prog_svc.update_processing_progress(
                        client_id, _processed_count, dataset_id=dataset_id
                    )
                except Exception as _prog_err:
                    logger.warning(f"Failed to update processing progress for client {client_id}: {_prog_err}")

            metadata_result = await metadata_generator.generate_metadata_from_tables(tables_metadata, on_table_done=_on_table_done_uploads)
            logger.info(f"AI metadata generation complete: {metadata_result}")

            # Persist explorer_limits so GET /metadata can return it
            meta_dir = metadata_generator.data_sources_root / "meta_information"
            meta_dir.mkdir(parents=True, exist_ok=True)
            explorer_limits_path = meta_dir / "explorer_limits.json"
            limits_json = json.dumps(limit_metadata, indent=2)
            await asyncio.to_thread(explorer_limits_path.write_text, limits_json, "utf-8")

            # Generate suggested questions after metadata is created
            try:
                logger.info(f"Generating suggested questions for client {client_id} (batch)")
                from explorer.question_generator import QuestionGenerator
                question_generator = QuestionGenerator(client_id=client_id, db=db, dataset_id=dataset_id)
                questions = await question_generator.generate_questions(count=30)
                logger.info(f"Generated {len(questions)} suggested questions for client {client_id}")
            except Exception as qe:
                logger.warning(f"Failed to generate suggested questions: {str(qe)}", exc_info=True)
                # Don't fail the whole process if question generation fails
            
            # Copy/merge base prompts (planner.xml, python.xml, etc.) for this client
            try:
                await copy_base_prompts_for_client(client_id, output_root)
                logger.info(f"Copied base prompts for client {client_id} (batch file upload)")
            except Exception as bp_err:
                logger.warning(f"Failed to copy base prompts for client {client_id}: {bp_err}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to generate AI metadata: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to generate AI metadata: {str(e)}")

        logger.info(f"Batch file processing complete for client {client_id}")

        # Update processing_status to "complete" after successful processing
        try:
            cred_db = await get_db()
            cred_service = DBCredentialsService(cred_db)
            existing_cred = await cred_service.get_credentials(
                client_id=client_id, db_type=None, dataset_id=dataset_id, decrypt_password=False
            )
            if not existing_cred:
                raise RuntimeError("store_in_local not configured in DB Configs")
            configured_store_in_local = require_store_in_local(existing_cred)
            await cred_service.save_credentials(
                client_id=client_id,
                db_type="file_upload",
                db_host="",
                db_port=0,
                db_name="",
                db_username="",
                db_password="",
                db_url="",
                additional_params={
                    "file_count": len(items),
                    "total_tables": total_tables,
                    "processing_status": "complete"
                },
                store_in_local=configured_store_in_local,
                created_by=admin_user.get('email', 'unknown'),
                dataset_id=dataset_id,
            )
            logger.info(f"Updated file upload config to complete in db_credentials for client {client_id}")
            # Notify admin that the uploaded dataset is ready for analysis
            try:
                from notifications.notification_service import create_notification
                from notifications.notification_model import Notification
                _notif_uid = str(admin_user.get("user_id") or admin_user.get("_id") or "")
                if _notif_uid and client_id:
                    _notif_creds = await cred_service.get_credentials(client_id=client_id, db_type=None, dataset_id=dataset_id)
                    _dataset_name = (_notif_creds.get("dataset_name") or dataset_id or "Dataset") if _notif_creds else (dataset_id or "Dataset")
                    await create_notification(Notification(
                        client_id=client_id,
                        user_id=_notif_uid,
                        type="db_config_completed",
                        title="Dataset Ready",
                        message=f'"{_dataset_name}" has been configured and is ready for analysis.',
                        metadata={"dataset_id": str(dataset_id or ""), "db_type": "file_upload", "file_count": len(items)},
                        target_role="admin",
                    ))
            except Exception as _ne:
                logger.warning(f"Failed to send db_config_completed notification: {_ne}")
        except Exception as e:
            logger.error(f"Failed to update processing_status to complete: {e}", exc_info=True)
            if not credential_warning:
                credential_warning = "Files processed but status could not be updated. Data may not persist after refresh."

        aggregated_schema = {
            "total_tables": total_tables,
            "tables": aggregated_tables,
        }

        response = {
            "success": True,
            "client_id": client_id,
            "schema": aggregated_schema,
            "explorer_limits": limit_metadata,
            "message": f"Processed {len(items)} file(s). Parquet files created and AI metadata generated.",
        }
        if credential_warning:
            response["warning"] = credential_warning
        return response
    except HTTPException:
        # Reset processing_status to "error" so frontend stops polling
        try:
            db_err = await get_db()
            svc_err = DBCredentialsService(db_err)
            creds_err = await svc_err.get_credentials(
                client_id=client_id, db_type=None, dataset_id=dataset_id
            )
            if creds_err:
                params_err = creds_err.get("additional_params") or {}
                params_err["processing_status"] = "error"
                await svc_err.save_credentials(
                    client_id=client_id, db_type="file_upload",
                    db_host="", db_port=0, db_name="", db_username="",
                    db_password="", db_url="", additional_params=params_err,
                    dataset_id=creds_err.get("dataset_id"),
                )
        except Exception:
            pass
        raise
    except Exception as e:
        logger.error(f"Failed to process uploaded files: {str(e)}", exc_info=True)
        # Reset processing_status to "error" so frontend stops polling
        try:
            db_err = await get_db()
            svc_err = DBCredentialsService(db_err)
            creds_err = await svc_err.get_credentials(
                client_id=client_id, db_type=None, dataset_id=dataset_id
            )
            if creds_err:
                params_err = creds_err.get("additional_params") or {}
                params_err["processing_status"] = "error"
                params_err["processing_error"] = str(e)
                await svc_err.save_credentials(
                    client_id=client_id, db_type="file_upload",
                    db_host="", db_port=0, db_name="", db_username="",
                    db_password="", db_url="", additional_params=params_err,
                    dataset_id=creds_err.get("dataset_id"),
                )
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@router.post("/explore-file")
async def explore_file(
    client_id: str = Query(..., description="Client ID"),
    background_tasks: BackgroundTasks = None,
    admin_user: dict = Depends(require_admin)
):
    """
    Run Explorer Agent on uploaded file data (CSV/Excel/Parquet).
    Uses previously uploaded and converted Parquet files.
    """
    try:
        # Check if client has uploaded files
        parquet_dir = Path(f"assets/clients/{client_id}/datasets")
        if not parquet_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No data found for client {client_id}. Please upload files first."
            )
        
        parquet_files = list(parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise HTTPException(
                status_code=404,
                detail=f"No parquet files found for client {client_id}. Please upload files first."
            )
        
        # Build schema structure from parquet files
        import pandas as pd
        tables = []
        
        for parquet_file in parquet_files:
            df = await asyncio.to_thread(pd.read_parquet, parquet_file)
            table_name = parquet_file.stem
            
            columns = []
            for col in df.columns:
                dtype = df[col].dtype
                sql_type = 'VARCHAR'
                if 'int' in str(dtype):
                    sql_type = 'INTEGER'
                elif 'float' in str(dtype):
                    sql_type = 'FLOAT'
                elif 'bool' in str(dtype):
                    sql_type = 'BOOLEAN'
                elif 'datetime' in str(dtype):
                    sql_type = 'TIMESTAMP'
                
                columns.append({
                    'name': col,
                    'type': sql_type,
                    'nullable': df[col].isnull().any(),
                    'primary_key': False
                })
            
            tables.append({
                'name': table_name,
                'schema': 'public',
                'columns': columns,
                'column_count': len(columns),
                'row_count': len(df)
            })
        
        # Apply subscription plan table/column limits
        limits = await get_explorer_limits(client_id, role=admin_user.get("role"))
        trimmed_tables, limit_metadata = apply_table_column_limits(
            tables,
            limits.get("table_limit"),
            limits.get("column_limit"),
        )
        exploration_result = {
            'tables': trimmed_tables,
            'total_tables': len(trimmed_tables),
            'source_type': 'file',
            'client_id': client_id,
            'explorer_limits': limit_metadata,
        }
        
        # Trigger XML generation from parquet files
        # The explorer agent will read the parquet files and generate metadata
        output_root = Path("xml_prompts/clients") / client_id
        output_root.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"File-based exploration complete for client {client_id}: {len(trimmed_tables)} tables")
        
        # Audit log
        await audit_logger.log_event(
            event_type=AuditEventType.EXPLORATION_RUN,
            severity=AuditSeverity.INFO,
            user_id=admin_user.get('email', 'unknown'),
            client_id=client_id,
            details={
                'source_type': 'file',
                'tables_count': len(trimmed_tables)
            }
        )
        
        return {
            "success": True,
            "message": "File-based exploration completed successfully",
            "result": exploration_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to explore files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File exploration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ERD extraction failed: {str(e)}")
