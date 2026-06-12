import logging
import asyncio
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from bson import ObjectId
from cryptography.fernet import Fernet
import os
import base64
from motor.motor_asyncio import AsyncIOMotorClient
from util.time_utils import utcnow
from util.dataset_paths import normalize_credential_dict
from util.data_source import require_store_in_local
logger = logging.getLogger(__name__)

class DBCredentialsService:
    DB_TYPES = ['postgres', 'mysql', 'mongodb', 'sqlserver', 'sap_oracle', 'sap_hana', 'sap_sybase']

    def __init__(self, db):
        self.db = db
        if db is not None:
            self.collection = db.db_credentials
        else:
            self.collection = None
        encryption_key = os.getenv('DB_CREDENTIALS_ENCRYPTION_KEY')
        if not encryption_key:
            raise RuntimeError('DB_CREDENTIALS_ENCRYPTION_KEY environment variable is not set. This is required for encrypting/decrypting database credentials. Generate one with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"')
        try:
            self.cipher = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
        except Exception as e:
            logger.error(f'Failed to initialize encryption: {e}')
            raise ValueError('DB_CREDENTIALS_ENCRYPTION_KEY is required ')

    def _ensure_collection(self, operation: str, *, allow_missing: bool=False) -> bool:
        if self.collection is not None:
            return True
        message = f'DB credentials collection is unavailable during {operation}. Mongo database handle was not provided to DBCredentialsService.'
        if allow_missing:
            logger.warning(message)
            return False
        raise RuntimeError(message)

    def _encrypt_password(self, password: str) -> str:
        try:
            encrypted = self.cipher.encrypt(password.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f'Encryption failed: {e}')
            raise ValueError('Failed to encrypt password')

    def _decrypt_password(self, encrypted_password: str) -> str:
        try:
            decoded = base64.b64decode(encrypted_password.encode())
            decrypted = self.cipher.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f'Decryption failed: {e}')
            raise ValueError('Failed to decrypt password')

    async def _next_display_order(self, client_id: str) -> int:
        last = await self.collection.find({'client_id': client_id}).sort('display_order', -1).limit(1).to_list(1)
        if not last:
            return 0
        return int(last[0].get('display_order', 0)) + 1

    def _doc_to_result(self, doc: Dict[str, Any], *, decrypt_password: bool, password_override: Optional[str]=None) -> Dict[str, Any]:
        password = password_override
        if password is None and decrypt_password and ('db_password_encrypted' in doc):
            encrypted_pwd = doc['db_password_encrypted']
            if doc.get('db_type') == 'file_upload' or not encrypted_pwd or (isinstance(encrypted_pwd, str) and encrypted_pwd.strip() == ''):
                password = ''
            else:
                try:
                    password = self._decrypt_password(encrypted_pwd)
                except Exception as e:
                    logger.error('Failed to decrypt password for client %s: %s', doc.get('client_id'), e)
                    password = None
        store_in_local = require_store_in_local(doc)
        result = {'credential_id': str(doc['_id']), 'client_id': doc['client_id'], 'dataset_id': doc.get('dataset_id'), 'dataset_name': doc.get('dataset_name') or doc.get('db_name') or doc.get('db_type') or 'Dataset', 'is_enabled': doc.get('is_enabled', True), 'display_order': doc.get('display_order', 0), 'db_type': doc['db_type'], 'additional_params': doc.get('additional_params', {}), 'store_in_local': store_in_local, 'created_at': doc['created_at'].isoformat() if isinstance(doc['created_at'], datetime) else doc['created_at'], 'updated_at': doc['updated_at'].isoformat() if isinstance(doc['updated_at'], datetime) else doc['updated_at'], 'created_by': doc.get('created_by', 'unknown'), 'updated_by': doc.get('updated_by', 'unknown')}
        if doc['db_type'] != 'file_upload':
            result.update({'db_host': doc.get('db_host', ''), 'db_port': doc.get('db_port', 0), 'db_name': doc.get('db_name', ''), 'db_username': doc.get('db_username', ''), 'db_password': password if decrypt_password else None, 'db_url': doc.get('db_url', '')})
        return normalize_credential_dict(result)

    async def save_credentials(self, client_id: str, db_type: str, db_host: str, db_port: int, db_name: str, db_username: str, db_password: str, db_url: str, store_in_local: bool, additional_params: Optional[Dict[str, Any]]=None, created_by: str='system', dataset_name: Optional[str]=None, dataset_id: Optional[str]=None) -> Dict[str, Any]:
        try:
            self._ensure_collection('save_credentials')
            allowed_types = self.DB_TYPES + ['file_upload']
            if db_type not in allowed_types:
                raise ValueError(f"Unsupported database type: {db_type}. Supported: {', '.join(allowed_types)}")
            if db_type == 'file_upload' or not db_password or db_password.strip() == '':
                encrypted_password = ''
            else:
                encrypted_password = self._encrypt_password(db_password)
            now = utcnow()
            add_params = additional_params or {}
            if dataset_id:
                existing = await self.collection.find_one({'client_id': client_id, 'dataset_id': dataset_id})
                if not existing:
                    raise ValueError(f'No dataset found for dataset_id={dataset_id}')
                if db_type == 'file_upload':
                    update_doc = {'$set': {'db_type': db_type, 'additional_params': add_params, 'store_in_local': store_in_local, 'updated_at': now, 'updated_by': created_by}, '$unset': {'db_host': '', 'db_port': '', 'db_name': '', 'db_username': '', 'db_password_encrypted': '', 'db_url': ''}}
                else:
                    update_doc = {'$set': {'db_type': db_type, 'db_host': db_host, 'db_port': db_port, 'db_name': db_name, 'db_username': db_username, 'db_password_encrypted': encrypted_password, 'db_url': db_url, 'additional_params': add_params, 'store_in_local': store_in_local, 'updated_at': now, 'updated_by': created_by}}
                await self.collection.update_one({'_id': existing['_id']}, update_doc)
                doc = await self.collection.find_one({'_id': existing['_id']})
                return self._doc_to_result(doc, decrypt_password=False)
            if dataset_name is not None and str(dataset_name).strip() != '':
                name_clean = str(dataset_name).strip()
                dup = await self.collection.find_one({'client_id': client_id, 'dataset_name': name_clean})
                if dup:
                    raise ValueError('DUPLICATE_DATASET_NAME')
                new_dataset_id = str(uuid.uuid4())
                if db_type == 'file_upload':
                    doc = {'client_id': client_id, 'dataset_id': new_dataset_id, 'dataset_name': name_clean, 'is_enabled': True, 'display_order': await self._next_display_order(client_id), 'db_type': db_type, 'additional_params': add_params, 'store_in_local': store_in_local, 'created_at': now, 'created_by': created_by, 'updated_at': now, 'updated_by': created_by}
                else:
                    doc = {'client_id': client_id, 'dataset_id': new_dataset_id, 'dataset_name': name_clean, 'is_enabled': True, 'display_order': await self._next_display_order(client_id), 'db_type': db_type, 'db_host': db_host, 'db_port': db_port, 'db_name': db_name, 'db_username': db_username, 'db_password_encrypted': encrypted_password, 'db_url': db_url, 'additional_params': add_params, 'store_in_local': store_in_local, 'created_at': now, 'created_by': created_by, 'updated_at': now, 'updated_by': created_by}
                ins = await self.collection.insert_one(doc)
                doc = await self.collection.find_one({'_id': ins.inserted_id})
                logger.info('Inserted new dataset %s for client %s', new_dataset_id, client_id)
                return self._doc_to_result(doc, decrypt_password=False)
            count = await self.collection.count_documents({'client_id': client_id})
            if count > 1:
                raise ValueError('dataset_id is required when multiple datasets exist for this client')
            existing = await self.collection.find_one({'client_id': client_id})
            if existing:
                if db_type == 'file_upload':
                    update_doc = {'$set': {'db_type': db_type, 'additional_params': add_params, 'store_in_local': store_in_local, 'updated_at': now, 'updated_by': created_by}, '$unset': {'db_host': '', 'db_port': '', 'db_name': '', 'db_username': '', 'db_password_encrypted': '', 'db_url': ''}}
                else:
                    update_doc = {'$set': {'db_type': db_type, 'db_host': db_host, 'db_port': db_port, 'db_name': db_name, 'db_username': db_username, 'db_password_encrypted': encrypted_password, 'db_url': db_url, 'additional_params': add_params, 'store_in_local': store_in_local, 'updated_at': now, 'updated_by': created_by}}
                if not existing.get('dataset_id'):
                    update_doc['$set']['dataset_id'] = str(uuid.uuid4())
                if not existing.get('dataset_name'):
                    update_doc['$set']['dataset_name'] = db_name or db_type or 'Default Dataset'
                if 'is_enabled' not in existing:
                    update_doc['$set']['is_enabled'] = True
                if 'display_order' not in existing:
                    update_doc['$set']['display_order'] = 0
                await self.collection.update_one({'_id': existing['_id']}, update_doc)
                doc = await self.collection.find_one({'_id': existing['_id']})
                logger.info('Updated legacy DB credentials for client %s', client_id)
                return self._doc_to_result(doc, decrypt_password=False)
            new_dataset_id = str(uuid.uuid4())
            ds_name = db_name or db_type or 'Default Dataset'
            if db_type == 'file_upload':
                doc = {'client_id': client_id, 'dataset_id': new_dataset_id, 'dataset_name': ds_name, 'is_enabled': True, 'display_order': 0, 'db_type': db_type, 'additional_params': add_params, 'store_in_local': store_in_local, 'created_at': now, 'created_by': created_by, 'updated_at': now, 'updated_by': created_by}
            else:
                doc = {'client_id': client_id, 'dataset_id': new_dataset_id, 'dataset_name': ds_name, 'is_enabled': True, 'display_order': 0, 'db_type': db_type, 'db_host': db_host, 'db_port': db_port, 'db_name': db_name, 'db_username': db_username, 'db_password_encrypted': encrypted_password, 'db_url': db_url, 'additional_params': add_params, 'store_in_local': store_in_local, 'created_at': now, 'created_by': created_by, 'updated_at': now, 'updated_by': created_by}
            ins = await self.collection.insert_one(doc)
            doc = await self.collection.find_one({'_id': ins.inserted_id})
            logger.info('Saved first DB credentials for client %s dataset_id=%s', client_id, new_dataset_id)
            return self._doc_to_result(doc, decrypt_password=False)
        except Exception as e:
            logger.error(f'Failed to save credentials for client {client_id}: {e}')
            raise

    async def get_credentials(self, client_id: str, db_type: Optional[str]=None, decrypt_password: bool=True, dataset_id: Optional[str]=None) -> Optional[Dict[str, Any]]:
        return {'credential_id': 'env_override', 'client_id': client_id, 'db_type': 'postgres', 'db_host': os.getenv('REDSHIFT_HOST', ''), 'db_port': int(os.getenv('REDSHIFT_PORT', 5439)), 'db_name': os.getenv('REDSHIFT_DB', ''), 'db_username': os.getenv('REDSHIFT_USER', ''), 'db_password': os.getenv('REDSHIFT_PASSWORD', ''), 'db_url': '', 'store_in_local': False, 'additional_params': {}}

    async def get_active_datasets(self, client_id: str) -> List[Dict[str, Any]]:
        if not self._ensure_collection('get_active_datasets', allow_missing=True):
            return []
        cursor = self.collection.find({'client_id': client_id, 'is_enabled': True, '$or': [{'additional_params.processing_status': {'$exists': False}}, {'additional_params.processing_status': {'$in': [None, '', 'complete']}}]}).sort([('display_order', 1), ('created_at', 1)])
        out: List[Dict[str, Any]] = []
        async for doc in cursor:
            out.append({'dataset_id': doc.get('dataset_id'), 'dataset_name': doc.get('dataset_name') or doc.get('db_name') or doc.get('db_type'), 'db_type': doc.get('db_type'), 'store_in_local': require_store_in_local(doc)})
        return out

    async def count_enabled_datasets(self, client_id: str) -> int:
        if not self._ensure_collection('count_enabled_datasets', allow_missing=True):
            return 0
        return await self.collection.count_documents({'client_id': client_id, 'is_enabled': True})

    async def list_credentials(self, client_id: str, include_password: bool=False, enabled_only: bool=False) -> List[Dict[str, Any]]:
        try:
            if not self._ensure_collection('list_credentials', allow_missing=True):
                return []
            query: Dict[str, Any] = {'client_id': client_id}
            if enabled_only:
                query['is_enabled'] = True
            cursor = self.collection.find(query).sort([('display_order', 1), ('created_at', 1)])
            credentials: List[Dict[str, Any]] = []
            async for doc in cursor:
                pwd = None
                if include_password and doc.get('db_type') != 'file_upload':
                    enc = doc.get('db_password_encrypted')
                    if enc and isinstance(enc, str) and enc.strip():
                        try:
                            pwd = self._decrypt_password(enc)
                        except Exception as e:
                            logger.error('Failed to decrypt password for credential %s: %s', doc['_id'], e)
                credentials.append(self._doc_to_result(doc, decrypt_password=include_password, password_override=pwd))
            return credentials
        except Exception as e:
            logger.error(f'Failed to list credentials for client {client_id}: {e}')
            raise

    async def update_store_in_local(self, client_id: str, db_type: str, store_in_local: bool, dataset_id: Optional[str]=None) -> bool:
        try:
            self._ensure_collection('update_store_in_local')
            filt: Dict[str, Any] = {'client_id': client_id, 'db_type': db_type}
            if dataset_id:
                filt['dataset_id'] = dataset_id
            result = await self.collection.update_one(filt, {'$set': {'store_in_local': store_in_local, 'updated_at': utcnow()}})
            return result.modified_count > 0
        except Exception as e:
            logger.error(f'Failed to update store_in_local for client {client_id}: {e}')
            raise

    async def delete_credentials(self, client_id: str, dataset_id: str) -> bool:
        try:
            self._ensure_collection('delete_credentials')
            doc = await self.collection.find_one({'client_id': client_id, 'dataset_id': dataset_id})
            if not doc:
                return False
            if doc.get('is_enabled', True):
                enabled = await self.count_enabled_datasets(client_id)
                if enabled <= 1:
                    raise ValueError('LAST_ENABLED_DATASET')
            result = await self.collection.delete_one({'client_id': client_id, 'dataset_id': dataset_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f'Failed to delete credentials for client {client_id}: {e}')
            raise

    async def toggle_dataset(self, client_id: str, dataset_id: str, is_enabled: bool) -> bool:
        self._ensure_collection('toggle_dataset')
        doc = await self.collection.find_one({'client_id': client_id, 'dataset_id': dataset_id})
        if not doc:
            return False
        if doc.get('is_enabled', True) and (not is_enabled):
            if await self.count_enabled_datasets(client_id) <= 1:
                raise ValueError('LAST_ENABLED_DATASET')
        await self.collection.update_one({'_id': doc['_id']}, {'$set': {'is_enabled': is_enabled, 'updated_at': utcnow()}})
        return True

    async def update_dataset_name(self, client_id: str, dataset_id: str, new_name: str) -> bool:
        name_clean = str(new_name).strip()
        if not name_clean:
            raise ValueError('dataset_name cannot be empty')
        self._ensure_collection('update_dataset_name')
        dup = await self.collection.find_one({'client_id': client_id, 'dataset_name': name_clean, 'dataset_id': {'$ne': dataset_id}})
        if dup:
            raise ValueError('DUPLICATE_DATASET_NAME')
        result = await self.collection.update_one({'client_id': client_id, 'dataset_id': dataset_id}, {'$set': {'dataset_name': name_clean, 'updated_at': utcnow()}})
        return result.modified_count > 0

    async def update_processing_progress(self, client_id: str, processed_tables: int, dataset_id: Optional[str]=None) -> None:
        if not self._ensure_collection('update_processing_progress', allow_missing=True):
            return
        filt: Dict[str, Any] = {'client_id': client_id}
        if dataset_id:
            filt['dataset_id'] = dataset_id
        await self.collection.update_one(filt, {'$set': {'additional_params.processed_tables': processed_tables, 'updated_at': utcnow()}})

    async def delete_credentials_for_client(self, client_id: str) -> int:
        try:
            self._ensure_collection('delete_credentials_for_client')
            result = await self.collection.delete_many({'client_id': client_id})
            if result.deleted_count > 0:
                logger.info('Deleted %s credential doc(s) for client %s', result.deleted_count, client_id)
            return int(result.deleted_count)
        except Exception as e:
            logger.error(f'Failed to delete credentials for client {client_id}: {e}')
            raise

    async def test_connection(self, db_type: str, db_host: str, db_port: int, db_name: str, db_username: str, db_password: str, db_url: Optional[str]=None, ssh_config: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        try:
            if db_type == 'postgres':
                from db_config.connectors.postgres_connector import PostgresConnector
                dsn = db_url or f'postgresql+asyncpg://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
                connector = PostgresConnector(dsn, ssh_config=ssh_config)
                try:
                    await connector.connect()
                    await connector.disconnect()
                    return {'success': True, 'message': 'Connection successful', 'db_type': db_type, 'db_host': db_host, 'db_name': db_name}
                except Exception as e:
                    await connector.disconnect()
                    return {'success': False, 'message': f'Connection failed: {str(e)}', 'db_type': db_type, 'db_host': db_host, 'db_name': db_name}
            elif db_type == 'mysql':
                from db_config.connectors.mysql_connector import MySQLConnector
                dsn = db_url or f'mysql+aiomysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
                connector = MySQLConnector(dsn, ssh_config=ssh_config)
                try:
                    await connector.connect()
                    await connector.disconnect()
                    return {'success': True, 'message': 'Connection successful', 'db_type': db_type, 'db_host': db_host, 'db_name': db_name}
                except Exception as e:
                    await connector.disconnect()
                    return {'success': False, 'message': f'Connection failed: {str(e)}', 'db_type': db_type, 'db_host': db_host, 'db_name': db_name}
            elif db_type == 'mongodb':
                from db_config.connectors.mongo_connector import MongoConnector
                if db_url and ('mongodb+srv' in db_url or 'mongodb.net' in db_url):
                    uri = db_url
                else:
                    uri = f'mongodb+srv://{db_username}:{db_password}@{db_host}/{db_name}'
                connector = MongoConnector(uri, db_name, ssh_config=ssh_config)
                try:
                    await connector.connect()
                    db = connector.get_db()
                    await db.client.admin.command('ping')
                    await connector.disconnect()
                    return {'success': True, 'message': 'Connection successful', 'db_type': db_type, 'db_host': db_host, 'db_name': db_name}
                except Exception as e:
                    await connector.disconnect()
                    return {'success': False, 'message': f'Connection failed: {str(e)}', 'db_type': db_type, 'db_host': db_host, 'db_name': db_name}
            elif db_type == 'sap_hana':
                try:
                    from hdbcli import dbapi
                    if db_url:
                        import re
                        match = re.search('hana://.+:.+@(.+):(\\d+)', db_url)
                        if match:
                            host, port = match.groups()
                        else:
                            return {'success': False, 'message': 'Invalid SAP HANA URL format', 'db_type': db_type}
                    else:
                        host = db_host
                        port = db_port

                    def _test_hana():
                        conn = dbapi.connect(address=host, port=int(port), user=db_username, password=db_password)
                        conn.close()
                    await asyncio.to_thread(_test_hana)
                    return {'success': True, 'message': 'Successfully connected to SAP HANA database', 'db_type': db_type, 'db_host': host, 'db_name': db_name}
                except Exception as e:
                    logger.error(f'SAP HANA connection test failed: {e}')
                    return {'success': False, 'message': f'SAP HANA connection failed: {str(e)}', 'db_type': db_type, 'db_host': db_host, 'db_name': db_name}
            elif db_type == 'sap_oracle':
                from db_config.connectors.oracle_connector import OracleConnector
                dsn = db_url or f'oracle+oracledb_async://{db_username}:{db_password}@{db_host}:{db_port}/?service_name={db_name}'
                connector = OracleConnector(dsn, ssh_config=ssh_config)
                try:
                    await connector.connect()
                    await connector.disconnect()
                    return {'success': True, 'message': 'Connection successful'}
                except Exception as e:
                    await connector.disconnect()
                    return {'success': False, 'message': f'Connection failed: {str(e)}', 'db_type': db_type, 'db_host': db_host, 'db_name': db_name}
            elif db_type == 'sap_sybase':
                from sqlalchemy import create_engine, text
                from db_config.connectors.sybase_connector import build_sybase_url
                import pyodbc
                odbc_conn_str = build_sybase_url(db_host, db_port, db_username, db_password, db_name)

                def pyodbc_creator():
                    return pyodbc.connect(odbc_conn_str)
                engine = create_engine('sybase+pyodbc://', creator=pyodbc_creator)
                try:
                    connection = engine.connect()
                    try:
                        connection.execute(text('SELECT 1'))
                    finally:
                        connection.close()
                    engine.dispose()
                    return {'success': True, 'message': 'Connection successful'}
                except Exception as e:
                    engine.dispose()
                    return {'success': False, 'message': f'Connection failed: {str(e)}', 'db_type': db_type, 'db_host': db_host, 'db_name': db_name}
            else:
                return {'success': False, 'message': f'Connection testing not implemented for {db_type}', 'db_type': db_type}
        except Exception as e:
            logger.error(f'Connection test failed: {e}')
            return {'success': False, 'message': f'Connection test error: {str(e)}', 'db_type': db_type}