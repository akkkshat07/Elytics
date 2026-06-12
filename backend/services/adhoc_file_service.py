from __future__ import annotations
import logging
import os
import shutil
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from config.system_config import ADHOC_FILE_CONFIG
from util.file_security import sanitize_filename, validate_path_within_directory
from services.session_memory import session_memory
logger = logging.getLogger(__name__)
CLIENTS_BASE_DIR = Path(ADHOC_FILE_CONFIG['base_dir'])
ADHOC_SUBDIR = 'adhoc_uploads'
MAX_SIZE_BYTES = ADHOC_FILE_CONFIG['max_file_size_mb'] * 1024 * 1024
ALLOWED_EXTENSIONS = set(ADHOC_FILE_CONFIG['allowed_extensions'])
FILE_TTL_HOURS = ADHOC_FILE_CONFIG['file_ttl_hours']

def _session_dir(client_id: str, session_id: str) -> Path:
    path = CLIENTS_BASE_DIR / client_id / ADHOC_SUBDIR / session_id
    is_valid, error = validate_path_within_directory(path, CLIENTS_BASE_DIR)
    if not is_valid:
        raise AdhocFileError(f'Invalid session path: {error}')
    return path

def _storage_session_prefix(client_id: str, session_id: str) -> str:
    from config.system_config import STORAGE_BACKEND
    if STORAGE_BACKEND == 'gcs':
        return f'clients/{client_id}/adhoc_uploads/{session_id}'
    return str(_session_dir(client_id, session_id))

class AdhocFileError(Exception):
    pass

async def upload_file(file_content: bytes, original_filename: str, session_id: str, client_id: str) -> Dict[str, Any]:
    ext = Path(original_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise AdhocFileError(f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    if len(file_content) > MAX_SIZE_BYTES:
        raise AdhocFileError(f"File too large ({len(file_content) / (1024 * 1024):.1f} MB). Maximum: {ADHOC_FILE_CONFIG['max_file_size_mb']} MB.")
    if len(file_content) == 0:
        raise AdhocFileError('File is empty.')
    await delete_file(session_id, client_id)
    safe_name = sanitize_filename(original_filename)
    session_dir = _session_dir(client_id, session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    if ext in ('.xlsx', '.xls'):
        file_paths, file_names = await _convert_excel_to_csv(file_content, safe_name, session_dir)
    else:
        csv_path = session_dir / safe_name
        csv_path.write_bytes(file_content)
        stem = Path(safe_name).stem
        file_paths = [str(csv_path)]
        file_names = [stem]
    metadata = {'original_filename': original_filename, 'file_paths': file_paths, 'file_names': file_names, 'file_size_bytes': len(file_content), 'uploaded_at': time.time(), 'sheet_count': len(file_paths), 'client_id': client_id, 'session_id': session_id}
    from config.system_config import STORAGE_BACKEND
    if STORAGE_BACKEND == 'gcs':
        from util.storage.backend import get_storage_backend
        storage = get_storage_backend()
        prefix = _storage_session_prefix(client_id, session_id)
        for fp in file_paths:
            local_file = Path(fp)
            remote_key = f'{prefix}/{local_file.name}'
            await storage.upload_file(str(local_file), remote_key)
        metadata['storage_paths'] = [f'{prefix}/{Path(fp).name}' for fp in file_paths]
    session_memory.set_adhoc_file(session_id, metadata)
    logger.info('Ad-hoc file uploaded | session=%s | client=%s | file=%s | size=%d | sheets=%d', session_id, client_id, original_filename, len(file_content), len(file_paths))
    return metadata

async def _convert_excel_to_csv(content: bytes, safe_name: str, session_dir: Path) -> tuple[List[str], List[str]]:
    import pandas as pd
    import tempfile
    tmp_path = session_dir / safe_name
    tmp_path.write_bytes(content)
    try:
        xls = await asyncio.to_thread(pd.ExcelFile, tmp_path)
        sheet_names = xls.sheet_names
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise AdhocFileError(f'Failed to read Excel file: {e}')
    file_paths = []
    file_names = []
    try:
        for sheet in sheet_names:
            try:
                df = await asyncio.to_thread(pd.read_excel, xls, sheet_name=sheet)
            except Exception as e:
                logger.warning("Skipping sheet '%s': %s", sheet, e)
                continue
            if df.empty:
                logger.warning("Skipping empty sheet '%s'", sheet)
                continue
            csv_name = sanitize_filename(f'{sheet}.csv') if len(sheet_names) > 1 else Path(safe_name).stem + '.csv'
            csv_path = session_dir / csv_name
            df.to_csv(csv_path, index=False)
            file_paths.append(str(csv_path))
            file_names.append(Path(csv_name).stem)
    finally:
        xls.close()
    tmp_path.unlink(missing_ok=True)
    if not file_paths:
        raise AdhocFileError('Excel file has no readable sheets with data.')
    return (file_paths, file_names)

async def delete_file(session_id: str, client_id: Optional[str]=None) -> bool:
    metadata = session_memory.get_adhoc_file(session_id)
    if not metadata:
        return False
    cid = client_id or metadata.get('client_id')
    if not cid:
        logger.error('delete_file called without client_id and metadata has no client_id for session %s', session_id)
        raise ValueError('client_id is required for file operations — cannot delete without tenant context')
    session_dir = _session_dir(cid, session_id)
    if session_dir.exists():
        try:
            shutil.rmtree(session_dir)
        except OSError as e:
            logger.warning('Failed to remove adhoc dir %s: %s', session_dir, e)
    from config.system_config import STORAGE_BACKEND
    if STORAGE_BACKEND == 'gcs':
        try:
            from util.storage.backend import get_storage_backend
            storage = get_storage_backend()
            prefix = _storage_session_prefix(cid, session_id)
            await storage.delete_prefix(prefix)
        except Exception as e:
            logger.warning('Failed to delete GCS adhoc files: %s', e)
    session_memory.clear_adhoc_file(session_id)
    logger.info('Ad-hoc file deleted | session=%s | client=%s', session_id, cid)
    return True

def get_file_metadata(session_id: str) -> Optional[Dict[str, Any]]:
    return session_memory.get_adhoc_file(session_id)

def cleanup_expired_files() -> int:
    if not CLIENTS_BASE_DIR.exists():
        return 0
    cutoff = time.time() - FILE_TTL_HOURS * 3600
    removed = 0
    for client_dir in CLIENTS_BASE_DIR.iterdir():
        if not client_dir.is_dir():
            continue
        adhoc_dir = client_dir / ADHOC_SUBDIR
        if not adhoc_dir.exists():
            continue
        for session_dir in adhoc_dir.iterdir():
            if not session_dir.is_dir():
                continue
            try:
                mtime = session_dir.stat().st_mtime
                if mtime < cutoff:
                    from config.system_config import STORAGE_BACKEND
                    if STORAGE_BACKEND == 'gcs':
                        try:
                            from util.storage.backend import get_storage_backend
                            import asyncio
                            storage = get_storage_backend()
                            client_id = client_dir.name
                            session_id = session_dir.name
                            prefix = _storage_session_prefix(client_id, session_id)
                            try:
                                loop = asyncio.get_running_loop()
                                loop.create_task(storage.delete_prefix(prefix))
                            except RuntimeError:
                                asyncio.run(storage.delete_prefix(prefix))
                        except Exception as e:
                            logger.warning('Failed to delete GCS adhoc files for %s/%s: %s', client_dir.name, session_dir.name, e)
                    shutil.rmtree(session_dir)
                    removed += 1
            except OSError:
                continue
    if removed:
        logger.info('Cleaned up %d expired ad-hoc file directories', removed)
    return removed