from __future__ import annotations
import logging
import os
from pathlib import Path
import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from middleware.auth_middleware import require_auth
from util.audit_logger import AuditEventType, AuditSeverity, audit_logger
from util.file_security import sanitize_filename, validate_path_within_directory
logger = logging.getLogger(__name__)
router = APIRouter(tags=['Uploaded data'])
ALLOWED_EXTENSIONS = frozenset({'.csv', '.xlsx', '.xls'})
_NOT_WRITABLE_HINT = 'The upload folder exists but the CoreSight server user cannot write to it. On the VM, fix ownership/permissions on that directory (e.g. `sudo chown -R <service-user>:<group> /data/uploaded_data` or `chmod` / ACL), or set CORESIGHT_VM_UPLOAD_DIR to a directory the app user can write.'

def _upload_root() -> Path:
    raw = os.getenv('CORESIGHT_VM_UPLOAD_DIR', '/data/uploaded_data').strip()
    return Path(raw).expanduser()

def _unique_destination(upload_dir: Path, safe_name: str) -> Path:
    candidate = upload_dir / safe_name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    n = 1
    while True:
        alt = upload_dir / f'{stem}_{n}{suffix}'
        if not alt.exists():
            return alt
        n += 1

@router.post('/uploaded-data')
async def upload_vm_data_file(file: UploadFile=File(...), current_user: dict=Depends(require_auth())):
    if not file.filename:
        raise HTTPException(status_code=400, detail='Missing filename')
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Only CSV and Excel files are allowed (.csv, .xlsx, .xls). Got: {ext or '(none)'}")
    max_bytes = int(os.getenv('MAX_UPLOAD_SIZE', 200 * 1024 * 1024))
    upload_dir = _upload_root()
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        logger.error('No permission to create upload directory %s: %s', upload_dir, e)
        raise HTTPException(status_code=503, detail=_NOT_WRITABLE_HINT)
    except OSError as e:
        logger.error('Could not create upload directory %s: %s', upload_dir, e)
        raise HTTPException(status_code=500, detail='Upload directory is not available')
    safe_name = sanitize_filename(file.filename)
    if Path(safe_name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail='Invalid file type after sanitization')
    dest_path = _unique_destination(upload_dir, safe_name)
    is_valid, err_msg = validate_path_within_directory(dest_path, upload_dir)
    if not is_valid:
        logger.error('Path validation failed for VM upload: %s', err_msg)
        raise HTTPException(status_code=400, detail='Invalid file path')
    CHUNK_SIZE = 1024 * 1024
    total = 0
    try:
        async with aiofiles.open(dest_path, 'wb') as out:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    try:
                        dest_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise HTTPException(status_code=400, detail=f'File too large. Maximum size is {max_bytes // (1024 * 1024)} MB.')
                await out.write(chunk)
    except HTTPException:
        raise
    except PermissionError as e:
        logger.error('VM data upload permission denied | path=%s | %s', dest_path, e)
        try:
            dest_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(status_code=503, detail=_NOT_WRITABLE_HINT)
    except Exception as e:
        logger.exception('VM data upload failed: %s', e)
        try:
            dest_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail='Failed to save file')
    logger.info('VM data upload | user=%s | client_id=%s | path=%s | bytes=%s', current_user.get('email'), current_user.get('client_id'), dest_path, total)
    try:
        await audit_logger.log_event(event_type=AuditEventType.FILE_UPLOAD, severity=AuditSeverity.INFO, user_id=current_user.get('email', 'unknown'), client_id=current_user.get('client_id'), details={'kind': 'vm_uploaded_data', 'original_filename': file.filename, 'saved_path': str(dest_path), 'bytes': total})
    except Exception as e:
        logger.warning('Audit log for VM upload failed (non-fatal): %s', e)
    return {'success': True, 'filename': dest_path.name, 'path': str(dest_path), 'bytes': total, 'message': f'File saved to {dest_path}'}