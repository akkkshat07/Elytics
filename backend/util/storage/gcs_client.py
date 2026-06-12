import asyncio
import logging
import os
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple
from google.cloud import storage
from google.cloud.storage import Blob
logger = logging.getLogger(__name__)

def _resolve_gcs_credentials_path(explicit_path: Optional[str]) -> Tuple[Optional[str], str]:
    env_candidates = [('GOOGLE_APPLICATION_CREDENTIALS', os.getenv('GOOGLE_APPLICATION_CREDENTIALS')), ('GCP_SA_KEY_FILE', os.getenv('GCP_SA_KEY_FILE'))]
    if explicit_path:
        resolved = Path(explicit_path).expanduser()
        source = 'credentials_path'
    else:
        source = 'adc'
        resolved = None
        for env_name, env_value in env_candidates:
            if env_value:
                resolved = Path(env_value).expanduser()
                source = env_name
                break
    if resolved is None:
        return (None, source)
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f'GCS credential file not found at {resolved}. Set GOOGLE_APPLICATION_CREDENTIALS or GCP_SA_KEY_FILE to a valid service-account JSON path.')
    os.environ.setdefault('GOOGLE_APPLICATION_CREDENTIALS', str(resolved))
    return (str(resolved), source)

class GCSClient:

    def __init__(self, bucket_name: Optional[str]=None, project_id: Optional[str]=None, credentials_path: Optional[str]=None) -> None:
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET', 'coresight-data')
        self.project_id = project_id or os.getenv('GCS_PROJECT_ID')
        creds_path, auth_source = _resolve_gcs_credentials_path(credentials_path)
        if creds_path:
            self._client = storage.Client.from_service_account_json(creds_path, project=self.project_id)
            auth_mode = 'service_account_json'
        else:
            self._client = storage.Client(project=self.project_id)
            auth_mode = 'application_default_credentials'
        self._bucket = self._client.bucket(self.bucket_name)
        logger.info('GCSClient initialized: bucket=%s, project=%s, auth_mode=%s, auth_source=%s', self.bucket_name, self.project_id, auth_mode, auth_source)

    async def read_bytes(self, gcs_path: str) -> bytes:

        def _download() -> bytes:
            blob = self._bucket.blob(gcs_path)
            return blob.download_as_bytes()
        try:
            return await asyncio.to_thread(_download)
        except Exception as e:
            error_str = str(e)
            if '404' in error_str or 'No such object' in error_str:
                logger.debug(f'Object not found: gs://{self.bucket_name}/{gcs_path}')
            else:
                logger.error(f'Failed to read bytes from gs://{self.bucket_name}/{gcs_path}: {e}')
            raise

    async def read_text(self, gcs_path: str, encoding: str='utf-8') -> str:
        data = await self.read_bytes(gcs_path)
        return data.decode(encoding)

    async def download_to_file(self, gcs_path: str, local_path: str) -> None:

        def _download() -> None:
            blob = self._bucket.blob(gcs_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
        try:
            await asyncio.to_thread(_download)
            logger.debug(f'Downloaded gs://{self.bucket_name}/{gcs_path} → {local_path}')
        except Exception as e:
            logger.error(f'Failed to download gs://{self.bucket_name}/{gcs_path}: {e}')
            raise

    async def write_bytes(self, gcs_path: str, data: bytes, content_type: Optional[str]=None) -> None:

        def _upload() -> None:
            blob = self._bucket.blob(gcs_path)
            blob.upload_from_string(data, content_type=content_type)
        try:
            await asyncio.to_thread(_upload)
            logger.debug(f'Uploaded {len(data)} bytes → gs://{self.bucket_name}/{gcs_path}')
        except Exception as e:
            logger.error(f'Failed to write bytes to gs://{self.bucket_name}/{gcs_path}: {e}')
            raise

    async def write_text(self, gcs_path: str, text: str, encoding: str='utf-8') -> None:
        await self.write_bytes(gcs_path, text.encode(encoding), content_type='text/plain')

    async def upload_from_file(self, local_path: str, gcs_path: str, content_type: Optional[str]=None) -> None:

        def _upload() -> None:
            blob = self._bucket.blob(gcs_path)
            blob.upload_from_filename(local_path, content_type=content_type)
        try:
            await asyncio.to_thread(_upload)
            logger.debug(f'Uploaded {local_path} → gs://{self.bucket_name}/{gcs_path}')
        except Exception as e:
            logger.error(f'Failed to upload {local_path} to gs://{self.bucket_name}/{gcs_path}: {e}')
            raise

    async def blob_exists(self, gcs_path: str) -> bool:

        def _exists() -> bool:
            blob = self._bucket.blob(gcs_path)
            return blob.exists()
        try:
            return await asyncio.to_thread(_exists)
        except Exception as e:
            logger.error(f'Failed to check existence of gs://{self.bucket_name}/{gcs_path}: {e}')
            raise

    async def list_blobs(self, prefix: str, delimiter: Optional[str]=None) -> List[str]:

        def _list() -> List[str]:
            blobs = self._client.list_blobs(self.bucket_name, prefix=prefix, delimiter=delimiter)
            return [blob.name for blob in blobs]
        try:
            return await asyncio.to_thread(_list)
        except Exception as e:
            logger.error(f'Failed to list blobs at gs://{self.bucket_name}/{prefix}: {e}')
            raise

    async def get_blob_size(self, gcs_path: str) -> Optional[int]:

        def _size() -> Optional[int]:
            blob = self._bucket.blob(gcs_path)
            blob.reload()
            return blob.size
        try:
            return await asyncio.to_thread(_size)
        except Exception as e:
            logger.debug(f'Could not get size for gs://{self.bucket_name}/{gcs_path}: {e}')
            return None

    async def delete_blob(self, gcs_path: str) -> bool:

        def _delete() -> bool:
            blob = self._bucket.blob(gcs_path)
            if blob.exists():
                blob.delete()
                return True
            return False
        try:
            result = await asyncio.to_thread(_delete)
            if result:
                logger.debug(f'Deleted gs://{self.bucket_name}/{gcs_path}')
            return result
        except Exception as e:
            logger.error(f'Failed to delete gs://{self.bucket_name}/{gcs_path}: {e}')
            raise

    async def delete_prefix(self, prefix: str) -> int:

        def _delete_all() -> int:
            blobs = list(self._client.list_blobs(self.bucket_name, prefix=prefix))
            if not blobs:
                return 0
            self._bucket.delete_blobs(blobs)
            return len(blobs)
        try:
            count = await asyncio.to_thread(_delete_all)
            logger.info(f'Deleted {count} blobs under gs://{self.bucket_name}/{prefix}')
            return count
        except Exception as e:
            logger.error(f'Failed to delete prefix gs://{self.bucket_name}/{prefix}: {e}')
            raise
_gcs_client: Optional[GCSClient] = None

def get_gcs_client() -> GCSClient:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = GCSClient()
    return _gcs_client

def initialize_gcs_client(bucket_name: Optional[str]=None, project_id: Optional[str]=None, credentials_path: Optional[str]=None) -> GCSClient:
    global _gcs_client
    _gcs_client = GCSClient(bucket_name=bucket_name, project_id=project_id, credentials_path=credentials_path)
    return _gcs_client