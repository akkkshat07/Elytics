import asyncio
import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
logger = logging.getLogger(__name__)

class StorageBackend(ABC):

    @abstractmethod
    async def read_bytes(self, path: str) -> bytes:
        ...

    @abstractmethod
    async def read_text(self, path: str, encoding: str='utf-8') -> str:
        ...

    @abstractmethod
    async def write_bytes(self, path: str, data: bytes, content_type: Optional[str]=None) -> None:
        ...

    @abstractmethod
    async def write_text(self, path: str, text: str, encoding: str='utf-8') -> None:
        ...

    @abstractmethod
    async def exists(self, path: str) -> bool:
        ...

    @abstractmethod
    async def list_files(self, prefix: str, delimiter: Optional[str]=None) -> List[str]:
        ...

    @abstractmethod
    async def delete(self, path: str) -> bool:
        ...

    @abstractmethod
    async def delete_prefix(self, prefix: str) -> int:
        ...

    @abstractmethod
    async def upload_file(self, local_path: str, remote_path: str, content_type: Optional[str]=None) -> None:
        ...

    @abstractmethod
    async def download_file(self, remote_path: str, local_path: str) -> None:
        ...

    @abstractmethod
    def get_uri(self, path: str) -> str:
        ...

    @abstractmethod
    async def ensure_directory(self, path: str) -> None:
        ...

class LocalStorageBackend(StorageBackend):

    def __init__(self, base_dir: Optional[str]=None) -> None:
        if base_dir:
            self._base = Path(base_dir)
        else:
            self._base = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        logger.info(f'LocalStorageBackend initialized: base_dir={self._base}')

    def _resolve(self, path: str) -> Path:
        return self._base / path

    async def read_bytes(self, path: str) -> bytes:
        resolved = self._resolve(path)
        try:
            return await asyncio.to_thread(resolved.read_bytes)
        except Exception as e:
            logger.error(f'Failed to read bytes from {resolved}: {e}')
            raise

    async def read_text(self, path: str, encoding: str='utf-8') -> str:
        resolved = self._resolve(path)
        try:
            return await asyncio.to_thread(resolved.read_text, encoding)
        except Exception as e:
            logger.error(f'Failed to read text from {resolved}: {e}')
            raise

    async def write_bytes(self, path: str, data: bytes, content_type: Optional[str]=None) -> None:
        resolved = self._resolve(path)
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(resolved.write_bytes, data)
            logger.debug(f'Wrote {len(data)} bytes to {resolved}')
        except Exception as e:
            logger.error(f'Failed to write bytes to {resolved}: {e}')
            raise

    async def write_text(self, path: str, text: str, encoding: str='utf-8') -> None:
        resolved = self._resolve(path)
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(resolved.write_text, text, encoding)
            logger.debug(f'Wrote text to {resolved}')
        except Exception as e:
            logger.error(f'Failed to write text to {resolved}: {e}')
            raise

    async def exists(self, path: str) -> bool:
        resolved = self._resolve(path)
        return await asyncio.to_thread(resolved.exists)

    async def list_files(self, prefix: str, delimiter: Optional[str]=None) -> List[str]:
        resolved = self._resolve(prefix)
        if not resolved.exists():
            return []

        def _list() -> List[str]:
            results = []
            if resolved.is_dir():
                for item in resolved.rglob('*'):
                    if item.is_file():
                        results.append(str(item.relative_to(self._base)))
            return results
        try:
            return await asyncio.to_thread(_list)
        except Exception as e:
            logger.error(f'Failed to list files at {resolved}: {e}')
            raise

    async def delete(self, path: str) -> bool:
        resolved = self._resolve(path)
        try:
            if resolved.exists():
                await asyncio.to_thread(resolved.unlink)
                return True
            return False
        except Exception as e:
            logger.error(f'Failed to delete {resolved}: {e}')
            raise

    async def delete_prefix(self, prefix: str) -> int:
        resolved = self._resolve(prefix)
        if not resolved.exists():
            return 0

        def _delete() -> int:
            if resolved.is_dir():
                items = list(resolved.rglob('*'))
                files = [f for f in items if f.is_file()]
                count = len(files)
                shutil.rmtree(resolved)
                return count
            return 0
        try:
            return await asyncio.to_thread(_delete)
        except Exception as e:
            logger.error(f'Failed to delete prefix {resolved}: {e}')
            raise

    async def upload_file(self, local_path: str, remote_path: str, content_type: Optional[str]=None) -> None:
        resolved = self._resolve(remote_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, local_path, str(resolved))

    async def download_file(self, remote_path: str, local_path: str) -> None:
        resolved = self._resolve(remote_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        await asyncio.to_thread(shutil.copy2, str(resolved), local_path)

    def get_uri(self, path: str) -> str:
        return str(self._resolve(path))

    async def ensure_directory(self, path: str) -> None:
        resolved = self._resolve(path)
        resolved.mkdir(parents=True, exist_ok=True)

class GCSStorageBackend(StorageBackend):

    def __init__(self, bucket_name: Optional[str]=None, project_id: Optional[str]=None, credentials_path: Optional[str]=None) -> None:
        from util.storage.gcs_client import GCSClient
        self._gcs = GCSClient(bucket_name=bucket_name, project_id=project_id, credentials_path=credentials_path)
        self._bucket_name = self._gcs.bucket_name
        logger.info(f'GCSStorageBackend initialized: bucket={self._bucket_name}')

    async def read_bytes(self, path: str) -> bytes:
        return await self._gcs.read_bytes(path)

    async def read_text(self, path: str, encoding: str='utf-8') -> str:
        return await self._gcs.read_text(path, encoding=encoding)

    async def write_bytes(self, path: str, data: bytes, content_type: Optional[str]=None) -> None:
        await self._gcs.write_bytes(path, data, content_type=content_type)

    async def write_text(self, path: str, text: str, encoding: str='utf-8') -> None:
        await self._gcs.write_text(path, text, encoding=encoding)

    async def exists(self, path: str) -> bool:
        return await self._gcs.blob_exists(path)

    async def list_files(self, prefix: str, delimiter: Optional[str]=None) -> List[str]:
        return await self._gcs.list_blobs(prefix, delimiter=delimiter)

    async def delete(self, path: str) -> bool:
        return await self._gcs.delete_blob(path)

    async def delete_prefix(self, prefix: str) -> int:
        return await self._gcs.delete_prefix(prefix)

    async def upload_file(self, local_path: str, remote_path: str, content_type: Optional[str]=None) -> None:
        await self._gcs.upload_from_file(local_path, remote_path, content_type=content_type)

    async def download_file(self, remote_path: str, local_path: str) -> None:
        await self._gcs.download_to_file(remote_path, local_path)

    def get_uri(self, path: str) -> str:
        return f'gs://{self._bucket_name}/{path}'

    async def ensure_directory(self, path: str) -> None:
        pass
_storage_backend: Optional[StorageBackend] = None
STORAGE_BACKEND_TYPE = os.getenv('STORAGE_BACKEND', 'local')

def get_storage_backend() -> StorageBackend:
    global _storage_backend
    if _storage_backend is None:
        if STORAGE_BACKEND_TYPE == 'gcs':
            _storage_backend = GCSStorageBackend()
        else:
            _storage_backend = LocalStorageBackend()
        logger.info(f'Storage backend initialized: {STORAGE_BACKEND_TYPE}')
    return _storage_backend

def initialize_storage_backend(backend_type: Optional[str]=None, **kwargs) -> StorageBackend:
    global _storage_backend
    bt = backend_type or STORAGE_BACKEND_TYPE
    if bt == 'gcs':
        _storage_backend = GCSStorageBackend(**kwargs)
    else:
        _storage_backend = LocalStorageBackend(**kwargs)
    logger.info(f'Storage backend explicitly initialized: {bt}')
    return _storage_backend