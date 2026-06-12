import threading
import logging
from typing import Dict, List, Optional
logger = logging.getLogger(__name__)

class EmbeddingCache:

    def __init__(self):
        self._store: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def put(self, run_id: str, embedding: List[float]) -> None:
        with self._lock:
            self._store[run_id] = embedding

    def get(self, run_id: str) -> Optional[List[float]]:
        with self._lock:
            return self._store.get(run_id)

    def remove(self, run_id: str) -> None:
        with self._lock:
            self._store.pop(run_id, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
embedding_cache = EmbeddingCache()