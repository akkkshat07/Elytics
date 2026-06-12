import logging
import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
logger = logging.getLogger(__name__)
_remote_client: Optional[QdrantClient] = None
_local_client_pool: Dict[str, QdrantClient] = {}

def _get_config():
    from config.system_config import VECTOR_DB_CONFIG
    return VECTOR_DB_CONFIG

def get_qdrant_client(client_id: Optional[str]=None) -> QdrantClient:
    import os
    import pathlib
    global _remote_client
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_host = os.getenv('QDRANT_HOST')
    if qdrant_url or qdrant_host:
        if _remote_client is None:
            if qdrant_url:
                _remote_client = QdrantClient(url=qdrant_url)
                logger.info('QdrantClient connected to remote URL: %s', qdrant_url)
            else:
                port = int(os.getenv('QDRANT_PORT', '6333'))
                _remote_client = QdrantClient(host=qdrant_host, port=port)
                logger.info('QdrantClient connected to remote host: %s:%d', qdrant_host, port)
        return _remote_client
    pool_key = client_id if client_id else '__global__'
    if pool_key not in _local_client_pool:
        cfg = _get_config()
        base_path = cfg.get('db_path', './assets/data/vector_db')
        if client_id:
            local_path = str(pathlib.Path(base_path) / client_id)
        else:
            local_path = base_path
        pathlib.Path(local_path).mkdir(parents=True, exist_ok=True)
        try:
            _local_client_pool[pool_key] = QdrantClient(path=local_path)
            logger.info('QdrantClient (local) client=%r → %s', client_id or 'global', local_path)
        except RuntimeError as e:
            if 'already accessed' in str(e).lower():
                logger.warning('Local Qdrant storage locked (client=%r) — falling back to in-memory mode (data will not persist): %s', client_id, e)
                _local_client_pool[pool_key] = QdrantClient(':memory:')
            else:
                raise
    return _local_client_pool[pool_key]

def string_id_to_uuid(string_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

def ensure_collection(collection_name: str, vector_size: int, client_id: Optional[str]=None) -> None:
    client = get_qdrant_client(client_id)
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        try:
            client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))
            logger.info("Created Qdrant collection '%s' (dim=%d, cosine)", collection_name, vector_size)
        except Exception as e:
            if 'already exists' in str(e).lower() or 'file exists' in str(e).lower():
                logger.warning("Qdrant collection '%s' has stale storage directory; cleaning up and recreating.", collection_name)
                try:
                    client.delete_collection(collection_name=collection_name)
                except Exception:
                    pass
                client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))
                logger.info("Recreated Qdrant collection '%s' after stale cleanup (dim=%d, cosine)", collection_name, vector_size)
            else:
                raise
    else:
        logger.debug("Qdrant collection '%s' already exists", collection_name)

def upsert_points(collection_name: str, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]], documents: Optional[List[str]]=None, client_id: Optional[str]=None) -> None:
    client = get_qdrant_client(client_id)
    points = []
    for idx, (sid, vec, payload) in enumerate(zip(ids, vectors, payloads)):
        p = dict(payload)
        if documents and idx < len(documents):
            p['document'] = documents[idx]
        p['_string_id'] = sid
        points.append(PointStruct(id=string_id_to_uuid(sid), vector=vec, payload=p))
    client.upsert(collection_name=collection_name, points=points, wait=True)
    logger.debug("Upserted %d points into '%s'", len(points), collection_name)

def search_vectors(collection_name: str, query_vector: List[float], limit: int=5, score_threshold: Optional[float]=None, query_filter: Optional[Dict[str, Any]]=None, client_id: Optional[str]=None) -> List[Dict[str, Any]]:
    client = get_qdrant_client(client_id)
    qdrant_filter: Optional[Filter] = None
    if query_filter:
        conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in query_filter.items()]
        qdrant_filter = Filter(must=conditions)
    results = client.query_points(collection_name=collection_name, query=query_vector, limit=limit, score_threshold=score_threshold, query_filter=qdrant_filter, with_payload=True)
    out = []
    for point in results.points:
        payload = dict(point.payload) if point.payload else {}
        out.append({'id': str(point.id), 'score': point.score, 'payload': payload, 'document': payload.get('document', '')})
    return out

def scroll_by_filter(collection_name: str, filter_dict: Dict[str, Any], limit: int=100, client_id: Optional[str]=None) -> List[Dict[str, Any]]:
    client = get_qdrant_client(client_id)
    conditions = []
    for key, value in filter_dict.items():
        conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    results, _next_offset = client.scroll(collection_name=collection_name, scroll_filter=Filter(must=conditions), limit=limit, with_payload=True, with_vectors=False)
    out = []
    for point in results:
        payload = dict(point.payload) if point.payload else {}
        out.append({'id': str(point.id), 'payload': payload, 'document': payload.get('document', '')})
    return out

def scroll_all(collection_name: str, limit: int=1000, with_vectors: bool=False, client_id: Optional[str]=None) -> List[Dict[str, Any]]:
    client = get_qdrant_client(client_id)
    results, _next_offset = client.scroll(collection_name=collection_name, limit=limit, with_payload=True, with_vectors=with_vectors)
    out = []
    for point in results:
        payload = dict(point.payload) if point.payload else {}
        out.append({'id': str(point.id), 'payload': payload, 'document': payload.get('document', '')})
    return out

def delete_by_ids(collection_name: str, ids: List[str], client_id: Optional[str]=None) -> None:
    client = get_qdrant_client(client_id)
    uuids = [string_id_to_uuid(sid) for sid in ids]
    client.delete(collection_name=collection_name, points_selector=uuids)
    logger.debug("Deleted %d points from '%s'", len(uuids), collection_name)

def count_points(collection_name: str, client_id: Optional[str]=None) -> int:
    client = get_qdrant_client(client_id)
    info = client.get_collection(collection_name=collection_name)
    return info.points_count

def get_by_ids(collection_name: str, ids: List[str], client_id: Optional[str]=None) -> List[Dict[str, Any]]:
    client = get_qdrant_client(client_id)
    uuids = [string_id_to_uuid(sid) for sid in ids]
    points = client.retrieve(collection_name=collection_name, ids=uuids, with_payload=True, with_vectors=False)
    out = []
    for point in points:
        payload = dict(point.payload) if point.payload else {}
        out.append({'id': str(point.id), 'payload': payload, 'document': payload.get('document', '')})
    return out

def collection_exists(collection_name: str, client_id: Optional[str]=None) -> bool:
    client = get_qdrant_client(client_id)
    collections = [c.name for c in client.get_collections().collections]
    return collection_name in collections

def recreate_collection(collection_name: str, vector_size: int, client_id: Optional[str]=None) -> None:
    client = get_qdrant_client(client_id)
    if collection_exists(collection_name, client_id=client_id):
        client.delete_collection(collection_name=collection_name)
        logger.info("Deleted existing Qdrant collection '%s'", collection_name)
    client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))
    logger.info("Recreated Qdrant collection '%s' (dim=%d, cosine)", collection_name, vector_size)