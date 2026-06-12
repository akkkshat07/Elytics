import asyncio
from typing import Dict, List, Any, Optional, Callable
import uuid
import json
from datetime import datetime
_collections: Dict[str, Dict[str, Any]] = {}

class MemoryCollection:

    def __init__(self, name: str):
        self.name = name
        if name not in _collections:
            _collections[name] = {}

    async def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for doc in _collections[self.name].values():
            if all((k in doc and doc[k] == v for k, v in query.items())):
                return doc
        return None

    async def find(self, query: Dict[str, Any]=None, projection: Dict[str, Any]=None) -> List[Dict[str, Any]]:
        if query is None:
            query = {}
        results = []
        for doc in _collections[self.name].values():
            if all((k in doc and doc[k] == v for k, v in query.items())):
                if projection:
                    projected_doc = {}
                    for k, v in projection.items():
                        if v == 1 and k in doc:
                            projected_doc[k] = doc[k]
                    results.append(projected_doc)
                else:
                    results.append(doc)
        return results

    async def insert_one(self, document: Dict[str, Any]) -> Dict[str, Any]:
        if '_id' not in document:
            document['_id'] = str(uuid.uuid4())
        _collections[self.name][document['_id']] = document
        return {'inserted_id': document['_id']}

    async def insert_many(self, documents: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        inserted_ids = []
        for doc in documents:
            result = await self.insert_one(doc)
            inserted_ids.append(result['inserted_id'])
        return {'inserted_ids': inserted_ids}

    async def update_one(self, query: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        doc = await self.find_one(query)
        if not doc:
            return {'matched_count': 0, 'modified_count': 0}
        if '$set' in update:
            for k, v in update['$set'].items():
                doc[k] = v
        for k, v in update.items():
            if not k.startswith('$'):
                doc[k] = v
        return {'matched_count': 1, 'modified_count': 1}

    async def delete_one(self, query: Dict[str, Any]) -> Dict[str, Any]:
        doc = await self.find_one(query)
        if not doc:
            return {'deleted_count': 0}
        del _collections[self.name][doc['_id']]
        return {'deleted_count': 1}

    async def delete_many(self, query: Dict[str, Any]) -> Dict[str, Any]:
        docs = await self.find(query)
        if not docs:
            return {'deleted_count': 0}
        for doc in docs:
            del _collections[self.name][doc['_id']]
        return {'deleted_count': len(docs)}

    async def count_documents(self, query: Dict[str, Any]=None) -> int:
        if query is None:
            query = {}
        docs = await self.find(query)
        return len(docs)

class MemoryDatabase:

    def __init__(self, name: str):
        self.name = name

    def __getitem__(self, collection_name: str) -> MemoryCollection:
        return MemoryCollection(collection_name)

    def __getattr__(self, collection_name: str) -> MemoryCollection:
        if collection_name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{collection_name}'")
        return MemoryCollection(collection_name)

    async def command(self, command: str, *args, **kwargs) -> Dict[str, Any]:
        if command == 'ping':
            return {'ok': 1.0}
        return {'ok': 0.0, 'error': f'Command {command} not implemented'}
_memory_db = None

def get_memory_db() -> MemoryDatabase:
    global _memory_db
    if _memory_db is None:
        _memory_db = MemoryDatabase('core-sight')
    return _memory_db