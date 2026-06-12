import asyncio
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
from datetime import datetime
from response_caching.cache_manager import CacheManager
from middleware.auth_middleware import require_auth
from db_config.mongo_server import get_db
logger = logging.getLogger(__name__)
cache_router = APIRouter(tags=['Cache Management'])

class DeleteRequest(BaseModel):
    question_ids: List[int]

class DeleteResponse(BaseModel):
    success: bool
    deleted_from_csv: int
    deleted_from_parquet: int
    deleted_from_vector_db: int
    errors: List[str]

class PreviewResponse(BaseModel):
    success: bool
    questions_to_delete: List[dict]
    count: int
    not_found_ids: List[int]
    total_before: int
    total_after: int

class UpdateRequest(BaseModel):
    question: Optional[str] = None
    planner_agent_response: Optional[str] = None
    python_agent_response: Optional[str] = None
    business_agent_response: Optional[str] = None

class UpdateResponse(BaseModel):
    success: bool
    data: Dict

@cache_router.get('/questions')
async def list_questions(page: int=Query(1, ge=1, description='Page number (1-indexed)'), page_size: int=Query(50, ge=1, le=500, description='Items per page'), search: Optional[str]=Query(None, description='Search query'), id_min: Optional[int]=Query(None, description='Minimum question ID'), id_max: Optional[int]=Query(None, description='Maximum question ID'), sort_by: Optional[str]=Query(None, description="Sort field: 'id', 'question', or 'user_id'"), sort_order: Optional[str]=Query(None, description="Sort order: 'asc' or 'desc'"), storage_filter: Optional[str]=Query(None, description="Filter by storage: 'csv', 'parquet', or 'vector_db'"), date_from: Optional[str]=Query(None, description='Start date (ISO) for created_at filter'), date_to: Optional[str]=Query(None, description='End date (ISO, exclusive) for created_at filter'), dataset_id: Optional[str]=Query(None, description='Dataset to scope cache operations to'), current_user: dict=Depends(require_auth()), db=Depends(get_db)):
    try:
        client_id = current_user.get('client_id')
        if not client_id:
            logger.error(f"SECURITY: Token missing client_id for user {current_user.get('user_id', 'unknown')}")
            raise HTTPException(status_code=403, detail='Invalid authentication token: missing client identifier. Please log in again.')
        logger.info(f"[MULTI-TENANT] Cache list_questions request from client='{client_id}', dataset='{dataset_id}', user='{current_user.get('user_id')}'")

        def _list():
            return CacheManager(client_id=client_id, dataset_id=dataset_id).list_questions(page=page, page_size=page_size, search_query=search, id_min=id_min, id_max=id_max, sort_by=sort_by, sort_order=sort_order, storage_filter=storage_filter, date_from=date_from, date_to=date_to)
        result = await asyncio.to_thread(_list)
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        data = result.get('data') or []
        need_ts = [item['question'] for item in data if not item.get('created_at')]
        if need_ts:
            question_to_created_at = {}
            cursor = db.conversations.find({'client_id': client_id, '$or': [{'input': {'$in': need_ts}}, {'enhanced_question': {'$in': need_ts}}]}, {'input': 1, 'enhanced_question': 1, 'created_at': 1}).sort('created_at', -1)
            async for doc in cursor:
                created_at = doc.get('created_at')
                if created_at is None:
                    continue
                ts = created_at.isoformat() if isinstance(created_at, datetime) else str(created_at)
                for q in (doc.get('input'), doc.get('enhanced_question')):
                    if q and q in need_ts and (q not in question_to_created_at):
                        question_to_created_at[q] = ts
            for item in data:
                if not item.get('created_at') and item.get('question') in question_to_created_at:
                    item['created_at'] = question_to_created_at[item['question']]
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error in list_questions endpoint: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@cache_router.get('/search')
async def search_questions(q: str=Query(..., description='Search query'), dataset_id: Optional[str]=Query(None, description='Dataset to scope cache operations to'), current_user: dict=Depends(require_auth())):
    try:
        client_id = current_user.get('client_id')
        if not client_id:
            logger.error(f"SECURITY: Token missing client_id for user {current_user.get('user_id', 'unknown')}")
            raise HTTPException(status_code=403, detail='Invalid authentication token: missing client identifier. Please log in again.')
        logger.info(f"[MULTI-TENANT] Cache search request from client='{client_id}', dataset='{dataset_id}', user='{current_user.get('user_id')}'")
        result = await asyncio.to_thread(lambda: CacheManager(client_id=client_id, dataset_id=dataset_id).search_questions(q))
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error in search_questions endpoint: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@cache_router.get('/questions/{question_id}')
async def get_question_detail(question_id: int, dataset_id: Optional[str]=Query(None, description='Dataset to scope cache operations to'), current_user: dict=Depends(require_auth())):
    try:
        client_id = current_user.get('client_id')
        if not client_id:
            logger.error(f"SECURITY: Token missing client_id for user {current_user.get('user_id', 'unknown')}")
            raise HTTPException(status_code=403, detail='Invalid authentication token: missing client identifier. Please log in again.')
        logger.info(f"[MULTI-TENANT] Cache get_question_detail request from client='{client_id}', dataset='{dataset_id}', user='{current_user.get('user_id')}', question_id={question_id}")
        result = await asyncio.to_thread(lambda: CacheManager(client_id=client_id, dataset_id=dataset_id).get_question_detail(question_id))
        if not result.get('success'):
            code = result.get('code')
            if code == 'not_found':
                raise HTTPException(status_code=404, detail='Question not found')
            raise HTTPException(status_code=400, detail=result.get('error', 'Unknown error'))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error in get_question_detail endpoint: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@cache_router.post('/preview-delete')
async def preview_delete(request: DeleteRequest, dataset_id: Optional[str]=Query(None, description='Dataset to scope cache operations to'), current_user: dict=Depends(require_auth())):
    try:
        client_id = current_user.get('client_id')
        if not client_id:
            logger.error(f"SECURITY: Token missing client_id for user {current_user.get('user_id', 'unknown')}")
            raise HTTPException(status_code=403, detail='Invalid authentication token: missing client identifier. Please log in again.')
        if not request.question_ids:
            raise HTTPException(status_code=400, detail='No question IDs provided')
        logger.info(f"[MULTI-TENANT] Cache preview_delete request from client='{client_id}', dataset='{dataset_id}', user='{current_user.get('user_id')}', question_ids={request.question_ids}")
        result = await asyncio.to_thread(lambda: CacheManager(client_id=client_id, dataset_id=dataset_id).preview_delete(request.question_ids))
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error in preview_delete endpoint: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@cache_router.post('/questions/delete')
async def delete_questions(request: DeleteRequest, dataset_id: Optional[str]=Query(None, description='Dataset to scope cache operations to'), current_user: dict=Depends(require_auth())):
    try:
        client_id = current_user.get('client_id')
        if not client_id:
            logger.error(f"SECURITY: Token missing client_id for user {current_user.get('user_id', 'unknown')}")
            raise HTTPException(status_code=403, detail='Invalid authentication token: missing client identifier. Please log in again.')
        if not request.question_ids:
            raise HTTPException(status_code=400, detail='No question IDs provided')
        logger.info(f"[MULTI-TENANT] User {current_user.get('user_id')} (client='{client_id}', dataset='{dataset_id}') deleting questions: {request.question_ids}")
        result = await asyncio.to_thread(lambda: CacheManager(client_id=client_id, dataset_id=dataset_id).delete_questions(request.question_ids))
        if not result['success']:
            raise HTTPException(status_code=500, detail=f"Deletion failed: {', '.join(result.get('errors', ['Unknown error']))}")
        logger.info(f"[MULTI-TENANT] Successfully deleted {result['deleted_from_csv']} from CSV, {result['deleted_from_parquet']} from Parquet, {result['deleted_from_vector_db']} from Vector DB for client '{client_id}', dataset '{dataset_id}'")
        try:
            from response_caching.suggester import warm_reload
            await asyncio.to_thread(warm_reload, client_id, dataset_id)
        except Exception as _e:
            logger.warning(f'Could not reload suggester cache after deletion: {_e}')
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error in delete_questions endpoint: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@cache_router.post('/questions/{question_id}/update')
async def update_question(question_id: int, request: UpdateRequest, dataset_id: Optional[str]=Query(None, description='Dataset to scope cache operations to'), current_user: dict=Depends(require_auth())):
    try:
        client_id = current_user.get('client_id')
        if not client_id:
            logger.error(f"SECURITY: Token missing client_id for user {current_user.get('user_id', 'unknown')}")
            raise HTTPException(status_code=403, detail='Invalid authentication token: missing client identifier. Please log in again.')
        if not any([request.question, request.planner_agent_response, request.python_agent_response, request.business_agent_response]):
            raise HTTPException(status_code=400, detail='No update fields provided')
        logger.info(f"[MULTI-TENANT] Cache update_question request from client='{client_id}', dataset='{dataset_id}', user='{current_user.get('user_id')}', question_id={question_id}")
        payload = request.dict(exclude_unset=True)
        result = await asyncio.to_thread(lambda: CacheManager(client_id=client_id, dataset_id=dataset_id).update_question(question_id, payload))
        if not result.get('success'):
            code = result.get('code')
            if code == 'not_found':
                raise HTTPException(status_code=404, detail='Question not found')
            if code == 'missing_api_key':
                raise HTTPException(status_code=500, detail=result.get('error'))
            if code == 'invalid_request':
                raise HTTPException(status_code=400, detail=result.get('error'))
            raise HTTPException(status_code=500, detail=result.get('error', 'Update failed'))
        try:
            from response_caching.suggester import warm_reload
            warm_reload(client_id, dataset_id=dataset_id)
        except Exception as _e:
            logger.warning(f'Could not reload suggester cache after update: {_e}')
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error in update_question endpoint: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@cache_router.get('/stats')
async def get_stats(dataset_id: Optional[str]=Query(None, description='Dataset to scope cache operations to'), current_user: dict=Depends(require_auth())):
    try:
        client_id = current_user.get('client_id')
        if not client_id:
            logger.error(f"SECURITY: Token missing client_id for user {current_user.get('user_id', 'unknown')}")
            raise HTTPException(status_code=403, detail='Invalid authentication token: missing client identifier. Please log in again.')
        logger.info(f"[MULTI-TENANT] Cache get_stats request from client='{client_id}', dataset='{dataset_id}', user='{current_user.get('user_id')}'")
        result = await asyncio.to_thread(lambda: CacheManager(client_id=client_id, dataset_id=dataset_id).get_stats())
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error in get_stats endpoint: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@cache_router.post('/rebuild')
async def rebuild_vector_db(dataset_id: Optional[str]=Query(None, description='Dataset to scope cache operations to'), current_user: dict=Depends(require_auth())):
    try:
        client_id = current_user.get('client_id')
        if not client_id:
            logger.error(f"SECURITY: Token missing client_id for user {current_user.get('user_id', 'unknown')}")
            raise HTTPException(status_code=403, detail='Invalid authentication token: missing client identifier. Please log in again.')
        logger.info(f"[MULTI-TENANT] User {current_user.get('user_id')} (client='{client_id}', dataset='{dataset_id}') initiated vector DB rebuild")
        result = await asyncio.to_thread(lambda: CacheManager(client_id=client_id, dataset_id=dataset_id).rebuild_vector_db())
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Rebuild failed'))
        logger.info(f"[MULTI-TENANT] Vector DB rebuild completed successfully for client '{client_id}', dataset '{dataset_id}'")
        try:
            from response_caching.suggester import warm_reload
            warm_reload(client_id, dataset_id=dataset_id)
        except Exception as _e:
            logger.warning(f'Could not reload suggester cache after rebuild: {_e}')
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error in rebuild_vector_db endpoint: {e}')
        raise HTTPException(status_code=500, detail=str(e))