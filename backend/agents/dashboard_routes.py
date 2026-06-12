from __future__ import annotations
import logging
import re
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from db_config.mongo_server import get_db
from domains.dashboard.repository import DashboardRepository
from middleware.auth_middleware import require_auth, require_auth_flexible
from services import dashboard_service
from services.live_table_view_service import LiveTableViewError, LiveTableViewService, ViewReportExportRequest, ViewReportQueryRequest
from util.time_utils import utcnow
logger = logging.getLogger(__name__)
router = APIRouter(prefix='/dashboard', tags=['Dashboard'])

def _validate_report_id(report_id: str) -> str:
    if not report_id or len(report_id) > 64:
        raise HTTPException(status_code=400, detail='Invalid report_id')
    if not re.match('^[a-zA-Z0-9_-]+$', report_id):
        raise HTTPException(status_code=400, detail='Invalid report_id format')
    return report_id

class AddFromConversationRequest(BaseModel):
    run_id: str = Field(..., min_length=1, max_length=100)
    title: str = Field('', max_length=200)

class RenameReportRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)

class ReorderRequest(BaseModel):
    ordered_ids: List[str] = Field(..., min_length=1)

class InteractRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

@router.get('/reports')
async def list_reports(current_user: dict=Depends(require_auth()), db=Depends(get_db)) -> Dict[str, Any]:
    user_id: str = current_user.get('_id') or current_user.get('user_id', '')
    client_id: str = current_user.get('client_id', '')
    if not client_id:
        raise HTTPException(status_code=400, detail='client_id is required')
    repo = DashboardRepository(db)
    doc = await repo.get_or_create(user_id, client_id)
    reports = sorted(doc.get('reports', []), key=lambda r: r.get('order', 0))
    return {'reports': reports, 'total': len(reports)}

@router.post('/reports/from-conversation')
async def add_report_from_conversation(body: AddFromConversationRequest, current_user: dict=Depends(require_auth()), db=Depends(get_db)) -> Dict[str, Any]:
    user_id: str = current_user.get('_id') or current_user.get('user_id', '')
    client_id: str = current_user.get('client_id', '')
    if not client_id:
        raise HTTPException(status_code=400, detail='client_id is required')
    if not re.match('^[a-zA-Z0-9_-]+$', body.run_id):
        raise HTTPException(status_code=400, detail='Invalid run_id format')
    try:
        report = await dashboard_service.create_report_from_run(run_id=body.run_id, title=body.title, user_id=user_id, client_id=client_id, db=db)
    except dashboard_service.AdhocQuickUploadReportBlockedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except dashboard_service.DuplicateReportFromConversationError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if report is None:
        raise HTTPException(status_code=404, detail='Conversation not found or does not contain executable code')
    try:
        from notifications.notification_service import create_notification
        from notifications.notification_model import Notification
        import logging as _log
        await create_notification(Notification(client_id=client_id, user_id=str(user_id), type='saved_as_report', title='Saved as Report', message=f'"{body.title}" has been added to your Dashboard.', metadata={'run_id': body.run_id, 'report_id': str(report.id) if hasattr(report, 'id') else ''}, target_role='any'))
    except Exception as _e:
        _log.getLogger(__name__).warning(f'Failed to create saved_as_report notification: {_e}')
    return {'report': report.model_dump(mode='python'), 'message': 'Report added to dashboard'}

@router.delete('/reports/{report_id}')
async def delete_report(report_id: str, current_user: dict=Depends(require_auth()), db=Depends(get_db)) -> Dict[str, Any]:
    report_id = _validate_report_id(report_id)
    user_id: str = current_user.get('_id') or current_user.get('user_id', '')
    client_id: str = current_user.get('client_id', '')
    if not client_id:
        raise HTTPException(status_code=400, detail='client_id is required')
    repo = DashboardRepository(db)
    deleted = await repo.delete_report(user_id, client_id, report_id)
    if not deleted:
        raise HTTPException(status_code=404, detail='Report not found')
    return {'message': 'Report deleted'}

@router.post('/reports/reorder')
async def reorder_reports(body: ReorderRequest, current_user: dict=Depends(require_auth()), db=Depends(get_db)) -> Dict[str, Any]:
    user_id: str = current_user.get('_id') or current_user.get('user_id', '')
    client_id: str = current_user.get('client_id', '')
    if not client_id:
        raise HTTPException(status_code=400, detail='client_id is required')
    for rid in body.ordered_ids:
        _validate_report_id(rid)
    repo = DashboardRepository(db)
    await repo.reorder_reports(user_id, client_id, body.ordered_ids)
    return {'message': 'Reports reordered'}

@router.patch('/reports/{report_id}/rename')
async def rename_report(report_id: str, body: RenameReportRequest, current_user: dict=Depends(require_auth()), db=Depends(get_db)) -> Dict[str, Any]:
    report_id = _validate_report_id(report_id)
    user_id: str = current_user.get('_id') or current_user.get('user_id', '')
    client_id: str = current_user.get('client_id', '')
    if not client_id:
        raise HTTPException(status_code=400, detail='client_id is required')
    repo = DashboardRepository(db)
    updated = await repo.update_report_fields(user_id, client_id, report_id, {'title': body.title.strip()})
    if not updated:
        raise HTTPException(status_code=404, detail='Report not found')
    return {'title': body.title.strip()}

@router.post('/reports/{report_id}/execute')
async def execute_report(report_id: str, current_user: dict=Depends(require_auth()), db=Depends(get_db)) -> Dict[str, Any]:
    report_id = _validate_report_id(report_id)
    user_id: str = current_user.get('_id') or current_user.get('user_id', '')
    client_id: str = current_user.get('client_id', '')
    if not client_id:
        raise HTTPException(status_code=400, detail='client_id is required')
    result = await dashboard_service.refresh_single_report(user_id=user_id, client_id=client_id, report_id=report_id, db=db)
    if result is None:
        raise HTTPException(status_code=404, detail='Report not found')
    return {'result': result.model_dump(mode='python')}

@router.post('/reports/refresh-all')
async def refresh_all(current_user: dict=Depends(require_auth()), db=Depends(get_db)) -> Dict[str, Any]:
    user_id: str = current_user.get('_id') or current_user.get('user_id', '')
    client_id: str = current_user.get('client_id', '')
    if not client_id:
        raise HTTPException(status_code=400, detail='client_id is required')
    updated_reports = await dashboard_service.refresh_all_reports(user_id=user_id, client_id=client_id, db=db)
    return {'reports': updated_reports, 'total': len(updated_reports)}

@router.get('/reports/{report_id}/insights/stream')
async def stream_report_insights(report_id: str, current_user: dict=Depends(require_auth_flexible()), db=Depends(get_db)):
    report_id = _validate_report_id(report_id)
    user_id: str = current_user.get('_id') or current_user.get('user_id', '')
    client_id: str = current_user.get('client_id', '')
    if not client_id:
        raise HTTPException(status_code=400, detail='client_id is required')
    repo = DashboardRepository(db)
    doc = await repo.get_by_user(user_id, client_id)
    if not doc:
        raise HTTPException(status_code=404, detail='Dashboard not found')
    report = next((r for r in doc.get('reports', []) if r.get('report_id') == report_id), None)
    if not report:
        raise HTTPException(status_code=404, detail='Report not found')
    last_result = report.get('last_result')
    if not last_result or last_result.get('status') == 'error':
        raise HTTPException(status_code=422, detail='Report has no successful execution result to generate insights from')
    input_query = report.get('cached_question') or report.get('original_query', '')
    from services.orchestrator_manager import OrchestratorManager
    orchestrator = OrchestratorManager()
    data = {'input': input_query, 'executor_response': last_result, 'user_id': user_id, 'client_id': client_id, 'session_id': f'dashboard_insights_{report_id}', 'report_id': report_id, 'db': db}
    generator = orchestrator.stream_business_only(data)
    headers = {'Cache-Control': 'no-cache', 'Connection': 'keep-alive'}
    return StreamingResponse(generator, media_type='text/event-stream', headers=headers)

@router.post('/reports/{report_id}/interact')
async def interact_query(report_id: str, body: InteractRequest, current_user: dict=Depends(require_auth()), db=Depends(get_db)) -> Dict[str, Any]:
    report_id = _validate_report_id(report_id)
    user_id: str = current_user.get('_id') or current_user.get('user_id', '')
    client_id: str = current_user.get('client_id', '')
    if not client_id:
        raise HTTPException(status_code=400, detail='client_id is required')
    result = await dashboard_service.run_interact_query(report_id=report_id, user_id=user_id, client_id=client_id, question=body.question.strip(), db=db)
    if result is None:
        raise HTTPException(status_code=404, detail='Report not found')
    return {'result': result.model_dump(mode='python')}

def _view_reports_client_id(current_user: dict) -> str:
    client_id = current_user.get('client_id', '')
    if not client_id:
        raise HTTPException(status_code=400, detail='client_id is required')
    return client_id

def _handle_view_reports_error(exc: LiveTableViewError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=exc.message)

@router.get('/view-reports/tables')
async def list_view_report_tables(current_user: dict=Depends(require_auth()), db=Depends(get_db)) -> Dict[str, Any]:
    client_id = _view_reports_client_id(current_user)
    try:
        svc = LiveTableViewService(db)
        tables = await svc.list_viewable_tables(client_id)
        return {'tables': tables, 'total': len(tables)}
    except LiveTableViewError as e:
        _handle_view_reports_error(e)

@router.post('/view-reports/query')
async def query_view_report_table(body: ViewReportQueryRequest, current_user: dict=Depends(require_auth()), db=Depends(get_db)) -> Dict[str, Any]:
    client_id = _view_reports_client_id(current_user)
    try:
        svc = LiveTableViewService(db)
        return await svc.query_table(client_id, body)
    except LiveTableViewError as e:
        _handle_view_reports_error(e)
    except Exception as e:
        logger.error('view-reports query failed: %s', e, exc_info=True)
        raise HTTPException(status_code=500, detail=f'Query failed: {e}')

@router.post('/view-reports/export')
async def export_view_report_table(body: ViewReportExportRequest, current_user: dict=Depends(require_auth()), db=Depends(get_db)):
    client_id = _view_reports_client_id(current_user)
    try:
        svc = LiveTableViewService(db)
        filename, stream = await svc.export_table_csv(client_id, body)
        headers = {'Content-Disposition': f'attachment; filename="{filename}"'}
        return StreamingResponse(stream, media_type='text/csv', headers=headers)
    except LiveTableViewError as e:
        _handle_view_reports_error(e)
    except Exception as e:
        logger.error('view-reports export failed: %s', e, exc_info=True)
        raise HTTPException(status_code=500, detail=f'Export failed: {e}')