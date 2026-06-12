from __future__ import annotations
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from middleware.auth_middleware import require_auth
from services.report_db_utils import ALL_SEGMENTS, parse_segments
from services import ai_insights_service
logger = logging.getLogger(__name__)
router = APIRouter(prefix='/insights', tags=['AI Insights'])

@router.get('/ribbon')
async def get_ribbon(business_month: str=Query(..., description='Business month in YYYY-MM'), segments: str=Query(default=ALL_SEGMENTS), report_date: Optional[str]=Query(default=None, description='Optional as-of date (YYYY-MM-DD)'), current_user: dict=Depends(require_auth())):
    try:
        seg_list = parse_segments(segments)
        return await ai_insights_service.compute_ribbon_insights(seg_list, business_month, report_date)
    except Exception as e:
        logger.exception('AI Insights ribbon error')
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/full')
async def get_full(business_month: str=Query(..., description='Business month in YYYY-MM'), segments: str=Query(default=ALL_SEGMENTS), report_date: Optional[str]=Query(default=None, description='Optional as-of date (YYYY-MM-DD)'), narrative: bool=Query(default=False, description='Include a composed narrative summary'), current_user: dict=Depends(require_auth())):
    try:
        seg_list = parse_segments(segments)
        return await ai_insights_service.compute_full_insights(seg_list, business_month, report_date, narrative=narrative)
    except Exception as e:
        logger.exception('AI Insights full error')
        raise HTTPException(status_code=500, detail=str(e))