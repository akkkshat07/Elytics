"""Lead Indicators Report API routes.

Static report endpoints that query PostgreSQL lead_indicators_master tables.
All endpoints are authenticated via require_auth.

Prefix: /api/lead-indicators  (registered in main.py)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from middleware.auth_middleware import require_auth
from services.report_db_utils import ALL_SEGMENTS, parse_segments
from services import lead_indicators_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/lead-indicators", tags=["Lead Indicators"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/metadata")
async def get_metadata(
    segments: str = Query(default=ALL_SEGMENTS, description="Comma-separated: ne_retail,ne_parlor,bh"),
    current_user: dict = Depends(require_auth()),
):
    """Return available business months, zones, and state groups."""
    try:
        seg_list = parse_segments(segments)
        return await lead_indicators_service.get_metadata(seg_list)
    except Exception as e:
        logger.exception("Lead Indicators metadata error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_summary(
    business_month: str = Query(..., description="Business month in YYYY-MM format"),
    segments: str = Query(default=ALL_SEGMENTS),
    current_user: dict = Depends(require_auth()),
):
    """Return aggregate KPIs for the given segments and business month."""
    try:
        seg_list = parse_segments(segments)
        return await lead_indicators_service.get_summary(seg_list, business_month)
    except Exception as e:
        logger.exception("Lead Indicators summary error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state-wise")
async def get_state_wise(
    business_month: str = Query(..., description="Business month in YYYY-MM format"),
    segments: str = Query(default=ALL_SEGMENTS),
    current_user: dict = Depends(require_auth()),
):
    """Return state-level aggregated lead indicators."""
    try:
        seg_list = parse_segments(segments)
        return await lead_indicators_service.get_state_wise(seg_list, business_month)
    except Exception as e:
        logger.exception("Lead Indicators state-wise error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/asm-wise")
async def get_asm_wise(
    business_month: str = Query(..., description="Business month in YYYY-MM format"),
    segments: str = Query(default=ALL_SEGMENTS),
    state: Optional[str] = Query(default=None, description="Optional state filter"),
    current_user: dict = Depends(require_auth()),
):
    """Return ASM/TeamLeader-level aggregated lead indicators."""
    try:
        seg_list = parse_segments(segments)
        return await lead_indicators_service.get_asm_wise(seg_list, business_month, state)
    except Exception as e:
        logger.exception("Lead Indicators asm-wise error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tso-tsi-wise")
async def get_tso_tsi_wise(
    business_month: str = Query(..., description="Business month in YYYY-MM format"),
    segments: str = Query(default=ALL_SEGMENTS),
    asm: Optional[str] = Query(default=None, description="Optional ASM/TeamLeader filter"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=10, le=200),
    search: Optional[str] = Query(default=None, description="Search by employee name or UUID"),
    current_user: dict = Depends(require_auth()),
):
    """Return employee-level detail with pagination and optional filters."""
    try:
        seg_list = parse_segments(segments)
        return await lead_indicators_service.get_tso_tsi_wise(
            seg_list, business_month, asm, page, page_size, search
        )
    except Exception as e:
        logger.exception("Lead Indicators tso-tsi-wise error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/day-wise")
async def get_day_wise(
    business_month: str = Query(...),
    segments: str = Query(default=ALL_SEGMENTS),
    zone: Optional[str] = Query(default=None),
    current_user: dict = Depends(require_auth()),
):
    """Return day-by-day TC, PC, P3/P2 NSV, ECO from dsr_master."""
    try:
        seg_list = parse_segments(segments)
        return await lead_indicators_service.get_day_wise(seg_list, business_month, zone)
    except Exception as e:
        logger.exception("Lead Indicators day-wise error")
        raise HTTPException(status_code=500, detail=str(e))
