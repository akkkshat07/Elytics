from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class DashboardDataframe(BaseModel):
    name: str
    json_data: Optional[str] = None
    html_data: Optional[str] = None
    column_mapping: Dict[str, str] = Field(default_factory=dict)
    column_metadata: Dict[str, str] = Field(default_factory=dict)

class DashboardPlotlyChart(BaseModel):
    name: str
    figure: str

class DashboardTextOutput(BaseModel):
    name: str
    value: str

class DashboardReportResult(BaseModel):
    dataframes: List[Dict[str, Any]] = Field(default_factory=list)
    plotly_charts: List[Dict[str, Any]] = Field(default_factory=list)
    text_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    executed_at: datetime
    status: str = 'success'
    error: Optional[str] = None

    class Config:
        extra = 'allow'

class DashboardReport(BaseModel):
    report_id: str
    title: str
    description: Optional[str] = None
    original_query: str
    cached_question: Optional[str] = None
    code: str
    dataset_id: Optional[str] = None
    source_run_id: Optional[str] = None
    source_conversation_id: Optional[str] = None
    order: int = 0
    created_at: datetime
    updated_at: datetime
    last_result: Optional[DashboardReportResult] = None

    class Config:
        extra = 'allow'

class DashboardDocument(BaseModel):
    user_id: str
    client_id: str
    reports: List[DashboardReport] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    class Config:
        extra = 'allow'
        populate_by_name = True