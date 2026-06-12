import json
import logging
from fastapi import APIRouter, HTTPException, Depends, Body, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import tempfile
from pathlib import Path
from agents.data_science_agent import DataScienceAgent
from auth.auth import get_current_user
logger = logging.getLogger(__name__)
data_science_router = APIRouter(prefix='/api/data-science', tags=['Data Science'])

class DataScienceQuery(BaseModel):
    query: str = Field(..., description='The data science question or task')
    dataset_path: Optional[str] = Field(None, description='Path to dataset file')
    context: Optional[Dict[str, Any]] = Field(None, description='Additional context')
    max_iterations: Optional[int] = Field(5, description='Maximum iterations')

@data_science_router.post('/analyze')
async def analyze_data(request: DataScienceQuery, current_user: dict=Depends(get_current_user)) -> StreamingResponse:

    async def event_generator():
        try:
            agent = DataScienceAgent(client_id=current_user.get('client_id', 'default'), db=None)
            async for event in agent.execute_analysis(user_query=request.query, dataset_path=request.dataset_path, context=request.context):
                yield f'data: {json.dumps(event)}\n\n'
        except Exception as e:
            logger.error(f'Error in analyze_data: {e}')
            yield f"data: {json.dumps({'type': 'error', 'content': {'message': str(e)}})}\n\n"
    return StreamingResponse(event_generator(), media_type='text/event-stream')

@data_science_router.post('/analyze-with-upload')
async def analyze_with_upload(file: UploadFile=File(...), query: str=Body(..., description='Analysis query'), current_user: dict=Depends(get_current_user)) -> StreamingResponse:

    async def event_generator():
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                content = await file.read()
                tmp.write(content)
                dataset_path = tmp.name
            agent = DataScienceAgent(client_id=current_user.get('client_id', 'default'), db=None)
            async for event in agent.execute_analysis(user_query=query, dataset_path=dataset_path):
                yield f'data: {json.dumps(event)}\n\n'
        except Exception as e:
            logger.error(f'Error in analyze_with_upload: {e}')
            yield f"data: {json.dumps({'type': 'error', 'content': {'message': str(e)}})}\n\n"
        finally:
            if 'dataset_path' in locals():
                Path(dataset_path).unlink(missing_ok=True)
    return StreamingResponse(event_generator(), media_type='text/event-stream')