from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from datetime import datetime
import logging
import time
from services.monitoring_service import monitoring_service
logger = logging.getLogger(__name__)

class MonitoringMiddleware(BaseHTTPMiddleware):

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        logger.info('Monitoring middleware initialized')

    async def dispatch(self, request: Request, call_next):
        method = request.method
        path = request.url.path
        client_id = getattr(request.state, 'client_id', 'unknown')
        start_time = time.time()
        with monitoring_service.track_http_in_progress(method, path):
            try:
                response = await call_next(request)
                duration = time.time() - start_time
                monitoring_service.track_http_request(method=method, endpoint=path, status_code=response.status_code, duration=duration, client_id=client_id)
                return response
            except Exception as e:
                duration = time.time() - start_time
                monitoring_service.track_http_request(method=method, endpoint=path, status_code=500, duration=duration, client_id=client_id)
                logger.error(f'Error processing request {method} {path}: {e}')
                raise