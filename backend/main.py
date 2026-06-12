from __future__ import annotations
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path.cwd() / '.env', override=True)
from util.sshtunnel_compat import ensure_sshtunnel_compat
ensure_sshtunnel_compat()
from util.logging_config import setup_logging
setup_logging()
import logging
logging.getLogger(__name__).info('[startup] CWD=%s | .env path=%s', Path.cwd(), Path.cwd() / '.env')
from fastapi import FastAPI, Request, HTTPException, Response, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from db_config.database import lifespan
from agents.agents_routes import agents_router
from response_caching.cache_routes import cache_router

from util.metrics import metrics_collector
from middleware.auth_middleware import require_admin, require_auth
from middleware.rate_limit_middleware import rate_limit_middleware, rate_limiter
from middleware.security_headers_middleware import security_headers_middleware, initialize_security_headers, get_security_grade
from middleware.timezone_json_middleware import timezone_json_middleware
import asyncio
import json
import logging
import os
logger = logging.getLogger(__name__)
environment = os.getenv('ENVIRONMENT', 'production')
initialize_security_headers(environment)
logger.info(f'Security headers initialized for environment: {environment}')
_env = os.getenv('ENVIRONMENT', 'production')
app = FastAPI(title='CoreSight', description='Decision Support AI Assistant API', version='1.0.0', openapi_url='/api/openapi.json' if _env != 'production' else None, docs_url='/api/docs' if _env != 'production' else None, redoc_url='/api/redoc' if _env != 'production' else None, lifespan=lifespan)

@app.exception_handler(404)
async def not_found_error(request: Request, exc: HTTPException):
    content = {'message': 'Resource not found', 'detail': 'The requested resource could not be found'}
    return Response(content=json.dumps(content), status_code=404, media_type='application/json')

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    from util.error_handler import sanitize_error_message, generate_correlation_id
    correlation_id = generate_correlation_id()
    logger.warning(f'[{correlation_id}] HTTPException: {exc.status_code} - {exc.detail}')
    sanitized_detail = sanitize_error_message(str(exc.detail))
    return Response(content=json.dumps({'error': sanitized_detail, 'status_code': exc.status_code, 'correlation_id': correlation_id}), status_code=exc.status_code, media_type='application/json')

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    from util.error_handler import create_safe_error_response, generate_correlation_id
    correlation_id = generate_correlation_id()
    logger.error(f'[{correlation_id}] Unhandled exception: {exc}', exc_info=True)
    error_type = None
    exc_str = str(type(exc)).lower()
    if 'database' in exc_str or 'sql' in exc_str:
        error_type = 'database'
    elif 'file' in exc_str or 'path' in exc_str:
        error_type = 'file'
    elif 'validation' in exc_str:
        error_type = 'validation'
    elif 'auth' in exc_str:
        error_type = 'authentication'
    error_response = create_safe_error_response(exc, error_type=error_type, status_code=500, correlation_id=correlation_id)
    return Response(content=json.dumps(error_response), status_code=error_response['status_code'], media_type='application/json')

@app.get('/health')
async def health_check():
    try:
        security_grade = get_security_grade()
        return {'status': 'healthy', 'service': 'CoreSight API', 'version': '1.0.0', 'environment': environment, 'security': {'grade': security_grade['grade'], 'score': f"{security_grade['score']}/{security_grade['max_score']}", 'headers_present': security_grade['headers_present'], 'headers_missing': security_grade.get('missing_headers', [])}, 'features': {'multi_tenant': True, 'rate_limiting': True, 'audit_logging': True, 'security_headers': True, 'self_service_registration': True}}
    except Exception as e:
        logger.error(f'Health check failed: {e}')
        return Response(content=json.dumps({'status': 'unhealthy', 'error': str(e)}), status_code=500, media_type='application/json')
metrics_router = APIRouter(prefix='/api/metrics', tags=['Metrics'])

@metrics_router.get('/')
async def get_all_metrics(admin_user: dict=Depends(require_admin)):
    return metrics_collector.get_all_metrics()

@metrics_router.get('/client/{client_id}')
async def get_client_metrics(client_id: str, current_user: dict=Depends(require_auth())):
    if current_user.get('client_id') != client_id and current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail='Access denied')
    return metrics_collector.get_client_summary(client_id)

@metrics_router.get('/authentication')
async def get_auth_metrics(admin_user: dict=Depends(require_admin)):
    return metrics_collector.get_authentication_stats()

@metrics_router.get('/queries')
async def get_query_metrics(admin_user: dict=Depends(require_admin)):
    return metrics_collector.get_query_stats()

@metrics_router.get('/prompts')
async def get_prompt_metrics(admin_user: dict=Depends(require_admin)):
    return metrics_collector.get_prompt_stats()

@metrics_router.get('/database')
async def get_database_metrics(admin_user: dict=Depends(require_admin)):
    return metrics_collector.get_database_stats()

@metrics_router.post('/reset')
async def reset_metrics(admin_user: dict=Depends(require_admin)):
    metrics_collector.reset_metrics()
    logger.info(f"Metrics reset by admin | admin_email={admin_user.get('email')}")
    return {'status': 'success', 'message': 'Metrics reset'}
timezone_router = APIRouter(prefix='/api', tags=['Client timezone'])

@timezone_router.post('/set-client-timezone')
async def set_client_timezone(request: Request, response: Response):
    try:
        body = await request.json()
        tz = (body.get('tz') or '').strip()
    except Exception:
        tz = request.query_params.get('tz') or request.query_params.get('timezone') or ''
        tz = (tz or '').strip()
    if not tz:
        return {'ok': True, 'message': 'No timezone provided; UTC will be used'}
    if len(tz) > 64 or not all((c.isalnum() or c in '/_-' for c in tz)):
        raise HTTPException(status_code=400, detail='Invalid timezone')
    is_production = os.getenv('ENVIRONMENT') == 'production'
    response.set_cookie(key='client_timezone', value=tz, path='/', max_age=365 * 24 * 60 * 60, samesite='none' if is_production else 'lax', secure=is_production, httponly=True)
    return {'ok': True, 'tz': tz}
app.include_router(timezone_router)
app.include_router(agents_router, prefix='/api/agents')
app.include_router(cache_router, prefix='/api/cache')
app.include_router(metrics_router)
from middleware.monitoring_middleware import MonitoringMiddleware
app.add_middleware(MonitoringMiddleware)
app.middleware('http')(security_headers_middleware)
app.middleware('http')(rate_limit_middleware)
app.middleware('http')(timezone_json_middleware)
frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
environment = os.getenv('ENVIRONMENT', 'development')
cors_allow_origins = [frontend_url]
if environment != 'production':
    cors_allow_origins.extend(['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:8024', 'http://127.0.0.1:8024'])
cors_allow_headers = ['Authorization', 'Content-Type', 'X-Requested-With', 'Accept', 'Accept-Language', 'Origin']
app.add_middleware(CORSMiddleware, allow_origins=cors_allow_origins, allow_credentials=True, allow_methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'], allow_headers=cors_allow_headers)
_warm_pool_manager = None

@app.on_event('startup')
async def startup_event():
    global _warm_pool_manager
    from services.monitoring_service import monitoring_service
    required_secrets = {'SECRET_KEY': {'required': True, 'min_length': 32, 'description': 'JWT token signing key'}, 'DB_CREDENTIALS_ENCRYPTION_KEY': {'required': True, 'min_length': 32, 'description': 'Database credentials encryption key'}}
    missing_secrets = []
    weak_secrets = []
    for secret_name, config in required_secrets.items():
        secret_value = os.getenv(secret_name)
        if config['required']:
            if not secret_value:
                missing_secrets.append(f"{secret_name} ({config['description']})")
            elif len(secret_value) < config.get('min_length', 0):
                weak_secrets.append(f"{secret_name} is too short (min {config['min_length']} chars)")
    if missing_secrets:
        error_msg = f'CRITICAL: Missing required environment variables:\n' + '\n'.join((f'  - {s}' for s in missing_secrets))
        logger.error(error_msg)
        raise ValueError(error_msg)
    if weak_secrets:
        warning_msg = f'WARNING: Weak environment variables detected:\n' + '\n'.join((f'  - {s}' for s in weak_secrets))
        logger.warning(warning_msg)
    secret_key = os.getenv('SECRET_KEY', '')
    default_secret = 'q9Zf7uHk3pL_6sV8rX0bNf2yWz4aTq1JcM5Gv8Yp4sZf7uHk3pL'
    if secret_key == default_secret:
        logger.warning('SECRET_KEY is using default value! Change this in production!')
    logger.info('CoreSight application starting up')
    logger.info('Multi-tenant framework initialized')
    logger.info('Metrics collector ready')
    logger.info('Rate limiting middleware enabled')
    logger.info('Monitoring middleware enabled')
    try:
        from db_config.mongo_server import get_db
        db = await get_db()
        await db.command('ping')
        logger.info('MongoDB connection pre-warmed and ready')
    except Exception as e:
        logger.error(f'Failed to pre-warm MongoDB connection: {e}')
    import asyncio
    asyncio.create_task(monitoring_service.start_background_monitoring(interval=15))
    logger.info('Security headers middleware enabled')
    try:
        from domains.conversation.service import ConversationService
        from config.system_config import ENABLE_BACKGROUND_JOBS, BACKGROUND_JOB_CONFIG
        if ENABLE_BACKGROUND_JOBS:
            conv_service = ConversationService(db)
            stale_count = await conv_service.mark_stale_background_failed(stale_threshold_minutes=BACKGROUND_JOB_CONFIG['stale_threshold_minutes'])
            logger.info(f'Background conversations initialized: {stale_count} stale entries cleaned up')
    except Exception as e:
        logger.error(f'Failed to initialize background conversations: {e}', exc_info=True)
    try:
        from domains.dashboard.repository import DashboardRepository
        dashboard_repo = DashboardRepository(db)
        await dashboard_repo.ensure_indexes()
        logger.info('Dashboard reports indexes ensured')
    except Exception as e:
        logger.error(f'Failed to initialize dashboard indexes: {e}', exc_info=True)
    try:
        from notifications.notification_service import ensure_indexes as ensure_notification_indexes
        await ensure_notification_indexes()
    except Exception as e:
        logger.error(f'Failed to initialize notification indexes: {e}', exc_info=True)

    async def _adhoc_cleanup_loop():
        from services.adhoc_file_service import cleanup_expired_files
        while True:
            await asyncio.sleep(1800)
            try:
                removed = cleanup_expired_files()
                if removed:
                    logger.info(f'Ad-hoc cleanup: removed {removed} expired file directories')
            except Exception as e:
                logger.warning(f'Ad-hoc cleanup failed: {e}')
    asyncio.create_task(_adhoc_cleanup_loop())
    logger.info('Ad-hoc file cleanup scheduled (every 30 min, TTL from ADHOC_FILE_TTL_HOURS)')

    async def _session_kernel_cleanup_loop():
        from util.session_kernel_store import cleanup_idle_session_kernels
        while True:
            await asyncio.sleep(300)
            try:
                await cleanup_idle_session_kernels()
            except Exception as e:
                logger.warning(f'Session kernel cleanup failed: {e}')
    asyncio.create_task(_session_kernel_cleanup_loop())
    logger.info('Session kernel cleanup scheduled (every 5 min, idle timeout from SESSION_KERNEL_IDLE_TIMEOUT)')
    try:
        grade_info = get_security_grade()
        logger.info(f"Security headers grade: {grade_info['grade']} ({grade_info['score']}/{grade_info['max_score']} - {grade_info['percentage']:.1f}%)")
    except Exception as e:
        logger.warning(f'Failed to calculate security grade: {e}')
    try:
        from util.warm_pool_manager import WarmPoolManager
        _warm_pool_manager = WarmPoolManager()
        await _warm_pool_manager.start()
        logger.info('WarmPoolManager started')
    except Exception as e:
        logger.warning(f'WarmPoolManager failed to start (non-fatal): {e}')

    async def _boot_kernel_prewarm():
        try:
            from util.kernel_manager import get_kernel_manager, release_kernel_manager
            from config.system_config import AGENT_CONFIG
            idle_timeout = float(AGENT_CONFIG.get('data_science_agent', {}).get('idle_timeout_minutes', 30.0))
            logger.info('Boot kernel pre-warm: starting Jupyter kernel...')
            mgr = await get_kernel_manager(client_id='__boot_prewarm__', idle_timeout_minutes=idle_timeout, use_docker=False)
            success = await mgr.start()
            if success:
                await release_kernel_manager(mgr, use_pool=True)
                logger.info('Boot kernel pre-warm: kernel ready and added to warm pool')
            else:
                logger.warning('Boot kernel pre-warm: kernel start failed')
        except Exception as e:
            logger.warning(f'Boot kernel pre-warm failed (non-fatal): {e}')
    asyncio.create_task(_boot_kernel_prewarm())

@app.on_event('shutdown')
async def shutdown_event():
    global _warm_pool_manager
    try:
        from util.session_kernel_store import cleanup_all_session_kernels
        await cleanup_all_session_kernels()
    except Exception as e:
        logger.warning(f'Session kernel shutdown cleanup error: {e}')
    if _warm_pool_manager is not None:
        try:
            await _warm_pool_manager.stop()
        except Exception as e:
            logger.warning(f'WarmPoolManager stop error: {e}')
    try:
        from util.kernel_manager import _kernel_pool, _get_pool_lock
        async with _get_pool_lock():
            total_stopped = 0
            for client_id, pool in list(_kernel_pool.items()):
                for mgr in pool:
                    try:
                        await asyncio.wait_for(mgr.stop(), timeout=10)
                        total_stopped += 1
                    except Exception as e:
                        logger.warning(f"Error stopping kernel {getattr(mgr, 'container_name', '?')}: {e}")
                pool.clear()
            _kernel_pool.clear()
            if total_stopped:
                logger.info(f'Graceful shutdown: stopped {total_stopped} kernel manager(s)')
    except Exception as e:
        logger.warning(f'Error during kernel pool cleanup: {e}')
    try:
        metrics_collector.export_to_json('metrics_snapshot.json')
        logger.info('Metrics snapshot exported successfully')
    except Exception as e:
        logger.error(f'Failed to export metrics snapshot: {e}', exc_info=True)
    try:
        await rate_limiter.close()
        logger.info('Rate limiter connection closed')
    except Exception as e:
        logger.error(f'Failed to close rate limiter: {e}', exc_info=True)
    logger.info('CoreSight application shutting down')