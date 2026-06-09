from dotenv import load_dotenv
from pathlib import Path
# Load .env FIRST before importing any modules that use environment variables.
# Path.cwd() is used instead of Path(__file__) because __file__ resolution is
# unreliable inside Nuitka-compiled .so modules. systemd's WorkingDirectory
# guarantees CWD is the project root, so the .env path is always correct.
load_dotenv(Path.cwd() / ".env", override=True)

from util.sshtunnel_compat import ensure_sshtunnel_compat

ensure_sshtunnel_compat()

# Initialize structured JSON logging for GCP Cloud Logging BEFORE any other imports
from util.logging_config import setup_logging
setup_logging()

import logging
logging.getLogger(__name__).info(
    "[startup] CWD=%s | .env path=%s", Path.cwd(), Path.cwd() / ".env"
)

from fastapi import FastAPI, Request, HTTPException, Response, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from db_config.database import lifespan
# from users.user_route import user_router
from agents.agents_routes import agents_router
from response_caching.cache_routes import cache_router
from admin.admin_routes import router as admin_router
from admin.super_admin_routes import router as super_admin_router
from auth.registration_routes import router as registration_router
from util.metrics import metrics_collector
from middleware.auth_middleware import require_admin, require_auth
from middleware.rate_limit_middleware import rate_limit_middleware, rate_limiter
from middleware.security_headers_middleware import (  # SECURITY: Phase 3 Task 3.2
    security_headers_middleware,
    initialize_security_headers,
    get_security_grade
)
from middleware.timezone_json_middleware import timezone_json_middleware
import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)

# Initialize security headers (Phase 3 Task 3.2)
environment = os.getenv("ENVIRONMENT", "production")
initialize_security_headers(environment)
logger.info(f"Security headers initialized for environment: {environment}")
# Create app instance
_env = os.getenv("ENVIRONMENT", "production")
app = FastAPI(
    title="CoreSight",
    description="Decision Support AI Assistant API",
    version="1.0.0",
    openapi_url="/api/openapi.json" if _env != "production" else None,
    docs_url="/api/docs" if _env != "production" else None,
    redoc_url="/api/redoc" if _env != "production" else None,
    lifespan=lifespan
)

# Add 404 error handler at application level
@app.exception_handler(404)
async def not_found_error(request: Request, exc: HTTPException):
    content = {
        "message": "Resource not found",
        "detail": "The requested resource could not be found"
    }
    return Response(
        content=json.dumps(content),
        status_code=404,
        media_type="application/json"
    )

# Add HTTPException handler to sanitize error messages
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    HTTPException handler that sanitizes error messages in production.
    Returns a correlation_id so clients can reference server-side logs.
    """
    from util.error_handler import sanitize_error_message, generate_correlation_id

    correlation_id = generate_correlation_id()

    # Log full original detail server-side with correlation ID
    logger.warning(f"[{correlation_id}] HTTPException: {exc.status_code} - {exc.detail}")

    # Sanitize before sending to client
    sanitized_detail = sanitize_error_message(str(exc.detail))

    return Response(
        content=json.dumps({
            "error": sanitized_detail,
            "status_code": exc.status_code,
            "correlation_id": correlation_id,
        }),
        status_code=exc.status_code,
        media_type="application/json"
    )

# Add global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to sanitize all unhandled exceptions.
    Prevents information disclosure in production.
    Returns a correlation_id so clients can reference server-side logs.
    """
    from util.error_handler import create_safe_error_response, generate_correlation_id

    correlation_id = generate_correlation_id()

    # Log full error server-side with correlation ID
    logger.error(f"[{correlation_id}] Unhandled exception: {exc}", exc_info=True)

    # Determine error type for generic message selection
    error_type = None
    exc_str = str(type(exc)).lower()
    if "database" in exc_str or "sql" in exc_str:
        error_type = "database"
    elif "file" in exc_str or "path" in exc_str:
        error_type = "file"
    elif "validation" in exc_str:
        error_type = "validation"
    elif "auth" in exc_str:
        error_type = "authentication"

    error_response = create_safe_error_response(
        exc, error_type=error_type, status_code=500, correlation_id=correlation_id
    )

    return Response(
        content=json.dumps(error_response),
        status_code=error_response["status_code"],
        media_type="application/json"
    )

# Health check endpoint with security validation
@app.get("/health")
async def health_check():
    """
    Health check endpoint with security headers validation.
    
    Returns application health status and security posture.
    """
    try:
        security_grade = get_security_grade()
        
        return {
            "status": "healthy",
            "service": "CoreSight API",
            "version": "1.0.0",
            "environment": environment,
            "security": {
                "grade": security_grade["grade"],
                "score": f"{security_grade['score']}/{security_grade['max_score']}",
                "headers_present": security_grade["headers_present"],
                "headers_missing": security_grade.get("missing_headers", [])
            },
            "features": {
                "multi_tenant": True,
                "rate_limiting": True,
                "audit_logging": True,
                "security_headers": True,
                "self_service_registration": True
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return Response(
            content=json.dumps({"status": "unhealthy", "error": str(e)}),
            status_code=500,
            media_type="application/json"
        )

# Create metrics router
metrics_router = APIRouter(prefix="/api/metrics", tags=["Metrics"])

@metrics_router.get("/")
async def get_all_metrics(admin_user: dict = Depends(require_admin)):
    """Get all metrics (admin only)"""
    return metrics_collector.get_all_metrics()

@metrics_router.get("/client/{client_id}")
async def get_client_metrics(
    client_id: str,
    current_user: dict = Depends(require_auth())
):
    """Get metrics for specific client"""
    # Verify user has access to this client
    if current_user.get("client_id") != client_id and current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    return metrics_collector.get_client_summary(client_id)

@metrics_router.get("/authentication")
async def get_auth_metrics(admin_user: dict = Depends(require_admin)):
    """Get authentication metrics"""
    return metrics_collector.get_authentication_stats()

@metrics_router.get("/queries")
async def get_query_metrics(admin_user: dict = Depends(require_admin)):
    """Get query metrics"""
    return metrics_collector.get_query_stats()

@metrics_router.get("/prompts")
async def get_prompt_metrics(admin_user: dict = Depends(require_admin)):
    """Get prompt loading metrics"""
    return metrics_collector.get_prompt_stats()

@metrics_router.get("/database")
async def get_database_metrics(admin_user: dict = Depends(require_admin)):
    """Get database metrics"""
    return metrics_collector.get_database_stats()

@metrics_router.post("/reset")
async def reset_metrics(admin_user: dict = Depends(require_admin)):
    """Reset all metrics (admin only)"""
    metrics_collector.reset_metrics()
    logger.info(f"Metrics reset by admin | admin_email={admin_user.get('email')}")
    return {"status": "success", "message": "Metrics reset"}

from explorer.explorer_routes import router as explorer_router
from api.db_credentials_routes import router as db_credentials_router
from api.uploaded_data_routes import router as uploaded_data_router
from api.upgrade_request_routes import router as upgrade_request_router
from admin.llm_config_routes import create_llm_config_router
from admin.custom_prompts_routes import router as custom_prompts_router
from agents.data_science_routes import data_science_router
from agents.dashboard_routes import router as dashboard_router
from agents.dsr_routes import router as dsr_router
from agents.focus_npd_routes import router as focus_npd_router
from agents.lead_indicators_routes import router as lead_indicators_router
from agents.category_wise_routes import router as category_wise_router
from agents.ai_insights_routes import router as ai_insights_router
from notifications.notification_routes import router as notification_router

# Timezone cookie endpoint (public, no auth) - frontend calls on load so API responses use client timezone
timezone_router = APIRouter(prefix="/api", tags=["Client timezone"])


@timezone_router.post("/set-client-timezone")
async def set_client_timezone(request: Request, response: Response):
    """
    Set client timezone in a cookie. Frontend calls this on app load with credentials.
    Middleware reads the cookie and converts JSON datetime fields to this timezone.
    """
    try:
        body = await request.json()
        tz = (body.get("tz") or "").strip()
    except Exception:
        tz = request.query_params.get("tz") or request.query_params.get("timezone") or ""
        tz = (tz or "").strip()
    if not tz:
        return {"ok": True, "message": "No timezone provided; UTC will be used"}
    if len(tz) > 64 or not all(c.isalnum() or c in "/_-" for c in tz):
        raise HTTPException(status_code=400, detail="Invalid timezone")
    is_production = os.getenv("ENVIRONMENT") == "production"
    response.set_cookie(
        key="client_timezone",
        value=tz,
        path="/",
        max_age=365 * 24 * 60 * 60,
        samesite="none" if is_production else "lax",
        secure=is_production,
        httponly=True,
    )
    return {"ok": True, "tz": tz}


# Mount routers
# Public routes (no authentication required)
app.include_router(registration_router, prefix="/api")  # Provides /api/public/register
app.include_router(timezone_router)  # POST /api/set-client-timezone (public)

# Protected routes (authentication required)
# app.include_router(user_router, prefix="/api/users")
app.include_router(agents_router, prefix="/api/agents")
app.include_router(cache_router, prefix="/api/cache")
app.include_router(admin_router, prefix="/api/admin")
app.include_router(custom_prompts_router, prefix="/api/admin")  # Custom prompts management
app.include_router(super_admin_router, prefix="/api/super-admin")  # Super admin routes
app.include_router(metrics_router)
app.include_router(explorer_router, prefix="/api/explorer", tags=["Explorer"])
app.include_router(db_credentials_router, prefix="/api", tags=["DB Credentials"])
app.include_router(uploaded_data_router, prefix="/api")
app.include_router(upgrade_request_router, prefix="/api")
app.include_router(create_llm_config_router())  # LLM configuration routes
app.include_router(data_science_router)
app.include_router(dashboard_router, prefix="/api")
app.include_router(dsr_router, prefix="/api")
app.include_router(focus_npd_router, prefix="/api")
app.include_router(lead_indicators_router, prefix="/api")
app.include_router(category_wise_router, prefix="/api")
app.include_router(ai_insights_router, prefix="/api")
app.include_router(notification_router)

# Middleware execution order (Starlette LIFO for app.middleware("http")):
#   1. CORS       (added last via add_middleware → wraps outermost)
#   2. Rate limit  (added third via app.middleware → runs first of the http stack)
#   3. Security headers
#   4. Monitoring  (added first via add_middleware → innermost)
#
# Rate limiting intentionally runs BEFORE route-level auth (Depends).
# It extracts identity from the JWT token internally.

from middleware.monitoring_middleware import MonitoringMiddleware
app.add_middleware(MonitoringMiddleware)

app.middleware("http")(security_headers_middleware)
app.middleware("http")(rate_limit_middleware)

# Add timezone conversion middleware - FOURTH
app.middleware("http")(timezone_json_middleware)

# Configure CORS - restrict to specific origins for security
# Allow frontend URL from environment, with local dev origins in non-production
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
environment = os.getenv("ENVIRONMENT", "development")

cors_allow_origins = [frontend_url]
# Allow local dev origins in non-production for convenience
if environment != "production":
    cors_allow_origins.extend([
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8024",  # Backend dev server
        "http://127.0.0.1:8024"
    ])

# Explicit allowed headers (avoid allow_headers="*" for security)
# X-Client-Timezone omitted; timezone is sent via cookie (set-client-timezone) so no CORS preflight for custom header
# Accept-Language included so browser-sent preflights (common in production) succeed
cors_allow_headers = [
    "Authorization",
    "Content-Type",
    "X-Requested-With",
    "Accept",
    "Accept-Language",
    "Origin",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=True,  # Required for client_timezone cookie to be sent cross-origin
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=cors_allow_headers,
)

# Startup and shutdown events
_warm_pool_manager = None

@app.on_event("startup")
async def startup_event():
    """Application startup with security validation"""
    global _warm_pool_manager
    from services.monitoring_service import monitoring_service
    
    # SECURITY: Validate required environment variables
    required_secrets = {
        "SECRET_KEY": {
            "required": True,
            "min_length": 32,
            "description": "JWT token signing key"
        },
        "DB_CREDENTIALS_ENCRYPTION_KEY": {
            "required": True,
            "min_length": 32,
            "description": "Database credentials encryption key"
        }
    }
    
    missing_secrets = []
    weak_secrets = []
    
    for secret_name, config in required_secrets.items():
        secret_value = os.getenv(secret_name)
        
        if config["required"]:
            if not secret_value:
                missing_secrets.append(f"{secret_name} ({config['description']})")
            elif len(secret_value) < config.get("min_length", 0):
                weak_secrets.append(f"{secret_name} is too short (min {config['min_length']} chars)")
    
    if missing_secrets:
        error_msg = f"CRITICAL: Missing required environment variables:\n" + "\n".join(f"  - {s}" for s in missing_secrets)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if weak_secrets:
        warning_msg = f"WARNING: Weak environment variables detected:\n" + "\n".join(f"  - {s}" for s in weak_secrets)
        logger.warning(warning_msg)
        # Don't fail startup for weak secrets, but log warning
    
    # Validate SECRET_KEY is not the default
    secret_key = os.getenv("SECRET_KEY", "")
    default_secret = "q9Zf7uHk3pL_6sV8rX0bNf2yWz4aTq1JcM5Gv8Yp4sZf7uHk3pL"
    if secret_key == default_secret:
        logger.warning("SECRET_KEY is using default value! Change this in production!")
    
    logger.info("CoreSight application starting up")
    logger.info("Multi-tenant framework initialized")
    logger.info("Metrics collector ready")
    logger.info("Rate limiting middleware enabled")
    logger.info("Monitoring middleware enabled")
    
    # Pre-warm MongoDB connection for faster first login
    try:
        from db_config.mongo_server import get_db
        db = await get_db()
        await db.command("ping")
        logger.info("MongoDB connection pre-warmed and ready")
    except Exception as e:
        logger.error(f"Failed to pre-warm MongoDB connection: {e}")
    
    # Start background monitoring
    import asyncio
    asyncio.create_task(monitoring_service.start_background_monitoring(interval=15))
    logger.info("Security headers middleware enabled")

    # Background conversations: clean up stale ones from previous crashes.
    # NOTE: background conversation indexes are now managed centrally in db_config/indexes.py
    # (idx_conv_client_bg_status, idx_conv_client_user_bg_status).
    # PROD ACTION REQUIRED (one-time): drop the old unnamed system-generated indexes on the
    # conversations collection that were created before this change:
    #   db.conversations.dropIndex("client_id_1_is_background_1_status_1")
    #   db.conversations.dropIndex("client_id_1_user_id_1_is_background_1_status_1")
    try:
        from domains.conversation.service import ConversationService
        from config.system_config import ENABLE_BACKGROUND_JOBS, BACKGROUND_JOB_CONFIG

        if ENABLE_BACKGROUND_JOBS:
            conv_service = ConversationService(db)
            stale_count = await conv_service.mark_stale_background_failed(
                stale_threshold_minutes=BACKGROUND_JOB_CONFIG["stale_threshold_minutes"]
            )
            logger.info(
                f"Background conversations initialized: {stale_count} stale entries cleaned up"
            )
    except Exception as e:
        logger.error(f"Failed to initialize background conversations: {e}", exc_info=True)
    
    # Dashboard reports: ensure indexes
    try:
        from domains.dashboard.repository import DashboardRepository
        dashboard_repo = DashboardRepository(db)
        await dashboard_repo.ensure_indexes()
        logger.info("Dashboard reports indexes ensured")
    except Exception as e:
        logger.error(f"Failed to initialize dashboard indexes: {e}", exc_info=True)

    # Notification center: ensure indexes
    try:
        from notifications.notification_service import ensure_indexes as ensure_notification_indexes
        await ensure_notification_indexes()
    except Exception as e:
        logger.error(f"Failed to initialize notification indexes: {e}", exc_info=True)

    # Schedule periodic ad-hoc file cleanup (every 30 min)
    async def _adhoc_cleanup_loop():
        from services.adhoc_file_service import cleanup_expired_files
        while True:
            await asyncio.sleep(1800)  # 30 minutes
            try:
                removed = cleanup_expired_files()
                if removed:
                    logger.info(f"Ad-hoc cleanup: removed {removed} expired file directories")
            except Exception as e:
                logger.warning(f"Ad-hoc cleanup failed: {e}")

    asyncio.create_task(_adhoc_cleanup_loop())
    logger.info("Ad-hoc file cleanup scheduled (every 30 min, TTL from ADHOC_FILE_TTL_HOURS)")

    # Schedule periodic session kernel cleanup (every 5 min)
    async def _session_kernel_cleanup_loop():
        from util.session_kernel_store import cleanup_idle_session_kernels
        while True:
            await asyncio.sleep(300)  # 5 minutes
            try:
                await cleanup_idle_session_kernels()
            except Exception as e:
                logger.warning(f"Session kernel cleanup failed: {e}")

    asyncio.create_task(_session_kernel_cleanup_loop())
    logger.info("Session kernel cleanup scheduled (every 5 min, idle timeout from SESSION_KERNEL_IDLE_TIMEOUT)")

    # Log security grade
    try:
        grade_info = get_security_grade()
        logger.info(
            f"Security headers grade: {grade_info['grade']} "
            f"({grade_info['score']}/{grade_info['max_score']} - {grade_info['percentage']:.1f}%)"
        )
    except Exception as e:
        logger.warning(f"Failed to calculate security grade: {e}")

    # Start pre-warm pool manager (K8s only — no-op on other backends)
    try:
        from util.warm_pool_manager import WarmPoolManager
        _warm_pool_manager = WarmPoolManager()
        await _warm_pool_manager.start()
        logger.info("WarmPoolManager started")
    except Exception as e:
        logger.warning(f"WarmPoolManager failed to start (non-fatal): {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown - export metrics snapshot and cleanup"""
    global _warm_pool_manager

    # Stop all session kernels before releasing the global pool
    try:
        from util.session_kernel_store import cleanup_all_session_kernels
        await cleanup_all_session_kernels()
    except Exception as e:
        logger.warning(f"Session kernel shutdown cleanup error: {e}")

    # Stop pre-warm pool manager
    if _warm_pool_manager is not None:
        try:
            await _warm_pool_manager.stop()
        except Exception as e:
            logger.warning(f"WarmPoolManager stop error: {e}")

    # FIX #8: Gracefully stop all kernel managers in the pool to prevent
    # orphan K8s pods and leaked semaphore slots.
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
                logger.info(f"Graceful shutdown: stopped {total_stopped} kernel manager(s)")
    except Exception as e:
        logger.warning(f"Error during kernel pool cleanup: {e}")

    try:
        metrics_collector.export_to_json("metrics_snapshot.json")
        logger.info("Metrics snapshot exported successfully")
    except Exception as e:
        logger.error(f"Failed to export metrics snapshot: {e}", exc_info=True)
    
    # Close rate limiter Redis connection
    try:
        await rate_limiter.close()
        logger.info("Rate limiter connection closed")
    except Exception as e:
        logger.error(f"Failed to close rate limiter: {e}", exc_info=True)
    
    logger.info("CoreSight application shutting down")

