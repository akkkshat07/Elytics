from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import psutil
import asyncio
import logging
from functools import wraps
logger = logging.getLogger(__name__)

class MonitoringService:

    def __init__(self):
        self.registry = CollectorRegistry()
        self.system_cpu_usage = Gauge('system_cpu_usage_percent', 'Current CPU usage percentage', registry=self.registry)
        self.system_memory_usage = Gauge('system_memory_usage_bytes', 'Current memory usage in bytes', registry=self.registry)
        self.system_memory_percent = Gauge('system_memory_usage_percent', 'Current memory usage percentage', registry=self.registry)
        self.system_disk_usage = Gauge('system_disk_usage_percent', 'Current disk usage percentage', registry=self.registry)
        self.http_requests_total = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status_code', 'client_id'], registry=self.registry)
        self.http_request_duration_seconds = Histogram('http_request_duration_seconds', 'HTTP request latency in seconds', ['method', 'endpoint', 'client_id'], buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0], registry=self.registry)
        self.http_requests_in_progress = Gauge('http_requests_in_progress', 'Number of HTTP requests currently being processed', ['method', 'endpoint'], registry=self.registry)
        self.db_query_duration_seconds = Histogram('db_query_duration_seconds', 'Database query latency in seconds', ['operation', 'collection'], buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0], registry=self.registry)
        self.db_connections_active = Gauge('db_connections_active', 'Number of active database connections', registry=self.registry)
        self.db_operations_total = Counter('db_operations_total', 'Total database operations', ['operation', 'collection', 'status'], registry=self.registry)
        self.redis_cache_hits_total = Counter('redis_cache_hits_total', 'Total Redis cache hits', ['cache_type'], registry=self.registry)
        self.redis_cache_misses_total = Counter('redis_cache_misses_total', 'Total Redis cache misses', ['cache_type'], registry=self.registry)
        self.redis_rate_limit_checks_total = Counter('redis_rate_limit_checks_total', 'Total rate limit checks', ['limit_type', 'result'], registry=self.registry)
        self.ai_agent_requests_total = Counter('ai_agent_requests_total', 'Total AI agent requests', ['agent_type', 'client_id', 'status'], registry=self.registry)
        self.ai_agent_duration_seconds = Histogram('ai_agent_duration_seconds', 'AI agent processing time in seconds', ['agent_type'], buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0], registry=self.registry)
        self.ai_tokens_used_total = Counter('ai_tokens_used_total', 'Total AI tokens consumed', ['agent_type', 'client_id'], registry=self.registry)
        self.active_users_total = Gauge('active_users_total', 'Number of active users', ['client_id'], registry=self.registry)
        self.queries_per_hour = Gauge('queries_per_hour', 'Number of queries in the last hour', ['client_id'], registry=self.registry)
        self.user_sessions_active = Gauge('user_sessions_active', 'Number of active user sessions', ['client_id'], registry=self.registry)
        self.service_health = Gauge('service_health_status', 'Overall service health (1=healthy, 0=unhealthy)', ['component'], registry=self.registry)
        self.service_info = Info('service', 'Service information', registry=self.registry)
        self.service_info.info({'name': 'CoreSight', 'version': '2.0.0', 'environment': 'production'})
        self.alert_thresholds = {'cpu_usage_percent': 80.0, 'memory_usage_percent': 85.0, 'disk_usage_percent': 90.0, 'http_error_rate_percent': 5.0, 'db_latency_seconds': 1.0, 'ai_agent_latency_seconds': 30.0}
        self._monitoring_task = None
        self._is_running = False
        logger.info('Monitoring service initialized with Prometheus metrics')

    def collect_system_metrics(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.system_cpu_usage.set(cpu_percent)
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.used)
            self.system_memory_percent.set(memory.percent)
            disk = psutil.disk_usage('/')
            self.system_disk_usage.set(disk.percent)
            self._check_system_alerts(cpu_percent, memory.percent, disk.percent)
        except Exception as e:
            logger.error(f'Error collecting system metrics: {e}')

    def _check_system_alerts(self, cpu: float, memory: float, disk: float):
        if cpu > self.alert_thresholds['cpu_usage_percent']:
            logger.warning(f"⚠️ HIGH CPU USAGE: {cpu:.1f}% (threshold: {self.alert_thresholds['cpu_usage_percent']}%)")
        if memory > self.alert_thresholds['memory_usage_percent']:
            logger.warning(f"⚠️ HIGH MEMORY USAGE: {memory:.1f}% (threshold: {self.alert_thresholds['memory_usage_percent']}%)")
        if disk > self.alert_thresholds['disk_usage_percent']:
            logger.warning(f"⚠️ HIGH DISK USAGE: {disk:.1f}% (threshold: {self.alert_thresholds['disk_usage_percent']}%)")

    def track_http_request(self, method: str, endpoint: str, status_code: int, duration: float, client_id: str='unknown'):
        self.http_requests_total.labels(method=method, endpoint=endpoint, status_code=str(status_code), client_id=client_id).inc()
        self.http_request_duration_seconds.labels(method=method, endpoint=endpoint, client_id=client_id).observe(duration)

    def track_http_in_progress(self, method: str, endpoint: str):

        class InProgressTracker:

            def __init__(tracker_self, monitoring_service, method, endpoint):
                tracker_self.monitoring_service = monitoring_service
                tracker_self.method = method
                tracker_self.endpoint = endpoint

            def __enter__(tracker_self):
                tracker_self.monitoring_service.http_requests_in_progress.labels(method=tracker_self.method, endpoint=tracker_self.endpoint).inc()
                return tracker_self

            def __exit__(tracker_self, exc_type, exc_val, exc_tb):
                tracker_self.monitoring_service.http_requests_in_progress.labels(method=tracker_self.method, endpoint=tracker_self.endpoint).dec()
        return InProgressTracker(self, method, endpoint)

    def track_db_query(self, operation: str, collection: str, duration: float, success: bool=True):
        self.db_query_duration_seconds.labels(operation=operation, collection=collection).observe(duration)
        status = 'success' if success else 'error'
        self.db_operations_total.labels(operation=operation, collection=collection, status=status).inc()
        if duration > self.alert_thresholds['db_latency_seconds']:
            logger.warning(f"⚠️ SLOW DB QUERY: {operation} on {collection} took {duration:.2f}s (threshold: {self.alert_thresholds['db_latency_seconds']}s)")

    def update_db_connections(self, active_connections: int):
        self.db_connections_active.set(active_connections)

    def track_cache_hit(self, cache_type: str='response'):
        self.redis_cache_hits_total.labels(cache_type=cache_type).inc()

    def track_cache_miss(self, cache_type: str='response'):
        self.redis_cache_misses_total.labels(cache_type=cache_type).inc()

    def track_rate_limit_check(self, limit_type: str, allowed: bool):
        result = 'allowed' if allowed else 'blocked'
        self.redis_rate_limit_checks_total.labels(limit_type=limit_type, result=result).inc()

    def track_ai_agent_request(self, agent_type: str, client_id: str, duration: float, tokens_used: int, success: bool=True):
        status = 'success' if success else 'error'
        self.ai_agent_requests_total.labels(agent_type=agent_type, client_id=client_id, status=status).inc()
        self.ai_agent_duration_seconds.labels(agent_type=agent_type).observe(duration)
        self.ai_tokens_used_total.labels(agent_type=agent_type, client_id=client_id).add(tokens_used)
        if duration > self.alert_thresholds['ai_agent_latency_seconds']:
            logger.warning(f"⚠️ SLOW AI AGENT: {agent_type} took {duration:.1f}s (threshold: {self.alert_thresholds['ai_agent_latency_seconds']}s)")

    def update_active_users(self, client_id: str, count: int):
        self.active_users_total.labels(client_id=client_id).set(count)

    def update_queries_per_hour(self, client_id: str, count: int):
        self.queries_per_hour.labels(client_id=client_id).set(count)

    def update_active_sessions(self, client_id: str, count: int):
        self.user_sessions_active.labels(client_id=client_id).set(count)

    def update_component_health(self, component: str, healthy: bool):
        status = 1.0 if healthy else 0.0
        self.service_health.labels(component=component).set(status)

    async def perform_health_checks(self, db_client, redis_client) -> Dict[str, bool]:
        health_status = {}
        try:
            await db_client.admin.command('ping')
            health_status['mongodb'] = True
            self.update_component_health('mongodb', True)
        except Exception as e:
            logger.error(f'MongoDB health check failed: {e}')
            health_status['mongodb'] = False
            self.update_component_health('mongodb', False)
        try:
            redis_client.ping()
            health_status['redis'] = True
            self.update_component_health('redis', True)
        except Exception as e:
            logger.error(f'Redis health check failed: {e}')
            health_status['redis'] = False
            self.update_component_health('redis', False)
        all_healthy = all(health_status.values())
        self.update_component_health('overall', all_healthy)
        return health_status

    def export_metrics(self) -> bytes:
        self.collect_system_metrics()
        return generate_latest(self.registry)

    def get_metrics_content_type(self) -> str:
        return CONTENT_TYPE_LATEST

    async def start_background_monitoring(self, interval: int=15):
        if self._is_running:
            logger.warning('Background monitoring already running')
            return
        self._is_running = True
        logger.info(f'Starting background monitoring (interval: {interval}s)')
        while self._is_running:
            try:
                self.collect_system_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f'Error in background monitoring: {e}')
                await asyncio.sleep(interval)

    async def stop_background_monitoring(self):
        self._is_running = False
        logger.info('Stopping background monitoring')
monitoring_service = MonitoringService()

def track_execution_time(metric_name: str):

    def decorator(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = await func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.debug(f'{metric_name} completed in {duration:.3f}s')
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f'{metric_name} failed after {duration:.3f}s: {e}')
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.debug(f'{metric_name} completed in {duration:.3f}s')
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f'{metric_name} failed after {duration:.3f}s: {e}')
                raise
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator