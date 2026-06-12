from __future__ import annotations
import time
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from util.time_utils import utcnow
from collections import defaultdict
from threading import Lock
import json
logger = logging.getLogger(__name__)

class MetricsCollector:

    def __init__(self):
        self._lock = Lock()
        self._metrics = {'authentication': defaultdict(lambda: {'success': 0, 'failure': 0, 'total': 0}), 'queries': defaultdict(lambda: {'count': 0, 'total_duration_ms': 0, 'errors': 0}), 'prompts': defaultdict(lambda: {'loads': 0, 'cache_hits': 0, 'cache_misses': 0, 'total_duration_ms': 0, 'by_type': defaultdict(int)}), 'database': defaultdict(lambda: {'queries': 0, 'total_duration_ms': 0, 'errors': 0}), 'admin_api': defaultdict(lambda: {'operations': 0, 'by_endpoint': defaultdict(int), 'errors': 0})}
        self._start_time = time.time()
        self._last_reset = utcnow()
        logger.info('Metrics collector initialized')

    def record_authentication(self, client_id: str, success: bool=True):
        with self._lock:
            stats = self._metrics['authentication'][client_id]
            stats['total'] += 1
            if success:
                stats['success'] += 1
            else:
                stats['failure'] += 1
        logger.debug(f'Auth metric recorded | client_id={client_id} | success={success}')

    def get_authentication_stats(self, client_id: Optional[str]=None) -> Dict[str, Any]:
        with self._lock:
            if client_id:
                stats = self._metrics['authentication'].get(client_id, {})
                return {'client_id': client_id, 'total_attempts': stats.get('total', 0), 'successful': stats.get('success', 0), 'failed': stats.get('failure', 0), 'success_rate': stats.get('success', 0) / stats.get('total', 1) * 100 if stats.get('total', 0) > 0 else 0}
            else:
                total_attempts = sum((s['total'] for s in self._metrics['authentication'].values()))
                total_success = sum((s['success'] for s in self._metrics['authentication'].values()))
                total_failure = sum((s['failure'] for s in self._metrics['authentication'].values()))
                return {'total_attempts': total_attempts, 'successful': total_success, 'failed': total_failure, 'success_rate': total_success / total_attempts * 100 if total_attempts > 0 else 0, 'clients': len(self._metrics['authentication'])}

    def record_query(self, client_id: str, duration_ms: float, error: bool=False):
        with self._lock:
            stats = self._metrics['queries'][client_id]
            stats['count'] += 1
            stats['total_duration_ms'] += duration_ms
            if error:
                stats['errors'] += 1
        logger.debug(f'Query metric recorded | client_id={client_id} | duration={duration_ms}ms | error={error}')

    def get_query_stats(self, client_id: Optional[str]=None) -> Dict[str, Any]:
        with self._lock:
            if client_id:
                stats = self._metrics['queries'].get(client_id, {})
                count = stats.get('count', 0)
                total_duration = stats.get('total_duration_ms', 0)
                return {'client_id': client_id, 'total_queries': count, 'total_errors': stats.get('errors', 0), 'avg_duration_ms': total_duration / count if count > 0 else 0, 'error_rate': stats.get('errors', 0) / count * 100 if count > 0 else 0}
            else:
                total_count = sum((s['count'] for s in self._metrics['queries'].values()))
                total_duration = sum((s['total_duration_ms'] for s in self._metrics['queries'].values()))
                total_errors = sum((s['errors'] for s in self._metrics['queries'].values()))
                return {'total_queries': total_count, 'total_errors': total_errors, 'avg_duration_ms': total_duration / total_count if total_count > 0 else 0, 'error_rate': total_errors / total_count * 100 if total_count > 0 else 0, 'clients': len(self._metrics['queries'])}

    def record_prompt_load(self, client_id: str, prompt_type: str, duration_ms: float, cache_hit: bool=False):
        with self._lock:
            stats = self._metrics['prompts'][client_id]
            stats['loads'] += 1
            stats['total_duration_ms'] += duration_ms
            stats['by_type'][prompt_type] += 1
            if cache_hit:
                stats['cache_hits'] += 1
            else:
                stats['cache_misses'] += 1
        logger.debug(f'Prompt metric recorded | client_id={client_id} | type={prompt_type} | duration={duration_ms}ms | cache_hit={cache_hit}')

    def get_prompt_stats(self, client_id: Optional[str]=None) -> Dict[str, Any]:
        with self._lock:
            if client_id:
                stats = self._metrics['prompts'].get(client_id, {})
                loads = stats.get('loads', 0)
                total_duration = stats.get('total_duration_ms', 0)
                cache_hits = stats.get('cache_hits', 0)
                cache_misses = stats.get('cache_misses', 0)
                return {'client_id': client_id, 'total_loads': loads, 'avg_duration_ms': total_duration / loads if loads > 0 else 0, 'cache_hits': cache_hits, 'cache_misses': cache_misses, 'cache_hit_rate': cache_hits / loads * 100 if loads > 0 else 0, 'by_type': dict(stats.get('by_type', {}))}
            else:
                total_loads = sum((s['loads'] for s in self._metrics['prompts'].values()))
                total_duration = sum((s['total_duration_ms'] for s in self._metrics['prompts'].values()))
                total_cache_hits = sum((s['cache_hits'] for s in self._metrics['prompts'].values()))
                total_cache_misses = sum((s['cache_misses'] for s in self._metrics['prompts'].values()))
                return {'total_loads': total_loads, 'avg_duration_ms': total_duration / total_loads if total_loads > 0 else 0, 'cache_hits': total_cache_hits, 'cache_misses': total_cache_misses, 'cache_hit_rate': total_cache_hits / total_loads * 100 if total_loads > 0 else 0, 'clients': len(self._metrics['prompts'])}

    def record_database_query(self, collection: str, duration_ms: float, error: bool=False):
        with self._lock:
            stats = self._metrics['database'][collection]
            stats['queries'] += 1
            stats['total_duration_ms'] += duration_ms
            if error:
                stats['errors'] += 1
        logger.debug(f'Database metric recorded | collection={collection} | duration={duration_ms}ms | error={error}')

    def get_database_stats(self, collection: Optional[str]=None) -> Dict[str, Any]:
        with self._lock:
            if collection:
                stats = self._metrics['database'].get(collection, {})
                queries = stats.get('queries', 0)
                total_duration = stats.get('total_duration_ms', 0)
                return {'collection': collection, 'total_queries': queries, 'total_errors': stats.get('errors', 0), 'avg_duration_ms': total_duration / queries if queries > 0 else 0, 'error_rate': stats.get('errors', 0) / queries * 100 if queries > 0 else 0}
            else:
                total_queries = sum((s['queries'] for s in self._metrics['database'].values()))
                total_duration = sum((s['total_duration_ms'] for s in self._metrics['database'].values()))
                total_errors = sum((s['errors'] for s in self._metrics['database'].values()))
                return {'total_queries': total_queries, 'total_errors': total_errors, 'avg_duration_ms': total_duration / total_queries if total_queries > 0 else 0, 'error_rate': total_errors / total_queries * 100 if total_queries > 0 else 0, 'collections': list(self._metrics['database'].keys())}

    def record_admin_operation(self, admin_email: str, endpoint: str, error: bool=False):
        with self._lock:
            stats = self._metrics['admin_api'][admin_email]
            stats['operations'] += 1
            stats['by_endpoint'][endpoint] += 1
            if error:
                stats['errors'] += 1
        logger.debug(f'Admin API metric recorded | admin={admin_email} | endpoint={endpoint} | error={error}')

    def get_admin_stats(self, admin_email: Optional[str]=None) -> Dict[str, Any]:
        with self._lock:
            if admin_email:
                stats = self._metrics['admin_api'].get(admin_email, {})
                operations = stats.get('operations', 0)
                return {'admin_email': admin_email, 'total_operations': operations, 'total_errors': stats.get('errors', 0), 'error_rate': stats.get('errors', 0) / operations * 100 if operations > 0 else 0, 'by_endpoint': dict(stats.get('by_endpoint', {}))}
            else:
                total_operations = sum((s['operations'] for s in self._metrics['admin_api'].values()))
                total_errors = sum((s['errors'] for s in self._metrics['admin_api'].values()))
                return {'total_operations': total_operations, 'total_errors': total_errors, 'error_rate': total_errors / total_operations * 100 if total_operations > 0 else 0, 'admins': len(self._metrics['admin_api'])}

    def get_all_metrics(self) -> Dict[str, Any]:
        uptime_seconds = time.time() - self._start_time
        return {'timestamp': utcnow().isoformat(), 'uptime_seconds': uptime_seconds, 'last_reset': self._last_reset.isoformat(), 'authentication': self.get_authentication_stats(), 'queries': self.get_query_stats(), 'prompts': self.get_prompt_stats(), 'database': self.get_database_stats(), 'admin_api': self.get_admin_stats()}

    def get_client_summary(self, client_id: str) -> Dict[str, Any]:
        return {'client_id': client_id, 'timestamp': utcnow().isoformat(), 'authentication': self.get_authentication_stats(client_id), 'queries': self.get_query_stats(client_id), 'prompts': self.get_prompt_stats(client_id)}

    def reset_metrics(self):
        with self._lock:
            self._metrics = {'authentication': defaultdict(lambda: {'success': 0, 'failure': 0, 'total': 0}), 'queries': defaultdict(lambda: {'count': 0, 'total_duration_ms': 0, 'errors': 0}), 'prompts': defaultdict(lambda: {'loads': 0, 'cache_hits': 0, 'cache_misses': 0, 'total_duration_ms': 0, 'by_type': defaultdict(int)}), 'database': defaultdict(lambda: {'queries': 0, 'total_duration_ms': 0, 'errors': 0}), 'admin_api': defaultdict(lambda: {'operations': 0, 'by_endpoint': defaultdict(int), 'errors': 0})}
            self._last_reset = utcnow()
        logger.info('Metrics reset completed')

    def export_to_json(self, filepath: str):
        try:
            metrics = self.get_all_metrics()
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f'Metrics exported to {filepath}')
        except Exception as e:
            logger.error(f'Error exporting metrics: {e}', exc_info=True)
metrics_collector = MetricsCollector()

def record_auth(client_id: str, success: bool=True):
    metrics_collector.record_authentication(client_id, success)

def record_query(client_id: str, duration_ms: float, error: bool=False):
    metrics_collector.record_query(client_id, duration_ms, error)

def record_prompt_load(client_id: str, prompt_type: str, duration_ms: float, cache_hit: bool=False):
    metrics_collector.record_prompt_load(client_id, prompt_type, duration_ms, cache_hit)

def record_db_query(collection: str, duration_ms: float, error: bool=False):
    metrics_collector.record_database_query(collection, duration_ms, error)

def record_admin_op(admin_email: str, endpoint: str, error: bool=False):
    metrics_collector.record_admin_operation(admin_email, endpoint, error)