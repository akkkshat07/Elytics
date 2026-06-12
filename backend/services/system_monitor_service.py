from __future__ import annotations
import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import psutil
from config.system_config import REDIS_URL
from util.cancellation import cancellation_manager
from util.time_utils import utcnow
logger = logging.getLogger(__name__)
PROCESS_START_TIME = time.time()

class SystemMonitorService:

    def get_system_resources(self) -> Dict[str, Any]:
        proc = psutil.Process(os.getpid())
        mem_info = proc.memory_info()
        vm = psutil.virtual_memory()
        try:
            proc_cpu = proc.cpu_percent(interval=0.1)
        except Exception:
            proc_cpu = 0.0
        cpu_freq = psutil.cpu_freq()
        disk = psutil.disk_usage('/')
        return {'system': {'cpu_count': psutil.cpu_count(logical=True), 'cpu_count_physical': psutil.cpu_count(logical=False), 'cpu_percent': psutil.cpu_percent(interval=0.1), 'cpu_freq_mhz': round(cpu_freq.current, 1) if cpu_freq else None, 'ram_total_mb': round(vm.total / 1024 / 1024, 1), 'ram_used_mb': round(vm.used / 1024 / 1024, 1), 'ram_available_mb': round(vm.available / 1024 / 1024, 1), 'ram_percent': vm.percent, 'disk_total_gb': round(disk.total / 1024 / 1024 / 1024, 1), 'disk_used_gb': round(disk.used / 1024 / 1024 / 1024, 1), 'disk_free_gb': round(disk.free / 1024 / 1024 / 1024, 1), 'disk_percent': disk.percent}, 'process': {'pid': proc.pid, 'cpu_percent': proc_cpu, 'ram_rss_mb': round(mem_info.rss / 1024 / 1024, 2), 'ram_vms_mb': round(mem_info.vms / 1024 / 1024, 2), 'threads': proc.num_threads(), 'open_files': len(proc.open_files()), 'connections': len(proc.net_connections()), 'uptime_seconds': round(time.time() - PROCESS_START_TIME)}}

    def get_active_stream_sessions(self) -> List[str]:
        return list(cancellation_manager._events.keys())

    async def get_active_sessions_detailed(self, db) -> Dict[str, Any]:
        streaming_ids = set(self.get_active_stream_sessions())
        recent_cutoff = utcnow() - timedelta(minutes=5)
        conditions = []
        if streaming_ids:
            conditions.append({'session_id': {'$in': list(streaming_ids)}})
        conditions.append({'status': {'$in': ['streaming', 'pending', 'processing']}, 'created_at': {'$gte': recent_cutoff}})
        if not conditions:
            return {'streaming_session_ids': list(streaming_ids), 'streaming_count': len(streaming_ids), 'active_conversations': [], 'active_conversation_count': 0, 'timestamp': utcnow().isoformat()}
        active_filter = {'$or': conditions}
        projection = {'run_id': 1, 'session_id': 1, 'user_id': 1, 'client_id': 1, 'status': 1, 'input': 1, 'route_decision': 1, 'created_at': 1, 'started_at': 1, 'model': 1, 'llm_config.provider': 1, 'llm_config.model': 1, 'total_token_usage': 1, 'is_background': 1, 'metadata.user_agent': 1}
        client_defaults: Dict[str, Dict[str, str]] = {}
        try:
            async for cfg_doc in db.llm_configurations.find({}, {'client_id': 1, 'configurations': 1}):
                cid = cfg_doc.get('client_id')
                for cfg in cfg_doc.get('configurations', []):
                    if cfg.get('is_default'):
                        client_defaults[cid] = {'provider': cfg.get('provider', ''), 'model': cfg.get('model', '')}
                        break
        except Exception as e:
            logger.error(f'Error pre-fetching client LLM defaults: {e}')
        conversations = []
        try:
            cursor = db.conversations.find(active_filter, projection).sort('created_at', -1).limit(200)
            async for doc in cursor:
                doc['_id'] = str(doc['_id'])
                created = doc.get('created_at')
                started = doc.get('started_at')
                if isinstance(created, datetime):
                    doc['created_at'] = created.isoformat()
                if isinstance(started, datetime):
                    doc['started_at'] = started.isoformat()
                sid = doc.get('session_id')
                db_status = doc.get('status', '')
                is_streaming = sid in streaming_ids
                if is_streaming:
                    effective = 'streaming'
                elif db_status in ('completed', 'success'):
                    effective = 'completed'
                elif db_status == 'error':
                    effective = 'error'
                elif db_status in ('pending', 'processing', 'streaming'):
                    is_recent = isinstance(created, datetime) and created >= recent_cutoff
                    effective = 'processing' if is_recent else 'stale'
                else:
                    effective = db_status or 'unknown'
                doc['is_streaming'] = is_streaming
                doc['effective_status'] = effective
                llm_cfg = doc.get('llm_config') or {}
                conv_provider = llm_cfg.get('provider') or ''
                conv_model = doc.get('model') or llm_cfg.get('model') or ''
                is_resolved = bool(conv_provider and conv_model)
                if not is_resolved:
                    fallback = client_defaults.get(doc.get('client_id'), {})
                    conv_provider = conv_provider or fallback.get('provider', '')
                    conv_model = conv_model or fallback.get('model', '')
                doc['resolved_provider'] = conv_provider
                doc['resolved_model'] = conv_model
                doc['model_is_confirmed'] = is_resolved
                conversations.append(doc)
        except Exception as e:
            logger.error(f'Error fetching active conversations: {e}')
        session_ids_in_view = list({c.get('session_id') for c in conversations if c.get('session_id')})
        session_token_summary: Dict[str, Any] = {}
        if session_ids_in_view:
            try:
                pipeline = [{'$match': {'session_id': {'$in': session_ids_in_view}}}, {'$group': {'_id': '$session_id', 'conversation_count': {'$sum': 1}, 'total_prompt_tokens': {'$sum': {'$ifNull': ['$total_token_usage.prompt_tokens', 0]}}, 'total_completion_tokens': {'$sum': {'$ifNull': ['$total_token_usage.completion_tokens', 0]}}, 'total_tokens': {'$sum': {'$ifNull': ['$total_token_usage.total_tokens', 0]}}, 'total_cost_usd': {'$sum': {'$ifNull': ['$estimated_cost.total_cost_usd', 0]}}, 'first_created': {'$min': '$created_at'}, 'last_created': {'$max': '$created_at'}, 'providers_used': {'$addToSet': '$llm_config.provider'}, 'models_used': {'$addToSet': '$model'}, 'client_id': {'$first': '$client_id'}, 'user_id': {'$first': '$user_id'}, 'error_count': {'$sum': {'$cond': [{'$eq': ['$status', 'error']}, 1, 0]}}}}, {'$sort': {'total_tokens': -1}}]
                async for doc in db.conversations.aggregate(pipeline):
                    sid = doc['_id']
                    fc = doc.get('first_created')
                    lc = doc.get('last_created')
                    session_token_summary[sid] = {'session_id': sid, 'conversation_count': doc['conversation_count'], 'prompt_tokens': doc['total_prompt_tokens'], 'completion_tokens': doc['total_completion_tokens'], 'total_tokens': doc['total_tokens'], 'total_cost_usd': round(doc.get('total_cost_usd', 0), 6), 'first_conversation': fc.isoformat() if isinstance(fc, datetime) else fc, 'last_conversation': lc.isoformat() if isinstance(lc, datetime) else lc, 'providers_used': [p for p in doc.get('providers_used') or [] if p], 'models_used': [m for m in doc.get('models_used') or [] if m], 'client_id': doc.get('client_id'), 'user_id': doc.get('user_id'), 'error_count': doc.get('error_count', 0), 'is_streaming': sid in streaming_ids}
            except Exception as e:
                logger.error(f'Error aggregating session tokens: {e}')
        return {'streaming_session_ids': list(streaming_ids), 'streaming_count': len(streaming_ids), 'active_conversations': conversations, 'active_conversation_count': len(conversations), 'session_token_summary': list(session_token_summary.values()), 'timestamp': utcnow().isoformat()}

    async def get_llm_provider_health(self, db) -> Dict[str, Any]:
        from services.llm_config_service import llm_config_service, PROVIDER_MODELS
        from config.system_config import LLM_PROVIDERS
        results: List[Dict[str, Any]] = []
        meta_config = await db.llm_meta_config.find_one({'is_active': True})
        platform_providers = {}
        if meta_config and meta_config.get('providers'):
            for prov in meta_config['providers']:
                name = prov.get('provider_name')
                if name:
                    platform_providers[name] = prov
        client_provider_counts: Dict[str, int] = {}
        try:
            pipeline = [{'$unwind': '$configurations'}, {'$group': {'_id': '$configurations.provider', 'client_count': {'$sum': 1}, 'models_used': {'$addToSet': '$configurations.model'}}}]
            async for doc in db.llm_configurations.aggregate(pipeline):
                provider_name = doc['_id']
                client_provider_counts[provider_name] = {'client_count': doc['client_count'], 'models_used': doc.get('models_used', [])}
        except Exception as e:
            logger.error(f'Error aggregating client LLM configs: {e}')
        all_providers = set(PROVIDER_MODELS.keys()) | set(platform_providers.keys()) | set(client_provider_counts.keys())
        health_tasks = []
        provider_order = []
        key_sources: Dict[str, str] = {}
        for provider_name in sorted(all_providers):
            env_cfg = LLM_PROVIDERS.get(provider_name, {})
            env_var_name = env_cfg.get('api_key_env_var')
            api_key = os.getenv(env_var_name) if env_var_name else None
            key_source = 'env' if api_key else None
            prov_meta = platform_providers.get(provider_name, {})
            if not api_key and prov_meta.get('api_key'):
                api_key = prov_meta['api_key']
                key_source = 'database'
            test_model = env_cfg.get('default_model')
            if prov_meta.get('models'):
                first_meta_model = prov_meta['models'][0] if isinstance(prov_meta['models'], list) else None
                if first_meta_model:
                    if isinstance(first_meta_model, dict):
                        test_model = first_meta_model.get('model_name', test_model)
                    else:
                        test_model = first_meta_model
            if not test_model:
                models = PROVIDER_MODELS.get(provider_name, [])
                test_model = models[0] if models else None
            provider_order.append(provider_name)
            key_sources[provider_name] = key_source or 'none'
            if api_key and test_model:
                health_tasks.append(llm_config_service.healthcheck(provider_name, api_key, test_model, timeout=8.0))
            else:
                _pn = provider_name
                _ev = env_var_name or 'N/A'

                async def _no_key_result(_pn=_pn, _ev=_ev):
                    return {'healthy': False, 'latency_ms': 0, 'error': f'No API key — set {_ev} in .env'}
                health_tasks.append(_no_key_result())
        health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
        for idx, provider_name in enumerate(provider_order):
            result = health_results[idx]
            if isinstance(result, Exception):
                result = {'healthy': False, 'latency_ms': 0, 'error': str(result)}
            usage = client_provider_counts.get(provider_name, {})
            env_var = LLM_PROVIDERS.get(provider_name, {}).get('api_key_env_var')
            results.append({'provider': provider_name, 'healthy': result.get('healthy', False), 'latency_ms': result.get('latency_ms', 0), 'error': result.get('error'), 'available_models': PROVIDER_MODELS.get(provider_name, []), 'clients_using': usage.get('client_count', 0), 'models_in_use': usage.get('models_used', []), 'has_platform_key': bool(os.getenv(env_var)) if env_var else provider_name in platform_providers, 'key_source': key_sources.get(provider_name, 'none'), 'env_var': env_var})
        healthy_count = sum((1 for r in results if r['healthy']))
        return {'providers': results, 'total_providers': len(results), 'healthy_count': healthy_count, 'unhealthy_count': len(results) - healthy_count, 'timestamp': utcnow().isoformat()}

    async def get_session_resources(self, db) -> Dict[str, Any]:
        streaming_ids = self.get_active_stream_sessions()
        active_count = max(len(streaming_ids), 1)
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        try:
            cpu = proc.cpu_percent(interval=0.1)
        except Exception:
            cpu = 0.0
        total_rss_mb = round(mem.rss / 1024 / 1024, 2)
        total_vms_mb = round(mem.vms / 1024 / 1024, 2)
        per_session_data = []
        if streaming_ids:
            try:
                cursor = db.conversations.find({'session_id': {'$in': streaming_ids}, 'status': {'$in': ['streaming', 'pending', 'processing']}}, {'session_id': 1, 'user_id': 1, 'client_id': 1, 'status': 1, 'model': 1, 'llm_config.provider': 1, 'total_token_usage': 1, 'created_at': 1})
                async for doc in cursor:
                    tokens = doc.get('total_token_usage', {})
                    per_session_data.append({'session_id': doc.get('session_id'), 'client_id': doc.get('client_id'), 'user_id': doc.get('user_id'), 'status': doc.get('status'), 'model': doc.get('model'), 'provider': doc.get('llm_config', {}).get('provider'), 'total_tokens': tokens.get('total_tokens', 0), 'prompt_tokens': tokens.get('prompt_tokens', 0), 'completion_tokens': tokens.get('completion_tokens', 0), 'est_ram_mb': round(total_rss_mb / active_count, 2), 'est_cpu_percent': round(cpu / active_count, 2)})
            except Exception as e:
                logger.error(f'Error fetching per-session resources: {e}')
        return {'total_active_sessions': len(streaming_ids), 'process_ram_rss_mb': total_rss_mb, 'process_ram_vms_mb': total_vms_mb, 'process_cpu_percent': cpu, 'per_session_avg_ram_mb': round(total_rss_mb / active_count, 2), 'per_session_avg_cpu_percent': round(cpu / active_count, 2), 'sessions': per_session_data, 'timestamp': utcnow().isoformat()}

    async def get_system_overview(self, db) -> Dict[str, Any]:
        resources = self.get_system_resources()
        streaming_ids = self.get_active_stream_sessions()
        now = utcnow()
        recent_24h = now - timedelta(hours=24)
        recent_1h = now - timedelta(hours=1)
        stats = {}
        try:
            stats['total_conversations_24h'] = await db.conversations.count_documents({'created_at': {'$gte': recent_24h}})
            stats['total_conversations_1h'] = await db.conversations.count_documents({'created_at': {'$gte': recent_1h}})
            stats['total_active_clients'] = len(await db.conversations.distinct('client_id', {'created_at': {'$gte': recent_24h}}))
            stats['total_active_users'] = len(await db.conversations.distinct('user_id', {'created_at': {'$gte': recent_24h}}))
            stats['error_count_24h'] = await db.conversations.count_documents({'status': 'error', 'created_at': {'$gte': recent_24h}})
            stats['total_sessions_24h'] = len(await db.conversations.distinct('session_id', {'created_at': {'$gte': recent_24h}}))
        except Exception as e:
            logger.error(f'Error fetching overview stats: {e}')
        redis_status = 'unknown'
        try:
            import redis
            r = redis.from_url(REDIS_URL, socket_connect_timeout=2)
            r.ping()
            redis_info = r.info('memory')
            redis_status = 'healthy'
            stats['redis_used_memory_mb'] = round(redis_info.get('used_memory', 0) / 1024 / 1024, 2)
            stats['redis_connected_clients'] = r.info('clients').get('connected_clients', 0)
        except Exception:
            redis_status = 'unhealthy'
        mongo_status = 'unknown'
        try:
            await db.command('ping')
            mongo_status = 'healthy'
        except Exception:
            mongo_status = 'unhealthy'
        vm = psutil.virtual_memory()
        est_max_concurrent = max(1, int(vm.available / 1024 / 1024 / 50))
        return {**resources, 'active_streams': len(streaming_ids), 'streaming_session_ids': streaming_ids, 'infrastructure': {'mongodb': mongo_status, 'redis': redis_status}, 'stats': stats, 'capacity': {'estimated_max_concurrent_sessions': est_max_concurrent, 'current_utilization_percent': round(len(streaming_ids) / max(est_max_concurrent, 1) * 100, 1), 'ram_headroom_mb': round(vm.available / 1024 / 1024, 1), 'cpu_headroom_percent': round(100 - resources['system']['cpu_percent'], 1)}, 'timestamp': utcnow().isoformat()}

    async def get_throughput_timeseries(self, db, hours: int=24) -> List[Dict[str, Any]]:
        now = utcnow()
        start = now - timedelta(hours=hours)
        pipeline = [{'$match': {'created_at': {'$gte': start}}}, {'$group': {'_id': {'$dateToString': {'format': '%Y-%m-%dT%H:00:00Z', 'date': '$created_at'}}, 'count': {'$sum': 1}, 'errors': {'$sum': {'$cond': [{'$eq': ['$status', 'error']}, 1, 0]}}, 'unique_users': {'$addToSet': '$user_id'}, 'unique_clients': {'$addToSet': '$client_id'}, 'avg_tokens': {'$avg': '$total_token_usage.total_tokens'}}}, {'$sort': {'_id': 1}}]
        result = []
        try:
            async for doc in db.conversations.aggregate(pipeline):
                result.append({'hour': doc['_id'], 'conversations': doc['count'], 'errors': doc['errors'], 'unique_users': len(doc.get('unique_users', [])), 'unique_clients': len(doc.get('unique_clients', [])), 'avg_tokens': round(doc.get('avg_tokens') or 0)})
        except Exception as e:
            logger.error(f'Error fetching throughput timeseries: {e}')
        return result
system_monitor_service = SystemMonitorService()