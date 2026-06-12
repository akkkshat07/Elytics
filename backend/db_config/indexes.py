import logging
from motor.motor_asyncio import AsyncIOMotorDatabase
logger = logging.getLogger(__name__)
_AUDIT_LOG_TTL_SECONDS = 90 * 24 * 3600
_LLM_METRICS_TTL_SECONDS = 30 * 24 * 3600

async def ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    await _conversations(db)
    await _conversation_history(db)
    await _users(db)
    await _audit_logs(db)
    await _client_prompts(db)
    await _llm_configurations(db)
    await _subscriptions(db)
    await _db_credentials(db)
    await _data_export_jobs(db)
    await _llm_call_metrics(db)
    await _notifications(db)
    await _client_configs(db)
    await _upgrade_requests(db)
    await _subscription_plans(db)
    await _agent_personas(db)
    await _client_personas(db)
    logger.info('MongoDB indexes ensured for all collections')

async def _conversations(db: AsyncIOMotorDatabase) -> None:
    coll = db.conversations
    specs = [([('client_id', 1), ('created_at', 1)], {'name': 'idx_conv_client_created'}), ([('client_id', 1), ('user_id', 1), ('session_id', 1), ('created_at', -1)], {'name': 'idx_conv_client_user_session_created'}), ([('run_id', 1)], {'name': 'idx_conv_run_id', 'unique': True, 'sparse': True}), ([('client_id', 1), ('is_background', 1), ('status', 1)], {'name': 'idx_conv_client_bg_status'}), ([('client_id', 1), ('user_id', 1), ('is_background', 1), ('status', 1)], {'name': 'idx_conv_client_user_bg_status'}), ([('session_id', 1), ('client_id', 1), ('is_background', 1), ('status', 1)], {'name': 'idx_conv_session_client_bg_status'}), ([('client_id', 1), ('user_id', 1), ('is_read', 1)], {'name': 'idx_conv_client_user_isread'}), ([('created_at', -1)], {'name': 'idx_conv_created_at'})]
    await _create_indexes(coll, specs, 'conversations')

async def _users(db: AsyncIOMotorDatabase) -> None:
    coll = db.users
    specs = [([('client_id', 1), ('is_active', 1)], {'name': 'idx_users_client_active'}), ([('email', 1)], {'name': 'idx_users_email', 'unique': True})]
    await _create_indexes(coll, specs, 'users')

async def _audit_logs(db: AsyncIOMotorDatabase) -> None:
    coll = db.audit_logs
    specs = [([('client_id', 1), ('timestamp', -1)], {'name': 'idx_audit_client_ts'}), ([('timestamp', 1)], {'name': 'idx_audit_ttl', 'expireAfterSeconds': _AUDIT_LOG_TTL_SECONDS})]
    await _create_indexes(coll, specs, 'audit_logs')

async def _client_prompts(db: AsyncIOMotorDatabase) -> None:
    coll = db.client_prompts
    specs = [([('client_id', 1), ('prompt_path', 1), ('active', 1)], {'name': 'idx_prompts_client_path_active'})]
    await _create_indexes(coll, specs, 'client_prompts')

async def _llm_configurations(db: AsyncIOMotorDatabase) -> None:
    coll = db.llm_configurations
    specs = [([('client_id', 1)], {'name': 'idx_llmcfg_client_id'})]
    await _create_indexes(coll, specs, 'llm_configurations')

async def _subscriptions(db: AsyncIOMotorDatabase) -> None:
    coll = db.subscriptions
    specs = [([('client_id', 1)], {'name': 'idx_sub_client_id'})]
    await _create_indexes(coll, specs, 'subscriptions')

async def _db_credentials(db: AsyncIOMotorDatabase) -> None:
    coll = db.db_credentials
    specs = [([('client_id', 1), ('is_enabled', 1), ('display_order', 1)], {'name': 'idx_dbcred_client_enabled_order'}), ([('client_id', 1), ('dataset_id', 1)], {'name': 'idx_dbcred_client_dataset', 'unique': True, 'sparse': True}), ([('client_id', 1), ('dataset_name', 1)], {'name': 'idx_dbcred_client_dataset_name', 'unique': True, 'sparse': True})]
    await _create_indexes(coll, specs, 'db_credentials')

async def _conversation_history(db: AsyncIOMotorDatabase) -> None:
    coll = db.conversation_history
    specs = [([('client_id', 1), ('user_id', 1)], {'name': 'idx_convhist_client_user'})]
    await _create_indexes(coll, specs, 'conversation_history')

async def _data_export_jobs(db: AsyncIOMotorDatabase) -> None:
    coll = db.data_export_jobs
    specs = [([('job_id', 1)], {'name': 'idx_exportjobs_job_id', 'unique': True}), ([('status', 1), ('created_at', 1)], {'name': 'idx_exportjobs_status_created'})]
    await _create_indexes(coll, specs, 'data_export_jobs')

async def _llm_call_metrics(db: AsyncIOMotorDatabase) -> None:
    coll = db.llm_call_metrics
    specs = [([('client_id', 1), ('ts', -1)], {'name': 'idx_llmmetrics_client_ts'}), ([('ts', 1)], {'name': 'idx_llmmetrics_ttl', 'expireAfterSeconds': _LLM_METRICS_TTL_SECONDS})]
    await _create_indexes(coll, specs, 'llm_call_metrics')

async def _notifications(db: AsyncIOMotorDatabase) -> None:
    coll = db.notifications
    specs = [([('client_id', 1), ('user_id', 1), ('created_at', -1)], {'name': 'idx_notif_client_user_created'}), ([('client_id', 1), ('user_id', 1), ('read', 1)], {'name': 'idx_notif_client_user_read'}), ([('id', 1), ('user_id', 1)], {'name': 'idx_notif_id_user', 'unique': True, 'sparse': True})]
    await _create_indexes(coll, specs, 'notifications')

async def _client_configs(db: AsyncIOMotorDatabase) -> None:
    coll = db.client_configs
    specs = [([('client_id', 1)], {'name': 'idx_clientcfg_client_id', 'unique': True}), ([('client_id', 1), ('deleted_at', 1)], {'name': 'idx_clientcfg_client_deleted'}), ([('created_at', -1)], {'name': 'idx_clientcfg_created_at'})]
    await _create_indexes(coll, specs, 'client_configs')

async def _upgrade_requests(db: AsyncIOMotorDatabase) -> None:
    coll = db.upgrade_requests
    specs = [([('status', 1), ('created_at', -1)], {'name': 'idx_upgradeq_status_created'}), ([('client_id', 1)], {'name': 'idx_upgradeq_client_id', 'unique': True, 'sparse': True})]
    await _create_indexes(coll, specs, 'upgrade_requests')

async def _subscription_plans(db: AsyncIOMotorDatabase) -> None:
    coll = db.subscription_plans
    specs = [([('plan_name', 1)], {'name': 'idx_subplans_plan_name', 'unique': True})]
    await _create_indexes(coll, specs, 'subscription_plans')

async def _agent_personas(db: AsyncIOMotorDatabase) -> None:
    coll = db.agent_personas
    specs = [([('slug', 1)], {'name': 'idx_agentpersona_slug', 'unique': True})]
    await _create_indexes(coll, specs, 'agent_personas')

async def _client_personas(db: AsyncIOMotorDatabase) -> None:
    coll = db.client_personas
    specs = [([('client_id', 1)], {'name': 'idx_clientpersona_client_id'})]
    await _create_indexes(coll, specs, 'client_personas')

async def _create_indexes(coll, specs: list, collection_name: str) -> None:
    for keys, kwargs in specs:
        index_name = kwargs.get('name', str(keys))
        try:
            await coll.create_index(keys, **kwargs)
            logger.debug('Index ready: %s.%s', collection_name, index_name)
        except Exception as exc:
            logger.warning('Index skipped on %s.%s: %s', collection_name, index_name, exc)