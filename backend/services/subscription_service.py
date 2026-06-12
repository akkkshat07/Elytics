import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from util.time_utils import utcnow
from motor.motor_asyncio import AsyncIOMotorDatabase
from db_config.mongo_server import get_db
logger = logging.getLogger(__name__)
_plans_seeded = False
DEFAULT_SUBSCRIPTION_PLANS = {'freemium': {'plan_name': 'freemium', 'display_order': 0, 'description': 'Get started for free', 'features': {'max_users': 1, 'max_conversations': 100, 'table_limit': 10, 'column_limit': 25, 'prompt_overrides': 0, 'max_cached_questions': 25, 'session_memory': 'limited', 'rate_limits': {'client_rpm': 200, 'user_rpm': 30, 'endpoint_rpm': 15}}}, 'starter': {'plan_name': 'starter', 'display_order': 1, 'description': 'Perfect for small teams', 'features': {'max_users': 2, 'max_conversations': 500, 'table_limit': 20, 'column_limit': 50, 'prompt_overrides': 2, 'max_cached_questions': 50, 'session_memory': None, 'rate_limits': {'client_rpm': 500, 'user_rpm': 60, 'endpoint_rpm': 30}}}, 'pro': {'plan_name': 'pro', 'display_order': 2, 'description': 'For growing businesses', 'features': {'max_users': 5, 'max_conversations': 2500, 'table_limit': 50, 'column_limit': 75, 'prompt_overrides': 5, 'max_cached_questions': 100, 'session_memory': None, 'advanced_agents': True, 'rate_limits': {'client_rpm': 1500, 'user_rpm': 120, 'endpoint_rpm': 60}}}, 'premium': {'plan_name': 'premium', 'display_order': 3, 'description': 'Enterprise solution', 'features': {'max_users': 10, 'max_conversations': None, 'table_limit': 100, 'column_limit': 100, 'prompt_overrides': 10, 'max_cached_questions': None, 'session_memory': None, 'advanced_agents': True, 'rate_limits': {'client_rpm': 5000, 'user_rpm': 300, 'endpoint_rpm': 120}}}}
SUBSCRIPTION_PLANS = DEFAULT_SUBSCRIPTION_PLANS

async def _ensure_plans_seeded(db: AsyncIOMotorDatabase) -> None:
    global _plans_seeded
    if _plans_seeded:
        return
    try:
        pipeline = [{'$group': {'_id': '$plan_name', 'ids': {'$push': '$_id'}, 'count': {'$sum': 1}}}, {'$match': {'count': {'$gt': 1}}}]
        async for group in db.subscription_plans.aggregate(pipeline):
            ids_to_delete = group['ids'][1:]
            if ids_to_delete:
                await db.subscription_plans.delete_many({'_id': {'$in': ids_to_delete}})
                logger.info(f"Removed {len(ids_to_delete)} duplicate(s) for plan '{group['_id']}'")
        await db.subscription_plans.create_index('plan_name', unique=True)
        count = await db.subscription_plans.count_documents({})
        if count == 0:
            now = datetime.utcnow()
            for plan in DEFAULT_SUBSCRIPTION_PLANS.values():
                await db.subscription_plans.update_one({'plan_name': plan['plan_name']}, {'$setOnInsert': {'plan_name': plan['plan_name'], 'features': plan['features'], 'display_order': plan.get('display_order', 99), 'description': plan.get('description', ''), 'created_at': now, 'updated_at': now}}, upsert=True)
            logger.info('Seeded default subscription plans into MongoDB')
        _plans_seeded = True
    except Exception as e:
        logger.warning(f'Failed to seed subscription plans: {e}')
        _plans_seeded = True

async def get_all_subscription_plans(db: Optional[AsyncIOMotorDatabase]=None) -> List[Dict[str, Any]]:
    if db is None:
        db = await get_db()
    if hasattr(db, 'db') and (not hasattr(db, 'list_collection_names')):
        db = db.db
    try:
        await _ensure_plans_seeded(db)
        cursor = db.subscription_plans.find({}, {'_id': 0})
        plans = await cursor.to_list(length=100)
        if plans:
            defaults_by_name = {p['plan_name']: p for p in DEFAULT_SUBSCRIPTION_PLANS.values()}
            for plan in plans:
                name = plan.get('plan_name')
                if name and name in defaults_by_name:
                    d = defaults_by_name[name]
                    if plan.get('description') is None:
                        plan['description'] = d.get('description', '')
                    if plan.get('display_order') is None:
                        plan['display_order'] = d.get('display_order', 99)
            return plans
    except Exception as e:
        logger.warning(f'Failed to read plans from MongoDB, using defaults: {e}')
    return [{'plan_name': p['plan_name'], 'features': p['features'], 'display_order': p.get('display_order', 99), 'description': p.get('description', '')} for p in DEFAULT_SUBSCRIPTION_PLANS.values()]

async def get_plan_by_name(plan_name: str, db: Optional[AsyncIOMotorDatabase]=None) -> Optional[Dict[str, Any]]:
    if db is None:
        db = await get_db()
    if hasattr(db, 'db') and (not hasattr(db, 'list_collection_names')):
        db = db.db
    try:
        await _ensure_plans_seeded(db)
        plan = await db.subscription_plans.find_one({'plan_name': plan_name.lower()}, {'_id': 0})
        return plan
    except Exception as e:
        logger.warning(f"Failed to read plan '{plan_name}' from MongoDB: {e}")
        fallback = DEFAULT_SUBSCRIPTION_PLANS.get(plan_name.lower())
        if fallback:
            return {'plan_name': fallback['plan_name'], 'features': fallback['features']}
        return None

async def get_subscription_plan(plan_name: str='freemium') -> Dict[str, Any]:
    plan_name_lower = plan_name.lower() if plan_name else 'freemium'
    try:
        db = await get_db()
        if hasattr(db, 'db') and (not hasattr(db, 'list_collection_names')):
            db = db.db
        await _ensure_plans_seeded(db)
        plan = await db.subscription_plans.find_one({'plan_name': plan_name_lower}, {'_id': 0})
        if plan:
            return {'plan_name': plan['plan_name'], 'features': plan['features']}
    except Exception as e:
        logger.warning(f"MongoDB plan lookup failed for '{plan_name_lower}': {e}")
    if plan_name_lower in DEFAULT_SUBSCRIPTION_PLANS:
        plan = DEFAULT_SUBSCRIPTION_PLANS[plan_name_lower].copy()
        return plan
    else:
        logger.warning(f"Invalid plan name '{plan_name}', defaulting to freemium")
        return DEFAULT_SUBSCRIPTION_PLANS['freemium'].copy()

async def get_client_subscription(client_id: str, db: Optional[AsyncIOMotorDatabase]=None) -> Dict[str, Any]:
    if db is None:
        db = await get_db()
    if hasattr(db, 'db') and (not hasattr(db, 'list_collection_names')):
        db = db.db
    if db is None:
        db = await get_db()
    try:
        subscription = await db.subscriptions.find_one({'client_id': client_id})
        if subscription:
            plan_name = subscription.get('plan_name', 'freemium')
            plan = await get_subscription_plan(plan_name)
            return {'plan_name': plan_name, 'features': plan['features']}
        else:
            return await get_subscription_plan('freemium')
    except Exception as e:
        logger.error(f'Error getting subscription for client {client_id}: {e}')
        return await get_subscription_plan('freemium')

async def get_rate_limits(client_id: str, db: Optional[AsyncIOMotorDatabase]=None) -> Dict[str, int]:
    _hardcoded_fallback = DEFAULT_SUBSCRIPTION_PLANS['freemium']['features']['rate_limits']
    try:
        plan = await get_client_subscription(client_id, db=db)
        rate_limits = plan.get('features', {}).get('rate_limits')
        if rate_limits:
            return rate_limits
        freemium = await get_subscription_plan('freemium')
        return freemium.get('features', {}).get('rate_limits', _hardcoded_fallback)
    except Exception as e:
        logger.error(f'Error getting rate limits for client {client_id}: {e}')
        return _hardcoded_fallback

async def get_explorer_limits(client_id: str, db: Optional[AsyncIOMotorDatabase]=None, role: Optional[str]=None) -> Dict[str, Optional[int]]:
    if role == 'super_admin':
        return {'table_limit': None, 'column_limit': None}
    plan = await get_client_subscription(client_id, db=db)
    features = plan.get('features') or {}
    return {'table_limit': features.get('table_limit'), 'column_limit': features.get('column_limit')}

async def count_client_conversations(client_id: str, db: Optional[AsyncIOMotorDatabase]=None, monthly: bool=False) -> int:
    if db is None:
        db = await get_db()
    try:
        query = {'client_id': client_id}
        if monthly:
            now = utcnow()
            start_of_month = datetime(now.year, now.month, 1)
            if now.month == 12:
                end_of_month = datetime(now.year + 1, 1, 1)
            else:
                end_of_month = datetime(now.year, now.month + 1, 1)
            query['created_at'] = {'$gte': start_of_month, '$lt': end_of_month}
        count = await db.conversations.count_documents(query)
        return count
    except Exception as e:
        logger.error(f'Error counting conversations for client {client_id}: {e}')
        return 0

async def count_client_users(client_id: str, db: Optional[AsyncIOMotorDatabase]=None) -> int:
    if db is None:
        db = await get_db()
    try:
        count = await db.users.count_documents({'client_id': client_id, 'is_active': True})
        return count
    except Exception as e:
        logger.error(f'Error counting users for client {client_id}: {e}')
        return 0

async def check_conversation_limit(client_id: str, db: Optional[AsyncIOMotorDatabase]=None, role: Optional[str]=None) -> Tuple[bool, int, Optional[int]]:
    if role == 'super_admin':
        return (True, 0, None)
    subscription = await get_client_subscription(client_id, db)
    limit = subscription['features']['max_conversations']
    current_count = await count_client_conversations(client_id, db, monthly=True)
    if limit is None:
        is_allowed = True
    else:
        is_allowed = current_count < limit
    return (is_allowed, current_count, limit)

async def check_user_limit(client_id: str, db: Optional[AsyncIOMotorDatabase]=None, role: Optional[str]=None) -> Tuple[bool, int, int]:
    if role == 'super_admin':
        return (True, 0, 999999)
    subscription = await get_client_subscription(client_id, db)
    limit = subscription['features']['max_users']
    current_count = await count_client_users(client_id, db)
    is_allowed = current_count < limit
    return (is_allowed, current_count, limit)

async def get_subscription_with_usage(client_id: str, db: Optional[AsyncIOMotorDatabase]=None) -> Dict[str, Any]:
    if db is None:
        db = await get_db()
    subscription = await get_client_subscription(client_id, db)
    conversations_used = await count_client_conversations(client_id, db, monthly=True)
    users_used = await count_client_users(client_id, db)
    from admin.custom_prompts_routes import get_prompt_override_count
    prompt_overrides_used = await get_prompt_override_count(client_id, db)
    conversations_limit = subscription['features']['max_conversations']
    users_limit = subscription['features']['max_users']
    prompt_overrides_limit = subscription['features'].get('prompt_overrides', 0)
    cache_limit = subscription['features'].get('max_cached_questions')
    if cache_limit is not None:
        from response_caching.feedback_processor import get_total_cache_count
        cache_used = get_total_cache_count(client_id)
    else:
        cache_used = 0
        cache_limit = None
    conversations_remaining = None if conversations_limit is None else max(0, conversations_limit - conversations_used)
    return {'plan_name': subscription['plan_name'], 'features': subscription['features'], 'usage': {'conversations': {'used': conversations_used, 'limit': conversations_limit, 'remaining': conversations_remaining}, 'users': {'used': users_used, 'limit': users_limit, 'remaining': max(0, users_limit - users_used)}, 'prompt_overrides': {'used': prompt_overrides_used, 'limit': prompt_overrides_limit}, 'cache': {'used': cache_used, 'limit': cache_limit}}, 'upgrade_message': 'To upgrade your plan, please contact our customer support team.'}