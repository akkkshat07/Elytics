import logging
import re
from typing import List, Optional, Set
from bson import ObjectId
logger = logging.getLogger(__name__)
_TABLE_KEY_RE = re.compile('[^A-Z0-9]+')

def _table_key(name: str) -> str:
    if not name:
        return ''
    return _TABLE_KEY_RE.sub('', name.upper())

async def get_denied_tables_for_user(user_id: str, client_id: str, db, dataset_id: Optional[str]=None) -> Set[str]:
    if not user_id or db is None:
        return set()
    db_handle = db
    try:
        db_dict = getattr(db, '__dict__', None)
        if isinstance(db_dict, dict) and db_dict.get('db') is not None:
            db_handle = db_dict['db']
    except Exception:
        db_handle = db
    users_collection = getattr(db_handle, 'users', None)
    if users_collection is None:
        logger.warning("[TablePermissions] db has no 'users' collection; returning no restrictions.")
        return set()
    try:
        try:
            obj_id = ObjectId(user_id)
            user = await users_collection.find_one({'_id': obj_id, 'client_id': client_id})
        except Exception:
            user = await users_collection.find_one({'_id': user_id, 'client_id': client_id})
        if not user:
            logger.debug(f"[TablePermissions] User '{user_id}' not found for client '{client_id}', returning no restrictions (empty denied set).")
            return set()
        role = user.get('role', 'user')
        if role in ('admin', 'super_admin'):
            logger.debug(f"[TablePermissions] User '{user_id}' has role '{role}' – bypassing table restrictions.")
            return set()
        by_ds = user.get('denied_tables_by_dataset')
        if isinstance(by_ds, dict) and by_ds:
            ds_key = str(dataset_id).strip() if dataset_id is not None else ''
            raw_list = by_ds.get(ds_key)
            if raw_list is None:
                raw_list = []
            denied_keys = {_table_key(t) for t in raw_list if t and _table_key(t)}
        else:
            denied_tables: List[str] = user.get('denied_tables') or []
            denied_keys = {_table_key(t) for t in denied_tables if t and _table_key(t)}
        if denied_keys:
            logger.info(f"[TablePermissions] User '{user_id}' (client '{client_id}') has {len(denied_keys)} denied tables.")
        return denied_keys
    except Exception as exc:
        logger.error(f"[TablePermissions] Error fetching denied tables for user '{user_id}': {exc}", exc_info=True)
        return set()

def get_allowed_tables(all_table_names: List[str], denied_tables: Set[str]) -> List[str]:
    if not denied_tables:
        return list(all_table_names)
    return [t for t in all_table_names if _table_key(t) not in denied_tables]

def check_tables_access(requested_tables: List[str], denied_tables: Set[str]) -> Optional[List[str]]:
    if not denied_tables or not requested_tables:
        return None
    violations = [t for t in requested_tables if _table_key(t) in denied_tables]
    return violations if violations else None