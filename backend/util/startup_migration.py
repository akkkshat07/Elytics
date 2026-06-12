from __future__ import annotations
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from util.time_utils import utcnow
logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[1]

def _move_children_into_subdir(parent: Path, subdir_name: str) -> int:
    if not parent.is_dir():
        return 0
    dest = parent / subdir_name
    children = [c for c in parent.iterdir() if c.name != subdir_name]
    if not children:
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    for child in children:
        shutil.move(str(child), str(dest / child.name))
    logger.info('migration: moved %d item(s) %s → %s/', len(children), parent, subdir_name)
    return len(children)

def _migrate_xml_prompts(client_id: str, dataset_id: str) -> None:
    data_sources = _ROOT / 'xml_prompts' / 'clients' / client_id / 'data_sources'
    if not (data_sources / 'meta_information').exists():
        return
    _move_children_into_subdir(data_sources, dataset_id)

def _migrate_assets(client_id: str, dataset_id: str) -> None:
    client_assets = _ROOT / 'assets' / 'clients' / client_id
    for sub in ('datasets', 'uploads'):
        _move_children_into_subdir(client_assets / sub, dataset_id)

def _migrate_response_caching(client_id: str, dataset_id: str) -> None:
    rc_dir = _ROOT / 'assets' / 'clients' / client_id / 'response_caching'
    _move_children_into_subdir(rc_dir, dataset_id)

def _check_xml_prompts(client_id: str) -> bool:
    data_sources = _ROOT / 'xml_prompts' / 'clients' / client_id / 'data_sources'
    if not data_sources.is_dir():
        return True
    if (data_sources / 'meta_information').exists():
        return False
    for child in data_sources.iterdir():
        if child.is_dir():
            try:
                import uuid as _uuid
                _uuid.UUID(child.name)
                return True
            except ValueError:
                pass
    return True

def _check_assets(client_id: str) -> bool:
    client_assets = _ROOT / 'assets' / 'clients' / client_id
    import uuid as _uuid
    for sub in ('datasets', 'uploads'):
        folder = client_assets / sub
        if not folder.is_dir():
            continue
        for child in folder.iterdir():
            try:
                _uuid.UUID(child.name)
            except ValueError:
                return False
    return True

def _check_response_caching(client_id: str) -> bool:
    rc_dir = _ROOT / 'assets' / 'clients' / client_id / 'response_caching'
    if not rc_dir.is_dir():
        return True
    import uuid as _uuid
    for child in rc_dir.iterdir():
        try:
            _uuid.UUID(child.name)
        except ValueError:
            return False
    return True

async def _check_conversations(db: Any, client_id: str) -> bool:
    count = await db.conversations.count_documents({'client_id': client_id, '$or': [{'dataset_id': {'$exists': False}}, {'dataset_id': None}, {'dataset_id': ''}]})
    return count == 0

async def _check_dashboard_reports(db: Any, client_id: str) -> bool:
    count = await db.dashboard_reports.count_documents({'client_id': client_id, 'reports': {'$elemMatch': {'$or': [{'dataset_id': {'$exists': False}}, {'dataset_id': None}, {'dataset_id': ''}]}}})
    return count == 0

async def migrate_client(coll: Any, client_id: str, doc: Dict[str, Any], db: Any) -> Dict[str, Any]:
    checks: Dict[str, str] = {}
    existing_id = doc.get('dataset_id')
    already_had_id = bool(existing_id and str(existing_id).strip())
    if already_had_id:
        dataset_id = str(existing_id).strip()
        dataset_name = doc.get('dataset_name') or doc.get('db_name') or doc.get('db_type') or 'Default Dataset'
        checks['db_credentials'] = 'ok'
    else:
        dataset_id = str(uuid.uuid4())
        dataset_name = doc.get('dataset_name') or doc.get('db_name') or doc.get('db_type') or 'Default Dataset'
        try:
            await coll.update_one({'_id': doc['_id']}, {'$set': {'dataset_id': dataset_id, 'dataset_name': dataset_name, 'is_enabled': doc.get('is_enabled', True), 'display_order': doc.get('display_order', 0), 'updated_at': utcnow()}})
            logger.info('migration: client=%s assigned dataset_id=%s', client_id, dataset_id)
            checks['db_credentials'] = 'fixed'
        except Exception as exc:
            logger.error('migration: db_credentials update failed for client=%s: %s', client_id, exc)
            checks['db_credentials'] = 'error'
            checks['xml_prompts'] = 'skipped'
            checks['assets'] = 'skipped'
            checks['conversations'] = 'skipped'
            return {'client_id': client_id, 'dataset_id': None, 'dataset_name': None, 'checks': checks, 'conversations_updated': 0}
    xml_ok_before = _check_xml_prompts(client_id)
    if xml_ok_before:
        checks['xml_prompts'] = 'ok'
    else:
        try:
            _migrate_xml_prompts(client_id, dataset_id)
            checks['xml_prompts'] = 'fixed'
        except Exception as exc:
            logger.warning('migration: xml_prompts failed for client=%s: %s', client_id, exc)
            checks['xml_prompts'] = 'error'
    assets_ok_before = _check_assets(client_id)
    if assets_ok_before:
        checks['assets'] = 'ok'
    else:
        try:
            _migrate_assets(client_id, dataset_id)
            checks['assets'] = 'fixed'
        except Exception as exc:
            logger.warning('migration: assets failed for client=%s: %s', client_id, exc)
            checks['assets'] = 'error'
    rc_ok_before = _check_response_caching(client_id)
    if rc_ok_before:
        checks['response_caching'] = 'ok'
    else:
        try:
            _migrate_response_caching(client_id, dataset_id)
            checks['response_caching'] = 'fixed'
        except Exception as exc:
            logger.warning('migration: response_caching failed for client=%s: %s', client_id, exc)
            checks['response_caching'] = 'error'
    conv_ok_before = await _check_conversations(db, client_id)
    if conv_ok_before:
        checks['conversations'] = 'ok'
        conversations_updated = 0
    else:
        try:
            result = await db.conversations.update_many({'client_id': client_id, '$or': [{'dataset_id': {'$exists': False}}, {'dataset_id': None}, {'dataset_id': ''}]}, {'$set': {'dataset_id': dataset_id}})
            conversations_updated = result.modified_count
            logger.info('migration: backfilled %d conversation(s) for client=%s', conversations_updated, client_id)
            checks['conversations'] = 'fixed'
        except Exception as exc:
            logger.warning('migration: conversation backfill failed for client=%s: %s', client_id, exc)
            checks['conversations'] = 'error'
            conversations_updated = 0
    dr_ok_before = await _check_dashboard_reports(db, client_id)
    if dr_ok_before:
        checks['dashboard_reports'] = 'ok'
        dashboard_reports_updated = 0
    else:
        try:
            result = await db.dashboard_reports.update_many({'client_id': client_id, 'reports': {'$elemMatch': {'$or': [{'dataset_id': {'$exists': False}}, {'dataset_id': None}, {'dataset_id': ''}]}}}, {'$set': {'reports.$[r].dataset_id': dataset_id}}, array_filters=[{'$or': [{'r.dataset_id': {'$exists': False}}, {'r.dataset_id': None}, {'r.dataset_id': ''}]}])
            dashboard_reports_updated = result.modified_count
            logger.info('migration: backfilled dashboard_reports for %d doc(s) for client=%s', dashboard_reports_updated, client_id)
            checks['dashboard_reports'] = 'fixed'
        except Exception as exc:
            logger.warning('migration: dashboard_reports backfill failed for client=%s: %s', client_id, exc)
            checks['dashboard_reports'] = 'error'
            dashboard_reports_updated = 0
    return {'client_id': client_id, 'dataset_id': dataset_id, 'dataset_name': dataset_name, 'checks': checks, 'conversations_updated': conversations_updated, 'dashboard_reports_updated': dashboard_reports_updated}

async def run_all_migrations(db: Any) -> List[Dict[str, Any]]:
    coll = db.db_credentials
    results: List[Dict[str, Any]] = []
    try:
        seen_clients: set = set()
        async for doc in coll.find({}, sort=[('display_order', 1), ('created_at', 1)]):
            client_id: str = doc.get('client_id', '')
            if not client_id or client_id in seen_clients:
                continue
            seen_clients.add(client_id)
            result = await migrate_client(coll, client_id, doc, db)
            results.append(result)
    except Exception as exc:
        logger.error('run_all_migrations: unexpected error: %s', exc, exc_info=True)
    return results