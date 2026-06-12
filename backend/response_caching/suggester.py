import logging
import re
import threading
from typing import Dict, List, Optional
import pandas as pd
from response_caching.config_manager import get_client_cache_dir
logger = logging.getLogger(__name__)
_lock = threading.Lock()
_raw_questions: Dict[str, List[str]] = {}
_norm_questions: Dict[str, List[str]] = {}

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub('[^a-z0-9\\s]', ' ', text)
    return re.sub('\\s+', ' ', text).strip()

def _cache_key(client_id: str, dataset_id: Optional[str]=None) -> str:
    return f"{client_id}:{dataset_id or ''}"

def _load_client(client_id: str, dataset_id: Optional[str]=None) -> None:
    key = _cache_key(client_id, dataset_id)
    with _lock:
        if key in _raw_questions:
            return
        csv_path = get_client_cache_dir(client_id, dataset_id) / 'correct_responses.csv'
        if not csv_path.exists():
            logger.debug(f"No cached questions CSV for client '{client_id}', dataset '{dataset_id}' at {csv_path}")
            return
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c.lower() == 'question')
            df.columns = [c.lower() for c in df.columns]
            questions = df['question'].dropna().astype(str).tolist()
            _raw_questions[key] = questions
            _norm_questions[key] = [_normalize(q) for q in questions]
            logger.info(f"Loaded {len(questions)} cached questions for client '{client_id}', dataset '{dataset_id or 'default'}'")
        except Exception as exc:
            logger.warning(f"Failed to load suggestions for client '{client_id}', dataset '{dataset_id}': {exc}")
            _raw_questions[key] = []
            _norm_questions[key] = []

def suggest(q: str, limit: int, client_id: str, dataset_id: Optional[str]=None) -> List[dict]:
    if not q or not q.strip():
        return []
    _load_client(client_id, dataset_id)
    key = _cache_key(client_id, dataset_id)
    raw = _raw_questions.get(key, [])
    norm = _norm_questions.get(key, [])
    if not raw:
        return []
    norm_q = _normalize(q)
    tokens = norm_q.split()
    if not tokens:
        return []
    scored: List[tuple] = []
    for idx, (candidate_raw, candidate_norm) in enumerate(zip(raw, norm)):
        score = 0.0
        for token in tokens:
            if token in candidate_norm:
                score += 1.0
        if score == 0.0:
            continue
        if norm_q in candidate_norm:
            score += 1.5
        if candidate_norm.startswith(tokens[0]):
            score += 0.5
        scored.append((score, idx))
    scored.sort(key=lambda x: -x[0])
    return [{'question': raw[idx], 'score': round(score, 2)} for score, idx in scored[:limit]]

def warm_reload(client_id: str, dataset_id: Optional[str]=None) -> None:
    key = _cache_key(client_id, dataset_id)
    with _lock:
        _raw_questions.pop(key, None)
        _norm_questions.pop(key, None)
    _load_client(client_id, dataset_id)
    logger.info(f"Warm-reloaded suggestions cache for client '{client_id}', dataset '{dataset_id or 'default'}'")