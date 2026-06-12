import os
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
_EMBED_CACHE_MAXSIZE = 1000
_EMBED_CACHE_TTL = 3600
_embed_cache: Dict[tuple, Tuple[List[float], float]] = {}
_embed_cache_order: List[tuple] = []
_embed_cache_lock = threading.Lock()

def _cache_get(key: tuple) -> Optional[List[float]]:
    with _embed_cache_lock:
        entry = _embed_cache.get(key)
        if entry is None:
            return None
        vector, expires_at = entry
        if time.monotonic() > expires_at:
            _embed_cache.pop(key, None)
            try:
                _embed_cache_order.remove(key)
            except ValueError:
                pass
            return None
        try:
            _embed_cache_order.remove(key)
        except ValueError:
            pass
        _embed_cache_order.append(key)
        return vector

def _cache_set(key: tuple, vector: List[float]) -> None:
    with _embed_cache_lock:
        expires_at = time.monotonic() + _EMBED_CACHE_TTL
        if key in _embed_cache:
            try:
                _embed_cache_order.remove(key)
            except ValueError:
                pass
        elif len(_embed_cache) >= _EMBED_CACHE_MAXSIZE:
            oldest = _embed_cache_order.pop(0)
            _embed_cache.pop(oldest, None)
        _embed_cache[key] = (vector, expires_at)
        _embed_cache_order.append(key)

def generate_embedding(text: str) -> List[float]:
    from config.system_config import VECTOR_DB_CONFIG
    provider = VECTOR_DB_CONFIG.get('embedding_provider', 'openai')
    model = VECTOR_DB_CONFIG.get('embedding_model', 'text-embedding-3-small')
    api_key_env = VECTOR_DB_CONFIG.get('api_key_env_var', 'OPENAI_API_KEY')
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f'{api_key_env} environment variable not set')
    cache_key = (text, provider, model)
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug('Embedding cache hit: provider=%s model=%s text_len=%d', provider, model, len(text))
        return cached
    if provider == 'gemini':
        vector = _generate_gemini_embedding(text, model, api_key)
    else:
        vector = _generate_openai_embedding(text, model, api_key)
    _cache_set(cache_key, vector)
    return vector

def generate_embedding_with_usage(text: str) -> Tuple[List[float], Dict]:
    from config.system_config import VECTOR_DB_CONFIG
    provider = VECTOR_DB_CONFIG.get('embedding_provider', 'openai')
    model = VECTOR_DB_CONFIG.get('embedding_model', 'text-embedding-3-small')
    api_key_env = VECTOR_DB_CONFIG.get('api_key_env_var', 'OPENAI_API_KEY')
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f'{api_key_env} environment variable not set')
    cache_key = (text, provider, model)
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug('Embedding cache hit (with_usage): provider=%s model=%s', provider, model)
        estimated_tokens = max(1, len(text) // 4)
        usage = {'prompt_tokens': estimated_tokens, 'completion_tokens': 0, 'total_tokens': estimated_tokens, 'provider': provider, 'model': model, 'estimated': True, 'cache_hit': True}
        return (cached, usage)
    if provider == 'gemini':
        embedding = _generate_gemini_embedding(text, model, api_key)
        estimated_tokens = max(1, len(text) // 4)
        usage = {'prompt_tokens': estimated_tokens, 'completion_tokens': 0, 'total_tokens': estimated_tokens, 'provider': 'gemini', 'model': model, 'estimated': True}
        _cache_set(cache_key, embedding)
        return (embedding, usage)
    else:
        embedding, usage = _generate_openai_embedding_with_usage(text, model, api_key)
        _cache_set(cache_key, embedding)
        return (embedding, usage)

def get_embedding_dimension() -> int:
    from config.system_config import VECTOR_DB_CONFIG
    return int(VECTOR_DB_CONFIG.get('embedding_dimension', 1536))

def _generate_gemini_embedding(text: str, model: str, api_key: str) -> List[float]:
    try:
        from google import genai
        from config.system_config import gemini_use_vertex_ai
        client = genai.Client(api_key=api_key, vertexai=gemini_use_vertex_ai())
        result = client.models.embed_content(model=model, contents=text)
        embedding = result.embeddings[0].values
        logger.debug('Generated Gemini embedding: model=%s, dims=%d', model, len(embedding))
        return list(embedding)
    except Exception as e:
        logger.error('Gemini embedding generation failed: %s', e)
        raise

def _generate_openai_embedding(text: str, model: str, api_key: str) -> List[float]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(input=text, model=model)
        embedding = response.data[0].embedding
        logger.debug('Generated OpenAI embedding: model=%s, dims=%d', model, len(embedding))
        return embedding
    except Exception as e:
        logger.error('OpenAI embedding generation failed: %s', e)
        raise

def _generate_openai_embedding_with_usage(text: str, model: str, api_key: str) -> Tuple[List[float], Dict]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(input=text, model=model)
        embedding = response.data[0].embedding
        usage_obj = getattr(response, 'usage', None)
        usage = {'prompt_tokens': getattr(usage_obj, 'prompt_tokens', 0) or 0, 'completion_tokens': 0, 'total_tokens': getattr(usage_obj, 'total_tokens', 0) or 0, 'provider': 'openai', 'model': model}
        logger.debug('Generated OpenAI embedding with usage: model=%s, dims=%d, tokens=%d', model, len(embedding), usage['total_tokens'])
        return (embedding, usage)
    except Exception as e:
        logger.error('OpenAI embedding generation failed: %s', e)
        raise