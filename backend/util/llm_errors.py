import asyncio
import enum
import logging
import re
from datetime import datetime, timezone
from typing import Optional
logger = logging.getLogger(__name__)
try:
    import openai as _openai
except ImportError:
    _openai = None
try:
    import groq as _groq
except ImportError:
    _groq = None
try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None

class LLMErrorCategory(enum.Enum):
    RETRYABLE = 'retryable'
    PROVIDER_FALLBACK = 'provider_fallback'
    HARD_FAILURE = 'hard_failure'

class LLMHardFailureError(Exception):

    def __init__(self, provider: str, original_exc: Exception, message: str) -> None:
        super().__init__(message)
        self.provider = provider
        self.original_exc = original_exc
        self.message = message

    def __repr__(self) -> str:
        return f'LLMHardFailureError(provider={self.provider!r}, message={self.message!r}, original={type(self.original_exc).__name__})'

def _extract_status_code(exc: Exception) -> Optional[int]:
    code = getattr(exc, 'status_code', None)
    if isinstance(code, int):
        return code
    response = getattr(exc, 'response', None)
    if response is not None:
        code = getattr(response, 'status_code', None)
        if isinstance(code, int):
            return code
    code = getattr(exc, 'code', None)
    if isinstance(code, int):
        return code
    exc_str = str(exc)
    patterns = ['status_code[=:\\s]+(\\d{3})', '\\b(4\\d{2}|5\\d{2})\\b', 'HTTP[/ ]+(\\d{3})', '\\((\\d{3})\\)']
    for pattern in patterns:
        match = re.search(pattern, exc_str, re.IGNORECASE)
        if match:
            try:
                candidate = int(match.group(1))
                if 400 <= candidate <= 599:
                    return candidate
            except (ValueError, IndexError):
                continue
    return None
_RETRY_AFTER_MAX_SECONDS: float = 30.0

def _parse_retry_after_raw(exc: Exception) -> Optional[float]:
    raw: Optional[str] = None
    response = getattr(exc, 'response', None)
    if response is not None:
        headers = getattr(response, 'headers', {}) or {}
        raw = headers.get('retry-after') or headers.get('Retry-After')
    if raw is None:
        headers = getattr(exc, 'headers', {}) or {}
        raw = headers.get('retry-after') or headers.get('Retry-After')
    if raw is None:
        return None
    raw = raw.strip()
    if raw.isdigit():
        return float(raw)
    try:
        import email.utils
        parsed_dt = email.utils.parsedate_to_datetime(raw)
        now = datetime.now(timezone.utc)
        return max(0.0, (parsed_dt - now).total_seconds())
    except Exception:
        pass
    return None

def _extract_retry_after(exc: Exception) -> Optional[float]:
    raw_seconds = _parse_retry_after_raw(exc)
    if raw_seconds is None:
        return None
    return max(1.0, min(raw_seconds, _RETRY_AFTER_MAX_SECONDS))

def _retry_after_exceeds_threshold(exc: Exception) -> bool:
    raw_seconds = _parse_retry_after_raw(exc)
    if raw_seconds is None:
        return False
    return raw_seconds > _RETRY_AFTER_MAX_SECONDS

def classify_error(provider: str, exc: Exception) -> LLMErrorCategory:
    if _openai is not None and provider in ('openai', 'groq'):
        sdk = _openai if provider == 'openai' else _groq
        if sdk is not None:
            if isinstance(exc, sdk.BadRequestError):
                return LLMErrorCategory.HARD_FAILURE
            if isinstance(exc, sdk.AuthenticationError):
                return LLMErrorCategory.PROVIDER_FALLBACK
            if isinstance(exc, (sdk.RateLimitError, sdk.APIConnectionError, sdk.InternalServerError)):
                return LLMErrorCategory.RETRYABLE
    if _openai is not None and provider == 'openai':
        if isinstance(exc, _openai.BadRequestError):
            return LLMErrorCategory.HARD_FAILURE
        if isinstance(exc, _openai.AuthenticationError):
            return LLMErrorCategory.PROVIDER_FALLBACK
        if isinstance(exc, (_openai.RateLimitError, _openai.APIConnectionError, _openai.InternalServerError)):
            return LLMErrorCategory.RETRYABLE
    if _groq is not None and provider == 'groq':
        if isinstance(exc, _groq.BadRequestError):
            return LLMErrorCategory.HARD_FAILURE
        if isinstance(exc, _groq.AuthenticationError):
            return LLMErrorCategory.PROVIDER_FALLBACK
        if isinstance(exc, (_groq.RateLimitError, _groq.APIConnectionError, _groq.InternalServerError)):
            return LLMErrorCategory.RETRYABLE
    if _anthropic is not None and provider == 'claude':
        if isinstance(exc, _anthropic.BadRequestError):
            return LLMErrorCategory.HARD_FAILURE
        if isinstance(exc, _anthropic.AuthenticationError):
            return LLMErrorCategory.PROVIDER_FALLBACK
        if isinstance(exc, (_anthropic.RateLimitError, _anthropic.APIConnectionError, _anthropic.InternalServerError, _anthropic.OverloadedError)):
            return LLMErrorCategory.RETRYABLE
    status_code = _extract_status_code(exc)
    if status_code is not None:
        if status_code == 400:
            return LLMErrorCategory.HARD_FAILURE
        if status_code in (401, 403):
            return LLMErrorCategory.PROVIDER_FALLBACK
        if status_code == 429 or 500 <= status_code <= 504:
            return LLMErrorCategory.RETRYABLE
    if isinstance(exc, asyncio.TimeoutError):
        return LLMErrorCategory.RETRYABLE
    if provider == 'gemini':
        exc_lower = str(exc).lower()
        if 'quota' in exc_lower or 'resource exhausted' in exc_lower:
            return LLMErrorCategory.RETRYABLE
        if 'permission denied' in exc_lower or 'iam_permission' in exc_lower:
            return LLMErrorCategory.PROVIDER_FALLBACK
        if 'invalid argument' in exc_lower or 'bad request' in exc_lower:
            return LLMErrorCategory.HARD_FAILURE
        if 'unavailable' in exc_lower or 'deadline exceeded' in exc_lower:
            return LLMErrorCategory.RETRYABLE
    logger.warning(f"classify_error: unrecognised exception type '{type(exc).__name__}' for provider '{provider}' — defaulting to RETRYABLE. Consider adding an explicit mapping. exc={exc!r}")
    return LLMErrorCategory.RETRYABLE