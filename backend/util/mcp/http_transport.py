from __future__ import annotations
import asyncio
import logging
import ssl
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Literal, Tuple
import httpx
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.client.sse import sse_client
from mcp.shared.message import SessionMessage
from util.mcp.transport import TransportError
logger = logging.getLogger(__name__)
DEFAULT_HTTP_TIMEOUT = 30.0
DEFAULT_SSE_READ_TIMEOUT = 300.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_RETRY_BACKOFF = 2.0

class HttpTransportError(TransportError):

    def __init__(self, message: str):
        super().__init__(message)

class HttpConnectionError(HttpTransportError):

    def __init__(self, url: str, original_error: Exception):
        super().__init__(f'Failed to connect to {url}: {original_error}')
        self.url = url
        self.original_error = original_error

class HttpAuthenticationError(HttpTransportError):

    def __init__(self, url: str, status_code: int, detail: str=''):
        message = f'Authentication failed for {url}: HTTP {status_code}'
        if detail:
            message += f' - {detail}'
        super().__init__(message)
        self.url = url
        self.status_code = status_code

class SseConnectionError(HttpTransportError):

    def __init__(self, message: str, url: str | None=None):
        super().__init__(message)
        self.url = url

class SseReconnectionError(HttpTransportError):

    def __init__(self, url: str, attempts: int, last_error: Exception | None=None):
        super().__init__(f'SSE reconnection failed after {attempts} attempts to {url}')
        self.url = url
        self.attempts = attempts
        self.last_error = last_error

class SslVerificationError(HttpTransportError):

    def __init__(self, url: str, original_error: Exception):
        super().__init__(f'SSL verification failed for {url}: {original_error}')
        self.url = url
        self.original_error = original_error

@dataclass
class AuthConfig:
    type: Literal['bearer', 'api_key', 'basic', 'custom']
    token: str | None = None
    username: str | None = None
    password: str | None = None
    header_name: str | None = None
    header_value_prefix: str | None = None

    def __post_init__(self):
        if self.type == 'bearer':
            if not self.token:
                raise ValueError("Bearer auth requires 'token'")
            self.header_name = self.header_name or 'Authorization'
            self.header_value_prefix = self.header_value_prefix or 'Bearer '
        elif self.type == 'api_key':
            if not self.token:
                raise ValueError("API key auth requires 'token'")
            self.header_name = self.header_name or 'X-API-Key'
            self.header_value_prefix = self.header_value_prefix or ''
        elif self.type == 'basic':
            if not self.username or not self.password:
                raise ValueError("Basic auth requires 'username' and 'password'")
        elif self.type == 'custom':
            if not self.header_name or not self.token:
                raise ValueError("Custom auth requires 'header_name' and 'token'")
            self.header_value_prefix = self.header_value_prefix or ''

@dataclass
class HttpSseConfig:
    url: str
    headers: Dict[str, str] | None = None
    auth: AuthConfig | None = None
    timeout: float = DEFAULT_HTTP_TIMEOUT
    sse_read_timeout: float = DEFAULT_SSE_READ_TIMEOUT
    verify_ssl: bool = True
    ssl_cert: str | None = None
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
    retry_backoff: float = DEFAULT_RETRY_BACKOFF

    def __post_init__(self):
        if not self.url:
            raise ValueError('URL is required')
        if not self.url.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        if self.timeout <= 0:
            raise ValueError('Timeout must be positive')
        if self.sse_read_timeout <= 0:
            raise ValueError('SSE read timeout must be positive')
        if self.max_retries < 0:
            raise ValueError('Max retries cannot be negative')
        if self.retry_delay <= 0:
            raise ValueError('Retry delay must be positive')
        if self.retry_backoff < 1:
            raise ValueError('Retry backoff must be >= 1')

class HttpSseTransport:

    def __init__(self, config: HttpSseConfig):
        self._config = config
        self._session_id: str | None = None

    @property
    def url(self) -> str:
        return self._config.url

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self._config.headers:
            headers.update(self._config.headers)
        if self._config.auth:
            auth = self._config.auth
            if auth.type in ('bearer', 'api_key', 'custom'):
                header_name = auth.header_name or 'Authorization'
                prefix = auth.header_value_prefix or ''
                headers[header_name] = f'{prefix}{auth.token}'
        return headers

    def _build_httpx_auth(self) -> httpx.Auth | None:
        if self._config.auth and self._config.auth.type == 'basic':
            return httpx.BasicAuth(self._config.auth.username or '', self._config.auth.password or '')
        return None

    def _create_httpx_client_factory(self) -> Callable[..., httpx.AsyncClient]:
        verify: bool | ssl.SSLContext = self._config.verify_ssl
        if self._config.ssl_cert and self._config.verify_ssl:
            try:
                ssl_context = ssl.create_default_context()
                ssl_context.load_verify_locations(self._config.ssl_cert)
                verify = ssl_context
            except Exception as e:
                raise SslVerificationError(self._config.url, e) from e

        @asynccontextmanager
        async def factory(headers: Dict[str, Any] | None=None, auth: httpx.Auth | None=None, timeout: httpx.Timeout | None=None) -> AsyncIterator[httpx.AsyncClient]:
            async with httpx.AsyncClient(headers=headers, auth=auth, timeout=timeout, verify=verify, follow_redirects=True) as client:
                yield client
        return factory

    def _on_session_created(self, session_id: str) -> None:
        self._session_id = session_id
        logger.debug(f'Session created with ID: {session_id}')

    async def _connect_once(self) -> AsyncIterator[Tuple[MemoryObjectReceiveStream[SessionMessage | Exception], MemoryObjectSendStream[SessionMessage]]]:
        try:
            async with sse_client(url=self._config.url, headers=self._build_headers(), timeout=self._config.timeout, sse_read_timeout=self._config.sse_read_timeout, httpx_client_factory=self._create_httpx_client_factory(), auth=self._build_httpx_auth(), on_session_created=self._on_session_created) as (read_stream, write_stream):
                yield (read_stream, write_stream)
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if status_code in (401, 403):
                raise HttpAuthenticationError(self._config.url, status_code, e.response.text[:200] if e.response.text else '') from e
            raise HttpConnectionError(self._config.url, e) from e
        except ssl.SSLError as e:
            raise SslVerificationError(self._config.url, e) from e
        except ssl.SSLCertVerificationError as e:
            raise SslVerificationError(self._config.url, e) from e
        except httpx.ConnectError as e:
            raise HttpConnectionError(self._config.url, e) from e
        except httpx.ConnectTimeout as e:
            raise HttpConnectionError(self._config.url, e) from e
        except httpx.TimeoutException as e:
            raise HttpConnectionError(self._config.url, e) from e

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[Tuple[MemoryObjectReceiveStream[SessionMessage | Exception], MemoryObjectSendStream[SessionMessage]]]:
        last_error: Exception | None = None
        delay = self._config.retry_delay
        for attempt in range(self._config.max_retries + 1):
            try:
                async with self._connect_once() as streams:
                    logger.info(f'Connected to MCP server at {self._config.url}' + (f' (attempt {attempt + 1})' if attempt > 0 else ''))
                    yield streams
                    return
            except (HttpAuthenticationError, SslVerificationError):
                raise
            except (HttpConnectionError, SseConnectionError) as e:
                last_error = e
                if attempt < self._config.max_retries:
                    logger.warning(f'Connection attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...')
                    await asyncio.sleep(delay)
                    delay *= self._config.retry_backoff
                else:
                    logger.error(f'All {self._config.max_retries + 1} connection attempts failed')
            except Exception as e:
                raise HttpTransportError(f'Unexpected error: {e}') from e
        raise SseReconnectionError(self._config.url, self._config.max_retries + 1, last_error)

    async def __aenter__(self) -> Tuple[MemoryObjectReceiveStream[SessionMessage | Exception], MemoryObjectSendStream[SessionMessage]]:
        self._context_manager = self.connect()
        return await self._context_manager.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self, '_context_manager'):
            await self._context_manager.__aexit__(exc_type, exc_val, exc_tb)

    def __repr__(self) -> str:
        auth_type = self._config.auth.type if self._config.auth else 'none'
        ssl_status = 'verified' if self._config.verify_ssl else 'unverified'
        return f"HttpSseTransport(url='{self._config.url}', auth={auth_type}, ssl={ssl_status})"