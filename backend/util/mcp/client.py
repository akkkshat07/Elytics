from __future__ import annotations
import asyncio
import logging
from datetime import timedelta
from types import TracebackType
from typing import Any, Protocol, runtime_checkable
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.client.session import ClientSession
from mcp.shared.message import SessionMessage
from mcp.shared.session import RequestResponder
import mcp.types as types
logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT_SECONDS = 180

class McpError(Exception):

    def __init__(self, message: str, code: int | None=None, data: Any=None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data

    def __str__(self) -> str:
        if self.code is not None:
            return f'McpError(code={self.code}): {self.message}'
        return f'McpError: {self.message}'

class McpTimeoutError(McpError):

    def __init__(self, message: str='Operation timed out', timeout_seconds: float=DEFAULT_TIMEOUT_SECONDS):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds

    def __str__(self) -> str:
        return f'McpTimeoutError: {self.message} (timeout: {self.timeout_seconds}s)'

class McpConnectionError(McpError):
    pass

class McpToolError(McpError):

    def __init__(self, message: str, tool_name: str, code: int | None=None, data: Any=None):
        super().__init__(message, code, data)
        self.tool_name = tool_name

    def __str__(self) -> str:
        if self.code is not None:
            return f'McpToolError(tool={self.tool_name}, code={self.code}): {self.message}'
        return f'McpToolError(tool={self.tool_name}): {self.message}'

class McpNotInitializedError(McpError):

    def __init__(self):
        super().__init__('Client not initialized. Call initialize() first.')

@runtime_checkable
class Transport(Protocol):

    async def aclose(self) -> None:
        ...

async def _default_message_handler(message: RequestResponder[types.ServerRequest, types.ClientResult] | types.ServerNotification | Exception) -> None:
    if isinstance(message, Exception):
        logger.warning(f'Received exception from server: {message}')
    elif isinstance(message, types.ServerNotification):
        logger.debug(f'Received notification: {message}')
    else:
        logger.debug(f'Received request: {message}')

class McpClient:

    def __init__(self, read_stream: MemoryObjectReceiveStream[SessionMessage | Exception], write_stream: MemoryObjectSendStream[SessionMessage], timeout_seconds: float=DEFAULT_TIMEOUT_SECONDS, client_info: types.Implementation | None=None):
        self._read_stream = read_stream
        self._write_stream = write_stream
        self._timeout_seconds = timeout_seconds
        self._client_info = client_info or types.Implementation(name='coresight-mcp-client', version='1.0.0')
        self._session: ClientSession | None = None
        self._initialized = False
        self._server_capabilities: types.ServerCapabilities | None = None
        self._tools_cache: list[types.Tool] | None = None
        self._init_result: types.InitializeResult | None = None
        self._entered_task_id: int | None = None

    async def __aenter__(self) -> 'McpClient':
        try:
            task = asyncio.current_task()
            self._entered_task_id = id(task) if task else None
            self._session = ClientSession(read_stream=self._read_stream, write_stream=self._write_stream, read_timeout_seconds=timedelta(seconds=self._timeout_seconds), client_info=self._client_info, message_handler=_default_message_handler)
            await self._session.__aenter__()
            self._init_result = await asyncio.wait_for(self._session.initialize(), timeout=self._timeout_seconds)
            self._server_capabilities = self._init_result.capabilities
            self._initialized = True
            logger.info(f'MCP client initialized successfully. Server: {self._init_result.serverInfo.name} v{self._init_result.serverInfo.version}, Protocol: {self._init_result.protocolVersion}')
            return self
        except asyncio.TimeoutError as e:
            if self._session is not None:
                try:
                    await self._session.__aexit__(type(e), e, e.__traceback__)
                except Exception:
                    pass
            self._initialized = False
            self._session = None
            raise McpTimeoutError('Initialization handshake timed out', timeout_seconds=self._timeout_seconds) from e
        except Exception as e:
            if self._session is not None:
                try:
                    await self._session.__aexit__(type(e), e, e.__traceback__)
                except Exception:
                    pass
            self._initialized = False
            self._session = None
            raise McpConnectionError(f'Failed to initialize MCP connection: {e}') from e

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> bool | None:
        self._initialized = False
        self._tools_cache = None
        self._server_capabilities = None
        if self._session is not None:
            try:
                current_task = asyncio.current_task()
                current_task_id = id(current_task) if current_task else None
                if self._entered_task_id is not None and current_task_id != self._entered_task_id:
                    logger.warning('Skipping MCP session __aexit__ in different task (entered_task_id=%s, current_task_id=%s)', self._entered_task_id, current_task_id)
                    self._session = None
                    return None
                result = await asyncio.wait_for(self._session.__aexit__(exc_type, exc_val, exc_tb), timeout=5.0)
                self._session = None
                self._entered_task_id = None
                return result
            except Exception as e:
                logger.warning(f'Error during session cleanup: {e}')
                self._session = None
                self._entered_task_id = None
        return None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def server_capabilities(self) -> types.ServerCapabilities | None:
        return self._server_capabilities

    @property
    def server_info(self) -> types.Implementation | None:
        if self._init_result:
            return self._init_result.serverInfo
        return None

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @timeout_seconds.setter
    def timeout_seconds(self, value: float) -> None:
        if value <= 0:
            raise ValueError('Timeout must be a positive number')
        self._timeout_seconds = value

    def _ensure_initialized(self) -> None:
        if not self._initialized or self._session is None:
            raise McpNotInitializedError()

    async def list_tools(self, use_cache: bool=True) -> list[types.Tool]:
        self._ensure_initialized()
        if use_cache and self._tools_cache is not None:
            return self._tools_cache
        try:
            result = await asyncio.wait_for(self._session.list_tools(), timeout=self._timeout_seconds)
            self._tools_cache = result.tools
            logger.debug(f'Listed {len(result.tools)} tools from MCP server')
            return result.tools
        except asyncio.TimeoutError as e:
            raise McpTimeoutError('list_tools request timed out', timeout_seconds=self._timeout_seconds) from e
        except Exception as e:
            raise McpError(f'Failed to list tools: {e}') from e

    async def call_tool(self, name: str, arguments: dict[str, Any] | None=None, timeout_seconds: float | None=None) -> Any:
        self._ensure_initialized()
        effective_timeout = timeout_seconds if timeout_seconds is not None else self._timeout_seconds
        try:
            result = await asyncio.wait_for(self._session.call_tool(name=name, arguments=arguments, read_timeout_seconds=timedelta(seconds=effective_timeout)), timeout=effective_timeout + 10)
            if result.isError:
                error_message = self._extract_error_message(result)
                raise McpToolError(message=error_message, tool_name=name)
            return self._extract_result(result)
        except asyncio.TimeoutError as e:
            raise McpTimeoutError(f"Tool '{name}' execution timed out after {effective_timeout} seconds", timeout_seconds=effective_timeout) from e
        except McpToolError:
            raise
        except Exception as e:
            if isinstance(e, McpError):
                raise
            message = str(e)
            if 'Timed out while waiting for response' in message:
                raise McpTimeoutError(f"Tool '{name}' execution timed out after {effective_timeout} seconds: {message}", timeout_seconds=effective_timeout) from e
            raise McpError(f"Failed to call tool '{name}': {e}") from e

    def _extract_result(self, result: types.CallToolResult) -> Any:
        if result.structuredContent is not None:
            return result.structuredContent
        if not result.content:
            return None
        if len(result.content) == 1:
            content = result.content[0]
            if isinstance(content, types.TextContent):
                text = content.text
                return self._try_parse_value(text)
            elif isinstance(content, types.ImageContent):
                return {'type': 'image', 'data': content.data, 'mimeType': content.mimeType}
            elif isinstance(content, types.ResourceContent):
                return {'type': 'resource', 'uri': str(content.resource.uri)}
        results = []
        for content in result.content:
            if isinstance(content, types.TextContent):
                results.append(self._try_parse_value(content.text))
            elif isinstance(content, types.ImageContent):
                results.append({'type': 'image', 'data': content.data, 'mimeType': content.mimeType})
            elif isinstance(content, types.ResourceContent):
                results.append({'type': 'resource', 'uri': str(content.resource.uri)})
            else:
                results.append(str(content))
        return results if len(results) > 1 else results[0] if results else None

    def _try_parse_value(self, text: str) -> Any:
        text = text.strip()
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            pass
        if text.lower() == 'true':
            return True
        if text.lower() == 'false':
            return False
        if text.lower() in ('none', 'null'):
            return None
        return text

    def _extract_error_message(self, result: types.CallToolResult) -> str:
        if result.content:
            messages = []
            for content in result.content:
                if isinstance(content, types.TextContent):
                    messages.append(content.text)
            if messages:
                return '\n'.join(messages)
        return 'Tool execution failed with unknown error'

    async def ping(self) -> bool:
        self._ensure_initialized()
        try:
            await asyncio.wait_for(self._session.send_ping(), timeout=self._timeout_seconds)
            return True
        except Exception as e:
            logger.warning(f'Ping failed: {e}')
            return False

    def clear_tools_cache(self) -> None:
        self._tools_cache = None

    def __repr__(self) -> str:
        status = 'initialized' if self._initialized else 'not initialized'
        return f'McpClient(status={status}, timeout={self._timeout_seconds}s)'