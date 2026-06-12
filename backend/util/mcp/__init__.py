class McpClient:
    pass
class McpError(Exception):
    pass
class McpTimeoutError(Exception):
    pass
class McpConnectionError(Exception):
    pass
class McpToolError(Exception):
    pass
class McpNotInitializedError(Exception):
    pass
DEFAULT_TIMEOUT_SECONDS = 180

class StdioTransport:
    pass
class TransportError(Exception):
    pass
class TransportNotStartedError(Exception):
    pass
class TransportAlreadyStartedError(Exception):
    pass
class TransportClosedError(Exception):
    pass
class MessageSerializationError(Exception):
    pass
class MessageDeserializationError(Exception):
    pass
class ProcessStartError(Exception):
    pass
DEFAULT_READ_TIMEOUT_SECONDS = 60

class HttpSseTransport:
    pass
class HttpSseConfig:
    pass
class AuthConfig:
    pass
class HttpTransportError(Exception):
    pass
class HttpConnectionError(Exception):
    pass
class HttpAuthenticationError(Exception):
    pass
class SseConnectionError(Exception):
    pass
class SseReconnectionError(Exception):
    pass
class SslVerificationError(Exception):
    pass
DEFAULT_HTTP_TIMEOUT = 60
DEFAULT_SSE_READ_TIMEOUT = 60

__all__ = ['McpClient', 'McpError', 'McpTimeoutError', 'McpConnectionError', 'McpToolError', 'McpNotInitializedError', 'DEFAULT_TIMEOUT_SECONDS', 'StdioTransport', 'TransportError', 'TransportNotStartedError', 'TransportAlreadyStartedError', 'TransportClosedError', 'MessageSerializationError', 'MessageDeserializationError', 'ProcessStartError', 'DEFAULT_READ_TIMEOUT_SECONDS', 'HttpSseTransport', 'HttpSseConfig', 'AuthConfig', 'HttpTransportError', 'HttpConnectionError', 'HttpAuthenticationError', 'SseConnectionError', 'SseReconnectionError', 'SslVerificationError', 'DEFAULT_HTTP_TIMEOUT', 'DEFAULT_SSE_READ_TIMEOUT']