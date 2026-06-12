from __future__ import annotations
import logging
import os
import re
import uuid
logger = logging.getLogger(__name__)

def _should_expose_detailed_errors() -> bool:
    return os.getenv('EXPOSE_DETAILED_ERRORS', 'false').lower() == 'true'

def generate_correlation_id() -> str:
    return str(uuid.uuid4())[:8]

def sanitize_error_message(error_detail: str) -> str:
    if not error_detail:
        return 'An error occurred'
    environment = os.getenv('ENVIRONMENT', 'production')
    if environment == 'development' and _should_expose_detailed_errors():
        return error_detail
    sanitized = error_detail
    sanitized = re.sub('^(Failed to [^:]+):.*$', '\\1.', sanitized, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub('^(An error occurred[^:]*):.*$', '\\1.', sanitized, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub('^(Internal server error):.*$', '\\1.', sanitized, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub('[A-Za-z]:\\\\[^\\s]+|/[^\\s]+', '[path redacted]', sanitized)
    sanitized = re.sub('table\\s+["\\\']?(\\w+)["\\\']?', 'table [redacted]', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub('column\\s+["\\\']?(\\w+)["\\\']?', 'column [redacted]', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub('(postgresql|postgresql\\+asyncpg|postgresql\\+psycopg2|mysql|mysql\\+aiomysql|mysql\\+pymysql|oracle|oracle\\+cx_oracle|mssql|mssql\\+pyodbc|mongodb|redis)://[^\\s]+', '[connection string redacted]', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub('File\\s+"[^"]+",\\s+line\\s+\\d+', '[source location redacted]', sanitized)
    sanitized = sanitized.strip()
    if len(sanitized) < 10:
        return 'An internal error occurred. Please contact support if the issue persists.'
    return sanitized

def create_safe_error_response(exc: Exception, error_type: str=None, status_code: int=500, correlation_id: str=None) -> dict:
    environment = os.getenv('ENVIRONMENT', 'production')
    cid = correlation_id or generate_correlation_id()
    if environment == 'development' and _should_expose_detailed_errors():
        return {'error': str(exc), 'type': str(type(exc).__name__), 'status_code': status_code, 'correlation_id': cid}
    generic_messages = {'database': 'A database operation failed. Please try again later.', 'file': 'A file operation failed. Please check your input and try again.', 'validation': 'Invalid input provided. Please check your request and try again.', 'authentication': 'Authentication failed. Please check your credentials.', 'authorization': 'You do not have permission to perform this action.'}
    error_message = generic_messages.get(error_type, 'An error occurred. Please try again later.')
    return {'error': error_message, 'status_code': status_code, 'correlation_id': cid}