from __future__ import annotations
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Optional
PYTHON_TO_GCP_SEVERITY = {logging.DEBUG: 'DEBUG', logging.INFO: 'INFO', logging.WARNING: 'WARNING', logging.ERROR: 'ERROR', logging.CRITICAL: 'CRITICAL'}
SERVICE_NAME = 'coresight-backend'
_RESET = '\x1b[0m'
_BOLD = '\x1b[1m'
_DIM = '\x1b[2m'
_SEVERITY_STYLES = {logging.DEBUG: '\x1b[36m', logging.INFO: '\x1b[32m', logging.WARNING: '\x1b[33m', logging.ERROR: '\x1b[31m', logging.CRITICAL: '\x1b[1;31m'}
_IS_TTY = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
if not _IS_TTY:
    _RESET = _BOLD = _DIM = ''
    _SEVERITY_STYLES = {k: '' for k in _SEVERITY_STYLES}
_EXTRA_KEYS = ('request_id', 'client_id', 'request_path', 'http_method', 'user_id')

class GCPStructuredFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {'severity': PYTHON_TO_GCP_SEVERITY.get(record.levelno, 'DEFAULT'), 'message': record.getMessage(), 'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(), 'logging.googleapis.com/sourceLocation': {'file': record.pathname, 'line': record.lineno, 'function': record.funcName}, 'serviceContext': {'service': SERVICE_NAME}, 'logger': record.name, 'module': record.module, 'pid': record.process}
        if record.exc_info and record.exc_info[0] is not None:
            log_entry['exception'] = {'type': record.exc_info[0].__name__, 'message': str(record.exc_info[1]), 'stackTrace': ''.join(traceback.format_exception(*record.exc_info))}
        for key in _EXTRA_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value
        return json.dumps(log_entry, default=str, ensure_ascii=False)

class HumanReadableFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        color = _SEVERITY_STYLES.get(record.levelno, '')
        severity = f'{color}{record.levelname:<8}{_RESET}'
        name = f'{_DIM}{record.name}{_RESET}'
        msg = record.getMessage()
        extras = []
        for key in _EXTRA_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                extras.append(f'{key}={value}')
        extra_str = f" {_DIM}[{', '.join(extras)}]{_RESET}" if extras else ''
        line = f'{_DIM}{ts}{_RESET} | {severity} | {name} — {msg}{extra_str}'
        if record.exc_info and record.exc_info[0] is not None:
            tb = ''.join(traceback.format_exception(*record.exc_info))
            line += f"\n{_SEVERITY_STYLES.get(logging.ERROR, '')}{tb}{_RESET}"
        return line

def _resolve_log_format() -> str:
    explicit = os.getenv('LOG_FORMAT', '').strip().lower()
    if explicit in ('json', 'text'):
        return explicit
    return 'text'

def setup_logging(level: Optional[str]=None) -> None:
    log_level = level or os.getenv('LOG_LEVEL', 'INFO')
    log_format = _resolve_log_format()
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    if log_format == 'json':
        handler.setFormatter(GCPStructuredFormatter())
    else:
        handler.setFormatter(HumanReadableFormatter())
    root_logger.addHandler(handler)
    for noisy_logger in ('uvicorn.access', 'uvicorn.error', 'httpx', 'httpcore', 'watchfiles', 'motor', 'pymongo'):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    root_logger.info(f'Logging initialized | service={SERVICE_NAME} | level={log_level} | format={log_format}')