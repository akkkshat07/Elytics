import json
import logging
import os
import subprocess
import threading
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
DEFAULT_READ_TIMEOUT_SECONDS = 60

class TransportError(Exception):

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f'TransportError: {self.message}'

class TransportNotStartedError(TransportError):

    def __init__(self):
        super().__init__('Transport not started. Call start() first.')

class TransportAlreadyStartedError(TransportError):

    def __init__(self):
        super().__init__('Transport already started.')

class TransportClosedError(TransportError):

    def __init__(self):
        super().__init__('Transport has been closed.')

class MessageSerializationError(TransportError):

    def __init__(self, message: str, original_error: Exception):
        super().__init__(f'Failed to serialize message: {message}')
        self.original_error = original_error

class MessageDeserializationError(TransportError):

    def __init__(self, raw_data: str, original_error: Exception):
        super().__init__(f'Failed to deserialize response: {raw_data[:100]}...')
        self.raw_data = raw_data
        self.original_error = original_error

class ProcessStartError(TransportError):

    def __init__(self, command: str, original_error: Exception):
        super().__init__(f'Failed to start subprocess: {command}')
        self.command = command
        self.original_error = original_error

class StdioTransport:

    def __init__(self, command: str, args: Optional[List[str]]=None, env: Optional[Dict[str, str]]=None, read_timeout: float=DEFAULT_READ_TIMEOUT_SECONDS, working_directory: Optional[str]=None):
        self._command = command
        self._args = args or []
        self._env = env
        self._read_timeout = read_timeout
        self._working_directory = working_directory
        self._process: Optional[subprocess.Popen] = None
        self._started = False
        self._closed = False
        self._stderr_thread: Optional[threading.Thread] = None
        self._stop_stderr_thread = threading.Event()
        self._lock = threading.Lock()
        logger.debug(f"StdioTransport initialized with command: {self._command} {' '.join(self._args)}")

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    @property
    def command_line(self) -> str:
        return f"{self._command} {' '.join(self._args)}".strip()

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise TransportClosedError()
            if self._started:
                raise TransportAlreadyStartedError()
            full_command = [self._command] + self._args
            process_env = os.environ.copy()
            if self._env:
                process_env.update(self._env)
            try:
                logger.info(f'Starting subprocess: {self.command_line}')
                self._process = subprocess.Popen(full_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=process_env, cwd=self._working_directory, bufsize=1, text=True, encoding='utf-8')
                self._started = True
                self._stop_stderr_thread.clear()
                self._stderr_thread = threading.Thread(target=self._stderr_reader, name=f'StdioTransport-stderr-{self._process.pid}', daemon=True)
                self._stderr_thread.start()
                logger.info(f'Subprocess started successfully (PID: {self._process.pid})')
            except FileNotFoundError as e:
                logger.error(f'Command not found: {self._command}')
                raise ProcessStartError(self.command_line, e) from e
            except PermissionError as e:
                logger.error(f'Permission denied executing: {self._command}')
                raise ProcessStartError(self.command_line, e) from e
            except OSError as e:
                logger.error(f'OS error starting subprocess: {e}')
                raise ProcessStartError(self.command_line, e) from e
            except Exception as e:
                logger.error(f'Unexpected error starting subprocess: {e}', exc_info=True)
                raise ProcessStartError(self.command_line, e) from e

    def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_running()
        try:
            json_message = json.dumps(message)
        except (TypeError, ValueError) as e:
            logger.error(f'Failed to serialize message: {e}')
            raise MessageSerializationError(str(message), e) from e
        try:
            logger.debug(f'Sending message: {json_message[:200]}...')
            self._process.stdin.write(json_message + '\n')
            self._process.stdin.flush()
        except BrokenPipeError as e:
            logger.error('Broken pipe - subprocess may have terminated')
            raise TransportError('Subprocess terminated unexpectedly') from e
        except Exception as e:
            logger.error(f'Error writing to stdin: {e}', exc_info=True)
            raise TransportError(f'Failed to send message: {e}') from e
        return self.read_message()

    def read_message(self) -> Dict[str, Any]:
        self._ensure_running()
        try:
            line = self._process.stdout.readline()
            if not line:
                return_code = self._process.poll()
                if return_code is not None:
                    logger.error(f'Subprocess terminated with code: {return_code}')
                    raise TransportError(f'Subprocess terminated unexpectedly (exit code: {return_code})')
                raise TransportError('Received empty response from subprocess')
            line = line.strip()
            logger.debug(f'Received response: {line[:200]}...')
            try:
                return json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f'Failed to parse JSON response: {line[:100]}...')
                raise MessageDeserializationError(line, e) from e
        except TransportError:
            raise
        except Exception as e:
            logger.error(f'Error reading from stdout: {e}', exc_info=True)
            raise TransportError(f'Failed to read message: {e}') from e

    def close(self) -> None:
        with self._lock:
            if self._closed:
                logger.debug('Transport already closed')
                return
            self._closed = True
            logger.info('Closing transport...')
            self._stop_stderr_thread.set()
            if self._process is not None:
                try:
                    if self._process.poll() is None:
                        logger.debug('Terminating subprocess...')
                        if self._process.stdin:
                            try:
                                self._process.stdin.close()
                            except Exception as e:
                                logger.debug(f'Error closing stdin: {e}')
                        self._process.terminate()
                        try:
                            self._process.wait(timeout=5)
                            logger.debug(f'Subprocess terminated gracefully (exit code: {self._process.returncode})')
                        except subprocess.TimeoutExpired:
                            logger.warning('Subprocess did not terminate gracefully, forcing kill')
                            self._process.kill()
                            self._process.wait(timeout=2)
                            logger.debug('Subprocess killed')
                    else:
                        logger.debug(f'Subprocess already terminated (exit code: {self._process.returncode})')
                except Exception as e:
                    logger.error(f'Error during subprocess cleanup: {e}', exc_info=True)
                finally:
                    for pipe in [self._process.stdin, self._process.stdout, self._process.stderr]:
                        if pipe:
                            try:
                                pipe.close()
                            except Exception:
                                pass
                    self._process = None
            if self._stderr_thread is not None and self._stderr_thread.is_alive():
                self._stderr_thread.join(timeout=2)
                if self._stderr_thread.is_alive():
                    logger.warning('Stderr thread did not terminate in time')
                self._stderr_thread = None
            self._started = False
            logger.info('Transport closed successfully')

    def _ensure_running(self) -> None:
        if self._closed:
            raise TransportClosedError()
        if not self._started:
            raise TransportNotStartedError()
        if self._process is None or self._process.poll() is not None:
            exit_code = self._process.returncode if self._process else 'unknown'
            raise TransportError(f'Subprocess is not running (exit code: {exit_code})')

    def _stderr_reader(self) -> None:
        logger.debug('Stderr reader thread started')
        try:
            while not self._stop_stderr_thread.is_set():
                if self._process is None or self._process.stderr is None:
                    break
                try:
                    line = self._process.stderr.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        logger.warning(f'[Subprocess stderr] {line}')
                except Exception as e:
                    if not self._stop_stderr_thread.is_set():
                        logger.debug(f'Error reading stderr: {e}')
                    break
        except Exception as e:
            logger.debug(f'Stderr reader thread error: {e}')
        finally:
            logger.debug('Stderr reader thread stopped')

    def __enter__(self) -> 'StdioTransport':
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        status = 'closed' if self._closed else 'running' if self.is_running else 'not started'
        return f"StdioTransport(command='{self.command_line}', status={status})"