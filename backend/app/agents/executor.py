import logging
import json
import math
import re
import sys
import threading
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from ..state import QueryState
logger = logging.getLogger(__name__)
_SAFE_BUILTINS = {'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict, 'dir': dir, 'enumerate': enumerate, 'filter': filter, 'float': float, 'format': format, 'frozenset': frozenset, 'getattr': getattr, 'hasattr': hasattr, 'hash': hash, 'int': int, 'isinstance': isinstance, 'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list, 'map': map, 'max': max, 'min': min, 'next': next, 'object': object, 'print': print, 'range': range, 'repr': repr, 'reversed': reversed, 'round': round, 'set': set, 'slice': slice, 'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple, 'type': type, 'vars': vars, 'zip': zip, 'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError, 'KeyError': KeyError, 'IndexError': IndexError, 'ZeroDivisionError': ZeroDivisionError}
_EXEC_TIMEOUT_SECONDS = 30

class ExecutionTimeoutError(Exception):
  pass

class ExecutorAgent:

  def __init__(self):
    logger.info('ExecutorAgent initialized')

  def _build_execution_namespace(self, query_results: List[Dict]) -> Dict[str, Any]:
    return {'__builtins__': _SAFE_BUILTINS, 'pd': pd, 'np': np, 'px': px, 'go': go, 'json': json, 'math': math, 're': re, 'query_results': query_results, 'charts': [], 'statistics': {}, 'text_outputs': []}

  def _execute_with_timeout(self, code: str, namespace: Dict) -> bool:
    execution_error = [None]
    completed = [False]

    def run():
      try:
        exec(code, namespace)
        completed[0] = True
      except Exception as e:
        execution_error[0] = e
        completed[0] = True
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=_EXEC_TIMEOUT_SECONDS)
    if not completed[0]:
      raise ExecutionTimeoutError(f'Code execution exceeded {_EXEC_TIMEOUT_SECONDS} second timeout')
    if execution_error[0] is not None:
      raise execution_error[0]
    return True

  def _serialize_charts(self, charts: list) -> List[Dict]:
    serialized = []
    for chart in charts:
      try:
        if hasattr(chart, 'to_dict'):
          chart = chart.to_dict()
        chart_json_str = json.dumps(chart, cls=plotly.utils.PlotlyJSONEncoder)
        serialized.append(json.loads(chart_json_str))
      except Exception as e:
        logger.warning(f'Could not serialize chart: {e}')
    return serialized

  def _serialize_statistics(self, statistics: dict) -> Dict[str, Any]:
    safe_stats = {}
    for key, value in statistics.items():
      try:
        if isinstance(value, (np.integer,)):
          safe_stats[key] = int(value)
        elif isinstance(value, (np.floating,)):
          safe_stats[key] = float(value)
        elif isinstance(value, np.ndarray):
          safe_stats[key] = value.tolist()
        elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
          safe_stats[key] = None
        else:
          safe_stats[key] = value
      except Exception:
        safe_stats[key] = str(value)
    return safe_stats

  def process(self, state: QueryState) -> Dict[str, Any]:
    code = state.get('generated_python', '')
    query_results = state.get('query_results', [])
    if not code or not code.strip():
      logger.warning('ExecutorAgent: No code to execute')
      return {'execution_results': {'error': 'No Python code was generated'}, 'charts': [], 'step_log': [' Executor: No code to execute']}
    logger.info(f'ExecutorAgent executing {len(code)} chars of code')
    logger.debug(f'Code preview:\n{code[:500]}')
    namespace = self._build_execution_namespace(query_results)
    try:
      self._execute_with_timeout(code, namespace)
      raw_charts = namespace.get('charts', [])
      raw_statistics = namespace.get('statistics', {})
      raw_text_outputs = namespace.get('text_outputs', [])
      serialized_charts = self._serialize_charts(raw_charts)
      serialized_stats = self._serialize_statistics(raw_statistics)
      execution_results = {'statistics': serialized_stats, 'text_outputs': [str(t) for t in raw_text_outputs], 'total_charts': len(serialized_charts), 'error': None}
      logger.info(f'ExecutorAgent complete | charts={len(serialized_charts)} | stats_keys={list(serialized_stats.keys())} | text_outputs={len(raw_text_outputs)}')
      return {'execution_results': execution_results, 'charts': serialized_charts, 'step_log': [f' Executor: Ran successfully | charts={len(serialized_charts)} | stats={len(serialized_stats)} keys | findings={len(raw_text_outputs)}']}
    except ExecutionTimeoutError as e:
      error_msg = f'Code execution timed out after {_EXEC_TIMEOUT_SECONDS}s'
      logger.error(error_msg)
      return {'execution_results': {'error': error_msg, 'statistics': {}, 'text_outputs': []}, 'charts': [], 'error': error_msg, 'step_log': [f' Executor: TIMEOUT — {error_msg}']}
    except Exception as e:
      error_msg = f'Code execution failed: {type(e).__name__}: {e}'
      logger.error(error_msg, exc_info=True)
      partial_charts = self._serialize_charts(namespace.get('charts', []))
      partial_stats = self._serialize_statistics(namespace.get('statistics', {}))
      return {'execution_results': {'error': error_msg, 'statistics': partial_stats, 'text_outputs': [str(t) for t in namespace.get('text_outputs', [])]}, 'charts': partial_charts, 'error': error_msg, 'step_log': [f' Executor: Runtime error — {error_msg}']}