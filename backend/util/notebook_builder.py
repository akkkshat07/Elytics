import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
NBFORMAT_MAJOR = 4
NBFORMAT_MINOR = 5

class NotebookBuilder:

    def __init__(self, output_dir: str='test_outputs', name_prefix: str='analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.filename = f'{name_prefix}_{self.timestamp}.ipynb'
        self.filepath = self.output_dir / self.filename
        self._notebook: Dict[str, Any] = {'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.10.0', 'mimetype': 'text/x-python', 'file_extension': '.py'}}, 'nbformat': NBFORMAT_MAJOR, 'nbformat_minor': NBFORMAT_MINOR, 'cells': []}
        logger.info(f'NotebookBuilder created — will save to {self.filepath}')

    def add_markdown_cell(self, text: str) -> None:
        cell = self._make_markdown_cell(text)
        self._notebook['cells'].append(cell)

    def add_code_cell(self, source: str, stdout: str='', stderr: str='', exception: Optional[str]=None, execution_count: Optional[int]=None) -> None:
        outputs: List[Dict[str, Any]] = []
        if stdout:
            outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': self._to_lines(stdout)})
        if stderr:
            outputs.append({'output_type': 'stream', 'name': 'stderr', 'text': self._to_lines(stderr)})
        if exception:
            outputs.append({'output_type': 'error', 'ename': 'ExecutionError', 'evalue': exception.split('\n')[-1] if exception else '', 'traceback': exception.split('\n')})
        cell = self._make_code_cell(source, outputs, execution_count)
        self._notebook['cells'].append(cell)

    def save(self) -> Path:
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self._notebook, f, indent=1, ensure_ascii=False)
            logger.debug(f"Notebook saved ({len(self._notebook['cells'])} cells) → {self.filepath}")
        except Exception as e:
            logger.error(f'Failed to save notebook: {e}')
        return self.filepath

    def add_output_to_last_cell(self, stdout: str='', stderr: str='', exception: Optional[str]=None) -> None:
        cells = self._notebook.get('cells', [])
        for cell in reversed(cells):
            if cell.get('cell_type') == 'code':
                if stdout:
                    cell['outputs'].append({'output_type': 'stream', 'name': 'stdout', 'text': self._to_lines(stdout)})
                if stderr:
                    cell['outputs'].append({'output_type': 'stream', 'name': 'stderr', 'text': self._to_lines(stderr)})
                if exception:
                    cell['outputs'].append({'output_type': 'error', 'ename': 'ExecutionError', 'evalue': exception.split('\n')[-1] if exception else '', 'traceback': exception.split('\n')})
                return
        logger.warning('add_output_to_last_cell: no code cell found to append to')

    @property
    def cell_count(self) -> int:
        return len(self._notebook['cells'])

    @staticmethod
    def _to_lines(text: str) -> List[str]:
        if not text:
            return []
        lines = text.split('\n')
        return [line + '\n' if i < len(lines) - 1 else line for i, line in enumerate(lines)]

    @staticmethod
    def _make_markdown_cell(source: str) -> Dict[str, Any]:
        return {'cell_type': 'markdown', 'metadata': {}, 'source': NotebookBuilder._to_lines(source)}

    @staticmethod
    def _make_code_cell(source: str, outputs: List[Dict[str, Any]], execution_count: Optional[int]=None) -> Dict[str, Any]:
        return {'cell_type': 'code', 'metadata': {}, 'source': NotebookBuilder._to_lines(source), 'outputs': outputs, 'execution_count': execution_count}