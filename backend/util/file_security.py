import re
from pathlib import Path
from typing import Tuple

def sanitize_filename(filename: str) -> str:
    if not filename:
        return 'file'
    filename = Path(filename).name
    filename = filename.replace('..', '').replace('../', '').replace('..\\', '')
    filename = re.sub('[^a-zA-Z0-9._-]', '_', filename)
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_len = 255 - len(ext) - 1 if ext else 255
        filename = name[:max_name_len] + ('.' + ext if ext else '')
    if not filename or filename == '.' or filename == '..':
        filename = 'file'
    return filename

def validate_path_within_directory(file_path: Path, base_dir: Path) -> Tuple[bool, str]:
    try:
        resolved_file = file_path.resolve()
        resolved_base = base_dir.resolve()
        try:
            resolved_file.relative_to(resolved_base)
            return (True, '')
        except ValueError:
            return (False, f'Path traversal detected: {file_path} is outside allowed directory')
    except Exception as e:
        return (False, f'Path validation error: {str(e)}')