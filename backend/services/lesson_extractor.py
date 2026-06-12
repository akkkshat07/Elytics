import difflib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)

class LessonExtractor:

    @staticmethod
    def extract_from_error_recovery(error_type: str, error_text: str, failed_code: str, fixed_code: str, file_schemas: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        if not failed_code or not fixed_code:
            return []
        diff_lines = list(difflib.unified_diff(failed_code.splitlines(), fixed_code.splitlines(), lineterm=''))
        if len(diff_lines) < 3:
            return []
        if len(diff_lines) > 60:
            return []
        tables = LessonExtractor._infer_tables_from_schemas(file_schemas)
        result = None
        if error_type == 'COLUMN_NOT_FOUND':
            result = LessonExtractor._lesson_column_not_found(error_text, diff_lines, tables)
        elif error_type == 'MISSING_COLUMN':
            result = LessonExtractor._lesson_missing_column(error_text, tables)
        elif error_type in ('ARROW_TYPE_ERROR', 'TYPE_MISMATCH'):
            result = LessonExtractor._lesson_type_error(error_text, diff_lines, tables)
        elif error_type == 'ZERO_ROW_RESULT':
            result = LessonExtractor._lesson_zero_row(failed_code, fixed_code, diff_lines, tables)
        elif error_type == 'IMPORT_ERROR':
            result = LessonExtractor._lesson_import_error(error_text)
        elif error_type == 'FILE_NOT_FOUND':
            result = LessonExtractor._lesson_file_not_found(error_text, tables)
        elif error_type == 'MEMORY_ERROR':
            result = LessonExtractor._lesson_memory_error(diff_lines, tables)
        elif error_type == 'UNDEFINED_VARIABLE':
            result = LessonExtractor._lesson_undefined_var(error_text, diff_lines, tables)
        elif len(diff_lines) <= 15:
            summary = LessonExtractor._summarize_diff(diff_lines)
            if summary:
                result = {'lesson_type': 'general', 'category': 'A', 'lesson': f'Error type {error_type}: {summary}', 'tables_involved': tables, 'source': 'error_recovery', 'source_error_type': error_type}
        return [result] if result else []

    @staticmethod
    def _lesson_column_not_found(error_text: str, diff_lines: List[str], tables: List[str]) -> Optional[Dict]:
        wrong_col = LessonExtractor._extract_field_ref(error_text)
        right_col = LessonExtractor._extract_added_column(diff_lines)
        if wrong_col and right_col and (wrong_col != right_col):
            table_hint = LessonExtractor._guess_table_from_error(error_text)
            prefix = f'In table {table_hint}: ' if table_hint else ''
            return {'lesson_type': 'column_name_quirk', 'category': 'A', 'lesson': f"{prefix}use '{right_col}' not '{wrong_col}'", 'tables_involved': [table_hint] if table_hint else tables, 'source': 'error_recovery', 'source_error_type': 'COLUMN_NOT_FOUND'}
        if wrong_col:
            return {'lesson_type': 'column_name_quirk', 'category': 'A', 'lesson': f"Column '{wrong_col}' does not exist — check FILE SCHEMAS for correct name", 'tables_involved': tables, 'source': 'error_recovery', 'source_error_type': 'COLUMN_NOT_FOUND'}
        return None

    @staticmethod
    def _lesson_missing_column(error_text: str, tables: List[str]) -> Optional[Dict]:
        col = LessonExtractor._extract_key_error_col(error_text)
        if col:
            return {'lesson_type': 'load_pattern', 'category': 'A', 'lesson': f"Column '{col}' must be explicitly included in columns= parameter when loading", 'tables_involved': tables, 'source': 'error_recovery', 'source_error_type': 'MISSING_COLUMN'}
        return None

    @staticmethod
    def _lesson_type_error(error_text: str, diff_lines: List[str], tables: List[str]) -> Optional[Dict]:
        col = LessonExtractor._extract_problematic_column_from_diff(diff_lines)
        fix = LessonExtractor._extract_type_fix(diff_lines)
        if col and fix:
            return {'lesson_type': 'dtype_quirk', 'category': 'B', 'lesson': f"Column '{col}' has type issues; apply {fix} before operations", 'tables_involved': tables, 'source': 'error_recovery', 'source_error_type': 'ARROW_TYPE_ERROR'}
        col_from_err = LessonExtractor._extract_column_from_type_error(error_text)
        if col_from_err:
            return {'lesson_type': 'dtype_quirk', 'category': 'B', 'lesson': f"Column '{col_from_err}' has mixed types; use pd.to_numeric(errors='coerce') or .astype(str)", 'tables_involved': tables, 'source': 'error_recovery', 'source_error_type': 'ARROW_TYPE_ERROR'}
        return None

    @staticmethod
    def _lesson_zero_row(failed_code: str, fixed_code: str, diff_lines: List[str], tables: List[str]) -> Optional[Dict]:
        wrong_filter = LessonExtractor._extract_filter_column(failed_code)
        right_filter = LessonExtractor._extract_filter_column(fixed_code)
        if wrong_filter and right_filter and (wrong_filter != right_filter):
            return {'lesson_type': 'filter_pattern', 'category': 'D', 'lesson': f"When filtering: use column '{right_filter}' instead of '{wrong_filter}'", 'tables_involved': tables, 'source': 'error_recovery', 'source_error_type': 'ZERO_ROW_RESULT'}
        wrong_val = LessonExtractor._extract_filter_value(failed_code)
        right_val = LessonExtractor._extract_filter_value(fixed_code)
        if wrong_val and right_val and (wrong_val != right_val):
            return {'lesson_type': 'filter_pattern', 'category': 'D', 'lesson': f"Filter value '{wrong_val}' yields 0 rows; use '{right_val}' instead", 'tables_involved': tables, 'source': 'error_recovery', 'source_error_type': 'ZERO_ROW_RESULT'}
        return None

    @staticmethod
    def _lesson_import_error(error_text: str) -> Optional[Dict]:
        pkg = LessonExtractor._extract_package_name(error_text)
        if pkg:
            return {'lesson_type': 'package_issue', 'category': 'H', 'lesson': f"Package '{pkg}' is not available in the execution environment", 'tables_involved': [], 'source': 'error_recovery', 'source_error_type': 'IMPORT_ERROR'}
        return None

    @staticmethod
    def _lesson_file_not_found(error_text: str, tables: List[str]) -> Optional[Dict]:
        return {'lesson_type': 'load_pattern', 'category': 'E', 'lesson': 'Always use EXACT absolute file paths from LOADED DATASETS; never use bare filenames', 'tables_involved': tables, 'source': 'error_recovery', 'source_error_type': 'FILE_NOT_FOUND'}

    @staticmethod
    def _lesson_memory_error(diff_lines: List[str], tables: List[str]) -> Optional[Dict]:
        for line in diff_lines:
            if line.startswith('+') and 'columns=' in line:
                return {'lesson_type': 'load_pattern', 'category': 'E', 'lesson': 'Table is too large; always use columns= parameter to load only needed columns', 'tables_involved': tables, 'source': 'error_recovery', 'source_error_type': 'MEMORY_ERROR'}
        return None

    @staticmethod
    def _lesson_undefined_var(error_text: str, diff_lines: List[str], tables: List[str]) -> Optional[Dict]:
        wrong_var = re.search("name '(\\w+)' is not defined", error_text, re.I)
        right_var = None
        for line in diff_lines:
            if line.startswith('+') and (not line.startswith('+++')):
                names = re.findall('\\b(df\\w*|data\\w*|result\\w*)\\b', line)
                if names:
                    right_var = names[0]
                    break
        if wrong_var and right_var:
            return {'lesson_type': 'column_name_quirk', 'category': 'A', 'lesson': f"Variable '{wrong_var.group(1)}' does not exist; use '{right_var}' instead", 'tables_involved': tables, 'source': 'error_recovery', 'source_error_type': 'UNDEFINED_VARIABLE'}
        return None

    @staticmethod
    def extract_from_data_profile(profile: Dict[str, Any], file_schemas: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        lessons: List[Dict[str, Any]] = []
        if not profile:
            return lessons
        for df_name, info in profile.items():
            table_name = LessonExtractor._df_name_to_table(df_name, file_schemas)
            tables = [table_name] if table_name else []
            columns = info.get('columns', [])
            dtypes = info.get('dtypes', {})
            null_counts = info.get('null_counts', {})
            shape = info.get('shape', [0, 0])
            sample_row = info.get('sample_row', [{}])
            string_values = info.get('string_values', {})
            num_rows = shape[0] if shape else 0
            for col, dtype in dtypes.items():
                if dtype == 'object' and sample_row:
                    sample_val = str(sample_row[0].get(col, '')) if sample_row else ''
                    if sample_val and re.match('^[\\d,.\\-]+$', sample_val) and (len(sample_val) > 0):
                        if not any((col.upper().endswith(s) for s in ('_ID', '_CODE', '_KEY', '_NUM'))):
                            lessons.append({'lesson_type': 'dtype_quirk', 'category': 'B', 'lesson': f"Column '{col}' in {df_name} is stored as string but looks numeric; use pd.to_numeric(errors='coerce')", 'tables_involved': tables, 'source': 'data_profile', 'source_error_type': ''})
            numeric_ranges = info.get('numeric_ranges', {})
            for col, rng in numeric_ranges.items():
                min_val = rng.get('min', 0)
                max_val = rng.get('max', 0)
                if 19000101 <= min_val <= 20301231 and 19000101 <= max_val <= 20301231 and (dtypes.get(col, '') in ('int64', 'int32', 'float64')):
                    lessons.append({'lesson_type': 'dtype_quirk', 'category': 'B', 'lesson': f"Column '{col}' in {df_name} stores dates as integers (YYYYMMDD format); convert with pd.to_datetime(format='%Y%m%d')", 'tables_involved': tables, 'source': 'data_profile', 'source_error_type': ''})
            for col, null_ct in null_counts.items():
                if num_rows > 0:
                    null_pct = null_ct / num_rows * 100
                    if null_pct > 50:
                        lessons.append({'lesson_type': 'dtype_quirk', 'category': 'B', 'lesson': f"Column '{col}' in {df_name} has {null_pct:.0f}% null values; handle with fillna() or dropna()", 'tables_involved': tables, 'source': 'data_profile', 'source_error_type': ''})
            for col, sv in string_values.items():
                top_values = sv.get('top_values', [])
                for val in top_values[:5]:
                    if re.match('^\\d{1,3}(,\\d{2,3})+(\\.\\d+)?$', val):
                        lessons.append({'lesson_type': 'dtype_quirk', 'category': 'B', 'lesson': f"Column '{col}' in {df_name} has comma-formatted numbers (e.g., '{val}'); strip commas with str.replace(',','') before converting", 'tables_involved': tables, 'source': 'data_profile', 'source_error_type': ''})
                        break
            for col, sv in string_values.items():
                unique_count = sv.get('unique_count', 0)
                if 2 <= unique_count <= 6:
                    vals = sv.get('top_values', [])[:6]
                    lessons.append({'lesson_type': 'filter_pattern', 'category': 'D', 'lesson': f"Column '{col}' in {df_name} is categorical with {unique_count} values: {vals}", 'tables_involved': tables, 'source': 'data_profile', 'source_error_type': ''})
            if num_rows > 500000 and len(columns) > 20:
                lessons.append({'lesson_type': 'load_pattern', 'category': 'E', 'lesson': f'Table {df_name} is large ({num_rows:,} rows, {len(columns)} cols); always use columns= parameter to avoid MemoryError', 'tables_involved': tables, 'source': 'data_profile', 'source_error_type': ''})
        return lessons

    @staticmethod
    def extract_from_code_pattern(code: str, file_schemas: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        lessons: List[Dict[str, Any]] = []
        if not code:
            return lessons
        tables = LessonExtractor._infer_tables_from_schemas(file_schemas)
        for m in re.finditer('pd\\.to_datetime\\(\\s*\\w+\\[?[\'\\"](\\w+)[\'\\"]\\]?\\s*,\\s*format=[\'\\"]([^\'\\"]+)[\'\\"]', code):
            col, fmt = (m.group(1), m.group(2))
            lessons.append({'lesson_type': 'dtype_quirk', 'category': 'B', 'lesson': f"Column '{col}' uses date format '{fmt}'; convert with pd.to_datetime(format='{fmt}')", 'tables_involved': tables, 'source': 'code_pattern', 'source_error_type': ''})
        if re.search('str\\.replace\\([\'\\"],[\'\\"]\\s*,\\s*[\'\\"][\'\\"]', code):
            col_match = re.search('\\[?[\'\\"](\\w+)[\'\\"]\\]?\\s*\\.\\s*str\\.replace\\([\'\\"],', code)
            col = col_match.group(1) if col_match else 'unknown'
            lessons.append({'lesson_type': 'dtype_quirk', 'category': 'B', 'lesson': f"Column '{col}' has comma-formatted numbers; strip commas before numeric conversion", 'tables_involved': tables, 'source': 'code_pattern', 'source_error_type': ''})
        for m in re.finditer('pd\\.merge\\([^)]*left_on=[\'\\"](\\w+)[\'\\"][^)]*right_on=[\'\\"](\\w+)[\'\\"]', code):
            left_col, right_col = (m.group(1), m.group(2))
            if left_col != right_col:
                lessons.append({'lesson_type': 'join_pattern', 'category': 'C', 'lesson': f"Join pattern: left_on='{left_col}' maps to right_on='{right_col}' (columns have different names)", 'tables_involved': tables, 'source': 'code_pattern', 'source_error_type': ''})
        for m in re.finditer('#\\s*(?:status|code|type)\\s*[:=]\\s*[\'\\"]?(\\w)[\'\\"]?\\s*(?:=|→|->|means?|is)\\s*[\'\\"]?(\\w[\\w\\s]+)', code, re.I):
            code_val, meaning = (m.group(1), m.group(2).strip())
            lessons.append({'lesson_type': 'filter_pattern', 'category': 'D', 'lesson': f"Status code '{code_val}' means '{meaning}'", 'tables_involved': tables, 'source': 'code_pattern', 'source_error_type': ''})
        if re.search('fiscal|fy\\d{2,4}|april.*march|apr.*mar', code, re.I):
            lessons.append({'lesson_type': 'business_logic', 'category': 'F', 'lesson': 'This company uses April-March fiscal year', 'tables_involved': tables, 'source': 'code_pattern', 'source_error_type': ''})
        return lessons

    @staticmethod
    def _extract_field_ref(error_text: str) -> Optional[str]:
        m = re.search('FieldRef\\.Name\\((\\w+)\\)', error_text)
        if m:
            return m.group(1)
        m = re.search("no match for FieldRef[^']*'(\\w+)'", error_text, re.I)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _extract_key_error_col(error_text: str) -> Optional[str]:
        m = re.search('KeyError:\\s*[\'\\"]([^\'\\"]+)[\'\\"]', error_text)
        return m.group(1) if m else None

    @staticmethod
    def _extract_added_column(diff_lines: List[str]) -> Optional[str]:
        for line in diff_lines:
            if line.startswith('+') and (not line.startswith('+++')):
                m = re.search('[\'\\"](\\w{2,})[\'\\"]', line)
                if m:
                    candidate = m.group(1)
                    if candidate.lower() not in ('parquet', 'csv', 'json', 'records', 'index', 'coerce', 'object', 'float', 'int', 'str'):
                        return candidate
        return None

    @staticmethod
    def _guess_table_from_error(error_text: str) -> Optional[str]:
        m = re.search('[\\\\/]([\\w\\-]+)\\.parquet', error_text)
        if m:
            return m.group(1)
        m = re.search('[\\\\/]([\\w\\-]+)\\.csv', error_text)
        return m.group(1) if m else None

    @staticmethod
    def _extract_problematic_column_from_diff(diff_lines: List[str]) -> Optional[str]:
        for line in diff_lines:
            if line.startswith('+') and (not line.startswith('+++')):
                m = re.search('\\[?[\'\\"](\\w+)[\'\\"]\\]?', line)
                if m:
                    return m.group(1)
        return None

    @staticmethod
    def _extract_type_fix(diff_lines: List[str]) -> Optional[str]:
        for line in diff_lines:
            if line.startswith('+') and (not line.startswith('+++')):
                if 'to_numeric' in line:
                    return "pd.to_numeric(errors='coerce')"
                if 'to_datetime' in line:
                    return 'pd.to_datetime()'
                if '.astype(str)' in line:
                    return '.astype(str)'
                if '.astype(float)' in line:
                    return '.astype(float)'
                if '.astype(int)' in line:
                    return '.astype(int)'
        return None

    @staticmethod
    def _extract_column_from_type_error(error_text: str) -> Optional[str]:
        m = re.search("could not convert.*?'(\\w+)'", error_text, re.I)
        if m:
            return m.group(1)
        m = re.search("column '(\\w+)'", error_text, re.I)
        return m.group(1) if m else None

    @staticmethod
    def _extract_filter_column(code: str) -> Optional[str]:
        m = re.search('\\[?\\w+\\[?\\s*\\[?[\'\\"](\\w+)[\'\\"]', code)
        return m.group(1) if m else None

    @staticmethod
    def _extract_filter_value(code: str) -> Optional[str]:
        m = re.search('==\\s*[\'\\"]([^\'\\"]+)[\'\\"]', code)
        return m.group(1) if m else None

    @staticmethod
    def _extract_package_name(error_text: str) -> Optional[str]:
        m = re.search('No module named [\'\\"]([^\'\\"]+)[\'\\"]', error_text)
        if m:
            return m.group(1).split('.')[0]
        m = re.search('ModuleNotFoundError.*?[\'\\"]([^\'\\"]+)[\'\\"]', error_text)
        return m.group(1).split('.')[0] if m else None

    @staticmethod
    def _summarize_diff(diff_lines: List[str]) -> Optional[str]:
        removed = [l[1:].strip() for l in diff_lines if l.startswith('-') and (not l.startswith('---'))]
        added = [l[1:].strip() for l in diff_lines if l.startswith('+') and (not l.startswith('+++'))]
        if removed and added:
            return f"Changed '{removed[0][:50]}' to '{added[0][:50]}'"
        if added:
            return f'Added: {added[0][:60]}'
        return None

    @staticmethod
    def _infer_tables_from_schemas(file_schemas: Optional[Dict[str, Any]]) -> List[str]:
        if not file_schemas:
            return []
        return [Path(f).stem for f in file_schemas.keys()]

    @staticmethod
    def _df_name_to_table(df_name: str, file_schemas: Optional[Dict[str, Any]]) -> Optional[str]:
        if not file_schemas:
            return df_name
        for fname in file_schemas:
            stem = Path(fname).stem.lower().replace('-', '_')
            if df_name.lower().replace('-', '_') in (stem, f'df_{stem}', f'df{stem}'):
                return Path(fname).stem
        return df_name