from decimal import Decimal, InvalidOperation
import re
from typing import Union, List

def _strip_trailing_zeros(s: str) -> str:
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s

def format_number_indian_system(text: str) -> str:

    def _to_indian_format(num_str: str) -> str:
        num_str = num_str.strip()
        prefix = '₹' if num_str.startswith('₹') else ''
        clean = num_str.replace('₹', '').replace(',', '').strip()
        sign = ''
        if clean.startswith('-'):
            sign = '-'
            clean = clean[1:]
        if '.' in clean:
            int_part, frac_part = clean.split('.', 1)
        else:
            int_part, frac_part = (clean, '')
        if len(int_part) > 3:
            last_three = int_part[-3:]
            remaining = int_part[:-3]
            parts = []
            while len(remaining) > 2:
                parts.insert(0, remaining[-2:])
                remaining = remaining[:-2]
            if remaining:
                parts.insert(0, remaining)
            grouped = ','.join(parts) + ',' + last_three
        else:
            grouped = int_part
        formatted = prefix + sign + grouped
        if frac_part:
            formatted += '.' + frac_part.rstrip('0').rstrip('.')
        return formatted
    pattern = '₹?\\s*\\d{1,3}(?:,\\d{2,3})+(?:\\.\\d+)?'
    return re.sub(pattern, lambda m: _to_indian_format(m.group(0)), text)

def format_indian_currency(value: Union[int, float, str, Decimal], include_inr: bool=True) -> str:
    suffix = ' INR' if include_inr else ''
    try:
        if isinstance(value, str):
            cleaned = value.replace(',', '').replace('₹', '').strip()
            dec = Decimal(cleaned)
        elif isinstance(value, Decimal):
            dec = value
        else:
            dec = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return str(value)
    sign = '-' if dec < 0 else ''
    dec = abs(dec)
    CRORE = Decimal('10000000')
    LAC = Decimal('100000')
    THOUSAND = Decimal('1000')

    def fmt(num: Decimal, denom: Decimal, label: str):
        s = f'{num / denom:.2f}'
        s = _strip_trailing_zeros(s)
        return f'{sign}{s} {label}{suffix}'
    if dec >= CRORE:
        return fmt(dec, CRORE, 'Crore')
    elif dec >= LAC:
        return fmt(dec, LAC, 'Lac')
    elif dec >= THOUSAND:
        return fmt(dec, THOUSAND, 'Thousand')
    else:
        s = f'{dec:.2f}' if dec != dec.to_integral() else str(int(dec))
        s = _strip_trailing_zeros(s)
        return f'{sign}{s}{suffix}'
_NUMBER_PATTERN = re.compile('(₹)?(\\s*)(\\d+(?:,\\d{2,3})*(?:\\.\\d+)?)')
_CURRENCY_KEYWORDS = ['value', 'total', 'amount', 'price', 'cost', 'revenue', 'sales', 'stock', 'stock value', 'balance', 'inr', 'rupee', 'rate']
CODE_DIGIT_LENGTH_THRESHOLD = 13
_PRECISION_COLUMN_KEYWORDS = ['qty', 'quantity', 'rate', 'weight', 'ratio', 'psl_rate', 'onhand_qty']

def _is_currency_context(text: str, start: int, end: int, keywords: List[str]=None) -> bool:
    if keywords is None:
        keywords = _CURRENCY_KEYWORDS
    before = text[max(0, start - 40):start].lower()
    kw_alt = '|'.join((re.escape(k) for k in keywords))
    immediate_pattern = re.compile('(?:' + kw_alt + ')\\b[\\s:\\-]*$', re.IGNORECASE)
    if immediate_pattern.search(before):
        return True
    after_end = min(len(text), end + 20)
    window = text[max(0, start - 80):after_end].lower()
    return any((kw in window for kw in keywords))
_ID_CONTEXT_KEYWORDS = ['requisition', 'group_code', 'item_code', 'item code', 'item', 'code', 'codes', 'sku', 'serial', 'id', 'base code', 'base_code']

def _is_identifier_context(text: str, start: int, end: int, keywords: List[str]=None) -> bool:
    if keywords is None:
        keywords = _ID_CONTEXT_KEYWORDS
    before_window = text[max(0, start - 80):start]
    before = before_window.lower()
    for kw in keywords:
        if kw in before:
            return True
    m = re.search('([A-Z0-9_]{3,})\\b', before_window)
    if m:
        token = m.group(1)
        if '_' in token or token.upper() == token:
            return True
    return False

def _normalize_non_currency_number_str(num_str: str) -> str:
    try:
        n = Decimal(num_str)
    except (InvalidOperation, ValueError):
        return num_str
    if '.' in num_str:
        frac = num_str.split('.')[-1]
        if frac and any((ch != '0' for ch in frac)):
            return _strip_trailing_zeros(f'{n:.2f}')
    if n == n.to_integral():
        return str(int(n))
    return _strip_trailing_zeros(f'{n:.2f}')

def _format_count_number(num: Decimal) -> str:
    if num != num.to_integral():
        return _strip_trailing_zeros(f'{num:.2f}')
    return f'{int(num):,}'

def format_numbers_in_text(text: str, include_inr: bool=False) -> str:

    def repl(m: re.Match) -> str:
        symbol = m.group(1) or ''
        spacing = m.group(2) or ''
        num_str = m.group(3)
        prefix = f'{symbol}{spacing}'
        clean = num_str.replace(',', '')
        has_symbol = bool(symbol)
        int_part = clean.split('.')[0]
        digits_only_int = re.sub('\\D', '', int_part)
        start, end = m.span(0)
        in_currency_context = has_symbol or _is_currency_context(text, start, end)
        if not has_symbol and (not in_currency_context) and (len(digits_only_int) >= CODE_DIGIT_LENGTH_THRESHOLD):
            return prefix + _normalize_non_currency_number_str(clean)
        if _is_identifier_context(text, start, end):
            return prefix + _normalize_non_currency_number_str(clean)
        try:
            number = Decimal(clean)
        except (InvalidOperation, ValueError):
            return m.group(0)
        if has_symbol:
            formatted = format_indian_currency(number, include_inr)
            return f'{prefix}{formatted}'
        return prefix + _format_count_number(number)
    return _NUMBER_PATTERN.sub(repl, text)

def format_dataframe_currency_columns(df_dict: dict, currency_columns: List=None) -> dict:
    if not df_dict or 'data' not in df_dict or 'columns' not in df_dict:
        return df_dict
    columns = list(df_dict.get('columns', []))
    data = df_dict.get('data', [])
    if not columns or not data:
        return df_dict
    if currency_columns is None:
        keywords = ['value', 'amount', 'price', 'cost', 'revenue', 'sales', 'total', 'rate']
        currency_columns = [c for c in columns if any((k in str(c).lower() for k in keywords))]
    ID_KEYS = ['code', 'id', 'item_code', 'item', 'sku', 'serial']
    id_columns = [c for c in columns if any((k in str(c).lower() for k in ID_KEYS))]
    currency_indices = [columns.index(c) for c in currency_columns if c in columns]
    id_indices = [columns.index(c) for c in id_columns if c in columns]
    precision_indices = [i for i, c in enumerate(columns) if any((k in str(c).lower() for k in _PRECISION_COLUMN_KEYWORDS))]
    formatted_data = []
    for row in data:
        r = list(row)
        for idx in id_indices:
            if idx < len(r) and r[idx] is not None:
                try:
                    n = Decimal(str(r[idx]))
                    r[idx] = str(int(n)) if n == n.to_integral() else _strip_trailing_zeros(f'{n:.2f}')
                except Exception:
                    pass
        for idx in currency_indices:
            if idx < len(r) and r[idx] is not None:
                try:
                    n = Decimal(str(r[idx]))
                    if n >= Decimal('100000'):
                        r[idx] = format_indian_currency(n, include_inr=False)
                    elif idx in precision_indices:
                        r[idx] = _normalize_non_currency_number_str(str(r[idx]))
                    else:
                        r[idx] = _normalize_non_currency_number_str(str(r[idx]))
                except Exception:
                    pass
        for idx in precision_indices:
            if idx < len(r) and r[idx] is not None and (idx not in currency_indices):
                try:
                    r[idx] = _normalize_non_currency_number_str(str(r[idx]))
                except Exception:
                    pass
        formatted_data.append(r)
    return {'columns': columns, 'data': formatted_data, 'index': df_dict.get('index')}