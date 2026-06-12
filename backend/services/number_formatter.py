import logging
from typing import Optional, Any
from services.schema_mapper import SchemaMapper
logger = logging.getLogger(__name__)

class NumberFormatter:
    LOCALE_CONFIG = {'indian': {'currency_symbol': '₹', 'units': [('Cr', 10000000), ('L', 100000), ('K', 1000)], 'decimal_places': 2}, 'us': {'currency_symbol': '$', 'units': [('B', 1000000000), ('M', 1000000), ('K', 1000)], 'decimal_places': 2}}

    def __init__(self, client_id: str, db: Any=None):
        self.client_id = client_id
        self.db = db
        self.locale = 'indian'
        self.config = self.LOCALE_CONFIG['indian']

    @classmethod
    async def create(cls, client_id: str, db: Any=None) -> 'NumberFormatter':
        instance = cls(client_id, db)
        instance.locale = await instance._get_client_locale()
        instance.config = cls.LOCALE_CONFIG.get(instance.locale, cls.LOCALE_CONFIG['indian'])
        logger.info(f"NumberFormatter initialized for client '{client_id}' with locale '{instance.locale}'")
        return instance

    async def _get_client_locale(self) -> str:
        try:
            if self.db is not None:
                schema_mapper = await SchemaMapper.create(self.client_id, self.db)
                config = schema_mapper.get_number_format_config()
                fmt = config.get('number_format', 'us')
                return 'us' if fmt == 'standard' else fmt
            else:
                logger.warning(f'No database connection, defaulting to Indian locale')
                return 'indian'
        except Exception as e:
            logger.error(f'Error determining locale: {e}', exc_info=True)
            return 'indian'

    def format_currency(self, value: float, include_symbol: bool=True, decimal_places: Optional[int]=None) -> str:
        if value is None or (isinstance(value, str) and value.strip() == ''):
            return 'N/A'
        try:
            value = float(value)
        except (ValueError, TypeError):
            return str(value)
        is_negative = value < 0
        abs_value = abs(value)
        decimals = decimal_places if decimal_places is not None else self.config['decimal_places']
        formatted_value = None
        for unit_name, unit_value in self.config['units']:
            if abs_value >= unit_value:
                scaled_value = abs_value / unit_value
                formatted_num = f'{scaled_value:.{decimals}f}'.rstrip('0').rstrip('.')
                formatted_value = f'{formatted_num} {unit_name}'
                break
        if formatted_value is None:
            if abs_value >= 1:
                if abs_value == int(abs_value):
                    formatted_value = str(int(abs_value))
                else:
                    formatted_value = f'{abs_value:.{decimals}f}'.rstrip('0').rstrip('.')
            else:
                formatted_value = f'{abs_value:.{decimals}f}'
        if include_symbol:
            symbol = self.config['currency_symbol']
            formatted_value = f'{symbol}{formatted_value}'
        if is_negative:
            formatted_value = f'-{formatted_value}'
        return formatted_value

    def format_quantity(self, value: float, decimal_places: int=0) -> str:
        return self.format_currency(value, include_symbol=False, decimal_places=decimal_places)

    def format_percentage(self, value: float, decimal_places: int=1) -> str:
        try:
            value = float(value)
            if value > 1:
                pct_value = value
            else:
                pct_value = value * 100
            formatted = f'{pct_value:.{decimal_places}f}%'
            return formatted
        except (ValueError, TypeError):
            return str(value)

    def parse_formatted_number(self, formatted_str: str) -> Optional[float]:
        try:
            cleaned = formatted_str.replace('₹', '').replace('$', '').strip()
            parts = cleaned.split()
            if len(parts) == 1:
                return float(parts[0].replace(',', ''))
            numeric_part = float(parts[0].replace(',', ''))
            unit = parts[1]
            for unit_name, unit_value in self.config['units']:
                if unit == unit_name:
                    result = numeric_part * unit_value
                    return round(result, 2) if result >= 1 else result
            return numeric_part
        except Exception as e:
            logger.warning(f"Error parsing formatted number '{formatted_str}': {e}")
            return None

    @staticmethod
    def format_indian_number(value: float, decimal_places: int=2) -> str:
        formatter = NumberFormatter(client_id='indian_default', db=None)
        formatter.locale = 'indian'
        formatter.config = NumberFormatter.LOCALE_CONFIG['indian']
        return formatter.format_currency(value, include_symbol=True, decimal_places=decimal_places)

    @staticmethod
    def format_us_number(value: float, decimal_places: int=2) -> str:
        formatter = NumberFormatter(client_id='us_default', db=None)
        formatter.locale = 'us'
        formatter.config = NumberFormatter.LOCALE_CONFIG['us']
        return formatter.format_currency(value, include_symbol=True, decimal_places=decimal_places)

def format_numbers_in_text(text: str, client_id: str='default', db: Any=None) -> str:
    return text