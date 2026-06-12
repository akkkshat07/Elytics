import re
from typing import List, Optional

class BaseSQLValidator:

    def validate(self, sql: str) -> List[str]:
        return []

class MSSQLValidator(BaseSQLValidator):
    RESERVED_ALIASES = {'RowCount', 'User', 'Table', 'Index', 'Order', 'Group', 'Select', 'From', 'Where'}

    def validate(self, sql: str) -> List[str]:
        errors = []
        for reserved in self.RESERVED_ALIASES:
            pattern = f'\\bAS\\s+{reserved}\\b|\\s+{reserved}\\s*[,)]'
            if re.search(pattern, sql, re.IGNORECASE):
                errors.append(f"POTENTIAL SYNTAX ERROR: '{reserved}' is a reserved word or metadata field in MSSQL. Do NOT use '{reserved}' as an alias. Use something else like '{reserved}_val' or '[{reserved}]'.")
        if re.search('\\bLIMIT\\s+\\d+', sql, re.IGNORECASE) and (not re.search('OFFSET', sql, re.IGNORECASE)):
            errors.append("POTENTIAL SYNTAX ERROR: MSSQL uses 'SELECT TOP N' instead of 'LIMIT N'. Update your query to use TOP or OFFSET/FETCH NEXT.")
        if 'GROUP BY' in sql.upper() and ('YEAR(' in sql.upper() or 'MONTH(' in sql.upper()):
            if 'DATEFROMPARTS' not in sql.upper():
                pass
        return errors

class SQLValidatorFactory:

    @staticmethod
    def get_validator(db_type: str) -> BaseSQLValidator:
        db_type = db_type.lower()
        if db_type in ('sqlserver', 'mssql'):
            return MSSQLValidator()
        return BaseSQLValidator()