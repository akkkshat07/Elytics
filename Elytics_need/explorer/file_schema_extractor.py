"""
File-based Schema Extractor
Converts CSV, Excel, Parquet, and ZIP files into database-like schema structures
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import zipfile
import io
from datetime import datetime
from util.time_utils import utcnow

logger = logging.getLogger(__name__)


class FileSchemaExtractor:
    """Extract schema from various file formats."""
    
    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls', '.parquet', '.zip'}
    MAX_SAMPLE_ROWS = 100  # For type inference
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_ext = file_path.suffix.lower()
        
        if self.file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {self.file_ext}. Supported: {self.SUPPORTED_FORMATS}")
    
    def extract_schema(self) -> Dict[str, Any]:
        """
        Extract schema from file and return database-like structure.
        
        Returns:
            {
                'tables': [
                    {
                        'name': 'table_name',
                        'columns': [{'name': 'col1', 'type': 'VARCHAR', 'nullable': True, 'primary_key': False}, ...],
                        'row_count': 1000,
                        'file_source': 'customers.csv'
                    }
                ],
                'total_tables': 1
            }
        """
        if self.file_ext == '.zip':
            return self._extract_from_zip()
        else:
            return self._extract_from_single_file()
    
    def _extract_from_single_file(self) -> Dict[str, Any]:
        """Extract schema from a single CSV/Excel/Parquet file."""
        try:
            logger.info(f"Extracting schema from {self.file_path.name} (type: {self.file_ext})")
            
            tables = []
            
            # Load dataframe based on file type
            if self.file_ext == '.csv':
                try:
                    df = pd.read_csv(self.file_path, nrows=self.MAX_SAMPLE_ROWS)
                except UnicodeDecodeError:
                    logger.warning(f"UTF-8 decode failed for {self.file_path.name}, retrying with latin-1 encoding")
                    df = pd.read_csv(self.file_path, nrows=self.MAX_SAMPLE_ROWS, encoding="latin-1")
                # Extract table name from filename
                table_name = self.file_path.stem
                
                # Build column schema
                columns = self._build_columns_schema(df)
                
                # Try to detect primary key
                primary_key = self._detect_primary_key(df, columns)
                
                table = {
                    'name': table_name,
                    'schema': 'public',  # Default schema for files
                    'columns': columns,
                    'column_count': len(columns),
                    'row_count': len(df),
                    'file_source': self.file_path.name,
                    'primary_key': primary_key,
                    'sheet_name': None  # No sheet for CSV
                }
                tables.append(table)
                
            elif self.file_ext in ['.xlsx', '.xls']:
                # Use appropriate engine for Excel formats
                engine = 'openpyxl' if self.file_ext == '.xlsx' else 'xlrd'
                try:
                    # Read all sheets - sheet_name=None returns a dict of {sheet_name: DataFrame}
                    excel_data = pd.read_excel(self.file_path, sheet_name=None, nrows=self.MAX_SAMPLE_ROWS, engine=engine)
                    
                    # If only one sheet, excel_data might be a DataFrame instead of dict
                    if isinstance(excel_data, pd.DataFrame):
                        excel_data = {None: excel_data}
                    
                    base_table_name = self.file_path.stem
                    
                    # Process each sheet as a separate table
                    for sheet_name, df in excel_data.items():
                        if df.empty:
                            logger.warning(f"Skipping empty sheet '{sheet_name}' in {self.file_path.name}")
                            continue
                        
                        # Create table name: filename_sheetname (or just filename if sheet_name is None)
                        if sheet_name:
                            # Sanitize sheet name for use as table name
                            safe_sheet_name = sheet_name.replace(' ', '_').replace('-', '_')
                            # Remove special characters that might cause issues
                            safe_sheet_name = ''.join(c for c in safe_sheet_name if c.isalnum() or c == '_')
                            table_name = f"{base_table_name}_{safe_sheet_name}"
                        else:
                            table_name = base_table_name
                        
                        # Build column schema
                        columns = self._build_columns_schema(df)
                        
                        # Try to detect primary key
                        primary_key = self._detect_primary_key(df, columns)
                        
                        table = {
                            'name': table_name,
                            'schema': 'public',  # Default schema for files
                            'columns': columns,
                            'column_count': len(columns),
                            'row_count': len(df),
                            'file_source': self.file_path.name,
                            'primary_key': primary_key,
                            'sheet_name': sheet_name
                        }
                        tables.append(table)
                        logger.info(f"Extracted schema from sheet '{sheet_name}' in {self.file_path.name}: {len(columns)} columns, {len(df)} rows")
                        
                except zipfile.BadZipFile:
                    raise ValueError(
                        f"The file '{self.file_path.name}' appears to be corrupted or is not a valid Excel file. "
                        f"Please ensure the file is a proper .xlsx or .xls file and try uploading again."
                    )
            elif self.file_ext == '.parquet':
                df = pd.read_parquet(self.file_path)
                if len(df) > self.MAX_SAMPLE_ROWS:
                    df = df.head(self.MAX_SAMPLE_ROWS)
                
                # Extract table name from filename
                table_name = self.file_path.stem
                
                # Build column schema
                columns = self._build_columns_schema(df)
                
                # Try to detect primary key
                primary_key = self._detect_primary_key(df, columns)
                
                table = {
                    'name': table_name,
                    'schema': 'public',  # Default schema for files
                    'columns': columns,
                    'column_count': len(columns),
                    'row_count': len(df),
                    'file_source': self.file_path.name,
                    'primary_key': primary_key,
                    'sheet_name': None  # No sheet for Parquet
                }
                tables.append(table)
            else:
                raise ValueError(f"Unsupported file extension: {self.file_ext}")
            
            if not tables:
                raise ValueError(f"No valid data found in file '{self.file_path.name}'")
            
            return {
                'tables': tables,
                'total_tables': len(tables),
                'source_type': 'file',
                'extraction_time': utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error extracting schema from {self.file_path}: {e}")
            raise
    
    def _extract_from_zip(self) -> Dict[str, Any]:
        """Extract schema from multiple files inside a ZIP archive."""
        tables = []
        
        try:
            with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    file_ext = Path(file_info.filename).suffix.lower()
                    
                    # Skip directories and unsupported files
                    if file_info.is_dir() or file_ext not in {'.csv', '.xlsx', '.xls', '.parquet'}:
                        continue
                    
                    try:
                        # Read file from ZIP
                        with zip_ref.open(file_info.filename) as file:
                            file_bytes = io.BytesIO(file.read())
                            
                            # Load dataframe based on type
                            if file_ext == '.csv':
                                try:
                                    df = pd.read_csv(file_bytes, nrows=self.MAX_SAMPLE_ROWS)
                                except UnicodeDecodeError:
                                    logger.warning(f"UTF-8 decode failed for {file_info.filename} in ZIP, retrying with latin-1 encoding")
                                    file_bytes.seek(0)
                                    df = pd.read_csv(file_bytes, nrows=self.MAX_SAMPLE_ROWS, encoding="latin-1")
                                # Extract table name
                                table_name = Path(file_info.filename).stem
                                
                                # Build columns
                                columns = self._build_columns_schema(df)
                                primary_key = self._detect_primary_key(df, columns)
                                
                                table = {
                                    'name': table_name,
                                    'schema': 'public',
                                    'columns': columns,
                                    'column_count': len(columns),
                                    'row_count': len(df),
                                    'file_source': file_info.filename,
                                    'primary_key': primary_key,
                                    'sheet_name': None
                                }
                                
                                tables.append(table)
                                logger.info(f"Extracted schema from {file_info.filename}: {len(columns)} columns, {len(df)} rows")
                                
                            elif file_ext in ['.xlsx', '.xls']:
                                engine = 'openpyxl' if file_ext == '.xlsx' else 'xlrd'
                                # Read all sheets - sheet_name=None returns a dict of {sheet_name: DataFrame}
                                excel_data = pd.read_excel(file_bytes, sheet_name=None, nrows=self.MAX_SAMPLE_ROWS, engine=engine)
                                
                                # If only one sheet, excel_data might be a DataFrame instead of dict
                                if isinstance(excel_data, pd.DataFrame):
                                    excel_data = {None: excel_data}
                                
                                base_table_name = Path(file_info.filename).stem
                                
                                # Process each sheet as a separate table
                                for sheet_name, df in excel_data.items():
                                    if df.empty:
                                        logger.warning(f"Skipping empty sheet '{sheet_name}' in {file_info.filename}")
                                        continue
                                    
                                    # Create table name: filename_sheetname (or just filename if sheet_name is None)
                                    if sheet_name:
                                        # Sanitize sheet name for use as table name
                                        safe_sheet_name = sheet_name.replace(' ', '_').replace('-', '_')
                                        # Remove special characters that might cause issues
                                        safe_sheet_name = ''.join(c for c in safe_sheet_name if c.isalnum() or c == '_')
                                        table_name = f"{base_table_name}_{safe_sheet_name}"
                                    else:
                                        table_name = base_table_name
                                    
                                    # Build columns
                                    columns = self._build_columns_schema(df)
                                    primary_key = self._detect_primary_key(df, columns)
                                    
                                    table = {
                                        'name': table_name,
                                        'schema': 'public',
                                        'columns': columns,
                                        'column_count': len(columns),
                                        'row_count': len(df),
                                        'file_source': file_info.filename,
                                        'primary_key': primary_key,
                                        'sheet_name': sheet_name
                                    }
                                    
                                    tables.append(table)
                                    logger.info(f"Extracted schema from sheet '{sheet_name}' in {file_info.filename}: {len(columns)} columns, {len(df)} rows")
                                    
                            elif file_ext == '.parquet':
                                df = pd.read_parquet(file_bytes)
                                if len(df) > self.MAX_SAMPLE_ROWS:
                                    df = df.head(self.MAX_SAMPLE_ROWS)
                                
                                # Extract table name
                                table_name = Path(file_info.filename).stem
                                
                                # Build columns
                                columns = self._build_columns_schema(df)
                                primary_key = self._detect_primary_key(df, columns)
                                
                                table = {
                                    'name': table_name,
                                    'schema': 'public',
                                    'columns': columns,
                                    'column_count': len(columns),
                                    'row_count': len(df),
                                    'file_source': file_info.filename,
                                    'primary_key': primary_key,
                                    'sheet_name': None
                                }
                                
                                tables.append(table)
                                logger.info(f"Extracted schema from {file_info.filename}: {len(columns)} columns, {len(df)} rows")
                            
                    except Exception as e:
                        logger.warning(f"Failed to extract schema from {file_info.filename}: {e}")
                        continue
        
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid ZIP file: {self.file_path}")
            raise ValueError(f"The file '{self.file_path.name}' is not a valid ZIP archive. Please ensure the file is properly compressed.")
        except Exception as e:
            logger.error(f"Error reading ZIP file {self.file_path}: {e}")
            raise
        
        if not tables:
            raise ValueError(f"No valid data files (CSV, Excel, Parquet) found in ZIP archive '{self.file_path.name}'")
        
        return {
            'tables': tables,
            'total_tables': len(tables),
            'source_type': 'zip',
            'extraction_time': utcnow().isoformat()
        }
    
    def _build_columns_schema(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build column schema from dataframe."""
        columns = []
        
        for col_name in df.columns:
            dtype = df[col_name].dtype
            nullable = df[col_name].isnull().any()
            
            # Map pandas dtype to SQL-like type
            sql_type = self._map_dtype_to_sql(dtype)
            
            columns.append({
                'name': col_name,
                'type': sql_type,
                'nullable': bool(nullable),
                'primary_key': False  # Will be set later if detected
            })
        
        return columns
    
    def _map_dtype_to_sql(self, dtype) -> str:
        """Map pandas dtype to SQL-like type string."""
        dtype_str = str(dtype)
        
        if 'int' in dtype_str:
            return 'INTEGER'
        elif 'float' in dtype_str:
            return 'FLOAT'
        elif 'bool' in dtype_str:
            return 'BOOLEAN'
        elif 'datetime' in dtype_str:
            return 'TIMESTAMP'
        elif 'date' in dtype_str:
            return 'DATE'
        elif 'object' in dtype_str:
            return 'VARCHAR'
        else:
            return 'VARCHAR'
    
    def _detect_primary_key(self, df: pd.DataFrame, columns: List[Dict]) -> Optional[str]:
        """
        Attempt to detect primary key by checking for unique, non-null columns.
        Common patterns: 'id', '*_id', columns with 100% unique values.
        """
        for col_dict in columns:
            col_name = col_dict['name']
            
            # Check if column name suggests it's a primary key
            is_id_column = (
                col_name.lower() == 'id' or 
                col_name.lower().endswith('_id') or
                col_name.lower() == 'pk'
            )
            
            # Check uniqueness and non-null
            is_unique = df[col_name].nunique() == len(df)
            is_not_null = not df[col_name].isnull().any()
            
            if is_id_column and is_unique and is_not_null:
                col_dict['primary_key'] = True
                logger.info(f"Detected primary key: {col_name}")
                return col_name
        
        # Fallback: find first column with 100% unique non-null values
        for col_dict in columns:
            col_name = col_dict['name']
            if df[col_name].nunique() == len(df) and not df[col_name].isnull().any():
                col_dict['primary_key'] = True
                logger.info(f"Inferred primary key based on uniqueness: {col_name}")
                return col_name
        
        logger.warning("No primary key detected")
        return None
    
    @staticmethod
    def save_as_parquet(df: pd.DataFrame, output_path: Path) -> Path:
        """Save dataframe as parquet file.
        Coerce object-typed columns to pandas StringDtype to avoid
        pyarrow conversion errors with mixed types (e.g., int + str).
        """
        import pandas as pd
        df_to_save = df.copy()
        for col in df_to_save.columns:
            if pd.api.types.is_object_dtype(df_to_save[col].dtype):
                # Preserve nulls while converting to string
                df_to_save[col] = df_to_save[col].astype("string")
        df_to_save.to_parquet(output_path, index=False)
        logger.info(f"Saved parquet file: {output_path}")
        return output_path
