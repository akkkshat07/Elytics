"""
Database Helper Functions for Code Execution

Provides helper functions that can be injected into the executor namespace
to enable database queries when store_in_local=False.
"""

import logging
import pandas as pd
from typing import Optional, Dict, Any
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from util.time_utils import utcnow

logger = logging.getLogger(__name__)


def read_sql_table(
    table_name: str,
    schema: str = "public",
    columns: Optional[list] = None,
    session_factory=None
) -> pd.DataFrame:
    """
    Read a PostgreSQL table into a pandas DataFrame.
    
    This is a synchronous wrapper that works with async SQLAlchemy sessions.
    It should be called from synchronous code execution context.
    
    Args:
        table_name: Name of the table to read
        schema: Schema name (default: 'public')
        columns: Optional list of column names to select
        session_factory: AsyncSession factory from PostgresConnector
        
    Returns:
        pandas DataFrame with table data
        
    Raises:
        RuntimeError: If session_factory is not provided or connection fails
    """
    if session_factory is None:
        raise RuntimeError(
            "Database connection not available. "
            "Please ensure store_in_local=False and database credentials are configured."
        )
    
    try:
        import asyncio
        
        async def _read_table():
            async with session_factory() as session:
                is_mysql = session.bind.dialect.name == "mysql"
                is_hana = session.bind.dialect.name == "hana"
                is_oracle = session.bind.dialect.name == "oracle"        
                actual_schema = schema
                
                if is_mysql and (schema is None or schema == 'public'):
                    db_name_result = await session.execute(text("SELECT DATABASE()"))
                    actual_schema = db_name_result.scalar()
                elif is_hana and (schema is None or schema == 'public'):
                    # HANA uses current schema if not specified and DUMMY is a special built-in table in HANA (with exactly one row) used for selecting system variables, constants, or expressions without needing an actual data table.
                    schema_result = await session.execute(text("SELECT CURRENT_SCHEMA FROM DUMMY"))
                    actual_schema = schema_result.scalar()
                elif is_oracle and (schema is None or schema == 'public'):
                    # Oracle: Get current schema (user) - this will be SAPSR3
                    schema_result = await session.execute(text("SELECT USER FROM DUAL"))
                    actual_schema = schema_result.scalar()

                if is_mysql:
                    # For MySQL, we often don't use schema prefix if it's just the current database
                    # Use backticks for quoting
                    if columns:
                        cols_str = ", ".join([f'`{col}`' for col in columns])
                        query = text(f'SELECT {cols_str} FROM `{table_name}`')
                    else:
                        query = text(f'SELECT * FROM `{table_name}`')
                elif is_hana:
                    # SAP HANA quoting (double quotes)
                    if columns:
                        cols_str = ", ".join([f'"{col}"' for col in columns])
                        query = text(f'SELECT {cols_str} FROM "{actual_schema}"."{table_name}"')
                    else:
                        query = text(f'SELECT * FROM "{actual_schema}"."{table_name}"')
                elif is_oracle:
                    # Oracle quoting (double quotes, schema.table format)
                    if columns:
                        cols_str = ", ".join([f'"{col}"' for col in columns])
                        query = text(f'SELECT {cols_str} FROM "{actual_schema}"."{table_name}"')
                        
                        try:
                            result = await session.execute(query)
                            rows = result.mappings().fetchall()
                        except Exception as e:
                            error_str = str(e)
                            # Check if it's a column not found error (ORA-00904)
                            if "ORA-00904" in error_str or "invalid identifier" in error_str.lower():
                                logger.warning(
                                    f"Column(s) not found in table {actual_schema}.{table_name}. "
                                    f"Requested columns: {columns}. Error: {error_str[:200]}. "
                                    f"Falling back to SELECT * and filtering to available columns."
                                )
                                # Fall back to SELECT * and filter columns in pandas
                                query = text(f'SELECT * FROM "{actual_schema}"."{table_name}"')
                                
                                result = await session.execute(query)
                                rows = result.mappings().fetchall()
                                
                                if rows:
                                    df = pd.DataFrame(rows)
                                    # Normalize Oracle column names to uppercase
                                    df.columns = [col.upper() for col in df.columns]
                                    available_cols = [col.upper() for col in df.columns]
                                    requested_cols_upper = [col.upper() for col in columns]
                                    missing_cols = [col for col in requested_cols_upper if col not in available_cols]
                                    
                                    if missing_cols:
                                        logger.warning(
                                            f"Fallback successful: Returning all {len(df.columns)} available columns "
                                            f"from {actual_schema}.{table_name}. "
                                            f"Requested columns {columns} had {len(missing_cols)} missing: {missing_cols}. "
                                            f"Available columns: {available_cols[:30]}..."
                                        )
                                    else:
                                        logger.info(
                                            f"Fallback successful: Returning all {len(df.columns)} available columns "
                                            f"from {actual_schema}.{table_name}."
                                        )
                                    # Return all columns - don't filter, so LLM can access any column
                                    return df
                                else:
                                    # Empty table - continue to metadata query below
                                    rows = []
                            else:
                                # Re-raise if it's a different error
                                raise
                    else:
                        query = text(f'SELECT * FROM "{actual_schema}"."{table_name}"')
                        
                        result = await session.execute(query)
                        rows = result.mappings().fetchall()
                else:
                    # PostgreSQL/standard quoting
                    if columns:
                        cols_str = ", ".join([f'"{col}"' for col in columns])
                        query = text(f'SELECT {cols_str} FROM "{actual_schema}"."{table_name}"')
                    else:
                        query = text(f'SELECT * FROM "{actual_schema}"."{table_name}"')
                    
                    result = await session.execute(query)
                    rows = result.mappings().fetchall()
                
                if rows:
                    return pd.DataFrame(rows)
                else:
                    # Return empty DataFrame with correct schema if table exists but is empty
                    # We'll need to get column info - use database-specific queries
                    if is_oracle:
                        # Oracle uses ALL_TAB_COLUMNS (information_schema doesn't exist in Oracle)
                        metadata_query = text("""
                            SELECT COLUMN_NAME as column_name, DATA_TYPE as data_type 
                            FROM ALL_TAB_COLUMNS 
                            WHERE OWNER = :schema AND TABLE_NAME = :table_name
                            ORDER BY COLUMN_ID
                        """)
                    else:
                        # PostgreSQL/MySQL use information_schema
                        metadata_query = text("""
                            SELECT column_name, data_type 
                            FROM information_schema.columns 
                            WHERE table_schema = :schema AND table_name = :table_name
                            ORDER BY ordinal_position
                        """)
                    
                    col_result = await session.execute(
                        metadata_query,
                        {"schema": actual_schema, "table_name": table_name}
                    )
                    col_rows = col_result.mappings().fetchall()
                    if col_rows:
                        col_names = [row["column_name"] for row in col_rows]
                        # Normalize Oracle column names to uppercase (SAP convention)
                        if is_oracle:
                            col_names = [col.upper() for col in col_names]
                        return pd.DataFrame(columns=col_names)
                    return pd.DataFrame()
        
        # Run async function in sync context
        # Since we're called from exec() which runs synchronously, we need to handle async execution
        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in a running event loop - use nest_asyncio to allow nested loops
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    return loop.run_until_complete(_read_table())
                except ImportError:
                    # nest_asyncio not available - use run_coroutine_threadsafe
                    import concurrent.futures
                    import threading
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = asyncio.run_coroutine_threadsafe(_read_table(), loop)
                        return future.result(timeout=300)  # 5 minute timeout
            except RuntimeError:
                # No running loop - try to get or create one
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # This shouldn't happen if get_running_loop() failed, but handle it
                        import nest_asyncio
                        try:
                            nest_asyncio.apply()
                            return loop.run_until_complete(_read_table())
                        except ImportError:
                            raise RuntimeError("nest_asyncio is required when running in an async context")
                    else:
                        return loop.run_until_complete(_read_table())
                except RuntimeError:
                    # No event loop exists, create a new one
                    return asyncio.run(_read_table())
        except Exception as e:
            # Re-raise as RuntimeError for outer handler
            raise RuntimeError(f"Failed to execute async database operation: {str(e)}") from e
            
    except Exception as e:
        logger.error(f"Error reading table {schema}.{table_name}: {e}")
        raise RuntimeError(f"Failed to read table {schema}.{table_name}: {str(e)}")


def read_sql_query(
    sql_query: str,
    session_factory=None
) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a pandas DataFrame.
    
    This is a synchronous wrapper that works with async SQLAlchemy sessions.
    It should be called from synchronous code execution context.
    
    Args:
        sql_query: SQL query string to execute
        session_factory: AsyncSession factory from PostgresConnector
        
    Returns:
        pandas DataFrame with query results
        
    Raises:
        RuntimeError: If session_factory is not provided or query fails
    """
    if session_factory is None:
        raise RuntimeError(
            "Database connection not available. "
            "Please ensure store_in_local=False and database credentials are configured."
        )
    
    try:
        import asyncio
        
        async def _execute_query():
            async with session_factory() as session:
                # Detect Oracle dialect 
                is_oracle = session.bind.dialect.name == "oracle"
                
                query = text(sql_query)
                result = await session.execute(query)
                rows = result.mappings().fetchall()
                
                if rows:
                    df = pd.DataFrame(rows)
                    # Normalize Oracle column names to uppercase (matches read_sql_table behavior)
                    if is_oracle:
                        df.columns = [col.upper() for col in df.columns]
                    return df
                else:
                    return pd.DataFrame()
        
        # Run async function in sync context
        # Since we're called from exec() which runs synchronously, we need to handle async execution
        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in a running event loop - use nest_asyncio to allow nested loops
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    return loop.run_until_complete(_execute_query())
                except ImportError:
                    # nest_asyncio not available - use run_coroutine_threadsafe
                    import concurrent.futures
                    import threading
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = asyncio.run_coroutine_threadsafe(_execute_query(), loop)
                        return future.result(timeout=300)  # 5 minute timeout
            except RuntimeError:
                # No running loop - try to get or create one
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # This shouldn't happen if get_running_loop() failed, but handle it
                        import nest_asyncio
                        try:
                            nest_asyncio.apply()
                            return loop.run_until_complete(_execute_query())
                        except ImportError:
                            raise RuntimeError("nest_asyncio is required when running in an async context")
                    else:
                        return loop.run_until_complete(_execute_query())
                except RuntimeError:
                    # No event loop exists, create a new one
                    return asyncio.run(_execute_query())
        except Exception as e:
            # Re-raise as RuntimeError for outer handler
            raise RuntimeError(f"Failed to execute async database operation: {str(e)}") from e
            
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        raise RuntimeError(f"Failed to execute SQL query: {str(e)}")


def format_date_for_sap(date_input) -> int:
    """
    Convert a date to SAP YYYYMMDD numeric format.
    
    SAP systems (Oracle, Sybase, etc.) store dates as numeric in YYYYMMDD format (e.g., 20240101 for Jan 1, 2024).
    This helper converts various date formats to the numeric format SAP expects.
    
    Args:
        date_input: Can be:
            - String in format 'YYYY-MM-DD' (e.g., '2024-01-01')
            - String in format 'YYYYMMDD' (e.g., '20240101') - returns as-is
            - datetime object
            - Already numeric YYYYMMDD (returns as-is)
    
    Returns:
        int: Date in YYYYMMDD format (e.g., 20240101)
    
    Examples:
        format_date_for_sap('2024-01-01')  # Returns: 20240101
        format_date_for_sap('20240101')     # Returns: 20240101
        format_date_for_sap(datetime(2024, 1, 1))  # Returns: 20240101
    """
    # If already numeric, return as-is
    if isinstance(date_input, int):
        return date_input
    
    # If string in YYYYMMDD format, convert to int
    if isinstance(date_input, str) and date_input.isdigit() and len(date_input) == 8:
        return int(date_input)
    
    # Parse date string or datetime object
    if isinstance(date_input, str):
        # Try YYYY-MM-DD format
        try:
            dt = datetime.strptime(date_input, '%Y-%m-%d')
        except ValueError:
            # Try YYYYMMDD format
            try:
                return int(date_input)
            except ValueError:
                raise ValueError(f"Invalid date format: {date_input}. Use 'YYYY-MM-DD' or 'YYYYMMDD'")
    elif isinstance(date_input, datetime):
        dt = date_input
    else:
        raise ValueError(f"Unsupported date type: {type(date_input)}")
    
    # Convert to YYYYMMDD format
    return int(dt.strftime('%Y%m%d'))


def verify_db_connection(session_factory=None) -> Dict[str, Any]:
    """
    Verify that the database connection is active and return connection metadata.
    This function can be used to confirm that queries are hitting the live database.
    
    Args:
        session_factory: AsyncSession factory from PostgresConnector
        
    Returns:
        Dictionary with connection verification info including:
        - connected: bool
        - database_name: str
        - server_version: str
        - current_timestamp: str (from database)
        - connection_time: str (local time when verified)
        
    Raises:
        RuntimeError: If session_factory is not provided or connection fails
    """
    if session_factory is None:
        raise RuntimeError(
            "Database connection not available. "
            "Please ensure store_in_local=False and database credentials are configured."
        )
    
    try:
        import asyncio
        
        async def _verify():
            async with session_factory() as session:
                # Get database name
                is_mysql = session.bind.dialect.name == "mysql"
                is_hana = session.bind.dialect.name == "hana"
                
                if is_mysql:
                    db_name_query = "SELECT database()"
                elif is_hana:
                    db_name_query = "SELECT CURRENT_SCHEMA FROM DUMMY"
                else:
                    db_name_query = "SELECT current_database()"
                
                db_name_result = await session.execute(text(db_name_query))
                db_name = db_name_result.scalar()
                
                # Get version
                if is_hana:
                    version_query = "SELECT VERSION FROM SYS.M_DATABASE"
                else:
                    version_query = "SELECT version()"
                
                version_result = await session.execute(text(version_query))
                version = version_result.scalar()
                
                # Get current timestamp from database (proves it's live)
                if is_hana:
                    timestamp_query = "SELECT CURRENT_TIMESTAMP FROM DUMMY"
                else:
                    timestamp_query = "SELECT NOW()"
                
                timestamp_result = await session.execute(text(timestamp_query))
                db_timestamp = timestamp_result.scalar()
                
                conn_type = "LIVE_POSTGRESQL"
                if is_mysql: conn_type = "LIVE_MYSQL"
                elif is_hana: conn_type = "LIVE_SAPHANA"

                return {
                    "connected": True,
                    "database_name": db_name,
                    "server_version": version,
                    "current_timestamp": str(db_timestamp),
                    "connection_time": utcnow().isoformat(),
                    "connection_type": conn_type
                }
        
        # Run async function in sync context
        try:
            try:
                loop = asyncio.get_running_loop()
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    return loop.run_until_complete(_verify())
                except ImportError:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = asyncio.run_coroutine_threadsafe(_verify(), loop)
                        return future.result(timeout=30)
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import nest_asyncio
                        try:
                            nest_asyncio.apply()
                            return loop.run_until_complete(_verify())
                        except ImportError:
                            raise RuntimeError("nest_asyncio is required when running in an async context")
                    else:
                        return loop.run_until_complete(_verify())
                except RuntimeError:
                    return asyncio.run(_verify())
        except Exception as e:
            raise RuntimeError(f"Failed to verify database connection: {str(e)}") from e
            
    except Exception as e:
        logger.error(f"Error verifying database connection: {e}")
        raise RuntimeError(f"Failed to verify database connection: {str(e)}")
