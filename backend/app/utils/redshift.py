"""
utils/redshift.py — Amazon Redshift Query Executor

PURPOSE:
    Provides a single helper class that connects to Amazon Redshift and executes
    SQL queries, returning results as a list of dicts (one dict per row).

WHY psycopg2?
    Amazon Redshift is PostgreSQL-compatible. The psycopg2 library is the standard
    PostgreSQL driver for Python. This is the same approach the reference code (db_helpers.py)
    uses — synchronous queries wrapped to work with the rest of the system.

WHY return list[dict] instead of raw rows?
    Returning a list of dicts lets us pass results directly into Pandas with
    pd.DataFrame(rows) in the Analysis Agent without any transformation step.

WINDOWS NOTE:
    Uses psycopg[binary] which bundles the libpq client — no separate PostgreSQL
    installation is needed on Windows. This is why we use it over psycopg2-binary.
"""

import logging
import psycopg
from psycopg.rows import dict_row
from typing import List, Dict, Any, Optional

from ..config import settings

logger = logging.getLogger(__name__)


class RedshiftClient:
    """
    Synchronous Amazon Redshift client using psycopg3 (psycopg).
    
    WHY synchronous and not async?
        LangGraph nodes in Phase 1 are synchronous. We keep this simple.
        In a future phase, this can be upgraded to psycopg's async mode.
    """

    def __init__(self):
        """
        Build the connection string from our config settings.
        
        WHY a connection string instead of separate params?
            psycopg3 accepts a DSN (Data Source Name) string, which is easy to
            construct from our environment variables.
        """
        self.connection_string = (
            f"host={settings.redshift_host} "
            f"port={settings.redshift_port} "
            f"dbname={settings.redshift_db} "
            f"user={settings.redshift_user} "
            f"password={settings.redshift_password} "
            f"sslmode=require"  # Redshift always requires SSL
        )
        logger.info(
            f"RedshiftClient configured | host={settings.redshift_host} "
            f"| db={settings.redshift_db}"
        )

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute a SQL query against Redshift and return results as a list of dicts.

        Args:
            sql: A valid Redshift SQL SELECT statement.

        Returns:
            List of dicts, where each dict = one row. Column names are the keys.
            Returns an empty list if the query returns no rows.

        WHY dict_row?
            psycopg3's dict_row row factory makes each row a dict automatically,
            so we don't have to zip column names manually.

        Example:
            rows = client.execute_query("SELECT customer_id, name FROM customers LIMIT 10")
            # rows = [{"customer_id": 1, "name": "Alice"}, ...]
        """
        try:
            # 'with' ensures the connection is always closed, even on error
            with psycopg.connect(self.connection_string, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    logger.info(f"Executing Redshift query | SQL={sql[:200]}...")
                    cur.execute(sql)
                    rows = cur.fetchall()
                    logger.info(f"Query returned {len(rows)} rows")
                    # Each row is already a dict thanks to dict_row — convert to plain dicts
                    return [dict(row) for row in rows]

        except psycopg.OperationalError as e:
            logger.error(f"Redshift connection error: {e}")
            raise RuntimeError(
                f"Could not connect to Redshift. Check your .env credentials. Error: {e}"
            ) from e
        except psycopg.Error as e:
            logger.error(f"Redshift query error: {e}")
            raise RuntimeError(f"SQL execution failed on Redshift: {e}") from e

    def get_schema(self, schema_name: str = "public") -> List[Dict[str, Any]]:
        """
        Introspect the Redshift schema to discover tables and their columns.

        WHY:
            The Schema Agent needs to know what tables and columns exist in the
            database before it can help the SQL Agent write the correct query.
            This queries Redshift's information_schema — the same system view
            PostgreSQL uses for metadata.

        Args:
            schema_name: The Redshift schema (default "public"). In Redshift,
                         schemas are like namespaces/folders for tables.

        Returns:
            List of dicts, one per column, with keys:
            - table_name, column_name, data_type, ordinal_position
        """
        introspection_sql = f"""
            SELECT
                table_name,
                column_name,
                data_type,
                ordinal_position
            FROM information_schema.columns
            WHERE table_schema = '{schema_name}'
            ORDER BY table_name, ordinal_position;
        """
        try:
            return self.execute_query(introspection_sql)
        except RuntimeError as e:
            logger.error(f"Schema introspection failed: {e}")
            return []
