import logging
import psycopg
from psycopg.rows import dict_row
from typing import List, Dict, Any, Optional
from ..config import settings
logger = logging.getLogger(__name__)

class RedshiftClient:

  def __init__(self):
    self.connection_string = f'host={settings.redshift_host} port={settings.redshift_port} dbname={settings.redshift_db} user={settings.redshift_user} password={settings.redshift_password} sslmode=require'
    logger.info(f'RedshiftClient configured | host={settings.redshift_host} | db={settings.redshift_db}')

  def execute_query(self, sql: str) -> List[Dict[str, Any]]:
    try:
      with psycopg.connect(self.connection_string, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
          logger.info(f'Executing Redshift query | SQL={sql[:200]}...')
          cur.execute(sql)
          rows = cur.fetchall()
          logger.info(f'Query returned {len(rows)} rows')
          return [dict(row) for row in rows]
    except psycopg.OperationalError as e:
      logger.error(f'Redshift connection error: {e}')
      raise RuntimeError(f'Could not connect to Redshift. Check your .env credentials. Error: {e}') from e
    except psycopg.Error as e:
      logger.error(f'Redshift query error: {e}')
      raise RuntimeError(f'SQL execution failed on Redshift: {e}') from e

  def get_schema(self, schema_name: str='public') -> List[Dict[str, Any]]:
    introspection_sql = f"\n      SELECT\n        table_name,\n        column_name,\n        data_type,\n        ordinal_position\n      FROM information_schema.columns\n      WHERE table_schema = '{schema_name}'\n      ORDER BY table_name, ordinal_position;\n    "
    try:
      return self.execute_query(introspection_sql)
    except RuntimeError as e:
      logger.error(f'Schema introspection failed: {e}')
      return []