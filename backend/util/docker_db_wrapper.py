import os
import pandas as pd
import logging
logger = logging.getLogger('docker_db_wrapper')
DB_TYPE = os.environ.get('CS_DB_TYPE', 'postgres')
DB_HOST = os.environ.get('CS_DB_HOST')
DB_PORT = os.environ.get('CS_DB_PORT')
DB_NAME = os.environ.get('CS_DB_NAME')
DB_USER = os.environ.get('CS_DB_USER')
DB_PASSWORD = os.environ.get('CS_DB_PASSWORD')
_sql_engine = None
_mongo_client = None

def _get_sql_engine():
    global _sql_engine
    if _sql_engine is not None:
        return _sql_engine
    try:
        from sqlalchemy import create_engine
        if DB_TYPE == 'postgres':
            port = DB_PORT or '5432'
            url = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{port}/{DB_NAME}'
        elif DB_TYPE == 'mysql':
            port = DB_PORT or '3306'
            url = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{port}/{DB_NAME}'
        elif DB_TYPE == 'sqlserver':
            port = DB_PORT or '1433'
            url = f'mssql+pymssql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{port}/{DB_NAME}'
        elif DB_TYPE == 'sap_oracle':
            port = DB_PORT or '1521'
            url = f'oracle+oracledb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{port}/?service_name={DB_NAME}'
        elif DB_TYPE == 'sap_hana':
            port = DB_PORT or '39015'
            url = f'hana://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{port}'
        elif DB_TYPE == 'sap_sybase':
            port = DB_PORT or '5000'
            url = f'sybase+pysybase://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{port}/{DB_NAME}'
        else:
            raise ValueError(f'Unsupported SQL DB Type: {DB_TYPE}')
        _sql_engine = create_engine(url)
        return _sql_engine
    except Exception as e:
        logger.error(f'Failed to create SQL engine: {e}')
        raise

def _get_mongo_db():
    global _mongo_client
    if _mongo_client is not None:
        return _mongo_client[DB_NAME]
    try:
        from pymongo import MongoClient
        port = DB_PORT or '27017'
        if DB_USER and DB_PASSWORD:
            uri = f'mongodb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{port}/{DB_NAME}?authSource=admin'
        else:
            uri = f'mongodb://{DB_HOST}:{port}/{DB_NAME}'
        _mongo_client = MongoClient(uri)
        return _mongo_client[DB_NAME]
    except Exception as e:
        logger.error(f'Failed to create MongoDB client: {e}')
        raise

def query_sql(query: str, params: dict=None) -> pd.DataFrame:
    try:
        engine = _get_sql_engine()
        from sqlalchemy import text
        with engine.connect() as connection:
            if params:
                result = connection.execute(text(query), params)
            else:
                result = connection.execute(text(query))
            return pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception as e:
        logger.error(f'Failed to execute query_sql: {e}')
        raise

def read_sql_query(query: str, params: dict=None) -> pd.DataFrame:
    return query_sql(query, params)

def read_table(table_name: str, schema: str=None) -> pd.DataFrame:
    query = f'SELECT * FROM {table_name}' if not schema else f'SELECT * FROM {schema}.{table_name}'
    return query_sql(query)

def format_date_for_sap(date_str: str) -> str:
    return date_str.replace('-', '')

def find(collection_name: str, filter_query: dict=None, projection: dict=None, sort=None, limit: int=None) -> list:
    try:
        db = _get_mongo_db()
        collection = db[collection_name]
        cursor = collection.find(filter_query or {}, projection)
        if sort:
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)
    except Exception as e:
        logger.error(f'Failed to execute find on {collection_name}: {e}')
        raise

def find_one(collection_name: str, filter_query: dict=None, projection: dict=None) -> dict:
    try:
        db = _get_mongo_db()
        return db[collection_name].find_one(filter_query or {}, projection)
    except Exception as e:
        logger.error(f'Failed to execute find_one on {collection_name}: {e}')
        raise

def aggregate(collection_name: str, pipeline: list) -> list:
    try:
        db = _get_mongo_db()
        return list(db[collection_name].aggregate(pipeline))
    except Exception as e:
        logger.error(f'Failed to execute aggregate on {collection_name}: {e}')
        raise
if __name__ == '__main__':
    if DB_TYPE in ['postgres', 'mysql', 'sqlserver', 'sap_oracle', 'sap_hana', 'sap_sybase']:
        print('Testing SQL connection...')
        try:
            _get_sql_engine().connect().close()
            print('SQL connection successful!')
        except Exception as e:
            print(f'SQL connection failed: {e}')
    else:
        print('Testing MongoDB connection...')
        try:
            _get_mongo_db().command('ping')
            print('MongoDB connection successful!')
        except Exception as e:
            print(f'MongoDB connection failed: {e}')