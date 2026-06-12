import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)
DATABASE_URL = os.getenv('DATABASE_URL', 'mongodb://localhost:27017')
DB_NAME = os.getenv('DATABASE_NAME', 'core-sight')
_mongo_client = None

def start_mongodb_server():
    global _mongo_client
    if _mongo_client is None:
        try:
            logger.info(f'Connecting to real MongoDB at {DATABASE_URL}...')
            _mongo_client = AsyncIOMotorClient(DATABASE_URL, maxPoolSize=50, minPoolSize=10, maxIdleTimeMS=60000, serverSelectionTimeoutMS=2000, connectTimeoutMS=5000, socketTimeoutMS=30000)
            logger.info(f'✅ Connected to MongoDB successfully')
            logger.info(f'   Database: {DB_NAME}')
            return _mongo_client
        except Exception as e:
            logger.error(f'❌ Failed to connect to MongoDB: {e}')
            return None
    return _mongo_client

def stop_mongodb_server():
    global _mongo_client
    if _mongo_client is not None:
        try:
            _mongo_client.close()
            _mongo_client = None
            logger.info(' MongoDB connection closed')
        except Exception as e:
            logger.error(f'❌ Failed to close MongoDB connection: {e}')

async def get_db():
    client = start_mongodb_server()
    if client is None:
        raise Exception('Failed to connect to database')
    return client[DB_NAME]

async def test_mongodb_connection():
    client = start_mongodb_server()
    if client is None:
        logger.error('❌ MongoDB client is None')
        return False
    try:
        db = client[DB_NAME]
        await db.command('ping')
        collection = db['test_collection']
        await collection.insert_one({'test': 'connection'})
        result = await collection.find_one({'test': 'connection'})
        if result and result.get('test') == 'connection':
            logger.info('✅ MongoDB connection and operations test successful')
            return True
        else:
            logger.error('❌ MongoDB operations test failed')
            return False
    except Exception as e:
        logger.error(f'❌ MongoDB connection test failed: {e}')
        return False

async def run_comprehensive_test():
    client = start_mongodb_server()
    if client is None:
        logger.error('❌ MongoDB client is None')
        return False
    try:
        db = client[DB_NAME]
        collection = db['test_collection']
        await collection.delete_many({})
        docs = [{'name': 'User 1', 'email': 'user1@example.com', 'age': 25}, {'name': 'User 2', 'email': 'user2@example.com', 'age': 30}, {'name': 'User 3', 'email': 'user3@example.com', 'age': 35}]
        result = await collection.insert_many(docs)
        logger.info(f'Inserted {len(result.inserted_ids)} documents')
        cursor = collection.find({'age': {'$gt': 25}})
        documents = await cursor.to_list(length=10)
        logger.info(f'Found {len(documents)} documents with age > 25')
        update_result = await collection.update_many({'age': {'$gt': 30}}, {'$set': {'status': 'senior'}})
        logger.info(f'Updated {update_result.modified_count} documents')
        count = await collection.count_documents({'age': {'$gt': 25}})
        logger.info(f'Count of documents with age > 25: {count}')
        delete_result = await collection.delete_many({'age': {'$lt': 30}})
        logger.info(f'Deleted {delete_result.deleted_count} documents')
        logger.info('✅ Comprehensive MongoDB test completed successfully')
        return True
    except Exception as e:
        logger.error(f'❌ Comprehensive MongoDB test failed: {e}')
        return False
if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    client = start_mongodb_server()
    if client is not None:
        success = loop.run_until_complete(test_mongodb_connection())
        if success:
            loop.run_until_complete(run_comprehensive_test())
        stop_mongodb_server()