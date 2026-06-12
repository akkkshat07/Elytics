import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Request
from jose import jwt, JWTError
import os
logger = logging.getLogger(__name__)
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
ALGORITHM = os.getenv('ALGORITHM', 'HS256')

async def get_client_id_from_token(request: Request) -> str:
    try:
        authorization: str = request.headers.get('Authorization')
        if not authorization:
            logger.error('No authorization header found')
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authorization header is required')
        if not authorization.startswith('Bearer '):
            logger.error('Invalid authorization header format')
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization header format. Expected 'Bearer <token>'")
        token = authorization.split(' ')[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        client_id = payload.get('client_id')
        if not client_id:
            logger.error('JWT token missing client_id field')
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='client_id is REQUIRED in JWT token payload')
        user_id = payload.get('sub') or payload.get('user_id')
        logger.info(f"Extracted client_id '{client_id}' for user '{user_id}'")
        return client_id
    except JWTError as e:
        logger.error(f'JWT decode error: {e}')
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f'Invalid JWT token: {str(e)}')
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error extracting client_id from token: {e}')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Error processing authentication token')

async def get_client_id_from_payload(token_payload: Dict[str, Any]) -> str:
    try:
        client_id = token_payload.get('client_id')
        if not client_id:
            raise ValueError('client_id is REQUIRED in JWT token payload')
        user_id = token_payload.get('sub') or token_payload.get('user_id')
        logger.info(f"Extracted client_id '{client_id}' from payload for user '{user_id}'")
        return client_id
    except Exception as e:
        logger.error(f'Error extracting client_id from payload: {e}')
        raise ValueError(f'client_id is REQUIRED in JWT token payload: {e}')

async def validate_client(client_id: str, db) -> bool:
    try:
        client_config = await db.client_configs.find_one({'client_id': client_id, 'enabled': True})
        if not client_config:
            logger.warning(f"Client '{client_id}' not found or disabled")
            return False
        logger.debug(f"Client '{client_id}' validated successfully")
        return True
    except Exception as e:
        logger.error(f"Error validating client '{client_id}': {e}")
        return False

async def get_client_context(client_id: str, db) -> Optional[Dict[str, Any]]:
    try:
        client_config = await db.client_configs.find_one({'client_id': client_id})
        if not client_config:
            logger.warning(f"Client context not found for '{client_id}'")
            return None
        if not client_config.get('enabled', False):
            logger.warning(f"Client '{client_id}' is disabled")
            return None
        if '_id' in client_config:
            del client_config['_id']
        logger.debug(f"Retrieved context for client '{client_id}'")
        return client_config
    except Exception as e:
        logger.error(f"Error getting client context for '{client_id}': {e}")
        return None

async def validate_client_or_raise(client_id: str, db) -> str:
    try:
        is_valid = await validate_client(client_id, db)
        if not is_valid:
            logger.error(f"Client '{client_id}' validation failed")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Client '{client_id}' not found or disabled")
        return client_id
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating client '{client_id}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Error validating client: {str(e)}')

async def ensure_client_id(client_id: Optional[str]) -> str:
    if not client_id or client_id.strip() == '':
        logger.error('client_id is required but was not provided')
        raise ValueError('client_id is REQUIRED for multi-tenant operation')
    return client_id.strip()

async def get_client_id_dependency(request: Request) -> str:
    return await get_client_id_from_token(request)

def get_client_id_from_user_input(user_input: Dict[str, Any]) -> str:
    try:
        client_id = user_input.get('client_id')
        if not client_id or client_id.strip() == '':
            logger.error('client_id not found in user input')
            raise ValueError('client_id is REQUIRED in user input')
        logger.debug(f"Extracted client_id '{client_id}' from user input")
        return client_id
    except Exception as e:
        logger.error(f'Error extracting client_id from user input: {e}')
        raise ValueError(f'client_id is REQUIRED in user input: {e}')
__all__ = ['get_client_id_from_token', 'get_client_id_from_request', 'get_client_id_from_payload', 'validate_client', 'get_client_context', 'validate_client_or_raise', 'ensure_client_id', 'get_client_id_dependency', 'get_client_id_from_user_input']
get_client_id_from_request = get_client_id_from_token