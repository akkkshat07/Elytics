import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field
from db_config.mongo_server import get_db
from middleware.auth_middleware import AuthMiddleware
from services.subscription_service import get_client_subscription
from util.time_utils import utcnow
logger = logging.getLogger(__name__)
ALLOWED_REQUESTED_PLANS = {'starter', 'pro', 'premium'}
router = APIRouter(tags=['Upgrade Request'])

class RequestUpgradeBody(BaseModel):
    requested_plan: str = Field(..., description='Plan to upgrade to (e.g. starter, pro, premium)')
    current_plan: Optional[str] = Field(None, description='Current plan (optional; derived from subscription if not provided)')
    name: Optional[str] = Field(None, description='Contact name (optional; derived from user profile if not provided)', min_length=1)
    phone: str = Field(..., min_length=5, description='Contact phone number')

@router.post('/request-upgrade', status_code=201)
async def request_upgrade(body: RequestUpgradeBody, current_user: dict=Depends(AuthMiddleware.get_current_user_from_token), db: AsyncIOMotorDatabase=Depends(get_db)):
    requested_plan = (body.requested_plan or '').strip().lower()
    if not requested_plan:
        raise HTTPException(status_code=400, detail='requested_plan is required')
    if requested_plan == 'freemium':
        raise HTTPException(status_code=400, detail='Cannot request upgrade to freemium')
    if requested_plan not in ALLOWED_REQUESTED_PLANS:
        raise HTTPException(status_code=400, detail=f"requested_plan must be one of: {', '.join(sorted(ALLOWED_REQUESTED_PLANS))}")
    client_id = current_user.get('client_id')
    client_email = current_user.get('email')
    if not client_id or not client_email:
        raise HTTPException(status_code=400, detail='Authentication missing client_id or email. Please sign in again.')
    current_plan = (body.current_plan or '').strip().lower() or None
    if not current_plan:
        try:
            subscription = await get_client_subscription(client_id, db)
            current_plan = (subscription.get('plan_name') or 'freemium').lower()
        except Exception as e:
            logger.warning(f'Could not resolve current plan for client {client_id}: {e}')
            current_plan = 'freemium'
    if current_plan and requested_plan == current_plan:
        raise HTTPException(status_code=400, detail='You are already on this plan.')
    raw_name = (body.name or '').strip()
    profile_name = (current_user.get('full_name') or current_user.get('name') or '').strip() if isinstance(current_user, dict) else ''
    if not raw_name:
        raw_name = profile_name
    first_name = ''
    last_name = ''
    if raw_name:
        parts = raw_name.split()
        first_name = parts[0]
        last_name = ' '.join(parts[1:]) if len(parts) > 1 else ''
    requested_at = utcnow()
    doc = {'client_id': client_id, 'client_email': client_email, 'current_plan': current_plan, 'requested_plan': requested_plan, 'status': 'pending', 'created_at': requested_at, 'first_name': first_name, 'last_name': last_name, 'phone': body.phone.strip()}
    try:
        await db.upgrade_requests.update_one({'client_id': client_id}, {'$set': doc}, upsert=True)
    except Exception as e:
        logger.error(f'Failed to insert upgrade request: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail='Failed to save upgrade request')
    try:
        from util.email_sender import send_upgrade_request_notification
        await send_upgrade_request_notification(client_id=client_id, client_email=client_email, current_plan=current_plan, requested_plan=requested_plan, requested_at=requested_at, first_name=first_name, last_name=last_name, phone=body.phone.strip())
    except Exception as e:
        logger.warning(f'Upgrade request email notification failed (request was saved): {e}')
    return {'success': True, 'message': 'Upgrade request submitted.'}