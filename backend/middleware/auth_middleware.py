def require_auth(*args, **kwargs):
    async def _dependency():
        return {'email': 'admin@elytics.com', 'is_admin': True, 'user_id': 'admin_123', 'client_id': 'default_client', 'role': 'super_admin'}
    return _dependency

def require_auth_flexible(*args, **kwargs):
    async def _dependency():
        return {'email': 'admin@elytics.com', 'is_admin': True, 'user_id': 'admin_123', 'client_id': 'default_client', 'role': 'super_admin'}
    return _dependency
def require_admin(*args, **kwargs):
    async def _dependency():
        return {'email': 'admin@elytics.com', 'is_admin': True, 'user_id': 'admin_123', 'client_id': 'default_client', 'role': 'super_admin'}
    return _dependency
