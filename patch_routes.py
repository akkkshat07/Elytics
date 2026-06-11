import sys

patch_code = """
# ==========================================
# PHASE 1 BYPASS MOCKS FOR ELYTICS
# ==========================================
import sys
from unittest.mock import MagicMock

class MockModule(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Mock CoreSight Dependencies that don't exist in Elytics yet (Phase 2)
sys.modules['util'] = MockModule()
sys.modules['util.Mongodb'] = MockModule()
sys.modules['util.cancellation'] = MockModule()
sys.modules['util.time_utils'] = MockModule()
sys.modules['util.metrics'] = MockModule()
sys.modules['util.audit_logger'] = MockModule()
sys.modules['services'] = MockModule()
sys.modules['services.orchestrator_manager'] = MockModule()
sys.modules['domains'] = MockModule()
sys.modules['domains.conversation.service'] = MockModule()
sys.modules['auth'] = MockModule()
sys.modules['auth.auth'] = MockModule()
sys.modules['auth.token_blacklist'] = MockModule()
sys.modules['config'] = MockModule()
sys.modules['config.system_config'] = MockModule()
sys.modules['middleware'] = MockModule()
sys.modules['middleware.auth_middleware'] = MockModule()
sys.modules['middleware.client_middleware'] = MockModule()

def mock_depends():
    return {}

# We also need to override the requires_auth functions so FastAPI doesn't crash on dependency parsing
import middleware.auth_middleware
middleware.auth_middleware.require_auth = MagicMock(return_value=mock_depends)
middleware.auth_middleware.require_auth_flexible = MagicMock(return_value=mock_depends)

# ==========================================

"""

filepath = "/Users/aksha/Desktop/Project/Elytics/backend/app/api/agents_routes.py"
with open(filepath, "r") as f:
    content = f.read()

with open(filepath, "w") as f:
    f.write(patch_code + content)

