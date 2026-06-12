from dotenv import load_dotenv
import os
import uvicorn
import shutil
from pathlib import Path

def _patch_jupyter_mcp_server():
    try:
        import importlib.util
        spec = importlib.util.find_spec('jupyter_mcp_server')
        if spec and spec.origin:
            server_file = os.path.join(os.path.dirname(spec.origin), 'server.py')
            if os.path.exists(server_file):
                with open(server_file, 'r') as f:
                    content = f.read()
                if 'le=60' in content:
                    content = content.replace('le=60', 'le=3000')
                    with open(server_file, 'w') as f:
                        f.write(content)
                    print('Successfully patched jupyter_mcp_server execution timeout limit.')
    except Exception as e:
        print(f'Warning: Failed to patch jupyter_mcp_server timeout limit: {e}')
load_dotenv()
HOST = os.getenv('BACKEND_HOST', '0.0.0.0')
PORT = int(os.getenv('BACKEND_PORT', '8024'))
RELOAD = os.getenv('UVICORN_RELOAD', 'true').lower() in ('1', 'true', 'yes', 'y')
if __name__ == '__main__':
    _patch_jupyter_mcp_server()
    uvicorn.run('main:app', host=HOST, port=PORT, reload=RELOAD)