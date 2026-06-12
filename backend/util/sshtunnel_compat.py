from __future__ import annotations
_compat_applied = False

def ensure_sshtunnel_compat() -> None:
    global _compat_applied
    if _compat_applied:
        return
    import paramiko
    if not hasattr(paramiko, 'DSSKey'):
        paramiko.DSSKey = paramiko.RSAKey
    _compat_applied = True