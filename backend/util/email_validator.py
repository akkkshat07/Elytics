import re
from typing import Optional, Tuple
BLOCKED_DOMAINS = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com', 'icloud.com', 'mail.com', 'protonmail.com', 'yandex.com', 'zoho.com', 'gmx.com', 'live.com', 'msn.com', 'rediffmail.com', 'inbox.com', 'mailinator.com', 'tempmail.com', '10minutemail.com', 'guerrillamail.com', 'throwaway.email'}

def extract_domain(email: str) -> Optional[str]:
    if not email or '@' not in email:
        return None
    try:
        domain = email.split('@')[1].lower().strip()
        return domain
    except (IndexError, AttributeError):
        return None

def is_blocked_domain(domain: str) -> bool:
    return domain in BLOCKED_DOMAINS

def validate_email_domain(user_email: str, organization_domain: str, allow_blocked: bool=False) -> Tuple[bool, Optional[str]]:
    user_domain = extract_domain(user_email)
    if not user_domain:
        return (False, 'Invalid email address format')
    if not allow_blocked and is_blocked_domain(user_domain):
        return (False, f"Generic email domains (like @{user_domain}) are not allowed. Please use an official business email address from your organization's domain.")
    if user_domain != organization_domain.lower():
        return (False, f"Email domain must match your organization's domain. Expected: @{organization_domain}, Got: @{user_domain}")
    return (True, None)