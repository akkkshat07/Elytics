import asyncio
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path
from typing import Optional
logger = logging.getLogger(__name__)
SMTP_HOST = os.getenv('SMTP_HOST', '').strip()
try:
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587') or '587')
except (TypeError, ValueError):
    SMTP_PORT = 587
SMTP_USER = os.getenv('SMTP_USER', '').strip()
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '').strip()
SUPPORT_EMAIL = os.getenv('SUPPORT_EMAIL', '').strip()
MAIL_FROM = os.getenv('MAIL_FROM', '').strip()
_TEMPLATE_DIR = Path(__file__).resolve().parent / 'email_templates'

def _is_smtp_configured() -> bool:
    return bool(SMTP_HOST and SUPPORT_EMAIL)

def _load_template(template_name: str) -> str:
    path = _TEMPLATE_DIR / template_name
    if not path.exists():
        raise FileNotFoundError(f'Email template not found: {path}')
    return path.read_text(encoding='utf-8')

def _send_email_sync(to: str, subject: str, body: str, html_body: Optional[str]=None, from_addr: Optional[str]=None) -> None:
    if not _is_smtp_configured():
        logger.info('SMTP not configured (SMTP_HOST or SUPPORT_EMAIL missing); skipping email')
        return
    sender = from_addr or MAIL_FROM or SMTP_USER or 'noreply@coresight'
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = to
    msg.attach(MIMEText(body, 'plain'))
    if html_body:
        msg.attach(MIMEText(html_body, 'html'))
    use_tls = SMTP_PORT == 587
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        if use_tls:
            server.starttls()
        if SMTP_USER and SMTP_PASSWORD:
            server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(sender, [to], msg.as_string())
    logger.info(f'Email sent to {to!r}')

async def send_email(to: str, subject: str, body: str, html_body: Optional[str]=None, from_addr: Optional[str]=None) -> None:
    try:
        await asyncio.to_thread(_send_email_sync, to=to, subject=subject, body=body, html_body=html_body, from_addr=from_addr)
    except Exception as e:
        logger.warning(f'Failed to send email to {to!r} (subject: {subject!r}): {e}', exc_info=True)

async def send_upgrade_request_notification(client_id: str, client_email: str, current_plan: str, requested_plan: str, requested_at: datetime, first_name: str='', last_name: str='', phone: str='') -> None:
    subject = f'[CORESight] Upgrade request: {client_id} → {requested_plan}'
    body = f"An upgrade request has been submitted.\n\nContact name: {first_name} {last_name}\nPhone: {phone}\nClient ID: {client_id}\nClient email: {client_email}\nCurrent plan: {current_plan}\nRequested plan: {requested_plan}\nRequested at: {(requested_at.isoformat() if requested_at else 'N/A')}\n"
    await send_email(to=SUPPORT_EMAIL, subject=subject, body=body)

async def send_verification_email(to: str, user_name: str='', verification_url: str='') -> None:
    subject = 'Verify Your Email — CORESight'
    greeting = f', {user_name}' if user_name else ''
    plain_body = f"Verify your email{greeting}!\n\nThanks for signing up for CORESight! Please verify your email address to activate your account.\n\nClick this link to verify (expires in 24 hours):\n{verification_url}\n\nIf you didn't create a CORESight account, you can safely ignore this email.\n\n- The CORESight Team\n"
    html_body = None
    try:
        template = _load_template('verify_email.html')
        current_year = datetime.utcnow().year
        user_greeting = f', {user_name}' if user_name else ''
        html_body = template.replace('{user_greeting}', user_greeting).replace('{verification_url}', verification_url).replace('{year}', str(current_year))
    except Exception as e:
        logger.warning(f'Could not load verify_email template, sending plain text: {e}')
    await send_email(to=to, subject=subject, body=plain_body, html_body=html_body)

async def send_welcome_email(to: str, user_name: str='', created_by_admin: bool=False, creator_email: str='', client_name: str='') -> None:
    subject = "Welcome to CORESight - Let's Turn Data Into Decisions"
    greeting = f', {user_name}' if user_name else ''
    if created_by_admin:
        org = client_name or 'your organization'
        admin_display = creator_email or 'your administrator'
        plain_body = f'Welcome to CORESight{greeting}!\n\nAn administrator at {org} has created a CORESight account for you.\nUse the email address {to} and the password shared with you by {admin_display} to sign in and start exploring your data.\n\nCORESight is an AI-powered decision-support platform that lets you ask questions about your business data in plain English. Specialised AI agents collaborate to plan, code, execute, and synthesise clear, actionable business insights, no SQL required.\n\nGet started: https://coresight.coreops.ai\n\n- The CORESight Team\n'
    else:
        plain_body = f"Welcome to CORESight{greeting}!\n\nYour account has been created successfully. You're all set to start turning your data into actionable insights.\n\nCORESight is an AI-powered decision-support platform that lets you ask questions about your business data in natural language. Specialised AI agents collaborate to plan, code, execute, and synthesise clear, actionable business insights, no SQL required.\n\nGet started: https://coresight.coreops.ai\n\n- The CORESight Team\n"
    html_body = None
    try:
        template = _load_template('welcome.html')
        current_year = datetime.utcnow().year
        user_greeting = f', {user_name}' if user_name else ''
        if created_by_admin:
            org = client_name or 'your organization'
            admin_display = creator_email or 'your administrator'
            intro_paragraph = f'An administrator at {org} has created a CORESight account for you. Use the email address {to} and the password shared with you by {admin_display} to sign in and start exploring your data.'
            footer_reason = f"""You're receiving this email because an administrator ({admin_display}) created a CORESight account for you on <a href="https://coresight.coreops.ai" target="_blank" class="footer-link" style="color:#2563eb; text-decoration:none;">coresight.coreops.ai</a>."""
        else:
            intro_paragraph = "Your account has been created successfully. You're all set to start turning your data into actionable insights."
            footer_reason = 'You\'re receiving this because you created an account on <a href="https://coresight.coreops.ai" target="_blank" class="footer-link" style="color:#2563eb; text-decoration:none;">coresight.coreops.ai</a>.'
        html_body = template.replace('{user_greeting}', user_greeting).replace('{intro_paragraph}', intro_paragraph).replace('{footer_reason}', footer_reason).replace('{year}', str(current_year))
    except Exception as e:
        logger.warning(f'Could not load welcome email template, sending plain text: {e}')
    await send_email(to=to, subject=subject, body=plain_body, html_body=html_body)

async def send_password_reset_email(to: str, reset_url: str, user_name: str='') -> None:
    subject = 'Reset Your Password - CORESight'
    greeting = f', {user_name}' if user_name else ''
    plain_body = f'Hi{greeting},\n\nWe received a request to reset your CORESight password.\n\nReset your password using this link (expires in 15 minutes):\n{reset_url}\n\nIf you did not request a password reset, you can safely ignore this email.\n\n- The CORESight Team\n'
    html_body = None
    try:
        template = _load_template('reset_password.html')
        current_year = datetime.utcnow().year
        user_greeting = f', {user_name}' if user_name else ''
        html_body = template.replace('{user_greeting}', user_greeting).replace('{reset_url}', reset_url).replace('{year}', str(current_year))
    except Exception as e:
        logger.warning(f'Could not load reset password template, sending plain text: {e}')
    await send_email(to=to, subject=subject, body=plain_body, html_body=html_body)