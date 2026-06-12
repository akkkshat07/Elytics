import logging
from typing import Dict, Optional
from fastapi import Request
from fastapi.responses import Response
import os
logger = logging.getLogger(__name__)

class SecurityHeadersConfig:

    def __init__(self, environment: str='production'):
        self.environment = environment
        self.is_production = environment == 'production'
        self.frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
        self.csp_directives = self._build_csp_directives()
        self.hsts_max_age = 31536000
        self.hsts_include_subdomains = True
        self.hsts_preload = self.is_production
        self.frame_options = 'DENY'
        self.referrer_policy = 'strict-origin-when-cross-origin'
        self.permissions_policy = self._build_permissions_policy()

    def _build_csp_directives(self) -> Dict[str, str]:
        csp = {'default-src': "'none'", 'script-src': "'none'", 'style-src': "'none'", 'img-src': "'self' data:", 'font-src': "'none'", 'connect-src': f"'self' {self.frontend_url} http://localhost:3000 http://localhost:8024", 'media-src': "'none'", 'object-src': "'none'", 'frame-src': "'none'", 'base-uri': "'self'", 'form-action': "'self'", 'frame-ancestors': "'none'", 'upgrade-insecure-requests': ''}
        return csp

    def _build_permissions_policy(self) -> str:
        policies = ['accelerometer=()', 'ambient-light-sensor=()', 'autoplay=()', 'battery=()', 'camera=()', 'display-capture=()', 'document-domain=()', 'encrypted-media=()', 'fullscreen=()', 'geolocation=()', 'gyroscope=()', 'magnetometer=()', 'microphone=()', 'midi=()', 'payment=()', 'picture-in-picture=()', 'publickey-credentials-get=()', 'sync-xhr=()', 'usb=()', 'wake-lock=()', 'xr-spatial-tracking=()']
        return ', '.join(policies)

    def get_csp_header_value(self) -> str:
        directives = []
        for directive, value in self.csp_directives.items():
            if value:
                directives.append(f'{directive} {value}')
            else:
                directives.append(directive)
        return '; '.join(directives)

    def get_hsts_header_value(self) -> str:
        hsts = f'max-age={self.hsts_max_age}'
        if self.hsts_include_subdomains:
            hsts += '; includeSubDomains'
        if self.hsts_preload:
            hsts += '; preload'
        return hsts

    def get_all_headers(self) -> Dict[str, str]:
        headers = {'Content-Security-Policy': self.get_csp_header_value(), 'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': self.frame_options, 'X-XSS-Protection': '1; mode=block', 'Referrer-Policy': self.referrer_policy, 'Permissions-Policy': self.permissions_policy, 'Cross-Origin-Embedder-Policy': 'require-corp', 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Resource-Policy': 'same-origin', 'Server': 'CoreSight-API', 'Cache-Control': 'no-store, no-cache, must-revalidate, private', 'Pragma': 'no-cache', 'Expires': '0'}
        if self.is_production:
            headers['Strict-Transport-Security'] = self.get_hsts_header_value()
        else:
            logger.debug('HSTS disabled in non-production environment')
        return headers
_config: Optional[SecurityHeadersConfig] = None

def initialize_security_headers(environment: str='production'):
    global _config
    _config = SecurityHeadersConfig(environment=environment)
    logger.info(f'Security headers initialized for environment: {environment}')

def get_security_headers_config() -> SecurityHeadersConfig:
    global _config
    if _config is None:
        environment = os.getenv('ENVIRONMENT', 'production')
        initialize_security_headers(environment)
    return _config

async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    config = get_security_headers_config()
    for header_name, header_value in config.get_all_headers().items():
        response.headers[header_name] = header_value
    if request.url.path.startswith('/api/docs') or request.url.path.startswith('/api/openapi.json') or request.url.path.startswith('/api/redoc'):
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com; frame-ancestors 'self'; base-uri 'self'; form-action 'self'; "
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    logger.debug(f'Applied security headers to {request.method} {request.url.path}')
    return response

def validate_security_headers(response_headers: Dict[str, str]) -> Dict[str, bool]:
    required_headers = ['Content-Security-Policy', 'X-Content-Type-Options', 'X-Frame-Options', 'X-XSS-Protection', 'Referrer-Policy', 'Permissions-Policy', 'Cross-Origin-Embedder-Policy', 'Cross-Origin-Opener-Policy', 'Cross-Origin-Resource-Policy']
    validation_result = {}
    for header in required_headers:
        validation_result[header] = header in response_headers
    return validation_result

def get_security_grade() -> Dict[str, any]:
    config = get_security_headers_config()
    headers = config.get_all_headers()
    score = 0
    max_score = 100
    checks = {'Content-Security-Policy': 25, 'Strict-Transport-Security': 20, 'X-Content-Type-Options': 10, 'X-Frame-Options': 10, 'Referrer-Policy': 10, 'Permissions-Policy': 15, 'Cross-Origin-Embedder-Policy': 5, 'Cross-Origin-Opener-Policy': 5}
    for header, points in checks.items():
        if header in headers:
            score += points
    if score >= 90:
        grade = 'A+'
    elif score >= 80:
        grade = 'A'
    elif score >= 70:
        grade = 'B'
    elif score >= 60:
        grade = 'C'
    elif score >= 50:
        grade = 'D'
    else:
        grade = 'F'
    return {'grade': grade, 'score': score, 'max_score': max_score, 'percentage': score / max_score * 100, 'headers_present': len([h for h in checks if h in headers]), 'headers_missing': len([h for h in checks if h not in headers]), 'missing_headers': [h for h in checks if h not in headers]}