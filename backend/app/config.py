from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    host: str = '0.0.0.0'
    port: int = 8000
    environment: str = 'development'
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_default_region: str = 'us-east-1'
    redshift_host: Optional[str] = None
    redshift_port: int = 5439
    redshift_db: Optional[str] = None
    redshift_user: Optional[str] = None
    redshift_password: Optional[str] = None
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
settings = Settings()