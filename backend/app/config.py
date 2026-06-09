from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Why: Using Pydantic for settings ensures that missing or invalid environment
    variables fail fast during startup rather than crashing later. This is critical
    for robust backend applications.
    """
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    environment: str = "development"
    
    # AWS Settings (used for Bedrock LLM access)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_default_region: str = "us-east-1"
    
    # Redshift Database Settings
    redshift_host: Optional[str] = None
    redshift_port: int = 5439
    redshift_db: Optional[str] = None
    redshift_user: Optional[str] = None
    redshift_password: Optional[str] = None
    
    # This config tells Pydantic to read variables from a .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Create a global instance of settings to be imported by other modules
settings = Settings()
