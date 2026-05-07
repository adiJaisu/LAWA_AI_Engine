"""
Defines the configuration settings for the development environment.

Includes:
- Debug mode enabled.
- Database connection details.
- Secret key for application security.
- Logging level set to DEBUG.
- Host and port configurations.

Author: HCLTech
"""
from pydantic_settings import BaseSettings


class DevelopmentConfig(BaseSettings):
    AZURE_TENANT_ID: str = "xxxx"
    AZURE_CLIENT_ID: str = "xxxx"
    AZURE_ISSUER: str = "xxxx"
    AZURE_JWKS_URI: str = "xxxx"

    ENV: str = "development"
    DEBUG: bool = True
    DB_URL_SYNC: str = "postgresql+psycopg2://ai_vms_user:ai_vms_password@localhost:5432/ai_vms_db"
    HOST: str = "0.0.0.0"
    PORT: int = 8010
    
    @classmethod
    def validate(cls):
        if not cls.AZURE_TENANT_ID:
            raise ValueError("AZURE_TENANT_ID is not set")
        if not cls.AZURE_CLIENT_ID:
            raise ValueError("AZURE_CLIENT_ID is not set")


