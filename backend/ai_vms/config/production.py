"""
Defines the configuration settings for the production environment.

Includes:
- Debug mode enabled.
- Database connection details.
- Secret key for application security.
- Logging level set to DEBUG.
- Host and port configurations.

Author: HCLTech
"""
from pydantic_settings import BaseSettings


class ProductionConfig(BaseSettings):

    AZURE_TENANT_ID: str = "xxxx"
    AZURE_CLIENT_ID: str = "xxxx"
    AZURE_ISSUER: str = "xxxx"
    AZURE_ISSUER: str = "xxxx"
    AZURE_JWKS_URI: str = "xxxx"
    ENV: str = "production"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8010
    DB_URL_SYNC: str = ""