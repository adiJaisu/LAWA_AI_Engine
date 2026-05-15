"""
Defines the configuration settings for the testing environment.

Includes:
- Debug mode enabled.
- Database connection details.
- Secret key for application security.
- Logging level set to DEBUG.
- Host and port configurations.

Author: HCLTech
"""
from pydantic_settings import BaseSettings
import secrets

class TestingConfig(BaseSettings):
    ENV: str = "testing"
    DEBUG: bool = True
    DATABASE_URL: str = ""
    HOST: str = "0.0.0.0"
    PORT: int = 8010
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1400
    SECRET_KEY: str = "e5a1c3b8d749f7b44c9a5d6e1f7c2a8b3d5e6f8a1c2b3d4e5f6a7b8c9d0e1f2a"

    