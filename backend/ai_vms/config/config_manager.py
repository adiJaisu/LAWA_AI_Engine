"""
Loads the application configuration based on the environment.

Includes:
- Fetching the environment variable (`APP_ENV`).
- Loading the corresponding configuration (Development, Testing, or Production).

Author: HCLTech
"""
import os
from .development import DevelopmentConfig
from .production import ProductionConfig
from .testing import TestingConfig

# Get environment from system env variable (default to development)
env = os.getenv("APP_ENV", "development").lower()

# Load corresponding config
if env == "production":
    config = ProductionConfig()
elif env == "testing":
    config = TestingConfig()
else:
    config = DevelopmentConfig()
