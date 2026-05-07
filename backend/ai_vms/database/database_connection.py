"""
Configures database connectivity using SQLAlchemy.

Includes:
- Database engine initialization based on environment configuration.
- Session management with autocommit and autoflush settings.
- Declarative base model setup for ORM.
- Dependency function (`get_db`) for managing database sessions.

Author: HCLTech
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ai_vms.config.config_manager import config

engine = create_engine(config.DB_URL_SYNC)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to get database session
def get_db():
    """
    Provides a database session for request handling.

    This function initializes a new database session using `SessionLocal` and ensures
    it is properly closed after the request is completed.

    Yields:
        Session: A SQLAlchemy database session.

    Usage:
        The function is typically used as a FastAPI dependency for routes 
        that require database access.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
