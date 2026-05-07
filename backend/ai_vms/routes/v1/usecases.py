"""
Defines API routes for usecases management.

Includes:
- Usecases List API route
- Database integration using SQLAlchemy
- Request logging

Author: HCLTech
"""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from ai_vms.database.database_connection import get_db
from ai_vms.config.logging_config import LoggingConfig
from ai_vms.services.v1.usecase_service import UsecaseService
from ai_vms.constant.constants import Constants as c
from ai_vms.utils.auth.auth import require_permission

# Initialize logger for request tracing
logger = LoggingConfig().setup_logging()

# Create router instance for usecase endpoints
router = APIRouter()


# Get all usecases (requires user read permission)
@router.get("/getallusecases", dependencies=[Depends(require_permission("user:read"))])
def get_all_usecases(db: Session = Depends(get_db)):
    """
    Handles request to fetch all usecases.

    This endpoint retrieves the list of all registered usecases from the system.

    Args:
        db (Session): Database session dependency.

    Returns:
        UsecaseListSuccessResponse | UsecaseListFailureResponse:
        - Success response with list of usecases.
        - Error response if a database issue occurs.
    """
    # Log the request for audit and debugging
    logger.info("Request Received to Fetch All Usecases...")

    # Use the service layer to fetch usecases
    usecase_service = UsecaseService(db)
    return usecase_service.getusecases()