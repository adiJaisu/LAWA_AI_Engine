"""
API routes for Role module.

Exposes endpoints for role retrieval, validation, and update operations.

Author: HCLTech
"""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from ai_vms.database.database_connection import get_db
from ai_vms.config.logging_config import LoggingConfig
from ai_vms.services.v1.roles_service import RoleService
from ai_vms.constant.constants import Constants as c
from ai_vms.utils.auth.auth import require_permission

# Initialize logger for this module
logger = LoggingConfig().setup_logging()

# Create router instance for role endpoints
router = APIRouter()


# Get all roles (requires user read permission)
@router.get("/getallroles", dependencies=[Depends(require_permission("user:read"))])
def get_all_roles(db: Session = Depends(get_db)):
    """
    Handles request to fetch all roles.

    This endpoint retrieves the list of all registered roles from the system.

    Args:
        db (Session): Database session dependency.

    Returns:
        RoleListSuccessResponse | RoleListFailureResponse:
        - Success response with list of roles.
        - Error response if a database issue occurs.
    """
    # Log request entry for traceability
    logger.info("Request Received to Fetch All Roles...")

    # Use role service to fetch roles from database
    role_service = RoleService(db)
    return role_service.getRoles()