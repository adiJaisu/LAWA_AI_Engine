"""
Defines API routes for user authentication and account management.

Includes:
- User sign-up functionality.
- User login functionality
- Database integration using SQLAlchemy.
- Request logging.

Author: HCLTech
"""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from ai_vms.database.database_connection import get_db
from ai_vms.schemas.users import AddUserRequest, UpdateUserRequest
from ai_vms.config.logging_config import LoggingConfig
from ai_vms.services.v1.users_service import UserService
from ai_vms.constant.constants import Constants as c
from ai_vms.utils.auth.auth import require_permission, get_current_user
# Initialize logger for request tracing
logger = LoggingConfig().setup_logging()

# Create router instance for user endpoints
router = APIRouter()


# Get all users (requires user read permission)
@router.get("/getallusers", dependencies=[Depends(require_permission("user:read"))])
def get_all_users(db: Session = Depends(get_db)):
    """
    Handles request to fetch all users.

    This endpoint retrieves the list of all registered users from the system.

    Args:
        db (Session): Database session dependency.

    Returns:
        UserListSuccessResponse | UserListFailureResponse:
        - Success response with list of users.
        - Error response if a database issue occurs.
    """
    # Log the request for audit and debugging
    logger.info("Request Received to Fetch All Users...")

    # Use the service layer to fetch all users
    user_service = UserService(db)
    return user_service.get_all_users()


# Add a new user (requires user create permission)
@router.post("/adduser")
def add_user(
    payload: AddUserRequest,
    db: Session = Depends(get_db),
    current=Depends(require_permission("user:create"))
):
    """
    Handles request to add a new user.

    This endpoint creates a new user entry in the Users table.
    It validates that the provided email is unique and the given roleId is valid
    before inserting the record.

    Args:
        payload (AddUserRequest): Request body containing user details such as
            email, firstName, lastName, roleId, and isActive.
        db (Session): Database session dependency.

    Returns:
        AddUserSuccessResponse | AddUserFailureResponse:
        - Success response with created userId if insertion is successful.
        - Failure response if validation fails (duplicate email/invalid role)
          or a database error occurs.
    """
    # Log the request for audit and debugging
    logger.info("Request Received to Add User...")

    # Use the service layer to create the user
    user_service = UserService(db)
    return user_service.add_user(payload, current)


# Update an existing user (requires user update permission)
@router.put("/updateuser")
def update_user(
    payload: UpdateUserRequest,
    db: Session = Depends(get_db),
    current=Depends(require_permission("user:update"))
):
    """
    Handles request to update an existing user.

    This endpoint updates an existing user record in the Users table based on the
    provided userId. It also validates that the updated email (if changed) remains
    unique and the provided roleId is valid.

    Args:
        payload (UpdateUserRequest): Request body containing userId and user details
            to be updated such as email, firstName, lastName, roleId, and isActive.
        db (Session): Database session dependency.

    Returns:
        UpdateUserSuccessResponse | UpdateUserFailureResponse:
        - Success response with userId if the update is successful.
        - Failure response if userId is not found, validation fails (duplicate email/invalid role),
          or a database error occurs.
    """
    # Log the request for audit and debugging
    logger.info("Request Received to Update User...")

    # Use the service layer to update the user
    user_service = UserService(db)
    return user_service.update_user(payload, current)


# Get user details by userId passed as query parameter (requires user read permission)
@router.get("/getuserdetail/{userId}", dependencies=[Depends(require_permission("user:read"))])
def get_user_detail(userId: int, db: Session = Depends(get_db)):
    """
    Fetch single user details based on userId.
    """
    

    # Log the request with userId for traceability
    logger.info(f"Request Received to Fetch User Detail for userId={userId}...")

    # Use the service layer to fetch user detail
    user_service = UserService(db)
    return user_service.get_user_detail(int(userId))


# Soft delete a user by userId passed as query parameter (requires user delete permission)
@router.delete("/deleteuser/{userId}")
def delete_user(
    userId: int, 
    db: Session = Depends(get_db),
    current=Depends(require_permission("user:delete")),
):
    """
    Soft delete user based on userId (sets is_active = 0).
    """

    # Log the request with userId for traceability
    logger.info(f"Request Received to Delete User userId={userId}...")

    # Use the service layer to soft delete the user
    user_service = UserService(db)
    return user_service.delete_user(userId, current)
