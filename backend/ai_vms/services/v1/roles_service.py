"""
Service layer for Role management.

Handles role validation, updates, and business logic related to role operations.

Author: HCLTech
"""

from typing import Union
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from ai_vms.schemas.roles import (
    RoleDetailsResponse,
    GetAllRoleErrorResponse,
    GetAllRoleSuccessResponse,
    )
from ai_vms.crud.roles import get_all_roles
from ai_vms.constant.constants import Constants as c
from ai_vms.config.logging_config import LoggingConfig

logger = LoggingConfig().setup_logging()

class RoleService:
    def __init__(self, db: Session):
        self.db = db

    def getRoles(self) -> Union[RoleDetailsResponse, GetAllRoleErrorResponse, GetAllRoleSuccessResponse]:
        """
        Retrieves user roles, excluding the SUPER_ADMIN role.
        
        Returns:
            - RoleDetailsResponse: Success response with user roles.
            - GetAllRoleErrorResponse: Error response if there is a database failure.
            - GetAllRoleSuccessResponse: Success response with user roles.
        """
        try:
            logger.info("Retrieving user roles...")
            roles = get_all_roles(self.db)
            
            # Exclude SUPER_ADMIN
            filtered_roles = [role for role in roles]
            
            if not filtered_roles:
                return GetAllRoleErrorResponse(code=500, message="No assignable roles found in the system")
            
            logger.info("User roles retrieved successfully")
            roles_list = [
                RoleDetailsResponse(
                    roleId=role.id,
                    roleName=role.name
                )
                for role in filtered_roles
            ]
            return GetAllRoleSuccessResponse(code=200, message="User roles retrieved successfully", roleDetails=roles_list)
        
        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            raise GetAllRoleErrorResponse(status_code=500, detail="Something went wrong | Database error")
        
        except Exception as e:
            logger.error(f"Internal Server error: {e}")
            raise GetAllRoleErrorResponse(status_code=500, detail="Something went wrong | Internal Server error")        

