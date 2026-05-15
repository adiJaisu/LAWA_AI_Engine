"""
Defines request and response models for role-based permissions.

Includes:
- Role creation and retrieval.
- Permission management.
- Access control mapping.
- Data validation using Pydantic.

Author: HCLTech
"""
from pydantic import BaseModel
from typing import Optional, List

class RoleDetailsResponse(BaseModel):
    """
    Model for role-specific details.
    Attributes:
        roleId (int): Unique identifier for the role.
        roleName (str): Name of the role.
    """
    roleId: int
    roleName: str

class GetAllRoleSuccessResponse(BaseModel):
    """
    Response model for successful role retrieval.
    Attributes:
        code (int): HTTP status code (default: 200).
        message (str): Success message.
        roleDetails (List[RoleDetailsResponse]): List of role details.
    """
    code: int = 200
    message: str = "Role Details Retrieved Successfully"
    roleDetails: List[RoleDetailsResponse]

class GetAllRoleErrorResponse(BaseModel):
    """
    Response model for failed role retrieval.
    Attributes:
        code (int): HTTP status code (default: 500).
        message (str): Error message.
    """
    code: int = 500
    message: str

class UpdateRoleApiRequest(BaseModel):
    """
    Request model for updating role permissions.
    Attributes:
        roleId (int): Unique identifier for the role.
        userId (int): Unique identifier for the user.
    """
    roleId: int
    userId: int

class UpdateRoleApiSuccessResponse(BaseModel):
    """
    Response model for successful role permission update.
    Attributes:
        code (int): HTTP status code (default: 200).
        message (str): Success message.
    """
    code: int = 200
    message: str = "User Role Updated Successfully"

class UpdateRoleApiErrorResponse(BaseModel):
    """
    Response model for failed role permission update.
    Attributes:
        code (int): HTTP status code (default: 500).
        message (str): Error message.
    """
    code: int = 500
    message: str

