"""
Defines request and response models for user sign-up and login functionality.

Includes request and response models for user management, excluding workstation/location dependencies.

Author: HCLTech
"""

from pydantic import BaseModel, EmailStr
from typing import Optional, List, Any
from datetime import datetime

class SignUpRequest(BaseModel):
    email: str
    password: str
    firstName: str
    lastName: str

class SignUpSuccessResponse(BaseModel):
    code: int = 200
    message: str = "User Added Successfully"

class SignUpErrorResponse(BaseModel):
    code: int = 500
    message: str

class LoginRequest(BaseModel):
    email: str
    password: str

class UserDetailsResponse(BaseModel):
    userId: int
    username: str
    firstName: Optional[str]
    lastName: Optional[str]
    status: int
    lastUpdatedPasswordAt: Optional[Any]

class AllUsersInfoResponse(BaseModel):
    userId: int
    username: str
    firstName: Optional[str]
    lastName: Optional[str]
    roleId: int
    roleName: str
    status: int
    createdAt: datetime
    updatedAt: datetime

class UserInfoResponse(BaseModel):
    userId: int
    username: str
    firstName: Optional[str]
    lastName: Optional[str]
    roleId: int
    roleName: str
    status: int
    createdAt: datetime
    updatedAt: datetime


class GetAllUserSuccessResponse(BaseModel):
    code: int
    message: str
    users: List[AllUsersInfoResponse]

class GetAllUserFailureResponse(BaseModel):
    code: int
    message: str


class AddUserRequest(BaseModel):
    email: str
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    roleId: int
    isActive: Optional[bool] = True

class AddUserSuccessResponse(BaseModel):
    code: int
    message: str

class AddUserFailureResponse(BaseModel):
    code: int
    message: str

class UpdateUserRequest(BaseModel):
    userId: int
    email: EmailStr
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    roleId: int
    isActive: Optional[bool] = None


class UpdateUserSuccessResponse(BaseModel):
    code: int = 200
    message: str = "User updated successfully"
    userId: int


class UpdateUserFailureResponse(BaseModel):
    code: int = 400
    message: str


class GetUserDetailSuccessResponse(BaseModel):
    code: int = 200
    message: str = "User fetched successfully"
    user: UserInfoResponse


class GetUserDetailFailureResponse(BaseModel):
    code: int
    message: str


class DeleteUserSuccessResponse(BaseModel):
    code: int = 200
    message: str = "User deleted successfully"


class DeleteUserFailureResponse(BaseModel):
    code: int
    message: str
