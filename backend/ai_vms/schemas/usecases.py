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

class UsecaseDetailsResponse(BaseModel):
    """
    Model for Usecase-specific details.
    Attributes:
        UsecaseId (int): Unique identifier for the usecase.
        UsecaseName (str): Name of the usecase.
    """
    usecaseId: int
    usecaseName: str
    classes: List[str] = []
    tracking: bool = False
    fps: Optional[float] = None
    batch: Optional[int] = None
    frameSkip: Optional[int] = None
    aiResource: str = "cpu"

class GetAllUsecasesSuccessResponse(BaseModel):
    """
    Response model for successful usecases retrieval.
    Attributes:
        code (int): HTTP status code (default: 200).
        message (str): Success message.
        roleDetails (List[usecaseDetailsResponse]): List of usecase details.
    """
    code: int = 200
    message: str = "usecases Details Retrieved Successfully"
    usecaseDetails: List[UsecaseDetailsResponse]

class GetAllUsecaseErrorResponse(BaseModel):
    """
    Response model for failed usecase retrieval.
    Attributes:
        code (int): HTTP status code (default: 500).
        message (str): Error message.
    """
    code: int = 500
    message: str

class UpdateUsecaseApiRequest(BaseModel):
    """
    Request model for updating usecase permissions.
    Attributes:
        roleId (int): Unique identifier for the usecase.
        userId (int): Unique identifier for the usecase.
    """
    usecaseId: int
    cameraId: int

class UpdateUsecaseApiSuccessResponse(BaseModel):
    """
    Response model for successful usecase permission update.
    Attributes:
        code (int): HTTP status code (default: 200).
        message (str): Success message.
    """
    code: int = 200
    message: str = "User usecase Updated Successfully"

class UpdateUsecaseApiErrorResponse(BaseModel):
    """
    Response model for failed usecase permission update.
    Attributes:
        code (int): HTTP status code (default: 500).
        message (str): Error message.
    """
    code: int = 500
    message: str
