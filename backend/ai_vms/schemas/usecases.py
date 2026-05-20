from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from ai_vms.models.enums import AIResource


class UsecaseCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    tracking: bool = False
    fps: Optional[float] = None
    batch: Optional[int] = None
    frame_skip: Optional[int] = None
    ai_resource: AIResource = AIResource.cpu
    is_active: bool = True


class UsecaseUpdateRequest(BaseModel):
    usecase_id: int

    name: Optional[str] = None
    description: Optional[str] = None
    tracking: Optional[bool] = None
    fps: Optional[float] = None
    batch: Optional[int] = None
    frame_skip: Optional[int] = None
    ai_resource: Optional[AIResource] = None
    is_active: Optional[bool] = None


# class UsecaseResponse(BaseModel):
#     id: int
#     name: str
#     description: Optional[str]
#     tracking: bool
#     fps: Optional[float]
#     batch: Optional[int]
#     frame_skip: Optional[int]
#     ai_resource: AIResource
#     is_active: bool
#     created_at: datetime
#     updated_at: datetime

#     class Config:
#         from_attributes = True




# class GetAllUsecasesResponse(BaseModel):
#     code: int
#     message: str
#     data: List[UsecaseResponse]


class UsecaseResponse(BaseModel):
    usecaseId: int
    usecaseName: str
    description: Optional[str]
    classes: List[str]=[]
    tracking: bool
    fps: Optional[float]
    batch: Optional[int]
    frameSkip: Optional[int]
    aiResource: AIResource
    is_active: bool
    created_at: datetime
    updated_at: datetime
 
    class Config:
        from_attributes = True
 
 
 
 
class GetAllUsecasesResponse(BaseModel):
    code: int
    message: str
    usecaseDetails: List[UsecaseResponse]