"""
Defines request and response models for camera configuration and management.

Includes ROI definitions and camera CRUD models, excluding workstation/location dependencies.

Author: HCLTech
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, Any, List, Literal, Union
from datetime import datetime
from ai_vms.models.enums import DecodingResource, DecodingPipeline, DockerMode


class RoiRect(BaseModel):
    type: Literal["rect"]
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(ge=0.0, le=1.0)
    h: float = Field(ge=0.0, le=1.0)


class RoiPolygon(BaseModel):
    type: Literal["polygon"]
    points: List[List[float]]  # [[x,y], [x,y], ...]


Roi = Union[RoiRect, RoiPolygon]


class UsecaseInfoResponse(BaseModel):
    usecaseId: int
    usecaseName: str


# ---- Requests ----

class CameraCreateRequest(BaseModel):
    name: str
    type: Literal["rtsp", "usb"]
    rtspUrl: Optional[str] = None
    resolution: Optional[str] = None
    height: Optional[float] = None
    resolutionWidth: Optional[float] = None
    resolutionHeight: Optional[float] = None
    decodingResource: DecodingResource = DecodingResource.cpu
    decodingPipeline: DecodingPipeline = DecodingPipeline.ffmpeg
    dockerMode: DockerMode = DockerMode.load
    codec: Optional[str] = None
    fps: Optional[int] = None
    usecaseIds: List[int] = Field(default_factory=list)
    roi: Any = None
    status: Optional[int] = 1

    @model_validator(mode="after")
    def validate_camera_fields(self):
        if not self.name.strip():
            raise ValueError("Camera name is required")
        if not self.codec:
            raise ValueError("Codec is required")
        if self.type == "rtsp" and not self.rtspUrl:
            raise ValueError("rtspUrl is required when camera type is 'rtsp'")
        return self


class CameraUpdateRequest(BaseModel):
    cameraId: int
    name: Optional[str] = None
    type: Optional[str] = None
    rtspUrl: Optional[str] = None
    resolution: Optional[str] = None
    height: Optional[float] = None
    resolutionWidth: Optional[float] = None
    resolutionHeight: Optional[float] = None
    decodingResource: Optional[DecodingResource] = None
    decodingPipeline: Optional[DecodingPipeline] = None
    dockerMode: Optional[DockerMode] = None
    codec: Optional[str] = None
    fps: Optional[int] = None
    usecaseIds: Optional[List[int]] = None
    roi: Any = None
    status: Optional[int] = None


# ---- Responses ----

class CameraInfoResponse(BaseModel):
    cameraId: int
    name: str
    type: Optional[str] = None
    rtspUrl: Optional[str]
    resolution: Optional[str]
    height: Optional[float] = None
    resolutionWidth: Optional[float] = None
    resolutionHeight: Optional[float] = None
    decodingResource: DecodingResource = DecodingResource.cpu
    decodingPipeline: DecodingPipeline = DecodingPipeline.ffmpeg
    dockerMode: DockerMode = DockerMode.load
    codec: Optional[str] = None
    fps: Optional[int] = None
    usecases: List[UsecaseInfoResponse] = []
    roi: Optional[Any] = None
    status: int
    is_delete: int
    createdAt: datetime
    updatedAt: datetime


class GetAllCamerasSuccessResponse(BaseModel):
    code: int = 200
    message: str = "Cameras fetched successfully"
    cameras: List[CameraInfoResponse]


class GetCameraDetailSuccessResponse(BaseModel):
    code: int = 200
    message: str = "Camera fetched successfully"
    cameraDetails: CameraInfoResponse


class CreateCameraSuccessResponse(BaseModel):
    code: int = 200
    message: str = "Camera created successfully"
    camera: CameraInfoResponse


class UpdateCameraSuccessResponse(BaseModel):
    code: int = 200
    message: str = "Camera updated successfully"
    camera: CameraInfoResponse


class DeleteCameraSuccessResponse(BaseModel):
    code: int = 200
    message: str = "Camera deleted successfully"


class CommonFailureResponse(BaseModel):
    code: int = 500
    message: str
