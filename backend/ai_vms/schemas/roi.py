"""
Schemas for ROI-only operations tied to a Camera.
Rules:
- GET and DELETE: cameraId in query param (handled by router)
- POST (create) and PUT (update): cameraId in request body
Author: HCLTech
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal, Union, Dict, Any



class RoiRect(BaseModel):
    """
    Rectangular ROI with normalized coordinates in [0, 1].
    """
    type: Literal["rect"]
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(ge=0.0, le=1.0)
    h: float = Field(ge=0.0, le=1.0)


class RoiPolygon(BaseModel):
    """
    Polygon ROI as list of [x, y] points (normalized).
    """
    type: Literal["polygon"]
    points: List[List[float]]

    @validator("points")
    def validate_points(cls, pts: List[List[float]]):
        if not pts or len(pts) < 3:
            raise ValueError("Polygon must have at least 3 points")
        for p in pts:
            if not isinstance(p, list) or len(p) != 2:
                raise ValueError("Each point must be [x, y]")
            x, y = p
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("Point coordinates must be in [0.0, 1.0]")
        return pts



# Core shape(s)
RoiShape = Union[RoiRect, RoiPolygon]

# Flexible payload: allow shape, list of shapes, or ANY dict (including empty {})
RoiPayload = Union[RoiShape, List[RoiShape], Dict[str, Any]]



# -------------------------
# Requests
# -------------------------

class RoiCreateRequest(BaseModel):
    """
    Create ROI for a camera. Fails if ROI already exists.
    """
    cameraId: int
    roi: Any = None


class RoiUpdateRequest(BaseModel):
    """
    Update/replace ROI for a camera. Succeeds whether ROI existed or not.
    """
    cameraId: int
    roi: Any = None


# -------------------------
# Responses
# -------------------------

class RoiGetSuccessResponse(BaseModel):
    code: int = 200
    message: str = "ROI fetched successfully"
    roi: Optional[RoiPayload]


class RoiCreateSuccessResponse(BaseModel):
    code: int = 200
    message: str = "ROI created successfully"
    roi: RoiPayload


class RoiUpdateSuccessResponse(BaseModel):
    code: int = 200
    message: str = "ROI updated successfully"
    roi: RoiPayload


class RoiDeleteSuccessResponse(BaseModel):
    code: int = 200
    message: str = "ROI deleted successfully"


class CameraFrameSuccessResponse(BaseModel):
    code: int = 200
    message: str = "Frame fetched successfully"
    frameFile: str  
