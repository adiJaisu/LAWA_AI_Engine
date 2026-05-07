from typing import Union
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ai_vms.database.database_connection import get_db
from ai_vms.services.v1.roi_service import ROIService
from ai_vms.utils.auth.auth import require_permission
from ai_vms.schemas.cameras import CommonFailureResponse
from ai_vms.schemas.roi import (
    RoiCreateRequest,RoiUpdateRequest,RoiGetSuccessResponse,
    RoiCreateSuccessResponse,RoiUpdateSuccessResponse,RoiDeleteSuccessResponse,
    CameraFrameSuccessResponse
)

router = APIRouter()


#get roi route
@router.get(
    "/getroi",
    dependencies=[Depends(require_permission("camera:read"))],
    response_model=Union[RoiGetSuccessResponse, CommonFailureResponse],
)
def get_roi(
    cameraId: int = Query(..., description="Camera ID"),
    db: Session = Depends(get_db),
):
    """
    Fetch the ROI JSON for a camera. cameraId is provided as a query parameter.
    """
    service = ROIService(db)
    return service.get_camera_roi(camera_id=cameraId)


#create roi route
@router.post(
    "/createroi",
    response_model=Union[RoiCreateSuccessResponse, CommonFailureResponse],
)
def create_roi(
    payload: RoiCreateRequest,
    db: Session = Depends(get_db),
    current_user=Depends(require_permission("camera:create")),
):
    """
    Create ROI for a camera. cameraId and roi are provided in the request body.
    Fails with 409 if ROI already exists for the camera.
    """
    service = ROIService(db)
    return service.create_camera_roi(
        camera_id=payload.cameraId,
        roi_payload=payload.roi,
        actor_id=current_user["user"].id,
    )


#update roi route
@router.put(
    "/updateroi",
    response_model=Union[RoiUpdateSuccessResponse, CommonFailureResponse],
)
def update_roi(
    payload: RoiUpdateRequest,
    db: Session = Depends(get_db),
    current_user=Depends(require_permission("camera:update")),
):
    """
    Update/replace ROI for a camera. cameraId and roi are provided in the request body.
    """
    service = ROIService(db)
    return service.update_camera_roi(
        camera_id=payload.cameraId,
        roi_payload=payload.roi,
        actor_id=current_user["user"].id,
    )


#delete roi route
@router.delete(
    "/deleteroi",
    response_model=Union[RoiDeleteSuccessResponse, CommonFailureResponse],
)
def delete_roi(
    cameraId: int = Query(..., description="Camera ID"),
    db: Session = Depends(get_db),
    current_user=Depends(require_permission("camera:delete")),
):
    """
    Delete (clear to null) the ROI for a camera. cameraId is provided as a query parameter.
    """
    service = ROIService(db)
    return service.delete_camera_roi(
        camera_id=cameraId,
        actor_id=current_user["user"].id,
    )


@router.get(
    "/getcameraframe",
    dependencies=[Depends(require_permission("camera:read"))],
    response_model=Union[CameraFrameSuccessResponse, CommonFailureResponse],
)
def get_camera_frame(
    cameraId: int = Query(..., description="Camera ID"),
    db: Session = Depends(get_db),
    current_user=Depends(require_permission("camera:read")),  # ensures user is available if persist=True
):
    """
    Fetch a single JPEG frame for the given camera as Base64.
    - If a stored ROI/frame blob exists, returns it.
    - Otherwise captures from RTSP and returns it.
    - If `persist=true` and the model supports it, stores the captured blob.
    """
    service = ROIService(db)
    actor_id = getattr(current_user.get("user"), "id", None) if current_user else None
    return service.fetch_camera_frame(camera_id=cameraId, actor_id=actor_id)

@router.post(
    "/refreshcameraframe",
    response_model=Union[CameraFrameSuccessResponse, CommonFailureResponse],
)
def refresh_camera_frame(
    cameraId: int = Query(..., description="Camera ID"),
    db: Session = Depends(get_db),
    current_user=Depends(require_permission("camera:update")),
):
    """
    Refresh camera frame:
    - Deletes existing frame blob
    - Captures new frame from RTSP
    - Saves and returns it
    """
    service = ROIService(db)
    actor_id = getattr(current_user.get("user"), "id", None) if current_user else None

    return service.refresh_camera_frame(
        camera_id=cameraId,
        actor_id=actor_id,
    )
