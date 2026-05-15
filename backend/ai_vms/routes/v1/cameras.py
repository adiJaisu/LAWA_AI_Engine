from typing import Union

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ai_vms.database.database_connection import get_db
from ai_vms.services.v1.camera_service import CameraService
from ai_vms.schemas.cameras import (
    CameraCreateRequest,
    CameraUpdateRequest,
    GetAllCamerasSuccessResponse,
    GetCameraDetailSuccessResponse,
    CreateCameraSuccessResponse,
    UpdateCameraSuccessResponse,
    DeleteCameraSuccessResponse,
    CommonFailureResponse,
)
from ai_vms.utils.auth.auth import require_permission, get_current_user

router = APIRouter()

@router.get(
    "/getallcameras",
    dependencies=[Depends(require_permission("camera:read"))],
    response_model=Union[GetAllCamerasSuccessResponse, CommonFailureResponse],
)
def get_all_cameras(db: Session = Depends(get_db)):
    camera_service = CameraService(db)
    return camera_service.get_all_cameras()

@router.get(
    "/getcameradetail/{camera_id}",
    dependencies=[Depends(require_permission("camera:read"))],
    response_model=Union[GetCameraDetailSuccessResponse, CommonFailureResponse],
)
def get_camera_detail(camera_id: int,
    db: Session = Depends(get_db)):
    camera_service = CameraService(db)
    return camera_service.get_camera_detail(camera_id)

@router.post(
    "/addcamera",
    response_model=Union[CreateCameraSuccessResponse, CommonFailureResponse],
)
def create_camera(
    payload: CameraCreateRequest,
    db: Session = Depends(get_db),
    current_user=Depends(require_permission("camera:create")),
):
    camera_service = CameraService(db)
    return camera_service.create_camera(payload, actor_id=current_user["user"].id)

@router.put(
    "/updatecamera",
    response_model=Union[UpdateCameraSuccessResponse, CommonFailureResponse],
)
def update_camera(
    payload: CameraUpdateRequest,
    db: Session = Depends(get_db),
    current_user=Depends(require_permission("camera:update")),
):
    camera_service = CameraService(db)
    return camera_service.update_camera(
        payload, actor_id=current_user["user"].id)

@router.delete(
    "/deletecamera/{camera_id}",
    response_model=Union[DeleteCameraSuccessResponse, CommonFailureResponse],
)
def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(require_permission("camera:delete")),
):
    camera_service = CameraService(db)
    return camera_service.delete_camera(camera_id, actor_id=current_user["user"].id)