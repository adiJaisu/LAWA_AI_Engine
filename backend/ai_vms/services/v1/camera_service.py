"""
Service layer for Camera management.

Handles business logic for camera CRUD, validations, workflow management,
and updating associations with usecases.

Author: HCLTech
"""

import logging
from typing import Union, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from ai_vms.models.camera import Camera
try:
    from ai_vms.models.enums import CameraType
except Exception:
    CameraType = None  # fallback

from ai_vms.crud.cameras import (
    get_all_cameras,
    get_camera_by_id,
    get_camera_by_name,
    create_camera as create_camera_db,
    update_camera as update_camera_db,
    soft_delete_camera as soft_delete_camera_db,
    validate_usecase_ids,
    replace_camera_usecases,
)

from ai_vms.schemas.cameras import (
    CameraCreateRequest,
    CameraUpdateRequest,
    CameraInfoResponse,
    UsecaseInfoResponse,
    GetAllCamerasSuccessResponse,
    GetCameraDetailSuccessResponse,
    CreateCameraSuccessResponse,
    UpdateCameraSuccessResponse,
    DeleteCameraSuccessResponse,
    CommonFailureResponse,
)

logger = logging.getLogger(__name__)


class CameraService:
    def __init__(self, db: Session):
        self.db = db

    def _to_camera_info(self, cam: Camera) -> CameraInfoResponse:
        usecases: List[UsecaseInfoResponse] = []
        for link in (cam.usecase_links or []):
            if link.usecase:
                usecases.append(
                    UsecaseInfoResponse(
                        usecaseId=link.usecase.id,
                        usecaseName=link.usecase.name
                    )
                )

        cam_type = cam.type.value if hasattr(cam.type, "value") else str(cam.type)
        
        return CameraInfoResponse(
            cameraId=cam.id,
            name=cam.name,
            type=cam_type,
            rtspUrl=cam.rtsp_url,
            resolution=cam.resolution,
            height=cam.height,
            resolutionWidth=cam.resolution_width,
            resolutionHeight=cam.resolution_height,
            decodingResource=cam.decoding_resource or "CPU",
            decodingPipeline=cam.decoding_pipeline or "FFmpeg",
            dockerMode=cam.docker_mode or "load",
            codec=cam.codec,
            fps=cam.fps,
            usecases=usecases,
            roi=cam.roi,
            status=1 if cam.is_active else 0,
            is_delete=1 if cam.is_delete else 0,
            createdAt=cam.created_at,
            updatedAt=cam.updated_at,
        )

    def _validate_camera_fields(self, cam_type: str, rtsp: Optional[str]) -> Union[None, CommonFailureResponse]:
        if cam_type == "rtsp" and not rtsp:
            return CommonFailureResponse(code=400, message="rtspUrl is required for RTSP cameras")
        return None

    def _validate_usecase_ids(self, usecase_ids: List[int]) -> Union[List[int], CommonFailureResponse]:
        usecase_ids = usecase_ids or []
        valid_ids = validate_usecase_ids(self.db, usecase_ids)
        invalid = sorted(set(usecase_ids) - set(valid_ids))
        if invalid:
            return CommonFailureResponse(code=400, message=f"Invalid usecaseIds: {invalid}")
        return valid_ids

    def get_all_cameras(self) -> Union[GetAllCamerasSuccessResponse, CommonFailureResponse]:
        try:
            logger.info("API requested to fetch all cameras.")
            cams = get_all_cameras(self.db)
            camera_list = [self._to_camera_info(c) for c in cams]
            return GetAllCamerasSuccessResponse(code=200, message="Cameras fetched successfully", cameras=camera_list)

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database Error while fetching cameras: {str(e)}")
            return CommonFailureResponse(code=500, message="Database Error Occurred")

        except Exception as e:
            logger.error(f"Internal Server Error while fetching cameras: {str(e)}")
            return CommonFailureResponse(code=500, message="Internal Server Error")

    def get_camera_detail(self, camera_id: int) -> Union[GetCameraDetailSuccessResponse, CommonFailureResponse]:
        try:
            logger.info(f"API requested to fetch camera detail: cameraId={camera_id}")
            cam = get_camera_by_id(self.db, camera_id)
            if not cam:
                return CommonFailureResponse(code=404, message="Camera not found")

            # Load with relations
            return GetCameraDetailSuccessResponse(
                code=200,
                message="Camera fetched successfully",
                cameraDetails=self._to_camera_info(cam)
            )

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database Error while fetching camera detail: {str(e)}")
            return CommonFailureResponse(code=500, message="Database Error Occurred")

        except Exception as e:
            logger.error(f"Internal Server Error while fetching camera detail: {str(e)}")
            return CommonFailureResponse(code=500, message="Internal Server Error")

    def create_camera(
        self,
        payload: CameraCreateRequest,
        actor_id: int
    ) -> Union[CreateCameraSuccessResponse, CommonFailureResponse]:
        try:
            logger.info("API requested to create camera.")

            if not payload.name or not payload.type:
                return CommonFailureResponse(code=400, message="name and type are required")

            if get_camera_by_name(self.db, payload.name):
                return CommonFailureResponse(code=409, message="Camera name already exists")

            err = self._validate_camera_fields(payload.type, payload.rtspUrl)
            if err:
                return err

            valid_ids_or_error = self._validate_usecase_ids(payload.usecaseIds or [])
            if isinstance(valid_ids_or_error, CommonFailureResponse):
                return valid_ids_or_error
            valid_usecase_ids = valid_ids_or_error

            cam_type_value = payload.type
            if CameraType is not None:
                try:
                    cam_type_value = CameraType(payload.type)
                except Exception:
                    cam_type_value = payload.type

            cam = Camera(
                name=payload.name,
                type=cam_type_value,
                rtsp_url=payload.rtspUrl,
                resolution=payload.resolution,
                height=payload.height,
                resolution_width=payload.resolutionWidth,
                resolution_height=payload.resolutionHeight,
                decoding_resource=payload.decodingResource,
                decoding_pipeline=payload.decodingPipeline,
                docker_mode=payload.dockerMode,
                roi=payload.roi,
                codec=payload.codec,
                fps=payload.fps,
                is_active=True if int(payload.status or 1) == 1 else False,
                created_by=actor_id,
                updated_by=actor_id,
            )

            cam = create_camera_db(self.db, cam)

            if valid_usecase_ids:
                replace_camera_usecases(self.db, cam.id, valid_usecase_ids, actor_id=actor_id)

            cam = get_camera_by_id(self.db, cam.id)

            return CreateCameraSuccessResponse(
                code=200,
                message="Camera created successfully",
                camera=self._to_camera_info(cam)
            )

        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Integrity Error while creating camera: {str(e)}")
            return CommonFailureResponse(code=409, message="Duplicate/constraint violation")

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database Error while creating camera: {str(e)}")
            return CommonFailureResponse(code=500, message="Database Error Occurred")

        except Exception as e:
            logger.error(f"Internal Server Error while creating camera: {str(e)}")
            return CommonFailureResponse(code=500, message="Internal Server Error")

    def update_camera(
        self,
        payload: CameraUpdateRequest,
        actor_id: int
    ) -> Union[UpdateCameraSuccessResponse, CommonFailureResponse]:
        try:
            logger.info(f"API requested to update camera: cameraId={payload.cameraId}")

            cam = get_camera_by_id(self.db, payload.cameraId)
            if not cam:
                return CommonFailureResponse(code=404, message="Camera not found")

            if payload.name and payload.name != cam.name:
                if get_camera_by_name(self.db, payload.name):
                    return CommonFailureResponse(code=409, message="Camera name already exists")
                cam.name = payload.name

            if payload.type is not None:
                cam.type = payload.type

            if payload.rtspUrl is not None:
                cam.rtsp_url = payload.rtspUrl

            cam_type = cam.type.value if hasattr(cam.type, "value") else str(cam.type)
            err = self._validate_camera_fields(cam_type, cam.rtsp_url)
            if err:
                return err

            if payload.resolution is not None:
                cam.resolution = payload.resolution

            if payload.height is not None:
                cam.height = payload.height

            if payload.resolutionWidth is not None:
                cam.resolution_width = payload.resolutionWidth

            if payload.resolutionHeight is not None:
                cam.resolution_height = payload.resolutionHeight

            if payload.decodingResource is not None:
                cam.decoding_resource = payload.decodingResource

            if payload.decodingPipeline is not None:
                cam.decoding_pipeline = payload.decodingPipeline

            if payload.dockerMode is not None:
                cam.docker_mode = payload.dockerMode
            
            if payload.fps is not None:
                cam.fps = payload.fps

            if payload.codec is not None:
                cam.codec = payload.codec

            if payload.roi is not None:
                cam.roi = payload.roi

            if payload.status is not None:
                cam.is_active = True if int(payload.status) == 1 else False

            cam.updated_by = actor_id
            cam = update_camera_db(self.db, cam)

            if payload.usecaseIds is not None:
                valid_ids_or_error = self._validate_usecase_ids(payload.usecaseIds or [])
                if isinstance(valid_ids_or_error, CommonFailureResponse):
                    return valid_ids_or_error
                valid_usecase_ids = valid_ids_or_error

                replace_camera_usecases(self.db, payload.cameraId, valid_usecase_ids, actor_id=actor_id)

            cam = get_camera_by_id(self.db, payload.cameraId)

            return UpdateCameraSuccessResponse(
                code=200,
                message="Camera updated successfully",
                camera=self._to_camera_info(cam)
            )

        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Integrity Error while updating camera: {str(e)}")
            return CommonFailureResponse(code=409, message="Duplicate/constraint violation")

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database Error while updating camera: {str(e)}")
            return CommonFailureResponse(code=500, message="Database Error Occurred")

        except Exception as e:
            logger.error(f"Internal Server Error while updating camera: {str(e)}")
            return CommonFailureResponse(code=500, message="Internal Server Error")

    def delete_camera(
        self,
        camera_id: int,
        actor_id: int
    ) -> Union[DeleteCameraSuccessResponse, CommonFailureResponse]:
        try:
            logger.info(f"API requested to delete camera: cameraId={camera_id}")

            cam = get_camera_by_id(self.db, camera_id)
            if not cam:
                return CommonFailureResponse(code=404, message="Camera not found")

            cam.updated_by = actor_id
            soft_delete_camera_db(self.db, cam)

            return DeleteCameraSuccessResponse(code=200, message="Camera deleted successfully")

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database Error while deleting camera: {str(e)}")
            return CommonFailureResponse(code=500, message="Database Error Occurred")

        except Exception as e:
            logger.error(f"Internal Server Error while deleting camera: {str(e)}")
            return CommonFailureResponse(code=500, message="Internal Server Error")
