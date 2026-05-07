"""
Service layer for ROI management on Camera.

Author: HCLTech
"""

import logging
from typing import Any, Optional, Union, Dict
import base64
import time 

import cv2
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from ai_vms.crud.cameras import (
    get_camera_by_id,
    get_camera_roi as repo_get_camera_roi,
    set_camera_roi as repo_set_camera_roi,
    get_camera_frame_blob,
    set_camera_frame_blob,
)

from ai_vms.schemas.roi import (
    RoiGetSuccessResponse,
    RoiCreateSuccessResponse,
    RoiUpdateSuccessResponse,
    RoiDeleteSuccessResponse,
    CameraFrameSuccessResponse
)
from ai_vms.schemas.cameras import CommonFailureResponse

logger = logging.getLogger(__name__)


class ROIService:
    """
    Business logic for interacting with the ROI JSON stored on the Camera table.
    Returns:
      - Success schemas: Roi*SuccessResponse
      - Failure schema: CommonFailureResponse
    """

    def __init__(self, db: Session, roi_validator: Optional[callable] = None) -> None:
        self.db = db
        self._validate = roi_validator

    # -------- GET --------
    def get_camera_roi(self, camera_id: int) -> Union[RoiGetSuccessResponse, CommonFailureResponse]:
        try:
            camera = get_camera_by_id(self.db, camera_id)
            if not camera or getattr(camera, "is_delete", False):
                return CommonFailureResponse(code=404, message="Camera not found")

            roi = repo_get_camera_roi(self.db, camera_id)
            return RoiGetSuccessResponse(message="ROI fetched successfully", roi=roi)

        except SQLAlchemyError:
            logger.exception("DB error while fetching ROI for camera_id=%s", camera_id)
            return CommonFailureResponse(code=500, message="Internal server error while fetching ROI")

    # -------- CREATE --------
    def create_camera_roi(
        self,
        camera_id: int,
        roi_payload: Any,
        actor_id: int,
    ) -> Union[RoiCreateSuccessResponse, CommonFailureResponse]:
        try:
            camera = get_camera_by_id(self.db, camera_id)
            if not camera or getattr(camera, "is_delete", False):
                return CommonFailureResponse(code=404, message="Camera not found")

            if camera.roi is not None:
                return CommonFailureResponse(code=409, message="ROI already exists for this camera")

            if self._validate:
                self._validate(roi_payload)

            roi_payload = self.add_points_to_roi(roi_payload)

            updated = repo_set_camera_roi(self.db, camera_id, roi_payload, actor_id)
            return RoiCreateSuccessResponse(message="ROI created successfully", roi=updated)

        except IntegrityError:
            logger.exception("Integrity error while creating ROI for camera_id=%s", camera_id)
            return CommonFailureResponse(code=500, message="Database integrity error while creating ROI")
        except SQLAlchemyError:
            logger.exception("DB error while creating ROI for camera_id=%s", camera_id)
            return CommonFailureResponse(code=500, message="Internal server error while creating ROI")
        except ValueError as ve:
            # Validator can raise ValueError
            return CommonFailureResponse(code=400, message=str(ve))
        except Exception:
            logger.exception("Unexpected error while creating ROI for camera_id=%s", camera_id)
            return CommonFailureResponse(code=500, message="Unexpected error while creating ROI")

    # -------- UPDATE  --------
    def update_camera_roi(
        self,
        camera_id: int,
        roi_payload: Any,
        actor_id: int,
    ) -> Union[RoiUpdateSuccessResponse, CommonFailureResponse]:
        try:
            camera = get_camera_by_id(self.db, camera_id)
            if not camera or getattr(camera, "is_delete", False):
                return CommonFailureResponse(code=404, message="Camera not found")

            if self._validate:
                self._validate(roi_payload)

            roi_payload = self.add_points_to_roi(roi_payload)

            updated = repo_set_camera_roi(self.db, camera_id, roi_payload, actor_id)
            return RoiUpdateSuccessResponse(message="ROI updated successfully", roi=updated)

        except IntegrityError:
            logger.exception("Integrity error while updating ROI for camera_id=%s", camera_id)
            return CommonFailureResponse(code=500, message="Database integrity error while updating ROI")
        except SQLAlchemyError:
            logger.exception("DB error while updating ROI for camera_id=%s", camera_id)
            return CommonFailureResponse(code=500, message="Internal server error while updating ROI")
        except ValueError as ve:
            return CommonFailureResponse(code=400, message=str(ve))
        except Exception:
            logger.exception("Unexpected error while updating ROI for camera_id=%s", camera_id)
            return CommonFailureResponse(code=500, message="Unexpected error while updating ROI")

    # -------- DELETE (clear to null) --------
    def delete_camera_roi(self, camera_id: int, actor_id: int) -> Union[RoiDeleteSuccessResponse, CommonFailureResponse]:
        try:
            camera = get_camera_by_id(self.db, camera_id)
            if not camera or getattr(camera, "is_delete", False):
                return CommonFailureResponse(code=404, message="Camera not found")

            if camera.roi is None:
                return CommonFailureResponse(code=404, message="ROI not found for this camera")

            _ = repo_set_camera_roi(self.db, camera_id, None, actor_id)
            return RoiDeleteSuccessResponse(message="ROI deleted successfully")

        except IntegrityError:
            logger.exception("Integrity error while deleting ROI for camera_id=%s", camera_id)
            return CommonFailureResponse(code=500, message="Database integrity error while deleting ROI")
        except SQLAlchemyError:
            logger.exception("DB error while deleting ROI for camera_id=%s", camera_id)
            return CommonFailureResponse(code=500, message="Internal server error while deleting ROI")
        except Exception:
            logger.exception("Unexpected error while deleting ROI for camera_id=%s", camera_id)
            return CommonFailureResponse(code=500, message="Unexpected error while deleting ROI")
    
    #fetching camera frame for roi
    def fetch_camera_frame(
        self,
        camera_id: int,
        actor_id: Optional[int] = None,
    ) -> Union[CameraFrameSuccessResponse, CommonFailureResponse]:
        """
        Simple flow:
          1) If frame blob exists in DB -> return it.
          2) Else capture from RTSP, encode JPEG, optionally persist, return it.
        """
        try:
            camera = get_camera_by_id(self.db, camera_id)
            if not camera or getattr(camera, "is_delete", False):
                return CommonFailureResponse(code=404, message="Camera not found")

            # 1) Try stored blob
            blob = None
            try:
                blob = get_camera_frame_blob(self.db, camera_id)
            except Exception:
                logger.debug("get_camera_frame_blob failed; will try RTSP.", exc_info=True)

            if blob:
                return CameraFrameSuccessResponse(
                    code=200,
                    message="Frame fetched from database successfully",
                    frameFile=base64.b64encode(blob).decode("utf-8"),
                )

            # 2) Capture from RTSP (blob is missing → first time)
            if not getattr(camera, "rtsp_url", None):
                return CommonFailureResponse(code=400, message="RTSP URL not available for this camera")

            frame = self._capture_frame_from_rtsp(camera.rtsp_url)
            if frame is None:
                return CommonFailureResponse(code=500, message="Failed to capture frame from RTSP stream")

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                logger.error("Failed to encode frame as JPEG (camera_id=%s)", camera_id)
                return CommonFailureResponse(code=500, message="Failed to encode frame")

            blob = buffer.tobytes()

            # Always persist on first capture (since no blob existed)
            try:
                _ = set_camera_frame_blob(self.db, camera_id, blob, actor_id)
            except Exception:
                # Do not fail the request if persistence has a transient issue
                logger.debug("set_camera_frame_blob failed; continuing to return frame.", exc_info=True)

            return CameraFrameSuccessResponse(
                code=200,
                message="Frame fetched successfully",
                frameFile=base64.b64encode(blob).decode("utf-8"),
            )

        except SQLAlchemyError:
            self.db.rollback()
            logger.exception("DB error while fetching frame (camera_id=%s)", camera_id)
            return CommonFailureResponse(code=500, message="Database error occurred while fetching frame")
        except Exception:
            logger.exception("Internal server error while fetching frame (camera_id=%s)", camera_id)
            return CommonFailureResponse(code=500, message="Internal server error while fetching frame")

    # ---helper for capturing frame ---
    def _capture_frame_from_rtsp(self, rtsp_url: str):
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # small buffer for newer frames

        if not cap.isOpened():
            logger.error("Failed to open RTSP stream.")
            return None

        # Prime and read a fresh frame
        _ = cap.read()
        time.sleep(1.0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.error("Failed to read frame from RTSP stream.")
            return None
        return frame
    
    
    def refresh_camera_frame(
        self,
        camera_id: int,
        actor_id: Optional[int] = None,
    ) -> Union[CameraFrameSuccessResponse, CommonFailureResponse]:
        """
        Force refresh:
        1) Delete existing frame blob
        2) Capture new frame from RTSP
        3) Save and return
        """
        try:
            camera = get_camera_by_id(self.db, camera_id)
            if not camera or getattr(camera, "is_delete", False):
                return CommonFailureResponse(code=404, message="Camera not found")

            if not getattr(camera, "rtsp_url", None):
                return CommonFailureResponse(code=400, message="RTSP URL not available for this camera")

            # 1) Clear any existing stored frame
            try:
                _ = set_camera_frame_blob(self.db, camera_id, None, actor_id)
            except Exception:
                logger.debug("Failed to clear existing frame blob", exc_info=True)

            # 2) Capture a fresh frame from RTSP
            frame = self._capture_frame_from_rtsp(camera.rtsp_url)
            if frame is None:
                return CommonFailureResponse(code=500, message="Failed to capture frame from RTSP stream")

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                logger.error("Failed to encode refreshed frame (camera_id=%s)", camera_id)
                return CommonFailureResponse(code=500, message="Failed to encode frame")

            blob = buffer.tobytes()

            # 3) Persist the refreshed frame
            try:
                _ = set_camera_frame_blob(self.db, camera_id, blob, actor_id)
            except Exception:
                logger.debug("Failed to persist refreshed frame", exc_info=True)

            return CameraFrameSuccessResponse(
                code=200,
                message="Frame refreshed successfully",
                frameFile=base64.b64encode(blob).decode("utf-8"),
            )

        except SQLAlchemyError:
            self.db.rollback()
            logger.exception("DB error while refreshing frame (camera_id=%s)", camera_id)
            return CommonFailureResponse(code=500, message="Database error occurred while refreshing frame")
        except Exception:
            logger.exception("Unexpected error while refreshing frame (camera_id=%s)", camera_id)
            return CommonFailureResponse(code=500, message="Internal server error while refreshing frame")
    

    #converting rect coordinates to polygon coordinates
    def add_points_to_roi(self, roi_list):
        updated_roi = []

        for roi in roi_list:
            if roi.get("type") == "rect":
                x = roi["x"]
                y = roi["y"]
                w = roi["w"]
                h = roi["h"]

                roi["points"] = [
                    [x, y],
                    [x + w, y],
                    [x, y + h],
                    [x + w, y + h]
                ]

            updated_roi.append(roi)

        return updated_roi