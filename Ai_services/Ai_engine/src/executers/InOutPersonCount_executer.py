from time import time
from typing import List, Dict, Any
import cv2
import os
import numpy as np
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.constant.global_constant import VisionPipeline
from src.business_cases.InOutPersonCount import InOutPersonCount
from src.Exception.Exception import FrameProcessingException

logging_config = LoggingConfig()
logger = logging_config.setup_logging()

def execute_in_out_person_count(validated_msg_with_frames_and_metadatas: List[Dict], detector: Any = None) -> List[Dict[str, Any]]:
    """Processes a batch of frames to detect in/out cross counting."""
    
    try:
        processed_frames: List[Dict[str, Any]] = []
        alert_count = Constants.ZERO

        for idx, msg in enumerate(validated_msg_with_frames_and_metadatas):
            raw_frame = msg[Constants.FRAME_METADATA][Constants.FRAME]

            try:
                lines = msg[Constants.FRAME_METADATA][Constants.ROIS]
                logger.debug(f"Lines from executor: {lines}")
                camera_id = msg[Constants.CAMERA_METADATA][Constants.CAMERA_ID]
            except (KeyError, IndexError) as e:
                logger.warning(f"Frame {idx}: ROIs or camera id missing in metadata: {str(e)}. Skipping.")
                continue

            try:
                if camera_id not in VisionPipeline.camera_tracker:
                    VisionPipeline.camera_tracker[camera_id] = InOutPersonCount()

                alert, unannotated_frame, live_status = (
                    VisionPipeline.camera_tracker[camera_id].detect(
                        frame=raw_frame,
                        lines=lines,
                        detector=detector
                    )
                )

                msg[Constants.FRAME_METADATA][Constants.RAW_FRAME] = raw_frame
                msg[Constants.FRAME_METADATA][Constants.FRAME] = unannotated_frame
                msg[Constants.FRAME_METADATA][Constants.ALERT] = alert
                msg[Constants.FRAME_METADATA][Constants.DETECTIONS] = live_status
                processed_frames.append(msg)

            except Exception as e:
                logger.error(f"Frame {idx}: Error during detection: {str(e)}", exc_info=True)
                continue
                
        # Return all processed frames for downstream (Event Manager, streaming, etc)
        return processed_frames

    except FrameProcessingException as e:
        logger.critical(f"Critical error in in_out_person_count detection: {str(e)}", exc_info=True)
        raise
