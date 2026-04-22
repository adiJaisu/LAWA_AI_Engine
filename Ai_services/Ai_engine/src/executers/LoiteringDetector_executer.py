from time import time
from typing import List, Dict, Any
import cv2
import os
import numpy as np
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.constant.global_constant import VisionPipeline
from src.business_cases.LoiteringDetection import LoiteringDetection
from src.Exception.Exception import FrameProcessingException

logging_config = LoggingConfig()
logger = logging_config.setup_logging()

def execute_loitering_detection(validated_msg_with_frames_and_metadatas: List[Dict], detector: Any = None) -> List[Dict[str, Any]]:
    """Processes a batch of frames to detect loitering usage violations within specified ROIs."""
    
    try:
        processed_frames: List[Dict[str, Any]] = []
        alert_count = Constants.ZERO

        for idx, msg in enumerate(validated_msg_with_frames_and_metadatas):
            raw_frame = msg[Constants.FRAME_METADATA][Constants.FRAME]

            try:
                roi = msg[Constants.FRAME_METADATA][Constants.ROIS]
                logger.debug(f"ROI from executor: {roi}")
                camera_id = msg[Constants.CAMERA_METADATA][Constants.CAMERA_ID]
            except (KeyError, IndexError) as e:
                logger.warning(f"Frame {idx}: ROIs or camera id missing in metadata: {str(e)}. Skipping.")
                continue

            try:
                if camera_id not in VisionPipeline.camera_tracker:
                    VisionPipeline.camera_tracker[camera_id] = LoiteringDetection()

                loitering_alert, unannotated_frame, live_status = (
                    VisionPipeline.camera_tracker[camera_id].detect(
                        frame=raw_frame,
                        rois=roi,
                        detector=detector
                    )
                )

                if loitering_alert:
                    alert_count += Constants.ONE
                    
                    annotated_frame = unannotated_frame.copy()
                    bboxes = live_status.get("bboxes", [])
                    for box_info in bboxes:
                        x1, y1, x2, y2 = box_info["bbox"]
                        tid = box_info["id"]
                        loiter_zones = box_info.get("loitering_zones", [])
                        if loiter_zones:
                            color = (0, 0, 255) # Red for loitering
                            label = f"ID {tid} LOITER: {','.join(loiter_zones)}"
                        else:
                            color = (0, 255, 0) # Green for tracking
                            label = f"ID {tid}"
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Use absolute path for saving frames
                    detected_frames_dir = os.path.join(Constants.PARENT_DIR, "detected_frames")
                    os.makedirs(detected_frames_dir, exist_ok=True)
                    save_path = os.path.join(detected_frames_dir, f"loitering_alert_{camera_id}_{int(time())}_{idx}.jpg")
                    cv2.imwrite(save_path, annotated_frame)
                    logger.debug(f"Saved alert frame to {save_path}")

                msg[Constants.FRAME_METADATA][Constants.RAW_FRAME] = raw_frame
                msg[Constants.FRAME_METADATA][Constants.FRAME] = unannotated_frame
                msg[Constants.FRAME_METADATA][Constants.ALERT] = loitering_alert
                msg[Constants.FRAME_METADATA][Constants.DETECTIONS] = live_status
                processed_frames.append(msg)

            except Exception as e:
                logger.error(f"Frame {idx}: Error during detection: {str(e)}", exc_info=True)
                continue
                
        return processed_frames if alert_count > Constants.ZERO else []

    except FrameProcessingException as e:
        logger.critical(f"Critical error in loitering detection: {str(e)}", exc_info=True)
        raise
