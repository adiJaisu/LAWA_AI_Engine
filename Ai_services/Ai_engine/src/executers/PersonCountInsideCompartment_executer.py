from time import time
from typing import List, Dict, Any
import cv2
import os

from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.constant.global_constant import VisionPipeline
from src.business_cases.PersonCountInsideCompartment import PersonCountInsideCompartment
from src.Exception.Exception import FrameProcessingException

logging_config = LoggingConfig()
logger = logging_config.setup_logging()


def execute_person_count_inside_compartment(
    validated_msg_with_frames_and_metadatas: List[Dict],
    detector: Any = None
) -> List[Dict[str, Any]]:
    """
    Processes a batch of frames for person counting inside compartment.
    """

    try:
        processed_frames: List[Dict[str, Any]] = []
        alert_count = Constants.ZERO

        for idx, msg in enumerate(validated_msg_with_frames_and_metadatas):

            try:
                raw_frame = msg[Constants.FRAME_METADATA][Constants.FRAME]
                roi = msg[Constants.FRAME_METADATA][Constants.ROIS]
                camera_id = msg[Constants.CAMERA_METADATA][Constants.CAMERA_ID]

            except (KeyError, IndexError) as e:
                logger.warning(f"Frame {idx}: Missing metadata: {str(e)}. Skipping.")
                continue

            try:
                # ---------------- INIT PER CAMERA ----------------
                if camera_id not in VisionPipeline.camera_tracker:
                    VisionPipeline.camera_tracker[camera_id] = PersonCountInsideCompartment()

                # ---------------- RUN DETECTION ----------------
                processed_frame, alert_flag, status = (
                    VisionPipeline.camera_tracker[camera_id].detect(
                        frame=raw_frame,
                        rois=roi,
                        detector=detector
                    )
                )

                # ---------------- ALERT HANDLING ----------------
                if alert_flag:
                    alert_count += Constants.ONE

                    os.makedirs("src/detected_frames", exist_ok=True)
                    cv2.imwrite(
                        f"src/detected_frames/crowd_alert_{int(time())}_{idx}.jpg",
                        processed_frame
                    )

                # ---------------- UPDATE MESSAGE ----------------
                msg[Constants.FRAME_METADATA][Constants.RAW_FRAME] = raw_frame
                msg[Constants.FRAME_METADATA][Constants.FRAME] = processed_frame
                msg[Constants.FRAME_METADATA][Constants.ALERT] = alert_flag
                msg[Constants.FRAME_METADATA][Constants.DETECTIONS] = status

                processed_frames.append(msg)

            except Exception as e:
                logger.error(f"Frame {idx}: Error during processing: {str(e)}", exc_info=True)
                continue

        # ---------------- RETURN LOGIC ----------------
        if alert_count > Constants.ZERO:
            logger.debug(f"Alert detected in batch: {alert_count}")
            return processed_frames
        else:
            return []

    except FrameProcessingException as e:
        logger.critical(f"Critical error in processing: {str(e)}", exc_info=True)
        raise