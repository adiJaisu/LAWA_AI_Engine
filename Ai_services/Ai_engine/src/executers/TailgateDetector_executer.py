from time import time
from typing import List, Dict, Any
import cv2
import os
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.constant.global_constant import VisionPipeline
from src.business_cases.TailgateDetection import TailgateDetection, TailgateDetectionException
from src.Exception.Exception import FrameProcessingException

logging_config = LoggingConfig()
logger = logging_config.setup_logging()


def execute_tailgate_detection(validated_msg_with_frames_and_metadatas: List[Dict], detector: Any = None) -> List[Dict[str, Any]]:
    """Processes a batch of frames to detect tailgating violations inside configured ROIs."""
    try:
        processed_frames: List[Dict[str, Any]] = []
        alert_count = Constants.ZERO

        for idx, msg in enumerate(validated_msg_with_frames_and_metadatas):
            raw_frame = msg[Constants.FRAME_METADATA][Constants.FRAME]

            try:
                roi = msg[Constants.FRAME_METADATA][Constants.ROIS]
                camera_id = msg[Constants.CAMERA_METADATA][Constants.CAMERA_ID]
            except (KeyError, IndexError) as e:
                logger.warning(f"Frame {idx}: ROIs or camera id missing in metadata: {str(e)}. Skipping.")
                continue

            try:
                if camera_id not in VisionPipeline.camera_tracker:
                    VisionPipeline.camera_tracker[camera_id] = TailgateDetection()

                unannotated_frame, tailgate_alert, tailgate_status = (
                    VisionPipeline.camera_tracker[camera_id].detect(
                        frame=raw_frame,
                        rois=roi,
                        detector=detector,
                    )
                )

                if tailgate_alert:
                    alert_count += Constants.ONE
                    
                    # Annotate frame before saving
                    # annotated_frame = unannotated_frame.copy()
                    # for det in tailgate_status.get("detections", []):
                    #     x1, y1, x2, y2 = map(int, det["bbox"])
                    #     color = (0, 0, 255) if det["is_tailgate"] or det["is_invalid"] else (0, 255, 0)
                    #     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    #     cv2.putText(annotated_frame, f"ID:{det['id']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # cv2.putText(annotated_frame, tailgate_status.get("status", ""), (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                    os.makedirs("src/detected_frames", exist_ok=True)
                    cv2.imwrite(f"src/detected_frames/tailgate_alert_{int(time())}_{idx}.jpg", unannotated_frame)

                msg[Constants.FRAME_METADATA][Constants.RAW_FRAME] = raw_frame
                msg[Constants.FRAME_METADATA][Constants.FRAME] = unannotated_frame
                msg[Constants.FRAME_METADATA][Constants.ALERT] = tailgate_alert
                msg[Constants.FRAME_METADATA][Constants.DETECTIONS] = tailgate_status
                processed_frames.append(msg)

            except TailgateDetectionException as e:
                logger.error(f"Frame {idx}: Tailgate detection error: {str(e)}", exc_info=True)
                continue
            except Exception as e:
                logger.error(f"Frame {idx}: Error during tailgate detection: {str(e)}", exc_info=True)
                continue

        if alert_count > Constants.ZERO:
            logger.debug(f"Found alert for this batch for tailgate alert count {alert_count}")
            return processed_frames
        else:
            return []

    except FrameProcessingException as e:
        logger.critical(f"Critical error in tailgate detection: {str(e)}", exc_info=True)
        raise
