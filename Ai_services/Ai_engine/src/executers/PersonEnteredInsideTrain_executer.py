from time import time
from typing import List, Dict, Any
import cv2
import os
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.constant.global_constant import VisionPipeline
from src.business_cases.PersonEnteredInsideTrain import PersonEnteredInsideTrain, PersonEnteredInsideTrainException
from src.Exception.Exception import FrameProcessingException

logging_config = LoggingConfig()
logger = logging_config.setup_logging()

def execute_person_entered_inside_train(validated_msg_with_frames_and_metadatas: List[Dict], detector: Any = None) -> List[Dict[str, Any]]:
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
                    VisionPipeline.camera_tracker[camera_id] = PersonEnteredInsideTrain()

                unannotated_frame, train_alert, status_dict = (
                    VisionPipeline.camera_tracker[camera_id].detect(
                        frame=raw_frame,
                        rois=roi,
                        detector=detector
                    )
                )

                if train_alert:
                    alert_count += Constants.ONE
                    
                    os.makedirs("src/detected_frames", exist_ok=True)
                    cv2.imwrite(f"src/detected_frames/train_arrival_depart_alert_{int(time())}_{idx}.jpg", unannotated_frame)

                msg[Constants.FRAME_METADATA][Constants.RAW_FRAME] = raw_frame
                msg[Constants.FRAME_METADATA][Constants.FRAME] = unannotated_frame
                msg[Constants.FRAME_METADATA][Constants.ALERT] = train_alert
                msg[Constants.FRAME_METADATA][Constants.DETECTIONS] = status_dict
                processed_frames.append(msg)

            except PersonEnteredInsideTrainException as e:
                logger.error(f"Frame {idx}: person entered inside train error: {str(e)}", exc_info=True)
                continue
            except Exception as e:
                logger.error(f"Frame {idx}: Error during person entered inside train: {str(e)}", exc_info=True)
                continue

        if alert_count > Constants.ZERO:
            logger.debug(f"Found alert for this batch for person entered inside train alert count {alert_count}")
            return processed_frames
        else:
            return []

    except FrameProcessingException as e:
        logger.critical(f"Critical error in person entered inside train: {str(e)}", exc_info=True)
        raise
