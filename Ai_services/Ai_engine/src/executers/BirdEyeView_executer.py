import os
import cv2
import threading
from time import time
from typing import List, Dict, Any
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.constant.global_constant import VisionPipeline
from src.business_cases.BirdEyeView import BirdEyeView
from src.Exception.Exception import FrameProcessingException, BaseException, BirdEyeViewException
from src.utils.ConfigReader import cfg

logging_config = LoggingConfig()
logger = logging_config.setup_logging()

# Lock for synchronizing the buffer across threads
buffer_lock = threading.Lock()

def execute_bird_eye_view(validated_msg_with_frames_and_metadatas: List[Dict], detector: Any = None) -> List[Dict[str, Any]]:
    try:
        processed_batches: List[Dict[str, Any]] = []
        
        expected_cameras = int(cfg.get_value_config(Constants.BIRD_EYE_VIEW, "EXPECTED_CAMERAS"))

        for msg in validated_msg_with_frames_and_metadatas:
            location_id = msg[Constants.CAMERA_METADATA][Constants.LOCATION_ID]
            camera_id = msg[Constants.CAMERA_METADATA][Constants.CAMERA_ID]
            timestamp = msg[Constants.FRAME_METADATA][Constants.TIME_STAMP]

            with buffer_lock:
                if location_id not in VisionPipeline.location_buffer:
                    VisionPipeline.location_buffer[location_id] = {}
                
                if timestamp not in VisionPipeline.location_buffer[location_id]:
                    VisionPipeline.location_buffer[location_id][timestamp] = {}
                
                VisionPipeline.location_buffer[location_id][timestamp][camera_id] = msg

                # Check if we have all cameras for this timestamp
                if len(VisionPipeline.location_buffer[location_id][timestamp]) == expected_cameras:
                    # Collect frames for processing
                    batch_messages = list(VisionPipeline.location_buffer[location_id][timestamp].values())
                    
                    # Remove from buffer
                    del VisionPipeline.location_buffer[location_id][timestamp]
                    
                    # Cleanup old timestamps in buffer to prevent memory leaks
                    if len(VisionPipeline.location_buffer[location_id]) > 50:
                        sorted_ts = sorted(VisionPipeline.location_buffer[location_id].keys())
                        for old_ts in sorted_ts[:-25]:
                            del VisionPipeline.location_buffer[location_id][old_ts]
                else:
                    continue

            # If we reached here, we have a complete batch
            try:
                if location_id not in VisionPipeline.camera_tracker:
                    VisionPipeline.camera_tracker[location_id] = BirdEyeView(detector=detector)

                combined_vis, alert, status_dict = (
                    VisionPipeline.camera_tracker[location_id].process_fused_frames(
                        batch_messages=batch_messages,
                        detector=detector
                    )
                )

                result_msg = batch_messages[0]
                result_msg[Constants.FRAME_METADATA][Constants.RAW_FRAME] = result_msg[Constants.FRAME_METADATA][Constants.FRAME]
                result_msg[Constants.FRAME_METADATA][Constants.FRAME] = combined_vis
                result_msg[Constants.FRAME_METADATA][Constants.ALERT] = alert
                result_msg[Constants.FRAME_METADATA][Constants.DETECTIONS] = status_dict
                
                processed_batches.append(result_msg)

            except BirdEyeViewException as e:
                logger.error(f"Bird Eye View specific error for location {location_id}: {str(e)}")
                continue
            except FrameProcessingException as e:
                logger.error(f"Frame processing error in BEV for location {location_id}: {str(e)}")
                continue
            except BaseException as e:
                logger.error(f"General AI engine error for location {location_id}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error in Bird Eye View for location {location_id}: {str(e)}", exc_info=True)
                continue

        return processed_batches

    except Exception as e:
        logger.critical(f"Critical failure in Bird Eye View execution: {str(e)}", exc_info=True)
        raise
