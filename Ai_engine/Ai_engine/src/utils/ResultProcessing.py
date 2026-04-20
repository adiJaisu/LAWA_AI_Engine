from typing import List, Dict, Any
import time
import cv2
import os
import base64
import datetime
from zoneinfo import ZoneInfo
import numpy as np
from src.utils.ConfigReader import cfg
from src.constant.constants import Constants
from src.constant.global_constant import VisionPipeline
from src.utils.Logger import log_time
from src.utils.Logger import LoggingConfig,log_time
from src.utils.FrameQualityChecker import FrameQualityChecker
from src.Exception.Exception import  FrameProcessingError
logging_config = LoggingConfig()
quality_checker = FrameQualityChecker()
logger = logging_config.setup_logging()
# 原代码保留 / Original turbojpeg usage:
# from turbojpeg import TurboJPEG
# jpeg = TurboJPEG()

import turbojpeg
try:
    jpeg = turbojpeg.TurboJPEG()
except RuntimeError:
    logger.warning("TurboJPEG C-library not found. Falling back to OpenCV natively for JPEG encoding.")
    jpeg = None
 
@log_time("Time taken to preprocess results",True)
def process_detection_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Processes detection results, saves frames to video files for debugging, and prepares event evidence for alert frames."""    
    try:
        event_manager_evidence: List[Dict[str, Any]] = []
        need_to_send = False

        if results:
            for processed_msg in results:
                if processed_msg[Constants.FRAME_METADATA].get(Constants.ALERT, False):
                    need_to_send =True
                    break

            if need_to_send:
                for processed_msg in results:
                    processed_msg[Constants.FRAME_QUALITY] = False
                    evidence_msg = _prepare_frame_data(processed_msg=processed_msg)
                    event_manager_evidence.append(evidence_msg)
                return event_manager_evidence
            else:
                return []
        else:
            return []
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")

def _prepare_frame_data(processed_msg: Dict[str, Any]) -> Dict[str, Any]:
    """Prepares and cleans frame data for the vision pipeline."""
    try:
        frame_metadata = processed_msg.get(Constants.FRAME_METADATA, {})
        detections = frame_metadata.get(Constants.DETECTION_DETAILS, [])

        for det in detections:
            conf = det.get(Constants.CONF)
            if isinstance(conf, np.float32):
                det[Constants.CONF] = float(conf)

        if frame_metadata[Constants.USECASE_NAME] == Constants.SPEEDING_USECASE and  not frame_metadata[Constants.ALERT]:
            return None
        elif frame_metadata[Constants.USECASE_NAME] == Constants.SPEEDING_USECASE and  frame_metadata[Constants.ALERT]:
            frames = frame_metadata.get(Constants.FRAME)
            raw_frames = frame_metadata.get(Constants.RAW_FRAME)
            
            # Original Speeding Logic with aggressive TurboJPEG usage
            # frame_list = [base64.b64encode(jpeg.encode(img,quality=int(cfg.get_env_config(Constants.FRAME_SENT_QUALITY_TO_EVENT_MANAGER)))).decode(Constants.UTF_8_ENCODING)for img in frames]
            # raw_frame_list = [base64.b64encode(jpeg.encode(img,quality=int(cfg.get_env_config(Constants.FRAME_SENT_QUALITY_TO_EVENT_MANAGER)))).decode(Constants.UTF_8_ENCODING)for img in raw_frames]
            
            q = int(cfg.get_env_config(Constants.FRAME_SENT_QUALITY_TO_EVENT_MANAGER) or 45)
            
            if jpeg is not None:
                frame_list = [base64.b64encode(jpeg.encode(img, quality=q)).decode(Constants.UTF_8_ENCODING) for img in frames]
                raw_frame_list = [base64.b64encode(jpeg.encode(img, quality=q)).decode(Constants.UTF_8_ENCODING) for img in raw_frames]
            else:
                frame_list = [base64.b64encode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode(Constants.UTF_8_ENCODING) for img in frames]
                raw_frame_list = [base64.b64encode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode(Constants.UTF_8_ENCODING) for img in raw_frames]

            processed_msg[Constants.FRAME_METADATA][Constants.FRAME] = frame_list
            processed_msg[Constants.FRAME_METADATA][Constants.RAW_FRAME] = raw_frame_list

        else:
            frame = frame_metadata.get(Constants.FRAME)
            raw_frame = frame_metadata.get(Constants.RAW_FRAME)

            if frame is None or raw_frame is None:
                raise FrameProcessingError("Frame or raw_frame missing from metadata")

            # Original Single Frame Encoding Logic
            # encoded_frame = base64.b64encode(jpeg.encode(frame, quality=45)).decode(Constants.UTF_8)
            # encoded_raw_frame = base64.b64encode(jpeg.encode(raw_frame, quality=45)).decode(Constants.UTF_8)

            if jpeg is not None:
                encoded_frame = base64.b64encode(jpeg.encode(frame, quality=45)).decode(Constants.UTF_8)
                encoded_raw_frame = base64.b64encode(jpeg.encode(raw_frame, quality=45)).decode(Constants.UTF_8)
            else:
                encoded_frame = base64.b64encode(cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 45])[1]).decode(Constants.UTF_8)
                encoded_raw_frame = base64.b64encode(cv2.imencode('.jpg', raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 45])[1]).decode(Constants.UTF_8)

            processed_msg[Constants.FRAME_METADATA][Constants.FRAME] = encoded_frame
            processed_msg[Constants.FRAME_METADATA][Constants.RAW_FRAME] = encoded_raw_frame

        processed_msg[Constants.FRAME_METADATA][Constants.USECASE_TIMESTAMP] = datetime.datetime.now(ZoneInfo(Constants.TIME_ZONE_INFO)).strftime(Constants.TIME_ZONE_FORMAT)
        return processed_msg

    except FrameProcessingError as e:
        logger.error(f"[VISION STREAM] FrameProcessingError: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"[VISION STREAM] Unexpected error preparing frame data: {str(e)}")
        return None

@log_time("Time taken to sent one batch to event manager.",True)
def send_msg_to_event_manager(
    event_manager_evidence: List[Dict[str, Any]]
) -> None:
    """
    Sends a batch of processed event evidence to the event manager queue.

    This function pushes the provided event evidence list to the event manager
    via RabbitMQ, if the queue is configured. It logs the number of events sent
    and any errors encountered during the process.

    Args:
        event_manager_evidence (List[Dict[str, Any]]): List of event evidence dictionaries to send.

    Returns:
        None
    """
    try:
        event_manager_queue = os.environ.get(Constants.EVENT_MANAGER_QUEUE)
        if event_manager_queue:
            VisionPipeline.rabbitmq_service.send_messages_batch(
                event_manager_queue,
                event_manager_evidence
            )
            logger.debug(f"Sent {len(event_manager_evidence)} events to event manager queue '{event_manager_queue}'.")
        else:
            logger.warning("Event manager queue environment variable not set. Cannot send events.")
    except Exception as e:
        logger.error(f"Error sending event manager evidence: {str(e)}")