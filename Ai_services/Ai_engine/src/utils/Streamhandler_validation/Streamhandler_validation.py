import datetime
from zoneinfo import ZoneInfo
from src.constant.constants import Constants
from src.utils.GlobalConfig import GlobalConfig
from src.constant.global_constant import VisionPipeline
from typing import Dict,Optional,Any
from src.utils.Logger import LoggingConfig
logging_config = LoggingConfig()
logger=logging_config.setup_logging()
class Validator:
    def __init__(self):
        pass
    """
    A class to validate messages for the StreamHandler.
    """
    

    # @staticmethod
    # def getFullFrameRoi(frameHeight,frameWidth):
    #     """ Helper function to generate full frame roi if roi is not given in metadata"""
    #     # Full-frame ROI fallback
    #     full_frame_roi = [[
    #         [Constants.ZERO, Constants.ZERO],
    #         [frameWidth - Constants.ONE, Constants.ZERO],
    #         [frameWidth - Constants.ONE, frameHeight - Constants.ONE],
    #         [Constants.ZERO, frameHeight - Constants.ONE]
    #     ]]

    #     return full_frame_roi

    def validate_polygons(self, polygons, cameraName, frame_shape, usecase=None):
        """
        Validate a list of polygons.

        Parameters:
        - polygons: list of list of points, e.g., [[[x1, y1], [x2, y2], ...], [...], ...]
        - frame_shape: tuple (height, width)
        - usecase: The usecase name to apply specific validation rules

        Returns:
        - valid_polygons: list of polygons that are valid and within the frame
        - invalid_indices: list of indices of polygons that were invalid
        """
        valid_polygons = []
        invalid_indices = []
        height, width = frame_shape
        
        min_points = Constants.TWO if usecase == Constants.IN_OUT_PERSON_COUNT_USECASE else Constants.THREE
        
        for idx, polygon in enumerate(polygons):
            # Check if polygon has at least min_points
            if len(polygon) < min_points:
                logger.error(f"Polygon at index {idx} has less than {min_points} points for camera {cameraName}")
                invalid_indices.append(idx)
                continue

            # Check all points are within frame bounds
            is_inside_frame = all(
                Constants.ZERO <= point[Constants.ZERO] < width and Constants.ZERO <= point[Constants.ONE] < height 
                for point in polygon
            )

            if not is_inside_frame:
                logger.error(f"Polygon at index {idx} contains points outside frame boundsfor {cameraName}")
                invalid_indices.append(idx)
                continue

            valid_polygons.append(polygon)

        return valid_polygons, invalid_indices

    def set_global_configuration(self,metadata: Dict[str, Any]) -> None:
        """
        Sets the global configuration for the application.

        Args:
            metadata (Dict[str, Any]): Configuration metadata to be set globally.
        """
        try:
            VisionPipeline.global_config = GlobalConfig()
            VisionPipeline.global_config.set_value(metadata)
        except Exception as e:
            logger.error(f"Failed to set global configuration: {e}")

    def validate_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validates a message against required fields and formats, modifying it in-place.
        
        Args:
            msg (Dict[str, Any]): The message to validate
            
        Returns:
            Optional[Dict[str, Any]]: Validated message with updated metadata structure,
                or None if validation fails
        """
        if not isinstance(msg, dict):
            logger.error(f"Invalid message format. Expected a dict, but got {type(msg)}")
            return None

        try:
            frame_meta = msg[Constants.FRAME_METADATA]
            cam_meta = msg[Constants.CAMERA_METADATA]
        except KeyError as e:
            logger.error(f"Missing required metadata field: {e}")
            return None

        required_frame_fields = [
            Constants.FRAME_ID, Constants.TIME_STAMP, Constants.FRAME_SIZE_H,
            Constants.FRAME_SIZE_W, Constants.FRAME, Constants.ROIS,
            Constants.RABBITMQ_SENT_TIMING, Constants.USECASE_NAME
        ]
        required_camera_fields = [
            Constants.CAMERA_ID, Constants.LOCATION_ID, Constants.CAMERA_NAME,
            Constants.CODEC, Constants.RESOLUTION, Constants.MODEL,
            Constants.CAMERA_HEIGHT, Constants.RTSP_URL
        ]

        if any(field not in frame_meta for field in required_frame_fields):
            missing = [f for f in required_frame_fields if f not in frame_meta]
            logger.error(f"Missing frame metadata fields: {missing}")
            return None

        if any(field not in cam_meta for field in required_camera_fields):
            missing = [f for f in required_camera_fields if f not in cam_meta]
            logger.error(f"Missing camera metadata fields: {missing}")
            return None

        cam_name = cam_meta[Constants.CAMERA_NAME]
        usecase = frame_meta[Constants.USECASE_NAME]

        strm_rois = frame_meta[Constants.ROIS]

        valid_polygons, _ = self.validate_polygons(
            strm_rois,
            cam_name,
            (frame_meta[Constants.FRAME_SIZE_H], frame_meta[Constants.FRAME_SIZE_W]),
            usecase=usecase
        )

        if not valid_polygons:
            logger.debug(f"No valid ROIs found in the message from camera {cam_meta[Constants.CAMERA_ID]}.")
            return None

        self.set_global_configuration(msg)
        mexico_time = datetime.datetime.now(datetime.timezone.utc).astimezone(ZoneInfo(Constants.TIME_ZONE_INFO))
        
        frame_meta[Constants.ROIS] = valid_polygons
        frame_meta[Constants.RABBITMQ_CONSUME_TIMING] = mexico_time.strftime('%Y-%m-%dT%H:%M:%S.%f')

        return msg