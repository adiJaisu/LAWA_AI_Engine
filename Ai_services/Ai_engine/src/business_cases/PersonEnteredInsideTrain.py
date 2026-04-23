import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.utils.ConfigReader import cfg

class PersonEnteredInsideTrainException(Exception):
    pass

class PersonEnteredInsideTrain:
    """
    Person Entered Inside Train detection system.
    """

    def __init__(self):
        logging_config = LoggingConfig()
        self.logger = logging_config.setup_logging()
        self.logger.info("Initializing PersonEnteredInsideTrain detection class")
        
        # Load Configs
        self.person_class_id = int(cfg.get_value_config(Constants.PERSON_ENTERED_INSIDE_TRAIN, Constants.PERSON_CLASS_ID))
        self.confidence = float(cfg.get_value_config(Constants.PERSON_ENTERED_INSIDE_TRAIN, Constants.PERSON_ENTERED_INSIDE_TRAIN_CONFIDENCE))
        self.tracker_name = cfg.get_value_config(Constants.PERSON_ENTERED_INSIDE_TRAIN, Constants.TRACKER_NAME)
        
        # Event manager state mapped inside class
        self.was_on_platform = set()
        self.entered_ids = set()

    def update_active_ids(self, active_ids: List[int]):
         # Remove inactive ids to prevent memory leaks, similar to event_manager.clean_ids()
         self.was_on_platform.intersection_update(active_ids)
         self.entered_ids.intersection_update(active_ids)

    def detect(self, frame: np.ndarray, rois: List[List[List[int]]], detector: Any) -> Tuple[np.ndarray, bool, Dict]:
        """
        Runs the person entered inside train logic.
        """
        if not rois or len(rois) < 3:
            self.logger.warning("PersonEnteredInsideTrain requires at least 3 ROIs: Platform, Gate1, Gate2")
            return frame, False, {}

        # Assuming rois[0] is platform, rois[1] is gate1, rois[2] is gate2
        platform_polygon = np.array(rois[0], np.int32)
        gate1_polygon = np.array(rois[1], np.int32)
        gate2_polygon = np.array(rois[2], np.int32)

        # Draw zones
        cv2.polylines(frame, [platform_polygon], True, (255, 0, 0), 2)
        cv2.polylines(frame, [gate1_polygon], True, (0, 255, 255), 2)
        cv2.polylines(frame, [gate2_polygon], True, (255, 255, 0), 2)

        try:
            results = detector.make_prediction_with_tracking(
                frame=frame,
                classes_id=[self.person_class_id],
                confidence=self.confidence,
                tracker=self.tracker_name,
                iou=0.5
            )[0]
        except Exception as e:
            self.logger.error(f"Error during object tracking: {e}")
            raise PersonEnteredInsideTrainException(f"Tracking error: {e}")

        current_platform_ids = set()
        alert = False

        if results.boxes is None or results.boxes.id is None:
            platform_count = 0
            entered_count = len(self.entered_ids)
            return frame, alert, {"platform_count": platform_count, "entered_count": entered_count}

        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy()

        active_ids = []

        for box, track_id in zip(boxes, ids):
            track_id = int(track_id)
            active_ids.append(track_id)

            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            color = (180, 180, 180) # default grey

            inside_platform = cv2.pointPolygonTest(platform_polygon, (cx, cy), False)

            if inside_platform >= 0:
                current_platform_ids.add(track_id)
                self.was_on_platform.add(track_id)
                color = (0, 255, 0) # Green if on platform

            in_gate1 = cv2.pointPolygonTest(gate1_polygon, (cx, cy), False) >= 0
            in_gate2 = cv2.pointPolygonTest(gate2_polygon, (cx, cy), False) >= 0

            # Logic from user: was_on_platform and in_gate and not has_entered
            if (track_id in self.was_on_platform and (in_gate1 or in_gate2) and track_id not in self.entered_ids):
                self.entered_ids.add(track_id)
                alert = True

            if track_id in self.entered_ids:
                color = (0, 0, 255) # Red if entered

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        self.update_active_ids(active_ids)

        platform_count = len(current_platform_ids)
        entered_count = len(self.entered_ids)

        cv2.putText(frame, f"Entered: {entered_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"On Platform: {platform_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        status_dict = {
            "platform_count": platform_count,
            "entered_count": entered_count
        }
        cv2.imshow("Person Entered Inside Train Detection", frame)
        cv2.waitKey(1)
        
        return frame, alert, status_dict