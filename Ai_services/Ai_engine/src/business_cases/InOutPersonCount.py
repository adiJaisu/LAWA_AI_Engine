import cv2
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.utils.ConfigReader import cfg

class InOutPersonCountException(Exception):
    pass

def side(point: Tuple[int, int], line_p1: Tuple[int, int], line_p2: Tuple[int, int]) -> int:
    return (point[0] - line_p1[0]) * (line_p2[1] - line_p1[1]) - (point[1] - line_p1[1]) * (line_p2[0] - line_p1[0])

class InOutPersonCount:
    def __init__(self):
        logging_config = LoggingConfig()
        self.logger = logging_config.setup_logging()
        self.logger.info("Initializing InOutPersonCount detection class")
        self.track_history = defaultdict(list)
        self.crossed_A = set()
        self.crossed_B = set()
        self.total_in = Constants.ZERO
        self.total_out = Constants.ZERO
        
        # Load configurable parameters like other pipelines
        self.tracker_name = cfg.get_value_config(Constants.IN_OUT_PERSON_COUNT, Constants.TRACKER_NAME)
        self.conf = float(cfg.get_value_config(Constants.IN_OUT_PERSON_COUNT, Constants.IN_OUT_CONFIDENCE))
        self.PERSON_CLASS_ID = int(cfg.get_value_config(Constants.IN_OUT_PERSON_COUNT, Constants.PERSON_CLASS_ID))

    def detect(self, frame: np.ndarray, lines: List[List[Tuple[int, int]]], detector: Any) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Runs the people counting logic.
        Returns:
            alert: Always False for in_out_person_count
            unannotated_frame: Raw frame
            live_status: Metadata about the current count and detections
        """
        if len(lines) < Constants.TWO or len(lines[Constants.ZERO]) < Constants.TWO or len(lines[Constants.ONE]) < Constants.TWO:
            # Lines are not properly format
            return False, frame, {}

        # lineA is IN, lineB is OUT
        lineA_p1 = tuple(lines[Constants.ZERO][Constants.ZERO])
        lineA_p2 = tuple(lines[Constants.ZERO][Constants.ONE])
        lineB_p1 = tuple(lines[Constants.ONE][Constants.ZERO])
        lineB_p2 = tuple(lines[Constants.ONE][Constants.ONE])

        try:
            results = detector.make_prediction_with_tracking(
                frame = frame,
                classes_id =[self.PERSON_CLASS_ID],
                confidence = self.conf,
                tracker=self.tracker_name,
                iou=Constants.ZERO_POINT_FIVE
            )[Constants.ZERO]
        except Exception as e:
            self.logger.error(f"Error during object tracking: {e}")
            raise InOutPersonCountException(f"Tracking error: {e}")

        detections_info = []

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy
            ids = results.boxes.id

            for box, tid in zip(boxes, ids):
                tid = int(tid.item())
                x1, y1, x2, y2 = map(int, box)

                cx = (x1 + x2) // Constants.TWO
                cy = int(y2 * Constants.ZERO_POINT_NINE_EIGHT)
                foot = (cx, cy)

                self.track_history[tid].append(foot)

                if len(self.track_history[tid]) > Constants.TWENTY:
                    self.track_history[tid].pop(Constants.ZERO)

                hist = self.track_history[tid]

                if len(hist) > Constants.TWO:
                    prev = hist[Constants.MINUS_TWO]
                    curr = hist[Constants.MINUS_ONE]

                    prevA = side(prev, lineA_p1, lineA_p2)
                    currA = side(curr, lineA_p1, lineA_p2)

                    prevB = side(prev, lineB_p1, lineB_p2)
                    currB = side(curr, lineB_p1, lineB_p2)

                    if prevA * currA < Constants.ZERO:
                        self.crossed_A.add(tid)

                    if prevB * currB < Constants.ZERO:
                        self.crossed_B.add(tid)

                    if tid in self.crossed_A and tid in self.crossed_B:
                        if hist[Constants.ZERO][Constants.ONE] < hist[Constants.MINUS_ONE][Constants.ONE]:
                            self.total_in += Constants.ONE
                        else:
                            self.total_out += Constants.ONE

                        self.crossed_A.discard(tid)
                        self.crossed_B.discard(tid)
                        self.track_history[tid] = []

                detections_info.append({
                    "id": tid,
                    "bbox": [x1, y1, x2, y2]
                })
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), Constants.TWO)
                cv2.putText(frame, f"ID: {tid}", (x1, y1 - Constants.TEN), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), Constants.TWO)   
                cv2.line(frame, lineA_p1, lineA_p2, (255, 0, 0), Constants.TWO)
                cv2.line(frame, lineB_p1, lineB_p2, (0, 0, 255), Constants.TWO) 
                cv2.putText(frame, f"IN: {self.total_in}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), Constants.TWO)
                cv2.putText(frame, f"OUT: {self.total_out}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), Constants.TWO)
                cv2.circle(frame, foot, 5, (0, 255, 255), -1)
                cv2.imshow("In/Out Person Count", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        live_status = {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "detections": detections_info
        }

        self.logger.debug(f"Live IN/OUT Status: {live_status}")
        return False, frame, live_status
