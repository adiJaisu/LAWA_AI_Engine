import cv2
import numpy as np
from typing import Any, Dict, List, Tuple

from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.utils.ConfigReader import cfg

logging_config = LoggingConfig()


class PersonCountInsideCompartmentException(Exception):
    pass


class PersonCountInsideCompartment:
    """
    Person Count Inside Compartment using Head-based filtering
    """

    def __init__(self):
        self.logger = logging_config.setup_logging()

        # ---------------- CONFIG ----------------
        self.person_class_id = int(
            cfg.get_value_config(Constants.PERSON_COUNT_INSIDE_COMPARTMENT, Constants.PERSON_CLASS_ID)
        )

        self.confidence = float(
            cfg.get_value_config(Constants.PERSON_COUNT_INSIDE_COMPARTMENT, Constants.PERSON_COUNT_INSIDE_COMPARTMENT_CONFIDENCE)
        )

        self.resize_width = int(cfg.get_value_config(Constants.PERSON_COUNT_INSIDE_COMPARTMENT, "RESIZE_WIDTH"))
        self.resize_height = int(cfg.get_value_config(Constants.PERSON_COUNT_INSIDE_COMPARTMENT, "RESIZE_HEIGHT"))
        self.top_crop_ratio = float(cfg.get_value_config(Constants.PERSON_COUNT_INSIDE_COMPARTMENT, "TOP_CROP_RATIO"))
        self.bottom_crop_ratio = float(cfg.get_value_config(Constants.PERSON_COUNT_INSIDE_COMPARTMENT, "BOTTOM_CROP_RATIO"))
        self.min_box_height = int(cfg.get_value_config(Constants.PERSON_COUNT_INSIDE_COMPARTMENT, "MIN_BOX_HEIGHT"))
        self.duplicate_head_distance = int(cfg.get_value_config(Constants.PERSON_COUNT_INSIDE_COMPARTMENT, "DUPLICATE_HEAD_DISTANCE"))

        # ---------------- STATE ----------------
        self.current_count = 0

    # ---------------- DUPLICATE HEAD FILTER ----------------
    @staticmethod
    def remove_duplicate_heads(head_data, distance_thresh):
        filtered = []

        for data in head_data:
            x1, y1, x2, y2, hx, hy = data
            keep = True

            for f in filtered:
                _, _, _, _, fx, fy = f

                distance = np.linalg.norm(
                    np.array([hx, hy]) - np.array([fx, fy])
                )

                if distance < distance_thresh:
                    keep = False
                    break

            if keep:
                filtered.append(data)

        return filtered

    def detect(
        self,
        frame: np.ndarray,
        rois: List[List[List[int]]],
        detector: Any
    ) -> Tuple[np.ndarray, bool, Dict]:

        alert_event = False

        # ---------------- PREPROCESS ----------------
        frame = cv2.resize(frame, (self.resize_width, self.resize_height))

        h, w, _ = frame.shape

        top_crop = int(h * self.top_crop_ratio)
        bottom_crop = int(h * self.bottom_crop_ratio)

        roi_frame = frame[top_crop:bottom_crop, :]

        # ---------------- DETECTION ----------------
        results = detector.make_prediction(
            roi_frame,
            classes_id=[self.person_class_id],
            confidence=self.confidence
            
        )

        head_data = []

        for r in results:

            if r.boxes is None:
                continue

            for box in r.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # adjust coordinates
                y1 += top_crop
                y2 += top_crop

                box_h = y2 - y1

                if box_h < self.min_box_height:
                    continue

                # head estimation
                head_x = (x1 + x2) // 2
                head_y = y1 + int(box_h * 0.50)

                # blur head region
                head_roi = frame[y1:head_y, x1:x2]

                if head_roi.size > 0:
                    blurred = cv2.GaussianBlur(head_roi, (151, 151), 50)
                    frame[y1:head_y, x1:x2] = blurred

                head_data.append((x1, y1, x2, y2, head_x, head_y))

        # ---------------- FILTER ----------------
        filtered = self.remove_duplicate_heads(
            head_data,
            self.duplicate_head_distance
        )

        # ---------------- COUNT ----------------
        self.current_count = len(filtered)

        # ---------------- VISUALIZATION ----------------
        display_frame = frame.copy()

        for (x1, y1, x2, y2, hx, hy) in filtered:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(display_frame, (hx, hy), 6, (0, 255, 0), -1)

        cv2.putText(display_frame,
                    f"People Count: {self.current_count}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

        cv2.imshow("Person_count", display_frame)
        cv2.waitKey(1)

        return display_frame, alert_event, {
            "current_count": self.current_count
        }