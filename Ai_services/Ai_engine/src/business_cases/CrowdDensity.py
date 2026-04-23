import cv2
import numpy as np
from collections import deque
from typing import Any, Dict, List, Tuple
from scipy.ndimage import gaussian_filter

from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.utils.ConfigReader import cfg

logging_config = LoggingConfig()


class CrowdDensity:
    """
    Crowd Density + Heatmap System
    Refactored into existing CrowdDensity pipeline format.
    """

    def __init__(self):
        self.logger = logging_config.setup_logging()

        # ---------------- CONFIG ----------------
        self.person_class_id = int(
            cfg.get_value_config(Constants.CROWD_DENSITY, Constants.PERSON_CLASS_ID)
        )
        self.confidence = float(
            cfg.get_value_config(Constants.CROWD_DENSITY, Constants.CROWD_DENSITY_CONFIDENCE)
        )
        self.tracker_name = cfg.get_value_config(
            Constants.CROWD_DENSITY, Constants.TRACKER_NAME
        )

        self.heatmap_resolution = int(
            cfg.get_value_config(Constants.CROWD_DENSITY, "HEATMAP_RESOLUTION")
        )
        self.heatmap_decay = float(
            cfg.get_value_config(Constants.CROWD_DENSITY, "HEATMAP_DECAY")
        )

        self.history_size = int(
            cfg.get_value_config(Constants.CROWD_DENSITY, "CROWD_HISTORY")
        )

        # ---------------- STATE ----------------
        self.current_crowd_count = 0
        self.peak_crowd_count = 0

        self.density_history = deque(maxlen=self.history_size)

        self.heatmap = np.zeros(
            (self.heatmap_resolution, self.heatmap_resolution),
            dtype=np.float32
        )

        self.frame_width = None
        self.frame_height = None

        self.status_text = "NORMAL"
        self.status_color = (255, 255, 255)

    @staticmethod
    def _get_center(box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _update_heatmap(self, tracks: List[Tuple[int, int, int, int, int]]):
        """Update heatmap with decay"""
        try:
            self.current_crowd_count = len(tracks)
            self.peak_crowd_count = max(self.peak_crowd_count, self.current_crowd_count)

            self.density_history.append(self.current_crowd_count)

            # Apply decay
            self.heatmap *= self.heatmap_decay

            for x1, y1, x2, y2, _ in tracks:
                cx, cy = self._get_center((x1, y1, x2, y2))

                hx = int((cx / self.frame_width) * self.heatmap_resolution)
                hy = int((cy / self.frame_height) * self.heatmap_resolution)

                hx = max(0, min(self.heatmap_resolution - 1, hx))
                hy = max(0, min(self.heatmap_resolution - 1, hy))

                self.heatmap[hy, hx] += 1

        except Exception:
            self.logger.exception("Heatmap update failed")

    def _draw_heatmap(self, frame, tracks):
        """Overlay heatmap + bounding boxes"""
        try:
            smoothed = gaussian_filter(self.heatmap, sigma=2)

            if smoothed.max() > 0:
                normalized = smoothed / smoothed.max()
            else:
                normalized = smoothed

            heatmap_resized = cv2.resize(
                normalized,
                (self.frame_width, self.frame_height)
            )

            heatmap_colored = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            frame = cv2.addWeighted(frame, 0.5, heatmap_colored, 0.5, 0)

            # Draw boxes
            for x1, y1, x2, y2, tid in tracks:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(
                    frame,
                    f"ID:{int(tid)}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

            return frame

        except Exception:
            self.logger.exception("Frame drawing failed")
            return frame

    def detect(
        self,
        frame: np.ndarray,
        rois: List[List[List[int]]],
        detector: Any
    ) -> Tuple[np.ndarray, bool, Dict]:

        """
        Main detection loop (adapted to original format)
        """

        if self.frame_width is None:
            self.frame_height, self.frame_width = frame.shape[:2]

        tracks = []
        alert_event = False

        # ---------------- DETECTION ----------------
        results = detector.make_prediction_with_tracking(
            frame=frame,
            classes_id=[self.person_class_id],
            confidence=self.confidence,
            tracker=self.tracker_name,
            iou=0.35
        )

        if results and len(results) > 0 and hasattr(results[0], "boxes"):

            if results[0].boxes.id is not None:

                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()

                for box, tid in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    tracks.append((x1, y1, x2, y2, int(tid)))

        # ---------------- UPDATE ----------------
        self._update_heatmap(tracks)

        # ---------------- STATUS LOGIC ----------------
        if self.current_crowd_count > int(
            cfg.get_value_config(Constants.CROWD_DENSITY, Constants.CROWD_LIMIT)
        ):
            self.status_text = "HIGH CROWD"
            self.status_color = (0, 0, 255)
            alert_event = True
        else:
            self.status_text = "NORMAL"
            self.status_color = (0, 255, 0)

        # ---------------- DRAW ----------------
        display_frame = self._draw_heatmap(frame.copy(), tracks)

        # Overlay text
        cv2.putText(display_frame, f"Count: {self.current_crowd_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.status_color, 2)

        cv2.putText(display_frame, f"Peak: {self.peak_crowd_count}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(display_frame, self.status_text, (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.status_color, 3)

        cv2.imshow("Crowd Density", display_frame)
        cv2.waitKey(1)

        return display_frame, alert_event, {
            "current_count": self.current_crowd_count,
            "peak_count": self.peak_crowd_count,
            "status": self.status_text
        }