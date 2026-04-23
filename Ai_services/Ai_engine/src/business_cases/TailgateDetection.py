import cv2
import numpy as np
import time
from collections import deque
from typing import Any, Dict, List, Tuple
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.utils.ConfigReader import cfg
from src.utils.DatabaseManager import DatabaseManager

logging_config = LoggingConfig()

class TailgateDetectionException(Exception):
    pass

class TailgateDetection:
    """
    Advanced Tailgate Detection system integrating LED color monitoring, 
    punch-based access tokens, and database-backed employee verification.
    """

    def __init__(self):
        self.logger = logging_config.setup_logging()
        self.db = DatabaseManager()
        
        # Load Configs
        self.person_class_id = int(cfg.get_value_config(Constants.TAILGATE_DETECTION, Constants.PERSON_CLASS_ID))
        self.confidence = float(cfg.get_value_config(Constants.TAILGATE_DETECTION, Constants.TAILGATE_DETECTION_CONFIDENCE))
        self.tracker_name = cfg.get_value_config(Constants.TAILGATE_DETECTION, Constants.TRACKER_NAME)
        
        # Thresholds
        self.min_sustain_frames = int(cfg.get_value_config(Constants.TAILGATE_DETECTION, Constants.TAILGATE_MIN_SUSTAIN_FRAMES))
        self.min_punch_gap = float(cfg.get_value_config(Constants.TAILGATE_DETECTION, Constants.TAILGATE_MIN_PUNCH_GAP))
        self.roi_min_frames = int(cfg.get_value_config(Constants.TAILGATE_DETECTION, Constants.TAILGATE_ROI_MIN_FRAMES))
        self.green_min_frames = int(cfg.get_value_config(Constants.TAILGATE_DETECTION, Constants.TAILGATE_GREEN_MIN_FRAMES))
        self.global_odc = str(cfg.get_value_config(Constants.TAILGATE_DETECTION, Constants.TAILGATE_GLOBAL_ODC)).strip().upper()

        # LED State
        self.red_hist = deque(maxlen=10)
        self.green_hist = deque(maxlen=10)
        self.red_streak = 0
        self.green_streak = 0
        self.current_color = "RED"
        self.prev_color = "RED"
        
        # Token System
        self.allow_count = 0
        self.green_session_consumed = False
        self.last_punch_time = 0
        self.green_on_frames = 0
        self.tailgate_count_current_session = 0
        
        # Entry Tracking
        self.roi_entry_frames: Dict[int, int] = {}
        self.roi2_presence = set()
        self.entered_roi1 = set()
        self.counted_ids = set()
        self.invalid_ids = set()
        self.tailgate_ids = set()
        
        self.entry_count = 0
        self.status_text = "WAITING"
        self.status_color = (255, 255, 255)

    def _update_led_state(self, frame: np.ndarray, led_roi_poly: List[List[int]]):
        """Processes the LED ROI to determine color state."""
        if not led_roi_poly or len(led_roi_poly) < 3:
            return

        # Extract bounding box for LED ROI
        poly_arr = np.array(led_roi_poly, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(poly_arr)
        led_crop = frame[y:y+h, x:x+w]

        red_prob, green_prob = 0.0, 0.0
        if led_crop.size > 0:
            # Zoom and Analyze (Logic from prototype)
            zoom = cv2.resize(led_crop, None, fx=15, fy=15, interpolation=cv2.INTER_CUBIC)
            hsv = cv2.cvtColor(zoom, cv2.COLOR_BGR2HSV)
            bright_mask = hsv[:, :, 2] > 140

            if np.count_nonzero(bright_mask) > 10:
                hue = hsv[:, :, 0][bright_mask]
                sat = hsv[:, :, 1][bright_mask]

                red_mask = ((hue < 12) | (hue > 165)) & (sat > 60)
                green_mask = ((hue > 35) & (hue < 90)) & (sat > 60)

                total = np.count_nonzero(bright_mask)
                red_prob = np.count_nonzero(red_mask) / total
                green_prob = np.count_nonzero(green_mask) / total

        self.red_hist.append(red_prob)
        self.green_hist.append(green_prob)

        red_avg = np.mean(self.red_hist)
        green_avg = np.mean(self.green_hist)

        if green_avg > red_avg:
            self.green_streak += 1
            self.red_streak = 0
        else:
            self.red_streak += 1
            self.green_streak = 0

        self.prev_color = self.current_color
        if self.green_streak >= self.min_sustain_frames:
            self.current_color = "GREEN"
        elif self.red_streak >= self.min_sustain_frames:
            self.current_color = "RED"

        # Handle Transitions & Tokens
        if self.prev_color == "RED" and self.current_color == "GREEN" and self.green_streak >= self.min_sustain_frames:
            now = time.time()
            if now - self.last_punch_time > self.min_punch_gap:
                self.allow_count += 1
                self.green_session_consumed = False
                self.last_punch_time = now
                self.logger.info(f"PUNCH DETECTED: allow_count increased to {self.allow_count}")

        if self.prev_color != "RED" and self.current_color == "RED":
            self.green_session_consumed = False

        self.green_on_frames = (self.green_on_frames + 1) if self.current_color == "GREEN" else 0

    @staticmethod
    def _point_inside_polygon(point: Tuple[int, int], polygon: List[List[int]]) -> bool:
        return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

    def detect(self, frame: np.ndarray, rois: List[List[List[int]]], detector: Any) -> Tuple[np.ndarray, bool, Dict]:
        """
        Main detection loop for Tailgate logic.
        rois[0]: LED
        rois[1]: Entry ROI
        rois[2]: Presence/Side ROI (Optional)
        """
        # 1. Update LED State
        if len(rois) > 0:
            self._update_led_state(frame, rois[0])

        # 2. Tracking Inference
        results = detector.make_prediction_with_tracking(
            frame=frame,
            classes_id=[self.person_class_id],
            confidence=self.confidence,
            tracker=self.tracker_name,
            iou=0.35  # <--- Increased from default 0.5. Higher allowed overlap means two close people won't merge!
        )

        alert_event = False
        if results and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):
                track_id = int(track_id)
                x1, y1, x2, y2 = box
                # Feet point logic from prototype
                foot_point = (int((x1 + x2) / 2), int ((y1+y2) * 0.75))

                in_roi1 = self._point_inside_polygon(foot_point, rois[1]) if len(rois) > 1 else False
                in_roi2 = self._point_inside_polygon(foot_point, rois[2]) if len(rois) > 2 else False

                if track_id in self.counted_ids or track_id in self.invalid_ids:
                    continue

                if in_roi2:
                    self.roi2_presence.add(track_id)

                if in_roi1:
                    self.entered_roi1.add(track_id)
                    self.roi_entry_frames[track_id] = self.roi_entry_frames.get(track_id, 0) + 1

                    if self.roi_entry_frames[track_id] >= self.roi_min_frames:
                        self.counted_ids.add(track_id)
                        self.entry_count += 1
                        
                        # Validate Access
                        is_valid = (
                            self.allow_count > 0 and 
                            not self.green_session_consumed and 
                            self.green_on_frames >= self.green_min_frames
                        )

                        if is_valid:
                            self.allow_count -= 1
                            self.green_session_consumed = True
                            self.tailgate_count_current_session = 0
                            
                            
                            # Database Check
                            emp = self.db.get_employee_by_count(self.entry_count)
                            if emp:
                                emp_odc = str(emp.get("odc_no", "")).strip().upper()
                                if emp_odc == self.global_odc:
                                    self.status_text = "ACCESS OK: ODC MATCH"
                                    self.status_color = (0, 255, 0)
                                    self.db.log_access(emp, "SUCCESS_MATCH")
                                else:
                                    self.status_text = "ACCESS OK: ODC DIFFERENT"
                                    self.status_color = (0, 165, 255)
                                    self.db.log_access(emp, "SUCCESS_ODC_DIFF")
                        else:
                            # TAILGATE TRIGGERED
                            self.status_text = "TAILGATE DETECTED"
                            self.status_color = (0, 0, 255)
                            self.tailgate_ids.add(track_id)
                            alert_event = True
                            self.tailgate_count_current_session += 1
                            self.logger.warning(f"TAILGATE ALERT: ID {track_id} entered without valid token.")

                # Side exit/Invalid Entry logic
                if track_id in self.roi2_presence and not in_roi2:
                    if track_id not in self.entered_roi1:
                        self.invalid_ids.add(track_id)
                        self.status_text = "INVALID ENTRY"
                        self.status_color = (0, 0, 255)
                        self.logger.warning(f"INVALID ENTRY: ID {track_id} bypassed threshold.")
                    self.roi2_presence.discard(track_id)

        # 3. Visualization
        display_frame = frame.copy()
        
        # Draw Bounding Boxes for currently tracked people
        if results and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            for box, track_id in zip(boxes, ids):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                
                # Use RED for tailgaters, WHITE for others
                color = (0, 0, 255) if track_id in self.tailgate_ids or track_id in self.invalid_ids else (0, 255, 0)
                # If they are just "Access OK", maybe use green? 
                # For now, let's stick to RED for offenders and WHITE for others.
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(display_frame,(int((x1 + x2) / 2), int ((y1+y2) * 0.75)), 4 , color, -1)  # ID marker
                cv2.putText(display_frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw Zones
        for i, poly in enumerate(rois):
            color = (255, 255, 255) if i == 1 else (200, 200, 200)
            cv2.polylines(display_frame, [np.array(poly, np.int32)], True, color, 2)

        
        # Draw status overlays
        cv2.putText(display_frame, f"LED: {self.current_color}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, self.status_color, 2)
        cv2.putText(display_frame, f"Persons Entered: {self.entry_count}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Tailgates (Session): {self.tailgate_count_current_session}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, self.status_text, (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.status_color, 3)
        cv2.imshow("Tailgate Detection", display_frame)
        cv2.waitKey(1)
        print(f"=================================================================={alert_event}==================================================================")
        return display_frame, alert_event, {
            "status": self.status_text,
            "tokens": self.allow_count,
            "tailgate_count": self.tailgate_count_current_session,
            "entry_count": self.entry_count
        }

 