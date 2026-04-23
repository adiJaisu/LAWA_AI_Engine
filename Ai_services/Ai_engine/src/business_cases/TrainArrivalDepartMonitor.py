import cv2
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Tuple
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.utils.ConfigReader import cfg

logging_config = LoggingConfig()

class TrainArrivalDepartMonitorException(Exception):
    pass

class TrainArrivalDepartMonitor:
    """
    Train Arrival Depart Monitor system decoupled from visualization.
    """

    def __init__(self):
        self.logger = logging_config.setup_logging()
        
        # Load Configs
        self.train_class_id = int(cfg.get_value_config(Constants.TRAIN_ARRIVAL_DEPART_MONITOR, Constants.TRAIN_CLASS_ID))
        self.confidence = float(cfg.get_value_config(Constants.TRAIN_ARRIVAL_DEPART_MONITOR, Constants.TRAIN_ARRIVAL_DEPART_MONITOR_CONFIDENCE))
        
        # Thresholds
        self.fps = 10 
        self.min_train_presence_frames = 4 * self.fps # MIN_TRAIN_PRESENCE_SEC = 4
        self.missing_frames_threshold = 40
        self.post_departure_ignore_frames = 100
        
        # State variables
        self.frame_count = 0
        self.train_present = False
        self.missing_frames = 0
        self.presence_counter = 0
        self.first_frame_checked = False
        self.post_departure_counter = 0

        self.current_arrival_frame = None
        self.current_departure_frame = None
        self.start_clock_time = None
        
    def frame_to_clock(self, frame_no):
        if not self.start_clock_time:
            return "--:--:--"
        import datetime as dt
        elapsed_seconds = frame_no / self.fps
        real_time = self.start_clock_time + dt.timedelta(seconds=elapsed_seconds)
        return real_time.strftime("%H:%M:%S")

    def format_dwell(self, frames):
        secs = int(frames / self.fps)
        if secs < 60:
            return f"{secs} sec"
        else:
            m = secs // 60
            s = secs % 60
            return f"{m} min {s} sec"

    def detect(self, frame: np.ndarray, rois: List[List[List[int]]], detector: Any, timestamp: str = None) -> Tuple[np.ndarray, bool, Dict]:
        self.frame_count += 1
        
        if self.start_clock_time is None:
            if timestamp:
                try:
                    self.start_clock_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    self.start_clock_time = datetime.now()
            else:
                self.start_clock_time = datetime.now()

        results = detector.make_prediction(
            frame=frame,
            classes_id=[self.train_class_id],
            confidence=self.confidence
        )

        train_detected = False
        display_frame = frame.copy() # Keeps the same decoupled approach as tailgate

        detections = []
        if results and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes.xyxy is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            if len(boxes) > 0:
                train_detected = True
                for box in boxes:
                    detections.append(box.tolist())
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Train", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        alert_event = False

        if self.post_departure_counter > 0:
            self.post_departure_counter -= 1
            train_detected = False

        if not self.first_frame_checked:
            self.first_frame_checked = True
            if train_detected:
                self.train_present = True
                self.current_arrival_frame = self.frame_count

        elif train_detected and not self.train_present:
            self.presence_counter += 1
            if self.presence_counter >= self.min_train_presence_frames:
                self.train_present = True
                self.current_arrival_frame = self.frame_count - self.presence_counter + 1
                self.current_departure_frame = None
                self.presence_counter = 0
        else:
            self.presence_counter = 0

        if self.train_present and not train_detected:
            self.missing_frames += 1
        else:
            self.missing_frames = 0

        if self.train_present and self.missing_frames > self.missing_frames_threshold:
            dep_frame = self.frame_count - self.missing_frames_threshold
            self.train_present = False
            self.current_departure_frame = dep_frame
            self.missing_frames = 0
            self.post_departure_counter = self.post_departure_ignore_frames
            alert_event = True
            self.logger.info(f"TRAIN DEPARTED. Arrival: {self.current_arrival_frame}, Departure: {self.current_departure_frame}")

        dwell_frames = 0
        if self.current_arrival_frame is not None:
            if self.train_present:
                dwell_frames = self.frame_count - self.current_arrival_frame
            elif self.current_departure_frame is not None:
                dwell_frames = self.current_departure_frame - self.current_arrival_frame

        y0 = 30
        if self.current_arrival_frame is not None:
            arr_str = self.frame_to_clock(self.current_arrival_frame)
            cv2.putText(display_frame, f"Arrival Time: {arr_str}", (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            dwell_str = self.format_dwell(dwell_frames)
            cv2.putText(display_frame, f"Time at Platform: {dwell_str}", (30, y0 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if self.current_departure_frame is not None:
            dep_str = self.frame_to_clock(self.current_departure_frame)
            cv2.putText(display_frame, f"Departure Time: {dep_str}", (30, y0 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        status_dict = {
            "train_present": self.train_present,
            "arrival_frame": self.current_arrival_frame,
            "departure_frame": self.current_departure_frame,
            "dwell_frames": dwell_frames,
            "detections": detections,
            "arrival_str": self.frame_to_clock(self.current_arrival_frame) if self.current_arrival_frame is not None else None,
            "departure_str": self.frame_to_clock(self.current_departure_frame) if self.current_departure_frame is not None else None,
            "dwell_str": self.format_dwell(dwell_frames) if self.current_arrival_frame is not None else None
        }
        cv2.imshow("Train Arrival/Departure Monitor", display_frame)
        cv2.waitKey(1)

        return display_frame, alert_event, status_dict

