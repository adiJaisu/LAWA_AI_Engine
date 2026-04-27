import cv2
import os
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.utils.ConfigReader import cfg
from src.detectors.custom_tracker import CustomPersonTracker

class RestroomPersonTrackingException(Exception):
    pass

class RestroomPersonTracking:
    """
    Restroom Person Tracking detection system.
    """

    def __init__(self, detector):
        logging_config = LoggingConfig()
        self.logger = logging_config.setup_logging()
        self.logger.info("Initializing RestroomPersonTracking detection class")
        
        # Load Configs
        self.person_class_id = int(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.PERSON_CLASS_ID))
        self.confidence = float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.RESTROOM_PERSON_TRACKING_CONFIDENCE))
        self.alert_threshold = float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.RESTROOM_ALERT_THRESHOLD))
        
        reid_params = {
            "model_name": cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.REID_MODEL_NAME),
            "num_classes": int(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.REID_NUM_CLASSES)),
            "image_size": tuple(map(int, cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.REID_IMAGE_SIZE).split(","))),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "device": str(detector.device)
        }
        
        tracking_params = {
            "track_thresh": float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.TRACK_THRESH)),
            "track_buffer": int(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.TRACK_BUFFER)),
            "match_thresh": float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.MATCH_THRESH)),
            "min_tracklet_len": int(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.MIN_TRACKLET_LEN)),
            "embedding_weight": float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.EMBEDDING_WEIGHT)),
            "iou_weight": float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.IOU_WEIGHT))
        }
        
        identity_params = {
            "threshold": float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.IDENTITY_THRESHOLD)),
            "avg_threshold": float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.IDENTITY_AVG_THRESHOLD)),
            "keep_threshold": float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.IDENTITY_KEEP_THRESHOLD)),
            "color_threshold": float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.IDENTITY_COLOR_THRESHOLD)),
            "reactivation_window": int(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.IDENTITY_REACTIVATION_WINDOW)),
            "reactivation_centroid_thresh": float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.IDENTITY_REACTIVATION_CENTROID_THRESH)),
            "reactivation_color_thresh": float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.IDENTITY_REACTIVATION_COLOR_THRESH)),
            "long_absence_window": int(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.IDENTITY_LONG_ABSENCE_WINDOW)),
            "long_centroid_thresh": float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.IDENTITY_LONG_CENTROID_THRESH)),
            "long_color_thresh": float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.IDENTITY_LONG_COLOR_THRESH)),
            "max_embeddings": int(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.IDENTITY_MAX_EMBEDDINGS)),
            "max_color_hists": int(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.IDENTITY_MAX_COLOR_HISTS)),
            "score_centroid_weight": float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.SCORE_CENTROID_WEIGHT)),
            "score_avg_weight": float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.SCORE_AVG_WEIGHT)),
            "score_color_weight": float(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, Constants.SCORE_COLOR_WEIGHT))
        }
        
        self.tracker = CustomPersonTracker(detector, tracking_params, identity_params, reid_params, self.person_class_id)
        
        # Paths
        self.identities_dir = Path(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, "IDENTITY_SAVE_DIR")) if cfg.obj_config.has_option(Constants.RESTROOM_PERSON_TRACKING, "IDENTITY_SAVE_DIR") else Path("identities")
        self.alerts_dir = Path(cfg.get_value_config(Constants.RESTROOM_PERSON_TRACKING, "ALERTS_DIR")) if cfg.obj_config.has_option(Constants.RESTROOM_PERSON_TRACKING, "ALERTS_DIR") else Path("alerts")
        
        # State
        self.id_last_inside = {}
        self.id_timers = {}
        self.id_alerted = {}
        self.id_zone_state = {}
        self.id_has_entered = {}
        self.entry_count = 0
        self.exit_count = 0
        self.max_entry_count = 0
        self.max_exit_count = 0
        self.alert_snapshot_index = 1
        
        # Simulated time
        self.frame_count = 0
        self.fps = Constants.FPS

    def point_in_polygon(self, point, polygon):
        if len(polygon) < 3:
            return False
        pts = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(pts, point, False) >= 0

    def is_in_zone(self, box, zone_polygon):
        x1, y1, x2, y2 = box
        bottom_center_x = (x1 + x2) // 2
        bottom_center_y = y2
        return self.point_in_polygon((bottom_center_x, bottom_center_y), zone_polygon)

    def copy_latest_identity_crop(self, identity_id: int) -> Path | None:
        """
        Copies the latest or 4th available crop for an identity to the alerts directory.
        """
        person_dir = self.identities_dir / f"person_{identity_id}"
        if not person_dir.exists():
            return None

        crop_files = list(person_dir.glob("frame_*.jpg"))
        if not crop_files:
            crop_files = list(person_dir.glob("*.jpg"))
        if not crop_files:
            return None

        # Sort by creation time
        sorted_files = sorted(crop_files, key=lambda p: p.stat().st_mtime)
        if len(sorted_files) >= 4:
            chosen_file = sorted_files[3]  # Prefer 4th frame if available
        else:
            chosen_file = sorted_files[-1]  # Fallback to latest

        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        dest_filename = f"alert_{identity_id}_{self.alert_snapshot_index}{chosen_file.suffix}"
        dest_path = self.alerts_dir / dest_filename

        try:
            shutil.copy2(chosen_file, dest_path)
            self.alert_snapshot_index += 1
            return dest_path
        except OSError as exc:
            self.logger.warning(f"Failed to copy crop for ID {identity_id}: {exc}")
            return None

    def detect(self, frame: np.ndarray, rois: List[List[List[int]]], detector: Any) -> Tuple[np.ndarray, bool, Dict]:
        """
        Runs the restroom person tracking logic with enhanced logic and visualization.
        """
        if not rois or len(rois) < 3:
            self.logger.warning("RestroomPersonTracking requires at least 3 ROIs: Inside Zone, Zone A, Zone B")
            return frame, False, {}

        # Assumes rois[0] is detection inside zone, rois[1] is zone A, rois[2] is zone B
        inside_zone_poly = np.array(rois[0], np.int32)
        zone_a_poly = np.array(rois[1], np.int32)
        zone_b_poly = np.array(rois[2], np.int32)

        # Draw zones
        cv2.polylines(frame, [inside_zone_poly], True, (0, 255, 0), 2)
        cv2.polylines(frame, [zone_a_poly], True, (255, 0, 0), 2)
        cv2.polylines(frame, [zone_b_poly], True, (0, 0, 255), 2)

        self.frame_count += 1
        current_time = self.frame_count / self.fps
        has_new_alert = False

        try:
            boxes, ids = self.tracker.process_frame(frame, [self.person_class_id], self.confidence, device=detector.device)
        except Exception as e:
            self.logger.error(f"Error during custom object tracking: {e}")
            raise RestroomPersonTrackingException(f"Tracking error: {e}")

        ids_detected = set()

        for box, pid in zip(boxes, ids):
            ids_detected.add(pid)
            x1, y1, x2, y2 = box

            # Logic for Entry/Exit detection (Zone A/B)
            bottom_center_x = (x1 + x2) // 2
            bottom_center_y = y2
            pink_dot = (bottom_center_x, bottom_center_y)

            in_zone_a = self.point_in_polygon(pink_dot, zone_a_poly)
            in_zone_b = self.point_in_polygon(pink_dot, zone_b_poly)

            current_zone = None
            if in_zone_a:
                current_zone = "A"
            elif in_zone_b:
                current_zone = "B"

            previous_zone = self.id_zone_state.get(pid)

            if current_zone is not None and current_zone != previous_zone and previous_zone is not None:
                if previous_zone == "B" and current_zone == "A":
                    self.entry_count += 1
                    self.max_entry_count = max(self.max_entry_count, self.entry_count)
                    self.id_has_entered[pid] = True
                    self.logger.info(f"ID {pid} entered through Zone B -> A. Current entry count: {self.entry_count}")
                elif previous_zone == "A" and current_zone == "B":
                    self.exit_count += 1
                    self.max_exit_count = max(self.max_exit_count, self.exit_count)
                    if self.id_has_entered.get(pid, False):
                        self.entry_count = max(0, self.entry_count - 1)
                        self.id_has_entered[pid] = False
                    self.logger.info(f"ID {pid} exited through Zone A -> B. Current entry count: {self.entry_count}")

            if current_zone is not None:
                self.id_zone_state[pid] = current_zone

            # Visualization: Bounding Box
            color = (0, 255, 0)
            thickness = 2
            if self.id_alerted.get(pid, False):
                color = (0, 0, 255) # Red for alert
                thickness = 4
                # Padded bbox for extra visibility
                pad = 10
                rx1 = max(0, x1 - pad)
                ry1 = max(0, y1 - pad)
                rx2 = min(frame.shape[1] - 1, x2 + pad)
                ry2 = min(frame.shape[0] - 1, y2 + pad)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"ID:{pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Timer logic: check if person is inside detection zone
            in_inside_zone = self.is_in_zone(box, inside_zone_poly)
            self.id_last_inside[pid] = in_inside_zone

            if in_inside_zone:
                # If they are currently inside, they shouldn't have an active timer
                if pid in self.id_timers:
                    self.id_timers.pop(pid)

        # Start timers for persons who were inside but are no longer detected
        for pid, was_inside in list(self.id_last_inside.items()):
            if pid not in ids_detected and was_inside:
                if pid not in self.id_timers:
                    self.id_timers[pid] = current_time
                    self.id_alerted[pid] = False

        # Process active timers and trigger alerts
        active_timers = []
        for pid, start_time in list(self.id_timers.items()):
            elapsed = max(0.0, current_time - start_time)
            active_timers.append((pid, elapsed))

            if not self.id_alerted.get(pid, False) and elapsed >= self.alert_threshold:
                self.id_alerted[pid] = True
                has_new_alert = True
                self.logger.warning(f"[ALERT] ID {pid} exceeded occupancy threshold of {self.alert_threshold}s")
                
                # Copy identity crop for the alert
                saved_crop = self.copy_latest_identity_crop(pid)
                if saved_crop:
                    self.logger.info(f"[ALERT] Saved identity crop for ID {pid} at {saved_crop}")

        # Visualization: Timer Display (Top Right)
        y_offset = 30
        for pid, elapsed in active_timers:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            timer_text = f"ID {pid}: {minutes:02d}:{seconds:02d}"
            
            c = (0, 0, 255) if self.id_alerted.get(pid, False) else (255, 255, 0)
            text_size, _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(
                frame,
                timer_text,
                (frame.shape[1] - text_size[0] - 20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                c,
                2,
            )
            y_offset += 35

        # Visualization: Alert Labels (Bottom Left)
        alert_y_offset = frame.shape[0] - 30
        for pid, alerted in self.id_alerted.items():
            if alerted:
                alert_text = f"ALERT - ID {pid}"
                cv2.putText(
                    frame,
                    alert_text,
                    (20, alert_y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                alert_y_offset -= 30

        # Visualization: Counter Box (Bottom Right)
        box_width = 220
        box_height = 80
        box_margin = 20
        bx2 = frame.shape[1] - box_margin
        bx1 = bx2 - box_width
        by2 = frame.shape[0] - box_margin
        by1 = by2 - box_height
        
        # Draw counter background box
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)

        cv2.putText(frame, f"Entry : {self.entry_count}", (bx1 + 15, by1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Exit  : {self.exit_count}", (bx1 + 15, by1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        still_occupied = self.max_entry_count - self.max_exit_count

        status_dict = {
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "still_occupied": still_occupied
        }

        # Optionally show frame (inherited from original code)
        cv2.imshow("Restroom Person Tracking Detection", frame)
        cv2.waitKey(1)
        
        return frame, has_new_alert, status_dict
