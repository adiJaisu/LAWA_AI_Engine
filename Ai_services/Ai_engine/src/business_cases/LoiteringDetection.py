import cv2
import os
import math
import numpy as np
from typing import List, Tuple, Any, Dict
from src.constant.constants import Constants
from shapely.geometry import Polygon, Point
from datetime import datetime
import time

from src.utils.Logger import LoggingConfig
from src.constant.global_constant import VisionPipeline
from src.utils.ConfigReader import cfg
from src.Exception.Exception import FrameProcessingException, PersonDetectionError

class LoiteringDetectionException(Exception):
    pass

class LoiteringDetection:
    """
    Unified person monitoring system for loitering detection.
    """

    def __init__(self):
        """Initialize Loitering Detector with configuration parameters."""
        logging_config = LoggingConfig()
        self.logger = logging_config.setup_logging()
        
        self.PERSON_CLASS_ID = int(cfg.get_value_config(
            Constants.LOITERING_DETECTION,
            Constants.PERSON_CLASS_ID
        ))
        
        self.case = int(cfg.get_value_config(Constants.LOITERING_DETECTION, Constants.CASE))
        
        self.ZONE_LABELS = ["RPA", "NRUA", "NRPA"]
        self.ZONE_COLORS = {
            "RPA": (0, 0, 255),
            "NRUA": (0, 255, 255),
            "NRPA": (0, 255, 0),
        }

        self.ZONES = {}
        self.track_info = {}
        self.zone_polygons = {}
        self.live_loitering_status = {}
        self.zone_event_state = {}

        self.PIXEL_TO_METER = float(cfg.get_value_config(Constants.LOITERING_DETECTION, Constants.PIXEL_TO_METER))
        self.MAX_ALLOWED_MOVEMENT = int(cfg.get_value_config(Constants.LOITERING_DETECTION, Constants.MAX_ALLOWED_MOVEMENT))
        self.GRACE_PERIODS = {
            "NRUA": int(cfg.get_value_config(Constants.LOITERING_DETECTION, Constants.GRACE_PERIOD_NRUA)),
            "NRPA": int(cfg.get_value_config(Constants.LOITERING_DETECTION, Constants.GRACE_PERIOD_NRPA)),
            "RPA": int(cfg.get_value_config(Constants.LOITERING_DETECTION, Constants.GRACE_PERIOD_RPA)),
        }
        self.tracker_name = cfg.get_value_config(Constants.LOITERING_DETECTION, Constants.TRACKER_NAME)
        self.conf = float(cfg.get_value_config(Constants.LOITERING_DETECTION, Constants.LOITERING_DETECTION_CONFIDENCE))

        self.event_triggered = False

    def detect(self, frame: np.ndarray, rois: List[List[List[int]]], detector: Any) -> Tuple[np.ndarray, bool, Dict]:
        """
        Processes a video frame for loitering detection.
        NOTE: Since loitering might use predefined zones from constants or camera specific rois,
        we assume rois can be mapped or we fall back to self.ZONES.
        """
        current_time = time.time()
        
        # Only re-initialize zones if ROIs have changed to prevent state loss
        if rois and len(rois) > 0:
            # Check if conceptual ROIs changed (to avoid unnecessary reset)
            rois_list = [np.array(r).tolist() for r in rois]
            if not hasattr(self, 'prev_rois_list') or self.prev_rois_list != rois_list:
                self.prev_rois_list = rois_list
                new_zones = {}
                new_zone_polygons = {}
                
                for i, polygon_points in enumerate(rois):
                    if len(polygon_points) < 3:
                        continue
                    zone_label = self.ZONE_LABELS[i] if i < len(self.ZONE_LABELS) else f"ZONE_{i}"
                    new_zones[zone_label] = {
                        "polygon": np.array(polygon_points),
                        "color": self.ZONE_COLORS.get(zone_label, (0, 255, 0)),
                    }
                    new_zone_polygons[zone_label] = Polygon(new_zones[zone_label]["polygon"])

                # Update state dictionaries without wiping existing ones
                self.ZONES = new_zones
                self.zone_polygons = new_zone_polygons
                
                # Update live status and event state only for new keys, or keep existing
                new_live_status = {}
                new_event_state = {}
                for zone_label in self.ZONES.keys():
                    new_live_status[zone_label] = self.live_loitering_status.get(zone_label, {"count": 0, "ids": []})
                    new_event_state[zone_label] = self.zone_event_state.get(zone_label, {
                        "active": False,
                        "start_time": None,
                        "start_dt": None,
                        "ids": set(),
                        "trigger_dwell": 0.0,
                        "trigger_prox": 0.0,
                    })
                
                self.live_loitering_status = new_live_status
                self.zone_event_state = new_event_state
                self.logger.info(f"LoiteringDetection: ROIs updated. Active zones: {list(self.ZONES.keys())}")
        
        self.logger.debug(f"ROIS in detect: {rois}, Zones active: {list(self.ZONES.keys())}")

        try:
            results = detector.make_prediction_with_tracking(
                frame=frame, 
                classes_id=[self.PERSON_CLASS_ID], 
                confidence=self.conf, 
                inference_image_size=640, 
                tracker=self.tracker_name
            )
            print(f"Detection results: {results}")
            cv2.waitKey(1)
        except Exception as e:
            self.logger.error(f"Error during object tracking: {e}")
            raise LoiteringDetectionException(f"Tracking error: {e}")

        self.event_triggered = False
        loitering_in_zone = {z: set() for z in self.ZONES.keys()}
        bboxes_info = []
        
        try:
            if results and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                print(f"Boxes: {boxes}, Track IDs: {track_ids}")
                for box, track_id in zip(boxes, track_ids):
                    track_id = int(track_id)
                    x1, y1, x2, y2 = box.astype(int)
                    centroid_x = int((x1 + x2) / 2)
                    centroid_y = int((y1 + y2) / 2)
                    foot_x = centroid_x
                    foot_y = y2

                    centroid_point = Point(centroid_x, centroid_y)
                    foot_point = Point(foot_x, foot_y)

                    if track_id not in self.track_info:
                        self.track_info[track_id] = {}

                    for zone_code, poly in self.zone_polygons.items():
                        inside_zone = poly.contains(centroid_point) or poly.contains(foot_point)

                        if zone_code not in self.track_info[track_id]:
                            self.track_info[track_id][zone_code] = {
                                "enter_time": None,
                                "positions": [],
                                "loitering": False,
                            }

                        zone_state = self.track_info[track_id][zone_code]
                        grace_period = int(self.GRACE_PERIODS.get(zone_code, 0))

                        if inside_zone:
                            if zone_state["enter_time"] is None:
                                zone_state["enter_time"] = current_time

                            zone_state["positions"].append((foot_x, foot_y))
                            dwell_time = current_time - zone_state["enter_time"]
                            positions = zone_state["positions"]
                            dist_px = (
                                np.linalg.norm(np.array(positions[-1]) - np.array(positions[0]))
                                if len(positions) > 1
                                else 0
                            )
                            proximity_m = dist_px * self.PIXEL_TO_METER
                            if self.case == 1:
                                if dwell_time >= grace_period and dist_px >= self.MAX_ALLOWED_MOVEMENT:
                                    zone_state["loitering"] = True
                                    loitering_in_zone[zone_code].add(track_id)

                                    if not self.zone_event_state[zone_code]["active"]: 
                                        self.zone_event_state[zone_code]["trigger_dwell"] = dwell_time
                                        self.zone_event_state[zone_code]["trigger_prox"] = proximity_m
                                else:
                                    zone_state["loitering"] = False

                            elif self.case == 2:
                                if dwell_time >= grace_period:
                                    zone_state["loitering"] = True
                                    loitering_in_zone[zone_code].add(track_id)

                                    if not self.zone_event_state[zone_code]["active"]:
                                        self.zone_event_state[zone_code]["trigger_dwell"] = dwell_time
                                        self.zone_event_state[zone_code]["trigger_prox"] = proximity_m
                                else:
                                    zone_state["loitering"] = False

                        else:
                            zone_state["enter_time"] = None
                            zone_state["positions"].clear()
                            zone_state["loitering"] = False

                    loiter_zones = [z for z in self.ZONES.keys() if self.track_info[track_id].get(z, {}).get("loitering")]
                    bboxes_info.append({
                        "id": track_id,
                        "bbox": [x1, y1, x2, y2],
                        "loitering_zones": loiter_zones
                    })
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


            for zone_code in self.ZONES.keys():
                ids_set = loitering_in_zone[zone_code]
                self.live_loitering_status[zone_code]["count"] = len(ids_set)
                self.live_loitering_status[zone_code]["ids"] = sorted(list(ids_set))

            for zone_code in self.ZONES.keys():
                current_ids = loitering_in_zone[zone_code]
                state = self.zone_event_state[zone_code]

                if len(current_ids) > 0 and not state["active"]:
                    state["active"] = True
                    state["start_time"] = current_time
                    state["start_dt"] = datetime.now()
                    state["ids"] = set(current_ids)
                elif len(current_ids) > 0 and state["active"]:
                    state["ids"].update(current_ids)
                elif len(current_ids) == 0 and state["active"]:
                    end_dt = datetime.now()
                    duration_s = current_time - state["start_time"]
                    
                    # reset
                    state["active"] = False
                    state["start_time"] = None
                    state["start_dt"] = None
                    state["ids"] = set()
                    state["trigger_dwell"] = 0.0
                    state["trigger_prox"] = 0.0

            self.live_loitering_status["bboxes"] = bboxes_info
            
            i = 0
            for zone_code, zone in self.ZONES.items():
                if self.zone_event_state[zone_code]["active"]:
                    self.event_triggered = True
                i += 1

            cv2.imshow("Loitering Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

            self.logger.debug(f"Live Loitering Status: {self.live_loitering_status}, event triggered: {self.event_triggered}")
            return self.event_triggered, frame, self.live_loitering_status
        except Exception as e:
            self.logger.critical(f"Critical error in frame processing: {str(e)}", exc_info=True)
            raise FrameProcessingException(f"Error during detect: {str(e)}")