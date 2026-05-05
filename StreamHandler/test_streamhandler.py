import os
import cv2
import time
import pika
import msgpack
import argparse
import datetime
from zoneinfo import ZoneInfo
import numpy as np

# Configuration defaults
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.environ.get("RABBITMQ_PORT", 5672))
RABBITMQ_USER = os.environ.get("RABBITMQ_USERNAME", "guest")
RABBITMQ_PASS = os.environ.get("RABBITMQ_PASSWORD", "guest")
EXCHANGE_NAME = "LAWA_exchange"
EXCHANGE_TYPE = "direct"

USECASES = {
    "loitering": {
        "queue": "queue_loitering_detection",
        "name": "Loitering_Detection",
        "zones": ["RPA (Red)", "NRUA (Yellow)", "NRPA (Green)"],
        "colors": [(0, 0, 255), (0, 255, 255), (0, 255, 0)]
    },
    "tailgating": {
        "queue": "queue_tailgate_detection",
        "name": "Tailgate_Detection",
        "zones": ["LED Zone (White)", "Entry Zone (Blue)", "Presence Zone (Orange)"],
        "colors": [(255, 255, 255), (255, 0, 0), (0, 165, 255)]
    },
    "in_out_person_count": {
        "queue": "queue_in_out_person_count",
        "name": "In_Out_Person_Count",
        "zones": ["In Line (Green)", "Out Line (Red)"],
        "colors": [(0, 255, 0), (0, 0, 255)],
        "hardcoded_rois": [
            [(650, 440), (1400, 550)], # IN line
            [(650, 510), (1400, 620)]  # OUT line
        ]
    },
    "crowd_density": {
        "queue": "queue_crowd_density",
        "name": "Crowd_Density",
        "zones":["kuch_bhi"],
        "colors": [(255, 255, 255)]
    },
    "train_arrival_depart_monitor": {
        "queue": "queue_train_arrival_depart_monitor",
        "name": "Train_Arrival_Depart_Monitor",
        "zones": ["Train_Zone (Purple)"],
        "colors": [(128, 0, 128)]
    },
    "person_count_inside_compartment":{
        "queue": "queue_person_count_inside_compartment",
        "name": "Person_Count_Inside_Compartment",
        "zones":["kuch_bhi"],
        "colors": [(255, 255, 255)]
    },
    "person_entered_inside_train": {
        "queue": "queue_person_entered_inside_train",
        "name": "Person_Entered_Inside_Train",
        "zones": ["Platform (Blue)", "Gate1 (Yellow)", "Gate2 (Cyan)"],
        "colors": [(255, 0, 0), (0, 255, 255), (255, 255, 0)],
        "hardcoded_rois": [
            [(100, 719), (600, 420), (1100, 420), (1279, 540), (1279, 719)],
            [(120, 300), (280, 300), (280, 620), (120, 620)],
            [(400, 300), (460, 300), (460, 520), (400, 520)]
        ]
    },
    "restroom_person_tracking": {
        "queue": "queue_restroom_person_tracking",
        "name": "Restroom_Person_Tracking",
        "zones": ["Inside Zone (Green)", "Zone A (Blue)", "Zone B (Red)"],
        "colors": [(0, 255, 0), (255, 0, 0), (0, 0, 255)],
        "hardcoded_rois": [
            [(303, 93), (490, 92), (486, 410), (302, 402)], # Inside Zone points
            [(316, 360), (464, 398), (471, 359), (312, 311)], # Zone A points
            [(506, 413), (490, 448), (269, 398), (302, 364)]  # Zone B points
        ]
    },
    "queue_management": {
        "queue": "queue_queue_management",
        "name": "Queue_Management",
        "zones": ["Inside Zone (Green)", "Entry Zone (Blue)", "Service/Exit Zone (Red)"],
        "colors": [(0, 255, 0), (255, 0, 0), (0, 0, 255)],
        "hardcoded_rois": [
            [[1167, 1017], [932, 948], [0, 1006], [2, 1077]], # ROI 0: Inside Zone (Total Area)
            [[1167, 1017], [932, 948], [0, 1006], [2, 1077]], # ROI 1: Waiting Area (Waiters)
            [[1178, 1015], [996, 943], [1161, 900], [1396, 952]] # ROI 2: Service Area (Served)
        ]
    },
    "bird_eye_view": {
        "queue": "queue_bird_eye_view",
        "name": "Bird_Eye_View",
        "zones": ["Cam1 Poly", "Cam2 Poly", "Cam1 Src", "Cam2 Src", "Dst Points"],
        "colors": [(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0), (0, 0, 255)],
        "hardcoded_rois": [
            [[4, 703], [1993, 0], [2560, 0], [2560, 1440], [0, 1440]], # Cam1 Poly
            [[2, 508], [1247, 70], [2560, 0], [2560, 1440], [0, 1440]], # Cam2 Poly
            [[300, 1390], [2558, 1370], [2558, 300], [1000, 30], [500, 150], [12, 349]], # Cam1 Src
            [[1000, 20], [480, 155], [17, 348], [305, 1378], [2550, 1360], [2560, 300]], # Cam2 Src
            [[0, 750], [500, 1000], [1000, 750], [1000, 250], [500, 0], [0, 250]] # Dst Points
        ]
    }
}

def interactive_draw_zones(frame, usecase_info):
    """Interactively draw the required number of zones."""
    zones_drawn = []
    zone_names = usecase_info["zones"]
    colors = usecase_info["colors"]
    num_zones = len(zone_names)
    current_pts = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_pts.append([x, y])
            
    window_name = f"Draw {usecase_info['name']} Zones"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    for i in range(num_zones):
        current_pts.clear()
        print(f"Please draw polygon for {zone_names[i]}: Click to add points. Press ENTER to close.")
        while True:
            temp_frame = frame.copy()
            for j, pol in enumerate(zones_drawn):
                cv2.polylines(temp_frame, [np.array(pol)], isClosed=True, color=colors[j], thickness=2)
                
            if len(current_pts) > 0:
                for pt in current_pts:
                    cv2.circle(temp_frame, tuple(pt), 4, colors[i], -1)
                if len(current_pts) > 1:
                    cv2.polylines(temp_frame, [np.array(current_pts)], isClosed=False, color=colors[i], thickness=2)
                    
            cv2.putText(temp_frame, f"Drawing {zone_names[i]}. Press ENTER when done.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window_name, temp_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13 or key == 32: # Enter or Space
                if len(current_pts) >= 3:
                    zones_drawn.append(list(current_pts))
                    break
                else:
                    print("Polygon needs at least 3 points!")
            elif key == 27: # ESC
                cv2.destroyAllWindows()
                return None
                
    cv2.destroyAllWindows()
    return zones_drawn

def scale_rois(rois, current_w, current_h, ref_w=2560, ref_h=1440):
    """Scale hardcoded ROIs from reference resolution to current resolution."""
    scaled_rois = []
    for roi in rois:
        scaled_roi = []
        for pt in roi:
            scaled_pt = [
                min(current_w - 1, max(0, int(pt[0] * current_w / ref_w))),
                min(current_h - 1, max(0, int(pt[1] * current_h / ref_h)))
            ]
            scaled_roi.append(scaled_pt)
        scaled_rois.append(scaled_roi)
    return scaled_rois

def compress_frame_to_buffer(frame):
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buffer.tobytes()

def create_message_dict(frame_bytes, width, height, frame_id, rois, usecase_info, camera_id="test_cam_01", timestamp=None):
    if timestamp is None:
        actual_time = datetime.datetime.now(datetime.timezone.utc).astimezone(ZoneInfo("Asia/Kolkata"))
        timestamp = actual_time.strftime('%Y-%m-%dT%H:%M:%S.%f')
    
    msg = {
        "frame_metadata": {
            "frame_id": frame_id,
            "time_stamp": timestamp,
            "frame_height": height,
            "frame_width": width,
            "frame": frame_bytes,
            "rois": rois,
            "rabbitmq_sent_timing": timestamp,
            "usecase_name": usecase_info["name"]
        },
        "camera_metadata": {
            "camera_id": camera_id,
            "location_id": "LOC_TEST",
            "name": f"{camera_id}_TEST_CAMERA",
            "codec": "mp4v",
            "resolution": f"{width}x{height}",
            "model": "Simulated_Camera",
            "height": 45,
            "rtsp_url": "mock://rtsp_stream"
        },
        "config_metadata": []
    }
    return msg

def get_rabbitmq_channel(queue_name):
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, credentials=credentials))
    channel = connection.channel()
    channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type=EXCHANGE_TYPE, durable=True)
    channel.queue_declare(queue=queue_name, durable=True)
    channel.queue_bind(exchange=EXCHANGE_NAME, queue=queue_name, routing_key=queue_name)
    return channel, connection

def stream_bird_eye_view(cap, cap2, usecase_info, channel, args, drawn_rois):
    """Dedicated streaming logic for Bird Eye View usecase."""
    print(f"\nStreaming Bird Eye View to {usecase_info['queue']} at {args.fps} FPS...")
    frame_idx = 0
    while True:
        try:
            batch = []
            start_time = time.time()
            for _ in range(args.batch_size):
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                actual_time = datetime.datetime.now(datetime.timezone.utc).astimezone(ZoneInfo("Asia/Kolkata"))
                shared_timestamp = actual_time.strftime('%Y-%m-%dT%H:%M:%S.%f')

                # Camera 1
                h, w, _ = frame.shape
                current_rois1 = scale_rois(drawn_rois, w, h)
                msg1 = create_message_dict(compress_frame_to_buffer(frame), w, h, frame_idx, current_rois1, usecase_info, "test_cam_01", shared_timestamp)
                batch.append(msg1)

                # Camera 2
                if cap2:
                    ret2, frame2 = cap2.read()
                    if not ret2:
                        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret2, frame2 = cap2.read()
                else:
                    frame2 = frame
                
                h2, w2, _ = frame2.shape
                current_rois2 = scale_rois(drawn_rois, w2, h2)
                msg2 = create_message_dict(compress_frame_to_buffer(frame2), w2, h2, frame_idx, current_rois2, usecase_info, "test_cam_02", shared_timestamp)
                batch.append(msg2)

                frame_idx += 1

            payload = msgpack.packb(batch, use_bin_type=True)
            channel.basic_publish(exchange=EXCHANGE_NAME, routing_key=usecase_info["queue"], body=payload)
            print(f"[Bird_Eye_View] Sent Batch #{frame_idx // args.batch_size}")
            
            elapsed = time.time() - start_time
            time.sleep(max(0.01, (1.0 / args.fps) - elapsed))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error in BEV streaming: {e}")
            break

def stream_standard_usecase(cap, usecase_info, channel, args, drawn_rois):
    """Original streaming logic for all other usecases."""
    print(f"\nStreaming {usecase_info['name']} to {usecase_info['queue']} at {args.fps} FPS...")
    frame_idx = 0
    while True:
        try:
            batch = []
            start_time = time.time()
            for _ in range(args.batch_size):
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                
                h, w, _ = frame.shape
                msg = create_message_dict(compress_frame_to_buffer(frame), w, h, frame_idx, drawn_rois, usecase_info)
                batch.append(msg)
                frame_idx += 1

            payload = msgpack.packb(batch, use_bin_type=True)
            channel.basic_publish(exchange=EXCHANGE_NAME, routing_key=usecase_info["queue"], body=payload)
            print(f"[{usecase_info['name']}] Sent Batch #{frame_idx // args.batch_size}")
            
            elapsed = time.time() - start_time
            time.sleep(max(0.01, (1.0 / args.fps) - elapsed))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error in standard streaming: {e}")
            break

def main():
    parser = argparse.ArgumentParser(description="Unified Streamhandler")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--video2", type=str, help="Path to second video file (for Bird Eye View)")
    parser.add_argument("--usecase", type=str, choices=list(USECASES.keys()), required=True, help="Select usecase")
    parser.add_argument("--fps", type=int, default=5, help="Simulation FPS")
    parser.add_argument("--batch_size", type=int, default=1, help="Frames per batch")
    args = parser.parse_args()

    usecase_info = USECASES[args.usecase]
    channel, connection = get_rabbitmq_channel(usecase_info["queue"])
    
    cap = cv2.VideoCapture(args.video)
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read video")
        return
        
    print(f"\n--- {args.usecase.upper()} ROI SETUP ---")
    if "hardcoded_rois" in usecase_info:
        drawn_rois = usecase_info["hardcoded_rois"]
    else:
        drawn_rois = interactive_draw_zones(first_frame, usecase_info)
        
    if not drawn_rois:
        return
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if args.usecase == "bird_eye_view":
        cap2 = cv2.VideoCapture(args.video2) if args.video2 else None
        stream_bird_eye_view(cap, cap2, usecase_info, channel, args, drawn_rois)
    else:
        stream_standard_usecase(cap, usecase_info, channel, args, drawn_rois)

    cap.release()
    connection.close()

if __name__ == "__main__":
    main()
