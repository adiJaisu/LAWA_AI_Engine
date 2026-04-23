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
        # "hardcoded_rois": [
        #     [(982, 522), (997, 522), (997, 527), (982, 527)], # LED
        #     [(676, 1004), (765, 887), (1064, 921), (993, 1060), (675, 1006)], # Entry (roi_points)
        #     [(332, 1015), (438, 940), (872, 1033), (748, 1077), (503, 1079), (339, 1015)] # Presence (roi_points_2)
        # ]
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
        "colors": [(128, 0, 128)]},
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

def compress_frame_to_buffer(frame):
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buffer.tobytes()

def create_message_dict(frame_bytes, width, height, frame_id, rois, usecase_info, camera_id="test_cam_01"):
    mexico_time = datetime.datetime.now(datetime.timezone.utc).astimezone(ZoneInfo("America/Mexico_City"))
    timestamp = mexico_time.strftime('%Y-%m-%dT%H:%M:%S.%f')
    
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

def main():
    parser = argparse.ArgumentParser(description="Unified Streamhandler for Loitering, Tailgating, In/Out Person Count, CrowdDensity and PersonCountInsideCompartment")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--usecase", type=str, choices=["loitering", "tailgating", "train_arrival_depart_monitor", "in_out_person_count", "person_entered_inside_train", "crowd_density"], required=True, help="Select usecase")
    parser = argparse.ArgumentParser(description="Unified Streamhandler for Loitering, Tailgating, In/Out Person Count, CrowdDensity and PersonCountInsideCompartment")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--usecase", type=str, choices=["loitering", "tailgating", "in_out_person_count", "crowd_density", "person_count_inside_compartment"], required=True, help="Select usecase")
    parser.add_argument("--fps", type=int, default=5, help="Simulation FPS")
    parser.add_argument("--batch_size", type=int, default=1, help="Frames per batch")
    args = parser.parse_args()

    usecase_info = USECASES[args.usecase]
    channel, connection = get_rabbitmq_channel(usecase_info["queue"])
    os.environ["OPENCV_VIDEOIO_DEBUG"] = "1"  # Enable OpenCV video debugging
    cap = cv2.VideoCapture(args.video)
    print(f"Opened video {args.video} with OpenCV backend: {cap.getBackendName()}")
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read video")
        return
        
    print(f"\n--- {args.usecase.upper()} ROI SETUP ---")
    if "hardcoded_rois" in usecase_info:
        print(f"Using hardcoded ROIs for {args.usecase}.")
        drawn_rois = usecase_info["hardcoded_rois"]
    else:
        drawn_rois = interactive_draw_zones(first_frame, usecase_info)
        
    if not drawn_rois:
        return
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    
    print(f"\nStreaming to {usecase_info['queue']} at {args.fps} FPS...")
    
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
                frame_bytes = compress_frame_to_buffer(frame)
                msg = create_message_dict(frame_bytes, w, h, frame_idx, drawn_rois, usecase_info)
                batch.append(msg)
                frame_idx += 1
                
            payload = msgpack.packb(batch, use_bin_type=True)
            channel.basic_publish(exchange=EXCHANGE_NAME, routing_key=usecase_info["queue"], body=payload)
            
            print(f"[{args.usecase}] Sent Batch #{frame_idx // args.batch_size}")
            
            elapsed = time.time() - start_time
            time.sleep(max(0.01, (1.0 / args.fps) - elapsed))

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

    cap.release()
    connection.close()

if __name__ == "__main__":
    main()
