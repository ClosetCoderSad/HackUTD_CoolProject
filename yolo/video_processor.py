# video_processor.py
import cv2
import base64
import time
import socket
import struct
import numpy as np
from voxel_sdk.device_controller import DeviceController
from voxel_sdk.ble import BleVoxelTransport
from inference_sdk import InferenceHTTPClient
# Import from our other project files
from shared_state import detection_data, latest_annotated_frame, lock
from utils import _recv_exact, get_local_ip


def setup_stream_connection(device_name="voxel", stream_port=9000):
    """
    Connects to the Voxel device and sets up the TCP listener.
    Returns the transport, filesystem, and the connection socket from the device.
    """
    print("Helper: Connecting to Voxel device...")
    transport = BleVoxelTransport(device_name=device_name)
    transport.connect("")
    
    controller = DeviceController(transport)
    fs = controller.filesystem = DeviceController(transport).filesystem
    print("Helper: Connected.")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", stream_port))
    listener.listen(1)
    print(f"Helper: TCP server listening for video on port {stream_port}...")

    my_ip = get_local_ip()
    print(f"Helper: My local IP is {my_ip}. Telling device to connect...")
    response = fs.start_rdmp_stream(my_ip, stream_port)
    if "error" in response:
        raise RuntimeError(f"Device failed to start streaming: {response}")

    listener.settimeout(20.0)
    try:
        conn, addr = listener.accept()
        print(f"Helper: Stream connection received from device: {addr}")
        return transport, fs, conn
    except socket.timeout:
        raise TimeoutError("Timed out waiting for stream connection from device")
    finally:
        listener.close()


def receive_frame(conn):
    """
    Receives a single frame payload from the socket and decodes it.
    Returns the OpenCV frame or None if the stream ends.
    """
    header = _recv_exact(conn, 8)
    if not header or header[:4] != b"VXL0":
        return None
    
    frame_len = struct.unpack(">I", header[4:])[0]
    payload = _recv_exact(conn, frame_len)
    if not payload:
        return None
        
    return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)


def process_frame_and_update_state(frame, mode, model=None, api_client=None):
    """
    Processes a single frame using either a local model or the Roboflow API,
    and updates the shared global state.
    """
    global detection_data, latest_annotated_frame

    # --- DYNAMIC INFERENCE LOGIC ---
    if mode == 'wall_quality_api':
        # --- API PATH ---
        # Encode frame to JPEG in memory for the API call
        _, img_encoded = cv2.imencode(".jpg", frame)
        
        # Get predictions from Roboflow
        result = api_client.infer(img_encoded, model_id="wall-quality-detection/1")
        predictions = result.get('predictions', [])

        # Since the API doesn't support tracking, we generate a simple count/confidence summary
        aggregator = defaultdict(lambda: [0.0, 0])
        for p in predictions:
            aggregator[p['class']][0] += p['confidence']
            aggregator[p['class']][1] += 1

        final_detections_summary = {}
        for class_name, (conf_sum, count) in aggregator.items():
            final_detections_summary[class_name] = {
                "count": count, "average_confidence": conf_sum / count
            }
        
        # NOTE: Cannot provide a useful annotated frame without bounding boxes
        annotated_frame = frame 
        with lock:
            latest_annotated_frame = annotated_frame.copy()
            # Overwrite the global state with the simple summary
            detection_data = {"status": "success", "detections": final_detections_summary}

    else:
        # --- LOCAL MODEL TRACKING PATH ---
        results = model.track(frame, persist=True, verbose=False) 
        result = results[0]
        annotated_frame = result.plot()

        with lock:
            latest_annotated_frame = annotated_frame.copy()

        # Check if the tracker found any objects
        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()
            boxes_xyxy = result.boxes.xyxy.cpu()
            confs = result.boxes.conf.cpu().tolist()

            for i, track_id in enumerate(track_ids):
                # If this is a new detection, capture its image and create a record
                if track_id not in detection_data["detections"]:
                    x1, y1, x2, y2 = [int(coord) for coord in boxes_xyxy[i]]
                    cropped_image = frame[y1:y2, x1:x2]
                    _, buffer = cv2.imencode('.jpg', cropped_image)
                    b64_image = base64.b64encode(buffer).decode('utf-8')
                    
                    detection_data["detections"][track_id] = {
                        "track_id": track_id,
                        "class_name": result.names[class_ids[i]],
                        "first_seen_timestamp": time.time(),
                        "last_seen_timestamp": time.time(),
                        "confidence": confs[i],
                        "first_image_base64": b64_image
                    }
                else:
                    # Otherwise, just update existing record
                    detection_data["detections"][track_id]["last_seen_timestamp"] = time.time()
                    detection_data["detections"][track_id]["confidence"] = confs[i]

            # Update the summary after processing the frame
            detection_data["summary"] = {
                "unique_total_count_seen": len(detection_data["detections"]),
                "detections_in_current_frame": len(track_ids),
            }
        
        detection_data["status"] = "success"




def video_processing_thread(mode, model=None, device_name="voxel", stream_port=9000):
    """
    High-level orchestrator. Now initializes the Roboflow client if needed.
    """
    global detection_data
    transport, fs, conn = None, None, None
    api_client = None

    # --- MODIFICATION: Initialize Roboflow Client for API mode ---
    if mode == 'wall_quality_api':
        print("Background thread: Initializing Roboflow API client...")
        api_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            # !!! IMPORTANT: Replace with YOUR private API key from Roboflow !!!
            api_key="9i6G8SnX8usNN9yNSJZv" 
        )
        print("Background thread: Roboflow client initialized.")

    try:
        transport, fs, conn = setup_stream_connection(device_name, stream_port)
        while True:
            frame = receive_frame(conn)
            if frame is None:
                print("Background thread: Stream ended."); break
            
            # Pass the mode, model, and client to the processing function
            process_frame_and_update_state(frame, mode, model=model, api_client=api_client)

    except Exception as e:
        print(f"An error occurred in the background thread: {e}")
        with lock:
            detection_data = {"status": "error", "message": str(e), "detections": {}}
    finally:
        print("Background thread: Cleaning up...")
        if conn:
            conn.close()
        if fs:
            fs.stop_rdmp_stream()
        if transport and transport.is_connected():
            transport.disconnect()
        print("Background thread: Disconnected and shut down.")
