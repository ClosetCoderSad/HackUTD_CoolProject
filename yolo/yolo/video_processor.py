# # In video_processor.py

import cv2
import base64
import time
import struct
import socket
import numpy as np
from collections import defaultdict
from voxel_sdk.device_controller import DeviceController
from voxel_sdk.ble import BleVoxelTransport

# Import from our other project files
import shared_state
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
    # transport is already connected; expose the controller filesystem helper
    fs = controller.filesystem
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
    print("Receiving frame header...")
    header = _recv_exact(conn, 8)
    if not header:
        print("No header received")
        return None
    if header[:4] != b"VXL0":
        print(f"Invalid header magic: {header[:4]}")
        return None
    
    frame_len = struct.unpack(">I", header[4:])[0]
    print(f"Expecting frame payload of {frame_len} bytes")
    payload = _recv_exact(conn, frame_len)
    if not payload:
        print("No payload received")
        return None
    
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        print("Failed to decode frame")
        return None
    
    print(f"Successfully decoded frame: {frame.shape}")
    return frame


def process_frame_and_update_state(frame, mode, model=None):
    """
    Processes a single frame using local YOLO model and updates the shared global state.
    """
    # no module-level globals here; we update shared_state directly
    print(f"Processing frame shape: {frame.shape if frame is not None else 'None'}")

    try:
        # Process with local YOLO model
        t0 = time.time()
        print("Local: Running YOLO detection...")
        results = model.predict(frame, conf=0.4, iou=0.3, verbose=False)
        result = results[0]
        dt = time.time() - t0
        print(f"Local: Detection completed in {dt*1000:.0f}ms")

        # Get detections and annotate frame
        boxes = result.boxes
        annotated_frame = frame.copy()
        
        detections_summary = defaultdict(lambda: {"count": 0, "total_conf": 0.0})
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Get box coordinates and convert to int
                x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]]
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]
                
                # Draw box and label
                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Update detection summary
                detections_summary[class_name]["count"] += 1
                detections_summary[class_name]["total_conf"] += conf

        # Calculate average confidence for each class
        final_summary = {}
        for cls, stats in detections_summary.items():
            final_summary[cls] = {
                "count": stats["count"],
                "average_confidence": stats["total_conf"] / stats["count"]
            }

        # Update shared state
        print("Local: Updating shared state...")
        num_boxes = len(boxes) if boxes is not None else 0
        with shared_state.lock:
            shared_state.latest_annotated_frame = np.ascontiguousarray(annotated_frame)
            shared_state.detection_data = {"status": "success", "detections": final_summary}
            print(f"Local: Frame processed and updated, found {num_boxes} objects")

    except Exception as e:
        print(f"Error processing frame: {e}")
        with shared_state.lock:
            shared_state.detection_data = {"status": "error", "message": str(e)}




def video_processing_thread(mode, model=None, device_name="voxel", stream_port=9000):
    """
    High-level orchestrator for video processing using local YOLO model.
    """
    # video_processing_thread will update shared_state; no local globals needed
    transport, fs, conn = None, None, None
    frame_count = 0
    last_fps_print = time.time()
    
    # Initialize display frame
    with shared_state.lock:
        shared_state.latest_annotated_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(shared_state.latest_annotated_frame, "Starting up...", (480, 360), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    try:
        transport, fs, conn = setup_stream_connection(device_name, stream_port)
        while True:
            t0 = time.time()
            frame = receive_frame(conn)
            t_recv = time.time()
            if frame is None:
                print("Background thread: Stream ended."); break
            
            process_frame_and_update_state(frame, mode, model=model)
            t_done = time.time()

            frame_count += 1
            if t_done - last_fps_print >= 5.0:
                fps = frame_count / (t_done - last_fps_print)
                print(f"FPS: {fps:.1f} (recv: {(t_recv-t0)*1000:.0f}ms, proc: {(t_done-t_recv)*1000:.0f}ms)")
                frame_count = 0
                last_fps_print = t_done

            # Brief sleep to prevent CPU overload
            time.sleep(0.001)

    except Exception as e:
        print(f"An error occurred in the background thread: {e}")
        with shared_state.lock:
            shared_state.detection_data = {"status": "error", "message": str(e), "detections": {}}
    finally:
        print("Background thread: Cleaning up...")
        if conn:
            conn.close()
        if fs:
            fs.stop_rdmp_stream()
        if transport and transport.is_connected():
            try:
                transport.disconnect()
            except RuntimeError as re:
                # Some Ble implementations try to join the running loop thread from
                # inside the same thread which raises RuntimeError("cannot join current thread").
                # Ignore this in cleanup and log it.
                print(f"Warning: transport.disconnect() raised RuntimeError during cleanup: {re}")
            except Exception as ex:
                print(f"Warning: transport.disconnect() failed during cleanup: {ex}")
        print("Background thread: Disconnected and shut down.")
