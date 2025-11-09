# shared_state.py
import threading
import time

# This dictionary holds all our application's data.
detection_data = {
    "status": "initializing",
    "summary": {},
    "detections": {} 
}

# This holds the latest frame for the live GUI feed.
latest_annotated_frame = None
last_frame_update_time = None

# The lock to ensure thread-safe access to the above variables.
lock = threading.Lock()

def update_frame(frame):
    """Thread-safe function to update the latest frame"""
    global latest_annotated_frame, last_frame_update_time
    with lock:
        try:
            if frame is not None and frame.size > 0:
                latest_annotated_frame = frame.copy()
                last_frame_update_time = time.time()
                print(f"State: Frame updated successfully, shape: {frame.shape}")
            else:
                print("State: Warning - Received invalid frame")
        except Exception as e:
            print(f"State: Error updating frame: {e}")