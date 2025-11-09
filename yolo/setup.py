# server.py
import argparse
import threading
import time
import cv2
from flask import Flask, jsonify
from ultralytics import YOLO

# Scientific computing
import numpy as np
import os

# Import from our other project files
from video_processor import video_processing_thread
import shared_state

app = Flask(__name__)

@app.route('/')
def index():
    return "YOLO Detection Server is running. Use the /detections endpoint to get results."

@app.route('/detections')
def get_detections():
    with shared_state.lock:
        return jsonify(shared_state.detection_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a YOLO backend server.")
    parser.add_argument(
        '--mode', type=str, default='general', 
        choices=['car_damage_yolo11m', 'general'],
        help="The detection mode to use. 'car_damage_yolo11m' uses the local YOLO model."
    )
    parser.add_argument('--port', type=int, default=5002, help="Port to run the API server on.")
    args = parser.parse_args()

    # Always use local model processing
    MODEL_MAP = {
        'general': 'yolov8n.pt',
        'car_damage_yolo11m': 'yolo11m_car_damage.pt'
    }
    selected_model_file = MODEL_MAP[args.mode]
    print(f"Main thread: Loading local model for '{args.mode}' mode from '{selected_model_file}'...")

    # Fallback: if the requested model file isn't present, try to use the
    # existing car-damage model as a sensible default so testing can proceed.
    if not os.path.exists(selected_model_file):
        fallback = 'yolo11m_car_damage.pt'
        if os.path.exists(os.path.join(os.path.dirname(__file__), fallback)) or os.path.exists(fallback):
            print(f"Warning: '{selected_model_file}' not found. Falling back to '{fallback}' for testing.")
            selected_model_file = fallback
        else:
            print(f"ERROR: Model file '{selected_model_file}' not found and no fallback available.")
            exit(1)

    try:
        model = YOLO(selected_model_file)
        print("Main thread: Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model '{selected_model_file}': {e}")
        exit(1)
    # Event to signal threads to stop
    stop_event = threading.Event()
    
    # Start processing thread
    processing_thread = threading.Thread(target=video_processing_thread, args=(args.mode, model), daemon=True)
    processing_thread.start()

    # Start Flask server thread
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=args.port), daemon=True)
    flask_thread.start()
    print(f"Main thread: Starting Flask server in background on http://0.0.0.0:{args.port}")
    
    print("Main thread: Starting live feed display. Press 'q' in the window to quit.")
    
    print("Main thread: Setting up display...")
    window_name = "Live Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.moveWindow(window_name, 100, 100)  # Position the window
    
    frames_displayed = 0
    last_display_time = time.time()
    last_frame = None
    
    while True:
        current_time = time.time()
        
        # Get the latest frame with minimal lock time
        with shared_state.lock:
            if shared_state.latest_annotated_frame is not None:
                frame = shared_state.latest_annotated_frame.copy()
            else:
                frame = None
        
        # Process and display the frame
        if frame is not None and frame.size > 0:
            try:
                # Always resize to maintain consistent display
                if frame.shape != (720, 1280, 3):
                    frame = cv2.resize(frame, (1280, 720))
                
                # Create a copy for display
                display_frame = frame.copy()
                display_frame = np.ascontiguousarray(display_frame)
                
                # Force window to stay on top and show frame
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                cv2.imshow(window_name, display_frame)
                frames_displayed += 1
                last_frame = display_frame
                
                # Log display stats periodically
                if current_time - last_display_time >= 2.0:
                    fps = frames_displayed / (current_time - last_display_time)
                    print(f"Display: Showing frames at {fps:.1f} FPS")
                    frames_displayed = 0
                    last_display_time = current_time
            except Exception as e:
                print(f"Display error: {e}")
                # If we have a last good frame, try to keep showing it
                if last_frame is not None:
                    try:
                        cv2.imshow(window_name, last_frame)
                    except:
                        pass
        else:
            # Show waiting screen
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for frames...", (480, 360),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(window_name, blank)
        
        # Check for exit with a short wait time for smooth display
        key = cv2.waitKey(1) & 0xFF  # Shorter wait time for more responsive display
        if key == ord('q'):  # Check for 'q' key press
            print(f"Main thread: 'q' pressed, shutting down. Frames displayed this session: {frames_displayed}")
            break
            
    # --- Cleanup ---
    stop_event.set()  # Signal other threads to stop
    cv2.destroyAllWindows()
    print("Main thread: Display cleanup complete. Waiting for other threads...")
    
    # Give threads a moment to clean up
    time.sleep(1)
    print("Main thread: Exiting.")