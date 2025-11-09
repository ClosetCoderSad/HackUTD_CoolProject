# server.py
import argparse
import threading
import cv2
from flask import Flask, jsonify
from ultralytics import YOLO

# Import from our other project files
from video_processor import video_processing_thread
from shared_state import detection_data, latest_annotated_frame, lock

app = Flask(__name__)

@app.route('/')
def index():
    return "YOLO Detection Server is running. Use the /detections endpoint to get results."

@app.route('/detections')
def get_detections():
    with lock:
        results_to_return = latest_results_json
    return app.response_class(response=results_to_return, status=200, mimetype='application/json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a YOLO backend server.")
    parser.add_argument(
        '--mode', type=str, default='general', 
        # --- CLEANUP: Removed the redundant 'wall_quality' local mode ---
        choices=['car_damage_yolo11m', 'general', 'wall_quality_api'],
        help="The detection mode to use. 'wall_quality_api' uses the Roboflow hosted API."
    )
    parser.add_argument('--port', type=int, default=5002, help="Port to run the API server on.")
    args = parser.parse_args()

    model = None
    if args.mode != 'wall_quality_api':
        # --- CLEANUP: Removed the 'wall_quality.pt' entry ---
        MODEL_MAP = {
            'general': 'yolov8n.pt',
            'car_damage_yolo11m': 'yolo11m_car_damage.pt'
        }
        selected_model_file = MODEL_MAP[args.mode]
        print(f"Main thread: Loading local model for '{args.mode}' mode from '{selected_model_file}'...")
        try:
            model = YOLO(selected_model_file)
            print("Main thread: Model loaded successfully.")
        except FileNotFoundError:
            print(f"ERROR: Model file '{selected_model_file}' not found.")
            exit()
    else:
        print(f"Main thread: Running in '{args.mode}' mode. No local model will be loaded.")
    processing_thread = threading.Thread(target=video_processing_thread, args=(args.mode, model), daemon=True)
    processing_thread.start()

    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=args.port), daemon=True)
    flask_thread.start()
    print(f"Main thread: Starting Flask server in background on http://0.0.0.0:{args.port}")
    # The 'lambda' is used to run app.run in a simple, argument-less function for the thread
    
    print("Main thread: Starting live feed display. Press 'q' in the window to quit.")
    while True:
        frame_to_show = None
        with lock:
            if latest_annotated_frame is not None:
                frame_to_show = latest_annotated_frame.copy()
        
        if frame_to_show is not None:
            cv2.imshow("Live Feed", frame_to_show)

        # Check for 'q' key press to exit the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Main thread: 'q' pressed, shutting down.")
            break
            
    # --- Cleanup ---
    cv2.destroyAllWindows()
    print("Main thread: Exiting.")