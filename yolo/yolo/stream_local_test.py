import cv2
import numpy as np
import socket
import struct
import time
import os
from ultralytics import YOLO
from voxel_sdk.device_controller import DeviceController
from voxel_sdk.ble import BleVoxelTransport

from utils import _recv_exact, get_local_ip

# Configuration
DEFAULT_MODEL = 'yolov8n.pt'
FALLBACK_MODEL = 'yolo11m_car_damage.pt'
HOST = '0.0.0.0'
PORT = 9000
WINDOW_NAME = 'YOLO Stream'


def select_model(path=None):
    if path:
        if os.path.exists(path):
            return path
        else:
            raise FileNotFoundError(path)
    if os.path.exists(DEFAULT_MODEL):
        return DEFAULT_MODEL
    if os.path.exists(FALLBACK_MODEL):
        print(f"Warning: '{DEFAULT_MODEL}' not found. Falling back to '{FALLBACK_MODEL}'.")
        return FALLBACK_MODEL
    raise FileNotFoundError("No model file found. Place 'yolov8n.pt' or 'yolo11m_car_damage.pt' in the folder.")


if __name__ == '__main__':
    # Select model
    model_file = select_model()
    print(f"Loading model from '{model_file}'...")
    model = YOLO(model_file)
    print("Model loaded.")

    # Connect to Voxel
    print("Connecting to Voxel device (BLE)...")
    transport = BleVoxelTransport(device_name='voxel')
    transport.connect("")
    controller = DeviceController(transport)
    fs = controller.filesystem
    print("Connected to device.")

    # Setup listener
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((HOST, PORT))
    listener.listen(1)
    print(f"TCP server listening on {HOST}:{PORT}...")

    conn = None
    try:
        my_ip = get_local_ip()
        print(f"Local IP: {my_ip} -> requesting device to start RDMP stream...")
        response = fs.start_rdmp_stream(my_ip, PORT)
        if isinstance(response, dict) and 'error' in response:
            raise RuntimeError(f"Device failed to start streaming: {response}")

        listener.settimeout(15.0)
        conn, addr = listener.accept()
        print(f"Stream connection received from device: {addr}")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

        while True:
            header = _recv_exact(conn, 8)
            if not header:
                print("Stream closed or empty header. Exiting loop.")
                break
            if header[:4] != b'VXL0':
                print(f"Invalid header magic: {header[:4]}")
                break

            frame_len = struct.unpack('>I', header[4:])[0]
            payload = _recv_exact(conn, frame_len)
            if not payload:
                print("Payload empty. Breaking.")
                break

            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print("Failed to decode frame")
                continue

            # Run YOLO inference (use lightweight .predict API)
            t0 = time.time()
            results = model.predict(frame, conf=0.4, iou=0.3, verbose=False)
            dt = (time.time() - t0) * 1000
            annotated = results[0].plot()

            # Show detection info
            boxes = results[0].boxes
            num = len(boxes) if boxes is not None else 0
            print(f"Frame decoded {frame.shape} | Detections: {num} | inference {dt:.0f}ms")

            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, stopping.")
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Cleaning up...")
        try:
            if conn:
                conn.close()
        except Exception:
            pass
        try:
            listener.close()
        except Exception:
            pass
        try:
            fs.stop_rdmp_stream()
        except Exception:
            pass
        try:
            transport.disconnect()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Done.")
