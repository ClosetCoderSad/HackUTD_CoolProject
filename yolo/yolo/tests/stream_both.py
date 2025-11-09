"""
Display two MJPG streams side-by-side from Voxel devices.

Usage (after starting streams on the devices):
    python3 yolo/tests/multi_connect_test.py

The script will:
1. Connect to MJPG streams on ports 9000 and 9001
2. Display both streams side by side
3. Use threading to handle streams in parallel
4. Press 'q' to quit
"""

import cv2
import numpy as np
import socket
import struct
import threading
import time
from typing import Optional, Dict, Tuple

# Stream settings
STREAM_PORTS = [9000, 9001]  # Left and right stream ports
WINDOW_NAME = "Voxel Streams"
FRAME_WIDTH = 640  # Each stream
FRAME_HEIGHT = 480

class StreamReceiver:
    """Handles receiving and decoding frames from one MJPG stream."""
    
    def __init__(self, port: int):
        self.port = port
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_timestamp = 0.0
        self.running = False
        self.connected = False
        self._lock = threading.Lock()
    
    def _recv_exact(self, sock: socket.socket, length: int) -> Optional[bytes]:
        """Read exactly length bytes or return None if connection closed."""
        chunks = []
        remaining = length
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def get_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Get the latest frame and its timestamp. Thread-safe."""
        with self._lock:
            return self.latest_frame, self.latest_timestamp

    def _update_frame(self, frame: Optional[np.ndarray]):
        """Update the latest frame. Thread-safe."""
        with self._lock:
            self.latest_frame = frame
            self.latest_timestamp = time.time()

    def receive_frames(self):
        """Main receive loop - runs in its own thread."""
        while self.running:
            try:
                # Try to connect/reconnect
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)  # Timeout for connect and recv
                sock.connect(('localhost', self.port))
                self.connected = True
                print(f"Connected to stream on port {self.port}")

                while self.running:
                    # Read 8-byte header
                    header = self._recv_exact(sock, 8)
                    if not header:
                        print(f"Stream {self.port}: Connection closed")
                        break

                    # Verify magic and get length
                    if header[:4] != b"VXL0":
                        print(f"Stream {self.port}: Invalid magic in header")
                        break

                    frame_len = struct.unpack(">I", header[4:])[0]
                    if frame_len <= 0 or frame_len > 5 * 1024 * 1024:
                        print(f"Stream {self.port}: Invalid frame length {frame_len}")
                        break

                    # Read and decode the JPEG frame
                    jpeg_data = self._recv_exact(sock, frame_len)
                    if not jpeg_data:
                        print(f"Stream {self.port}: Failed reading frame data")
                        break

                    frame_array = np.frombuffer(jpeg_data, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        # Resize to target size
                        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                        self._update_frame(frame)

            except (socket.error, ConnectionError) as e:
                print(f"Stream {self.port}: Connection error: {e}")
                self.connected = False
                if self.running:
                    time.sleep(1.0)  # Wait before retry
                continue

            finally:
                try:
                    sock.close()
                except Exception:
                    pass

    def start(self):
        """Start the receiver thread."""
        self.running = True
        threading.Thread(target=self.receive_frames, daemon=True).start()

    def stop(self):
        """Stop the receiver thread."""
        self.running = False


def main():
    # Create window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    # Start receivers
    receivers = [StreamReceiver(port) for port in STREAM_PORTS]
    for r in receivers:
        r.start()

    try:
        blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        while True:
            # Get latest frames
            frames = []
            for i, receiver in enumerate(receivers):
                frame, ts = receiver.get_frame()
                if frame is None:
                    # If no frame, show info text on blank
                    info = blank.copy()
                    text = f"Waiting for stream {STREAM_PORTS[i]}..."
                    if receiver.connected:
                        text += " (Connected)"
                    cv2.putText(info, text,
                              (40, FRAME_HEIGHT//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                              (255, 255, 255), 2)
                    frames.append(info)
                else:
                    # Add port number to frame
                    labeled = frame.copy()
                    cv2.putText(labeled, f"Port {STREAM_PORTS[i]}",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                              0.8, (0, 255, 0), 2)
                    frames.append(labeled)

            # Combine side by side
            if len(frames) == 2:
                display = np.hstack(frames)
                cv2.imshow(WINDOW_NAME, display)
            
            # Check for quit
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:  # q or ESC
                break

    finally:
        # Clean up
        for r in receivers:
            r.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()