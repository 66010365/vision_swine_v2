# ======================================================
# Import Libraries
# ======================================================

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


# ======================================================
# Camera and Streaming Configuration
# ======================================================
CAMERA_INDEX = 0
HOST = "0.0.0.0"
PORT = 8080

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80


# ======================================================
# Shared Stream State
# ======================================================
frame_lock = threading.Lock()
latest_jpeg = None
running = threading.Event()


# ======================================================
# HTTP Stream Handler
# ======================================================

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_viewer_page()
        elif self.path == "/stream":
            self.send_mjpeg_stream()
        else:
            self.send_error(404, "Not Found")

    def send_viewer_page(self):
        page = b"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Vision Swine - Live Camera</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #111827;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
        }
        img {
            display: block;
            width: 100%;
            max-width: 960px;
            height: auto;
            margin: 20px auto;
            background: #000;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <h2>Vision Swine - Live Camera</h2>
    <p>Live stream from Jetson USB camera</p>
    <img src="/stream" alt="Camera stream">
</body>
</html>"""

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(page)

    def send_mjpeg_stream(self):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        try:
            while running.is_set():
                with frame_lock:
                    jpeg = latest_jpeg

                if jpeg is None:
                    time.sleep(0.05)
                    continue

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    b"Content-Length: "
                    + str(len(jpeg)).encode("ascii")
                    + b"\r\n\r\n"
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(0.02)

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Browser closed the stream.
            pass

    def log_message(self, format_string, *args):
        # Keep terminal output less noisy.
        return


# ======================================================
# MJPEG HTTP Server
# ======================================================

class MJPEGServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ======================================================
# Capture and Encode Camera Frames
# ======================================================

def capture_frames(camera):
    global latest_jpeg

    encode_params = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        JPEG_QUALITY,
    ]

    while running.is_set():
        success, frame = camera.read()

        if not success or frame is None:
            print("Warning: cannot read camera frame")
            time.sleep(0.1)
            continue

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            encode_params,
        )

        if success:
            with frame_lock:
                latest_jpeg = encoded.tobytes()


# ======================================================
# Main Application
# ======================================================

def main():
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    if not camera.isOpened():
        raise RuntimeError(
            "Cannot open camera. Check USB connection and /dev/video0."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    server = None
    capture_thread = None

    try:
        running.set()

        capture_thread = threading.Thread(
            target=capture_frames,
            args=(camera,),
            daemon=True,
        )
        capture_thread.start()

        server = MJPEGServer((HOST, PORT), StreamHandler)

        print("=" * 50)
        print("VISION SWINE - MJPEG STREAM SERVER")
        print("Camera: /dev/video0")
        print("Resolution target: {}x{}".format(
            FRAME_WIDTH, FRAME_HEIGHT
        ))
        print("Viewer: http://<JETSON_IP>:{}".format(PORT))
        print("Stream: http://<JETSON_IP>:{}/stream".format(PORT))
        print("Press Ctrl+C to stop")
        print("=" * 50)

        server.serve_forever()

    except KeyboardInterrupt:
        print("\nStopping camera stream...")

    finally:
        running.clear()

        if server is not None:
            server.server_close()

        camera.release()

        if capture_thread is not None:
            capture_thread.join(timeout=2)

        print("Camera released. Stream server stopped.")


# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":
    main()
