"""
Flask Backend Web Application for Vision-Based Distracted Driver Detection System.
Provides REST APIs, MJPEG video streaming, SocketIO real-time telemetry broadcasting,
and SQLite database integration.
"""

import os
import sys
import time
import threading
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, make_response
from flask_socketio import SocketIO, emit

import database
from detection import DistractionEngine

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vision_driver_safety_secret_key_2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize Distraction Engine
engine = DistractionEngine()

# Camera & Streaming state
class CameraManager:
    """Manages OpenCV VideoCapture thread, state, and frame distribution."""
    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.is_running = False
        self.is_paused = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_telemetry = {}
        self.camera_connected = False
        self.thread = None
        self.consecutive_failures = 0

    def _open_camera(self):
        """Attempts to open the best available camera backend and index."""
        # Try DirectShow first on Windows, then standard
        candidates = []
        if sys.platform.startswith('win'):
            for idx in [0, 1]:
                candidates.append((idx, cv2.CAP_DSHOW))
            for idx in [0, 1]:
                candidates.append((idx, None))
        else:
            candidates.append((0, None))

        for idx, backend in candidates:
            try:
                cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
                if cap and cap.isOpened():
                    # Warmup reads (some webcams require 2-3 frames to start streaming)
                    for _ in range(3):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            self.src = idx
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            cap.set(cv2.CAP_PROP_FPS, 30)
                            print(f"[INFO] Webcam connected successfully (Index: {idx}, Backend: {backend})")
                            return cap
                        time.sleep(0.05)
                    cap.release()
            except Exception as e:
                print(f"[DEBUG] Camera probe exception for idx {idx}: {e}")
        return None

    def start(self):
        with self.lock:
            if self.is_running and self.thread and self.thread.is_alive():
                return True

            self.cap = self._open_camera()
            self.camera_connected = self.cap is not None
            self.is_running = True
            self.is_paused = False
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            print("[INFO] Live camera capture and AI detection thread launched.")
            return True

    def stop(self):
        with self.lock:
            self.is_running = False
            self.camera_connected = False
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
            print("[INFO] Camera stopped.")

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def _capture_loop(self):
        while self.is_running:
            if self.is_paused:
                time.sleep(0.05)
                continue

            frame = None
            if self.cap and self.cap.isOpened():
                try:
                    ret, raw_frame = self.cap.read()
                    if ret and raw_frame is not None and raw_frame.size > 0:
                        frame = raw_frame
                        self.consecutive_failures = 0
                        self.camera_connected = True
                    else:
                        self.consecutive_failures += 1
                except Exception as e:
                    print(f"[WARN] Camera read error: {e}")
                    self.consecutive_failures += 1

            # If camera disconnected or failed repeatedly, attempt reconnection every 3 seconds
            if frame is None:
                if self.consecutive_failures % 30 == 0:
                    print("[INFO] Attempting webcam reconnection...")
                    if self.cap:
                        self.cap.release()
                    self.cap = self._open_camera()

                # Generate a informative diagnostic frame so user sees camera status
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "SEARCHING FOR WEBCAM...", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(blank, "Ensure camera permission is allowed in Windows Settings", (70, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
                cv2.putText(blank, "Or click 'Start' / 'Retry' on the toolbar", (160, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)

                with self.lock:
                    self.latest_frame = blank
                time.sleep(0.1)
                continue

            try:
                # Mirror horizontally for natural driver ergonomics
                frame = cv2.flip(frame, 1)

                # Process through Distraction Engine
                annotated_frame, telemetry = engine.process_frame(frame)

                with self.lock:
                    self.latest_frame = annotated_frame
                    self.latest_telemetry = telemetry

                # Emit telemetry via Socket.IO
                socketio.emit('telemetry_update', telemetry)
            except Exception as e:
                print(f"[ERROR] Frame processing error: {e}")

            time.sleep(0.01)

    def get_jpeg(self):
        with self.lock:
            if self.latest_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', self.latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    return jpeg.tobytes()

        # Standby frame if not yet captured
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        msg = "CAMERA PAUSED" if self.is_paused else "INITIALIZING CAMERA..."
        cv2.putText(blank, msg, (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2, cv2.LINE_AA)
        _, jpeg = cv2.imencode('.jpg', blank)
        return jpeg.tobytes()


camera_manager = CameraManager(src=0)

# Connect engine event callback to SocketIO
def broadcast_alert_event(telemetry):
    socketio.emit('alert_triggered', telemetry)

engine.set_event_callback(broadcast_alert_event)


# ==============================================================================
# FLASK HTTP ROUTES
# ==============================================================================

@app.route('/')
def index():
    """Renders the main automotive cockpit dashboard."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Multipart MJPEG video stream."""
    def generate():
        while True:
            frame_bytes = camera_manager.get_jpeg()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Returns real-time system hardware, camera, and CV module status."""
    return jsonify({
        'camera': 'CONNECTED' if camera_manager.camera_connected else 'DISCONNECTED',
        'is_running': camera_manager.is_running,
        'is_paused': camera_manager.is_paused,
        'face_detection': 'ACTIVE',
        'eye_tracking': 'ACTIVE',
        'head_pose': 'ACTIVE',
        'hand_detection': 'ACTIVE',
        'yawning_detection': 'ACTIVE',
        'alert_system': 'ACTIVE',
        'backend': 'CONNECTED',
        'fps': engine.fps,
        'demo_mode': engine.demo_mode
    })


@app.route('/api/telemetry', methods=['GET'])
def get_telemetry_route():
    """Returns the latest live driver monitoring telemetry."""
    return jsonify(camera_manager.latest_telemetry)



@app.route('/api/alerts', methods=['GET'])
def get_alerts_route():
    """Returns recent alert history from SQLite database."""
    severity = request.args.get('severity', None)
    limit = int(request.args.get('limit', 100))
    alerts = database.get_alerts(limit=limit, severity=severity)
    return jsonify(alerts)


@app.route('/api/alerts', methods=['POST'])
def post_alert_route():
    """Manually logs or simulates an alert event into SQLite."""
    data = request.get_json() or {}
    alert_type = data.get('alert_type', 'Manual Alert')
    severity = data.get('severity', 'WARNING')
    duration = float(data.get('duration', 2.0))
    details = data.get('details', 'Logged via API')
    
    alert_id = database.insert_alert(alert_type, severity, duration, details)
    return jsonify({'success': True, 'alert_id': alert_id}), 201


@app.route('/api/alerts', methods=['DELETE'])
def clear_alerts_route():
    """Clears all alert history."""
    database.clear_alerts()
    return jsonify({'success': True, 'message': 'Alert history cleared'})


@app.route('/api/alerts/export', methods=['GET'])
def export_alerts_csv():
    """Exports all alert records as a CSV download."""
    csv_content = database.export_csv_string()
    response = make_response(csv_content)
    response.headers["Content-Disposition"] = "attachment; filename=driver_alert_history.csv"
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    return response


@app.route('/api/statistics', methods=['GET'])
def get_statistics_route():
    """Returns analytics statistics for charts and widgets."""
    stats = database.get_statistics()
    return jsonify(stats)


@app.route('/api/camera/control', methods=['POST'])
def camera_control():
    """Handles camera start/stop/pause/resume actions."""
    data = request.get_json() or {}
    action = data.get('action', '').lower()

    if action == 'start':
        success = camera_manager.start()
        return jsonify({'success': success, 'message': 'Camera started' if success else 'Failed to open camera'})
    elif action == 'stop':
        camera_manager.stop()
        return jsonify({'success': True, 'message': 'Camera stopped'})
    elif action == 'pause':
        camera_manager.pause()
        return jsonify({'success': True, 'message': 'Detection paused'})
    elif action == 'resume':
        camera_manager.resume()
        return jsonify({'success': True, 'message': 'Detection resumed'})
    else:
        return jsonify({'error': 'Invalid action'}), 400


@app.route('/api/demo/simulate', methods=['POST'])
def simulate_scenario():
    """Triggers viva demonstration simulation scenarios."""
    data = request.get_json() or {}
    scenario = data.get('scenario', 'reset')
    duration = float(data.get('duration', 8.0))
    res = engine.trigger_simulation(scenario, duration)
    return jsonify(res)


@app.route('/api/demo/mode', methods=['POST'])
def toggle_demo_mode():
    """Toggles demonstration HUD and timer breakdown overlay."""
    data = request.get_json() or {}
    engine.demo_mode = bool(data.get('enabled', not engine.demo_mode))
    return jsonify({'demo_mode': engine.demo_mode})


# ==============================================================================
# SOCKET.IO EVENTS
# ==============================================================================

@socketio.on('connect')
def handle_connect():
    print("[SOCKET] Client connected to live telemetry stream.")
    emit('system_status', {'status': 'connected', 'demo_mode': engine.demo_mode})


@socketio.on('request_telemetry')
def handle_telemetry_request():
    emit('telemetry_update', camera_manager.latest_telemetry)


import base64

@socketio.on('process_browser_frame')
def handle_browser_frame(data_url):
    """Processes real-time webcam frame sent from client browser."""
    try:
        if not data_url or not isinstance(data_url, str):
            return
        if ',' in data_url:
            _, encoded = data_url.split(',', 1)
        else:
            encoded = data_url
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None and frame.size > 0:
            if frame.shape[1] > 640:
                frame = cv2.resize(frame, (640, 480))
            
            processed_frame, telemetry = engine.process_frame(frame)
            camera_manager.latest_telemetry = telemetry
            
            ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            if ret:
                processed_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
                emit('processed_frame_response', {
                    'image': processed_b64,
                    'telemetry': telemetry
                })
    except Exception as e:
        print(f"[DEBUG] handle_browser_frame exception: {e}")


# Auto-start camera upon launching
def start_app_camera():
    camera_manager.start()

if __name__ == '__main__':
    # Initialize DB
    database.init_db()
    # Start camera capture thread
    threading.Thread(target=start_app_camera, daemon=True).start()
    
    print("\n" + "="*70)
    print(" VISION-BASED DISTRACTED DRIVER DETECTION SYSTEM ")
    print(" Real-Time Driver Monitoring & Safety Alert System ")
    print(" Web Dashboard: http://127.0.0.1:5000 ")
    print("="*70 + "\n")
    
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
