"""
Face Detector Module.
Uses MediaPipe Face Mesh to detect facial landmarks and bounding box in real time.
Includes fail-safe Haar cascade fallback for headless cloud servers (Render / Heroku / AWS).
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

MP_AVAILABLE = False
mp_face_mesh = None
mp_drawing = None
mp_drawing_styles = None
mp_import_error = None

try:
    import mediapipe as mp
    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
        from mediapipe.python.solutions import drawing_utils as mp_drawing
        from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
    except Exception:
        mp_solutions = getattr(mp, 'solutions', None)
        if mp_solutions is not None:
            mp_face_mesh = getattr(mp_solutions, 'face_mesh', None)
            mp_drawing = getattr(mp_solutions, 'drawing_utils', None)
            mp_drawing_styles = getattr(mp_solutions, 'drawing_styles', None)

    if mp_face_mesh is not None:
        MP_AVAILABLE = True
        print("[INFO] MediaPipe Face Mesh module loaded successfully.")
    else:
        mp_import_error = "mp_face_mesh is None"
        print("[WARNING] MediaPipe loaded but face_mesh solution was not found.")
except Exception as e:
    mp_import_error = str(e)
    print(f"[WARNING] MediaPipe Face Mesh not available on cloud host: {e}")
    MP_AVAILABLE = False


class FaceDetector:
    """Detects face presence, 468+ facial landmarks, and computes the face bounding box."""

    def __init__(self, min_detection_confidence: float = 0.45, min_tracking_confidence: float = 0.45):
        self.mp_face_mesh = mp_face_mesh
        self.face_mesh = None
        self.init_error = None
        if MP_AVAILABLE and self.mp_face_mesh is not None:
            try:
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                print("[INFO] MediaPipe FaceMesh model initialized successfully.")
            except Exception as e:
                self.init_error = str(e)
                print(f"[WARNING] Failed to initialize FaceMesh object: {e}")
                self.face_mesh = None

        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles

        # Fallback Haar Cascade face detector
        self.haar_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            print(f"[DEBUG] Haar cascade initialization notice: {e}")

    def process(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        h, w, _ = frame_bgr.shape

        # 1. Primary detection: MediaPipe Face Mesh
        if MP_AVAILABLE and self.face_mesh is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            try:
                results = self.face_mesh.process(frame_rgb)
            except Exception as e:
                print(f"[DEBUG] FaceMesh.process error: {e}")
                results = None

            if results and getattr(results, 'multi_face_landmarks', None) and len(results.multi_face_landmarks) > 0:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks_norm = []
                landmarks_px = []
                x_coords = []
                y_coords = []

                for lm in face_landmarks.landmark:
                    landmarks_norm.append((lm.x, lm.y, lm.z))
                    px_x, px_y = int(lm.x * w), int(lm.y * h)
                    landmarks_px.append((px_x, px_y, lm.z * w))
                    x_coords.append(px_x)
                    y_coords.append(px_y)

                x_min = max(0, min(x_coords) - 10)
                y_min = max(0, min(y_coords) - 10)
                x_max = min(w, max(x_coords) + 10)
                y_max = min(h, max(y_coords) + 10)
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

                return {
                    'face_detected': True,
                    'landmarks_norm': landmarks_norm,
                    'landmarks_px': landmarks_px,
                    'bbox': bbox,
                    'mesh_results': results
                }

        # 2. Fallback: Haar Cascade face detection if FaceMesh has no detections or is unavailable
        if self.haar_cascade is not None:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60))
                if len(faces) > 0:
                    x, y, fw, fh = faces[0]
                    # Synthesize approximate center face landmarks so telemetry stays active
                    center_x, center_y = x + fw // 2, y + fh // 2
                    norm_cx, norm_cy = center_x / w, center_y / h
                    # Create baseline 478 landmarks centered on face
                    synthetic_norm = [(norm_cx, norm_cy, 0.0)] * 478
                    synthetic_px = [(center_x, center_y, 0.0)] * 478

                    return {
                        'face_detected': True,
                        'landmarks_norm': synthetic_norm,
                        'landmarks_px': synthetic_px,
                        'bbox': (int(x), int(y), int(fw), int(fh)),
                        'mesh_results': None
                    }
            except Exception as e:
                print(f"[DEBUG] Haar cascade detection notice: {e}")

        return {
            'face_detected': False,
            'landmarks_norm': [],
            'landmarks_px': [],
            'bbox': None,
            'mesh_results': None
        }

    def draw_face_mesh(self, frame: np.ndarray, mesh_results, draw_contours: bool = True) -> np.ndarray:
        if MP_AVAILABLE and self.mp_drawing and self.mp_face_mesh and mesh_results and getattr(mesh_results, 'multi_face_landmarks', None):
            try:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    if draw_contours:
                        self.mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                    else:
                        # Draw subtle tesselation / contours spec
                        self.mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=1, circle_radius=1)
                        )
            except Exception:
                pass
        return frame

    def draw_bounding_box(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], status_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        if bbox is None:
            return frame
        x, y, w, h = bbox
        corner_len = min(25, w // 4, h // 4)
        thickness = 2
        cv2.line(frame, (x, y), (x + corner_len, y), status_color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_len), status_color, thickness)
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), status_color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), status_color, thickness)
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), status_color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), status_color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), status_color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), status_color, thickness)
        return frame
