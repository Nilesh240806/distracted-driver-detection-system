"""
Face Detector Module.
Uses MediaPipe Face Mesh to detect facial landmarks and bounding box in real time.
Includes fail-safe fallback for headless cloud servers (Render / Heroku / AWS).
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

MP_AVAILABLE = False
mp_face_mesh = None
mp_drawing = None
mp_drawing_styles = None

try:
    import mediapipe as mp
    mp_solutions = getattr(mp, 'solutions', None)
    if mp_solutions is None:
        try:
            import mediapipe.python.solutions as mp_solutions
        except Exception:
            import mediapipe.solutions as mp_solutions
    mp_face_mesh = getattr(mp_solutions, 'face_mesh', None)
    mp_drawing = getattr(mp_solutions, 'drawing_utils', None)
    mp_drawing_styles = getattr(mp_solutions, 'drawing_styles', None)
    if mp_face_mesh is not None:
        MP_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] MediaPipe Face Mesh not available on cloud host: {e}")
    MP_AVAILABLE = False


class FaceDetector:
    """Detects face presence, 468+ facial landmarks, and computes the face bounding box."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_face_mesh = mp_face_mesh
        self.face_mesh = None
        if MP_AVAILABLE and self.mp_face_mesh is not None:
            try:
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
            except Exception as e:
                print(f"[WARNING] Failed to initialize FaceMesh object: {e}")
                self.face_mesh = None

        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles

    def process(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        if not MP_AVAILABLE or self.face_mesh is None:
            return {
                'face_detected': False,
                'landmarks_norm': [],
                'landmarks_px': [],
                'bbox': None,
                'mesh_results': None
            }

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        try:
            results = self.face_mesh.process(frame_rgb)
        except Exception as e:
            return {
                'face_detected': False,
                'landmarks_norm': [],
                'landmarks_px': [],
                'bbox': None,
                'mesh_results': None
            }

        if not results or not getattr(results, 'multi_face_landmarks', None):
            return {
                'face_detected': False,
                'landmarks_norm': [],
                'landmarks_px': [],
                'bbox': None,
                'mesh_results': results
            }

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
