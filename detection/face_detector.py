"""
Face Detector Module.
Uses MediaPipe Face Mesh to detect facial landmarks and bounding box in real time.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List, Dict, Any

mp_solutions = getattr(mp, 'solutions', None)
if mp_solutions is None:
    try:
        import mediapipe.python.solutions as mp_solutions
    except ModuleNotFoundError:
        import mediapipe.solutions as mp_solutions

mp_face_mesh = mp_solutions.face_mesh
mp_drawing = mp_solutions.drawing_utils
mp_drawing_styles = mp_solutions.drawing_styles


class FaceDetector:
    """Detects face presence, 468+ facial landmarks, and computes the face bounding box."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_face_mesh = mp_face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles

    def process(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Processes a BGR video frame and returns face detection metadata.
        
        Args:
            frame_bgr: OpenCV BGR image (H, W, 3)
            
        Returns:
            Dict containing:
                - face_detected: bool
                - landmarks_norm: List of (x, y, z) in [0, 1]
                - landmarks_px: List of (x, y, z) in pixel coords
                - bbox: (x_min, y_min, width, height) in pixel coords
                - mesh_results: raw mediapipe result for custom drawing
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
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

        # Compute tight bounding box with slight margin
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
        """Draws aesthetic subtle face mesh tessellation and contours on frame."""
        if mesh_results and mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                if draw_contours:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
        return frame

    def draw_bounding_box(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], status_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draws automotive-styled corner brackets around the detected face."""
        if bbox is None:
            return frame
            
        x, y, w, h = bbox
        corner_len = min(25, w // 4, h // 4)
        thickness = 2
        
        # Draw 4 aesthetic corner brackets
        # Top-Left
        cv2.line(frame, (x, y), (x + corner_len, y), status_color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_len), status_color, thickness)
        # Top-Right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), status_color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), status_color, thickness)
        # Bottom-Left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), status_color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), status_color, thickness)
        # Bottom-Right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), status_color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), status_color, thickness)

        return frame
