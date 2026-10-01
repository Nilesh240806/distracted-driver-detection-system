"""
Eye Detector & Eye Aspect Ratio (EAR) Module.
Calculates Eye Aspect Ratio (EAR) for both eyes to detect blinks, eye closure, and drowsiness.
"""

import time
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional


class EyeDetector:
    """
    Tracks eye state using Eye Aspect Ratio (EAR) calculated from 6 facial landmarks per eye.
    
    Formula:
        EAR = ( ||p2 - p6|| + ||p3 - p5|| ) / ( 2 * ||p1 - p4|| )
    """

    # MediaPipe Face Mesh Landmark Indices for Eyes
    # Left eye: 33 (outer), 160 (top outer), 158 (top inner), 133 (inner), 153 (bottom inner), 144 (bottom outer)
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    # Right eye: 362 (inner), 385 (top inner), 387 (top outer), 263 (outer), 373 (bottom outer), 380 (bottom inner)
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def __init__(self, ear_threshold: float = 0.22, closure_threshold_sec: float = 2.0):
        self.ear_threshold = ear_threshold
        self.closure_threshold_sec = closure_threshold_sec
        
        # Temporal state tracking
        self.eye_closed_start_time: Optional[float] = None
        self.eye_closure_duration: float = 0.0
        self.is_closed: bool = False
        self.drowsiness_alert_triggered: bool = False
        
        # Blink tracking
        self.total_blinks: int = 0
        self._blink_candidate: bool = False
        self._blink_start_time: Optional[float] = None

    @staticmethod
    def calculate_ear(eye_landmarks: List[Tuple[float, float, float]]) -> float:
        """
        Calculates EAR from a list of 6 landmark points [(x, y, z), ...].
        """
        if len(eye_landmarks) < 6:
            return 0.30

        p1 = np.array(eye_landmarks[0][:2], dtype=np.float32)
        p2 = np.array(eye_landmarks[1][:2], dtype=np.float32)
        p3 = np.array(eye_landmarks[2][:2], dtype=np.float32)
        p4 = np.array(eye_landmarks[3][:2], dtype=np.float32)
        p5 = np.array(eye_landmarks[4][:2], dtype=np.float32)
        p6 = np.array(eye_landmarks[5][:2], dtype=np.float32)

        # Vertical distances
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)

        # Horizontal distance
        h = np.linalg.norm(p1 - p4)

        if h < 1e-6:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return float(ear)

    def process(self, landmarks_norm: List[Tuple[float, float, float]], current_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyzes eye landmarks and updates eye closure durations and drowsiness status.
        
        Args:
            landmarks_norm: Normalized landmarks from FaceDetector
            current_time: Optional timestamp (defaults to time.time())
            
        Returns:
            Dict containing:
                - left_ear: float
                - right_ear: float
                - avg_ear: float
                - eye_status: 'OPEN' or 'CLOSED'
                - closure_time: float (seconds)
                - threshold_sec: float
                - drowsiness_alert: bool
                - new_alert_event: bool (true only once when threshold crossed)
                - total_blinks: int
        """
        if current_time is None:
            current_time = time.time()

        if not landmarks_norm or len(landmarks_norm) < 468:
            # No face detected
            self.eye_closed_start_time = None
            self.eye_closure_duration = 0.0
            self.is_closed = False
            self.drowsiness_alert_triggered = False
            return {
                'left_ear': 0.0,
                'right_ear': 0.0,
                'avg_ear': 0.0,
                'eye_status': 'UNKNOWN',
                'closure_time': 0.0,
                'threshold_sec': self.closure_threshold_sec,
                'drowsiness_alert': False,
                'new_alert_event': False,
                'total_blinks': self.total_blinks
            }

        left_points = [landmarks_norm[i] for i in self.LEFT_EYE_INDICES]
        right_points = [landmarks_norm[i] for i in self.RIGHT_EYE_INDICES]

        left_ear = self.calculate_ear(left_points)
        right_ear = self.calculate_ear(right_points)
        avg_ear = (left_ear + right_ear) / 2.0

        new_alert_event = False
        is_currently_closed = avg_ear < self.ear_threshold

        if is_currently_closed:
            if not self.is_closed:
                # Just transitioned to closed
                self.is_closed = True
                self.eye_closed_start_time = current_time
                self._blink_start_time = current_time
                self._blink_candidate = True

            self.eye_closure_duration = current_time - (self.eye_closed_start_time or current_time)

            # Check if threshold crossed
            if self.eye_closure_duration >= self.closure_threshold_sec:
                if not self.drowsiness_alert_triggered:
                    self.drowsiness_alert_triggered = True
                    new_alert_event = True
        else:
            # Eyes are open
            if self.is_closed:
                # Was closed, now open
                duration = current_time - (self.eye_closed_start_time or current_time)
                # Short closure (<0.45s) counts as a natural blink
                if 0.05 <= duration < 0.45:
                    self.total_blinks += 1

            self.is_closed = False
            self.eye_closed_start_time = None
            self.eye_closure_duration = 0.0
            self.drowsiness_alert_triggered = False
            self._blink_candidate = False

        return {
            'left_ear': round(left_ear, 3),
            'right_ear': round(right_ear, 3),
            'avg_ear': round(avg_ear, 3),
            'eye_status': 'CLOSED' if is_currently_closed else 'OPEN',
            'closure_time': round(self.eye_closure_duration, 2),
            'threshold_sec': self.closure_threshold_sec,
            'drowsiness_alert': self.drowsiness_alert_triggered,
            'new_alert_event': new_alert_event,
            'total_blinks': self.total_blinks
        }

    def draw_eye_overlays(self, frame: np.ndarray, landmarks_px: List[Tuple[int, int, float]], eye_status: str) -> np.ndarray:
        """Draws glowing contours around both eyes with color indicating open/closed state."""
        if not landmarks_px or len(landmarks_px) < 468:
            return frame

        color = (0, 0, 255) if eye_status == 'CLOSED' else (0, 255, 128)  # Red for closed, Cyan-Green for open

        # Left Eye
        left_pts = np.array([[landmarks_px[i][0], landmarks_px[i][1]] for i in self.LEFT_EYE_INDICES], dtype=np.int32)
        cv2.polylines(frame, [left_pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

        # Right Eye
        right_pts = np.array([[landmarks_px[i][0], landmarks_px[i][1]] for i in self.RIGHT_EYE_INDICES], dtype=np.int32)
        cv2.polylines(frame, [right_pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

        return frame
