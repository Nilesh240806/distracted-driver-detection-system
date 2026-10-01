"""
Yawn Detector & Mouth Aspect Ratio (MAR) Module.
Calculates Mouth Aspect Ratio (MAR) to detect sustained yawning while filtering brief speech movements.
"""

import time
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional


class YawnDetector:
    """
    Computes Mouth Aspect Ratio (MAR) using upper and lower lip landmark distances.
    
    Formula:
        MAR = ( ||p13 - p14|| + ||p37 - p84|| + ||p267 - p314|| ) / ( 3 * ||p61 - p291|| )
    """

    # Key landmark indices in MediaPipe Face Mesh for mouth
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    UPPER_LIP_MID = 13
    LOWER_LIP_MID = 14
    UPPER_LIP_LEFT = 37
    LOWER_LIP_LEFT = 84
    UPPER_LIP_RIGHT = 267
    LOWER_LIP_RIGHT = 314

    MOUTH_OUTLINE_INDICES = [61, 37, 0, 267, 291, 314, 17, 84]

    def __init__(self, mar_threshold: float = 0.55, yawn_duration_threshold_sec: float = 1.2):
        self.mar_threshold = mar_threshold
        self.yawn_duration_threshold_sec = yawn_duration_threshold_sec

        # Temporal state tracking
        self.is_mouth_wide: bool = False
        self.mouth_open_start_time: Optional[float] = None
        self.yawn_duration: float = 0.0
        self.yawn_detected: bool = False
        self.total_yawns: int = 0
        self.alert_triggered: bool = False

    def calculate_mar(self, landmarks_norm: List[Tuple[float, float, float]]) -> float:
        """Calculates normalized MAR value."""
        if not landmarks_norm or len(landmarks_norm) < 468:
            return 0.0

        p_left = np.array(landmarks_norm[self.MOUTH_LEFT][:2], dtype=np.float32)
        p_right = np.array(landmarks_norm[self.MOUTH_RIGHT][:2], dtype=np.float32)
        p_top_mid = np.array(landmarks_norm[self.UPPER_LIP_MID][:2], dtype=np.float32)
        p_bot_mid = np.array(landmarks_norm[self.LOWER_LIP_MID][:2], dtype=np.float32)
        p_top_l = np.array(landmarks_norm[self.UPPER_LIP_LEFT][:2], dtype=np.float32)
        p_bot_l = np.array(landmarks_norm[self.LOWER_LIP_LEFT][:2], dtype=np.float32)
        p_top_r = np.array(landmarks_norm[self.UPPER_LIP_RIGHT][:2], dtype=np.float32)
        p_bot_r = np.array(landmarks_norm[self.LOWER_LIP_RIGHT][:2], dtype=np.float32)

        # Vertical spans
        v_mid = np.linalg.norm(p_top_mid - p_bot_mid)
        v_l = np.linalg.norm(p_top_l - p_bot_l)
        v_r = np.linalg.norm(p_top_r - p_bot_r)
        v_avg = (v_mid + v_l + v_r) / 3.0

        # Horizontal width
        h_width = np.linalg.norm(p_left - p_right)
        if h_width < 1e-6:
            return 0.0

        mar = v_avg / h_width
        return float(mar)

    def process(self, landmarks_norm: List[Tuple[float, float, float]], current_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyzes mouth geometry to detect sustained yawning.
        """
        if current_time is None:
            current_time = time.time()

        if not landmarks_norm or len(landmarks_norm) < 468:
            self.is_mouth_wide = False
            self.mouth_open_start_time = None
            self.yawn_duration = 0.0
            self.yawn_detected = False
            self.alert_triggered = False
            return {
                'mar': 0.0,
                'mar_threshold': self.mar_threshold,
                'is_yawning': False,
                'yawn_duration': 0.0,
                'threshold_sec': self.yawn_duration_threshold_sec,
                'yawn_detected': False,
                'new_alert_event': False,
                'total_yawns': self.total_yawns
            }

        mar = self.calculate_mar(landmarks_norm)
        is_wide = mar > self.mar_threshold
        new_alert_event = False

        if is_wide:
            if not self.is_mouth_wide:
                # Just opened mouth wide
                self.is_mouth_wide = True
                self.mouth_open_start_time = current_time

            self.yawn_duration = current_time - (self.mouth_open_start_time or current_time)

            if self.yawn_duration >= self.yawn_duration_threshold_sec:
                self.yawn_detected = True
                if not self.alert_triggered:
                    self.alert_triggered = True
                    new_alert_event = True
                    self.total_yawns += 1
        else:
            # Mouth closed / normal
            self.is_mouth_wide = False
            self.mouth_open_start_time = None
            self.yawn_duration = 0.0
            self.yawn_detected = False
            self.alert_triggered = False

        return {
            'mar': round(mar, 3),
            'mar_threshold': self.mar_threshold,
            'is_yawning': self.yawn_detected,
            'yawn_duration': round(self.yawn_duration, 2),
            'threshold_sec': self.yawn_duration_threshold_sec,
            'yawn_detected': self.yawn_detected,
            'new_alert_event': new_alert_event,
            'total_yawns': self.total_yawns
        }

    def draw_mouth_overlay(self, frame: np.ndarray, landmarks_px: List[Tuple[int, int, float]], is_yawning: bool) -> np.ndarray:
        """Draws lip contours highlighting yawn status."""
        if not landmarks_px or len(landmarks_px) < 468:
            return frame

        color = (0, 140, 255) if is_yawning else (200, 200, 200)  # Orange for yawn
        mouth_pts = np.array([[landmarks_px[i][0], landmarks_px[i][1]] for i in self.MOUTH_OUTLINE_INDICES if i < len(landmarks_px)], dtype=np.int32)
        if len(mouth_pts) > 2:
            cv2.polylines(frame, [mouth_pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        return frame
