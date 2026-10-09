"""
Hand Detector Module.
Uses MediaPipe Hands to detect visible driver hands on the steering wheel zone.
Includes fail-safe fallback for headless cloud servers (Render / Heroku / AWS).
"""

import time
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

MP_HANDS_AVAILABLE = False
mp_hands = None
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
    mp_hands = getattr(mp_solutions, 'hands', None)
    mp_drawing = getattr(mp_solutions, 'drawing_utils', None)
    mp_drawing_styles = getattr(mp_solutions, 'drawing_styles', None)
    if mp_hands is not None:
        MP_HANDS_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] MediaPipe Hands not available on cloud host: {e}")
    MP_HANDS_AVAILABLE = False


class HandDetector:
    """
    Detects hands in the video frame, counts visible hands (0, 1, 2),
    and enforces single-hand driving and hands-off-wheel safety thresholds.
    """

    def __init__(self, one_hand_threshold_sec: float = 2.0, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.one_hand_threshold_sec = one_hand_threshold_sec
        self.mp_hands = mp_hands
        self.hands = None
        if MP_HANDS_AVAILABLE and self.mp_hands is not None:
            try:
                self.hands = self.mp_hands.Hands(
                    max_num_hands=2,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
            except Exception as e:
                print(f"[WARNING] Failed to initialize Hands object: {e}")
                self.hands = None

        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles

        # Temporal state tracking
        self.current_state: str = "TWO HANDS"
        self.state_start_time: Optional[float] = None
        self.state_duration: float = 0.0
        self.single_hand_alert_triggered: bool = False
        self.no_hands_alert_triggered: bool = False

    def process(self, frame_bgr: np.ndarray, current_time: Optional[float] = None) -> Dict[str, Any]:
        if current_time is None:
            current_time = time.time()

        if not MP_HANDS_AVAILABLE or self.hands is None:
            return {
                'num_hands': 2,
                'hand_landmarks_norm': [],
                'hand_landmarks_px': [],
                'hand_state': 'TWO HANDS',
                'state_duration': 0.0,
                'alert_level': 'NONE',
                'alert_message': 'System Active',
                'hands_results': None
            }

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        try:
            results = self.hands.process(frame_rgb)
        except Exception:
            return {
                'num_hands': 2,
                'hand_landmarks_norm': [],
                'hand_landmarks_px': [],
                'hand_state': 'TWO HANDS',
                'state_duration': 0.0,
                'alert_level': 'NONE',
                'alert_message': 'System Active',
                'hands_results': None
            }

        num_hands = 0
        all_landmarks_norm = []
        all_landmarks_px = []

        if results and getattr(results, 'multi_hand_landmarks', None):
            num_hands = len(results.multi_hand_landmarks)
            for hand_lms in results.multi_hand_landmarks:
                hand_norm = []
                hand_px = []
                for lm in hand_lms.landmark:
                    hand_norm.append((lm.x, lm.y, lm.z))
                    hand_px.append((int(lm.x * w), int(lm.y * h), lm.z * w))
                all_landmarks_norm.append(hand_norm)
                all_landmarks_px.append(hand_px)

        # Classify hand state
        if num_hands >= 2:
            new_state = "TWO HANDS"
        elif num_hands == 1:
            new_state = "ONE HAND"
        else:
            new_state = "NO HANDS"

        # Temporal duration tracking
        if new_state != self.current_state:
            self.current_state = new_state
            self.state_start_time = current_time
            self.state_duration = 0.0
            self.single_hand_alert_triggered = False
            self.no_hands_alert_triggered = False
        else:
            if self.state_start_time is None:
                self.state_start_time = current_time
            self.state_duration = current_time - self.state_start_time

        # Determine alert escalation
        alert_level = "NONE"
        alert_message = f"Hands Status: {self.current_state}"
        
        if self.current_state == "ONE HAND" and self.state_duration >= self.one_hand_threshold_sec:
            alert_level = "WARNING"
            alert_message = f"SINGLE-HAND DRIVING ({self.state_duration:.1f}s)"
        elif self.current_state == "NO HANDS" and self.state_duration >= 1.5:
            alert_level = "CRITICAL"
            alert_message = f"HANDS OFF WHEEL! ({self.state_duration:.1f}s)"

        return {
            'num_hands': num_hands,
            'hand_landmarks_norm': all_landmarks_norm,
            'hand_landmarks_px': all_landmarks_px,
            'hand_state': self.current_state,
            'state_duration': round(self.state_duration, 2),
            'alert_level': alert_level,
            'alert_message': alert_message,
            'hands_results': results
        }

    def draw_hands(self, frame: np.ndarray, hands_results) -> np.ndarray:
        if MP_HANDS_AVAILABLE and self.mp_drawing and self.mp_hands and hands_results and getattr(hands_results, 'multi_hand_landmarks', None):
            try:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            except Exception:
                pass
        return frame
