"""
Hand Detector Module.
Uses MediaPipe Hands to detect visible driver hands on the steering wheel zone.
Classifies into TWO HANDS, ONE HAND, or NO HANDS, and tracks single-hand / no-hands duration.
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Dict, Any, Optional


class HandDetector:
    """
    Detects hands in the video frame, counts visible hands (0, 1, 2),
    and enforces single-hand driving and hands-off-wheel safety thresholds.
    """

    def __init__(self, one_hand_threshold_sec: float = 2.0, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.one_hand_threshold_sec = one_hand_threshold_sec
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Temporal state tracking
        self.current_state: str = "TWO HANDS"  # 'TWO HANDS', 'ONE HAND', 'NO HANDS'
        self.state_start_time: Optional[float] = None
        self.state_duration: float = 0.0
        self.single_hand_alert_triggered: bool = False
        self.no_hands_alert_triggered: bool = False

    def process(self, frame_bgr: np.ndarray, current_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Processes frame to find hand landmarks and evaluate safety conditions.
        """
        if current_time is None:
            current_time = time.time()

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)

        num_hands = 0
        hand_boxes = []

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                x_pts = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_pts = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x_min, x_max = max(0, min(x_pts) - 5), min(w, max(x_pts) + 5)
                y_min, y_max = max(0, min(y_pts) - 5), min(h, max(y_pts) + 5)
                hand_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

        if num_hands >= 2:
            state = "TWO HANDS"
        elif num_hands == 1:
            state = "ONE HAND"
        else:
            state = "NO HANDS"

        new_alert_event = False
        alert_msg = ""
        severity = "INFO"

        if state != self.current_state:
            # Transitioned to different hand state
            self.current_state = state
            self.state_start_time = current_time
            self.state_duration = 0.0
            self.single_hand_alert_triggered = False
            self.no_hands_alert_triggered = False
        else:
            self.state_duration = current_time - (self.state_start_time or current_time)

        # Check thresholds
        if state == "ONE HAND":
            if self.state_duration >= self.one_hand_threshold_sec:
                if not self.single_hand_alert_triggered:
                    self.single_hand_alert_triggered = True
                    new_alert_event = True
                    alert_msg = "WARNING: SINGLE-HAND DRIVING DETECTED"
                    severity = "WARNING"
        elif state == "NO HANDS":
            if self.state_duration >= self.one_hand_threshold_sec:
                if not self.no_hands_alert_triggered:
                    self.no_hands_alert_triggered = True
                    new_alert_event = True
                    alert_msg = "CRITICAL: HANDS NOT DETECTED"
                    severity = "CRITICAL"
        else:
            # TWO HANDS -> fully safe
            self.single_hand_alert_triggered = False
            self.no_hands_alert_triggered = False

        return {
            'num_hands': num_hands,
            'state': self.current_state,
            'hand_time': round(self.state_duration, 2),
            'threshold_sec': self.one_hand_threshold_sec,
            'boxes': hand_boxes,
            'hands_results': results,
            'single_hand_alert': self.single_hand_alert_triggered,
            'no_hands_alert': self.no_hands_alert_triggered,
            'new_alert_event': new_alert_event,
            'alert_msg': alert_msg,
            'severity': severity
        }

    def draw_hands(self, frame: np.ndarray, hands_results) -> np.ndarray:
        """Draws hand skeleton and futuristic target joints on detected hands."""
        if hands_results and hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return frame
