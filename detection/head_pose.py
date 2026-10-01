"""
Head Pose Estimator Module.
Estimates 3D head orientation (Yaw, Pitch, Roll) and classifies head direction into CENTER, LEFT, or RIGHT.
Tracks prolonged distraction when looking away from the road.
"""

import time
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class HeadPoseEstimator:
    """
    Computes 3D head rotation angles using OpenCV's solvePnP and facial landmark geometry.
    Classifies orientation into 'CENTER', 'LEFT', 'RIGHT' and tracks looking-away duration.
    """

    # Key landmark indices in MediaPipe Face Mesh
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_CORNER = 33
    RIGHT_EYE_CORNER = 263
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291

    # Approximate 3D generic facial model coordinates in millimeters
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)

    def __init__(self, yaw_threshold_deg: float = 16.0, turn_threshold_sec: float = 3.0):
        self.yaw_threshold_deg = yaw_threshold_deg
        self.turn_threshold_sec = turn_threshold_sec

        # Temporal tracking
        self.current_direction: str = "CENTER"
        self.direction_start_time: Optional[float] = None
        self.turn_duration: float = 0.0
        self.alert_triggered: bool = False

    def estimate_pose(self, landmarks_px: List[Tuple[int, int, float]], frame_shape: Tuple[int, int, int]) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Estimates Euler angles (yaw, pitch, roll) in degrees from pixel landmarks.
        """
        h, w, _ = frame_shape
        image_points = np.array([
            (landmarks_px[self.NOSE_TIP][0], landmarks_px[self.NOSE_TIP][1]),
            (landmarks_px[self.CHIN][0], landmarks_px[self.CHIN][1]),
            (landmarks_px[self.LEFT_EYE_CORNER][0], landmarks_px[self.LEFT_EYE_CORNER][1]),
            (landmarks_px[self.RIGHT_EYE_CORNER][0], landmarks_px[self.RIGHT_EYE_CORNER][1]),
            (landmarks_px[self.LEFT_MOUTH_CORNER][0], landmarks_px[self.LEFT_MOUTH_CORNER][1]),
            (landmarks_px[self.RIGHT_MOUTH_CORNER][0], landmarks_px[self.RIGHT_MOUTH_CORNER][1]),
        ], dtype=np.float64)

        # Approximate camera matrix based on frame dimensions
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0, np.zeros((3, 1)), np.zeros((3, 1))

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose projection matrix to obtain Euler angles
        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        return yaw, pitch, roll, rvec, tvec

    def process(self, landmarks_px: List[Tuple[int, int, float]], frame_shape: Tuple[int, int, int], current_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculates head direction, duration of distraction, and triggers alerts if looking away > 3s.
        """
        if current_time is None:
            current_time = time.time()

        if not landmarks_px or len(landmarks_px) < 468:
            self.current_direction = "UNKNOWN"
            self.direction_start_time = None
            self.turn_duration = 0.0
            self.alert_triggered = False
            return {
                'direction': 'UNKNOWN',
                'yaw': 0.0,
                'pitch': 0.0,
                'roll': 0.0,
                'turn_time': 0.0,
                'threshold_sec': self.turn_threshold_sec,
                'head_alert': False,
                'new_alert_event': False
            }

        yaw, pitch, roll, rvec, tvec = self.estimate_pose(landmarks_px, frame_shape)

        # Secondary geometry ratio for extra stability
        nose_x = landmarks_px[self.NOSE_TIP][0]
        left_x = landmarks_px[self.LEFT_EYE_CORNER][0]
        right_x = landmarks_px[self.RIGHT_EYE_CORNER][0]
        eye_mid_x = (left_x + right_x) / 2.0
        eye_span = max(abs(right_x - left_x), 1.0)
        geom_ratio = (nose_x - eye_mid_x) / eye_span

        # Classify direction: Yaw threshold and geom_ratio confirmation
        # In mirrored webcam / solvePnP:
        # Looking Left (from driver's POV) or Looking Right (from driver's POV)
        if yaw < -self.yaw_threshold_deg or geom_ratio < -0.18:
            detected_dir = "LEFT"
        elif yaw > self.yaw_threshold_deg or geom_ratio > 0.18:
            detected_dir = "RIGHT"
        else:
            detected_dir = "CENTER"

        new_alert_event = False

        if detected_dir in ("LEFT", "RIGHT"):
            if self.current_direction != detected_dir:
                # Started turning in a new direction
                self.current_direction = detected_dir
                self.direction_start_time = current_time
                self.alert_triggered = False

            self.turn_duration = current_time - (self.direction_start_time or current_time)

            if self.turn_duration >= self.turn_threshold_sec:
                if not self.alert_triggered:
                    self.alert_triggered = True
                    new_alert_event = True
        else:
            # Returned to CENTER
            self.current_direction = "CENTER"
            self.direction_start_time = None
            self.turn_duration = 0.0
            self.alert_triggered = False

        return {
            'direction': self.current_direction,
            'yaw': round(yaw, 1),
            'pitch': round(pitch, 1),
            'roll': round(roll, 1),
            'turn_time': round(self.turn_duration, 2),
            'threshold_sec': self.turn_threshold_sec,
            'head_alert': self.alert_triggered,
            'new_alert_event': new_alert_event,
            'rvec': rvec,
            'tvec': tvec
        }

    def draw_head_direction(self, frame: np.ndarray, landmarks_px: List[Tuple[int, int, float]], rvec: np.ndarray, tvec: np.ndarray, direction: str) -> np.ndarray:
        """Projects a 3D nose orientation axis line to visualize driver's gaze line."""
        if not landmarks_px or rvec is None or tvec is None or len(landmarks_px) < 468:
            return frame

        h, w, _ = frame.shape
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 3D axis points starting from nose tip projecting forward (Z axis)
        axis_3d = np.array([
            (0.0, 0.0, 0.0),       # Nose origin
            (0.0, 0.0, 250.0),     # Forward gaze line
        ], dtype=np.float64)

        try:
            imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
            p_start = (int(imgpts[0].ravel()[0]), int(imgpts[0].ravel()[1]))
            p_end = (int(imgpts[1].ravel()[0]), int(imgpts[1].ravel()[1]))

            color = (0, 255, 0) if direction == "CENTER" else (0, 165, 255) if direction in ("LEFT", "RIGHT") else (0, 0, 255)
            cv2.line(frame, p_start, p_end, color, 3, cv2.LINE_AA)
            cv2.circle(frame, p_start, 4, (255, 255, 0), -1)
        except Exception:
            pass

        return frame
