"""
Distraction Engine Module.
Central coordinator orchestrating Face, Eye, Head Pose, Hand, and Yawn detectors.
Computes Distraction Score (0-100), runs the multi-tier Safety State Machine, manages event logging,
and renders real-time automotive HUD telemetry overlays on video frames.
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable

from .face_detector import FaceDetector
from .eye_detector import EyeDetector
from .head_pose import HeadPoseEstimator
from .hand_detector import HandDetector
from .yawn_detector import YawnDetector
import database


class DistractionEngine:
    """
    Main Computer Vision & Safety Analysis Engine.
    """

    # Safety State Machine constants
    STATE_NORMAL = "NORMAL"
    STATE_WARNING = "WARNING"
    STATE_CRITICAL = "CRITICAL"
    STATE_EMERGENCY = "EMERGENCY_SIMULATION"

    # Emergency escalation threshold (seconds in critical state)
    EMERGENCY_THRESHOLD_SEC = 5.0

    def __init__(self,
                 ear_thresh: float = 0.22,
                 eye_closure_sec: float = 1.8,
                 yaw_thresh_deg: float = 15.0,
                 head_turn_sec: float = 2.5,
                 one_hand_sec: float = 2.0,
                 mar_thresh: float = 0.52,
                 yawn_sec: float = 1.0):
        
        # Initialize sub-detectors
        self.face_detector = FaceDetector()
        self.eye_detector = EyeDetector(ear_threshold=ear_thresh, closure_threshold_sec=eye_closure_sec)
        self.head_pose = HeadPoseEstimator(yaw_threshold_deg=yaw_thresh_deg, turn_threshold_sec=head_turn_sec)
        self.hand_detector = HandDetector(one_hand_threshold_sec=one_hand_sec)
        self.yawn_detector = YawnDetector(mar_threshold=mar_thresh, yawn_duration_threshold_sec=yawn_sec)

        # Safety State Machine
        self.current_state = self.STATE_NORMAL
        self.driver_status = "SAFE"  # 'SAFE', 'WARNING', 'CRITICAL'
        self.critical_start_time: Optional[float] = None
        self.emergency_triggered: bool = False
        self.current_alert_msg: str = "Driver monitoring active"
        self.current_severity: str = "INFO"

        # Distraction Score (0 - 100)
        self.distraction_score: int = 0
        self.distraction_level: str = "Normal"

        # Live Real-time Camera Detection Mode (Pure Live Production)
        self.demo_mode: bool = False
        self.simulation_override: Optional[Dict[str, Any]] = None
        self.simulation_end_time: float = 0.0

        # Performance & FPS calculation
        self.fps: float = 0.0
        self._prev_frame_time: float = time.time()
        self._frame_count: int = 0

        # Event notification callback (e.g. SocketIO emission)
        self.event_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def set_event_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Sets an external callback for real-time alert broadcasts."""
        self.event_callback = callback

    def trigger_simulation(self, scenario: str, duration_sec: float = 8.0) -> Dict[str, Any]:
        """
        Activates a simulated driver behavior scenario for viva demonstration.
        Scenarios: 'drowsiness', 'head_left', 'head_right', 'single_hand', 'yawning', 'critical_emergency', 'reset'
        """
        now = time.time()
        if scenario == 'reset':
            self.simulation_override = None
            self.simulation_end_time = 0.0
            self.current_state = self.STATE_NORMAL
            self.driver_status = "SAFE"
            self.emergency_triggered = False
            return {'status': 'Simulation reset to normal'}

        self.simulation_end_time = now + duration_sec
        self.simulation_override = {
            'scenario': scenario,
            'start_time': now,
            'duration': duration_sec
        }
        return {'status': f'Simulating scenario: {scenario} for {duration_sec}s'}

    def compute_distraction_score(self,
                                   face_detected: bool,
                                   eye_status: str,
                                   head_dir: str,
                                   hand_state: str,
                                   is_yawning: bool) -> Tuple[int, str]:
        """
        Calculates rule-based distraction score (0-100) and severity category.
        
        Rules:
            Eyes closed: +50
            Looking left/right: +30
            One-hand driving: +20
            Yawning: +20
            No face detected: +40
            No hands detected: +40
        """
        score = 0
        if not face_detected:
            score += 40
        else:
            if eye_status == 'CLOSED':
                score += 50
            if head_dir in ('LEFT', 'RIGHT'):
                score += 30
            if is_yawning:
                score += 20

        if hand_state == 'ONE HAND':
            score += 20
        elif hand_state == 'NO HANDS':
            score += 40

        # Clamp between 0 and 100
        score = max(0, min(100, score))

        if score <= 20:
            level = "Normal"
        elif score <= 40:
            level = "Low"
        elif score <= 60:
            level = "Moderate"
        elif score <= 80:
            level = "High"
        else:
            level = "Critical"

        return score, level

    def update_state_machine(self,
                             current_time: float,
                             eye_data: Dict[str, Any],
                             head_data: Dict[str, Any],
                             hand_data: Dict[str, Any],
                             yawn_data: Dict[str, Any],
                             face_detected: bool) -> Tuple[str, str, str, str]:
        """
        Evaluates safety logic and advances state machine:
        NORMAL -> WARNING -> CRITICAL -> EMERGENCY_SIMULATION
        """
        # Determine highest active condition
        is_critical = False
        is_warning = False
        active_alert = "Driver monitoring active"
        severity = "INFO"

        # Check Critical triggers
        if eye_data['drowsiness_alert']:
            is_critical = True
            active_alert = "DROWSINESS ALERT: EYES CLOSED FOR MORE THAN 2 SECONDS"
            severity = "CRITICAL"
        elif head_data['head_alert']:
            is_critical = True
            active_alert = f"DISTRACTION ALERT: DRIVER LOOKING {head_data['direction']}"
            severity = "CRITICAL"
        elif hand_data['no_hands_alert']:
            is_critical = True
            active_alert = "CRITICAL: HANDS NOT DETECTED ON STEERING ZONE"
            severity = "CRITICAL"
        elif not face_detected and eye_data.get('no_face_duration', 0) > 3.0:
            is_critical = True
            active_alert = "CRITICAL: DRIVER NOT VISIBLE"
            severity = "CRITICAL"

        # Check Warning triggers if not already critical
        if not is_critical:
            if hand_data['single_hand_alert']:
                is_warning = True
                active_alert = "WARNING: SINGLE-HAND DRIVING DETECTED"
                severity = "WARNING"
            elif yawn_data['yawn_detected']:
                is_warning = True
                active_alert = "WARNING: YAWNING DETECTED (FATIGUE INDICATOR)"
                severity = "WARNING"
            elif eye_data['closure_time'] > 1.0:
                is_warning = True
                active_alert = "WARNING: Prolonged Eye Closure Candidate"
                severity = "WARNING"
            elif head_data['turn_time'] > 1.5:
                is_warning = True
                active_alert = f"WARNING: Driver Turning {head_data['direction']}"
                severity = "WARNING"

        # State transitions
        if is_critical:
            if self.current_state != self.STATE_CRITICAL and self.current_state != self.STATE_EMERGENCY:
                self.current_state = self.STATE_CRITICAL
                self.critical_start_time = current_time
                self.emergency_triggered = False

            # Check for prolonged critical condition -> Emergency simulation
            crit_duration = current_time - (self.critical_start_time or current_time)
            if crit_duration >= self.EMERGENCY_THRESHOLD_SEC:
                self.current_state = self.STATE_EMERGENCY
                self.emergency_triggered = True
                active_alert = "EMERGENCY SAFETY ALERT: PROLONGED CRITICAL DISTRACTION"
                severity = "CRITICAL"

            driver_status = "CRITICAL"

        elif is_warning:
            self.current_state = self.STATE_WARNING
            self.critical_start_time = None
            self.emergency_triggered = False
            driver_status = "WARNING"

        else:
            # Safe / Normal
            self.current_state = self.STATE_NORMAL
            self.critical_start_time = None
            self.emergency_triggered = False
            driver_status = "SAFE"
            active_alert = "Driver monitoring active - Safe"
            severity = "INFO"

        return self.current_state, driver_status, active_alert, severity

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Main pipeline entry point: processes frame through all detectors,
        evaluates safety states, overlays telemetry, and builds output JSON.
        """
        current_time = time.time()
        h, w, _ = frame_bgr.shape

        # Calculate FPS
        self._frame_count += 1
        elapsed = current_time - self._prev_frame_time
        if elapsed >= 0.5:
            self.fps = round(self._frame_count / elapsed, 1)
            self._frame_count = 0
            self._prev_frame_time = current_time

        # Check simulation override
        if self.simulation_override and current_time < self.simulation_end_time:
            return self._handle_simulation_frame(frame_bgr, current_time)
        elif self.simulation_override and current_time >= self.simulation_end_time:
            self.simulation_override = None

        # 1. Face & Landmark Detection
        face_res = self.face_detector.process(frame_bgr)
        face_detected = face_res['face_detected']
        landmarks_norm = face_res['landmarks_norm']
        landmarks_px = face_res['landmarks_px']
        bbox = face_res['bbox']

        # 2. Eye Aspect Ratio (EAR) & Drowsiness
        eye_res = self.eye_detector.process(landmarks_norm, current_time)

        # 3. Head Pose & Direction
        head_res = self.head_pose.process(landmarks_px, frame_bgr.shape, current_time)

        # 4. Hand Detection & Steering Zone
        hand_res = self.hand_detector.process(frame_bgr, current_time)

        # 5. Yawning & Mouth Aspect Ratio (MAR)
        yawn_res = self.yawn_detector.process(landmarks_norm, current_time)

        # 6. Distraction Score
        dist_score, dist_level = self.compute_distraction_score(
            face_detected=face_detected,
            eye_status=eye_res['eye_status'],
            head_dir=head_res['direction'],
            hand_state=hand_res['state'],
            is_yawning=yawn_res['is_yawning']
        )
        self.distraction_score = dist_score
        self.distraction_level = dist_level

        # 7. State Machine & Safety Status
        state, driver_status, alert_msg, severity = self.update_state_machine(
            current_time=current_time,
            eye_data=eye_res,
            head_data=head_res,
            hand_data=hand_res,
            yawn_data=yawn_res,
            face_detected=face_detected
        )
        self.driver_status = driver_status
        self.current_alert_msg = alert_msg
        self.current_severity = severity

        # 8. Check for newly confirmed alert events to persist in SQLite
        new_event = False
        event_type = ""
        event_duration = 0.0
        details = ""

        if eye_res['new_alert_event']:
            new_event = True
            event_type = "Eye Closure Drowsiness"
            event_duration = eye_res['closure_time']
            details = f"EAR: {eye_res['avg_ear']:.2f}, Threshold: {eye_res['threshold_sec']}s"
            database.insert_alert(event_type, "CRITICAL", event_duration, details)

        elif head_res['new_alert_event']:
            new_event = True
            event_type = f"Looking {head_res['direction']}"
            event_duration = head_res['turn_time']
            details = f"Yaw: {head_res['yaw']}°, Threshold: {head_res['threshold_sec']}s"
            database.insert_alert(event_type, "WARNING", event_duration, details)

        elif hand_res['new_alert_event']:
            new_event = True
            event_type = "Single-Hand Driving" if hand_res['state'] == 'ONE HAND' else "Hands Not Detected"
            event_duration = hand_res['hand_time']
            details = f"Hands count: {hand_res['num_hands']}, Duration: {event_duration}s"
            database.insert_alert(event_type, hand_res['severity'], event_duration, details)

        elif yawn_res['new_alert_event']:
            new_event = True
            event_type = "Yawning / Fatigue"
            event_duration = yawn_res['yawn_duration']
            details = f"MAR: {yawn_res['mar']:.2f}, Duration: {event_duration}s"
            database.insert_alert(event_type, "WARNING", event_duration, details)

        # 9. Render HUD Visual Overlays on Frame
        annotated_frame = self.render_hud_overlay(
            frame=frame_bgr.copy(),
            face_detected=face_detected,
            bbox=bbox,
            mesh_results=face_res['mesh_results'],
            landmarks_px=landmarks_px,
            eye_res=eye_res,
            head_res=head_res,
            hand_res=hand_res,
            yawn_res=yawn_res,
            driver_status=driver_status,
            dist_score=dist_score
        )

        # Build telemetry payload
        telemetry = {
            'face_detected': face_detected,
            'driver_status': driver_status,
            'eye_status': eye_res['eye_status'],
            'avg_ear': eye_res['avg_ear'],
            'eye_closure_time': eye_res['closure_time'],
            'eye_closure_threshold': eye_res['threshold_sec'],
            'head_direction': head_res['direction'],
            'yaw': head_res['yaw'],
            'pitch': head_res['pitch'],
            'head_turn_time': head_res['turn_time'],
            'head_turn_threshold': head_res['threshold_sec'],
            'hands': hand_res['state'],
            'num_hands': hand_res['num_hands'],
            'hand_time': hand_res['hand_time'],
            'hand_threshold': hand_res['threshold_sec'],
            'yawning': yawn_res['is_yawning'],
            'mar': yawn_res['mar'],
            'yawn_time': yawn_res['yawn_duration'],
            'yawn_threshold': yawn_res['threshold_sec'],
            'distraction_score': dist_score,
            'distraction_level': dist_level,
            'severity': severity,
            'alert': alert_msg,
            'emergency_simulation': self.emergency_triggered,
            'emergency_duration': round(current_time - (self.critical_start_time or current_time), 1) if self.critical_start_time else 0.0,
            'total_blinks': eye_res['total_blinks'],
            'total_yawns': yawn_res['total_yawns'],
            'fps': self.fps,
            'new_alert_recorded': new_event,
            'demo_mode': self.demo_mode,
            'simulation_active': self.simulation_override is not None
        }

        # Trigger callback if set and alert occurred
        if new_event and self.event_callback:
            self.event_callback(telemetry)

        return annotated_frame, telemetry

    def _handle_simulation_frame(self, frame_bgr: np.ndarray, current_time: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Handles synthetic demonstration scenarios for evaluation."""
        scenario = self.simulation_override['scenario']
        elapsed = current_time - self.simulation_override['start_time']

        face_detected = True
        eye_status = "OPEN"
        avg_ear = 0.28
        eye_time = 0.0
        head_dir = "CENTER"
        yaw = 0.0
        head_time = 0.0
        hands = "TWO HANDS"
        hand_time = 0.0
        yawning = False
        mar = 0.25
        yawn_time = 0.0
        driver_status = "SAFE"
        severity = "INFO"
        alert = "Simulation Active"
        score = 15
        emergency = False

        if scenario == 'drowsiness':
            eye_status = "CLOSED"
            avg_ear = 0.12
            eye_time = round(min(elapsed, 3.5), 2)
            if eye_time > 2.0:
                driver_status = "CRITICAL"
                severity = "CRITICAL"
                alert = "DROWSINESS ALERT: EYES CLOSED FOR MORE THAN 2 SECONDS"
                score = 85
            else:
                driver_status = "WARNING"
                severity = "WARNING"
                alert = "Simulating Eyes Closing..."
                score = 50

        elif scenario in ('head_left', 'head_right'):
            head_dir = "LEFT" if scenario == 'head_left' else "RIGHT"
            yaw = -26.0 if scenario == 'head_left' else 26.0
            head_time = round(min(elapsed, 4.2), 2)
            if head_time > 3.0:
                driver_status = "CRITICAL"
                severity = "CRITICAL"
                alert = f"DISTRACTION ALERT: DRIVER LOOKING {head_dir}"
                score = 75
            else:
                driver_status = "WARNING"
                severity = "WARNING"
                alert = f"Simulating Driver Looking {head_dir}..."
                score = 45

        elif scenario == 'single_hand':
            hands = "ONE HAND"
            hand_time = round(min(elapsed, 3.0), 2)
            if hand_time > 2.0:
                driver_status = "WARNING"
                severity = "WARNING"
                alert = "WARNING: SINGLE-HAND DRIVING DETECTED"
                score = 40
            else:
                score = 30

        elif scenario == 'yawning':
            yawning = True
            mar = 0.68
            yawn_time = round(min(elapsed, 2.5), 2)
            if yawn_time > 1.2:
                driver_status = "WARNING"
                severity = "WARNING"
                alert = "WARNING: YAWNING DETECTED (FATIGUE INDICATOR)"
                score = 45

        elif scenario == 'critical_emergency':
            eye_status = "CLOSED"
            avg_ear = 0.10
            eye_time = round(elapsed, 2)
            hands = "NO HANDS"
            head_dir = "LEFT"
            score = 95
            driver_status = "CRITICAL"
            severity = "CRITICAL"
            if elapsed >= 5.0:
                emergency = True
                alert = "EMERGENCY SAFETY ALERT: SIMULATED NOTIFICATION SENT"
            else:
                alert = "CRITICAL DISTRACTION: ESCALATION IN PROGRESS..."

        telemetry = {
            'face_detected': face_detected,
            'driver_status': driver_status,
            'eye_status': eye_status,
            'avg_ear': avg_ear,
            'eye_closure_time': eye_time,
            'eye_closure_threshold': 2.0,
            'head_direction': head_dir,
            'yaw': yaw,
            'pitch': 0.0,
            'head_turn_time': head_time,
            'head_turn_threshold': 3.0,
            'hands': hands,
            'num_hands': 1 if hands == "ONE HAND" else (0 if hands == "NO HANDS" else 2),
            'hand_time': hand_time,
            'hand_threshold': 2.0,
            'yawning': yawning,
            'mar': mar,
            'yawn_time': yawn_time,
            'yawn_threshold': 1.2,
            'distraction_score': score,
            'distraction_level': "Critical" if score > 80 else ("High" if score > 60 else "Moderate"),
            'severity': severity,
            'alert': alert,
            'emergency_simulation': emergency,
            'emergency_duration': round(elapsed, 1),
            'total_blinks': 12,
            'total_yawns': 3,
            'fps': 30.0,
            'new_alert_recorded': False,
            'demo_mode': True,
            'simulation_active': True
        }

        # Render visual banner for simulation
        annotated = frame_bgr.copy()
        cv2.putText(annotated, f"[DEMO SIMULATION: {scenario.upper()}]", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2, cv2.LINE_AA)
        return annotated, telemetry

    def render_hud_overlay(self,
                           frame: np.ndarray,
                           face_detected: bool,
                           bbox: Optional[Tuple[int, int, int, int]],
                           mesh_results: Any,
                           landmarks_px: List[Tuple[int, int, float]],
                           eye_res: Dict[str, Any],
                           head_res: Dict[str, Any],
                           hand_res: Dict[str, Any],
                           yawn_res: Dict[str, Any],
                           driver_status: str,
                           dist_score: int) -> np.ndarray:
        """
        Renders rich automotive HUD graphics, telemetry overlays, and status brackets directly on the frame.
        """
        h, w, _ = frame.shape

        # Select color theme based on safety state
        if driver_status == "SAFE":
            status_bgr = (0, 255, 120)    # Neon Green
            hud_bg = (20, 30, 20)
        elif driver_status == "WARNING":
            status_bgr = (0, 190, 255)    # Amber / Orange
            hud_bg = (30, 30, 15)
        else:
            status_bgr = (40, 40, 255)    # Crimson Red
            hud_bg = (40, 15, 15)

        # 1. Subtle Face Mesh & Bounding Brackets
        if face_detected and bbox:
            self.face_detector.draw_bounding_box(frame, bbox, status_bgr)
            self.face_detector.draw_face_mesh(frame, mesh_results, draw_contours=False)
            self.eye_detector.draw_eye_overlays(frame, landmarks_px, eye_res['eye_status'])
            self.head_pose.draw_head_direction(frame, landmarks_px, head_res.get('rvec'), head_res.get('tvec'), head_res['direction'])
            self.yawn_detector.draw_mouth_overlay(frame, landmarks_px, yawn_res['is_yawning'])

        # 2. Hand landmarks & bounding boxes
        self.hand_detector.draw_hands(frame, hand_res.get('hands_results'))

        # 3. Top-Left Automotive HUD Card (Glassmorphic Semi-transparent Box)
        card_w, card_h = 240, 160
        sub_img = frame[15:15+card_h, 15:15+card_w]
        overlay = np.full_like(sub_img, hud_bg, dtype=np.uint8)
        frame[15:15+card_h, 15:15+card_w] = cv2.addWeighted(sub_img, 0.35, overlay, 0.65, 0)
        cv2.rectangle(frame, (15, 15), (15+card_w, 15+card_h), status_bgr, 1)

        # Card Content Lines
        cv2.putText(frame, "DRIVER TELEMETRY", (25, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        face_str = "DETECTED" if face_detected else "NO FACE"
        eye_str = f"{eye_res['eye_status']} (EAR {eye_res['avg_ear']:.2f})"
        head_str = f"{head_res['direction']} ({head_res['yaw']:+.0f}°)"
        hand_str = f"{hand_res['state']}"
        yawn_str = "DETECTED" if yawn_res['is_yawning'] else "NO"

        items = [
            ("FACE", face_str, (0, 255, 120) if face_detected else (0, 0, 255)),
            ("EYES", eye_str, (0, 255, 120) if eye_res['eye_status'] == 'OPEN' else (0, 0, 255)),
            ("HEAD", head_str, (0, 255, 120) if head_res['direction'] == 'CENTER' else (0, 190, 255)),
            ("HANDS", hand_str, (0, 255, 120) if hand_res['state'] == 'TWO HANDS' else (0, 190, 255) if hand_res['state'] == 'ONE HAND' else (0, 0, 255)),
            ("YAWN", yawn_str, (0, 190, 255) if yawn_res['is_yawning'] else (200, 200, 200)),
            ("STATUS", driver_status, status_bgr)
        ]

        y_offset = 52
        for label, val, val_color in items:
            cv2.putText(frame, f"{label}:", (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1, cv2.LINE_AA)
            cv2.putText(frame, val, (85, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.38, val_color, 1, cv2.LINE_AA)
            y_offset += 19

        # 4. Top-Right FPS and Score Badge
        fps_str = f"FPS: {self.fps:.0f}"
        cv2.putText(frame, fps_str, (w - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        
        score_str = f"RISK: {dist_score}/100"
        cv2.putText(frame, score_str, (w - 120, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_bgr, 1, cv2.LINE_AA)

        # 5. Bottom Live Progress / Active Alert Bar
        if driver_status != "SAFE" or self.demo_mode:
            bar_h = 30
            y_bar = h - bar_h - 10
            bar_sub = frame[y_bar:y_bar+bar_h, 15:w-15]
            overlay_bar = np.full_like(bar_sub, (20, 20, 20), dtype=np.uint8)
            frame[y_bar:y_bar+bar_h, 15:w-15] = cv2.addWeighted(bar_sub, 0.3, overlay_bar, 0.7, 0)
            cv2.rectangle(frame, (15, y_bar), (w-15, y_bar+bar_h), status_bgr, 1)

            # Build alert string with active timers
            timer_text = ""
            if eye_res['closure_time'] > 0:
                timer_text += f"Eyes: {eye_res['closure_time']:.1f}/{eye_res['threshold_sec']:.0f}s  "
            if head_res['turn_time'] > 0:
                timer_text += f"Head: {head_res['turn_time']:.1f}/{head_res['threshold_sec']:.0f}s  "
            if hand_res['hand_time'] > 0 and hand_res['state'] != 'TWO HANDS':
                timer_text += f"Hands: {hand_res['hand_time']:.1f}/{hand_res['threshold_sec']:.0f}s"

            display_msg = f"{self.current_alert_msg} | {timer_text}" if timer_text else self.current_alert_msg
            cv2.putText(frame, display_msg[:75], (25, y_bar + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_bgr, 1, cv2.LINE_AA)

        return frame
