"""
Automated Unit Tests for Vision-Based Distracted Driver Detection System.
Validates Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), Distraction Scoring, State Machine logic,
and SQLite Database integration.
"""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection.eye_detector import EyeDetector
from detection.yawn_detector import YawnDetector
from detection.distraction_engine import DistractionEngine
import database


class TestDriverDetectionPipeline(unittest.TestCase):

    def setUp(self):
        """Set up test environment and initialize engine."""
        self.engine = DistractionEngine()
        database.init_db()

    def test_ear_calculation_open_vs_closed(self):
        """Test EAR formula on simulated open and closed eye coordinates."""
        # Open Eye: width=20, height=8
        # p1=(0,0), p2=(5,4), p3=(15,4), p4=(20,0), p5=(15,-4), p6=(5,-4)
        open_eye = [
            (0.0, 0.0, 0.0),    # p1
            (5.0, 4.0, 0.0),    # p2
            (15.0, 4.0, 0.0),   # p3
            (20.0, 0.0, 0.0),   # p4
            (15.0, -4.0, 0.0),  # p5
            (5.0, -4.0, 0.0)    # p6
        ]
        ear_open = EyeDetector.calculate_ear(open_eye)
        # Expected: (8 + 8) / (2 * 20) = 16 / 40 = 0.40
        self.assertAlmostEqual(ear_open, 0.40, places=2)
        self.assertGreater(ear_open, 0.22, "Open eye EAR should be well above threshold")

        # Closed Eye: width=20, height=1
        # p1=(0,0), p2=(5,0.5), p3=(15,0.5), p4=(20,0), p5=(15,-0.5), p6=(5,-0.5)
        closed_eye = [
            (0.0, 0.0, 0.0),
            (5.0, 0.5, 0.0),
            (15.0, 0.5, 0.0),
            (20.0, 0.0, 0.0),
            (15.0, -0.5, 0.0),
            (5.0, -0.5, 0.0)
        ]
        ear_closed = EyeDetector.calculate_ear(closed_eye)
        # Expected: (1 + 1) / (2 * 20) = 2 / 40 = 0.05
        self.assertAlmostEqual(ear_closed, 0.05, places=2)
        self.assertLess(ear_closed, 0.22, "Closed eye EAR should be below threshold")

    def test_mar_calculation(self):
        """Test MAR formula for normal mouth vs wide yawn."""
        yawn_detector = YawnDetector()
        
        # Synthetic landmark list of 478 points
        landmarks = [(0.0, 0.0, 0.0)] * 478
        
        # Set normal mouth landmarks
        landmarks[YawnDetector.MOUTH_LEFT] = (0.0, 0.0, 0.0)
        landmarks[YawnDetector.MOUTH_RIGHT] = (10.0, 0.0, 0.0)  # width = 10
        landmarks[YawnDetector.UPPER_LIP_MID] = (5.0, 1.0, 0.0)
        landmarks[YawnDetector.LOWER_LIP_MID] = (5.0, -1.0, 0.0) # height = 2
        landmarks[YawnDetector.UPPER_LIP_LEFT] = (3.0, 1.0, 0.0)
        landmarks[YawnDetector.LOWER_LIP_LEFT] = (3.0, -1.0, 0.0)
        landmarks[YawnDetector.UPPER_LIP_RIGHT] = (7.0, 1.0, 0.0)
        landmarks[YawnDetector.LOWER_LIP_RIGHT] = (7.0, -1.0, 0.0)

        mar_normal = yawn_detector.calculate_mar(landmarks)
        # Expected: 2 / 10 = 0.20
        self.assertAlmostEqual(mar_normal, 0.20, places=2)
        self.assertLess(mar_normal, 0.55, "Normal speech MAR should be below 0.55")

        # Set wide yawning mouth landmarks (height = 7)
        landmarks[YawnDetector.UPPER_LIP_MID] = (5.0, 3.5, 0.0)
        landmarks[YawnDetector.LOWER_LIP_MID] = (5.0, -3.5, 0.0)
        landmarks[YawnDetector.UPPER_LIP_LEFT] = (3.0, 3.5, 0.0)
        landmarks[YawnDetector.LOWER_LIP_LEFT] = (3.0, -3.5, 0.0)
        landmarks[YawnDetector.UPPER_LIP_RIGHT] = (7.0, 3.5, 0.0)
        landmarks[YawnDetector.LOWER_LIP_RIGHT] = (7.0, -3.5, 0.0)

        mar_yawn = yawn_detector.calculate_mar(landmarks)
        # Expected: 7 / 10 = 0.70
        self.assertAlmostEqual(mar_yawn, 0.70, places=2)
        self.assertGreater(mar_yawn, 0.55, "Yawning MAR should exceed 0.55")

    def test_distraction_scoring_rules(self):
        """Test rule-based distraction score additions and clamping."""
        # 1. Fully Safe Driver: face=True, eyes=OPEN, head=CENTER, hands=TWO HANDS, yawn=False
        score, level = self.engine.compute_distraction_score(
            face_detected=True,
            eye_status='OPEN',
            head_dir='CENTER',
            hand_state='TWO HANDS',
            is_yawning=False
        )
        self.assertEqual(score, 0)
        self.assertEqual(level, "Normal")

        # 2. Single-hand + Looking Right: 20 + 30 = 50
        score, level = self.engine.compute_distraction_score(
            face_detected=True,
            eye_status='OPEN',
            head_dir='RIGHT',
            hand_state='ONE HAND',
            is_yawning=False
        )
        self.assertEqual(score, 50)
        self.assertEqual(level, "Moderate")

        # 3. Eyes closed + No hands + Yawn: 50 + 40 + 20 = 110 -> Clamped to 100
        score, level = self.engine.compute_distraction_score(
            face_detected=True,
            eye_status='CLOSED',
            head_dir='CENTER',
            hand_state='NO HANDS',
            is_yawning=True
        )
        self.assertEqual(score, 100)
        self.assertEqual(level, "Critical")

    def test_state_machine_transitions(self):
        """Test State Machine progression: NORMAL -> WARNING -> CRITICAL -> EMERGENCY."""
        t0 = 100.0

        # Initial Normal
        eye_data = {'drowsiness_alert': False, 'closure_time': 0.0, 'threshold_sec': 2.0}
        head_data = {'head_alert': False, 'turn_time': 0.0, 'direction': 'CENTER'}
        hand_data = {'single_hand_alert': False, 'no_hands_alert': False, 'state': 'TWO HANDS', 'hand_time': 0.0}
        yawn_data = {'yawn_detected': False}

        state, status, msg, sev = self.engine.update_state_machine(t0, eye_data, head_data, hand_data, yawn_data, True)
        self.assertEqual(state, DistractionEngine.STATE_NORMAL)
        self.assertEqual(status, "SAFE")

        # Warning Trigger: Single hand
        hand_data['single_hand_alert'] = True
        state, status, msg, sev = self.engine.update_state_machine(t0 + 2.1, eye_data, head_data, hand_data, yawn_data, True)
        self.assertEqual(state, DistractionEngine.STATE_WARNING)
        self.assertEqual(status, "WARNING")
        self.assertEqual(sev, "WARNING")

        # Critical Trigger: Prolonged Eye closure
        eye_data['drowsiness_alert'] = True
        state, status, msg, sev = self.engine.update_state_machine(t0 + 3.0, eye_data, head_data, hand_data, yawn_data, True)
        self.assertEqual(state, DistractionEngine.STATE_CRITICAL)
        self.assertEqual(status, "CRITICAL")
        self.assertEqual(sev, "CRITICAL")

        # Emergency Escalation: Critical persists > 5 seconds
        state, status, msg, sev = self.engine.update_state_machine(t0 + 9.0, eye_data, head_data, hand_data, yawn_data, True)
        self.assertEqual(state, DistractionEngine.STATE_EMERGENCY)
        self.assertTrue(self.engine.emergency_triggered)

    def test_database_crud(self):
        """Test SQLite alert recording, retrieval, statistics, and clearing."""
        database.clear_alerts()

        # Insert alerts
        id1 = database.insert_alert("Eye Closure Drowsiness", "CRITICAL", 2.5, "EAR 0.12")
        id2 = database.insert_alert("Looking Left", "WARNING", 3.2, "Yaw -22 deg")
        id3 = database.insert_alert("Single-Hand Driving", "WARNING", 2.1, "1 hand")

        self.assertGreater(id1, 0)
        self.assertGreater(id2, 0)

        # Retrieve alerts
        alerts = database.get_alerts(limit=10)
        self.assertEqual(len(alerts), 3)
        self.assertEqual(alerts[0]['alert_type'], "Single-Hand Driving")

        # Statistics
        stats = database.get_statistics()
        self.assertEqual(stats['total_alerts'], 3)
        self.assertEqual(stats['critical_alerts'], 1)
        self.assertEqual(stats['warning_alerts'], 2)
        self.assertEqual(stats['longest_eye_closure'], 2.5)
        self.assertEqual(stats['longest_head_turn'], 3.2)

        # CSV Export
        csv_str = database.export_csv_string()
        self.assertIn("Eye Closure Drowsiness", csv_str)
        self.assertIn("CRITICAL", csv_str)


if __name__ == '__main__':
    unittest.main()
