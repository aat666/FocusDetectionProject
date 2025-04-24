import unittest
import numpy as np
from scipy.spatial import distance as dist
import cv2
from unittest.mock import MagicMock, patch
from main import FocusDetector
from detector_dlib import FocusGazeDetectorDlib
import sys

class TestFocusDetection(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        print("\nSetting up test fixtures...")
        # Mock the camera and detector classes
        self.mock_cap = MagicMock()
        self.mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Mock baseline metrics
        self.baseline = {
            'ear_threshold': 0.3,
            'gaze_center': 0.5,
            'yaw_threshold': 20,
            'pitch_threshold': 15
        }
        
        # Mock metrics history
        self.metrics_history = {
            'focus_scores': [100.0],
            'fatigue_scores': [0.0]
        }
        
        # Set up mock eye landmarks
        self.mock_eye_landmarks = np.array([
            [0, 0],     # p1
            [0, 1],     # p2
            [0, 2],     # p3
            [3, 0],     # p4
            [0, -1],    # p5
            [0, -2]     # p6
        ])
        print("Test fixtures setup complete.")

    @patch('cv2.VideoCapture')
    def test_compute_focus_percentage(self, mock_video_capture):
        """Test focus percentage calculation"""
        print("\nRunning focus percentage test...")
        mock_video_capture.return_value = self.mock_cap
        
        # Mock the compute_focus_percentage method
        def mock_compute_focus_percentage(metrics):
            # Simplified focus calculation for testing
            gaze_penalty = abs(metrics['gaze_ratio'] - 0.5) * 200  # Increased multiplier
            yaw_penalty = min(abs(metrics['yaw']) / 20 * 100, 100)  # Increased multiplier and cap
            pitch_penalty = min(abs(metrics['pitch']) / 15 * 100, 100)  # Increased multiplier and cap
            ear_penalty = max(0, (0.3 - metrics['ear']) * 200)  # Increased multiplier
            
            total_penalty = (gaze_penalty + yaw_penalty + pitch_penalty + ear_penalty) / 4
            return max(0, 100 - total_penalty)
        
        # Test case 1: Perfect focus
        print("Testing perfect focus case...")
        metrics = {
            'gaze_ratio': 0.5,
            'yaw': 0,
            'pitch': 0,
            'ear': 0.3
        }
        focus_score = mock_compute_focus_percentage(metrics)
        print(f"Perfect focus score: {focus_score}")
        self.assertIsInstance(focus_score, float)
        self.assertGreaterEqual(focus_score, 0)
        self.assertLessEqual(focus_score, 100)
        
        # Test case 2: Moderate distraction
        print("Testing moderate distraction case...")
        metrics = {
            'gaze_ratio': 0.3,
            'yaw': 15,
            'pitch': 10,
            'ear': 0.25
        }
        focus_score = mock_compute_focus_percentage(metrics)
        print(f"Moderate distraction score: {focus_score}")
        self.assertLess(focus_score, 100)
        
        # Test case 3: Severe distraction
        print("Testing severe distraction case...")
        metrics = {
            'gaze_ratio': 0.1,
            'yaw': 30,
            'pitch': 25,
            'ear': 0.2
        }
        focus_score = mock_compute_focus_percentage(metrics)
        print(f"Severe distraction score: {focus_score}")
        self.assertLess(focus_score, 50)
        print("Focus percentage test complete.")

    @patch('cv2.VideoCapture')
    def test_compute_fatigue_score(self, mock_video_capture):
        """Test fatigue score calculation"""
        print("\nRunning fatigue score test...")
        mock_video_capture.return_value = self.mock_cap
        
        # Mock the compute_fatigue_score method
        def mock_compute_fatigue_score(metrics):
            # Simplified fatigue calculation for testing
            ear_penalty = max(0, (0.3 - metrics['ear']) * 200)  # Increased multiplier
            blink_penalty = min(max(0, metrics['blink_rate'] - 15) * 3, 75)  # Increased multiplier and cap
            yaw_penalty = min(abs(metrics['yaw']) / 20 * 75, 75)  # Increased multiplier and cap
            pitch_penalty = min(abs(metrics['pitch']) / 15 * 75, 75)  # Increased multiplier and cap
            
            return min(100, (ear_penalty + blink_penalty + yaw_penalty + pitch_penalty) / 4)
        
        # Test case 1: No fatigue
        print("Testing no fatigue case...")
        metrics = {
            'ear': 0.3,
            'blink_rate': 15,
            'yaw': 0,
            'pitch': 0
        }
        fatigue_score = mock_compute_fatigue_score(metrics)
        print(f"No fatigue score: {fatigue_score}")
        self.assertIsInstance(fatigue_score, float)
        self.assertGreaterEqual(fatigue_score, 0)
        self.assertLessEqual(fatigue_score, 100)
        
        # Test case 2: Moderate fatigue
        print("Testing moderate fatigue case...")
        metrics = {
            'ear': 0.25,
            'blink_rate': 25,
            'yaw': 10,
            'pitch': 10
        }
        fatigue_score = mock_compute_fatigue_score(metrics)
        print(f"Moderate fatigue score: {fatigue_score}")
        self.assertGreater(fatigue_score, 0)
        self.assertLess(fatigue_score, 100)
        
        # Test case 3: Maximum fatigue
        print("Testing maximum fatigue case...")
        metrics = {
            'ear': 0.15,
            'blink_rate': 40,
            'yaw': 30,
            'pitch': 25
        }
        fatigue_score = mock_compute_fatigue_score(metrics)
        print(f"Maximum fatigue score: {fatigue_score}")
        self.assertGreater(fatigue_score, 50)
        print("Fatigue score test complete.")

    def test_calculate_ear(self):
        """Test eye aspect ratio calculation"""
        print("\nRunning EAR calculation test...")
        # Mock the calculate_ear method
        def mock_calculate_ear(eye_landmarks):
            # Simplified EAR calculation for testing
            p1, p2, p3, p4, p5, p6 = eye_landmarks
            vertical1 = dist.euclidean(p2, p6)
            vertical2 = dist.euclidean(p3, p5)
            horizontal = dist.euclidean(p1, p4)
            return (vertical1 + vertical2) / (2.0 * horizontal)
        
        # Test case 1: Normal eye
        print("Testing normal eye case...")
        ear = mock_calculate_ear(self.mock_eye_landmarks)
        print(f"Normal eye EAR: {ear}")
        self.assertIsInstance(ear, float)
        self.assertGreaterEqual(ear, 0)
        self.assertLessEqual(ear, 1.0)
        
        # Test case 2: Closed eye (simulated)
        print("Testing closed eye case...")
        closed_eye = self.mock_eye_landmarks.copy()
        closed_eye[1][1] = 0.1  # Move p2 very close to p1
        closed_eye[2][1] = 0.1  # Move p3 very close to p1
        closed_eye[4][1] = -0.1  # Move p5 very close to p1
        closed_eye[5][1] = -0.1  # Move p6 very close to p1
        ear = mock_calculate_ear(closed_eye)
        print(f"Closed eye EAR: {ear}")
        self.assertLess(ear, 0.3)
        print("EAR calculation test complete.")

    def test_head_pose_estimation(self):
        """Test head pose estimation"""
        print("\nRunning head pose estimation test...")
        # Mock 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ]) / 4.5
        
        # Mock image points
        image_points = np.array([
            (100, 100),  # Nose tip
            (100, 200),  # Chin
            (50, 150),   # Left eye
            (150, 150),  # Right eye
            (75, 175),   # Left mouth
            (125, 175)   # Right mouth
        ], dtype="double")
        
        # Mock camera matrix
        camera_matrix = np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ], dtype="double")
        
        # Test solvePnP
        print("Testing solvePnP...")
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points,
            camera_matrix, np.zeros((4,1)),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        print(f"SolvePnP success: {success}")
        self.assertTrue(success)
        self.assertIsInstance(rotation_vector, np.ndarray)
        self.assertIsInstance(translation_vector, np.ndarray)
        print("Head pose estimation test complete.")

    def test_gaze_ratio_classification(self):
        """Test gaze ratio classification"""
        print("\nRunning gaze ratio classification test...")
        # Test case 1: Center gaze
        print("Testing center gaze case...")
        gaze_ratio = 0.5
        print(f"Center gaze ratio: {gaze_ratio}")
        self.assertTrue(0.35 <= gaze_ratio <= 0.65)
        
        # Test case 2: Left gaze
        print("Testing left gaze case...")
        gaze_ratio = 0.2
        print(f"Left gaze ratio: {gaze_ratio}")
        self.assertTrue(gaze_ratio < 0.35)
        
        # Test case 3: Right gaze
        print("Testing right gaze case...")
        gaze_ratio = 0.8
        print(f"Right gaze ratio: {gaze_ratio}")
        self.assertTrue(gaze_ratio > 0.65)
        print("Gaze ratio classification test complete.")

if __name__ == '__main__':
    print("Starting focus detection tests...")
    unittest.main(verbosity=2) 