import cv2
import numpy as np
import os
from detector_dlib import FocusGazeDetectorDlib
from main import FocusDetector
import time
from scipy.spatial.transform import Rotation as R

class IntegrationTest:
    def __init__(self):
        print("\n=== Initializing Integration Test ===")
        print("Creating detector instances...")
        self.detector = FocusGazeDetectorDlib(cap=None)
        self.focus_scorer = FocusDetector(cap=None)
        self.test_image_path = "test_images/test_face.jpg"  # You'll need to provide a test image
        print("Initialization complete!\n")
        
    def load_test_image(self):
        """Load and validate test image"""
        try:
            print("Looking for test image...")
            if not os.path.exists(self.test_image_path):
                raise FileNotFoundError(f"Test image not found at {self.test_image_path}")
            
            print("Loading test image...")
            frame = cv2.imread(self.test_image_path)
            if frame is None:
                raise ValueError("Failed to load test image")
            
            print(f"✅ Successfully loaded test image: {frame.shape}")
            return frame
        except Exception as e:
            print(f"❌ Error loading test image: {e}")
            return None

    def test_face_detection(self, frame):
        """Test face detection component"""
        try:
            print("\n=== Testing Face Detection ===")
            print("Converting image to grayscale...")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("Running face detection...")
            rects = self.detector.detector(gray, 0)
            
            if len(rects) == 0:
                print("❌ No face detected in the test image")
                return None
            
            print(f"✅ Face detected successfully")
            return rects[0]
        except Exception as e:
            print(f"❌ Error in face detection: {e}")
            return None

    def test_landmark_detection(self, frame, rect):
        """Test facial landmark detection"""
        try:
            print("\n=== Testing Landmark Detection ===")
            print("Converting image to grayscale...")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("Detecting facial landmarks...")
            shape = self.detector.predictor(gray, rect)
            landmarks = self.detector.shape_to_np(shape)
            
            if len(landmarks) != 68:
                print("❌ Incorrect number of landmarks detected")
                return None
            
            print("✅ Landmarks detected successfully")
            return landmarks
        except Exception as e:
            print(f"❌ Error in landmark detection: {e}")
            return None

    def test_metrics_calculation(self, frame, landmarks):
        """Test calculation of various metrics"""
        try:
            print("\n=== Testing Metrics Calculation ===")
            metrics = {}
            
            print("Calculating Eye Aspect Ratio (EAR)...")
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            left_ear = self.detector.calculate_ear(left_eye)
            right_ear = self.detector.calculate_ear(right_eye)
            metrics['ear'] = (left_ear + right_ear) / 2.0
            print(f"✅ EAR calculated: {metrics['ear']:.3f}")
            
            print("Calculating gaze ratio...")
            nose_point = landmarks[30]
            metrics['gaze_ratio'] = nose_point[0] / frame.shape[1]
            print(f"✅ Gaze ratio calculated: {metrics['gaze_ratio']:.3f}")
            
            print("Calculating head pose...")
            image_points = np.array([
                landmarks[30], landmarks[8], landmarks[36],
                landmarks[45], landmarks[48], landmarks[54]
            ], dtype="double")
            
            success, rotation_vector, _ = cv2.solvePnP(
                self.detector.face_3d_model, image_points,
                self.detector.camera_matrix, self.detector.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                rmat, _ = cv2.Rodrigues(rotation_vector)
                euler_angles = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                metrics['yaw'] = euler_angles[0]
                metrics['pitch'] = euler_angles[1]
                metrics['roll'] = euler_angles[2]
                print(f"✅ Head pose calculated - Yaw: {metrics['yaw']:.1f}°, Pitch: {metrics['pitch']:.1f}°, Roll: {metrics['roll']:.1f}°")
            else:
                print("❌ Failed to calculate head pose")
                return None
            
            return metrics
        except Exception as e:
            print(f"❌ Error in metrics calculation: {e}")
            return None

    def test_focus_calculation(self, metrics):
        """Test focus score calculation"""
        try:
            print("\n=== Testing Focus Score Calculation ===")
            print("Computing focus score...")
            focus_score = self.focus_scorer.compute_focus_percentage(metrics)
            print(f"✅ Focus score calculated: {focus_score:.1f}%")
            return focus_score
        except Exception as e:
            print(f"❌ Error in focus score calculation: {e}")
            return None

    def test_fatigue_calculation(self, metrics):
        """Test fatigue score calculation"""
        try:
            print("\n=== Testing Fatigue Score Calculation ===")
            print("Computing fatigue score...")
            fatigue_score = self.focus_scorer.compute_fatigue_score(metrics)
            print(f"✅ Fatigue score calculated: {fatigue_score:.1f}%")
            return fatigue_score
        except Exception as e:
            print(f"❌ Error in fatigue score calculation: {e}")
            return None

    def run_test(self):
        """Run complete integration test"""
        print("\n=== Starting Integration Test ===")
        print("This test will verify all components of the focus detection system.")
        print("Each step will be clearly indicated with progress updates.\n")
        
        # Load test image
        print("Step 1: Loading test image...")
        frame = self.load_test_image()
        if frame is None:
            print("\n❌ Test failed at image loading stage")
            return
        
        # Test face detection
        print("\nStep 2: Testing face detection...")
        rect = self.test_face_detection(frame)
        if rect is None:
            print("\n❌ Test failed at face detection stage")
            return
        
        # Test landmark detection
        print("\nStep 3: Testing landmark detection...")
        landmarks = self.test_landmark_detection(frame, rect)
        if landmarks is None:
            print("\n❌ Test failed at landmark detection stage")
            return
        
        # Test metrics calculation
        print("\nStep 4: Testing metrics calculation...")
        metrics = self.test_metrics_calculation(frame, landmarks)
        if metrics is None:
            print("\n❌ Test failed at metrics calculation stage")
            return
        
        # Test focus calculation
        print("\nStep 5: Testing focus calculation...")
        focus_score = self.test_focus_calculation(metrics)
        if focus_score is None:
            print("\n❌ Test failed at focus calculation stage")
            return
        
        # Test fatigue calculation
        print("\nStep 6: Testing fatigue calculation...")
        fatigue_score = self.test_fatigue_calculation(metrics)
        if fatigue_score is None:
            print("\n❌ Test failed at fatigue calculation stage")
            return
        
        # Display final results
        print("\n=== Integration Test Results ===")
        print("All tests completed successfully!")
        print("\nFinal Metrics:")
        print(f"EAR: {metrics['ear']:.3f}")
        print(f"Gaze Ratio: {metrics['gaze_ratio']:.3f}")
        print(f"Head Pose - Yaw: {metrics['yaw']:.1f}°, Pitch: {metrics['pitch']:.1f}°")
        print(f"Focus Score: {focus_score:.1f}%")
        print(f"Fatigue Score: {fatigue_score:.1f}%")
        
        # Visualize results
        print("\nGenerating visualization...")
        self.visualize_results(frame, landmarks, focus_score, fatigue_score)
        print("\n=== Integration Test Complete ===")

    def visualize_results(self, frame, landmarks, focus_score, fatigue_score):
        """Visualize test results on the image"""
        try:
            print("Drawing landmarks and scores...")
            # Draw landmarks
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Draw focus score
            cv2.putText(frame, f"Focus: {focus_score:.1f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw fatigue score
            cv2.putText(frame, f"Fatigue: {fatigue_score:.1f}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save visualization
            print("Saving visualization...")
            output_path = "test_results/integration_test_result.jpg"
            os.makedirs("test_results", exist_ok=True)
            cv2.imwrite(output_path, frame)
            print(f"✅ Visualization saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error in visualization: {e}")

if __name__ == "__main__":
    test = IntegrationTest()
    test.run_test() 