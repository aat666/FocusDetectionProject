# main.py
import cv2
import dlib
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance as dist
import subprocess
import os
import pandas as pd
from datetime import datetime

class FocusDetector:
    def __init__(self, cap=None):
        # Initialize camera
        self.cap = cap if cap is not None else cv2.VideoCapture(0)
        
        # Initialize dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = os.path.join('models', 'shape_predictor_68_face_landmarks.dat')
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Initialize metrics tracking
        self.start_time = time.time()
        self.frame_count = 0
        self.total_blinks = 0
        self.blink_counter = 0
        self.last_blink_time = 0
        self.distraction_count = 0
        
        # Initialize history tracking
        self.focus_history = []
        self.metrics_history = {
            'raw_ear': [],
            'blink_intervals': [],
            'raw_angles': [],
            'gaze_ratios': [],
            'fatigue_scores': []
        }
        
        # Initialize smoothing buffers
        self.yaw_history = []
        self.pitch_history = []
        self.roll_history = []
        
        # Set up camera matrix for head pose estimation
        self.camera_matrix = np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ], dtype="double")
        self.dist_coeffs = np.zeros((4,1))
        
        # 3D model points for head pose estimation
        self.face_3d_model = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ]) / 4.5
        
        # Initialize baseline metrics
        self.baseline = {
            'ear_threshold': 0.3,
            'gaze_center': 0.5,
            'yaw_threshold': 20,
            'pitch_threshold': 15
        }
        
        # Constants
        self.EAR_THRESHOLD = 0.2
        self.CONSECUTIVE_FRAMES = 3
        self.DEBUG_VERBOSE = True
        
        print("Focus Detection System Initialized")
        print("Press 'q' to quit")
        print("Press 'c' to start calibration")
    
    def shape_to_np(self, shape):
        """Convert dlib shape to numpy array"""
        coords = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    
    def calculate_ear(self, eye):
        """Calculate eye aspect ratio"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def smooth_value(self, value, history, window_size=5):
        """Apply moving average smoothing"""
        history.append(value)
        if len(history) > window_size:
            history.pop(0)
        return np.mean(history)
    
    def compute_focus_percentage(self, metrics):
        """Compute focus percentage based on multiple metrics"""
        # Initialize focus components
        gaze_score = 100.0
        head_pose_score = 100.0
        blink_penalty = 0.0
        
        # Calculate gaze score
        gaze_ratio = metrics['gaze_ratio']
        gaze_deviation = abs(gaze_ratio - self.baseline['gaze_center'])
        gaze_score = max(0, 100 - (gaze_deviation * 200))
        
        # Calculate head pose score
        yaw_penalty = max(0, abs(metrics['yaw']) - self.baseline['yaw_threshold']) * 2
        pitch_penalty = max(0, abs(metrics['pitch']) - self.baseline['pitch_threshold']) * 2
        head_pose_score = max(0, 100 - (yaw_penalty + pitch_penalty))
        
        # Calculate blink penalty
        if metrics['blink_rate'] > 20:  # More than 20 blinks per minute
            blink_penalty = min(30, (metrics['blink_rate'] - 20) * 2)
        
        # Calculate final focus score
        focus_score = (gaze_score * 0.4 + head_pose_score * 0.4) * (1 - blink_penalty/100)
        return max(0, min(100, focus_score))
    
    def compute_fatigue_score(self, metrics):
        """Compute fatigue score based on various metrics"""
        # Initialize fatigue score components
        blink_fatigue = 0.0
        ear_fatigue = 0.0
        head_pose_fatigue = 0.0
        
        # Calculate blink rate fatigue (40% weight)
        baseline_blink_rate = 15.0
        blink_deviation = abs(metrics['blink_rate'] - baseline_blink_rate)
        blink_fatigue = min(40.0, blink_deviation * 2.0)
        
        # Calculate EAR fatigue (30% weight)
        baseline_ear = self.baseline['ear_threshold']
        ear_deviation = abs(metrics['ear'] - baseline_ear)
        ear_fatigue = min(30.0, ear_deviation * 100.0)
        
        # Calculate head pose fatigue (30% weight)
        yaw_penalty = min(15.0, abs(metrics['yaw']) * 0.5)
        pitch_penalty = min(15.0, abs(metrics['pitch']) * 0.5)
        head_pose_fatigue = yaw_penalty + pitch_penalty
        
        # Calculate total fatigue score
        total_fatigue = blink_fatigue + ear_fatigue + head_pose_fatigue
        fatigue_score = min(100.0, total_fatigue)
        
        # Store fatigue score in history
        self.metrics_history['fatigue_scores'].append(fatigue_score)
        
        return fatigue_score
    
    def start_calibration(self):
        """Perform calibration to establish baseline metrics"""
        print("Starting calibration...")
        calibration_frames = 30
        calibration_data = {
            'ear': [],
            'gaze_ratio': [],
            'yaw': [],
            'pitch': []
        }
        
        for _ in range(calibration_frames):
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            
            if len(rects) > 0:
                rect = rects[0]
                shape = self.predictor(gray, rect)
                landmarks = self.shape_to_np(shape)
                
                # Calculate metrics
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                ear = (self.calculate_ear(left_eye) + self.calculate_ear(right_eye)) / 2.0
                
                nose_point = landmarks[30]
                gaze_ratio = nose_point[0] / frame.shape[1]
                
                # Calculate head pose
                image_points = np.array([
                    landmarks[30], landmarks[8], landmarks[36],
                    landmarks[45], landmarks[48], landmarks[54]
                ], dtype="double")
                
                success, rotation_vector, _ = cv2.solvePnP(
                    self.face_3d_model, image_points,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    rmat, _ = cv2.Rodrigues(rotation_vector)
                    euler_angles = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                    
                    # Store calibration data
                    calibration_data['ear'].append(ear)
                    calibration_data['gaze_ratio'].append(gaze_ratio)
                    calibration_data['yaw'].append(euler_angles[0])
                    calibration_data['pitch'].append(euler_angles[1])
            
            time.sleep(0.1)
        
        # Update baseline metrics
        if calibration_data['ear']:
            self.baseline['ear_threshold'] = np.mean(calibration_data['ear'])
            self.baseline['gaze_center'] = np.mean(calibration_data['gaze_ratio'])
        
        print("Calibration completed!")
    
    def process_frame(self, frame):
        """Process frame and return display frame and metrics"""
        try:
            self.frame_count += 1
            current_time = time.time()
            
            metrics = {
                'ear': self.baseline['ear_threshold'],
                'focus_percentage': 100.0,
                'gaze_ratio': 0.5,
                'yaw': 0,
                'pitch': 0,
                'roll': 0,
                'blink_rate': 0,
                'fatigue_score': 0.0
            }
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            
            if len(rects) > 0:
                rect = rects[0]
                shape = self.predictor(gray, rect)
                landmarks = self.shape_to_np(shape)
                
                # Extract eyes landmarks
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                metrics['ear'] = (left_ear + right_ear) / 2.0
                
                # Blink detection
                if metrics['ear'] < self.EAR_THRESHOLD:
                    self.blink_counter += 1
                else:
                    if self.blink_counter >= self.CONSECUTIVE_FRAMES:
                        self.total_blinks += 1
                        current_time = time.time()
                        if self.last_blink_time > 0:
                            blink_interval = current_time - self.last_blink_time
                            self.metrics_history['blink_intervals'].append(blink_interval)
                        self.last_blink_time = current_time
                    self.blink_counter = 0
                
                # Calculate blink rate
                session_duration = current_time - self.start_time
                metrics['blink_rate'] = (self.total_blinks / session_duration) * 60 if session_duration > 0 else 0
                
                # Compute head pose
                image_points = np.array([
                    landmarks[30], landmarks[8], landmarks[36],
                    landmarks[45], landmarks[48], landmarks[54]
                ], dtype="double")
                
                success, rotation_vector, _ = cv2.solvePnP(
                    self.face_3d_model, image_points,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    rmat, _ = cv2.Rodrigues(rotation_vector)
                    euler_angles = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                    
                    metrics['yaw'] = self.smooth_value(euler_angles[0], self.yaw_history)
                    metrics['pitch'] = self.smooth_value(euler_angles[1], self.pitch_history)
                    metrics['roll'] = self.smooth_value(euler_angles[2], self.roll_history)
                    
                    self.metrics_history['raw_angles'].append(euler_angles)
                
                # Compute gaze ratio
                nose_point = landmarks[30]
                metrics['gaze_ratio'] = nose_point[0] / frame.shape[1]
                self.metrics_history['gaze_ratios'].append(metrics['gaze_ratio'])
                
                # Compute focus and fatigue scores
                metrics['focus_percentage'] = self.compute_focus_percentage(metrics)
                metrics['fatigue_score'] = self.compute_fatigue_score(metrics)
                
                # Track focus history
                self.focus_history.append(metrics['focus_percentage'])
                
                # Update distraction count
                if metrics['focus_percentage'] < 60:
                    self.distraction_count += 1
                
                # Draw visualization
                self._draw_debug_info(frame, metrics, current_time)
                
                # Draw landmarks
                for (x, y) in landmarks:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            return frame, metrics
            
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            return frame, {'focus_percentage': 0.0, 'fatigue_score': 0.0}
    
    def _draw_debug_info(self, frame, metrics, current_time):
        """Draw debug information on frame"""
        # Calculate session time
        session_duration = current_time - self.start_time
        hours = int(session_duration // 3600)
        minutes = int((session_duration % 3600) // 60)
        seconds = int(session_duration % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Draw metrics
        cv2.putText(frame, f"EAR: {metrics['ear']:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gaze: {metrics['gaze_ratio']:.3f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {time_str}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw focus information
        focus_percentage = metrics['focus_percentage']
        cv2.putText(frame, f"Focus: {focus_percentage:.1f}%", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw focus bar
        focus_color = (0, 255, 0) if focus_percentage >= 80 else \
                     (0, 255, 255) if focus_percentage >= 60 else \
                     (0, 0, 255)
        cv2.rectangle(frame, (200, 165), (200 + int(focus_percentage * 2), 185),
                     focus_color, -1)
        
        # Draw fatigue information
        fatigue_score = metrics['fatigue_score']
        cv2.putText(frame, f"Fatigue: {fatigue_score:.1f}%", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw fatigue bar
        fatigue_color = (0, 255, 0) if fatigue_score <= 30 else \
                       (0, 255, 255) if fatigue_score <= 60 else \
                       (0, 0, 255)
        cv2.rectangle(frame, (200, 195), (200 + int(fatigue_score * 2), 215),
                     fatigue_color, -1)
    
    def save_metrics_to_excel(self, filename=None):
        """Save focus and fatigue metrics to an Excel file"""
        try:
            # Calculate averages
            avg_focus = np.mean(self.focus_history) if self.focus_history else 0
            avg_fatigue = np.mean(self.metrics_history['fatigue_scores']) if self.metrics_history['fatigue_scores'] else 0
            
            # Calculate session duration
            session_duration = time.time() - self.start_time
            hours = int(session_duration // 3600)
            minutes = int((session_duration % 3600) // 60)
            seconds = int(session_duration % 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Create metrics dictionary
            metrics_data = {
                'Metric': ['Average Focus Score', 'Average Fatigue Score', 'Session Duration', 'Total Frames', 'Total Blinks'],
                'Value': [
                    f"{avg_focus:.2f}%",
                    f"{avg_fatigue:.2f}%",
                    duration_str,
                    self.frame_count,
                    self.total_blinks
                ]
            }
            
            # Create DataFrame
            df = pd.DataFrame(metrics_data)
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"focus_metrics_{timestamp}.xlsx"
            
            # Save to Excel
            df.to_excel(filename, index=False)
            
            print(f"Metrics saved to {filename}")
            print(f"Average Focus Score: {avg_focus:.2f}%")
            print(f"Average Fatigue Score: {avg_fatigue:.2f}%")
            
            return filename
            
        except Exception as e:
            print(f"Error saving metrics to Excel: {str(e)}")
            return None
    
    def cleanup(self):
        """Cleanup resources and save metrics"""
        try:
            # Save metrics to Excel
            excel_file = self.save_metrics_to_excel()
            if excel_file:
                print(f"\nMetrics saved to: {excel_file}")
            
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    # Initialize detector
    detector = FocusDetector()
    
    try:
        while True:
            ret, frame = detector.cap.read()
            if not ret:
                print("Error: Could not read from camera")
                break
            
            # Process frame
            display_frame, metrics = detector.process_frame(frame)
            
            # Show the frame
            cv2.imshow('Focus Detection', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("\nStarting calibration...")
                detector.start_calibration()
                print("Calibration completed!")
    
    except Exception as e:
        print(f"Error during execution: {e}")
    
    finally:
        # Cleanup
        detector.cleanup()
        print("\nSession ended. Metrics have been saved to Excel.")

if __name__ == "__main__":
    main()
