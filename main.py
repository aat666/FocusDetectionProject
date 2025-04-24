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
        
        # First release and re-open the camera to ensure clean state
        if cap is None:
            self.cap.release()
            self.cap = cv2.VideoCapture(0)
        
        # Set camera properties in specific order
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Verify resolution was set correctly
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera resolution set to: {actual_width}x{actual_height}")
        
        # If resolution is not what we want, try alternative resolutions in order
        if actual_width != 1280 or actual_height != 720:
            resolutions = [(1920, 1080), (1280, 720), (960, 540), (640, 480)]
            for width, height in resolutions:
                print(f"Trying resolution: {width}x{height}")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if actual_width == width and actual_height == height:
                    print(f"Successfully set resolution to: {width}x{height}")
                    break
        
        # Store the actual resolution for later use
        self.frame_width = int(actual_width)
        self.frame_height = int(actual_height)
        
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
            'fatigue_scores': [],
            'focus_scores': []
        }
        
        # Initialize smoothing buffers
        self.yaw_history = []
        self.pitch_history = []
        self.roll_history = []
        
        # Set up camera matrix for head pose estimation
        self.camera_matrix = np.array([
            [1000, 0, self.frame_width/2],  # Use actual camera width
            [0, 1000, self.frame_height/2],  # Use actual camera height
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
        """Compute focus percentage based on multiple metrics with simplified recovery mechanism"""
        # Get session duration
        session_duration = time.time() - self.start_time
        
        # Apply warm-up period for the first 10 seconds
        if session_duration < 10:
            return 100.0
        
        # Get previous focus score
        previous_score = self.metrics_history['focus_scores'][-1] if self.metrics_history['focus_scores'] else 100.0
        
        # Calculate attention indicators
        gaze_ratio = metrics['gaze_ratio']
        gaze_deviation = abs(gaze_ratio - self.baseline['gaze_center'])
        yaw_deviation = abs(metrics['yaw'])
        pitch_deviation = abs(metrics['pitch'])
        
        # Define thresholds
        GAZE_THRESHOLD = 0.4      # Maximum allowed gaze deviation
        HEAD_THRESHOLD = 25.0     # Maximum allowed head angle deviation
        
        # Calculate attention score (0 to 1)
        gaze_score = max(0, 1 - (gaze_deviation / GAZE_THRESHOLD))
        head_score = max(0, 1 - (max(yaw_deviation, pitch_deviation) / HEAD_THRESHOLD))
        
        # Weighted attention score (70% gaze, 30% head position)
        attention_score = (gaze_score * 0.7 + head_score * 0.3)
        
        # Time delta (assuming 30 FPS)
        time_delta = 1.0 / 30.0
        
        # Define states based on attention score
        HIGHLY_FOCUSED = 0.8
        FOCUSED = 0.6
        PARTIALLY_FOCUSED = 0.4
        
        # Calculate score change based on attention state
        if attention_score >= HIGHLY_FOCUSED:
            # Rapid recovery when highly focused
            recovery_rate = 60.0  # 60% per second
            score_change = recovery_rate * time_delta
            
            # Bonus recovery when returning from low focus
            if previous_score < 80.0:
                score_change *= 2.0  # Double recovery rate
                
        elif attention_score >= FOCUSED:
            # Standard recovery when focused
            recovery_rate = 40.0  # 40% per second
            score_change = recovery_rate * time_delta
            
        elif attention_score >= PARTIALLY_FOCUSED:
            # Slow recovery when partially focused
            recovery_rate = 20.0  # 20% per second
            score_change = recovery_rate * time_delta
            
        else:
            # Apply penalty when not focused
            penalty_rate = 30.0  # 30% per second
            score_change = -penalty_rate * time_delta * (1.0 - attention_score)
            
            # Reduce penalty for high scores to prevent sudden drops
            if previous_score > 90:
                score_change *= 0.3
            elif previous_score > 70:
                score_change *= 0.6
        
        # Calculate new focus score
        focus_score = previous_score + score_change
        
        # Ensure immediate recovery starts when attention returns
        if attention_score >= FOCUSED and score_change > 0:
            # Guarantee minimum score increase when focused
            min_increase = 0.5  # Minimum 0.5% increase per frame when focused
            focus_score = max(focus_score, previous_score + min_increase)
        
        # Prevent score from dropping when highly focused
        if attention_score >= HIGHLY_FOCUSED:
            focus_score = max(focus_score, previous_score)
        
        # Ensure score stays within valid range
        focus_score = max(0.0, min(100.0, focus_score))
        
        # Store focus score in history
        self.metrics_history['focus_scores'].append(focus_score)
        
        # Debug output
        if self.DEBUG_VERBOSE:
            print(f"\nFocus Score Analysis:")
            print(f"Session duration: {session_duration:.1f}s")
            print(f"Attention scores - Gaze: {gaze_score:.2f}, Head: {head_score:.2f}")
            print(f"Combined attention: {attention_score:.2f}")
            print(f"State: {'Highly Focused' if attention_score >= HIGHLY_FOCUSED else 'Focused' if attention_score >= FOCUSED else 'Partially Focused' if attention_score >= PARTIALLY_FOCUSED else 'Distracted'}")
            print(f"Score change: {score_change:+.2f}%")
            print(f"Score: {previous_score:.1f}% → {focus_score:.1f}%")
        
        return focus_score
    
    def compute_fatigue_score(self, metrics):
        """Compute fatigue score based on various metrics with recovery mechanism"""
        # Get session duration
        session_duration = time.time() - self.start_time
        
        # Start with 0% fatigue for the first 10 seconds
        if session_duration < 10:
            return 0.0
        
        # Get previous fatigue score
        previous_score = self.metrics_history['fatigue_scores'][-1] if self.metrics_history['fatigue_scores'] else 0.0
        
        # Calculate base fatigue score
        fatigue_score = previous_score
        
        # Gradually increase sensitivity from 10-20 seconds
        ramp_factor = min(1.0, (session_duration - 10) / 10) if session_duration < 20 else 1.0
        
        # Calculate alertness indicators
        ear = metrics['ear']
        blink_rate = metrics['blink_rate']
        yaw_deviation = abs(metrics['yaw'])
        pitch_deviation = abs(metrics['pitch'])
        
        # Define alertness thresholds
        EAR_THRESHOLD = self.baseline['ear_threshold'] * 0.9  # 90% of baseline EAR
        BLINK_RATE_THRESHOLD = 25  # blinks per minute
        YAW_THRESHOLD = 15.0
        PITCH_THRESHOLD = 15.0
        
        # Calculate alertness factors (0 to 1, where 1 is most alert)
        ear_factor = min(1.0, max(0.0, (ear - EAR_THRESHOLD) / (self.baseline['ear_threshold'] - EAR_THRESHOLD)))
        blink_factor = min(1.0, max(0.0, 1.0 - (blink_rate / BLINK_RATE_THRESHOLD)))
        head_factor = min(1.0, max(0.0, 1.0 - max(yaw_deviation/YAW_THRESHOLD, pitch_deviation/PITCH_THRESHOLD)))
        
        # Combined alertness factor (0 to 1)
        alertness_factor = (ear_factor * 0.4 + blink_factor * 0.4 + head_factor * 0.2)
        
        # Recovery and increase rates (percent per second)
        BASE_RECOVERY_RATE = 10.0  # Base recovery rate when alert
        MAX_RECOVERY_RATE = 20.0   # Maximum recovery rate when fully alert
        INCREASE_RATE = 8.0        # Rate of fatigue increase when showing fatigue signs
        
        # Time since last update
        time_delta = 1.0 / 30.0  # Assuming 30 FPS
        
        if alertness_factor > 0.6:  # Show good signs of alertness
            # Calculate recovery rate based on alertness
            recovery_rate = BASE_RECOVERY_RATE + (MAX_RECOVERY_RATE - BASE_RECOVERY_RATE) * alertness_factor
            recovery_amount = recovery_rate * time_delta
            fatigue_score = max(0.0, fatigue_score - recovery_amount)
        else:
            # Calculate fatigue increase based on metrics
            if ear < EAR_THRESHOLD:
                ear_penalty = min(40.0, (EAR_THRESHOLD - ear) * 200.0 * ramp_factor)
            else:
                ear_penalty = 0.0
            
            if blink_rate > BLINK_RATE_THRESHOLD:
                blink_penalty = min(40.0, (blink_rate - BLINK_RATE_THRESHOLD) * 2.0 * ramp_factor)
            else:
                blink_penalty = 0.0
            
            head_penalty = min(20.0, (max(yaw_deviation, pitch_deviation) - 10.0) * 0.5 * ramp_factor)
            head_penalty = max(0.0, head_penalty)
            
            # Apply increases gradually
            total_increase = (ear_penalty + blink_penalty + head_penalty) * (time_delta * INCREASE_RATE / 100.0)
            fatigue_score = min(100.0, fatigue_score + total_increase)
        
        # Store fatigue score in history
        self.metrics_history['fatigue_scores'].append(fatigue_score)
        
        # Debug output
        if self.DEBUG_VERBOSE:
            print(f"\nFatigue Score Analysis:")
            print(f"Alertness Factors - EAR: {ear_factor:.2f}, Blink: {blink_factor:.2f}, Head: {head_factor:.2f}")
            print(f"Combined alertness: {alertness_factor:.2f}")
            if alertness_factor > 0.6:
                print(f"Recovery mode: -{recovery_amount:.2f}% (rate={recovery_rate:.1f}%/s)")
            else:
                print(f"Increase mode: +{total_increase:.2f}%")
            print(f"EAR: {ear:.3f} (threshold={EAR_THRESHOLD:.3f})")
            print(f"Blink rate: {blink_rate:.1f}/min (threshold={BLINK_RATE_THRESHOLD})")
            print(f"Head pose - Yaw: {yaw_deviation:.1f}°, Pitch: {pitch_deviation:.1f}°")
            print(f"Fatigue Score: {previous_score:.1f}% → {fatigue_score:.1f}%")
        
        return fatigue_score
    
    def start_calibration(self):
        """Perform calibration to establish baseline metrics"""
        print("Starting calibration...")
        print("Please look directly at the camera and maintain a neutral expression.")
        print("Calibration will take 5 seconds...")
        
        calibration_frames = 30
        calibration_data = {
            'ear': [],
            'gaze_ratio': [],
            'yaw': [],
            'pitch': []
        }
        
        start_time = time.time()
        frames_processed = 0
        
        while frames_processed < calibration_frames and (time.time() - start_time) < 10:
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
                    
                    frames_processed += 1
                    
                    # Draw calibration progress
                    progress = frames_processed / calibration_frames
                    cv2.putText(frame, f"Calibrating: {int(progress * 100)}%", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Calibration', frame)
                    cv2.waitKey(1)
            
            time.sleep(0.1)
        
        cv2.destroyWindow('Calibration')
        
        # Update baseline metrics with more robust calculations
        if len(calibration_data['ear']) > 0:
            # Use median instead of mean for more robust baseline values
            self.baseline['ear_threshold'] = np.median(calibration_data['ear'])
            self.baseline['gaze_center'] = np.median(calibration_data['gaze_ratio'])
            
            # Calculate more forgiving thresholds for head pose
            yaw_std = np.std(calibration_data['yaw'])
            pitch_std = np.std(calibration_data['pitch'])
            
            # Set thresholds to be more forgiving (3 standard deviations)
            self.baseline['yaw_threshold'] = max(25, yaw_std * 3)
            self.baseline['pitch_threshold'] = max(20, pitch_std * 3)
            
            print(f"Calibration completed! Baseline values:")
            print(f"EAR threshold: {self.baseline['ear_threshold']:.3f}")
            print(f"Gaze center: {self.baseline['gaze_center']:.3f}")
            print(f"Yaw threshold: {self.baseline['yaw_threshold']:.1f}°")
            print(f"Pitch threshold: {self.baseline['pitch_threshold']:.1f}°")
            
            # Reset focus and fatigue scores after calibration
            self.focus_history = []
            self.metrics_history['focus_scores'] = []
            self.metrics_history['fatigue_scores'] = []
            self.frame_count = 0
            self.start_time = time.time()
        else:
            print("Calibration failed. Using default values.")
            # Set reasonable default values if calibration fails
            self.baseline['ear_threshold'] = 0.3
            self.baseline['gaze_center'] = 0.5
            self.baseline['yaw_threshold'] = 25  # More forgiving default
            self.baseline['pitch_threshold'] = 20  # More forgiving default
    
    def process_frame(self, frame):
        """Process frame and return display frame and metrics"""
        try:
            self.frame_count += 1
            current_time = time.time()
            
            # Initialize metrics with correct starting values
            metrics = {
                'ear': self.baseline['ear_threshold'],
                'focus_percentage': 100.0,  # Start at 100% focus
                'gaze_ratio': self.baseline['gaze_center'],
                'yaw': 0,
                'pitch': 0,
                'roll': 0,
                'blink_rate': 0,
                'fatigue_score': 0.0  # Start at 0% fatigue
            }
            
            # If this is the first frame, return initial values
            if self.frame_count == 1:
                return frame, metrics
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            
            # Handle case when no face is detected
            if len(rects) == 0:
                # Get previous focus score
                previous_score = self.metrics_history['focus_scores'][-1] if self.metrics_history['focus_scores'] else 100.0
                
                # Calculate time since last frame
                time_delta = 1.0 / 30.0  # Assuming 30 FPS
                
                # Define penalty rate for no face detection (faster than normal distraction)
                NO_FACE_PENALTY_RATE = 50.0  # 50% per second
                
                # Calculate penalty
                penalty = NO_FACE_PENALTY_RATE * time_delta
                
                # Apply penalty with protection for high scores
                if previous_score > 90:
                    penalty *= 0.5  # Reduced penalty for very high scores
                elif previous_score > 70:
                    penalty *= 0.7  # Slightly reduced penalty for high scores
                
                # Calculate new focus score
                new_focus_score = max(0.0, previous_score - penalty)
                
                # Update metrics
                metrics['focus_percentage'] = new_focus_score
                self.metrics_history['focus_scores'].append(new_focus_score)
                
                # Increment distraction count if focus is low
                if new_focus_score < 60:
                    self.distraction_count += 1
                
                # Draw warning on frame
                # Get frame dimensions
                frame_height, frame_width = frame.shape[:2]
                
                # Calculate text size and position
                text = "DISTRACTED"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Calculate position to center the text
                text_x = (frame_width - text_width) // 2
                text_y = (frame_height + text_height) // 2
                
                # Draw text with outline for better visibility
                # Draw black outline
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
                # Draw red text
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
                
                # Draw debug info
                self._draw_debug_info(frame, metrics, current_time)
                
                return frame, metrics
            
            # Normal face detection processing continues here
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
            
            # Compute gaze ratio using stored frame dimensions
            nose_point = landmarks[30]
            metrics['gaze_ratio'] = nose_point[0] / self.frame_width
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
            return frame, {'focus_percentage': 100.0, 'fatigue_score': 0.0}  # Return initial values on error
    
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
    
    def save_metrics_to_excel(self, first_name, last_name):
        """Save metrics to fixed Excel and CSV files with user information"""
        try:
            # Calculate session duration
            session_duration = time.time() - self.start_time
            
            # Calculate average focus and fatigue scores
            avg_focus = np.mean(self.metrics_history['focus_scores']) if self.metrics_history['focus_scores'] else 0
            avg_fatigue = np.mean(self.metrics_history['fatigue_scores']) if self.metrics_history['fatigue_scores'] else 0
            
            # Calculate other metrics
            avg_ear = np.mean(self.metrics_history['raw_ear']) if self.metrics_history['raw_ear'] else 0
            avg_gaze = np.mean(self.metrics_history['gaze_ratios']) if self.metrics_history['gaze_ratios'] else 0
            avg_yaw = np.mean(self.yaw_history) if self.yaw_history else 0
            avg_pitch = np.mean(self.pitch_history) if self.pitch_history else 0
            avg_roll = np.mean(self.roll_history) if self.roll_history else 0
            
            # Create metrics dictionary
            metrics_data = {
                'First Name': [first_name],
                'Last Name': [last_name],
                'Date': [datetime.now().strftime('%Y-%m-%d')],
                'Time': [datetime.now().strftime('%H:%M:%S')],
                'Session Duration (seconds)': [session_duration],
                'Total Blinks': [self.total_blinks],
                'Average Focus Score (%)': [avg_focus],
                'Average Fatigue Score (%)': [avg_fatigue],
                'Average EAR': [avg_ear],
                'Average Gaze Ratio': [avg_gaze],
                'Average Yaw (degrees)': [avg_yaw],
                'Average Pitch (degrees)': [avg_pitch],
                'Average Roll (degrees)': [avg_roll],
                'Distraction Count': [self.distraction_count]
            }
            
            # Create DataFrame for current session
            current_df = pd.DataFrame(metrics_data)
            
            # Define fixed filenames
            excel_file = "focus_detection_records.xlsx"
            csv_file = "focus_detection_records.csv"
            
            # Handle Excel file
            try:
                # Try to read existing Excel file
                existing_df = pd.read_excel(excel_file)
                # Append new data
                updated_df = pd.concat([existing_df, current_df], ignore_index=True)
            except FileNotFoundError:
                # If file doesn't exist, use current data
                updated_df = current_df
            
            # Save to Excel
            updated_df.to_excel(excel_file, index=False)
            
            # Handle CSV file
            try:
                # Try to read existing CSV file
                existing_csv_df = pd.read_csv(csv_file)
                # Append new data
                updated_csv_df = pd.concat([existing_csv_df, current_df], ignore_index=True)
            except FileNotFoundError:
                # If file doesn't exist, use current data
                updated_csv_df = current_df
            
            # Save to CSV
            updated_csv_df.to_csv(csv_file, index=False)
            
            print(f"\nMetrics saved to:")
            print(f"Excel file: {excel_file}")
            print(f"CSV file: {csv_file}")
            
            # Print summary
            print(f"\nSession Summary for {first_name} {last_name}:")
            print(f"Session Duration: {session_duration:.1f} seconds")
            print(f"Average Focus Score: {avg_focus:.2f}%")
            print(f"Average Fatigue Score: {avg_fatigue:.2f}%")
            print(f"Total Blinks: {self.total_blinks}")
            print(f"Distraction Count: {self.distraction_count}")
            print(f"\nTotal records in database: {len(updated_df)}")
            
            return excel_file, csv_file
            
        except Exception as e:
            print(f"Error saving metrics: {str(e)}")
            return None, None
    
    def cleanup(self):
        """Cleanup resources"""
        try:
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
