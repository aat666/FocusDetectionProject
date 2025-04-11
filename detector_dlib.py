# detector_dlib.py
import cv2
import dlib
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance as dist
import subprocess  # For audio alerts
import os
import pandas as pd
from datetime import datetime

def play_audio_alert(sound_file="alert.wav"):
    try:
        subprocess.run(["afplay", sound_file])
    except Exception as e:
        print("Error playing audio alert:", e)

class FocusGazeDetectorDlib:
    def __init__(self, cap=None, predictor_path=None):
        # Initialize dlib's face detector and shape predictor.
        self.detector = dlib.get_frontal_face_detector()
        
        # Use absolute path for predictor file
        if predictor_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            predictor_path = os.path.join(current_dir, "models", "shape_predictor_68_face_landmarks.dat")
            
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Store camera capture object if provided
        self.cap = cap
        
        # Enhanced debugging options
        self.DEBUG_MODE = True
        self.DEBUG_VERBOSE = False
        self.LOG_TO_FILE = False
        self.log_file = None
        
        # For head pose estimation, define 3D model points.
        self.face_3d_model = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ]) / 4.5  # Adjust scaling as needed

        # We'll need the camera matrix; if cap is provided, get width and height.
        if cap is not None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            width, height = 640, 480
        focal_length = width
        center = (width / 2, height / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        self.dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion

        # Parameters for blink detection and focus calculation
        self.EAR_THRESHOLD = 0.25
        self.CONSECUTIVE_FRAMES = 2
        self.BLINK_REFRACTORY_PERIOD = 0.2
        
        # Enhanced metric tracking
        self.metrics_history = {
            'raw_ear': [],           # Raw EAR values
            'blink_intervals': [],    # Time between blinks
            'raw_angles': [],        # Raw head pose angles
            'smoothed_angles': [],   # Smoothed head pose angles
            'gaze_ratios': [],       # Raw gaze ratios
            'focus_scores': [],      # Final focus scores
            'penalty_scores': [],     # Individual penalty contributions
            'fatigue_scores': []     # Fatigue scores
        }
        
        # Initialize counters and metrics
        self.blink_counter = 0
        self.total_blinks = 0
        self.last_blink_time = time.time()
        self.start_time = time.time()
        self.frame_count = 0
        self.focus_history = []
        
        # Set weights for focus calculation (customize as needed)
        self.weights = {'gaze': 0.5, 'head': 0.3, 'blink': 0.2}
        
        # Set thresholds for head pose (in degrees)
        self.YAW_THRESHOLD = 20
        self.PITCH_THRESHOLD = 15
        
        # Head pose smoothing
        self.SMOOTHING_WINDOW = 5  # Number of frames for moving average
        self.yaw_history = []
        self.pitch_history = []
        self.roll_history = []
        
        # Default baseline for focus score computation
        self.baseline = {
            'ear_threshold': self.EAR_THRESHOLD,
            'gaze_center': 0.5,  # This could be replaced by a more robust measure
            'gaze_range': 0.15,
            'blink_rate_min': 8,
            'blink_rate_max': 30
        }
        
        # Calibration parameters
        self.calibration_data = {
            'gaze_ratios': [],
            'blink_intervals': [],
            'yaw_values': [],
            'pitch_values': [],
            'roll_values': [],
            'last_blink_time': time.time()
        }
        self.is_calibrated = False
        self.calibrated_thresholds = {}
        
        # Performance tracking
        self.last_debug_time = time.time()
        self.DEBUG_LOG_INTERVAL = 1.0  # Log every second
        
        # Initialize focus tracking
        self.current_focus = 100.0  # Start at 100%
        self.distraction_count = 0
        
        # Add WARMUP_FRAMES attribute
        self.WARMUP_FRAMES = 30  # Number of frames to warm up the focus calculation
        
        if self.LOG_TO_FILE:
            self.log_file = open('focus_detector_debug.log', 'w')

    def _log_debug(self, message, force=False):
        """Enhanced debug logging with file output option"""
        if self.DEBUG_MODE and (force or self.DEBUG_VERBOSE):
            current_time = time.time()
            timestamp = time.strftime('%H:%M:%S')
            log_message = f"[{timestamp}] {message}"
            
            print(log_message)
            
            if self.LOG_TO_FILE and self.log_file:
                self.log_file.write(log_message + '\n')
                self.log_file.flush()

    def smooth_value(self, value, history, smoothing=5):
        """Apply smoothing to values"""
        history.append(value)
        if len(history) > smoothing:
            history.pop(0)
        return np.mean(history)

    def start_calibration(self, duration=60):
        """Run calibration routine for specified duration"""
        print("\nStarting calibration mode...")
        print("Please look naturally at the screen for the next 60 seconds.")
        print("Maintain a comfortable posture and blink normally.\n")
        
        start_time = time.time()
        calibration_frames = 0
        
        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Process frame without displaying metrics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            
            if len(rects) > 0:
                self._collect_calibration_data(rects[0], gray, frame)
                calibration_frames += 1
            
            # Display countdown
            remaining_time = int(duration - (time.time() - start_time))
            cv2.putText(frame, f"Calibrating: {remaining_time}s", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Calibration", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self._compute_calibrated_thresholds()
        cv2.destroyWindow("Calibration")
        return self.is_calibrated

    def _collect_calibration_data(self, rect, gray, frame):
        """Collect data during calibration"""
        try:
            shape = self.predictor(gray, rect)
            landmarks = self.shape_to_np(shape)
            
            # Collect gaze data
            nose_point = landmarks[30]
            gaze_ratio = nose_point[0] / frame.shape[1]
            self.calibration_data['gaze_ratios'].append(gaze_ratio)
            
            # Collect head pose data
            image_points = np.array([
                landmarks[30],   # Nose tip
                landmarks[8],    # Chin
                landmarks[36],   # Left eye left corner
                landmarks[45],   # Right eye right corner
                landmarks[48],   # Left Mouth corner
                landmarks[54]    # Right mouth corner
            ], dtype="double")
            
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.face_3d_model, image_points, self.camera_matrix,
                self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                rmat, _ = cv2.Rodrigues(rotation_vector)
                euler_angles = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                
                # Only append valid angles
                if not np.any(np.isnan(euler_angles)):
                    self.calibration_data['yaw_values'].append(euler_angles[0])
                    self.calibration_data['pitch_values'].append(euler_angles[1])
                    self.calibration_data['roll_values'].append(euler_angles[2])
                    
        except Exception as e:
            print(f"Error collecting calibration data: {e}")

    def _compute_calibrated_thresholds(self):
        """Compute personalized thresholds from calibration data"""
        # Check if we have enough data points (at least 30)
        min_required_samples = 30
        
        data_counts = {
            'Gaze ratios': len(self.calibration_data['gaze_ratios']),
            'Yaw values': len(self.calibration_data['yaw_values']),
            'Pitch values': len(self.calibration_data['pitch_values']),
            'Roll values': len(self.calibration_data['roll_values'])
        }
        
        print("\nCollected data points:")
        for metric, count in data_counts.items():
            print(f"{metric}: {count}")
        
        if any(count < min_required_samples for count in data_counts.values()):
            print(f"\nInsufficient calibration data collected. Need at least {min_required_samples} samples for each metric.")
            print("Please try calibration again and ensure your face is clearly visible to the camera.")
            return
        
        try:
            # Compute mean and standard deviation for each metric
            self.calibrated_thresholds = {
                'GAZE_CENTER': np.mean(self.calibration_data['gaze_ratios']),
                'GAZE_STD': max(0.1, np.std(self.calibration_data['gaze_ratios']) * 2),
                'YAW_THRESHOLD': np.std(self.calibration_data['yaw_values']) * 2,
                'PITCH_THRESHOLD': np.std(self.calibration_data['pitch_values']) * 2
            }
            
            # Update thresholds with wider margins
            center = self.calibrated_thresholds['GAZE_CENTER']
            spread = self.calibrated_thresholds['GAZE_STD']
            
            # Ensure minimum threshold differences and constrain to reasonable ranges
            self.GAZE_LEFT_THRESHOLD = max(0.2, min(0.4, center - spread))
            self.GAZE_RIGHT_THRESHOLD = min(0.8, max(0.6, center + spread))
            
            # Ensure minimum range between thresholds
            if (self.GAZE_RIGHT_THRESHOLD - self.GAZE_LEFT_THRESHOLD) < 0.2:
                center = (self.GAZE_LEFT_THRESHOLD + self.GAZE_RIGHT_THRESHOLD) / 2
                self.GAZE_LEFT_THRESHOLD = center - 0.1
                self.GAZE_RIGHT_THRESHOLD = center + 0.1
            
            # Set reasonable bounds for head pose thresholds
            self.YAW_THRESHOLD = max(15, min(30, self.calibrated_thresholds['YAW_THRESHOLD']))
            self.PITCH_THRESHOLD = max(12, min(25, self.calibrated_thresholds['PITCH_THRESHOLD']))
            
            self.is_calibrated = True
            print("\nCalibration completed successfully!")
            print("Calibrated thresholds:")
            print(f"  Gaze Left: {self.GAZE_LEFT_THRESHOLD:.2f}")
            print(f"  Gaze Right: {self.GAZE_RIGHT_THRESHOLD:.2f}")
            print(f"  Yaw: ±{self.YAW_THRESHOLD:.2f}°")
            print(f"  Pitch: ±{self.PITCH_THRESHOLD:.2f}°")
            
        except Exception as e:
            print(f"Error computing calibration thresholds: {e}")
            self.is_calibrated = False

    def compute_focus_percentage(self, metrics):
        """Compute focus percentage with non-linear penalty scaling"""
        try:
            # Configuration parameters for penalty scaling
            self.penalty_config = {
                'gaze': {
                    'alpha': 5.0,      # Gaze deviation sensitivity
                    'max_penalty': 30  # Maximum gaze penalty
                },
                'head': {
                    'beta': 0.1,       # Yaw sensitivity
                    'gamma': 0.1,      # Pitch sensitivity
                    'max_penalty': 30  # Maximum head pose penalty
                },
                'blink': {
                    'delta': 0.5,      # Blink rate sensitivity
                    'max_penalty': 20  # Maximum blink penalty
                }
            }

            if self.DEBUG_VERBOSE:
                print("\n=== Focus Score Calculation ===")
                print(f"Raw metrics: {metrics}")
                print(f"Current baselines: {self.baseline}")

            # Start with maximum focus
            focus_score = 100.0
            penalties = {'gaze': 0.0, 'head': 0.0, 'blink': 0.0}

            # 1. Compute Gaze Penalty (non-linear)
            gaze_deviation = abs(metrics['gaze_ratio'] - self.baseline['gaze_center'])
            if gaze_deviation > self.baseline['gaze_range']:
                alpha = self.penalty_config['gaze']['alpha']
                max_pen = self.penalty_config['gaze']['max_penalty']
                penalties['gaze'] = max_pen * (1 - np.exp(-alpha * (gaze_deviation - self.baseline['gaze_range'])))
                
                if self.DEBUG_VERBOSE:
                    print(f"Gaze deviation: {gaze_deviation:.3f}")
                    print(f"Gaze penalty: {penalties['gaze']:.2f}")

            # 2. Compute Head Pose Penalty (non-linear)
            yaw = abs(metrics['yaw'])
            pitch = abs(metrics['pitch'])
            
            # Separate penalties for yaw and pitch
            yaw_penalty = 0.0
            if yaw > self.YAW_THRESHOLD:
                beta = self.penalty_config['head']['beta']
                max_pen = self.penalty_config['head']['max_penalty']
                yaw_penalty = max_pen * (1 - np.exp(-beta * (yaw - self.YAW_THRESHOLD)))

            pitch_penalty = 0.0
            if pitch > self.PITCH_THRESHOLD:
                gamma = self.penalty_config['head']['gamma']
                max_pen = self.penalty_config['head']['max_penalty']
                pitch_penalty = max_pen * (1 - np.exp(-gamma * (pitch - self.PITCH_THRESHOLD)))

            # Take maximum of yaw and pitch penalties
            penalties['head'] = max(yaw_penalty, pitch_penalty)
            
            if self.DEBUG_VERBOSE:
                print(f"Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°")
                print(f"Head pose penalty: {penalties['head']:.2f}")

            # 3. Compute Blink Penalty (non-linear)
            blink_rate = metrics.get('blink_rate', self.baseline['blink_rate_min'])
            delta = self.penalty_config['blink']['delta']
            max_pen = self.penalty_config['blink']['max_penalty']

            if blink_rate < self.baseline['blink_rate_min']:
                deviation = self.baseline['blink_rate_min'] - blink_rate
                penalties['blink'] = max_pen * (1 - np.exp(-delta * deviation))
            elif blink_rate > self.baseline['blink_rate_max']:
                deviation = blink_rate - self.baseline['blink_rate_max']
                penalties['blink'] = max_pen * (1 - np.exp(-delta * deviation))

            if self.DEBUG_VERBOSE:
                print(f"Blink rate: {blink_rate:.1f}/min")
                print(f"Blink penalty: {penalties['blink']:.2f}")

            # Combine penalties using weights
            total_penalty = (
                self.weights['gaze'] * penalties['gaze'] +
                self.weights['head'] * penalties['head'] +
                self.weights['blink'] * penalties['blink']
            )

            # Calculate final score
            focus_score = max(0, min(100, 100 - total_penalty))

            if self.DEBUG_VERBOSE:
                print("\nPenalty Breakdown:")
                for key, value in penalties.items():
                    print(f"  {key}: {value:.2f} (weight: {self.weights[key]:.2f})")
                print(f"Total weighted penalty: {total_penalty:.2f}")
                print(f"Final focus score: {focus_score:.2f}")

            return focus_score

        except Exception as e:
            if self.DEBUG_VERBOSE:
                print(f"Error in focus calculation: {str(e)}")
            return 0.0

    def compute_fatigue_score(self, metrics):
        """
        Compute fatigue score based on various metrics.
        
        Args:
            metrics (dict): Dictionary containing current metrics including:
                - blink_rate: Blinks per minute
                - ear: Eye aspect ratio
                - yaw: Head yaw angle
                - pitch: Head pitch angle
                
        Returns:
            float: Fatigue score from 0 (fully alert) to 100 (extremely fatigued)
        """
        # Initialize fatigue score components
        blink_fatigue = 0.0
        ear_fatigue = 0.0
        head_pose_fatigue = 0.0
        
        # Calculate blink rate fatigue (40% weight)
        baseline_blink_rate = 15.0  # Normal blink rate per minute
        blink_deviation = abs(metrics['blink_rate'] - baseline_blink_rate)
        blink_fatigue = min(40.0, blink_deviation * 2.0)  # Cap at 40%
        
        # Calculate EAR fatigue (30% weight)
        baseline_ear = self.baseline['ear_threshold']
        ear_deviation = abs(metrics['ear'] - baseline_ear)
        ear_fatigue = min(30.0, ear_deviation * 100.0)  # Cap at 30%
        
        # Calculate head pose fatigue (30% weight)
        yaw_penalty = min(15.0, abs(metrics['yaw']) * 0.5)  # Cap at 15%
        pitch_penalty = min(15.0, abs(metrics['pitch']) * 0.5)  # Cap at 15%
        head_pose_fatigue = yaw_penalty + pitch_penalty
        
        # Calculate total fatigue score
        total_fatigue = blink_fatigue + ear_fatigue + head_pose_fatigue
        
        # Normalize to 0-100 range
        fatigue_score = min(100.0, total_fatigue)
        
        # Store fatigue score in history
        self.metrics_history['fatigue_scores'].append(fatigue_score)
        
        # Log detailed breakdown if in verbose mode
        if self.DEBUG_VERBOSE:
            self._log_debug(f"Fatigue Score Breakdown:")
            self._log_debug(f"  Blink Fatigue: {blink_fatigue:.1f}%")
            self._log_debug(f"  EAR Fatigue: {ear_fatigue:.1f}%")
            self._log_debug(f"  Head Pose Fatigue: {head_pose_fatigue:.1f}%")
            self._log_debug(f"  Total Fatigue Score: {fatigue_score:.1f}%")
        
        return fatigue_score

    def generate_session_summary(self):
        """Generate session summary with focus statistics"""
        try:
            session_duration = time.time() - self.start_time
            hours = int(session_duration // 3600)
            minutes = int((session_duration % 3600) // 60)
            seconds = int(session_duration % 60)
            
            # Calculate final focus statistics
            avg_focus = np.mean(self.focus_history) if self.focus_history else 100
            min_focus = np.min(self.focus_history) if self.focus_history else 100
            max_focus = np.max(self.focus_history) if self.focus_history else 100
            
            summary = (
                "\n=== Session Summary ===\n"
                f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}\n"
                f"Total Frames: {self.frame_count}\n"
                f"Total Distractions: {self.distraction_count}\n"
                f"\nFocus Statistics:\n"
                f"  Average: {avg_focus:.1f}%\n"
                f"  Range: {min_focus:.1f}% - {max_focus:.1f}%\n"
                f"  Distraction Rate: {self.distraction_count / (session_duration / 60):.1f} per minute\n"
            )
            
            return summary
            
        except Exception as e:
            return f"Error generating summary: {e}"

    def shape_to_np(self, shape, dtype="int"):
        # Convert dlib shape object to NumPy array
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def calculate_ear(self, eye):
        # Calculate eye aspect ratio
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C) if C > 0 else 0.0
        return ear

    def process_frame(self, frame):
        """Process frame and return display frame and metrics."""
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
                'fatigue_score': 0.0  # Add fatigue score to metrics
            }
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            if len(rects) > 0:
                # For simplicity, take the first detected face
                rect = rects[0]
                shape = self.predictor(gray, rect)
                landmarks = self.shape_to_np(shape)
                
                # Extract eyes landmarks (using 68-landmarks indices)
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                metrics['ear'] = (left_ear + right_ear) / 2.0
                
                # Blink detection with enhanced tracking
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
                
                # Calculate blink rate (blinks per minute)
                session_duration = current_time - self.start_time
                metrics['blink_rate'] = (self.total_blinks / session_duration) * 60 if session_duration > 0 else 0
                
                # Compute head pose with smoothing
                image_points = np.array([
                    landmarks[30],   # Nose tip
                    landmarks[8],    # Chin
                    landmarks[36],   # Left eye left corner
                    landmarks[45],   # Right eye right corner
                    landmarks[48],   # Left Mouth corner
                    landmarks[54]    # Right mouth corner
                ], dtype="double")
                
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    self.face_3d_model, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    rmat, _ = cv2.Rodrigues(rotation_vector)
                    euler_angles = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                    
                    # Apply smoothing to angles
                    metrics['yaw'] = self.smooth_value(euler_angles[0], self.yaw_history)
                    metrics['pitch'] = self.smooth_value(euler_angles[1], self.pitch_history)
                    metrics['roll'] = self.smooth_value(euler_angles[2], self.roll_history)
                    
                    # Store raw angles for history
                    self.metrics_history['raw_angles'].append(euler_angles)
                
                # Compute gaze ratio with enhanced tracking
                nose_point = landmarks[30]
                metrics['gaze_ratio'] = nose_point[0] / frame.shape[1]
                self.metrics_history['gaze_ratios'].append(metrics['gaze_ratio'])
                
                # Compute focus percentage using non-linear scoring
                metrics['focus_percentage'] = self.compute_focus_percentage(metrics)
                
                # Compute fatigue score
                metrics['fatigue_score'] = self.compute_fatigue_score(metrics)
                
                # Track focus history
                self.focus_history.append(metrics['focus_percentage'])
                
                # Update distraction count
                if metrics['focus_percentage'] < 60:  # Threshold for distraction
                    self.distraction_count += 1
                
                # Draw enhanced visualization
                self._draw_debug_info(frame, metrics, current_time)
                
                # Draw landmarks
                for (x, y) in landmarks:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            return frame, metrics
            
        except Exception as e:
            self._log_debug(f"Error in process_frame: {str(e)}", force=True)
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

    def cleanup(self):
        """Enhanced cleanup with debug log closing and metrics saving"""
        try:
            # Save metrics to Excel
            excel_file = self.save_metrics_to_excel()
            if excel_file:
                print(f"\nMetrics saved to: {excel_file}")
            
            if self.LOG_TO_FILE and self.log_file:
                self.log_file.close()
                
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def set_debug_mode(self, enabled=True, verbose=False):
        """Configure debug mode and verbosity"""
        self.DEBUG_MODE = enabled
        self.DEBUG_VERBOSE = verbose
        self._log_debug(f"Debug mode: {'ON' if enabled else 'OFF'}, Verbose: {'ON' if verbose else 'OFF'}", force=True)

    def set_thresholds(self, gaze_left=0.35, gaze_right=0.65, yaw=20.0, pitch=15.0):
        """Set detection thresholds for gaze and head pose."""
        self.GAZE_LEFT_THRESHOLD = gaze_left
        self.GAZE_RIGHT_THRESHOLD = gaze_right
        self.YAW_THRESHOLD = yaw
        self.PITCH_THRESHOLD = pitch
        
        if self.DEBUG_MODE:
            self._log_debug(
                f"Updated thresholds - Gaze L: {gaze_left:.3f}, "
                f"R: {gaze_right:.3f}, Yaw: {yaw}°, Pitch: {pitch}°"
            )

    def save_metrics_to_excel(self, filename=None):
        """
        Save focus and fatigue metrics to an Excel file.
        
        Args:
            filename (str, optional): Name of the Excel file. If None, generates a timestamp-based name.
        """
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
            
            if self.DEBUG_VERBOSE:
                self._log_debug(f"Metrics saved to {filename}")
                self._log_debug(f"Average Focus Score: {avg_focus:.2f}%")
                self._log_debug(f"Average Fatigue Score: {avg_fatigue:.2f}%")
            
            return filename
            
        except Exception as e:
            if self.DEBUG_VERBOSE:
                self._log_debug(f"Error saving metrics to Excel: {str(e)}")
            return None
