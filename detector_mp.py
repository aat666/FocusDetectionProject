import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import mediapipe as mp
from scipy.spatial import distance as dist
import subprocess  # For audio alerts

def play_audio_alert(sound_file="alert.wav"):
    try:
        subprocess.run(["afplay", sound_file])
    except Exception as e:
        print("Error playing audio alert:", e)

class FocusGazeDetectorMP:
    def __init__(self, cap=None):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.7
        )
        
        # Store camera capture object if provided
        self.cap = cap
        
        # Initialize camera parameters
        if cap is not None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            width, height = 640, 480  # Default values
            
        # Camera matrix initialization
        focal_length = width
        center = (width/2, height/2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        # Assuming no lens distortion
        self.dist_coeffs = np.zeros((4,1))
        
        # 3D model points for head pose estimation
        self.face_3d_model = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ]) / 4.5  # Scaling factor for better estimation
        
        # Enhanced debugging options
        self.DEBUG_MODE = True
        self.DEBUG_VERBOSE = False
        self.LOG_TO_FILE = False
        self.log_file = None
        
        # Enhanced metric tracking
        self.metrics_history = {
            'raw_ear': [],           # Raw EAR values
            'blink_intervals': [],    # Time between blinks
            'raw_angles': [],        # Raw head pose angles
            'smoothed_angles': [],   # Smoothed head pose angles
            'gaze_ratios': [],       # Raw gaze ratios
            'focus_scores': [],      # Final focus scores
            'penalty_scores': []     # Individual penalty contributions
        }
        
        # Blink detection parameters
        self.EAR_THRESHOLD = 0.25        # Threshold for detecting closed eyes
        self.CONSECUTIVE_FRAMES = 2       # Frames needed to confirm a blink
        self.BLINK_REFRACTORY_PERIOD = 0.2  # Minimum time (seconds) between blinks
        
        # Head pose smoothing
        self.SMOOTHING_WINDOW = 5  # Number of frames for moving average
        self.yaw_history = []
        self.pitch_history = []
        self.roll_history = []
        
        # Initialize tracking variables
        self.blink_counter = 0
        self.total_blinks = 0
        self.last_blink_time = time.time()
        self.start_time = time.time()
        self.frame_count = 0
        
        # Focus calculation parameters
        self.weights = {
            'gaze': 0.5,    # Gaze deviation has highest weight
            'head': 0.3,    # Head pose is second most important
            'blink': 0.2    # Blink rate has lowest weight
        }
        
        # Constants
        self.YAW_THRESHOLD = 20
        self.PITCH_THRESHOLD = 15
        self.GAZE_LEFT_THRESHOLD = 0.4
        self.GAZE_RIGHT_THRESHOLD = 0.6
        
        # Debugging and testing parameters
        self.TEST_GAZE_DEVIATION_THRESHOLD = 3.0  # Default 3 seconds
        self.FOCUS_RESET_THRESHOLD = 2.0  # Time needed in center to reset focus
        self.last_log_time = time.time()
        self.LOG_INTERVAL = 0.2  # Log every 200ms
        
        # State tracking
        self.gaze_deviation_start = None
        self.center_gaze_start = None
        self.current_gaze_state = "Center"
        self.current_focus_status = "Focus"
        
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
        
        # Initialize baseline values
        self.baseline = {
            'gaze_center': 0.5,
            'gaze_range': 0.15,
            'blink_rate_min': 8,
            'blink_rate_max': 30,
            'ear_threshold': 0.25,
            'normal_blink_rate': 15  # Average normal blink rate
        }
        
        # Session metrics tracking
        self.session_blinks = 0
        
        # Performance tracking
        self.last_debug_time = time.time()
        self.DEBUG_LOG_INTERVAL = 1.0  # Log every second
        
        # Initialize focus tracking
        self.current_focus = 100.0  # Start at 100%
        self.focus_history = []
        self.distraction_count = 0
        
        # Add WARMUP_FRAMES attribute
        self.WARMUP_FRAMES = 30  # Number of frames to warm up the focus calculation
        
        if self.LOG_TO_FILE:
            self.log_file = open('focus_detector_debug.log', 'w')

    def set_debug_mode(self, enabled=True, verbose=False):
        """Configure debug mode and verbosity"""
        self.DEBUG_MODE = enabled
        self.DEBUG_VERBOSE = verbose
        self._log_debug(f"Debug mode: {'ON' if enabled else 'OFF'}, Verbose: {'ON' if verbose else 'OFF'}", force=True)

    def set_test_threshold(self, seconds):
        """Set gaze deviation threshold for testing"""
        if seconds > 0:
            self._log_debug(f"Setting deviation threshold to {seconds}s", force=True)
            self.TEST_GAZE_DEVIATION_THRESHOLD = seconds
        else:
            print("Error: Threshold must be positive")

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

    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR) for given eye landmarks
        """
        try:
            # Vertical eye distances
            v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            
            # Horizontal eye distance
            h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            # Calculate EAR
            ear = (v1 + v2) / (2.0 * h) if h > 0 else 0.0
            
            if self.DEBUG_VERBOSE:
                self._log_debug(f"Raw EAR calculation - v1: {v1:.3f}, v2: {v2:.3f}, h: {h:.3f}, EAR: {ear:.3f}")
                
            return ear
            
        except Exception as e:
            self._log_debug(f"Error calculating EAR: {e}", force=True)
            return 0.0

    def smooth_value(self, value, history, smoothing=5):
        """Apply smoothing to values"""
        history.append(value)
        if len(history) > smoothing:
            history.pop(0)
        return np.mean(history)

    def draw_metrics(self, frame, metrics):
        # Create a completely black overlay for the metrics
        overlay = np.zeros_like(frame)
        
        # Draw metrics on fixed positions
        y_position = 30
        for key, value in metrics.items():
            text = f"{key}: {value}"
            # All metrics in green except Focus status
            color = (0, 255, 0)
            if key == 'Focus':
                color = (0, 255, 0) if value == "Focus" else (0, 0, 255)
            
            cv2.putText(
                overlay,
                text,
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            y_position += 30
        
        # Blend the overlay with the original frame
        alpha = 0.7
        return cv2.addWeighted(frame, 1, overlay, alpha, 0)

    def _check_gaze_deviation(self, gaze_ratio):
        """Check if gaze deviation exceeds time threshold with state persistence"""
        current_time = time.time()
        
        # Determine current gaze state
        if gaze_ratio < self.GAZE_LEFT_THRESHOLD:
            new_gaze_state = "Left"
        elif gaze_ratio > self.GAZE_RIGHT_THRESHOLD:
            new_gaze_state = "Right"
        else:
            new_gaze_state = "Center"
            
        # Log raw gaze data and state
        self._log_debug(
            f"Gaze Ratio: {gaze_ratio:.3f} | "
            f"Thresholds: [{self.GAZE_LEFT_THRESHOLD:.3f}, {self.GAZE_RIGHT_THRESHOLD:.3f}] | "
            f"State: {new_gaze_state}"
        )
            
        # Handle state transitions
        if new_gaze_state != self.current_gaze_state:
            if new_gaze_state == "Center":
                self.center_gaze_start = current_time
                if self.gaze_deviation_start is not None:
                    self._log_debug(f"Gaze returned to center after "
                                  f"{current_time - self.gaze_deviation_start:.1f}s",
                                  force=True)
                self.gaze_deviation_start = None
            else:
                self.center_gaze_start = None
                if self.gaze_deviation_start is None:
                    self.gaze_deviation_start = current_time
                    self._log_debug(f"Started deviation timer - New state: {new_gaze_state}",
                                  force=True)
            self.current_gaze_state = new_gaze_state
            
        # Check for sustained deviation
        is_sustained_deviation = False
        if self.gaze_deviation_start is not None:
            deviation_duration = current_time - self.gaze_deviation_start
            self._log_debug(
                f"Current deviation duration: {deviation_duration:.1f}s / "
                f"{self.TEST_GAZE_DEVIATION_THRESHOLD:.1f}s threshold"
            )
            
            if deviation_duration >= self.TEST_GAZE_DEVIATION_THRESHOLD:
                self._log_debug(
                    f"SUSTAINED DEVIATION DETECTED! "
                    f"Direction: {self.current_gaze_state} "
                    f"Duration: {deviation_duration:.1f}s",
                    force=True
                )
                is_sustained_deviation = True
                self.current_focus_status = "Not Focus"
                
        # Check for sustained center gaze (focus reset)
        elif self.center_gaze_start is not None and self.current_focus_status == "Not Focus":
            center_duration = current_time - self.center_gaze_start
            if center_duration >= self.FOCUS_RESET_THRESHOLD:
                self.current_focus_status = "Focus"
                self._log_debug(
                    f"Focus reset after {center_duration:.1f}s of centered gaze",
                    force=True
                )
                
        return is_sustained_deviation

    def _log_metrics(self, metrics, prefix=""):
        """Helper function to log metrics with optional prefix"""
        if self.DEBUG_VERBOSE:
            for key, value in metrics.items():
                self._log_debug(f"{prefix}{key}: {value}")
    
    def process_frame(self, frame):
        """Process frame and return display frame and metrics."""
        try:
            self.frame_count += 1
            current_time = time.time()
            
            # Initialize metrics with default values
            metrics = {
                'gaze_ratio': self.baseline['gaze_center'],
                'yaw': 0,
                'pitch': 0,
                'roll': 0,
                'ear': self.baseline['ear_threshold'],
                'blink_rate': 0,
                'focus_percentage': 100.0  # Start with maximum focus
            }
            
            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = np.array([[int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])]
                                    for lm in face_landmarks.landmark])
                
                # Calculate EAR
                left_eye = landmarks[[33, 160, 158, 133, 153, 144]]
                right_eye = landmarks[[362, 385, 387, 263, 373, 380]]
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                metrics['ear'] = (left_ear + right_ear) / 2.0
                
                # Calculate gaze ratio
                metrics['gaze_ratio'] = self.calculate_gaze_ratio(landmarks)
                
                # Calculate head pose
                pose_landmarks = np.array([
                    landmarks[1],    # Nose
                    landmarks[33],   # Left eye
                    landmarks[263],  # Right eye
                    landmarks[61],   # Mouth left
                    landmarks[291],  # Mouth right
                    landmarks[199]   # Chin
                ], dtype=np.float32)
                
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    self.face_3d_model, pose_landmarks, self.camera_matrix,
                    self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    rmat, _ = cv2.Rodrigues(rotation_vector)
                    angles = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                    metrics['yaw'] = angles[0]
                    metrics['pitch'] = angles[1]
                    metrics['roll'] = angles[2]
                
                # Compute focus percentage based on metrics
                metrics['focus_percentage'] = self.compute_focus_percentage(metrics)
                
            return frame, metrics
            
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            return frame, {'focus_percentage': 0.0}

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

    def get_face_3d_model(self):
        """Return the 3D face model points"""
        return self.face_3d_model

    def calculate_gaze_ratio(self, eye_points):
        """Calculate the gaze ratio based on eye landmarks"""
        eye_region = np.array(eye_points, dtype=np.int32)
        
        # Get the eye region dimensions
        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        
        # Calculate the width of the eye region
        eye_width = max_x - min_x
        if eye_width == 0:
            return 0.5  # Return center position if can't determine
        
        # Calculate the pupil position relative to the eye width
        pupil_x = np.mean(eye_region[:, 0])
        gaze_ratio = (pupil_x - min_x) / eye_width
        
        return gaze_ratio

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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                self._collect_calibration_data(results.multi_face_landmarks[0], frame)
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

    def _collect_calibration_data(self, face_landmarks, frame):
        """Collect data during calibration"""
        try:
            landmarks = np.array([[int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])]
                                for lm in face_landmarks.landmark])
            
            # Collect gaze data
            left_eye = landmarks[[33, 160, 158, 133, 153, 144]]
            right_eye = landmarks[[362, 385, 387, 263, 373, 380]]
            left_ratio = self.calculate_gaze_ratio(left_eye)
            right_ratio = self.calculate_gaze_ratio(right_eye)
            
            # Only append valid data
            if not (np.isnan(left_ratio) or np.isnan(right_ratio)):
                self.calibration_data['gaze_ratios'].append((left_ratio + right_ratio) / 2)
            
            # Collect head pose data
            image_points = np.array([
                landmarks[1], landmarks[33], landmarks[263],
                landmarks[61], landmarks[291], landmarks[199]
            ], dtype=np.float32)
            
            size = frame.shape
            focal_length = size[1]
            center_pt = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array([
                [focal_length, 0, center_pt[0]],
                [0, focal_length, center_pt[1]],
                [0, 0, 1]
            ], dtype="double")
            
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.get_face_3d_model(), image_points, camera_matrix,
                np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                rmat, _ = cv2.Rodrigues(rotation_vector)
                euler_angles = R.from_matrix(rmat).as_euler('zyx', degrees=True)
                
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
                'GAZE_STD': max(0.1, np.std(self.calibration_data['gaze_ratios']) * 2),  # Ensure minimum spread
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
        """
        Compute focus percentage with non-linear penalty scaling for better drowsiness detection.
        Uses exponential penalties to achieve better discrimination between drowsy and non-drowsy states.
        """
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

    def update_baseline_from_calibration(self, calibration_data):
        """Update baseline values from calibration data"""
        try:
            if 'gaze_center' in calibration_data:
                self.baseline['gaze_center'] = calibration_data['gaze_center']
            if 'gaze_range' in calibration_data:
                self.baseline['gaze_range'] = calibration_data['gaze_range']
            if 'ear_threshold' in calibration_data:
                self.baseline['ear_threshold'] = calibration_data['ear_threshold']
                self.EAR_THRESHOLD = calibration_data['ear_threshold']
            
            self._log_debug("Updated baseline values:", force=True)
            self._log_debug(f"Gaze center: {self.baseline['gaze_center']:.3f}")
            self._log_debug(f"Gaze range: {self.baseline['gaze_range']:.3f}")
            self._log_debug(f"EAR threshold: {self.baseline['ear_threshold']:.3f}")
            
        except Exception as e:
            self._log_debug(f"Error updating baseline: {e}", force=True)

    def cleanup(self):
        """Enhanced cleanup with debug log closing"""
        try:
            if self.LOG_TO_FILE and self.log_file:
                self.log_file.close()
                
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def set_thresholds(self, gaze_left=0.35, gaze_right=0.65, yaw=20.0, pitch=15.0):
        """Set detection thresholds"""
        self.GAZE_LEFT_THRESHOLD = gaze_left
        self.GAZE_RIGHT_THRESHOLD = gaze_right
        self.YAW_THRESHOLD = yaw
        self.PITCH_THRESHOLD = pitch
        
        if self.DEBUG_MODE:
            self._log_debug(
                f"Updated thresholds - Gaze L: {gaze_left:.3f}, "
                f"R: {gaze_right:.3f}, Yaw: {yaw}°, Pitch: {pitch}°"
            )

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
        
    # Initialize detector with camera
    detector = FocusGazeDetectorMP(cap)
    
    while detector.cap.isOpened():
        ret, frame = detector.cap.read()
        if not ret:
            break
            
        # Process frame and get the frame with metrics already drawn
        display_frame, _ = detector.process_frame(frame)
        
        # Display the frame
        cv2.imshow("Focus, Fatigue & Gaze Detection", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    detector.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()