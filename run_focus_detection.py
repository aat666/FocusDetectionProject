import os
import sys

# Try importing required packages with error handling
try:
    import cv2
    import numpy as np
    from detector_dlib import FocusGazeDetectorDlib
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("\nPlease make sure all dependencies are installed:")
    print("pip install numpy opencv-python dlib scipy pandas openpyxl")
    sys.exit(1)

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize detector
    try:
        detector = FocusGazeDetectorDlib(cap=cap)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        cap.release()
        return
    
    print("\nFocus Detection System Started")
    print("Press 'q' to quit")
    print("Press 'c' to start calibration")
    
    try:
        while True:
            ret, frame = cap.read()
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
        cap.release()
        cv2.destroyAllWindows()
        print("\nSession ended. Metrics have been saved to Excel.")

if __name__ == "__main__":
    main() 