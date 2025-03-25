# main.py
import cv2
from detector_dlib import FocusGazeDetectorDlib
import argparse

def main():
    parser = argparse.ArgumentParser(description='Focus and Gaze Detection using dlib')
    parser.add_argument('--calibrate', action='store_true', help='Run calibration mode before starting')
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Use the dlib-based detector
    detector = FocusGazeDetectorDlib(cap)
    
    if args.calibrate:
        # (If you want to implement calibration with dlib, add your routine here)
        print("Calibration not implemented in this example.")
    
    print("\nStarting focus detection...")
    print("Press 'q' to quit")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            display_frame, metrics = detector.process_frame(frame)
            # Optionally, print metrics to the console
            print(f"Focus: {metrics['focus_percentage']:.1f}% | EAR: {metrics['ear']:.3f} | Yaw: {metrics['yaw']:.1f}")
            
            cv2.imshow("Focus Detection (dlib)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
