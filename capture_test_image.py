import cv2
import os

def capture_test_image():
    # Create test_images directory if it doesn't exist
    os.makedirs('test_images', exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\n=== Webcam Test Image Capture ===")
    print("1. Position your face in the center of the frame")
    print("2. Make sure your face is well-lit and clearly visible")
    print("3. Press 'c' to capture the image")
    print("4. Press 'q' to quit without saving")
    print("Waiting for capture command...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Display the frame
        cv2.imshow('Press "c" to capture, "q" to quit', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Save the captured frame
            output_path = 'test_images/test_face.jpg'
            cv2.imwrite(output_path, frame)
            print(f"\n✅ Image saved successfully to: {output_path}")
            break
        elif key == ord('q'):
            print("\n❌ Capture cancelled")
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_test_image() 