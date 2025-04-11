import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
from main import FocusDetector
import os

class LoginDialog:
    def __init__(self, parent):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("User Login")
        self.dialog.geometry("300x150")
        self.dialog.resizable(False, False)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.geometry("+%d+%d" % (
            parent.winfo_rootx() + parent.winfo_width()/2 - 150,
            parent.winfo_rooty() + parent.winfo_height()/2 - 75
        ))
        
        # Create and pack widgets
        ttk.Label(self.dialog, text="Please enter your credentials:").pack(pady=10)
        
        # First name
        ttk.Label(self.dialog, text="First Name:").pack()
        self.first_name = ttk.Entry(self.dialog)
        self.first_name.pack(pady=2)
        
        # Last name
        ttk.Label(self.dialog, text="Last Name:").pack()
        self.last_name = ttk.Entry(self.dialog)
        self.last_name.pack(pady=2)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Start Session", command=self.ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.LEFT, padx=5)
        
        # Initialize result
        self.result = None
        
        # Bind Enter key to ok
        self.dialog.bind('<Return>', lambda e: self.ok())
        
        # Wait for dialog to close
        parent.wait_window(self.dialog)
    
    def ok(self):
        first = self.first_name.get().strip()
        last = self.last_name.get().strip()
        if first and last:
            self.result = (first, last)
            self.dialog.destroy()
        else:
            messagebox.showerror("Error", "Please enter both first and last name")
    
    def cancel(self):
        self.dialog.destroy()

class FocusDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Focus Detection System")
        self.root.geometry("1400x9000")  # Increased from 1200x800
        
        # Initialize variables
        self.is_running = False
        self.detector = None
        self.camera_thread = None
        self.user_credentials = None
        
        # Create main container with grid layout
        self.main_container = ttk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid weights
        self.main_container.grid_columnconfigure(0, weight=3)  # Increased weight for video feed
        self.main_container.grid_columnconfigure(1, weight=1)  # Controls take less space
        self.main_container.grid_rowconfigure(0, weight=1)
        
        # Create left panel for video feed
        self.video_frame = ttk.LabelFrame(self.main_container, text="Video Feed")
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create video label with fixed size
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(padx=10, pady=10)
        
        # Create right panel for controls and metrics
        self.control_frame = ttk.LabelFrame(self.main_container, text="Controls & Metrics")
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Create control buttons
        self.create_control_buttons()
        
        # Initialize metrics variables
        self.focus_var = tk.StringVar(value="Focus: 0%")
        self.fatigue_var = tk.StringVar(value="Fatigue: 0%")
        self.blink_var = tk.StringVar(value="Blinks: 0")
        self.time_var = tk.StringVar(value="Time: 00:00:00")
        self.user_var = tk.StringVar(value="User: Not logged in")
        
        # Create metrics display
        self.create_metrics_display()
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Set minimum window size
        self.root.minsize(1200, 800)  # Set minimum size to ensure controls are visible
    
    def create_control_buttons(self):
        """Create control buttons"""
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Start button
        self.start_button = ttk.Button(button_frame, text="Start Session", command=self.start_detection)
        self.start_button.pack(fill=tk.X, pady=2)
        
        # Stop button
        self.stop_button = ttk.Button(button_frame, text="Stop Session", command=self.stop_detection)
        self.stop_button.pack(fill=tk.X, pady=2)
        self.stop_button.config(state=tk.DISABLED)  # Initially disabled
        
        # Calibrate button
        self.calibrate_button = ttk.Button(button_frame, text="Calibrate", command=self.calibrate)
        self.calibrate_button.pack(fill=tk.X, pady=2)
        
        # Save metrics button
        self.save_button = ttk.Button(button_frame, text="Save Metrics", command=self.save_metrics)
        self.save_button.pack(fill=tk.X, pady=2)
        
        # Initially disable buttons
        self.calibrate_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
    
    def create_metrics_display(self):
        """Create metrics display area"""
        metrics_frame = ttk.LabelFrame(self.control_frame, text="Metrics")
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # User info
        ttk.Label(metrics_frame, textvariable=self.user_var).pack(pady=5)
        
        # Focus score
        ttk.Label(metrics_frame, textvariable=self.focus_var).pack(pady=5)
        self.focus_bar = ttk.Progressbar(metrics_frame, length=200, mode='determinate')
        self.focus_bar.pack(pady=5)
        
        # Fatigue score
        ttk.Label(metrics_frame, textvariable=self.fatigue_var).pack(pady=5)
        self.fatigue_bar = ttk.Progressbar(metrics_frame, length=200, mode='determinate')
        self.fatigue_bar.pack(pady=5)
        
        # Blink count
        ttk.Label(metrics_frame, textvariable=self.blink_var).pack(pady=5)
        
        # Session time
        ttk.Label(metrics_frame, textvariable=self.time_var).pack(pady=5)
    
    def start_detection(self):
        """Start focus detection"""
        # Show login dialog first
        login = LoginDialog(self.root)
        if login.result:
            first_name, last_name = login.result
            self.user_credentials = (first_name, last_name)
            self.user_var.set(f"User: {first_name} {last_name}")
            
            try:
                # Initialize camera first
                cap = cv2.VideoCapture(0)
                
                # Set camera properties before creating detector
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Try to set resolution in order of preference
                resolutions = [(1280, 720), (960, 540), (640, 480)]
                for width, height in resolutions:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    if actual_width == width and actual_height == height:
                        print(f"Successfully set resolution to: {width}x{height}")
                        break
                
                # Initialize detector with preconfigured camera
                self.detector = FocusDetector(cap=cap)
                
                self.is_running = True
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.calibrate_button.config(state=tk.NORMAL)
                self.save_button.config(state=tk.NORMAL)
                self.status_var.set("Detection running...")
                
                # Start camera thread
                self.camera_thread = threading.Thread(target=self.update_video)
                self.camera_thread.daemon = True
                self.camera_thread.start()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start detection: {str(e)}")
                if 'cap' in locals():
                    cap.release()
    
    def stop_detection(self):
        """Stop focus detection"""
        if messagebox.askyesno("Stop Session", "Are you sure you want to stop the current session?"):
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.calibrate_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            self.status_var.set("Detection stopped")
            
            if self.detector:
                self.detector.cleanup()
                self.detector = None
    
    def calibrate(self):
        """Start calibration process"""
        if self.detector:
            self.status_var.set("Calibrating...")
            self.calibrate_button.config(state=tk.DISABLED)
            
            def calibration_thread():
                self.detector.start_calibration()
                self.root.after(0, self.calibration_complete)
            
            threading.Thread(target=calibration_thread).start()
    
    def calibration_complete(self):
        """Handle calibration completion"""
        self.status_var.set("Calibration completed")
        self.calibrate_button.config(state=tk.NORMAL)
    
    def save_metrics(self):
        """Save current metrics to Excel"""
        if self.detector and self.user_credentials:
            try:
                first_name, last_name = self.user_credentials
                filename = self.detector.save_metrics_to_excel(first_name, last_name)
                if filename:
                    messagebox.showinfo("Success", f"Metrics saved to {filename}")
                else:
                    messagebox.showerror("Error", "Failed to save metrics")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save metrics: {str(e)}")
        else:
            messagebox.showerror("Error", "No active session or user not logged in")
    
    def update_video(self):
        """Update video feed and metrics"""
        last_update_time = time.time()
        target_fps = 30.0
        frame_delay = 1.0 / target_fps
        
        while self.is_running:
            try:
                # Maintain consistent frame rate
                current_time = time.time()
                elapsed = current_time - last_update_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                
                ret, frame = self.detector.cap.read()
                if ret:
                    # Process frame
                    display_frame, metrics = self.detector.process_frame(frame)
                    
                    # Resize frame to fit the video label while maintaining aspect ratio
                    height, width = display_frame.shape[:2]
                    max_size = 720  # Increased from 480 to 720 for larger display
                    scale = min(max_size/width, max_size/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))
                    
                    # Update metrics display
                    self.focus_var.set(f"Focus: {metrics['focus_percentage']:.1f}%")
                    self.fatigue_var.set(f"Fatigue: {metrics['fatigue_score']:.1f}%")
                    self.blink_var.set(f"Blinks: {self.detector.total_blinks}")
                    
                    # Update progress bars
                    self.focus_bar['value'] = metrics['focus_percentage']
                    self.fatigue_bar['value'] = metrics['fatigue_score']
                    
                    # Update time
                    session_duration = time.time() - self.detector.start_time
                    hours = int(session_duration // 3600)
                    minutes = int((session_duration % 3600) // 60)
                    seconds = int(session_duration % 60)
                    self.time_var.set(f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
                    
                    # Convert frame for display
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    # Update video label
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
                    
                    # Update last frame time
                    last_update_time = time.time()
                
                # Update GUI less frequently than frame processing
                if not hasattr(self, '_last_gui_update') or \
                   time.time() - self._last_gui_update > 0.033:  # ~30 fps for GUI updates
                    self.root.update()
                    self._last_gui_update = time.time()
                
            except Exception as e:
                print(f"Error in video update: {str(e)}")
                self.stop_detection()
                break
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            if messagebox.askokcancel("Quit", "Do you want to quit? This will stop the detection."):
                self.stop_detection()
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    root = tk.Tk()
    app = FocusDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 