import time
import numpy as np
import cv2
import dlib
from main import FocusDetector
from detector_dlib import FocusGazeDetectorDlib
import psutil
import os
from datetime import datetime
import sys

class PerformanceBenchmark:
    def __init__(self):
        print("Initializing benchmark components...")
        self.detector = FocusDetector(cap=None)
        self.dlib_detector = FocusGazeDetectorDlib(cap=None)
        self.results = []
        self.start_time = None
        self.process = psutil.Process(os.getpid())
        print("Initialization complete!")

    def start_benchmark(self):
        """Start the benchmark timer"""
        self.start_time = time.time()
        self.results = []

    def measure_metric(self, metric_name, func, *args, **kwargs):
        """Measure performance of a specific metric"""
        # Measure CPU usage before
        cpu_before = self.process.cpu_percent()
        
        # Measure memory usage before
        memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure execution time
        start = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start
        
        # Measure CPU usage after
        cpu_after = self.process.cpu_percent()
        
        # Measure memory usage after
        memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Store results
        self.results.append({
            'metric': metric_name,
            'execution_time': execution_time,
            'cpu_usage': (cpu_before + cpu_after) / 2,
            'memory_usage': memory_after - memory_before,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        return result

    def run_face_detection_benchmark(self, iterations=100):
        """Benchmark face detection performance"""
        print("\nRunning Face Detection Benchmark...")
        self.start_benchmark()
        
        # Create a mock frame
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        gray = cv2.cvtColor(mock_frame, cv2.COLOR_BGR2GRAY)
        
        for i in range(iterations):
            sys.stdout.write(f"\rProgress: {i+1}/{iterations} iterations")
            sys.stdout.flush()
            self.measure_metric(
                f'face_detection_{i}',
                self.dlib_detector.detector,
                gray, 0
            )
        print()  # New line after progress
        self.print_results("Face Detection")

    def run_landmark_detection_benchmark(self, iterations=100):
        """Benchmark facial landmark detection performance"""
        print("\nRunning Landmark Detection Benchmark...")
        self.start_benchmark()
        
        # Create mock face rectangle using dlib's rectangle class
        mock_rect = dlib.rectangle(0, 0, 100, 100)
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        gray = cv2.cvtColor(mock_frame, cv2.COLOR_BGR2GRAY)
        
        for i in range(iterations):
            sys.stdout.write(f"\rProgress: {i+1}/{iterations} iterations")
            sys.stdout.flush()
            self.measure_metric(
                f'landmark_detection_{i}',
                self.dlib_detector.predictor,
                gray,
                mock_rect
            )
        print()  # New line after progress
        self.print_results("Landmark Detection")

    def run_focus_calculation_benchmark(self, iterations=100):
        """Benchmark focus calculation performance"""
        print("\nRunning Focus Calculation Benchmark...")
        self.start_benchmark()
        
        # Create mock metrics
        mock_metrics = {
            'gaze_ratio': 0.5,
            'yaw': 0,
            'pitch': 0,
            'ear': 0.3
        }
        
        for i in range(iterations):
            sys.stdout.write(f"\rProgress: {i+1}/{iterations} iterations")
            sys.stdout.flush()
            self.measure_metric(
                f'focus_calculation_{i}',
                self.detector.compute_focus_percentage,
                mock_metrics
            )
        print()  # New line after progress
        self.print_results("Focus Calculation")

    def run_fatigue_calculation_benchmark(self, iterations=100):
        """Benchmark fatigue calculation performance"""
        print("\nRunning Fatigue Calculation Benchmark...")
        self.start_benchmark()
        
        # Create mock metrics
        mock_metrics = {
            'ear': 0.3,
            'blink_rate': 15,
            'yaw': 0,
            'pitch': 0
        }
        
        for i in range(iterations):
            sys.stdout.write(f"\rProgress: {i+1}/{iterations} iterations")
            sys.stdout.flush()
            self.measure_metric(
                f'fatigue_calculation_{i}',
                self.detector.compute_fatigue_score,
                mock_metrics
            )
        print()  # New line after progress
        self.print_results("Fatigue Calculation")

    def print_results(self, benchmark_name):
        """Print benchmark results"""
        if not self.results:
            return
            
        # Calculate statistics
        execution_times = [r['execution_time'] for r in self.results]
        cpu_usages = [r['cpu_usage'] for r in self.results]
        memory_usages = [r['memory_usage'] for r in self.results]
        
        print(f"\n{benchmark_name} Benchmark Results:")
        print("----------------------------------------")
        print(f"Total iterations: {len(self.results)}")
        print(f"Average execution time: {np.mean(execution_times):.6f} seconds")
        print(f"Min execution time: {np.min(execution_times):.6f} seconds")
        print(f"Max execution time: {np.max(execution_times):.6f} seconds")
        print(f"Standard deviation: {np.std(execution_times):.6f} seconds")
        print(f"Average CPU usage: {np.mean(cpu_usages):.2f}%")
        print(f"Average memory usage: {np.mean(memory_usages):.2f} MB")
        print("----------------------------------------")

    def run_all_benchmarks(self, iterations=100):
        """Run all benchmarks"""
        print("\nStarting Performance Benchmark Suite...")
        print("This will run 4 benchmarks with 100 iterations each.")
        print("You will see progress indicators for each benchmark.")
        print("Press Ctrl+C to stop the benchmark at any time.\n")
        
        try:
            self.run_face_detection_benchmark(iterations)
            self.run_landmark_detection_benchmark(iterations)
            self.run_focus_calculation_benchmark(iterations)
            self.run_fatigue_calculation_benchmark(iterations)
            print("\nAll benchmarks completed successfully!")
        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user. Partial results:")
            self.print_results("Partial Results")
        except Exception as e:
            print(f"\n\nAn error occurred: {str(e)}")
            print("Partial results:")
            self.print_results("Partial Results")

if __name__ == '__main__':
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks(iterations=100) 