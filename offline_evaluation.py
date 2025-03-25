#!/usr/bin/env python3

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from detector_dlib import FocusGazeDetectorDlib
import sys
import time
from datetime import datetime, timedelta

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def format_time(seconds):
    """Format time in seconds to a human-readable string."""
    return str(timedelta(seconds=int(seconds)))

def print_progress_bar(current, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end='\r'):
    """Print a colored progress bar."""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    # Color the progress bar based on completion
    if current == total:
        color = Colors.GREEN
    elif current / total > 0.7:
        color = Colors.CYAN
    elif current / total > 0.3:
        color = Colors.BLUE
    else:
        color = Colors.WARNING
    
    print(f'\r{prefix} {color}{bar}{Colors.ENDC} {percent}% {suffix}', end=print_end)

class FocusEvaluator:
    def __init__(self, dataset_path: str):
        """Initialize the focus evaluator with dataset path."""
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        # Initialize detector without camera capture
        self.detector = FocusGazeDetectorDlib(cap=None)
        self.results: Dict[str, List[float]] = {}
        
    def process_image(self, image_path: str) -> float:
        """Process a single image and return its focus percentage."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"{Colors.FAIL}Failed to read image: {image_path}{Colors.ENDC}")
                raise ValueError(f"Failed to read image: {image_path}")
            
            # Process frame using the detector's process_frame method
            _, metrics = self.detector.process_frame(image)
            
            # Get focus percentage from metrics
            focus_percentage = metrics.get('focus_percentage', 0.0)
            return focus_percentage
            
        except Exception as e:
            print(f"{Colors.FAIL}Error processing image {image_path}: {str(e)}{Colors.ENDC}")
            import traceback
            traceback.print_exc()
            return -1.0

    def process_folder(self, folder_path: Path) -> Tuple[List[float], int]:
        """Process all images in a folder and return scores and error count."""
        scores = []
        error_count = 0
        
        # Get list of images first
        images = list(folder_path.glob("*.jpg"))
        
        for img_path in images:
            score = self.process_image(str(img_path))
            if score >= 0:
                scores.append(score)
            else:
                error_count += 1
                
        return scores, error_count

    def evaluate_dataset(self) -> Dict[str, Dict]:
        """Evaluate the entire dataset and return statistics."""
        total_images = 0
        total_errors = 0
        all_scores = []
        
        # Get total number of images
        total_image_count = 0
        print(f"{Colors.BOLD}Counting images...{Colors.ENDC}", end="", flush=True)
        for folder_num in range(1, 57):
            folder_path = self.dataset_path / f"{folder_num:04d}"
            if folder_path.exists():
                total_image_count += len(list(folder_path.glob("*.jpg")))
        print(f" {Colors.GREEN}Found {total_image_count} images{Colors.ENDC}")

        # Progress tracking variables
        processed_images = 0
        start_time = time.time()
        current_time = start_time

        try:
            # Process each numbered folder
            for folder_num in range(1, 57):
                folder_name = f"{folder_num:04d}"
                folder_path = self.dataset_path / folder_name
                
                if not folder_path.exists():
                    continue
                
                # Show which folder is being processed
                print(f"\n{Colors.BOLD}Processing folder {folder_name}:{Colors.ENDC}")
                
                # Process images with loading bar
                images = list(folder_path.glob("*.jpg"))
                folder_start_time = time.time()
                
                for i, img_path in enumerate(images, 1):
                    # Calculate progress and time estimates
                    elapsed = time.time() - start_time
                    speed = processed_images / elapsed if elapsed > 0 else 0
                    remaining_images = total_image_count - processed_images
                    eta = remaining_images / speed if speed > 0 else 0
                    
                    # Update progress bar with time estimates
                    prefix = f'Processing: {i}/{len(images)}'
                    suffix = f'| Speed: {speed:.1f} img/s | ETA: {format_time(eta)}'
                    print_progress_bar(i, len(images), prefix=prefix, suffix=suffix)
                    
                    # Process image
                    score = self.process_image(str(img_path))
                    if score >= 0:
                        all_scores.append(score)
                    else:
                        total_errors += 1
                    
                    # Update counters
                    processed_images += 1
                    current_time = time.time()
                
                # Print folder completion stats
                folder_time = time.time() - folder_start_time
                print(f"\n{Colors.GREEN}Folder {folder_name} completed in {format_time(folder_time)}{Colors.ENDC}")
                
                # Update results
                self.results[folder_name] = all_scores[-len(images):]

        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Processing interrupted!{Colors.ENDC}")
            return {}
        except Exception as e:
            print(f"\n{Colors.FAIL}Error during processing: {e}{Colors.ENDC}")
            return {}
        finally:
            total_time = time.time() - start_time
            print(f"\n{Colors.BOLD}Processing complete!{Colors.ENDC}")
            print(f"Time taken: {format_time(total_time)}")
            print(f"Average speed: {processed_images/total_time:.1f} images/second")

        # Compute statistics
        stats = {
            'summary': {
                'total_images': processed_images,
                'total_errors': total_errors,
                'overall_mean': np.mean(all_scores) if all_scores else 0,
                'overall_std': np.std(all_scores) if all_scores else 0,
                'overall_min': min(all_scores) if all_scores else 0,
                'overall_max': max(all_scores) if all_scores else 0
            }
        }
        
        # Compute per-folder statistics
        for folder_name, scores in self.results.items():
            if scores:
                stats[folder_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }
                
        return stats

    def plot_histogram(self, save_path: str = None):
        """Generate and optionally save a histogram of all focus scores."""
        all_scores = []
        for scores in self.results.values():
            all_scores.extend(scores)
            
        plt.figure(figsize=(12, 6))
        plt.hist(all_scores, bins=50, edgecolor='black', color='#2ecc71', alpha=0.7)
        plt.title('Distribution of Focus Scores', fontsize=14, pad=20)
        plt.xlabel('Focus Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add mean and std lines
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_score:.2f}')
        plt.axvline(mean_score + std_score, color='gray', linestyle='dashed', linewidth=1, alpha=0.5)
        plt.axvline(mean_score - std_score, color='gray', linestyle='dashed', linewidth=1, alpha=0.5)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

def print_statistics(stats: Dict):
    """Print formatted statistics to the terminal."""
    print(f"\n{Colors.BOLD}=== Focus Evaluation Results ==={Colors.ENDC}")
    
    # Calculate averages across all folders
    folder_stats = [v for k, v in stats.items() if k != 'summary']
    avg_metrics = {
        'mean_focus': np.mean([s['mean'] for s in folder_stats]),
        'mean_std': np.mean([s['std'] for s in folder_stats]),
        'mean_min': np.mean([s['min'] for s in folder_stats]),
        'mean_max': np.mean([s['max'] for s in folder_stats]),
        'mean_count': np.mean([s['count'] for s in folder_stats])
    }
    
    print(f"\n{Colors.BOLD}Overall Summary:{Colors.ENDC}")
    print(f"Total images processed: {stats['summary']['total_images']}")
    print(f"Total errors: {stats['summary']['total_errors']}")
    print(f"Overall mean focus score: {stats['summary']['overall_mean']:.2f}")
    print(f"Overall std deviation: {stats['summary']['overall_std']:.2f}")
    print(f"Range: {stats['summary']['overall_min']:.2f} - {stats['summary']['overall_max']:.2f}")
    
    print(f"\n{Colors.BOLD}Average Metrics Across All Folders:{Colors.ENDC}")
    print(f"  Average Focus Score: {avg_metrics['mean_focus']:.2f}")
    print(f"  Average Std Deviation: {avg_metrics['mean_std']:.2f}")
    print(f"  Average Range: {avg_metrics['mean_min']:.2f} - {avg_metrics['mean_max']:.2f}")
    print(f"  Average Images per Folder: {avg_metrics['mean_count']:.1f}")
    
    print(f"\n{Colors.BOLD}Per-folder Statistics:{Colors.ENDC}")
    for folder_name, folder_stats in sorted(stats.items()):
        if folder_name != 'summary':
            print(f"\n{Colors.BLUE}Folder {folder_name}:{Colors.ENDC}")
            print(f"  Images: {folder_stats['count']}")
            print(f"  Mean: {folder_stats['mean']:.2f}")
            print(f"  Std: {folder_stats['std']:.2f}")
            print(f"  Range: {folder_stats['min']:.2f} - {folder_stats['max']:.2f}")

def main():
    # Hardcoded dataset path
    dataset_path = "/Users/ani/Downloads/columbia_dataset"
    
    parser = argparse.ArgumentParser(description='Evaluate focus detection on a dataset of images')
    parser.add_argument('--save_histogram', type=str, default=None,
                      help='Optional path to save the histogram plot')
    args = parser.parse_args()

    try:
        evaluator = FocusEvaluator(dataset_path)
        stats = evaluator.evaluate_dataset()
        print_statistics(stats)
        
        if args.save_histogram:
            evaluator.plot_histogram(args.save_histogram)
            print(f"\n{Colors.GREEN}Histogram saved to: {args.save_histogram}{Colors.ENDC}")
            
    except Exception as e:
        print(f"{Colors.FAIL}Error: {str(e)}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main() 