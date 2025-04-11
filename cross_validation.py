#!/usr/bin/env python3

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from detector_dlib import FocusGazeDetectorDlib
from sklearn.model_selection import KFold
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm

class FocusCrossValidator:
    def __init__(self, dataset_path: str, n_splits: int = 5):
        """Initialize the cross-validation system.
        
        Args:
            dataset_path (str): Path to the dataset directory
            n_splits (int): Number of folds for cross-validation
        """
        self.dataset_path = Path(dataset_path)
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Initialize results storage
        self.fold_results = {}
        self.aggregate_metrics = {}
        
        # Performance tracking
        self.processing_times = []
        
    def load_dataset(self) -> List[Tuple[str, List[str]]]:
        """Load and organize the dataset.
        
        Returns:
            List of tuples containing (folder_name, list_of_image_paths)
        """
        dataset = []
        for folder_num in range(1, 57):  # Assuming folders 0001 to 0056
            folder_name = f"{folder_num:04d}"
            folder_path = self.dataset_path / folder_name
            
            if folder_path.exists():
                image_paths = [str(p) for p in folder_path.glob("*.jpg")]
                if image_paths:
                    dataset.append((folder_name, image_paths))
        
        return dataset
    
    def process_fold(self, train_fold: List[Tuple[str, List[str]]], 
                    val_fold: List[Tuple[str, List[str]]], 
                    fold_idx: int) -> Dict:
        """Process a single fold of the cross-validation.
        
        Args:
            train_fold: Training data for this fold
            val_fold: Validation data for this fold
            fold_idx: Index of the current fold
            
        Returns:
            Dictionary containing fold results
        """
        print(f"\nProcessing Fold {fold_idx + 1}/{self.n_splits}")
        
        # Initialize detector
        detector = FocusGazeDetectorDlib(cap=None)
        
        # Process training data
        train_scores = []
        train_start_time = time.time()
        
        print("Processing training data...")
        for folder_name, image_paths in tqdm(train_fold):
            folder_scores = []
            for img_path in image_paths:
                try:
                    image = cv2.imread(img_path)
                    if image is not None:
                        _, metrics = detector.process_frame(image)
                        folder_scores.append(metrics['focus_percentage'])
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            if folder_scores:
                train_scores.extend(folder_scores)
        
        # Process validation data
        val_scores = []
        val_start_time = time.time()
        
        print("Processing validation data...")
        for folder_name, image_paths in tqdm(val_fold):
            folder_scores = []
            for img_path in image_paths:
                try:
                    image = cv2.imread(img_path)
                    if image is not None:
                        _, metrics = detector.process_frame(image)
                        folder_scores.append(metrics['focus_percentage'])
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            if folder_scores:
                val_scores.extend(folder_scores)
        
        # Calculate fold metrics
        fold_metrics = {
            'train_mean': np.mean(train_scores) if train_scores else 0,
            'train_std': np.std(train_scores) if train_scores else 0,
            'val_mean': np.mean(val_scores) if val_scores else 0,
            'val_std': np.std(val_scores) if val_scores else 0,
            'train_time': time.time() - train_start_time,
            'val_time': time.time() - val_start_time,
            'train_samples': len(train_scores),
            'val_samples': len(val_scores)
        }
        
        return fold_metrics
    
    def run_cross_validation(self):
        """Run the complete cross-validation process."""
        print("Starting cross-validation process...")
        
        # Load dataset
        dataset = self.load_dataset()
        if not dataset:
            raise ValueError("No data found in the specified dataset path")
        
        print(f"Loaded {len(dataset)} folders")
        
        # Prepare data for cross-validation
        X = np.array(range(len(dataset)))
        
        total_time = 0
        # Run cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(self.kf.split(X)):
            # Split data
            train_fold = [dataset[i] for i in train_idx]
            val_fold = [dataset[i] for i in val_idx]
            
            # Process fold
            fold_start_time = time.time()  # Start timing for this fold
            fold_metrics = self.process_fold(train_fold, val_fold, fold_idx)
            fold_duration = time.time() - fold_start_time  # Calculate duration for this fold
            total_time += fold_duration
            
            self.fold_results[f'fold_{fold_idx + 1}'] = fold_metrics
            
            # Print fold results
            print(f"\nFold {fold_idx + 1} Results:")
            print(f"Training Mean Score: {fold_metrics['train_mean']:.2f} ± {fold_metrics['train_std']:.2f}")
            print(f"Validation Mean Score: {fold_metrics['val_mean']:.2f} ± {fold_metrics['val_std']:.2f}")
            print(f"Training Time: {fold_metrics['train_time']:.2f}s")
            print(f"Validation Time: {fold_metrics['val_time']:.2f}s")
        
        # Calculate aggregate metrics
        self._calculate_aggregate_metrics()
        
        # Plot results
        self._plot_results()
    
    def _calculate_aggregate_metrics(self):
        """Calculate aggregate metrics across all folds."""
        train_means = [fold['train_mean'] for fold in self.fold_results.values()]
        val_means = [fold['val_mean'] for fold in self.fold_results.values()]
        
        self.aggregate_metrics = {
            'mean_train_score': np.mean(train_means),
            'std_train_score': np.std(train_means),
            'mean_val_score': np.mean(val_means),
            'std_val_score': np.std(val_means),
            'total_train_samples': sum(fold['train_samples'] for fold in self.fold_results.values()),
            'total_val_samples': sum(fold['val_samples'] for fold in self.fold_results.values()),
            'total_train_time': sum(fold['train_time'] for fold in self.fold_results.values()),
            'total_val_time': sum(fold['val_time'] for fold in self.fold_results.values())
        }
    
    def _plot_results(self):
        """Plot cross-validation results."""
        plt.figure(figsize=(12, 6))
        
        # Plot training and validation scores
        fold_numbers = range(1, self.n_splits + 1)
        train_means = [self.fold_results[f'fold_{i}']['train_mean'] for i in fold_numbers]
        val_means = [self.fold_results[f'fold_{i}']['val_mean'] for i in fold_numbers]
        
        plt.plot(fold_numbers, train_means, 'b-o', label='Training Score')
        plt.plot(fold_numbers, val_means, 'r-o', label='Validation Score')
        
        plt.title('Cross-Validation Results')
        plt.xlabel('Fold Number')
        plt.ylabel('Mean Focus Score')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig('cross_validation_results.png')
        plt.close()
    
    def print_summary(self):
        """Print summary of cross-validation results."""
        print("\n=== Cross-Validation Summary ===")
        print(f"Number of Folds: {self.n_splits}")
        print(f"\nAggregate Metrics:")
        print(f"Mean Training Score: {self.aggregate_metrics['mean_train_score']:.2f} ± {self.aggregate_metrics['std_train_score']:.2f}")
        print(f"Mean Validation Score: {self.aggregate_metrics['mean_val_score']:.2f} ± {self.aggregate_metrics['std_val_score']:.2f}")
        print(f"\nSample Counts:")
        print(f"Total Training Samples: {self.aggregate_metrics['total_train_samples']}")
        print(f"Total Validation Samples: {self.aggregate_metrics['total_val_samples']}")
        print(f"\nProcessing Times:")
        print(f"Total Training Time: {self.aggregate_metrics['total_train_time']:.2f}s")
        print(f"Total Validation Time: {self.aggregate_metrics['total_val_time']:.2f}s")
        print(f"Average Time per Sample: {(self.aggregate_metrics['total_train_time'] + self.aggregate_metrics['total_val_time']) / (self.aggregate_metrics['total_train_samples'] + self.aggregate_metrics['total_val_samples']):.3f}s")

def main():
    # Hardcoded dataset path
    dataset_path = "/Users/ani/Downloads/columbia_dataset"
    
    # Initialize and run cross-validation
    validator = FocusCrossValidator(dataset_path, n_splits=5)
    validator.run_cross_validation()
    validator.print_summary()

if __name__ == "__main__":
    main() 