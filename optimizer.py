import numpy as np
import random
from typing import Dict, List, Tuple
from offline_evaluation import FocusEvaluator
from detector_dlib import FocusGazeDetectorDlib
import time
from tqdm import tqdm
import cv2

class FocusOptimizer:
    def __init__(self, dataset_path: str):
        """Initialize the focus optimizer with dataset path."""
        self.dataset_path = dataset_path
        self.evaluator = FocusEvaluator(dataset_path)
        
        # Define parameter ranges for optimization
        self.parameter_ranges = {
            'ear_threshold': (0.2, 0.3),
            'yaw_threshold': (15, 30),
            'pitch_threshold': (10, 25),
            'gaze_weight': (0.4, 0.6),
            'head_weight': (0.2, 0.4),
            'blink_weight': (0.1, 0.3)
        }
        
        # Initialize best parameters
        self.best_parameters = None
        self.best_score = 0
        self.generation_history = []
        
        # Define target size for resizing
        self.target_width = 640  # Standard width for face detection
        self.target_height = 480  # Standard height for face detection
        
    def resize_image(self, image):
        """Resize image while maintaining aspect ratio."""
        height, width = image.shape[:2]
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Calculate new dimensions
        if width > height:
            new_width = self.target_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = self.target_height
            new_width = int(new_height * aspect_ratio)
            
        # Resize image
        resized = cv2.resize(image, (new_width, new_height))
        return resized
        
    def create_individual(self) -> Dict:
        """Create a random set of parameters within defined ranges."""
        return {
            param: random.uniform(ranges[0], ranges[1])
            for param, ranges in self.parameter_ranges.items()
        }
        
    def evaluate_fitness(self, parameters: Dict) -> float:
        """Evaluate how well these parameters perform."""
        try:
            # Create detector with current parameters
            detector = FocusGazeDetectorDlib(cap=None)
            detector.set_thresholds(
                gaze_left=0.35,
                gaze_right=0.65,
                yaw=parameters['yaw_threshold'],
                pitch=parameters['pitch_threshold']
            )
            
            # Update weights
            detector.weights = {
                'gaze': parameters['gaze_weight'],
                'head': parameters['head_weight'],
                'blink': parameters['blink_weight']
            }
            
            # Update EAR threshold
            detector.EAR_THRESHOLD = parameters['ear_threshold']
            
            # Process only first 20 folders
            all_scores = []
            total_images = 0
            processed_images = 0
            
            print("\nProcessing folders:")
            for folder_num in range(1, 21):  # Only process first 20 folders
                folder_name = f"{folder_num:04d}"
                folder_path = self.evaluator.dataset_path / folder_name
                
                if not folder_path.exists():
                    continue
                
                # Get list of images in the folder
                images = list(folder_path.glob("*.jpg"))
                total_images += len(images)
                
                # Process images with progress bar
                print(f"\nProcessing folder {folder_name}:")
                for img_path in tqdm(images, desc=f"Folder {folder_name}"):
                    try:
                        # Load and resize image
                        image = cv2.imread(str(img_path))
                        if image is None:
                            continue
                            
                        # Resize image while maintaining aspect ratio
                        resized_image = self.resize_image(image)
                        
                        # Process frame using the detector's process_frame method
                        _, metrics = detector.process_frame(resized_image)
                        focus_percentage = metrics.get('focus_percentage', 0.0)
                        
                        if focus_percentage >= 0:
                            all_scores.append(focus_percentage)
                        processed_images += 1
                        
                    except Exception as e:
                        print(f"Error processing image {img_path}: {str(e)}")
                        continue
                
                # Print folder statistics
                if all_scores:
                    folder_mean = np.mean(all_scores[-len(images):])
                    print(f"Folder {folder_name} mean score: {folder_mean:.2f}")
            
            # Calculate mean focus score
            mean_score = np.mean(all_scores) if all_scores else 0.0
            
            print(f"\nProcessed {processed_images}/{total_images} images")
            print(f"Overall mean focus score: {mean_score:.2f}")
            
            return mean_score
            
        except Exception as e:
            print(f"Error evaluating parameters: {e}")
            return 0.0
            
    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Perform crossover between two parents."""
        child = {}
        for param in self.parameter_ranges.keys():
            # Randomly choose which parent's value to use
            child[param] = parent1[param] if random.random() < 0.5 else parent2[param]
        return child
        
    def mutate(self, individual: Dict, mutation_rate: float = 0.1) -> Dict:
        """Mutate an individual's parameters."""
        mutated = individual.copy()
        for param, ranges in self.parameter_ranges.items():
            if random.random() < mutation_rate:
                # Add random noise to the parameter
                noise = random.uniform(-0.1, 0.1) * (ranges[1] - ranges[0])
                mutated[param] = max(ranges[0], min(ranges[1], mutated[param] + noise))
        return mutated
        
    def optimize(self, n_generations: int = 20, population_size: int = 50) -> Dict:
        """Run the optimization process."""
        print("\nStarting parameter optimization...")
        print(f"Processing {population_size} individuals for {n_generations} generations")
        print("Using first 20 folders of the dataset")
        print(f"Images will be resized to {self.target_width}x{self.target_height}")
        
        # Initialize population
        population = [self.create_individual() for _ in range(population_size)]
        
        for generation in range(n_generations):
            print(f"\nGeneration {generation + 1}/{n_generations}")
            generation_start_time = time.time()
            
            # Evaluate current population
            scores = []
            for i, individual in enumerate(population):
                print(f"\nEvaluating individual {i+1}/{population_size}")
                score = self.evaluate_fitness(individual)
                scores.append(score)
                print(f"Individual {i+1} score: {score:.2f}")
            
            # Update best parameters
            best_idx = np.argmax(scores)
            if scores[best_idx] > self.best_score:
                self.best_score = scores[best_idx]
                self.best_parameters = population[best_idx]
                print(f"\nNew best score: {self.best_score:.2f}")
                print("Best parameters:", self.best_parameters)
            
            # Store generation statistics
            self.generation_history.append({
                'generation': generation + 1,
                'mean_score': np.mean(scores),
                'best_score': np.max(scores),
                'worst_score': np.min(scores)
            })
            
            # Select best individuals
            best_indices = np.argsort(scores)[-population_size//2:]
            new_population = [population[i] for i in best_indices]
            
            # Create new individuals through crossover and mutation
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(new_population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
            
            # Calculate generation time
            generation_time = time.time() - generation_start_time
            print(f"\nGeneration {generation + 1} completed in {generation_time:.2f} seconds")
            print(f"Average time per individual: {generation_time/population_size:.2f} seconds")
        
        print("\nOptimization complete!")
        print(f"Best score achieved: {self.best_score:.2f}")
        print("Best parameters:", self.best_parameters)
        
        return self.best_parameters
        
    def plot_optimization_history(self):
        """Plot the optimization history."""
        import matplotlib.pyplot as plt
        
        generations = [h['generation'] for h in self.generation_history]
        mean_scores = [h['mean_score'] for h in self.generation_history]
        best_scores = [h['best_score'] for h in self.generation_history]
        worst_scores = [h['worst_score'] for h in self.generation_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, mean_scores, label='Mean Score')
        plt.plot(generations, best_scores, label='Best Score')
        plt.plot(generations, worst_scores, label='Worst Score')
        plt.xlabel('Generation')
        plt.ylabel('Focus Score')
        plt.title('Optimization Progress')
        plt.legend()
        plt.grid(True)
        plt.show() 