#!/usr/bin/env python3

import argparse
from optimizer import FocusOptimizer
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Optimize focus detection parameters using genetic algorithm')
    parser.add_argument('--dataset_path', type=str, default="/Users/ani/Downloads/columbia_dataset",
                      help='Path to the dataset directory')
    parser.add_argument('--generations', type=int, default=20,
                      help='Number of generations to run')
    parser.add_argument('--population_size', type=int, default=50,
                      help='Size of the population in each generation')
    parser.add_argument('--output_file', type=str, default='optimized_parameters.json',
                      help='File to save the optimized parameters')
    args = parser.parse_args()

    try:
        # Create optimizer
        optimizer = FocusOptimizer(args.dataset_path)
        
        # Run optimization
        best_parameters = optimizer.optimize(
            n_generations=args.generations,
            population_size=args.population_size
        )
        
        # Plot optimization history
        optimizer.plot_optimization_history()
        
        # Save results
        results = {
            'best_parameters': best_parameters,
            'best_score': optimizer.best_score,
            'generation_history': optimizer.generation_history
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"\nResults saved to {args.output_file}")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        raise

if __name__ == "__main__":
    main() 