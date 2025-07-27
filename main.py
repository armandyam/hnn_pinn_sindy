#!/usr/bin/env python3
"""
Main execution script for AI Safety and Physics project.
This script runs the complete pipeline:
1. Data Generation
2. Neural Network Training (Baseline, PINN, HNN)
3. Symbolic Regression
4. Long-term Validation and Comparison
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

# Add project modules to path
sys.path.append('.')

from data_generation import PhysicsDataGenerator
from models.nn_baseline import BaselineNN, BaselineTrainer
from models.pinn import PINN, PINNTrainer
from models.hnn import HNN, HNNTrainer, prepare_hnn_data
from regression.symbolic_regression import SymbolicRegression
from validation.compare_long_term import LongTermValidator
from config import NN_CONFIG, HNN_CONFIG

class PhysicsAISafetyPipeline:
    """Complete pipeline for AI Safety and Physics project"""
    
    def __init__(self):
        self.data_generator = PhysicsDataGenerator()
        self.symbolic_regression = SymbolicRegression()
        self.validator = LongTermValidator()
        self.results = {}
    
    def step1_generate_data(self, systems: List[str] = ['damped_oscillator', 'pendulum']):
        """Step 1: Generate synthetic physics data"""
        print("=" * 50)
        print("STEP 1: DATA GENERATION")
        print("=" * 50)
        
        for system in systems:
            print(f"\nGenerating data for {system}...")
            
            if system == 'damped_oscillator':
                # Use config parameters for undamped oscillator (c=0.0)
                from config import SYSTEMS
                params = SYSTEMS['damped_oscillator']['parameters']
                data = self.data_generator.generate_damped_oscillator_data(
                    m=params['m'], k=params['k'], c=params['c']
                )
            elif system == 'pendulum':
                data = self.data_generator.generate_pendulum_data()
            else:
                raise ValueError(f"Unknown system: {system}")
            
            self.data_generator.save_data(data, system)
            self.data_generator.plot_data(data)
            
            print(f"Data generation completed for {system}")
    
    def step2_train_models(self, systems: List[str] = ['damped_oscillator', 'pendulum']):
        """Step 2: Train neural network models"""
        print("\n" + "=" * 50)
        print("STEP 2: NEURAL NETWORK TRAINING")
        print("=" * 50)
        
        for system in systems:
            print(f"\nTraining models for {system}...")
            
            # Load data
            data = pd.read_csv(f'data/{system}.csv')
            
            # Load config for neural network parameters
            from config import NN_CONFIG
            
            # Train Baseline NN
            print("Training Baseline Neural Network...")
            baseline_model = BaselineNN(hidden_layers=NN_CONFIG['hidden_layers'])
            baseline_trainer = BaselineTrainer(baseline_model, lr=NN_CONFIG['learning_rate'])
            baseline_trainer.train(data['t'].values, data['x'].values, 
                                 epochs=NN_CONFIG['epochs'], val_split=NN_CONFIG['val_split'])
            baseline_trainer.save_model(f'baseline_nn_{system}')
            baseline_trainer.plot_training()
            
            # Train PINN
            print("Training Physics-Informed Neural Network...")
            pinn_model = PINN(hidden_layers=NN_CONFIG['hidden_layers'])
            pinn_trainer = PINNTrainer(pinn_model, lr=NN_CONFIG['learning_rate'], physics_weight=1.0)
            pinn_trainer.train(data['t'].values, data['x'].values, 
                             epochs=NN_CONFIG['epochs'], val_split=NN_CONFIG['val_split'], system=system)
            pinn_trainer.save_model(f'pinn_{system}')
            pinn_trainer.plot_training()
            
            # Train HNN
            print("Training Hamiltonian Neural Network...")
            if system == 'damped_oscillator':
                q, p, dq_dt, dp_dt = prepare_hnn_data(data['t'].values, 
                                                      data['x'].values, data['v'].values)
            else:  # pendulum
                q, p, dq_dt, dp_dt = prepare_hnn_data(data['t'].values, 
                                                      data['theta'].values, data['omega'].values)
            
            hnn_model = HNN(hidden_layers=NN_CONFIG['hidden_layers'])
            hnn_trainer = HNNTrainer(hnn_model, lr=HNN_CONFIG['learning_rate'], energy_weight=HNN_CONFIG['energy_weight'])
            hnn_trainer.train(q, p, dq_dt, dp_dt, epochs=HNN_CONFIG['epochs'], val_split=NN_CONFIG['val_split'])
            hnn_trainer.save_model(f'hnn_{system}')
            hnn_trainer.plot_training()
            
            print(f"Model training completed for {system}")
    
    def step3_symbolic_regression(self, systems: List[str] = ['damped_oscillator', 'pendulum']):
        """Step 3: Extract symbolic equations"""
        print("\n" + "=" * 50)
        print("STEP 3: SYMBOLIC REGRESSION")
        print("=" * 50)
        
        for system in systems:
            print(f"\nExtracting equations for {system}...")
            
            # Load data
            data = pd.read_csv(f'data/{system}.csv')
            
            # Load config for neural network parameters
            from config import NN_CONFIG
            
            # Load trained models
            baseline_model = BaselineNN(hidden_layers=NN_CONFIG['hidden_layers'])
            baseline_trainer = BaselineTrainer(baseline_model)
            baseline_trainer.load_model(f'baseline_nn_{system}')
            
            pinn_model = PINN(hidden_layers=NN_CONFIG['hidden_layers'])
            pinn_trainer = PINNTrainer(pinn_model)
            pinn_trainer.load_model(f'pinn_{system}')
            
            hnn_model = HNN(hidden_layers=NN_CONFIG['hidden_layers'])
            hnn_trainer = HNNTrainer(hnn_model)
            hnn_trainer.load_model(f'hnn_{system}')
            
            # Extract equations from each model
            models = {
                'baseline_nn': baseline_trainer,
                'pinn': pinn_trainer,
                'hnn': hnn_trainer
            }
            
            equations = {}
            for name, trainer in models.items():
                print(f"Extracting equations from {name}...")
                
                # Get derivatives from neural network
                x, dx_dt, d2x_dt2 = trainer.get_derivatives(data['t'].values, order=2)
                
                # Extract symbolic equation
                eq_data = self.symbolic_regression.extract_equations_from_nn(
                    data['t'].values, x, dx_dt, d2x_dt2, f'{name}_{system}')
                equations[name] = eq_data
            
            # Save results
            self.symbolic_regression.save_results(equations, f'symbolic_regression_{system}')
            
            print(f"Symbolic regression completed for {system}")
    
    def step4_validation(self, systems: List[str] = ['damped_oscillator', 'pendulum']):
        """Step 4: Long-term validation and comparison"""
        print("\n" + "=" * 50)
        print("STEP 4: LONG-TERM VALIDATION")
        print("=" * 50)
        
        for system in systems:
            print(f"\nValidating long-term dynamics for {system}...")
            
            # Set initial conditions based on system
            if system == 'damped_oscillator':
                initial_conditions = (1.0, 0.0)
            else:  # pendulum
                initial_conditions = (np.pi/4, 0.0)
            
            # Run comparison
            results = self.validator.compare_long_term_dynamics(
                system, t_span=(0, 100), initial_conditions=initial_conditions)
            
            # Plot results
            self.validator.plot_long_term_comparison(system)
            
            # Print summary
            self.validator.print_comparison_summary(system)
            
            print(f"Validation completed for {system}")
    
    def run_complete_pipeline(self, systems: List[str] = ['damped_oscillator', 'pendulum']):
        """Run the complete pipeline"""
        print("AI SAFETY AND PHYSICS PROJECT")
        print("=" * 50)
        print("This project compares different neural network approaches for")
        print("learning physics and extracting interpretable equations.")
        print("=" * 50)
        
        # Step 1: Generate data
        self.step1_generate_data(systems)
        
        # Step 2: Train models
        self.step2_train_models(systems)
        
        # Step 3: Symbolic regression
        self.step3_symbolic_regression(systems)
        
        # Step 4: Validation
        self.step4_validation(systems)
        
        print("\n" + "=" * 50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nResults saved in:")
        print("- data/: Generated datasets")
        print("- models/saved/: Trained neural networks")
        print("- results/: Symbolic regression results")
        print("- plots/: Visualization plots")
    
    def run_quick_test(self):
        """Run a quick test with just the damped oscillator"""
        print("Running quick test with damped oscillator...")
        self.run_complete_pipeline(['damped_oscillator'])

def main():
    """Main execution function"""
    
    # Create pipeline
    pipeline = PhysicsAISafetyPipeline()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            pipeline.run_quick_test()
        elif sys.argv[1] == 'full':
            pipeline.run_complete_pipeline()
        else:
            print("Usage: python main.py [test|full]")
            print("  test: Run quick test with damped oscillator only")
            print("  full: Run complete pipeline with all systems")
    else:
        # Default: run quick test
        print("No arguments provided. Running quick test...")
        pipeline.run_quick_test()

if __name__ == "__main__":
    main() 