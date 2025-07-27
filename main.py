#!/usr/bin/env python3
"""
AI Safety and Physics Project - Main Pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import random
from typing import List, Dict, Any

# Set random seeds for reproducibility - DO THIS FIRST!
def set_random_seeds(seed: int = 42):
    """Set all random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Initialize reproducibility BEFORE any other imports
set_random_seeds(42)

from data_generation import PhysicsDataGenerator
from models.nn_baseline import BaselineNN, BaselineTrainer
from models.pinn import PINN, PINNTrainer
from models.hnn import HNN, HNNTrainer
from models.cascaded_hnn import CascadedHNN, CascadedHNNTrainer
from models.sequential_cascaded_hnn import SequentialCascadedHNN, SequentialCascadedHNNTrainer
from regression.symbolic_regression import SymbolicRegression
from validation.compare_long_term import LongTermValidator
from utils.data_utils import prepare_hnn_data

# Import all config variables
from config import (
    NN_CONFIG, PINN_CONFIG, HNN_CONFIG, CASCADED_HNN_CONFIG, SEQUENTIAL_CASCADED_HNN_CONFIG,
    SYMBOLIC_CONFIG, DATA_CONFIG, VALIDATION_CONFIG, SYSTEMS, 
    EXPERIMENT_CONFIG, PATHS
)

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
            baseline_model = BaselineNN(
                hidden_layers=NN_CONFIG['hidden_layers'],
                activation=NN_CONFIG['activation']
            )
            baseline_trainer = BaselineTrainer(
                baseline_model, 
                lr=NN_CONFIG['learning_rate'],
                optimizer_type=NN_CONFIG['optimizer'],
                weight_decay=NN_CONFIG['weight_decay'],
                scheduler_type=NN_CONFIG['scheduler'],
                scheduler_params=NN_CONFIG['scheduler_params']
            )
            baseline_trainer.train(data['t'].values, data['x'].values, 
                                 epochs=NN_CONFIG['epochs'], val_split=NN_CONFIG['val_split'])
            baseline_trainer.save_model(f'baseline_nn_{system}')
            baseline_trainer.plot_training()
            
            # Train PINN
            print("Training Physics-Informed Neural Network...")
            pinn_model = PINN(
                hidden_layers=PINN_CONFIG['hidden_layers'],
                activation=PINN_CONFIG['activation']
            )
            pinn_trainer = PINNTrainer(
                pinn_model, 
                lr=NN_CONFIG['learning_rate'], 
                physics_weight=PINN_CONFIG['physics_weight'],
                optimizer_type=PINN_CONFIG['optimizer'],
                weight_decay=NN_CONFIG['weight_decay'],
                scheduler_type=PINN_CONFIG['scheduler'],
                scheduler_params=PINN_CONFIG['scheduler_params']
            )
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
            
            hnn_model = HNN(
                hidden_layers=HNN_CONFIG['hidden_layers'],
                activation=HNN_CONFIG['activation'],
                normalize_inputs=HNN_CONFIG['normalize_inputs']
            )
            # Set normalization if enabled
            if HNN_CONFIG['normalize_inputs']:
                hnn_model.set_normalization(q, p)
            hnn_trainer = HNNTrainer(
                hnn_model, 
                lr=HNN_CONFIG['learning_rate'], 
                energy_weight=HNN_CONFIG['energy_weight'],
                optimizer_type=HNN_CONFIG['optimizer'],
                weight_decay=HNN_CONFIG['weight_decay'],
                scheduler_type=HNN_CONFIG['scheduler'],
                scheduler_params=HNN_CONFIG['scheduler_params']
            )
            hnn_trainer.train(q, p, dq_dt, dp_dt, epochs=HNN_CONFIG['epochs'], val_split=NN_CONFIG['val_split'])
            hnn_trainer.save_model(f'hnn_{system}')
            hnn_trainer.plot_training()
            
            # Train Cascaded HNN
            print("Training Cascaded HNN...")
            cascaded_model = CascadedHNN(
                trajectory_config=CASCADED_HNN_CONFIG['trajectory_net'],
                hnn_config=CASCADED_HNN_CONFIG['hnn_net']
            )
            cascaded_trainer = CascadedHNNTrainer(
                cascaded_model,
                trajectory_weight=CASCADED_HNN_CONFIG['training']['trajectory_weight'],
                hnn_weight=CASCADED_HNN_CONFIG['training']['hnn_weight'],
                energy_weight=CASCADED_HNN_CONFIG['training']['energy_weight'],
                learning_rate=CASCADED_HNN_CONFIG['trajectory_net']['learning_rate'],
                optimizer_type=CASCADED_HNN_CONFIG['trajectory_net']['optimizer'],
                weight_decay=CASCADED_HNN_CONFIG['trajectory_net']['weight_decay'],
                scheduler_type=CASCADED_HNN_CONFIG['trajectory_net']['scheduler'],
                scheduler_params=CASCADED_HNN_CONFIG['trajectory_net']['scheduler_params']
            )
            cascaded_trainer.train(
                data['t'].values, data['x'].values, data['v'].values,
                epochs=CASCADED_HNN_CONFIG['training']['epochs'],
                val_split=CASCADED_HNN_CONFIG['training']['val_split'],
                patience=CASCADED_HNN_CONFIG['training']['patience']
            )
            cascaded_trainer.save_model(f'cascaded_hnn_{system}')
            cascaded_trainer.plot_training()
            
            # Train Sequential Cascaded HNN
            print("Training Sequential Cascaded HNN...")
            sequential_cascaded_model = SequentialCascadedHNN(
                trajectory_config=SEQUENTIAL_CASCADED_HNN_CONFIG['trajectory_net'],
                hnn_config=SEQUENTIAL_CASCADED_HNN_CONFIG['hnn_net']
            )
            sequential_cascaded_trainer = SequentialCascadedHNNTrainer(
                sequential_cascaded_model,
                trajectory_config=SEQUENTIAL_CASCADED_HNN_CONFIG['trajectory_net'],
                hnn_config=SEQUENTIAL_CASCADED_HNN_CONFIG['hnn_net'],
                energy_weight=SEQUENTIAL_CASCADED_HNN_CONFIG['training']['energy_weight']
            )
            sequential_cascaded_trainer.train(
                data['t'].values, data['x'].values, data['v'].values,
                stage1_epochs=SEQUENTIAL_CASCADED_HNN_CONFIG['training']['stage1_epochs'],
                stage2_epochs=SEQUENTIAL_CASCADED_HNN_CONFIG['training']['stage2_epochs'],
                val_split=SEQUENTIAL_CASCADED_HNN_CONFIG['training']['val_split'],
                patience=SEQUENTIAL_CASCADED_HNN_CONFIG['training']['patience']
            )
            sequential_cascaded_trainer.save_model(f'sequential_cascaded_hnn_{system}')
            sequential_cascaded_trainer.plot_training()
            
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
            from config import NN_CONFIG, SEQUENTIAL_CASCADED_HNN_CONFIG
            
            # Load trained models
            baseline_model = BaselineNN(
                hidden_layers=NN_CONFIG['hidden_layers'],
                activation=NN_CONFIG['activation']
            )
            baseline_trainer = BaselineTrainer(baseline_model)
            baseline_trainer.load_model(f'baseline_nn_{system}')
            
            pinn_model = PINN(
                hidden_layers=PINN_CONFIG['hidden_layers'],
                activation=PINN_CONFIG['activation']
            )
            pinn_trainer = PINNTrainer(pinn_model)
            pinn_trainer.load_model(f'pinn_{system}')
            
            hnn_model = HNN(
                hidden_layers=HNN_CONFIG['hidden_layers'],
                activation=HNN_CONFIG['activation'],
                normalize_inputs=HNN_CONFIG['normalize_inputs']
            )
            # Set normalization if enabled
            if HNN_CONFIG['normalize_inputs']:
                # Reload the normalization stats from training data
                if system == 'damped_oscillator':
                    q, p, _, _ = prepare_hnn_data(data['t'].values, data['x'].values, data['v'].values)
                else:
                    q, p, _, _ = prepare_hnn_data(data['t'].values, data['theta'].values, data['omega'].values)
                hnn_model.set_normalization(q, p)
            hnn_trainer = HNNTrainer(hnn_model)
            hnn_trainer.load_model(f'hnn_{system}')
            
            # Load cascaded HNN
            cascaded_model = CascadedHNN(
                trajectory_config=CASCADED_HNN_CONFIG['trajectory_net'],
                hnn_config=CASCADED_HNN_CONFIG['hnn_net']
            )
            cascaded_trainer = CascadedHNNTrainer(cascaded_model)
            cascaded_trainer.load_model(f'cascaded_hnn_{system}')
            
            # Load Sequential Cascaded HNN
            sequential_cascaded_model = SequentialCascadedHNN(
                trajectory_config=SEQUENTIAL_CASCADED_HNN_CONFIG['trajectory_net'],
                hnn_config=SEQUENTIAL_CASCADED_HNN_CONFIG['hnn_net']
            )
            sequential_cascaded_trainer = SequentialCascadedHNNTrainer(
                sequential_cascaded_model,
                trajectory_config=SEQUENTIAL_CASCADED_HNN_CONFIG['trajectory_net'],
                hnn_config=SEQUENTIAL_CASCADED_HNN_CONFIG['hnn_net'],
                energy_weight=SEQUENTIAL_CASCADED_HNN_CONFIG['training']['energy_weight']
            )
            sequential_cascaded_trainer.load_model(f'sequential_cascaded_hnn_{system}')
            
            # Extract equations from each model
            models = {
                'baseline_nn': baseline_trainer,
                'pinn': pinn_trainer,
                'hnn': hnn_trainer,
                'cascaded_hnn': cascaded_trainer,
                'sequential_cascaded_hnn': sequential_cascaded_trainer
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
        """Step 4: Long-term validation and comparison - FIXED to use VALIDATION_CONFIG"""
        print("\n" + "=" * 50)
        print("STEP 4: LONG-TERM VALIDATION")
        print("=" * 50)
        
        for system in systems:
            print(f"\nValidating long-term dynamics for {system}...")
            
            # Use VALIDATION_CONFIG for consistency
            if system == 'damped_oscillator':
                # Use config default initial conditions
                results = self.validator.compare_long_term_dynamics(system)
            else:  # pendulum
                # Override initial conditions for pendulum while using config t_span
                results = self.validator.compare_long_term_dynamics(
                    system, initial_conditions=(np.pi/4, 0.0))
            
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