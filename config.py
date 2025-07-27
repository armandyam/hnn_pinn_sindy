"""
Configuration file for AI Safety and Physics project.
Modify these parameters to customize the experiments.
"""

# Import numpy for system parameters
import numpy as np

# Data Generation Parameters
DATA_CONFIG = {
    't_span': (0, 15),  # Reduced from 20 to focus on core dynamics
    'n_points': 8000,   # Reduced from 10000 for less overfitting
    'noise_level': 0.02,  # Increased noise for robustness
    'subsample_factor': 3,  # Increased subsampling for more challenging learning
    'initial_conditions': [(1.0, 0.0), (0.5, 0.0), (1.0, 1.0), (0.3, 0.5), (1.5, 0.0)]  # More diverse conditions
}

# Neural Network Parameters
NN_CONFIG = {
    'hidden_layers': [64, 32],  # Simpler architecture to reduce overfitting
    'activation': 'tanh',
    'learning_rate': 1e-3,  # Increased learning rate for faster convergence
    'epochs': 8000,  # Reduced epochs to prevent overfitting
    'val_split': 0.2,
    'batch_size': 64,  # Increased batch size for more stable gradients
    'weight_decay': 1e-3,  # Increased regularization
    'optimizer': 'adamw',
    'scheduler': 'steplr',
    'scheduler_params': {
        'step_size': 1500,  # More frequent LR drops
        'gamma': 0.7  # More aggressive LR reduction
    }
}

# PINN Configuration  
PINN_CONFIG = {
    'hidden_layers': [128, 64, 32],  # Match HNN architecture
    'learning_rate': 1e-3,  # Match HNN learning rate
    'epochs': 8000,
    'batch_size': 64,
    'physics_weight': 10.0,  # Reduced from 100.0 for better data/physics balance
    'weight_decay': 1e-4,  # Match HNN regularization
    'activation': 'tanh',  # Tanh for smooth derivatives
    'optimizer': 'adamw',
    'scheduler': 'steplr',
    'scheduler_params': {
        'step_size': 1500,  # Match HNN scheduler
        'gamma': 0.8
    },
    'val_split': 0.2,
    'residual_points': 3000  # More collocation points for better physics coverage
}

# HNN Configuration
HNN_CONFIG = {
    'hidden_layers': [128, 64, 32],  # Deeper network for better Hamiltonian approximation
    'learning_rate': 1e-3,  # Higher learning rate for faster convergence
    'epochs': 8000,
    'batch_size': 64,
    'energy_weight': 10.0,  # Reduced weight to balance data and energy losses
    'weight_decay': 1e-4,  # Slightly higher regularization
    'activation': 'tanh',  # Tanh is good for Hamiltonian functions
    'optimizer': 'adamw',
    'scheduler': 'steplr',
    'scheduler_params': {
        'step_size': 1500,  # Decay learning rate every 1500 epochs
        'gamma': 0.8  # More aggressive decay
    },
    'val_split': 0.2,
    'normalize_inputs': True  # Enable normalization for better convergence
}

# Symbolic Regression Configuration
SYMBOLIC_CONFIG = {
    'library_type': 'polynomial',
    'max_degree': 1,  # Reduced from 2 for simpler, more stable equations
    'threshold': 0.01,  # Increased threshold for sparser equations
    'alpha': 0.01,  # Increased regularization
    'max_iter': 100,  # More iterations for better convergence
    'feature_names': ['x', 'dx_dt'],
    'normalize_columns': False  # Disable normalization for numerical stability
}

# Validation Parameters
VALIDATION_CONFIG = {
    't_span': (0, 50),  # Shorter validation period
    'n_points': 5000,
    'initial_conditions': [(1.0, 0.0)]  # Single condition for cleaner analysis
}

# System-specific Parameters
SYSTEMS = {
    'damped_oscillator': {
        'parameters': {
            'm': 1.0,
            'k': 1.0,
            'c': 0.0  # Undamped oscillator
        },
        'true_equation': 'x_ddot = -1.0 * x'  # Simple harmonic oscillator
    },
    'pendulum': {
        'parameters': {
            'g': 9.81,
            'L': 1.0,
            'c': 0.1
        },
        'true_equation': 'theta_ddot = -(g/L) * sin(theta) - c * theta_dot'
    }
}

# Experiment Configuration
EXPERIMENT_CONFIG = {
    'run_all_systems': False,
    'save_plots': True,
    'save_models': True,
    'verbose': True
}

# File Paths
PATHS = {
    'data': 'data/',
    'models': 'models/saved/',
    'results': 'results/',
    'plots': 'plots/'
} 