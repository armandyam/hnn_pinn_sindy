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

# PINN Parameters
PINN_CONFIG = {
    'hidden_layers': [64, 64],  # Simpler than before
    'activation': 'tanh',
    'physics_weight': 50.0,  # Much higher physics weight
    'optimizer': 'adamw',
    'scheduler': 'steplr',
    'scheduler_params': {
        'step_size': 1500,
        'gamma': 0.7
    },
    'residual_points': 2000  # More physics constraints
}

# HNN Parameters
HNN_CONFIG = {
    'hidden_layers': [64, 32],  # Simpler architecture
    'activation': 'tanh',
    'learning_rate': 5e-4,  # Slightly lower LR for stability
    'epochs': 8000,
    'batch_size': 64,
    'energy_weight': 100.0,  # Much higher energy conservation weight
    'weight_decay': 1e-3,
    'optimizer': 'adamw',
    'scheduler': 'steplr',
    'scheduler_params': {
        'step_size': 1500,
        'gamma': 0.7
    },
    'normalize_inputs': False  # Disable normalization for simplicity
}

# Symbolic Regression Parameters
SYMBOLIC_CONFIG = {
    'max_degree': 2,  # Reduced complexity - focus on linear/quadratic terms
    'threshold': 0.001,  # Lower threshold to allow more terms
    'alpha': 0.001,  # Lower regularization for more terms
    'max_iter': 50,  # More iterations for better convergence
    'feature_names': ['x', 'dx_dt']
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