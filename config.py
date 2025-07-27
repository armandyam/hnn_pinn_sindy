"""
Configuration file for AI Safety and Physics project.
Modify these parameters to customize the experiments.
"""

# Import numpy for system parameters
import numpy as np

# Data Generation Parameters
DATA_CONFIG = {
    'noise_level': 0.01,           # Gaussian noise standard deviation
    'subsample_factor': 5,          # Factor for data subsampling
    't_span': (0, 20),             # Increased time span for better conditioning
    'n_points': 10000,              # More points for better coverage
    'initial_conditions': [(1.0, 0.0), (0.5, 0.0), (1.0, 1.0)],  # Multiple ICs for generalization
}

# Neural Network Parameters
NN_CONFIG = {
    'hidden_layers': [128, 64, 32], # Deeper network with better capacity
    'learning_rate': 5e-4,          # Lower learning rate for stability
    'epochs': 15000,                # More training epochs
    'val_split': 0.2,               # Validation split ratio
    'batch_size': 32,               # Batch size for training
    'weight_decay': 1e-5,           # L2 regularization to prevent overfitting
    'activation': 'tanh',            # Activation function
    'optimizer': 'adamw',           # Optimizer type
    'scheduler': 'steplr',          # Learning rate scheduler
    'scheduler_params': {
        'step_size': 1000,
        'gamma': 0.9
    }
}

# PINN Parameters
PINN_CONFIG = {
    'physics_weight': 10.0,        # Increased weight for physics loss term
    'activation': 'tanh',          # Smooth activation for physics
    'hidden_layers': [128, 128],   # Shallow & wider for PINN
    'optimizer': 'adamw',          # Use AdamW for better optimization
    'scheduler': 'steplr',         # Learning rate scheduler
    'scheduler_params': {
        'step_size': 1000,
        'gamma': 0.9
    },
    'residual_points': 1000,       # Number of collocation points for physics loss
}

# HNN Parameters
HNN_CONFIG = {
    'energy_weight': 10.0,           # Increased weight for energy conservation
    'learning_rate': 5e-4,           # Lower learning rate for stability
    'epochs': 15000,                 # More training epochs
    'batch_size': 32,                # Batch size for training
    'weight_decay': 1e-5,            # L2 regularization
    'activation': 'tanh',            # Smooth activation for Hamiltonian
    'hidden_layers': [128, 128],     # Shallow & wider for HNN
    'optimizer': 'adamw',            # Use AdamW for better optimization
    'scheduler': 'steplr',           # Learning rate scheduler
    'scheduler_params': {
        'step_size': 1000,
        'gamma': 0.9
    },
    'normalize_inputs': True,        # Normalize q, p inputs to unit scale
}

# Symbolic Regression Parameters
SYMBOLIC_CONFIG = {
    'library_type': 'polynomial',   # 'polynomial' or 'fourier'
    'max_degree': 3,               # Lower degree for simpler equations
    'threshold': 0.001,            # Lower threshold for more terms
    'alpha': 0.001,                # Lower alpha for less regularization
    'max_iter': 20,                # More iterations for better convergence
}

# Validation Parameters
VALIDATION_CONFIG = {
    't_span': (0, 100),            # Time span for long-term validation
    'n_points': 10000,              # Number of points for integration
}

# System-specific Parameters
SYSTEMS = {
    'damped_oscillator': {
        'initial_conditions': (1.0, 0.0),
        'parameters': {'m': 1.0, 'k': 1.0, 'c': 0.0},
        'true_equation': 'ddot(x) + x = 0'  # Updated for undamped case
    },
    'pendulum': {
        'initial_conditions': (np.pi/4, 0.0),
        'parameters': {'g': 9.81, 'L': 1.0, 'c': 0.1},
        'true_equation': 'ddot(theta) + 0.1*dot(theta) + 9.81*sin(theta) = 0'
    }
}

# Experiment Configuration
EXPERIMENT_CONFIG = {
    'systems': ['damped_oscillator', 'pendulum'],  # Systems to study
    'models': ['baseline_nn', 'pinn', 'hnn'],     # Models to compare
    'save_plots': True,                            # Whether to save plots
    'save_results': True,                          # Whether to save results
}

# File Paths
PATHS = {
    'data_dir': 'data',
    'models_dir': 'models/saved',
    'results_dir': 'results',
    'plots_dir': 'plots'
} 