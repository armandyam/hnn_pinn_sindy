#!/usr/bin/env python3
"""
Test script for sequential cascaded HNN
"""

import pandas as pd
import numpy as np
from models.sequential_cascaded_hnn import SequentialCascadedHNN, SequentialCascadedHNNTrainer
from config import CASCADED_HNN_CONFIG

def test_sequential_cascaded_hnn():
    """Test the sequential cascaded HNN implementation"""
    
    print("=== TESTING SEQUENTIAL CASCADED HNN ===")
    
    # Load training data
    data = pd.read_csv('data/damped_oscillator.csv')
    print(f"Loaded {len(data)} data points")
    
    # Create sequential model with optimized configs
    seq_config = {
        'trajectory_net': {
            'hidden_layers': [128, 64, 32],  # Moderate size
            'activation': 'tanh',
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'optimizer': 'adamw',
            'scheduler': 'steplr',
            'scheduler_params': {'step_size': 1000, 'gamma': 0.8}
        },
        'hnn_net': {
            'hidden_layers': [64, 64],  # Simple HNN
            'activation': 'tanh',
            'normalize_inputs': False,
            'learning_rate': 5e-4,
            'weight_decay': 1e-4,
            'optimizer': 'adamw',
            'scheduler': 'steplr',
            'scheduler_params': {'step_size': 1000, 'gamma': 0.8}
        }
    }
    
    sequential_model = SequentialCascadedHNN(
        trajectory_config=seq_config['trajectory_net'],
        hnn_config=seq_config['hnn_net']
    )
    
    # Create trainer
    sequential_trainer = SequentialCascadedHNNTrainer(
        sequential_model,
        trajectory_config=seq_config['trajectory_net'],
        hnn_config=seq_config['hnn_net'],
        energy_weight=0.1
    )
    
    print("🚀 Starting sequential training...")
    
    # Train sequentially
    history = sequential_trainer.train(
        data['t'].values, data['x'].values, data['v'].values,
        stage1_epochs=3000,  # Shorter for testing
        stage2_epochs=3000,
        val_split=0.2,
        patience=1000
    )
    
    # Save model
    sequential_trainer.save_model('sequential_cascaded_hnn_damped_oscillator')
    sequential_trainer.plot_training()
    
    print("\n=== TESTING PREDICTION ===")
    t_test = np.linspace(0, 10, 20)
    x_pred = sequential_trainer.predict(t_test)
    
    print(f"Prediction shape: {x_pred.shape}")
    print(f"Prediction range: [{x_pred.min():.6f}, {x_pred.max():.6f}]")
    print(f"Sample values: {x_pred[:5]}")
    
    # Compare to true oscillator
    x_true = np.cos(t_test)
    error = np.mean(np.abs(x_pred - x_true))
    print(f"Mean absolute error vs cos(t): {error:.6f}")
    
    if error < 0.3:
        print("✅ Sequential prediction looks much better!")
    elif error < 0.8:
        print("⚠️ Sequential prediction is okay...")
    else:
        print("❌ Sequential prediction still needs work...")
    
    # Final comparison
    print(f"\n📊 COMPARISON:")
    print(f"Stage 1 final train loss: {history['stage1']['train_loss'][-1]:.6f}")
    print(f"Stage 1 final val loss: {history['stage1']['val_loss'][-1]:.6f}")
    print(f"Stage 2 final dynamics loss: {history['stage2']['dynamics_loss'][-1]:.6f}")
    print(f"Stage 2 final energy loss: {history['stage2']['energy_loss'][-1]:.6f}")

if __name__ == "__main__":
    test_sequential_cascaded_hnn() 