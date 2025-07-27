#!/usr/bin/env python3
"""
Retrain just the cascaded HNN with fixed parameters
"""

import pandas as pd
import numpy as np
from models.cascaded_hnn import CascadedHNN, CascadedHNNTrainer
from config import CASCADED_HNN_CONFIG

def retrain_cascaded_hnn():
    """Retrain the cascaded HNN with fixed parameters"""
    
    print("=== RETRAINING CASCADED HNN ===")
    
    # Load training data
    data = pd.read_csv('data/damped_oscillator.csv')
    print(f"Loaded {len(data)} data points")
    
    # Create new model
    cascaded_model = CascadedHNN(
        trajectory_config=CASCADED_HNN_CONFIG['trajectory_net'],
        hnn_config=CASCADED_HNN_CONFIG['hnn_net']
    )
    
    # Create trainer with FIXED parameters
    cascaded_trainer = CascadedHNNTrainer(
        cascaded_model,
        trajectory_weight=CASCADED_HNN_CONFIG['training']['trajectory_weight'],  # Now 1.0
        hnn_weight=CASCADED_HNN_CONFIG['training']['hnn_weight'],  # 1.0
        energy_weight=CASCADED_HNN_CONFIG['training']['energy_weight'],  # 0.1
        learning_rate=CASCADED_HNN_CONFIG['trajectory_net']['learning_rate'],
        optimizer_type=CASCADED_HNN_CONFIG['trajectory_net']['optimizer'],
        weight_decay=CASCADED_HNN_CONFIG['trajectory_net']['weight_decay'],
        scheduler_type=CASCADED_HNN_CONFIG['trajectory_net']['scheduler'],
        scheduler_params=CASCADED_HNN_CONFIG['trajectory_net']['scheduler_params']
    )
    
    print(f"Training with balanced weights: trajectory={CASCADED_HNN_CONFIG['training']['trajectory_weight']}, hnn={CASCADED_HNN_CONFIG['training']['hnn_weight']}, energy={CASCADED_HNN_CONFIG['training']['energy_weight']}")
    
    # Train
    history = cascaded_trainer.train(
        data['t'].values, data['x'].values, data['v'].values,
        epochs=CASCADED_HNN_CONFIG['training']['epochs'],
        val_split=CASCADED_HNN_CONFIG['training']['val_split'],
        patience=CASCADED_HNN_CONFIG['training']['patience']
    )
    
    # Save model
    cascaded_trainer.save_model('cascaded_hnn_damped_oscillator')
    cascaded_trainer.plot_training()
    
    print("✅ Cascaded HNN retrained successfully!")
    
    # Test prediction immediately
    print("\n=== TESTING PREDICTION ===")
    t_test = np.linspace(0, 10, 20)
    x_pred = cascaded_trainer.predict(t_test)
    
    print(f"Prediction shape: {x_pred.shape}")
    print(f"Prediction range: [{x_pred.min():.6f}, {x_pred.max():.6f}]")
    print(f"Sample values: {x_pred[:5]}")
    
    # Compare to true oscillator
    x_true = np.cos(t_test)
    error = np.mean(np.abs(x_pred - x_true))
    print(f"Mean absolute error vs cos(t): {error:.6f}")
    
    if error < 0.5:
        print("✅ Prediction looks reasonable!")
    else:
        print("⚠️ Prediction still looks wrong...")

if __name__ == "__main__":
    retrain_cascaded_hnn() 