#!/usr/bin/env python3
"""
Test script for cascaded HNN
"""

import numpy as np
import matplotlib.pyplot as plt
from models.cascaded_hnn import CascadedHNN, CascadedHNNTrainer
from config import CASCADED_HNN_CONFIG

def test_cascaded_hnn():
    """Test the cascaded HNN implementation"""
    
    print("Testing Cascaded HNN...")
    
    # Create simple test data
    t = np.linspace(0, 10, 100)
    x = np.sin(t)  # Simple harmonic motion
    v = np.cos(t)  # Velocity
    
    # Create model
    model = CascadedHNN(
        trajectory_config=CASCADED_HNN_CONFIG['trajectory_net'],
        hnn_config=CASCADED_HNN_CONFIG['hnn_net']
    )
    
    # Create trainer
    trainer = CascadedHNNTrainer(
        model,
        trajectory_weight=1.0,
        hnn_weight=1.0,
        energy_weight=0.1,
        learning_rate=1e-3
    )
    
    # Train for a few epochs
    print("Training cascaded HNN...")
    history = trainer.train(t, x, v, epochs=100, val_split=0.2)
    
    # Test prediction
    print("Testing prediction...")
    t_test = np.linspace(0, 15, 150)
    x_pred = trainer.predict(t_test)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(t, x, 'b-', label='True', linewidth=2)
    plt.plot(t_test, x_pred, 'r--', label='Cascaded HNN', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Cascaded HNN Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['trajectory_loss'], label='Trajectory Loss')
    plt.plot(history['hnn_loss'], label='HNN Loss')
    plt.plot(history['energy_loss'], label='Energy Loss')
    plt.plot(history['total_loss'], label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/cascaded_hnn_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Cascaded HNN test completed!")

if __name__ == "__main__":
    test_cascaded_hnn() 