#!/usr/bin/env python3
"""
Debug script to check what cascaded HNN is actually predicting
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from models.cascaded_hnn import CascadedHNN, CascadedHNNTrainer
from config import CASCADED_HNN_CONFIG

def debug_cascaded_hnn():
    """Debug the cascaded HNN prediction"""
    
    print("=== DEBUGGING CASCADED HNN PREDICTION ===")
    
    # Load the trained model
    cascaded_model = CascadedHNN(
        trajectory_config=CASCADED_HNN_CONFIG['trajectory_net'],
        hnn_config=CASCADED_HNN_CONFIG['hnn_net']
    )
    cascaded_trainer = CascadedHNNTrainer(cascaded_model)
    
    try:
        cascaded_trainer.load_model('cascaded_hnn_damped_oscillator')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test different prediction methods
    t_test = np.linspace(0, 10, 20)
    t_tensor = torch.tensor(t_test, dtype=torch.float32, requires_grad=True)
    
    print(f"\nTesting on {len(t_test)} time points from 0 to 10...")
    
    # Method 1: Current predict method (only trajectory_net)
    print("\n1. Current predict method:")
    try:
        x_pred_current = cascaded_trainer.predict(t_test)
        print(f"   Shape: {x_pred_current.shape}")
        print(f"   Range: [{x_pred_current.min():.6f}, {x_pred_current.max():.6f}]")
        print(f"   Sample values: {x_pred_current[:5]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        x_pred_current = None
    
    # Method 2: Full forward pass (FIXED)
    print("\n2. Full forward pass method (FIXED):")
    try:
        cascaded_model.eval()
        # DON'T use torch.no_grad() since we need gradients for autograd
        x_pred_full, v_pred_full, H_pred_full = cascaded_model(t_tensor)
        x_pred_full_np = x_pred_full.detach().numpy()
        print(f"   Shape: {x_pred_full_np.shape}")
        print(f"   Range: [{x_pred_full_np.min():.6f}, {x_pred_full_np.max():.6f}]")
        print(f"   Sample values: {x_pred_full_np[:5]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        x_pred_full_np = None
    
    # Method 3: Just trajectory net (FIXED)
    print("\n3. Just trajectory_net method (FIXED):")
    try:
        cascaded_model.eval()
        with torch.no_grad():
            t_tensor_no_grad = torch.tensor(t_test, dtype=torch.float32)  # No requires_grad
            x_pred_traj = cascaded_model.predict_trajectory(t_tensor_no_grad)
            x_pred_traj_np = x_pred_traj.detach().numpy()
        print(f"   Shape: {x_pred_traj_np.shape}")
        print(f"   Range: [{x_pred_traj_np.min():.6f}, {x_pred_traj_np.max():.6f}]")
        print(f"   Sample values: {x_pred_traj_np[:5]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        x_pred_traj_np = None
    
    # Compare predictions
    print("\n=== COMPARISON ===")
    if x_pred_current is not None and x_pred_full_np is not None:
        print(f"Current vs Full: Max diff = {np.max(np.abs(x_pred_current - x_pred_full_np)):.10f}")
    if x_pred_current is not None and x_pred_traj_np is not None:
        print(f"Current vs Traj: Max diff = {np.max(np.abs(x_pred_current - x_pred_traj_np)):.10f}")
    if x_pred_full_np is not None and x_pred_traj_np is not None:
        print(f"Full vs Traj: Max diff = {np.max(np.abs(x_pred_full_np - x_pred_traj_np)):.10f}")
    
    # True oscillator for comparison
    t_true = np.linspace(0, 10, 100)
    x_true = np.cos(t_true)  # Simple harmonic oscillator
    
    print("\n=== TRUE OSCILLATOR ===")
    print(f"True range: [{x_true.min():.6f}, {x_true.max():.6f}]")
    print(f"True samples: {x_true[:5]}")
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(t_true, x_true, 'k-', label='True', linewidth=2)
    if x_pred_current is not None:
        plt.plot(t_test, x_pred_current, 'r--', label='Current predict()', linewidth=2, marker='o')
    if x_pred_full_np is not None:
        plt.plot(t_test, x_pred_full_np, 'b:', label='Full forward', linewidth=2, marker='s')
    if x_pred_traj_np is not None:
        plt.plot(t_test, x_pred_traj_np, 'g-.', label='Trajectory only', linewidth=2, marker='^')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Cascaded HNN Prediction Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    if x_pred_current is not None:
        errors_current = np.abs(x_pred_current - np.cos(t_test))
        plt.plot(t_test, errors_current, 'r--', label='Current error', linewidth=2, marker='o')
    if x_pred_full_np is not None:
        errors_full = np.abs(x_pred_full_np - np.cos(t_test))
        plt.plot(t_test, errors_full, 'b:', label='Full error', linewidth=2, marker='s')
    if x_pred_traj_np is not None:
        errors_traj = np.abs(x_pred_traj_np - np.cos(t_test))
        plt.plot(t_test, errors_traj, 'g-.', label='Traj error', linewidth=2, marker='^')
    plt.xlabel('Time')
    plt.ylabel('Absolute Error')
    plt.title('Prediction Errors (Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    if x_pred_current is not None:
        plt.semilogy(t_test, errors_current + 1e-10, 'r--', label='Current error', linewidth=2, marker='o')
    if x_pred_full_np is not None:
        plt.semilogy(t_test, errors_full + 1e-10, 'b:', label='Full error', linewidth=2, marker='s')
    if x_pred_traj_np is not None:
        plt.semilogy(t_test, errors_traj + 1e-10, 'g-.', label='Traj error', linewidth=2, marker='^')
    plt.xlabel('Time')
    plt.ylabel('Absolute Error (Log Scale)')
    plt.title('Prediction Errors (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/debug_cascaded_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Debug plots saved to plots/debug_cascaded_predictions.png")

if __name__ == "__main__":
    debug_cascaded_hnn() 