#!/usr/bin/env python3
"""
Test HNN integrity - verify it's using learned dynamics, not true dynamics
"""

import numpy as np
from models.hnn import HNN, HNNTrainer
from config import HNN_CONFIG

def test_hnn_integrity():
    """Test that HNN is using learned dynamics, not hardcoded physics"""
    
    # Load the trained HNN model
    print("🔍 Loading trained HNN model...")
    hnn_model = HNN(
        input_dim=2,
        hidden_layers=HNN_CONFIG['hidden_layers'],
        activation=HNN_CONFIG['activation'],
        normalize_inputs=HNN_CONFIG['normalize_inputs']
    )
    
    hnn_trainer = HNNTrainer(hnn_model)
    hnn_trainer.load_model('hnn_damped_oscillator')
    
    # Test integrity with different initial conditions
    hnn_trainer.test_prediction_integrity()
    
    # Additional test: Compare prediction vs true dynamics for validation trajectory
    print("🔍 Testing on validation trajectory...")
    
    # Standard initial conditions used in training
    q0, p0 = 1.0, 0.0
    t_span = (0, 15)  # Same as training data
    
    # HNN prediction
    t_hnn, q_hnn, p_hnn = hnn_trainer.integrate_trajectory(q0, p0, t_span, 100)
    
    # True analytical solution
    t_true = np.linspace(*t_span, 100)
    q_true = q0 * np.cos(t_true) + p0 * np.sin(t_true)
    p_true = -q0 * np.sin(t_true) + p0 * np.cos(t_true)
    
    # Compute errors
    q_rmse = np.sqrt(np.mean((q_hnn - q_true)**2))
    p_rmse = np.sqrt(np.mean((p_hnn - p_true)**2))
    
    print(f"Validation trajectory comparison:")
    print(f"  Position RMSE: {q_rmse:.6f}")
    print(f"  Momentum RMSE: {p_rmse:.6f}")
    
    # Check energy conservation
    H_hnn = 0.5 * (p_hnn**2 + q_hnn**2)  # Approximate Hamiltonian
    energy_variance = np.var(H_hnn)
    print(f"  Energy variance: {energy_variance:.8f}")
    
    # Verdict
    print("\n🔍 INTEGRITY ASSESSMENT:")
    if q_rmse < 1e-8 and p_rmse < 1e-8:
        print("⚠️ SUSPICIOUS: Errors suspiciously small - possible cheating with true dynamics!")
    elif q_rmse < 0.1 and p_rmse < 0.1 and energy_variance < 0.01:
        print("✅ LEGITIMATE: Small errors with good energy conservation - well-trained HNN!")
    else:
        print("⚠️ PROBLEMATIC: Large errors or poor energy conservation")
    
    return q_rmse, p_rmse, energy_variance

if __name__ == "__main__":
    test_hnn_integrity() 