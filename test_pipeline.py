#!/usr/bin/env python3
"""
Test script to verify that all components of the AI Safety and Physics pipeline work correctly.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project modules to path
sys.path.append('.')

def test_data_generation():
    """Test data generation"""
    print("Testing data generation...")
    
    from data_generation import PhysicsDataGenerator
    
    generator = PhysicsDataGenerator(noise_level=0.01, subsample_factor=5)
    
    # Test damped oscillator
    oscillator_data = generator.generate_damped_oscillator_data()
    assert 't' in oscillator_data
    assert 'x' in oscillator_data
    assert 'v' in oscillator_data
    assert len(oscillator_data['t']) > 0
    
    # Test pendulum
    pendulum_data = generator.generate_pendulum_data()
    assert 't' in pendulum_data
    assert 'theta' in pendulum_data
    assert 'omega' in pendulum_data
    assert len(pendulum_data['t']) > 0
    
    print("✓ Data generation test passed")

def test_neural_networks():
    """Test neural network models"""
    print("Testing neural networks...")
    
    from models.nn_baseline import BaselineNN, BaselineTrainer
    from models.pinn import PINN, PINNTrainer
    from models.hnn import HNN, HNNTrainer, prepare_hnn_data
    
    # Generate test data
    t = np.linspace(0, 10, 100)
    x = np.sin(t) * np.exp(-0.1*t)
    v = np.cos(t) * np.exp(-0.1*t) - 0.1 * np.sin(t) * np.exp(-0.1*t)
    
    # Test Baseline NN
    baseline_model = BaselineNN()
    baseline_trainer = BaselineTrainer(baseline_model)
    baseline_trainer.train(t, x, epochs=10)  # Short training for test
    pred = baseline_trainer.predict(t)
    assert len(pred) == len(t)
    
    # Test PINN
    pinn_model = PINN()
    pinn_trainer = PINNTrainer(pinn_model)
    pinn_trainer.train(t, x, epochs=10, system='damped_oscillator')
    pred = pinn_trainer.predict(t)
    assert len(pred) == len(t)
    
    # Test HNN
    q, p, dq_dt, dp_dt = prepare_hnn_data(t, x, v)
    hnn_model = HNN()
    hnn_trainer = HNNTrainer(hnn_model)
    hnn_trainer.train(q, p, dq_dt, dp_dt, epochs=10)
    pred_dq, pred_dp = hnn_trainer.predict_dynamics(q, p)
    assert len(pred_dq) == len(q)
    
    print("✓ Neural network test passed")

def test_symbolic_regression():
    """Test symbolic regression"""
    print("Testing symbolic regression...")
    
    from regression.symbolic_regression import SymbolicRegression
    
    # Generate test data with strictly increasing time
    t = np.linspace(0, 10, 100)
    x = np.sin(t) * np.exp(-0.1*t)
    dx_dt = np.cos(t) * np.exp(-0.1*t) - 0.1 * np.sin(t) * np.exp(-0.1*t)
    d2x_dt2 = -np.sin(t) * np.exp(-0.1*t) - 0.2 * np.cos(t) * np.exp(-0.1*t) + 0.01 * np.sin(t) * np.exp(-0.1*t)
    
    # Ensure t is strictly increasing (remove any potential duplicates)
    unique_indices = np.unique(t, return_index=True)[1]
    t = t[unique_indices]
    x = x[unique_indices]
    dx_dt = dx_dt[unique_indices]
    d2x_dt2 = d2x_dt2[unique_indices]
    
    # Test symbolic regression
    sr = SymbolicRegression()
    equations = sr.extract_equations_from_nn(t, x, dx_dt, d2x_dt2, 'test_system')
    
    assert 'ddot_model' in equations
    assert hasattr(equations['ddot_model'], 'equations')
    
    print("✓ Symbolic regression test passed")

def test_validation():
    """Test validation components"""
    print("Testing validation...")
    
    from validation.compare_long_term import LongTermValidator
    
    validator = LongTermValidator()
    
    # Test true system integration
    t, x, v = validator.integrate_true_system((0, 10), (1.0, 0.0), 'damped_oscillator')
    assert len(t) > 0
    assert len(x) > 0
    assert len(v) > 0
    
    # Test metrics computation
    metrics = validator.compute_metrics(t, x, t, x)
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'energy_variance' in metrics
    
    print("✓ Validation test passed")

def test_complete_pipeline():
    """Test the complete pipeline with minimal data"""
    print("Testing complete pipeline...")
    
    # Create minimal test data
    os.makedirs('data', exist_ok=True)
    
    t = np.linspace(0, 10, 50)
    x = np.sin(t) * np.exp(-0.1*t)
    v = np.cos(t) * np.exp(-0.1*t) - 0.1 * np.sin(t) * np.exp(-0.1*t)
    
    # Save test data
    df = pd.DataFrame({'t': t, 'x': x, 'v': v})
    df.to_csv('data/damped_oscillator.csv', index=False)
    
    # Test pipeline components
    from models.nn_baseline import BaselineNN, BaselineTrainer
    from models.pinn import PINN, PINNTrainer
    from models.hnn import HNN, HNNTrainer, prepare_hnn_data
    
    # Train models with minimal epochs
    baseline_model = BaselineNN()
    baseline_trainer = BaselineTrainer(baseline_model)
    baseline_trainer.train(t, x, epochs=10)
    baseline_trainer.save_model('baseline_nn_damped_oscillator')
    
    pinn_model = PINN()
    pinn_trainer = PINNTrainer(pinn_model)
    pinn_trainer.train(t, x, epochs=10, system='damped_oscillator')
    pinn_trainer.save_model('pinn_damped_oscillator')
    
    q, p, dq_dt, dp_dt = prepare_hnn_data(t, x, v)
    hnn_model = HNN()
    hnn_trainer = HNNTrainer(hnn_model)
    hnn_trainer.train(q, p, dq_dt, dp_dt, epochs=10)
    hnn_trainer.save_model('hnn_damped_oscillator')
    
    print("✓ Complete pipeline test passed")

def main():
    """Run all tests"""
    print("Running AI Safety and Physics Pipeline Tests")
    print("=" * 50)
    
    try:
        test_data_generation()
        test_neural_networks()
        test_symbolic_regression()
        test_validation()
        test_complete_pipeline()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
        print("\nThe pipeline is ready to use.")
        print("Run 'python main.py test' to execute the full pipeline.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 