import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Dict, Tuple, List, Optional
import os
import sys
sys.path.append('..')

from models.nn_baseline import BaselineTrainer, BaselineNN
from models.pinn import PINNTrainer, PINN
from models.hnn import HNNTrainer, HNN, prepare_hnn_data
from regression.symbolic_regression import SymbolicRegression

class LongTermValidator:
    """Validate and compare long-term dynamics of different models"""
    
    def __init__(self):
        self.results = {}
        self.true_equations = {
            'damped_oscillator': 'ddot(x) + 0.1*dot(x) + x = 0',
            'pendulum': 'ddot(theta) + 0.1*dot(theta) + 9.81*sin(theta) = 0'
        }
    
    def load_trained_models(self, system: str):
        """Load trained models"""
        from config import NN_CONFIG, PINN_CONFIG, HNN_CONFIG
        
        # Load baseline model
        baseline_model = BaselineNN(
            hidden_layers=NN_CONFIG['hidden_layers'],
            activation=NN_CONFIG['activation']
        )
        baseline_trainer = BaselineTrainer(baseline_model)
        baseline_trainer.load_model(f'baseline_nn_{system}')
        
        # Load PINN model  
        pinn_model = PINN(
            hidden_layers=PINN_CONFIG['hidden_layers'],
            activation=PINN_CONFIG['activation']
        )
        pinn_trainer = PINNTrainer(pinn_model)
        pinn_trainer.load_model(f'pinn_{system}')
        
        # Load HNN model
        hnn_model = HNN(
            hidden_layers=HNN_CONFIG['hidden_layers'],
            activation=HNN_CONFIG['activation'],
            normalize_inputs=HNN_CONFIG['normalize_inputs']
        )
        # Set normalization if enabled
        if HNN_CONFIG['normalize_inputs']:
            # Reload training data to get normalization stats
            data = pd.read_csv(f'data/{system}.csv')
            if system == 'damped_oscillator':
                from main import prepare_hnn_data
                q, p, _, _ = prepare_hnn_data(data['t'].values, data['x'].values, data['v'].values)
            else:
                from main import prepare_hnn_data
                q, p, _, _ = prepare_hnn_data(data['t'].values, data['theta'].values, data['omega'].values)
            hnn_model.set_normalization(q, p)
        hnn_trainer = HNNTrainer(hnn_model)
        hnn_trainer.load_model(f'hnn_{system}')
        
        return {
            'baseline_nn': baseline_trainer,
            'pinn': pinn_trainer, 
            'hnn': hnn_trainer
        }
    
    def integrate_true_system(self, t_span: Tuple[float, float], 
                            initial_conditions: Tuple[float, float],
                            system: str = 'damped_oscillator') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate the true system for comparison"""
        
        if system == 'damped_oscillator':
            # Use config parameters for consistency
            from config import SYSTEMS
            params = SYSTEMS['damped_oscillator']['parameters']
            m, k, c = params['m'], params['k'], params['c']
            
            def true_system(t, y):
                x, v = y
                dx_dt = v
                dv_dt = -(k/m)*x - (c/m)*v  # Use config parameters
                return [dx_dt, dv_dt]
        elif system == 'pendulum':
            def true_system(t, y):
                theta, omega = y
                dtheta_dt = omega
                domega_dt = -9.81*np.sin(theta) - 0.1*omega  # g=9.81, L=1, c=0.1
                return [dtheta_dt, domega_dt]
        else:
            raise ValueError(f"Unknown system: {system}")
        
        t_eval = np.linspace(*t_span, 1000)
        sol = solve_ivp(true_system, t_span, initial_conditions, 
                       t_eval=t_eval, method='RK45')
        
        return sol.t, sol.y[0], sol.y[1]
    
    def integrate_discovered_equation(self, equation_model, t_span: Tuple[float, float],
                                   initial_conditions: Tuple[float, float],
                                   n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate discovered equation"""
        
        t = np.linspace(*t_span, n_points)
        dt = t[1] - t[0]
        
        x = np.zeros(n_points)
        v = np.zeros(n_points)
        x[0], v[0] = initial_conditions
        
        # Euler integration
        for i in range(1, n_points):
            # Create feature vector
            X_current = np.array([[x[i-1], v[i-1]]])
            
            # Predict acceleration
            a_pred = equation_model.predict(X_current)[0]
            
            # Update state
            v[i] = v[i-1] + dt * a_pred
            x[i] = x[i-1] + dt * v[i-1]
        
        return t, x
    
    def compute_metrics(self, t_true: np.ndarray, x_true: np.ndarray,
                       t_pred: np.ndarray, x_pred: np.ndarray) -> Dict[str, float]:
        """Compute comparison metrics"""
        
        # Interpolate predictions to true time points
        from scipy.interpolate import interp1d
        if len(t_pred) > 1:
            f_interp = interp1d(t_pred, x_pred, kind='linear', bounds_error=False, fill_value='extrapolate')
            x_pred_interp = f_interp(t_true)
        else:
            x_pred_interp = x_pred[0] * np.ones_like(t_true)
        
        # Compute metrics
        rmse = np.sqrt(np.mean((x_true - x_pred_interp)**2))
        mae = np.mean(np.abs(x_true - x_pred_interp))
        
        # Compute energy drift (for conservative systems)
        if len(x_pred) > 1:
            energy_variance = np.var(x_pred**2 + np.gradient(x_pred, t_pred[1]-t_pred[0])**2)
        else:
            energy_variance = 0.0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'energy_variance': energy_variance,
            'max_error': np.max(np.abs(x_true - x_pred_interp))
        }
    
    def compare_long_term_dynamics(self, system: str = 'damped_oscillator',
                                 t_span: Tuple[float, float] = (0, 100),
                                 initial_conditions: Tuple[float, float] = (1.0, 0.0)):
        """Compare long-term dynamics of all models"""
        
        print(f"Comparing long-term dynamics for {system}...")
        
        # Load data
        data = pd.read_csv(f'data/{system}.csv')
        
        # Load trained models
        models = self.load_trained_models(system)
        
        # Integrate true system
        t_true, x_true, v_true = self.integrate_true_system(t_span, initial_conditions, system)
        
        # Get predictions from neural networks
        predictions = {}
        for name, trainer in models.items():
            if name == 'hnn':
                # HNN gives trajectory directly
                t_pred, x_pred, v_pred = trainer.integrate_trajectory(
                    initial_conditions[0], initial_conditions[1], t_span)
            else:
                # Other models predict position vs time
                t_pred = np.linspace(*t_span, 1000)
                x_pred = trainer.predict(t_pred)
                v_pred = np.gradient(x_pred, t_pred[1]-t_pred[0])
            
            predictions[name] = {
                't': t_pred,
                'x': x_pred,
                'v': v_pred
            }
        
        # Extract symbolic equations
        sr = SymbolicRegression()
        equations = {}
        
        for name, trainer in models.items():
            if name == 'hnn':
                # For HNN, we need to get derivatives from the learned dynamics
                q, p, dq_dt, dp_dt = prepare_hnn_data(data['t'].values, data['x'].values, data['v'].values)
                # Use finite differences for second derivative
                d2q_dt2 = np.gradient(dq_dt, data['t'].values[1]-data['t'].values[0])
            else:
                # Get derivatives from neural network
                x, dx_dt, d2x_dt2 = trainer.get_derivatives(data['t'].values, order=2)
                q, p, dq_dt, dp_dt = x, dx_dt, dx_dt, d2x_dt2
            
            eq_data = sr.extract_equations_from_nn(data['t'].values, q, dq_dt, d2x_dt2, f'{name}_{system}')
            equations[name] = eq_data
        
        # Integrate discovered equations
        equation_predictions = {}
        for name, eq_data in equations.items():
            # Get the X data from the equation data
            X = eq_data['X']
            t_eq, x_eq = sr.integrate_discovered_equation(
                eq_data['ddot_model'], X, t_span, initial_conditions)
            equation_predictions[name] = {
                't': t_eq,
                'x': x_eq
            }
        
        # Compute metrics
        metrics = {}
        for name in models.keys():
            # Neural network metrics
            nn_metrics = self.compute_metrics(t_true, x_true, 
                                            predictions[name]['t'], predictions[name]['x'])
            
            # Equation metrics
            eq_metrics = self.compute_metrics(t_true, x_true,
                                           equation_predictions[name]['t'], equation_predictions[name]['x'])
            
            metrics[name] = {
                'neural_network': nn_metrics,
                'discovered_equation': eq_metrics
            }
        
        # Store results
        self.results[system] = {
            'true_trajectory': {'t': t_true, 'x': x_true, 'v': v_true},
            'predictions': predictions,
            'equation_predictions': equation_predictions,
            'metrics': metrics,
            'equations': equations
        }
        
        return self.results[system]
    
    def plot_long_term_comparison(self, system: str = 'damped_oscillator', 
                                save_plot: bool = True):
        """Plot long-term comparison of all models"""
        
        if system not in self.results:
            print(f"No results for {system}. Run compare_long_term_dynamics first.")
            return
        
        results = self.results[system]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Neural network predictions
        t_true = results['true_trajectory']['t']
        x_true = results['true_trajectory']['x']
        
        axes[0, 0].plot(t_true, x_true, 'k-', label='True', linewidth=2)
        
        colors = ['b', 'r', 'g']
        for i, (name, pred) in enumerate(results['predictions'].items()):
            axes[0, 0].plot(pred['t'], pred['x'], f'{colors[i]}-', 
                           label=name.upper(), alpha=0.7, linewidth=1.5)
        
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Position')
        axes[0, 0].set_title('Neural Network Predictions')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(-5, 5)  # Set y-axis limits for oscillator amplitude
        
        # Plot 2: Discovered equation predictions
        axes[0, 1].plot(t_true, x_true, 'k-', label='True', linewidth=2)
        
        for i, (name, pred) in enumerate(results['equation_predictions'].items()):
            axes[0, 1].plot(pred['t'], pred['x'], f'{colors[i]}-', 
                           label=f'{name.upper()} Eq', alpha=0.7, linewidth=1.5)
        
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Position')
        axes[0, 1].set_title('Discovered Equation Predictions')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(-5, 5)  # Set y-axis limits for oscillator amplitude
        
        # Plot 3: Error over time
        for i, (name, pred) in enumerate(results['predictions'].items()):
            # Interpolate to true time points
            from scipy.interpolate import interp1d
            f_interp = interp1d(pred['t'], pred['x'], kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
            x_pred_interp = f_interp(t_true)
            error = np.abs(x_true - x_pred_interp)
            axes[1, 0].plot(t_true, error, f'{colors[i]}-', 
                           label=f'{name.upper()} Error', alpha=0.7)
        
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Absolute Error')
        axes[1, 0].set_title('Neural Network Errors')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Plot 4: Metrics comparison
        metrics = results['metrics']
        model_names = list(metrics.keys())
        nn_rmse = [metrics[name]['neural_network']['rmse'] for name in model_names]
        eq_rmse = [metrics[name]['discovered_equation']['rmse'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, nn_rmse, width, label='Neural Network', alpha=0.7)
        axes[1, 1].bar(x + width/2, eq_rmse, width, label='Discovered Equation', alpha=0.7)
        
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('Final RMSE Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([name.upper() for name in model_names])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            os.makedirs('plots', exist_ok=True)
            plt.savefig(f'plots/{system}_long_term_comparison.png', dpi=300, bbox_inches='tight')
            print(f"Long-term comparison plot saved to plots/{system}_long_term_comparison.png")
        
        plt.show()
    
    def print_comparison_summary(self, system: str = 'damped_oscillator'):
        """Print summary of comparison results"""
        
        if system not in self.results:
            print(f"No results for {system}. Run compare_long_term_dynamics first.")
            return
        
        results = self.results[system]
        metrics = results['metrics']
        
        print(f"\n=== Long-term Dynamics Comparison for {system} ===")
        print("\nNeural Network RMSE:")
        for name, metric in metrics.items():
            print(f"  {name.upper()}: {metric['neural_network']['rmse']:.6f}")
        
        print("\nDiscovered Equation RMSE:")
        for name, metric in metrics.items():
            print(f"  {name.upper()}: {metric['discovered_equation']['rmse']:.6f}")
        
        print("\nEnergy Conservation (Variance):")
        for name, metric in metrics.items():
            print(f"  {name.upper()}: {metric['neural_network']['energy_variance']:.6f}")
        
        # Print discovered equations
        print("\nDiscovered Equations:")
        for name, eq_data in results['equations'].items():
            try:
                equation_str = str(eq_data['ddot_model'].equations())
                print(f"  {name.upper()}: {equation_str}")
            except Exception as e:
                print(f"  {name.upper()}: Error getting equation - {e}")

def main():
    """Run complete validation pipeline"""
    
    validator = LongTermValidator()
    
    # Compare damped oscillator
    print("=== Damped Harmonic Oscillator ===")
    validator.compare_long_term_dynamics('damped_oscillator', t_span=(0, 100))
    validator.plot_long_term_comparison('damped_oscillator')
    validator.print_comparison_summary('damped_oscillator')
    
    # Compare pendulum
    print("\n=== Simple Pendulum ===")
    validator.compare_long_term_dynamics('pendulum', t_span=(0, 100), 
                                      initial_conditions=(np.pi/4, 0.0))
    validator.plot_long_term_comparison('pendulum')
    validator.print_comparison_summary('pendulum')

if __name__ == "__main__":
    main() 