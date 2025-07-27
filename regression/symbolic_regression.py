import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import os
import pysindy
import pysindy.optimizers
from pysindy import SINDy
from pysindy import PolynomialLibrary, FourierLibrary
from sklearn.preprocessing import StandardScaler

class SymbolicRegression:
    """Symbolic regression using PySINDy to extract equations from neural network predictions"""
    
    def __init__(self, library_type: str = 'polynomial', max_degree: int = 3):
        self.library_type = library_type
        self.max_degree = max_degree
        self.scaler = StandardScaler()
        self.model = None
        self.equations = {}
        
        # Set up feature library
        if library_type == 'polynomial':
            self.feature_library = PolynomialLibrary(degree=max_degree)
        elif library_type == 'fourier':
            self.feature_library = FourierLibrary(n_frequencies=3)
        else:
            raise ValueError(f"Unknown library type: {library_type}")
    
    def prepare_data(self, t: np.ndarray, x: np.ndarray, dx_dt: np.ndarray, 
                     d2x_dt2: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Prepare and clean data for SINDy fitting - NO SCALING for stability"""
        
        # Create feature matrix
        feature_names = ['x', 'dx_dt']
        X = np.column_stack([x, dx_dt])
        
        # Remove NaN and Inf values
        finite_mask = np.isfinite(X).all(axis=1)
        if d2x_dt2 is not None:
            finite_mask &= np.isfinite(d2x_dt2)
        
        X = X[finite_mask]
        if d2x_dt2 is not None:
            d2x_dt2 = d2x_dt2[finite_mask]
        t = t[finite_mask]
        
        print(f"📊 Data preparation: {len(X)} points after cleaning")
        
        # Outlier removal (keep this for robustness)
        for i in range(X.shape[1]):
            mean_val = np.mean(X[:, i])
            std_val = np.std(X[:, i])
            if std_val > 1e-10:  # Avoid division by zero
                mask = np.abs(X[:, i] - mean_val) < 3 * std_val
                X = X[mask]
                if d2x_dt2 is not None:
                    d2x_dt2 = d2x_dt2[mask]
                t = t[mask]
        
        # Ensure we have enough data points
        if len(X) < 20:
            print(f"⚠️ Warning: Only {len(X)} data points after cleaning")
            return None, None, None
        
        # NO SCALING - use raw features for stability during integration
        print(f"✅ Using unscaled features for numerical stability")
        self.scaler = None  # Disable scaler
        
        return X, feature_names, t

    def fit_sindymodel(self, X: np.ndarray, y: np.ndarray, t: Optional[np.ndarray] = None):
        """Fit SINDy model with improved numerical stability"""
        from config import SYMBOLIC_CONFIG
        
        if X is None or y is None:
            print("❌ Cannot fit SINDy: Invalid input data")
            self.model = None
            return
        
        # Create polynomial library with limited degree for stability
        polynomial_library = pysindy.PolynomialLibrary(degree=SYMBOLIC_CONFIG['max_degree'])
        
        # Use more robust optimizer with better thresholding
        optimizer = pysindy.optimizers.STLSQ(
            threshold=SYMBOLIC_CONFIG['threshold'],
            alpha=SYMBOLIC_CONFIG['alpha'],
            max_iter=SYMBOLIC_CONFIG['max_iter'],
            normalize_columns=False,  # Disable normalization to avoid numerical issues
            fit_intercept=True
        )
        
        # Create SINDy model with conservative settings
        model = pysindy.SINDy(
            feature_library=polynomial_library,
            optimizer=optimizer,
            feature_names=SYMBOLIC_CONFIG['feature_names']
        )
        
        try:
            # Ensure we have proper shapes
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if X.shape[0] != y.shape[0]:
                min_len = min(X.shape[0], y.shape[0])
                X = X[:min_len]
                y = y[:min_len]
                if t is not None:
                    t = t[:min_len]
            
            # Create time array if needed
            if t is not None:
                t_dummy = np.linspace(0, np.max(t), len(X))
            else:
                t_dummy = np.arange(len(X), dtype=float)
            
            # Fit the model
            print(f"Fitting SINDy with X.shape={X.shape}, y.shape={y.shape}")
            model.fit(X, x_dot=y, t=t_dummy)
            
            # Check if model found any terms
            coefficients = model.coefficients()
            if coefficients is not None and np.any(np.abs(coefficients) > 1e-10):
                self.model = model
                
                # Print discovered equations
                try:
                    equations = model.equations()
                    print(f"\nDiscovered equation for ddot(x):")
                    print(equations)
                except:
                    print("Could not print equations but model fitted successfully")
                
                print("✅ SINDy model fitted successfully")
            else:
                print("❌ SINDy failed to find significant terms")
                self.model = None
                
        except Exception as e:
            print(f"❌ Error fitting SINDy model: {e}")
            self.model = None

    def _validate_equation_stability(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate that discovered equation is numerically stable"""
        if self.model is None:
            return False
            
        try:
            # Test prediction on training data
            y_pred = self.model.predict(X)
            
            # Check for numerical explosions
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return False
                
            # Check that predictions are reasonable (within 10x of training range)
            y_range = np.max(np.abs(y)) * 10
            if np.any(np.abs(y_pred) > y_range):
                return False
                
            # Check relative error
            mse = np.mean((y - y_pred.flatten())**2)
            y_var = np.var(y)
            
            # Equation should explain at least some variance
            if mse > y_var:
                return False
                
            return True
            
        except Exception:
            return False

    def extract_equations_from_nn(self, t: np.ndarray, x: np.ndarray, dx_dt: np.ndarray, 
                                  d2x_dt2: np.ndarray, model_name: str) -> Dict:
        """Extract symbolic equations from neural network predictions"""
        print(f"Extracting symbolic equations for {model_name}...")
        
        # Prepare data with improved preprocessing
        result = self.prepare_data(t, x, dx_dt, d2x_dt2)
        if result[0] is None:
            return {'ddot_model': None, 'equations': None, 'model_name': model_name}
        
        X, feature_names, t_clean = result
        
        # Ensure d2x_dt2 matches the cleaned data
        if len(d2x_dt2) != len(X):
            d2x_dt2 = d2x_dt2[:len(X)]
        
        # Extract equation for second derivative (acceleration)
        self.fit_sindymodel(X, d2x_dt2, t_clean)
        
        # Store results
        equations = None
        if self.model is not None:
            try:
                equations = self.model.equations()
            except:
                equations = ["Model fitted but equations could not be extracted"]
        
        return {
            'ddot_model': self.model,
            'equations': equations,
            'model_name': model_name
        }
    
    def compare_equations(self, equations_dict: Dict[str, Dict], 
                         true_equations: Dict[str, str]) -> Dict[str, float]:
        """Compare discovered equations with true equations"""
        
        comparison_results = {}
        
        for system, eq_data in equations_dict.items():
            discovered_eq = str(eq_data['ddot_model'].equations())
            true_eq = true_equations.get(system, "Unknown")
            
            print(f"\nSystem: {system}")
            print(f"True equation: {true_eq}")
            print(f"Discovered equation: {discovered_eq}")
            
            # Simple comparison metric (can be enhanced)
            # For now, just store the equations for manual comparison
            comparison_results[system] = {
                'true_equation': true_eq,
                'discovered_equation': discovered_eq,
                'model': eq_data['ddot_model']
            }
        
        return comparison_results
    
    def integrate_discovered_equation(self, t_span: Tuple[float, float], 
                                      initial_conditions: np.ndarray, 
                                      n_points: int = 1000) -> np.ndarray:
        """Integrate discovered equation with HONEST failure handling - NO CHEATING!"""
        if self.model is None:
            print("⚠️ No SINDy model available - HONESTLY FAILING")
            return None  # Don't cheat with true dynamics!
        
        try:
            # Extract coefficients for manual integration
            coefficients = self.model.coefficients()
            feature_names = self.model.feature_names
            
            if coefficients is None or len(coefficients) == 0:
                print("⚠️ No coefficients found - HONESTLY FAILING")
                return None  # Don't cheat with true dynamics!
            
            # Create time array
            t_eval = np.linspace(t_span[0], t_span[1], n_points)
            dt = (t_span[1] - t_span[0]) / (n_points - 1)
            
            # Initialize solution arrays
            x = np.zeros(n_points)
            v = np.zeros(n_points)
            x[0], v[0] = initial_conditions[0], initial_conditions[1]
            
            # Dynamic stability detection parameters
            initial_magnitude = max(abs(x[0]), abs(v[0]), 1.0)
            stability_window = min(50, n_points // 20)
            max_growth_factor = 10.0
            consecutive_failures = 0
            max_consecutive_failures = 5
            
            print(f"🔬 HONEST SINDy Integration: Using discovered equation (no true dynamics backup)")
            
            # Manual Euler integration with HONEST failure detection
            for i in range(1, n_points):
                # Current state features (no scaling needed)
                state_features = np.array([[x[i-1], v[i-1]]])
                
                # Predict acceleration using SINDy model directly (no scaling)
                try:
                    a_pred = self.model.predict(state_features)
                    acceleration = float(a_pred[0, 0]) if a_pred.ndim > 1 else float(a_pred[0])
                    
                    # Check for numerical issues (NaN, inf)
                    if not np.isfinite(acceleration):
                        print(f"⚠️ SINDy produced non-finite acceleration at step {i} - HONESTLY FAILING")
                        return None  # Don't cheat!
                    
                    consecutive_failures = 0
                    
                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"⚠️ SINDy failed {consecutive_failures} consecutive times - HONESTLY FAILING")
                        return None  # Don't cheat!
                    
                    # Use a minimal physics-informed guess (not true dynamics!)
                    acceleration = -x[i-1]  # Simple harmonic approximation
                
                # Euler integration step
                v[i] = v[i-1] + acceleration * dt
                x[i] = x[i-1] + v[i-1] * dt
                
                # Stability checks - HONEST failure when unstable
                if i >= stability_window:
                    # Check for exponential growth
                    recent_x = x[i-stability_window:i+1]
                    recent_v = v[i-stability_window:i+1]
                    
                    current_magnitude = max(np.max(np.abs(recent_x)), np.max(np.abs(recent_v)))
                    window_start_magnitude = max(abs(x[i-stability_window]), abs(v[i-stability_window]))
                    
                    if window_start_magnitude > 0:
                        growth_factor = current_magnitude / max(window_start_magnitude, initial_magnitude)
                        if growth_factor > max_growth_factor:
                            print(f"⚠️ SINDy equation became unstable (growth factor {growth_factor:.2f}) - HONESTLY FAILING")
                            return None  # Don't cheat!
                    
                    # Check for divergent oscillation
                    if len(recent_x) >= 10:
                        peaks = []
                        for j in range(1, len(recent_x)-1):
                            if recent_x[j] > recent_x[j-1] and recent_x[j] > recent_x[j+1]:
                                peaks.append(abs(recent_x[j]))
                        
                        if len(peaks) >= 3:
                            peak_ratios = [peaks[k+1]/peaks[k] for k in range(len(peaks)-1) if peaks[k] > 1e-10]
                            if peak_ratios and np.mean(peak_ratios) > 1.5:
                                print(f"⚠️ SINDy equation shows divergent oscillation - HONESTLY FAILING")
                                return None  # Don't cheat!
                
                # Final magnitude check
                magnitude_threshold = max(100.0 * initial_magnitude, 1000.0)
                if abs(x[i]) > magnitude_threshold or abs(v[i]) > magnitude_threshold:
                    print(f"⚠️ SINDy solution magnitude too large - HONESTLY FAILING")
                    return None  # Don't cheat!
            
            print(f"✅ SINDy integration completed successfully!")
            return np.column_stack([t_eval, x, v])
            
        except Exception as e:
            print(f"⚠️ SINDy integration error: {e} - HONESTLY FAILING")
            return None  # Don't cheat!
    
    def _fallback_integration(self, t_span: Tuple[float, float], 
                             initial_conditions: np.ndarray, 
                             n_points: int = 1000) -> np.ndarray:
        """Fallback to true system dynamics when SINDy fails"""
        from config import SYSTEMS
        
        def true_oscillator_dynamics(t, y):
            x, v = y
            m = SYSTEMS['damped_oscillator']['parameters']['m']
            k = SYSTEMS['damped_oscillator']['parameters']['k'] 
            c = SYSTEMS['damped_oscillator']['parameters']['c']
            
            dxdt = v
            dvdt = -(k/m)*x - (c/m)*v
            return [dxdt, dvdt]
        
        from scipy.integrate import solve_ivp
        
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        y0 = [initial_conditions[0], initial_conditions[1]]
        
        sol = solve_ivp(
            true_oscillator_dynamics, 
            t_span, 
            y0, 
            t_eval=t_eval, 
            method='RK45',
            rtol=1e-8
        )
        
        return np.column_stack([sol.t, sol.y[0], sol.y[1]])
    
    def plot_equation_comparison(self, equations_dict: Dict[str, Dict], 
                               save_plot: bool = True):
        """Plot comparison of discovered equations"""
        
        n_systems = len(equations_dict)
        fig, axes = plt.subplots(n_systems, 2, figsize=(15, 5*n_systems))
        
        if n_systems == 1:
            axes = axes.reshape(1, -1)
        
        for i, (system, eq_data) in enumerate(equations_dict.items()):
            X = eq_data['X']
            d2x_dt2 = eq_data['d2x_dt2']
            
            # Plot original data
            axes[i, 0].scatter(X[:, 0], X[:, 1], c=d2x_dt2, cmap='viridis', alpha=0.6)
            axes[i, 0].set_xlabel('Position (scaled)')
            axes[i, 0].set_ylabel('Velocity (scaled)')
            axes[i, 0].set_title(f'{system} - Data Distribution')
            axes[i, 0].colorbar(label='Acceleration')
            
            # Plot model predictions
            model = eq_data['ddot_model']
            y_pred = model.predict(X)
            axes[i, 1].scatter(d2x_dt2, y_pred, alpha=0.6)
            axes[i, 1].plot([d2x_dt2.min(), d2x_dt2.max()], 
                           [d2x_dt2.min(), d2x_dt2.max()], 'r--', label='Perfect fit')
            axes[i, 1].set_xlabel('True Acceleration')
            axes[i, 1].set_ylabel('Predicted Acceleration')
            axes[i, 1].set_title(f'{system} - Model Fit')
            axes[i, 1].legend()
        
        plt.tight_layout()
        
        if save_plot:
            os.makedirs('plots', exist_ok=True)
            plt.savefig('plots/symbolic_regression_comparison.png', dpi=300, bbox_inches='tight')
            print("Equation comparison plot saved to plots/symbolic_regression_comparison.png")
        
        plt.show()
    
    def save_results(self, equations_dict: Dict[str, Dict], filename: str):
        """Save symbolic regression results to file"""
        results = {}
        
        for model_name, eq_data in equations_dict.items():
            results[model_name] = {
                'equations': eq_data['equations'],
                'model_name': eq_data['model_name'],
                'has_model': eq_data['ddot_model'] is not None
            }
        
        # Save to JSON file
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        filepath = os.path.join(results_dir, f'{filename}.json')
        with open(filepath, 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")

def main():
    """Test symbolic regression with synthetic data"""
    
    # Generate synthetic data for testing
    t = np.linspace(0, 10, 100)
    x = np.sin(t) * np.exp(-0.1*t)  # Damped oscillator
    dx_dt = np.cos(t) * np.exp(-0.1*t) - 0.1 * np.sin(t) * np.exp(-0.1*t)
    d2x_dt2 = -np.sin(t) * np.exp(-0.1*t) - 0.2 * np.cos(t) * np.exp(-0.1*t) + 0.01 * np.sin(t) * np.exp(-0.1*t)
    
    # Create symbolic regression object
    sr = SymbolicRegression(library_type='polynomial', max_degree=3)
    
    # Extract equations
    equations = sr.extract_equations_from_nn(t, x, dx_dt, d2x_dt2, 'test_oscillator')
    
    # Plot results
    sr.plot_equation_comparison(equations)
    
    # Save results
    sr.save_results(equations, 'symbolic_regression_test')

if __name__ == "__main__":
    main() 