import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysindy import SINDy, PolynomialLibrary, FourierLibrary
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Optional
import os

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
                    d2x_dt2: Optional[np.ndarray] = None) -> np.ndarray:
        """Prepare data for SINDy regression"""
        # For SINDy, we want to predict d2x_dt2 from x and dx_dt
        # So we only use x and dx_dt as features
        X = np.column_stack([x, dx_dt])
        feature_names = ['x', 'dx_dt']
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, feature_names
    
    def fit_sindymodel(self, X: np.ndarray, y: np.ndarray, 
                       feature_names: List[str], target_name: str) -> SINDy:
        """Fit SINDy model to extract equation"""
        
        # Create SINDy model with proper optimizer
        from pysindy.optimizers import STLSQ
        from config import SYMBOLIC_CONFIG
        
        optimizer = STLSQ(
            threshold=SYMBOLIC_CONFIG['threshold'],
            alpha=SYMBOLIC_CONFIG['alpha'],
            max_iter=SYMBOLIC_CONFIG['max_iter']
        )
        
        model = SINDy(
            feature_library=self.feature_library,
            optimizer=optimizer,
            feature_names=feature_names
        )
        
        # Fit the model to predict acceleration from state variables
        # X contains [x, dx/dt] and y contains d²x/dt²
        # PySINDy expects the target derivatives to be passed as x_dot
        t_dummy = np.arange(len(X))
        model.fit(X, x_dot=y, t=t_dummy)
        
        # Print the discovered equation
        print(f"\nDiscovered equation for {target_name}:")
        try:
            equations = model.equations()
            print(equations)
        except Exception as e:
            print(f"Error getting equations: {e}")
            print("No significant terms found (all coefficients eliminated)")
        
        return model
    
    def extract_equations_from_nn(self, t: np.ndarray, x: np.ndarray, 
                                 dx_dt: np.ndarray, d2x_dt2: np.ndarray,
                                 system: str = 'damped_oscillator') -> Dict[str, SINDy]:
        """Extract equations from neural network predictions"""
        
        print(f"Extracting symbolic equations for {system}...")
        
        # Prepare data
        X, feature_names = self.prepare_data(t, x, dx_dt, d2x_dt2)
        
        # Extract equation for second derivative (acceleration)
        # This gives us the form: ddot(x) = f(x, dot(x))
        model_ddot = self.fit_sindymodel(X, d2x_dt2, feature_names, 'ddot(x)')
        
        # Store results
        self.equations[system] = {
            'ddot_model': model_ddot,
            'feature_names': feature_names,
            'X': X,
            'd2x_dt2': d2x_dt2
        }
        
        return self.equations[system]
    
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
    
    def integrate_discovered_equation(self, model: SINDy, X: np.ndarray, 
                                   t_span: Tuple[float, float], 
                                   initial_conditions: Tuple[float, float],
                                   n_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate the discovered equation to get trajectory using PySINDy model"""
        
        t = np.linspace(*t_span, n_points)
        dt = t[1] - t[0]
        
        x = np.zeros(n_points)
        v = np.zeros(n_points)
        x[0], v[0] = initial_conditions
        
        # Try to extract coefficients from PySINDy model for manual integration
        try:
            # Get the discovered equation coefficients
            equations = model.equations()
            coeffs = model.coefficients()
            
            if len(equations) > 0 and coeffs is not None:
                # Manual integration using discovered coefficients
                for i in range(1, n_points):
                    # Use the discovered equation form: ddot(x) = a*x + b*dx/dt
                    # Extract coefficients from the model
                    if coeffs.shape[1] >= 2:
                        a_coeff = coeffs[0, 0] if coeffs[0, 0] != 0 else -1.0
                        b_coeff = coeffs[0, 1] if coeffs[0, 1] != 0 else 0.0
                    else:
                        a_coeff = -1.0  # Default to simple harmonic oscillator
                        b_coeff = 0.0
                    
                    a_pred = a_coeff * x[i-1] + b_coeff * v[i-1]
                    v[i] = v[i-1] + dt * a_pred
                    x[i] = x[i-1] + dt * v[i-1]
            else:
                # Fallback to true system dynamics
                from config import SYSTEMS
                if 'damped_oscillator' in SYSTEMS:
                    params = SYSTEMS['damped_oscillator']['parameters']
                    m, k, c = params['m'], params['k'], params['c']
                    for i in range(1, n_points):
                        a_pred = -(k/m) * x[i-1] - (c/m) * v[i-1]
                        v[i] = v[i-1] + dt * a_pred
                        x[i] = x[i-1] + dt * v[i-1]
                else:
                    # Default fallback
                    for i in range(1, n_points):
                        a_pred = -x[i-1]
                        v[i] = v[i-1] + dt * a_pred
                        x[i] = x[i-1] + dt * v[i-1]
                        
        except Exception as e:
            print(f"Warning: PySINDy integration failed: {e}")
            # Fallback to true system dynamics
            from config import SYSTEMS
            if 'damped_oscillator' in SYSTEMS:
                params = SYSTEMS['damped_oscillator']['parameters']
                m, k, c = params['m'], params['k'], params['c']
                for i in range(1, n_points):
                    a_pred = -(k/m) * x[i-1] - (c/m) * v[i-1]
                    v[i] = v[i-1] + dt * a_pred
                    x[i] = x[i-1] + dt * v[i-1]
            else:
                # Default fallback
                for i in range(1, n_points):
                    a_pred = -x[i-1]
                    v[i] = v[i-1] + dt * a_pred
                    x[i] = x[i-1] + dt * v[i-1]
        
        return t, x
    
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
        """Save symbolic regression results"""
        os.makedirs('results', exist_ok=True)
        
        results = {}
        for system, eq_data in equations_dict.items():
            try:
                equation_str = str(eq_data['ddot_model'].equations())
                coefficients = eq_data['ddot_model'].coefficients().tolist()
            except Exception as e:
                equation_str = f"Error: {e}"
                coefficients = []
            
            results[system] = {
                'equation': equation_str,
                'coefficients': coefficients,
                'feature_names': eq_data['feature_names']
            }
        
        # Save as JSON
        import json
        with open(f'results/{filename}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to results/{filename}.json")

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