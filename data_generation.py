import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import os

class PhysicsDataGenerator:
    """Generate synthetic physics data for training neural networks"""
    
    def __init__(self, noise_level: float = 0.01, subsample_factor: int = 5):
        self.noise_level = noise_level
        self.subsample_factor = subsample_factor
    
    def damped_harmonic_oscillator(self, t: float, y: np.ndarray, 
                                  m: float = 1.0, k: float = 1.0, c: float = 0.1) -> np.ndarray:
        """Damped harmonic oscillator: m*ddx + c*dx + k*x = 0"""
        x, v = y
        dxdt = v
        dvdt = -(k/m)*x - (c/m)*v
        return [dxdt, dvdt]
    
    def simple_pendulum(self, t: float, y: np.ndarray, 
                       g: float = 9.81, L: float = 1.0, c: float = 0.1) -> np.ndarray:
        """Simple pendulum: ddot(theta) + (c/m)*dot(theta) + (g/L)*sin(theta) = 0"""
        theta, omega = y
        dthetadt = omega
        domegadt = -(g/L)*np.sin(theta) - c*omega
        return [dthetadt, domegadt]
    
    def generate_damped_oscillator_data(self, t_span: Tuple[float, float] = (0, 25),
                                       y0: Tuple[float, float] = (1.0, 0.0),
                                       n_points: int = 500, 
                                       m: float = 1.0, k: float = 1.0, c: float = 0.0) -> Dict[str, np.ndarray]:
        """Generate data for damped harmonic oscillator"""
        t_eval = np.linspace(*t_span, n_points)
        
        # Solve ODE with specified parameters
        sol = solve_ivp(lambda t, y: self.damped_harmonic_oscillator(t, y, m, k, c), 
                       t_span, y0, t_eval=t_eval, method='RK45')
        
        # Extract data
        t = sol.t
        x = sol.y[0]
        v = sol.y[1]
        
        # Add noise
        x_noisy = x + np.random.normal(0, self.noise_level, x.shape)
        v_noisy = v + np.random.normal(0, self.noise_level, v.shape)
        
        # Subsample
        t_sub = t[::self.subsample_factor]
        x_sub = x_noisy[::self.subsample_factor]
        v_sub = v_noisy[::self.subsample_factor]
        
        return {
            't': t_sub,
            'x': x_sub,
            'v': v_sub,
            't_full': t,
            'x_full': x,
            'v_full': v,
            'system': 'damped_oscillator'
        }
    
    def generate_pendulum_data(self, t_span: Tuple[float, float] = (0, 25),
                              y0: Tuple[float, float] = (np.pi/4, 0.0),
                              n_points: int = 500) -> Dict[str, np.ndarray]:
        """Generate data for simple pendulum"""
        t_eval = np.linspace(*t_span, n_points)
        
        # Solve ODE
        sol = solve_ivp(self.simple_pendulum, t_span, y0, 
                       t_eval=t_eval, method='RK45')
        
        # Extract data
        t = sol.t
        theta = sol.y[0]
        omega = sol.y[1]
        
        # Add noise
        theta_noisy = theta + np.random.normal(0, self.noise_level, theta.shape)
        omega_noisy = omega + np.random.normal(0, self.noise_level, omega.shape)
        
        # Subsample
        t_sub = t[::self.subsample_factor]
        theta_sub = theta_noisy[::self.subsample_factor]
        omega_sub = omega_noisy[::self.subsample_factor]
        
        return {
            't': t_sub,
            'theta': theta_sub,
            'omega': omega_sub,
            't_full': t,
            'theta_full': theta,
            'omega_full': omega,
            'system': 'pendulum'
        }
    
    def save_data(self, data: Dict[str, np.ndarray], filename: str):
        """Save generated data to CSV"""
        os.makedirs('data', exist_ok=True)
        
        # Create DataFrame
        if data['system'] == 'damped_oscillator':
            df = pd.DataFrame({
                't': data['t'],
                'x': data['x'],
                'v': data['v']
            })
        else:  # pendulum
            df = pd.DataFrame({
                't': data['t'],
                'theta': data['theta'],
                'omega': data['omega']
            })
        
        df.to_csv(f'data/{filename}.csv', index=False)
        print(f"Data saved to data/{filename}.csv")
    
    def plot_data(self, data: Dict[str, np.ndarray], save_plot: bool = True):
        """Plot generated data"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        if data['system'] == 'damped_oscillator':
            # Training data
            axes[0, 0].plot(data['t'], data['x'], 'b.', label='Training data')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Position')
            axes[0, 0].set_title('Damped Oscillator - Position')
            axes[0, 0].legend()
            
            axes[0, 1].plot(data['t'], data['v'], 'r.', label='Training data')
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Velocity')
            axes[0, 1].set_title('Damped Oscillator - Velocity')
            axes[0, 1].legend()
            
            # Full trajectory
            axes[1, 0].plot(data['t_full'], data['x_full'], 'b-', label='Full trajectory')
            axes[1, 0].plot(data['t'], data['x'], 'r.', label='Training points')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Position')
            axes[1, 0].set_title('Full vs Training Data')
            axes[1, 0].legend()
            
            # Phase space
            axes[1, 1].plot(data['x'], data['v'], 'g.', label='Training data')
            axes[1, 1].set_xlabel('Position')
            axes[1, 1].set_ylabel('Velocity')
            axes[1, 1].set_title('Phase Space')
            axes[1, 1].legend()
            
        else:  # pendulum
            # Training data
            axes[0, 0].plot(data['t'], data['theta'], 'b.', label='Training data')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Angle')
            axes[0, 0].set_title('Simple Pendulum - Angle')
            axes[0, 0].legend()
            
            axes[0, 1].plot(data['t'], data['omega'], 'r.', label='Training data')
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Angular Velocity')
            axes[0, 1].set_title('Simple Pendulum - Angular Velocity')
            axes[0, 1].legend()
            
            # Full trajectory
            axes[1, 0].plot(data['t_full'], data['theta_full'], 'b-', label='Full trajectory')
            axes[1, 0].plot(data['t'], data['theta'], 'r.', label='Training points')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Angle')
            axes[1, 0].set_title('Full vs Training Data')
            axes[1, 0].legend()
            
            # Phase space
            axes[1, 1].plot(data['theta'], data['omega'], 'g.', label='Training data')
            axes[1, 1].set_xlabel('Angle')
            axes[1, 1].set_ylabel('Angular Velocity')
            axes[1, 1].set_title('Phase Space')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_plot:
            os.makedirs('plots', exist_ok=True)
            plt.savefig(f'plots/{data["system"]}_data.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to plots/{data['system']}_data.png")
        
        plt.show()

def main():
    """Generate and save data for both systems"""
    generator = PhysicsDataGenerator(noise_level=0.01, subsample_factor=5)
    
    # Generate damped oscillator data
    print("Generating damped harmonic oscillator data...")
    oscillator_data = generator.generate_damped_oscillator_data()
    generator.save_data(oscillator_data, 'damped_oscillator')
    generator.plot_data(oscillator_data)
    
    # Generate pendulum data
    print("Generating simple pendulum data...")
    pendulum_data = generator.generate_pendulum_data()
    generator.save_data(pendulum_data, 'simple_pendulum')
    generator.plot_data(pendulum_data)

if __name__ == "__main__":
    main() 