#!/usr/bin/env python3
"""
Cascaded HNN: t → x (NN) → v (autograd) → (x,v) → H (HNN)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from scipy.integrate import solve_ivp
from models.hnn import HNN


class TrajectoryNN(nn.Module):
    """Neural network that predicts x from time t"""
    
    def __init__(self, hidden_layers: List[int] = [64, 32], activation: str = 'tanh'):
        super(TrajectoryNN, self).__init__()
        
        # Build layers dynamically
        layers = []
        input_dim = 1  # time t
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sine':
                layers.append(nn.SiLU())  # SiLU is close to sine
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))  # x(t)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, t):
        """Predict x from time t"""
        return self.network(t.unsqueeze(-1)).squeeze(-1)


class CascadedHNN(nn.Module):
    """Cascaded HNN: t → x (NN) → v (autograd) → (x,v) → H (HNN)"""
    
    def __init__(self, trajectory_config: Dict, hnn_config: Dict):
        super(CascadedHNN, self).__init__()
        
        # Trajectory network: t → x
        self.trajectory_net = TrajectoryNN(
            hidden_layers=trajectory_config['hidden_layers'],
            activation=trajectory_config['activation']
        )
        
        # HNN network: (x,v) → H
        self.hnn_net = HNN(
            hidden_layers=hnn_config['hidden_layers'],
            activation=hnn_config['activation'],
            normalize_inputs=hnn_config.get('normalize_inputs', False)
        )
        
        self.trajectory_config = trajectory_config
        self.hnn_config = hnn_config
    
    def forward(self, t):
        """Forward pass: t → x → v → H"""
        # Ensure t requires grad for autograd
        if not t.requires_grad:
            t = t.requires_grad_(True)
        
        # Step 1: Predict trajectory
        x = self.trajectory_net(t)
        
        # Step 2: Compute velocity via autograd
        v = torch.autograd.grad(x.sum(), t, create_graph=True, retain_graph=True)[0]
        
        # Step 3: HNN on learned trajectory
        H = self.hnn_net(x.unsqueeze(-1), v.unsqueeze(-1))
        
        return x, v, H
    
    def predict_trajectory(self, t):
        """Predict only trajectory (for comparison with other models)"""
        return self.trajectory_net(t)
    
    def get_derivatives(self, t, order=2):
        """Get derivatives for symbolic regression"""
        t_tensor = torch.tensor(t, dtype=torch.float32, requires_grad=True)
        x = self.trajectory_net(t_tensor)
        
        derivatives = [x.detach().numpy()]
        
        if order >= 1:
            dx_dt = torch.autograd.grad(x.sum(), t_tensor, create_graph=True)[0]
            derivatives.append(dx_dt.detach().numpy())
        
        if order >= 2:
            d2x_dt2 = torch.autograd.grad(dx_dt.sum(), t_tensor, create_graph=True)[0]
            derivatives.append(d2x_dt2.detach().numpy())
        
        return derivatives
    
    def predict_dynamics(self, x, v):
        """Get dynamics from HNN component"""
        # Convert to numpy arrays for the HNN predict_dynamics method
        if isinstance(x, torch.Tensor):
            x_np = x.detach().numpy()
        else:
            x_np = x
        if isinstance(v, torch.Tensor):
            v_np = v.detach().numpy()
        else:
            v_np = v
        
        # Use the HNN's get_dynamics method directly
        x_tensor = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
        v_tensor = torch.tensor(v_np, dtype=torch.float32, requires_grad=True)
        return self.hnn_net.get_dynamics(x_tensor, v_tensor)
    
    def predict_hamiltonian(self, x, v):
        """Get Hamiltonian from HNN component"""
        # Convert to tensors for the HNN forward method
        if isinstance(x, torch.Tensor):
            x_tensor = x
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32)
        if isinstance(v, torch.Tensor):
            v_tensor = v
        else:
            v_tensor = torch.tensor(v, dtype=torch.float32)
        
        # Use the HNN's forward method directly
        return self.hnn_net(x_tensor, v_tensor)


class CascadedHNNTrainer:
    """Trainer for cascaded HNN"""
    
    def __init__(self, model: CascadedHNN, 
                 trajectory_weight: float = 1.0,
                 hnn_weight: float = 1.0,
                 energy_weight: float = 0.1,
                 learning_rate: float = 1e-3,
                 optimizer_type: str = 'adamw',
                 weight_decay: float = 1e-4,
                 scheduler_type: str = 'steplr',
                 scheduler_params: Dict = None):
        
        self.model = model
        self.trajectory_weight = trajectory_weight
        self.hnn_weight = hnn_weight
        self.energy_weight = energy_weight
        
        # Setup SEPARATE optimizers for each component (like the individual models)
        if optimizer_type == 'adam':
            self.trajectory_optimizer = optim.Adam(model.trajectory_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.hnn_optimizer = optim.Adam(model.hnn_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            self.trajectory_optimizer = optim.AdamW(model.trajectory_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.hnn_optimizer = optim.AdamW(model.hnn_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Setup schedulers
        if scheduler_type == 'steplr':
            step_size = scheduler_params.get('step_size', 1000) if scheduler_params else 1000
            gamma = scheduler_params.get('gamma', 0.9) if scheduler_params else 0.9
            self.trajectory_scheduler = optim.lr_scheduler.StepLR(self.trajectory_optimizer, step_size=step_size, gamma=gamma)
            self.hnn_scheduler = optim.lr_scheduler.StepLR(self.hnn_optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'reduceonplateau':
            self.trajectory_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.trajectory_optimizer, patience=500, factor=0.5)
            self.hnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.hnn_optimizer, patience=500, factor=0.5)
        else:
            self.trajectory_scheduler = None
            self.hnn_scheduler = None
        
        self.history = {'trajectory_loss': [], 'hnn_loss': [], 'energy_loss': [], 'total_loss': [], 
                        'val_trajectory_loss': [], 'val_hnn_loss': [], 'val_energy_loss': [], 'val_total_loss': []}
    
    def compute_trajectory_loss(self, x_pred, x_true):
        """Compute trajectory prediction loss"""
        return torch.mean((x_pred - x_true) ** 2)
    
    def compute_hnn_loss(self, x_pred, v_pred, x_true, v_true):
        """Compute HNN dynamics loss - SIMPLIFIED to avoid gradient issues"""
        # Instead of complex dynamics loss, use a simpler approach
        # Just ensure the HNN learns to predict reasonable dynamics
        
        # Get dynamics from HNN
        dq_dt_pred, dp_dt_pred = self.model.predict_dynamics(
            x_pred.unsqueeze(-1), v_pred.unsqueeze(-1))
        
        # Simple loss: ensure dq_dt_pred ≈ v_pred (velocity consistency)
        velocity_consistency_loss = torch.mean((dq_dt_pred.squeeze() - v_pred) ** 2)
        
        # Simple loss: ensure dp_dt_pred is reasonable (not too large)
        acceleration_regularization = torch.mean(torch.abs(dp_dt_pred.squeeze()))
        
        return velocity_consistency_loss + 0.1 * acceleration_regularization
    
    def compute_energy_loss(self, x_pred, v_pred):
        """Compute energy conservation loss"""
        H_pred = self.model.predict_hamiltonian(x_pred.unsqueeze(-1), v_pred.unsqueeze(-1))
        energy_variance = torch.var(H_pred)
        return energy_variance
    
    def train(self, t: np.ndarray, x: np.ndarray, v: np.ndarray, 
              epochs: int = 5000, batch_size: int = 64, 
              val_split: float = 0.2, patience: int = 1000):
        """Train the cascaded HNN"""
        
        # Convert to tensors
        t_tensor = torch.tensor(t, dtype=torch.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        v_tensor = torch.tensor(v, dtype=torch.float32)
        
        # Split into train/val
        n_train = int(len(t) * (1 - val_split))
        t_train, x_train, v_train = t_tensor[:n_train], x_tensor[:n_train], v_tensor[:n_train]
        t_val, x_val, v_val = t_tensor[n_train:], x_tensor[n_train:], v_tensor[n_train:]
        
        print(f"Training cascaded HNN with {len(t_train)} training samples, {len(t_val)} validation samples")
        print(f"Trajectory weight: {self.trajectory_weight}, HNN weight: {self.hnn_weight}, Energy weight: {self.energy_weight}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            
            # Forward pass
            x_pred, v_pred, H_pred = self.model(t_train)
            
            # Compute losses
            trajectory_loss = self.compute_trajectory_loss(x_pred, x_train)
            hnn_loss = self.compute_hnn_loss(x_pred, v_pred, x_train, v_train)
            energy_loss = self.compute_energy_loss(x_pred, v_pred)
            
            # Total loss
            total_loss = (self.trajectory_weight * trajectory_loss + 
                         self.hnn_weight * hnn_loss + 
                         self.energy_weight * energy_loss)
            
            # Backward pass with separate optimizers and gradient clipping
            self.trajectory_optimizer.zero_grad()
            self.hnn_optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.trajectory_net.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.hnn_net.parameters(), max_norm=1.0)
            
            self.trajectory_optimizer.step()
            self.hnn_optimizer.step()
            
            # Validation
            self.model.eval()
            # Ensure validation tensors require grad for autograd
            t_val_with_grad = t_val.requires_grad_(True)
            x_val_pred, v_val_pred, H_val_pred = self.model(t_val_with_grad)
            val_trajectory_loss = self.compute_trajectory_loss(x_val_pred, x_val)
            val_hnn_loss = self.compute_hnn_loss(x_val_pred, v_val_pred, x_val, v_val)
            val_energy_loss = self.compute_energy_loss(x_val_pred, v_val_pred)
            val_total_loss = (self.trajectory_weight * val_trajectory_loss + 
                            self.hnn_weight * val_hnn_loss + 
                            self.energy_weight * val_energy_loss)
            
            # Update schedulers
            if self.trajectory_scheduler is not None:
                if isinstance(self.trajectory_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.trajectory_scheduler.step(val_total_loss)
                else:
                    self.trajectory_scheduler.step()
            
            if self.hnn_scheduler is not None:
                if isinstance(self.hnn_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.hnn_scheduler.step(val_total_loss)
                else:
                    self.hnn_scheduler.step()
            
            # Store history
            self.history['trajectory_loss'].append(trajectory_loss.item())
            self.history['hnn_loss'].append(hnn_loss.item())
            self.history['energy_loss'].append(energy_loss.item())
            self.history['total_loss'].append(total_loss.item())
            self.history['val_trajectory_loss'].append(val_trajectory_loss.item())
            self.history['val_hnn_loss'].append(val_hnn_loss.item())
            self.history['val_energy_loss'].append(val_energy_loss.item())
            self.history['val_total_loss'].append(val_total_loss.item())
            
            # Early stopping
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/saved/cascaded_hnn_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch}: Trajectory={trajectory_loss.item():.6f}, "
                      f"HNN={hnn_loss.item():.6f}, Energy={energy_loss.item():.6f}, "
                      f"Total={total_loss.item():.6f}, Val={val_total_loss.item():.6f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('models/saved/cascaded_hnn_best.pth'))
        print("Training completed!")
        
        return self.history
    
    def predict(self, t: np.ndarray) -> np.ndarray:
        """Predict x(t) using HNN dynamics integration - LIKE STANDALONE HNN"""
        # Use the same approach as standalone HNN: integrate learned dynamics
        if len(t) == 0:
            return np.array([])
        
        # Get initial conditions from the trajectory network at t=0
        t0_tensor = torch.tensor([t[0]], dtype=torch.float32, requires_grad=True)
        x0, v0, _ = self.model(t0_tensor)
        x0_val = x0.item()
        v0_val = v0.item()
        
        # Integrate using learned HNN dynamics (like standalone HNN)
        t_span = (t[0], t[-1])
        t_int, x_int, v_int = self.integrate_trajectory(x0_val, v0_val, t_span, len(t))
        
        # Interpolate to requested time points
        from scipy.interpolate import interp1d
        if len(t_int) > 1:
            f_interp = interp1d(t_int, x_int, kind='linear', bounds_error=False, fill_value='extrapolate')
            x_pred = f_interp(t)
        else:
            x_pred = np.full_like(t, x_int[0])
        
        return x_pred
    
    def get_derivatives(self, t, order=2):
        """Delegate to the underlying model's get_derivatives method."""
        return self.model.get_derivatives(t, order=order)
    
    def integrate_trajectory(self, q0: float, p0: float, t_span: Tuple[float, float], 
                           n_points: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate trajectory using learned dynamics"""
        from scipy.integrate import solve_ivp
        
        def cascaded_dynamics(t, y):
            """Dynamics function for scipy integration"""
            q_val, p_val = y[0], y[1]
            
            # Use HNN component for dynamics
            dq_dt, dp_dt = self.model.predict_dynamics(
                torch.tensor([[q_val]]), torch.tensor([[p_val]]))
            return [dq_dt[0, 0].item(), dp_dt[0, 0].item()]
        
        # Use scipy's RK45 integrator
        sol = solve_ivp(cascaded_dynamics, t_span, [q0, p0], 
                       t_eval=np.linspace(*t_span, n_points), 
                       method='RK45', rtol=1e-8)
        
        if sol.success:
            return sol.t, sol.y[0], sol.y[1]  # t, q, p
        else:
            print("⚠️ Cascaded HNN integration failed, falling back to Euler")
            # Fallback to Euler
            t = np.linspace(*t_span, n_points)
            dt = t[1] - t[0]
            
            q = np.zeros(n_points)
            p = np.zeros(n_points)
            q[0] = q0
            p[0] = p0
            
            # Euler integration
            for i in range(1, n_points):
                q_tensor = torch.tensor([[q[i-1]]])
                p_tensor = torch.tensor([[p[i-1]]])
                dq_dt, dp_dt = self.model.predict_dynamics(q_tensor, p_tensor)
                q[i] = q[i-1] + dt * dq_dt[0, 0].item()
                p[i] = p[i-1] + dt * dp_dt[0, 0].item()
            
            return t, q, p
    
    def save_model(self, filename: str):
        """Save the trained model"""
        torch.save(self.model.state_dict(), f'models/saved/{filename}.pth')
        print(f"Model saved to models/saved/{filename}.pth")
    
    def load_model(self, filename: str):
        """Load a trained model"""
        self.model.load_state_dict(torch.load(f'models/saved/{filename}.pth'))
        print(f"Model loaded from models/saved/{filename}.pth")
    
    def plot_training(self):
        """Plot training history with validation losses"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses with validation
        axes[0, 0].plot(self.history['trajectory_loss'], 'b-', label='Train Trajectory Loss')
        axes[0, 0].plot(self.history['val_trajectory_loss'], 'r--', label='Val Trajectory Loss')
        axes[0, 0].set_title('Trajectory Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.history['hnn_loss'], 'b-', label='Train HNN Loss')
        axes[0, 1].plot(self.history['val_hnn_loss'], 'r--', label='Val HNN Loss')
        axes[0, 1].set_title('HNN Loss')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.history['energy_loss'], 'b-', label='Train Energy Loss')
        axes[1, 0].plot(self.history['val_energy_loss'], 'r--', label='Val Energy Loss')
        axes[1, 0].set_title('Energy Loss')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(self.history['total_loss'], 'b-', label='Train Total Loss')
        axes[1, 1].plot(self.history['val_total_loss'], 'r--', label='Val Total Loss')
        axes[1, 1].set_title('Total Loss')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/cascaded_hnn_training.png', dpi=300, bbox_inches='tight')
        # plt.show() 