#!/usr/bin/env python3
"""
Sequential Cascaded HNN: Train trajectory network first, then HNN component
Two-stage training: Stage 1 (t -> x), Stage 2 (x,v -> H)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from scipy.integrate import solve_ivp
from models.hnn import HNN


class SequentialCascadedHNN(nn.Module):
    """Sequential Cascaded HNN with two-stage training"""
    
    def __init__(self, trajectory_config: Dict, hnn_config: Dict):
        super(SequentialCascadedHNN, self).__init__()
        
        # Trajectory network: t → x
        layers = []
        input_dim = 1  # time t
        
        for hidden_dim in trajectory_config['hidden_layers']:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if trajectory_config['activation'] == 'tanh':
                layers.append(nn.Tanh())
            elif trajectory_config['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif trajectory_config['activation'] == 'sine':
                layers.append(nn.SiLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))  # x(t)
        self.trajectory_net = nn.Sequential(*layers)
        
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
        if not t.requires_grad:
            t = t.requires_grad_(True)
        
        # Step 1: Predict trajectory
        x = self.trajectory_net(t.unsqueeze(-1)).squeeze(-1)
        
        # Step 2: Compute velocity via autograd
        v = torch.autograd.grad(x.sum(), t, create_graph=True, retain_graph=True)[0]
        
        # Step 3: HNN on learned trajectory
        H = self.hnn_net(x.unsqueeze(-1), v.unsqueeze(-1))
        
        return x, v, H
    
    def predict_trajectory(self, t):
        """Predict only trajectory (for stage 1 training)"""
        return self.trajectory_net(t.unsqueeze(-1)).squeeze(-1)
    
    def get_derivatives(self, t, order=2):
        """Get derivatives for symbolic regression"""
        t_tensor = torch.tensor(t, dtype=torch.float32, requires_grad=True)
        x = self.trajectory_net(t_tensor.unsqueeze(-1)).squeeze(-1)
        
        derivatives = [x.detach().numpy()]
        
        if order >= 1:
            dx_dt = torch.autograd.grad(x.sum(), t_tensor, create_graph=True)[0]
            derivatives.append(dx_dt.detach().numpy())
        
        if order >= 2:
            d2x_dt2 = torch.autograd.grad(dx_dt.sum(), t_tensor, create_graph=True)[0]
            derivatives.append(d2x_dt2.detach().numpy())
        
        return derivatives if order > 1 else derivatives[:order+1]
    
    def predict_dynamics(self, x, v):
        """Get dynamics from HNN"""
        x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True) if not isinstance(x, torch.Tensor) else x.float()
        v_tensor = torch.tensor(v, dtype=torch.float32, requires_grad=True) if not isinstance(v, torch.Tensor) else v.float()
        
        dq_dt, dp_dt = self.hnn_net.get_dynamics(x_tensor, v_tensor)
        return dq_dt, dp_dt
    
    def predict_hamiltonian(self, x, v):
        """Get Hamiltonian from HNN"""
        x_tensor = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
        v_tensor = torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v
        
        H = self.hnn_net.forward(x_tensor, v_tensor)
        return H


class SequentialCascadedHNNTrainer:
    """Sequential trainer for cascaded HNN"""
    
    def __init__(self, model: SequentialCascadedHNN, 
                 trajectory_config: Dict,
                 hnn_config: Dict,
                 energy_weight: float = 0.1):
        
        self.model = model
        self.energy_weight = energy_weight
        
        # Stage 1: Trajectory optimizer
        if trajectory_config['optimizer'] == 'adam':
            self.trajectory_optimizer = optim.Adam(model.trajectory_net.parameters(), 
                                                 lr=trajectory_config['learning_rate'], 
                                                 weight_decay=trajectory_config['weight_decay'])
        elif trajectory_config['optimizer'] == 'adamw':
            self.trajectory_optimizer = optim.AdamW(model.trajectory_net.parameters(), 
                                                  lr=trajectory_config['learning_rate'], 
                                                  weight_decay=trajectory_config['weight_decay'])
        
        # Stage 2: HNN optimizer
        if hnn_config['optimizer'] == 'adam':
            self.hnn_optimizer = optim.Adam(model.hnn_net.parameters(), 
                                          lr=hnn_config['learning_rate'], 
                                          weight_decay=hnn_config['weight_decay'])
        elif hnn_config['optimizer'] == 'adamw':
            self.hnn_optimizer = optim.AdamW(model.hnn_net.parameters(), 
                                           lr=hnn_config['learning_rate'], 
                                           weight_decay=hnn_config['weight_decay'])
        
        # Schedulers
        if trajectory_config['scheduler'] == 'steplr':
            step_size = trajectory_config['scheduler_params'].get('step_size', 1000)
            gamma = trajectory_config['scheduler_params'].get('gamma', 0.9)
            self.trajectory_scheduler = optim.lr_scheduler.StepLR(self.trajectory_optimizer, step_size=step_size, gamma=gamma)
        else:
            self.trajectory_scheduler = None
            
        if hnn_config['scheduler'] == 'steplr':
            step_size = hnn_config['scheduler_params'].get('step_size', 1000)
            gamma = hnn_config['scheduler_params'].get('gamma', 0.9)
            self.hnn_scheduler = optim.lr_scheduler.StepLR(self.hnn_optimizer, step_size=step_size, gamma=gamma)
        else:
            self.hnn_scheduler = None
        
        self.stage1_history = {'train_loss': [], 'val_loss': []}
        self.stage2_history = {'dynamics_loss': [], 'energy_loss': [], 'total_loss': [], 
                              'val_dynamics_loss': [], 'val_energy_loss': [], 'val_total_loss': []}
    
    def train_stage1(self, t: np.ndarray, x: np.ndarray, 
                     epochs: int = 5000, val_split: float = 0.2, 
                     patience: int = 1000):
        """Stage 1: Train trajectory network (t -> x)"""
        
        print("🔥 STAGE 1: Training Trajectory Network (t → x)")
        
        # Convert to tensors
        t_tensor = torch.tensor(t, dtype=torch.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        
        # Split data
        n_train = int(len(t) * (1 - val_split))
        t_train, x_train = t_tensor[:n_train], x_tensor[:n_train]
        t_val, x_val = t_tensor[n_train:], x_tensor[n_train:]
        
        print(f"Stage 1: {len(t_train)} train, {len(t_val)} val samples")
        
        # Freeze HNN
        for param in self.model.hnn_net.parameters():
            param.requires_grad = False
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            
            x_pred = self.model.predict_trajectory(t_train)
            train_loss = torch.mean((x_pred - x_train) ** 2)
            
            self.trajectory_optimizer.zero_grad()
            train_loss.backward()
            self.trajectory_optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                x_val_pred = self.model.predict_trajectory(t_val)
                val_loss = torch.mean((x_val_pred - x_val) ** 2)
            
            # Store history
            self.stage1_history['train_loss'].append(train_loss.item())
            self.stage1_history['val_loss'].append(val_loss.item())
            
            # Scheduler
            if self.trajectory_scheduler is not None:
                self.trajectory_scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/saved/sequential_cascaded_stage1_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}, best val loss: {best_val_loss:.6f}")
                    break
            
            if epoch % 500 == 0:
                print(f"Stage 1 Epoch {epoch}: Train={train_loss.item():.6f}, Val={val_loss.item():.6f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('models/saved/sequential_cascaded_stage1_best.pth'))
        print("✅ Stage 1 completed!")
    
    def train_stage2(self, t: np.ndarray, x: np.ndarray, v: np.ndarray,
                     epochs: int = 5000, val_split: float = 0.2, 
                     patience: int = 1000):
        """Stage 2: Train HNN component (x,v -> H)"""
        
        print("🔥 STAGE 2: Training HNN Component (x,v → H)")
        
        # Freeze trajectory network
        for param in self.model.trajectory_net.parameters():
            param.requires_grad = False
        
        # Unfreeze HNN
        for param in self.model.hnn_net.parameters():
            param.requires_grad = True
        
        # Prepare HNN data
        from utils.data_utils import prepare_hnn_data
        q, p, dq_dt, dp_dt = prepare_hnn_data(t, x, v)
        
        # Convert to tensors
        q_tensor = torch.tensor(q, dtype=torch.float32)
        p_tensor = torch.tensor(p, dtype=torch.float32)
        dq_dt_tensor = torch.tensor(dq_dt, dtype=torch.float32)
        dp_dt_tensor = torch.tensor(dp_dt, dtype=torch.float32)
        
        # Split data
        n_train = int(len(q) * (1 - val_split))
        q_train, p_train = q_tensor[:n_train], p_tensor[:n_train]
        dq_dt_train, dp_dt_train = dq_dt_tensor[:n_train], dp_dt_tensor[:n_train]
        q_val, p_val = q_tensor[n_train:], p_tensor[n_train:]
        dq_dt_val, dp_dt_val = dq_dt_tensor[n_train:], dp_dt_tensor[n_train:]
        
        print(f"Stage 2: {len(q_train)} train, {len(q_val)} val samples")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            
            # Get dynamics from HNN
            q_train.requires_grad_(True)
            p_train.requires_grad_(True)
            dq_dt_pred, dp_dt_pred = self.model.hnn_net.get_dynamics(q_train, p_train)
            
            # Dynamics loss
            dynamics_loss = torch.mean((dq_dt_pred - dq_dt_train) ** 2 + 
                                     (dp_dt_pred - dp_dt_train) ** 2)
            
            # Energy loss
            H_pred = self.model.hnn_net(q_train, p_train)
            energy_loss = torch.var(H_pred)
            
            total_loss = dynamics_loss + self.energy_weight * energy_loss
            
            self.hnn_optimizer.zero_grad()
            total_loss.backward()
            self.hnn_optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.enable_grad():
                q_val.requires_grad_(True)
                p_val.requires_grad_(True)
                dq_dt_val_pred, dp_dt_val_pred = self.model.hnn_net.get_dynamics(q_val, p_val)
                val_dynamics_loss = torch.mean((dq_dt_val_pred - dq_dt_val) ** 2 + 
                                             (dp_dt_val_pred - dp_dt_val) ** 2)
                
                H_val_pred = self.model.hnn_net(q_val, p_val)
                val_energy_loss = torch.var(H_val_pred)
                val_total_loss = val_dynamics_loss + self.energy_weight * val_energy_loss
            
            # Store history
            self.stage2_history['dynamics_loss'].append(dynamics_loss.item())
            self.stage2_history['energy_loss'].append(energy_loss.item())
            self.stage2_history['total_loss'].append(total_loss.item())
            self.stage2_history['val_dynamics_loss'].append(val_dynamics_loss.item())
            self.stage2_history['val_energy_loss'].append(val_energy_loss.item())
            self.stage2_history['val_total_loss'].append(val_total_loss.item())
            
            # Scheduler
            if self.hnn_scheduler is not None:
                self.hnn_scheduler.step()
            
            # Early stopping
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/saved/sequential_cascaded_stage2_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}, best val loss: {best_val_loss:.6f}")
                    break
            
            if epoch % 500 == 0:
                print(f"Stage 2 Epoch {epoch}: Dynamics={dynamics_loss.item():.6f}, "
                      f"Energy={energy_loss.item():.6f}, Total={total_loss.item():.6f}, "
                      f"Val={val_total_loss.item():.6f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('models/saved/sequential_cascaded_stage2_best.pth'))
        print("✅ Stage 2 completed!")
    
    def train(self, t: np.ndarray, x: np.ndarray, v: np.ndarray, 
              stage1_epochs: int = 5000, stage2_epochs: int = 5000,
              val_split: float = 0.2, patience: int = 1000):
        """Full sequential training: Stage 1 then Stage 2"""
        
        print("🚀 SEQUENTIAL CASCADED HNN TRAINING")
        
        # Stage 1: Train trajectory network
        self.train_stage1(t, x, stage1_epochs, val_split, patience)
        
        # Stage 2: Train HNN component
        self.train_stage2(t, x, v, stage2_epochs, val_split, patience)
        
        print("✅ Sequential training completed!")
        
        return {'stage1': self.stage1_history, 'stage2': self.stage2_history}
    
    def predict(self, t: np.ndarray) -> np.ndarray:
        """Predict using HNN dynamics integration"""
        if len(t) == 0:
            return np.array([])
        
        # Get initial conditions from trajectory network
        t0_tensor = torch.tensor([t[0]], dtype=torch.float32, requires_grad=True)
        x0, v0, _ = self.model(t0_tensor)
        x0_val = x0.item()
        v0_val = v0.item()
        
        # Integrate using HNN dynamics
        t_span = (t[0], t[-1])
        t_int, x_int, v_int = self.integrate_trajectory(x0_val, v0_val, t_span, len(t))
        
        # Interpolate
        from scipy.interpolate import interp1d
        if len(t_int) > 1:
            f_interp = interp1d(t_int, x_int, kind='linear', bounds_error=False, fill_value='extrapolate')
            x_pred = f_interp(t)
        else:
            x_pred = np.full_like(t, x_int[0])
        
        return x_pred
    
    def integrate_trajectory(self, q0: float, p0: float, t_span: Tuple[float, float], 
                           n_points: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate trajectory using learned HNN dynamics"""
        
        def sequential_dynamics(t, y):
            q_val, p_val = y[0], y[1]
            dq_dt, dp_dt = self.model.predict_dynamics(
                torch.tensor([[q_val]]), torch.tensor([[p_val]]))
            return [dq_dt[0, 0].item(), dp_dt[0, 0].item()]
        
        # Use scipy's RK45 integrator
        sol = solve_ivp(sequential_dynamics, t_span, [q0, p0], 
                       t_eval=np.linspace(*t_span, n_points), 
                       method='RK45', rtol=1e-8)
        
        if sol.success:
            return sol.t, sol.y[0], sol.y[1]
        else:
            print("⚠️ Sequential HNN integration failed, falling back to Euler")
            # Fallback to Euler
            t = np.linspace(*t_span, n_points)
            dt = t[1] - t[0]
            
            q = np.zeros(n_points)
            p = np.zeros(n_points)
            q[0] = q0
            p[0] = p0
            
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
        """Plot training history for both stages"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Stage 1 plots
        axes[0, 0].plot(self.stage1_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(self.stage1_history['val_loss'], 'r--', label='Val Loss')
        axes[0, 0].set_title('Stage 1: Trajectory Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Stage 2 plots
        axes[0, 1].plot(self.stage2_history['dynamics_loss'], 'b-', label='Train Dynamics')
        axes[0, 1].plot(self.stage2_history['val_dynamics_loss'], 'r--', label='Val Dynamics')
        axes[0, 1].set_title('Stage 2: Dynamics Loss')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(self.stage2_history['energy_loss'], 'b-', label='Train Energy')
        axes[0, 2].plot(self.stage2_history['val_energy_loss'], 'r--', label='Val Energy')
        axes[0, 2].set_title('Stage 2: Energy Loss')
        axes[0, 2].set_yscale('log')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.stage2_history['total_loss'], 'b-', label='Train Total')
        axes[1, 0].plot(self.stage2_history['val_total_loss'], 'r--', label='Val Total')
        axes[1, 0].set_title('Stage 2: Total Loss')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary plots
        axes[1, 1].text(0.1, 0.8, f"Stage 1 Final:\nTrain: {self.stage1_history['train_loss'][-1]:.6f}\nVal: {self.stage1_history['val_loss'][-1]:.6f}", 
                       transform=axes[1, 1].transAxes, fontsize=12, verticalalignment='top')
        axes[1, 1].text(0.1, 0.4, f"Stage 2 Final:\nDynamics: {self.stage2_history['dynamics_loss'][-1]:.6f}\nEnergy: {self.stage2_history['energy_loss'][-1]:.6f}", 
                       transform=axes[1, 1].transAxes, fontsize=12, verticalalignment='top')
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        axes[1, 2].axis('off')  # Unused subplot
        
        plt.tight_layout()
        plt.savefig('plots/sequential_cascaded_hnn_training.png', dpi=300, bbox_inches='tight')
        # plt.show()
    
    def get_derivatives(self, t, order=2):
        """Delegate to the underlying model's get_derivatives method."""
        return self.model.get_derivatives(t, order=order) 