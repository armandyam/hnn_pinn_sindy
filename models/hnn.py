import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import os

class HNN(nn.Module):
    """Hamiltonian Neural Network that learns H(q,p)"""
    
    def __init__(self, input_dim: int = 2, hidden_layers: list = [64, 32, 16], 
                 activation: str = 'tanh', normalize_inputs: bool = True):
        super(HNN, self).__init__()
        
        self.normalize_inputs = normalize_inputs
        
        # Build network dynamically from hidden_layers list
        layers = []
        prev_dim = input_dim
        
        # Choose activation function
        if activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'sine':
            act_fn = lambda x: torch.sin(x)
        else:
            act_fn = nn.Tanh()  # Default to tanh
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn
            ])
            prev_dim = hidden_dim
        
        # Add output layer (Hamiltonian is scalar)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, q, p):
        """Compute Hamiltonian H(q,p)"""
        inputs = torch.cat([q, p], dim=1)
        return self.network(inputs)
    
    def get_dynamics(self, q, p):
        """Compute dynamics: dq/dt = dH/dp, dp/dt = -dH/dq"""
        q.requires_grad_(True)
        p.requires_grad_(True)
        
        H = self.forward(q, p)
        
        # Compute gradients
        dH_dq = torch.autograd.grad(H.sum(), q, create_graph=True)[0]
        dH_dp = torch.autograd.grad(H.sum(), p, create_graph=True)[0]
        
        # Hamilton's equations
        dq_dt = dH_dp
        dp_dt = -dH_dq
        
        return dq_dt, dp_dt

class HNNTrainer:
    """Trainer for Hamiltonian Neural Network"""
    
    def __init__(self, model: HNN, lr: float = 5e-4, energy_weight: float = 1.0):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=500)
        self.energy_weight = energy_weight
        self.train_losses = []
        self.val_losses = []
        self.energy_losses = []
    
    def train(self, q_data: np.ndarray, p_data: np.ndarray, 
              dq_dt_data: np.ndarray, dp_dt_data: np.ndarray,
              epochs: int = 1000, val_split: float = 0.2) -> Dict[str, list]:
        """Train the HNN"""
        
        # Convert to tensors
        q_tensor = torch.FloatTensor(q_data.reshape(-1, 1))
        p_tensor = torch.FloatTensor(p_data.reshape(-1, 1))
        dq_dt_tensor = torch.FloatTensor(dq_dt_data.reshape(-1, 1))
        dp_dt_tensor = torch.FloatTensor(dp_dt_data.reshape(-1, 1))
        
        # Split into train/val
        n_train = int(len(q_data) * (1 - val_split))
        q_train = q_tensor[:n_train]
        p_train = p_tensor[:n_train]
        dq_dt_train = dq_dt_tensor[:n_train]
        dp_dt_train = dp_dt_tensor[:n_train]
        
        q_val = q_tensor[n_train:]
        p_val = p_tensor[n_train:]
        dq_dt_val = dq_dt_tensor[n_train:]
        dp_dt_val = dp_dt_tensor[n_train:]
        
        # Set requires_grad for training tensors
        q_train.requires_grad_(True)
        p_train.requires_grad_(True)
        
        # Set requires_grad for validation tensors
        q_val.requires_grad_(True)
        p_val.requires_grad_(True)
        
        print("Training Hamiltonian Neural Network...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            self.optimizer.zero_grad()
            
            # Predict dynamics
            dq_dt_pred, dp_dt_pred = self.model.get_dynamics(q_train, p_train)
            
            # Dynamics loss
            dynamics_loss = torch.mean((dq_dt_pred - dq_dt_train)**2 + 
                                     (dp_dt_pred - dp_dt_train)**2)
            
            # Energy conservation loss
            H_values = self.model(q_train, p_train)
            energy_loss = torch.var(H_values)  # Variance should be small for conservation
            
            # Total loss
            total_loss = dynamics_loss + self.energy_weight * energy_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            # Validation
            self.model.eval()
            dq_dt_val_pred, dp_dt_val_pred = self.model.get_dynamics(q_val, p_val)
            val_loss = torch.mean((dq_dt_val_pred - dq_dt_val)**2 + 
                                (dp_dt_val_pred - dp_dt_val)**2)
            
            self.train_losses.append(dynamics_loss.item())
            self.val_losses.append(val_loss.item())
            self.energy_losses.append(energy_loss.item())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Dynamics Loss = {dynamics_loss.item():.6f}, "
                      f"Energy Loss = {energy_loss.item():.6f}, "
                      f"Val Loss = {val_loss.item():.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'energy_losses': self.energy_losses
        }
    
    def predict_dynamics(self, q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict dynamics for given q, p"""
        self.model.eval()
        q_tensor = torch.FloatTensor(q.reshape(-1, 1))
        p_tensor = torch.FloatTensor(p.reshape(-1, 1))
        # Set requires_grad for gradient computation
        q_tensor.requires_grad_(True)
        p_tensor.requires_grad_(True)
        dq_dt_pred, dp_dt_pred = self.model.get_dynamics(q_tensor, p_tensor)
        return dq_dt_pred.detach().numpy().flatten(), dp_dt_pred.detach().numpy().flatten()
    
    def predict_hamiltonian(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Predict Hamiltonian values"""
        self.model.eval()
        q_tensor = torch.FloatTensor(q.reshape(-1, 1))
        p_tensor = torch.FloatTensor(p.reshape(-1, 1))
        H = self.model(q_tensor, p_tensor)
        return H.detach().numpy().flatten()
    
    def get_derivatives(self, t: np.ndarray, order: int = 2) -> Tuple[np.ndarray, ...]:
        """Get derivatives for symbolic regression - HNN doesn't predict time series directly"""
        # For HNN, we need to reconstruct the time series from the learned dynamics
        # This is a simplified approach - in practice, you might want to integrate the learned dynamics
        
        # Use the first few points to estimate derivatives
        n_points = min(len(t), 100)  # Use fewer points for efficiency
        t_subset = t[:n_points]
        
        # Create a simple trajectory using the learned dynamics
        q0, p0 = 1.0, 0.0  # Initial conditions
        dt = t_subset[1] - t_subset[0] if len(t_subset) > 1 else 0.1
        
        q = np.zeros(n_points)
        p = np.zeros(n_points)
        q[0] = q0
        p[0] = p0
        
        # Integrate using learned dynamics
        for i in range(1, n_points):
            dq_dt, dp_dt = self.predict_dynamics(np.array([q[i-1]]), np.array([p[i-1]]))
            q[i] = q[i-1] + dt * dq_dt[0]
            p[i] = p[i-1] + dt * dp_dt[0]
        
        # Compute derivatives using finite differences
        dx_dt = np.gradient(q, dt)
        d2x_dt2 = np.gradient(dx_dt, dt)
        
        return q, dx_dt, d2x_dt2
    
    def integrate_trajectory(self, q0: float, p0: float, t_span: Tuple[float, float], 
                           n_points: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate trajectory using learned dynamics"""
        t = np.linspace(*t_span, n_points)
        dt = t[1] - t[0]
        
        q = np.zeros(n_points)
        p = np.zeros(n_points)
        q[0] = q0
        p[0] = p0
        
        # Euler integration
        for i in range(1, n_points):
            dq_dt, dp_dt = self.predict_dynamics(np.array([q[i-1]]), np.array([p[i-1]]))
            q[i] = q[i-1] + dt * dq_dt[0]
            p[i] = p[i-1] + dt * dp_dt[0]
        
        return t, q, p
    
    def save_model(self, filename: str):
        """Save the trained model"""
        os.makedirs('models/saved', exist_ok=True)
        torch.save(self.model.state_dict(), f'models/saved/{filename}.pth')
        print(f"Model saved to models/saved/{filename}.pth")
    
    def load_model(self, filename: str):
        """Load a trained model"""
        self.model.load_state_dict(torch.load(f'models/saved/{filename}.pth'))
        print(f"Model loaded from models/saved/{filename}.pth")
    
    def plot_training(self, save_plot: bool = True):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Dynamics Loss', alpha=0.7)
        ax1.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        ax1.plot(self.energy_losses, label='Energy Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('HNN Training Curves')
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Loss comparison
        ax2.plot(self.train_losses, label='Dynamics Loss', alpha=0.7)
        ax2.plot(self.energy_losses, label='Energy Loss', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Dynamics vs Energy Loss')
        ax2.legend()
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            os.makedirs('plots', exist_ok=True)
            plt.savefig('plots/hnn_training.png', dpi=300, bbox_inches='tight')
            print("Training plot saved to plots/hnn_training.png")
        
        plt.show()

def prepare_hnn_data(t_data: np.ndarray, x_data: np.ndarray, v_data: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Prepare data for HNN training by computing derivatives"""
    # For damped oscillator: q = x, p = v
    q = x_data
    p = v_data
    
    # Compute derivatives using finite differences
    dt = t_data[1] - t_data[0]
    dq_dt = np.gradient(q, dt)
    dp_dt = np.gradient(p, dt)
    
    return q, p, dq_dt, dp_dt

def main():
    """Test the HNN"""
    # Load data
    import pandas as pd
    data = pd.read_csv('data/damped_oscillator.csv')
    
    # Prepare HNN data
    q, p, dq_dt, dp_dt = prepare_hnn_data(data['t'].values, data['x'].values, data['v'].values)
    
    # Create and train model
    model = HNN()
    trainer = HNNTrainer(model, energy_weight=0.1)
    
    # Train
    history = trainer.train(q, p, dq_dt, dp_dt, epochs=1000)
    
    # Plot training
    trainer.plot_training()
    
    # Save model
    trainer.save_model('hnn_oscillator')
    
    # Test trajectory integration
    t_int, q_int, p_int = trainer.integrate_trajectory(q[0], p[0], (0, 30))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(data['t'], data['x'], 'b.', label='Training data', alpha=0.7)
    plt.plot(t_int, q_int, 'r-', label='HNN prediction', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('HNN Trajectory Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    H_train = trainer.predict_hamiltonian(q, p)
    H_int = trainer.predict_hamiltonian(q_int, p_int)
    plt.plot(data['t'], H_train, 'b.', label='Training energy', alpha=0.7)
    plt.plot(t_int, H_int, 'r-', label='Predicted energy', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Hamiltonian')
    plt.title('Energy Conservation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 