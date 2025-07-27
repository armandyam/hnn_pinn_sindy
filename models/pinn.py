import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import os

class PINN(nn.Module):
    """Physics-Informed Neural Network"""
    
    def __init__(self, input_dim: int = 1, hidden_layers: list = [64, 32, 16], 
                 output_dim: int = 1, activation: str = 'tanh'):
        super(PINN, self).__init__()
        
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
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class PINNTrainer:
    """Trainer for Physics-Informed Neural Network"""
    
    def __init__(self, model: PINN, lr: float = 1e-3, physics_weight: float = 1.0,
                 optimizer_type: str = 'adam', weight_decay: float = 1e-5, 
                 scheduler_type: str = None, scheduler_params: dict = None):
        self.model = model
        self.physics_weight = physics_weight
        
        # Choose optimizer
        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Choose scheduler
        if scheduler_type == 'steplr' and scheduler_params:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=scheduler_params.get('step_size', 1000),
                gamma=scheduler_params.get('gamma', 0.9)
            )
        else:
            self.scheduler = None
        
        self.train_losses = []
        self.val_losses = []
        self.physics_losses = []
    
    def compute_physics_loss(self, t: torch.Tensor, system: str = 'damped_oscillator') -> torch.Tensor:
        """Compute physics-based loss using automatic differentiation"""
        t.requires_grad_(True)
        
        # Forward pass
        x = self.model(t)
        
        # First derivative
        dx_dt = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
        
        # Second derivative
        d2x_dt2 = torch.autograd.grad(dx_dt.sum(), t, create_graph=True)[0]
        
        if system == 'damped_oscillator':
            # Import config parameters instead of hardcoding
            from config import SYSTEMS
            params = SYSTEMS['damped_oscillator']['parameters']
            m, k, c = params['m'], params['k'], params['c']
            
            # Physics constraint: m*ddx + c*dx + k*x = 0
            # For damped oscillator: ddot(x) + (c/m)*dot(x) + (k/m)*x = 0
            physics_residual = d2x_dt2 + (c/m)*dx_dt + (k/m)*x
            return torch.mean(physics_residual**2)
        
        elif system == 'pendulum':
            # Import config parameters instead of hardcoding
            from config import SYSTEMS
            params = SYSTEMS['pendulum']['parameters']
            g, L, c = params['g'], params['L'], params['c']
            
            # Physics constraint: ddot(theta) + (c/m)*dot(theta) + (g/L)*sin(theta) = 0
            physics_residual = d2x_dt2 + c*dx_dt + (g/L)*torch.sin(x)
            return torch.mean(physics_residual**2)
        
        else:
            raise ValueError(f"Unknown system: {system}")
    
    def train(self, t: np.ndarray, x: np.ndarray, epochs: int = 1000, 
              val_split: float = 0.2, system: str = 'damped_oscillator', verbose: bool = True):
        """Train PINN with physics loss and early stopping"""
        # Convert to tensors
        t_tensor = torch.FloatTensor(t.reshape(-1, 1))
        x_tensor = torch.FloatTensor(x.reshape(-1, 1))
        
        # Split data
        n_val = int(len(t) * val_split)
        indices = np.random.permutation(len(t))
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        t_train, x_train = t_tensor[train_indices], x_tensor[train_indices]
        t_val, x_val = t_tensor[val_indices], x_tensor[val_indices]
        
        # Early stopping parameters - MORE PATIENCE FOR PHYSICS CONVERGENCE
        best_val_loss = float('inf')
        patience = 5000  # Much higher patience for PINN physics convergence
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            self.optimizer.zero_grad()
            
            # Data loss
            pred = self.model(t_train)
            data_loss = nn.MSELoss()(pred, x_train)
            
            # Physics loss
            physics_loss = self.compute_physics_loss(t_train, system)
            
            # Total loss
            total_loss = data_loss + self.physics_weight * physics_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            # Step scheduler if available
            if self.scheduler:
                self.scheduler.step()
            
            # Validation - compute TOTAL validation loss (data + physics)
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(t_val)
                val_data_loss = nn.MSELoss()(val_pred, x_val)
            
            # For physics loss, we need gradients enabled
            val_physics_loss = self.compute_physics_loss(t_val, system)
            val_total_loss = val_data_loss + self.physics_weight * val_physics_loss
            
            # Store losses
            self.train_losses.append(data_loss.item())
            self.val_losses.append(val_data_loss.item())  # Store data loss for plotting
            self.physics_losses.append(physics_loss.item())
            
            # Early stopping check (use TOTAL validation loss, not just data loss)
            if val_total_loss.item() < best_val_loss:
                best_val_loss = val_total_loss.item()
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Stop if total validation loss hasn't improved
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}, best total val loss: {best_val_loss:.6f}")
                # Restore best model
                self.model.load_state_dict(best_model_state)
                break
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Data Loss = {data_loss.item():.6f}, Physics Loss = {physics_loss.item():.6f}, Val Loss = {val_data_loss.item():.6f}")
    
    def predict(self, t: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            t_tensor = torch.FloatTensor(t.reshape(-1, 1))
            predictions = self.model(t_tensor).numpy().flatten()
        return predictions
    
    def get_derivatives(self, t: np.ndarray, order: int = 2) -> Tuple[np.ndarray, ...]:
        """Compute derivatives using autograd"""
        self.model.eval()
        t_tensor = torch.FloatTensor(t.reshape(-1, 1))
        t_tensor.requires_grad_(True)
        
        x = self.model(t_tensor)
        
        # First derivative
        dx_dt = torch.autograd.grad(x.sum(), t_tensor, create_graph=True)[0]
        
        derivatives = [x.detach().numpy().flatten(), dx_dt.detach().numpy().flatten()]
        
        if order >= 2:
            # Second derivative
            d2x_dt2 = torch.autograd.grad(dx_dt.sum(), t_tensor, create_graph=True)[0]
            derivatives.append(d2x_dt2.detach().numpy().flatten())
        
        return tuple(derivatives)
    
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
        ax1.plot(self.train_losses, label='Data Loss', alpha=0.7)
        ax1.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        # ax1.plot(self.physics_losses, label='Physics Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('PINN Training Curves')
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Loss comparison
        ax2.plot(self.train_losses, label='Data Loss', alpha=0.7)
        ax2.plot(self.physics_losses, label='Physics Loss', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Data vs Physics Loss')
        ax2.legend()
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            os.makedirs('plots', exist_ok=True)
            plt.savefig('plots/pinn_training.png', dpi=300, bbox_inches='tight')
            print("Training plot saved to plots/pinn_training.png")
        
        # plt.show()

def main():
    """Test the PINN"""
    # Load data
    import pandas as pd
    data = pd.read_csv('data/damped_oscillator.csv')
    
    # Create and train model
    model = PINN()
    trainer = PINNTrainer(model, physics_weight=1.0)
    
    # Train
    history = trainer.train(data['t'].values, data['x'].values, 
                          epochs=1000, system='damped_oscillator')
    
    # Plot training
    trainer.plot_training()
    
    # Save model
    trainer.save_model('pinn_oscillator')
    
    # Test predictions
    t_test = np.linspace(0, 30, 300)
    x_pred = trainer.predict(t_test)
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['t'], data['x'], 'b.', label='Training data', alpha=0.7)
    plt.plot(t_test, x_pred, 'r-', label='PINN prediction', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('PINN Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()

if __name__ == "__main__":
    main() 