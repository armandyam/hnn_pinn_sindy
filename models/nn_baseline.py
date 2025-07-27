import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import os

class BaselineNN(nn.Module):
    """Baseline neural network that simply fits the data without physics constraints"""
    
    def __init__(self, input_dim: int = 1, hidden_layers: list = [64, 32, 16], 
                 output_dim: int = 1, activation: str = 'tanh'):
        super(BaselineNN, self).__init__()
        
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

class BaselineTrainer:
    """Trainer for baseline neural network"""
    
    def __init__(self, model: BaselineNN, lr: float = 1e-3, optimizer_type: str = 'adam', 
                 weight_decay: float = 1e-5, scheduler_type: str = None, scheduler_params: dict = None):
        self.model = model
        
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
        
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
    
    def train(self, t_data: np.ndarray, x_data: np.ndarray, 
              epochs: int = 1000, val_split: float = 0.2) -> Dict[str, list]:
        """Train the baseline neural network"""
        
        # Convert to tensors
        t_tensor = torch.FloatTensor(t_data.reshape(-1, 1))
        x_tensor = torch.FloatTensor(x_data.reshape(-1, 1))
        
        # Split into train/val
        n_train = int(len(t_data) * (1 - val_split))
        t_train = t_tensor[:n_train]
        x_train = x_tensor[:n_train]
        t_val = t_tensor[n_train:]
        x_val = x_tensor[n_train:]
        
        print("Training baseline neural network...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            self.optimizer.zero_grad()
            
            x_pred = self.model(t_train)
            train_loss = self.criterion(x_pred, x_train)
            
            train_loss.backward()
            self.optimizer.step()
            
            # Step scheduler if available
            if self.scheduler:
                self.scheduler.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                x_val_pred = self.model(t_val)
                val_loss = self.criterion(x_val_pred, x_val)
            
            self.train_losses.append(train_loss.item())
            self.val_losses.append(val_loss.item())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss.item():.6f}, Val Loss = {val_loss.item():.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
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
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Baseline NN Training Curves')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            os.makedirs('plots', exist_ok=True)
            plt.savefig('plots/baseline_nn_training.png', dpi=300, bbox_inches='tight')
            print("Training plot saved to plots/baseline_nn_training.png")
        
        plt.show()

def main():
    """Test the baseline neural network"""
    # Load data
    import pandas as pd
    data = pd.read_csv('data/damped_oscillator.csv')
    
    # Create and train model
    model = BaselineNN()
    trainer = BaselineTrainer(model)
    
    # Train
    history = trainer.train(data['t'].values, data['x'].values, epochs=1000)
    
    # Plot training
    trainer.plot_training()
    
    # Save model
    trainer.save_model('baseline_nn_oscillator')
    
    # Test predictions
    t_test = np.linspace(0, 30, 300)
    x_pred = trainer.predict(t_test)
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['t'], data['x'], 'b.', label='Training data', alpha=0.7)
    plt.plot(t_test, x_pred, 'r-', label='NN prediction', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Baseline NN Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main() 