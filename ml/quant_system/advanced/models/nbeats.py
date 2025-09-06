"""
N-BEATS: Neural Basis Expansion Analysis for Time Series Forecasting

This module implements N-BEATS, a state-of-the-art deep learning architecture
for time series forecasting that is interpretable and requires no feature engineering.

Reference: Oreshkin, B. N., et al. "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class NBEATSConfig:
    """Configuration for N-BEATS model."""
    
    # Model architecture
    input_size: int = 252  # Lookback window (1 year of daily data)
    output_size: int = 30  # Forecast horizon (30 days)
    hidden_size: int = 512  # Hidden layer size
    num_blocks: int = 3  # Number of blocks
    num_layers: int = 4  # Number of layers per block
    num_stacks: int = 2  # Number of stacks
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Loss function
    loss_function: str = "mae"  # "mae", "mse", "huber"
    
    # Interpretability
    interpretable: bool = True
    trend_degree: int = 2  # Polynomial degree for trend
    seasonality_periods: List[int] = None  # Seasonality periods
    
    def __post_init__(self):
        if self.seasonality_periods is None:
            self.seasonality_periods = [7, 30, 90]  # Weekly, monthly, quarterly

class NBEATSBlock(nn.Module):
    """Single N-BEATS block."""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int, 
                 num_layers: int, block_type: str = "generic"):
        super(NBEATSBlock, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.block_type = block_type
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.fc_layers.append(nn.Linear(input_size, hidden_size))
            elif i == num_layers - 1:
                self.fc_layers.append(nn.Linear(hidden_size, output_size))
            else:
                self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Basis expansion layers
        if block_type == "trend":
            self.basis = self._create_trend_basis()
        elif block_type == "seasonality":
            self.basis = self._create_seasonality_basis()
        else:  # generic
            self.basis = self._create_generic_basis()
    
    def _create_trend_basis(self):
        """Create trend basis (polynomial)."""
        basis = torch.zeros(self.output_size, self.output_size)
        for i in range(self.output_size):
            for j in range(self.output_size):
                basis[i, j] = (i / self.output_size) ** j
        return nn.Parameter(basis, requires_grad=True)
    
    def _create_seasonality_basis(self):
        """Create seasonality basis (Fourier)."""
        basis = torch.zeros(self.output_size, self.output_size)
        for i in range(self.output_size):
            for j in range(self.output_size):
                if j % 2 == 0:
                    basis[i, j] = np.cos(2 * np.pi * i * j / self.output_size)
                else:
                    basis[i, j] = np.sin(2 * np.pi * i * j / self.output_size)
        return nn.Parameter(basis, requires_grad=True)
    
    def _create_generic_basis(self):
        """Create generic basis (learned)."""
        return nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    """
                    Forward pass through N-BEATS block.
                    
                    Args:
                        x: Input tensor of shape (batch_size, input_size)
                    
                    Returns:
                        Tuple of (backcast, forecast)
                    """
                    # Fully connected layers
                    for i, layer in enumerate(self.fc_layers):
                        if i == 0:
                            h = F.relu(layer(x))
                        else:
                            h = F.relu(layer(h))
                    
                    # Basis expansion
                    if self.block_type == "generic":
                        forecast = self.basis(h)
                        backcast = self.basis(h)  # For generic, use same basis
                    else:
                        forecast = torch.matmul(h, self.basis.T)
                        backcast = torch.matmul(h, self.basis.T)
                    
                    # Ensure backcast has same shape as input
                    if backcast.shape[1] != x.shape[1]:
                        # Pad or truncate to match input size
                        if backcast.shape[1] < x.shape[1]:
                            padding = torch.zeros(backcast.shape[0], x.shape[1] - backcast.shape[1], device=backcast.device)
                            backcast = torch.cat([backcast, padding], dim=1)
                        else:
                            backcast = backcast[:, :x.shape[1]]
                    
                    return backcast, forecast

class NBEATSStack(nn.Module):
    """N-BEATS stack containing multiple blocks."""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int, 
                 num_blocks: int, num_layers: int, block_type: str = "generic"):
        super(NBEATSStack, self).__init__()
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                NBEATSBlock(input_size, output_size, hidden_size, num_layers, block_type)
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through N-BEATS stack.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (backcast, forecast)
        """
        backcast = x
        forecast = torch.zeros(x.shape[0], self.blocks[0].output_size, device=x.device)
        
        for block in self.blocks:
            block_backcast, block_forecast = block(backcast)
            backcast = backcast - block_backcast
            forecast = forecast + block_forecast
        
        return backcast, forecast

class NBEATSModel(nn.Module):
    """Complete N-BEATS model."""
    
    def __init__(self, config: NBEATSConfig):
        super(NBEATSModel, self).__init__()
        
        self.config = config
        self.stacks = nn.ModuleList()
        
        # Create stacks
        for i in range(config.num_stacks):
            if i == 0 and config.interpretable:
                # First stack: trend
                self.stacks.append(
                    NBEATSStack(config.input_size, config.output_size, 
                               config.hidden_size, config.num_blocks, 
                               config.num_layers, "trend")
                )
            elif i == 1 and config.interpretable:
                # Second stack: seasonality
                self.stacks.append(
                    NBEATSStack(config.input_size, config.output_size, 
                               config.hidden_size, config.num_blocks, 
                               config.num_layers, "seasonality")
                )
            else:
                # Generic stacks
                self.stacks.append(
                    NBEATSStack(config.input_size, config.output_size, 
                               config.hidden_size, config.num_blocks, 
                               config.num_layers, "generic")
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through N-BEATS model.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Forecast tensor of shape (batch_size, output_size)
        """
        backcast = x
        forecast = torch.zeros(x.shape[0], self.config.output_size, device=x.device)
        
        for stack in self.stacks:
            stack_backcast, stack_forecast = stack(backcast)
            backcast = stack_backcast
            forecast = forecast + stack_forecast
        
        return forecast
    
    def get_interpretable_components(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get interpretable components (trend, seasonality, residuals).
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with trend, seasonality, and residual components
        """
        if not self.config.interpretable:
            return {"forecast": self.forward(x)}
        
        components = {}
        backcast = x
        forecast = torch.zeros(x.shape[0], self.config.output_size, device=x.device)
        
        for i, stack in enumerate(self.stacks):
            stack_backcast, stack_forecast = stack(backcast)
            
            if i == 0:  # Trend stack
                components["trend"] = stack_forecast
            elif i == 1:  # Seasonality stack
                components["seasonality"] = stack_forecast
            else:  # Generic stacks
                if "residual" not in components:
                    components["residual"] = stack_forecast
                else:
                    components["residual"] += stack_forecast
            
            backcast = stack_backcast
            forecast += stack_forecast
        
        components["forecast"] = forecast
        return components

class NBEATSTrainer:
    """Trainer for N-BEATS model."""
    
    def __init__(self, config: NBEATSConfig):
        self.config = config
        self.model = NBEATSModel(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Loss function
        if config.loss_function == "mae":
            self.criterion = nn.L1Loss()
        elif config.loss_function == "mse":
            self.criterion = nn.MSELoss()
        elif config.loss_function == "huber":
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {config.loss_function}")
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Training history
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "epochs": []
        }
    
    def prepare_data(self, data: pd.Series, train_ratio: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for training.
        
        Args:
            data: Time series data
            train_ratio: Ratio of data to use for training
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        # Create sequences
        X, y = [], []
        for i in range(len(data) - self.config.input_size - self.config.output_size + 1):
            X.append(data[i:i + self.config.input_size].values)
            y.append(data[i + self.config.input_size:i + self.config.input_size + self.config.output_size].values)
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # Split into train/validation
        split_idx = int(len(X) * train_ratio)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_val, y_val
    
    def train(self, data: pd.Series, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the N-BEATS model.
        
        Args:
            data: Time series data
            verbose: Whether to print training progress
            
        Returns:
            Training results
        """
        # Prepare data
        X_train, y_train, X_val, y_val = self.prepare_data(data)
        
        # Move to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for i in range(0, len(X_train), self.config.batch_size):
                batch_X = X_train[i:i + self.config.batch_size]
                batch_y = y_train[i:i + self.config.batch_size]
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = self.criterion(val_predictions, y_val).item()
            
            # Record history
            self.training_history["train_loss"].append(train_loss / len(X_train))
            self.training_history["val_loss"].append(val_loss)
            self.training_history["epochs"].append(epoch)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_nbeats_model.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss/len(X_train):.6f}, Val Loss = {val_loss:.6f}")
        
        # Load best model
        self.model.load_state_dict(torch.load("best_nbeats_model.pth"))
        
        return {
            "training_history": self.training_history,
            "best_val_loss": best_val_loss,
            "final_epoch": epoch
        }
    
    def predict(self, data: pd.Series) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            data: Input time series data
            
        Returns:
            Predictions
        """
        self.model.eval()
        
        # Prepare input
        if len(data) < self.config.input_size:
            raise ValueError(f"Input data too short. Need at least {self.config.input_size} points.")
        
        input_data = data[-self.config.input_size:].values
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        return prediction.cpu().numpy().flatten()
    
    def get_components(self, data: pd.Series) -> Dict[str, np.ndarray]:
        """
        Get interpretable components.
        
        Args:
            data: Input time series data
            
        Returns:
            Dictionary with trend, seasonality, and residual components
        """
        self.model.eval()
        
        # Prepare input
        input_data = data[-self.config.input_size:].values
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get components
        with torch.no_grad():
            components = self.model.get_interpretable_components(input_tensor)
        
        # Convert to numpy
        return {k: v.cpu().numpy().flatten() for k, v in components.items()}

# Global instance for easy access
nbeats_trainer = None

def get_nbeats_trainer(config: Optional[NBEATSConfig] = None) -> NBEATSTrainer:
    """Get or create N-BEATS trainer instance."""
    global nbeats_trainer
    if nbeats_trainer is None:
        config = config or NBEATSConfig()
        nbeats_trainer = NBEATSTrainer(config)
    return nbeats_trainer
