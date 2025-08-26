"""
Temporal Fusion Transformer (TFT) for Advanced Time Series Prediction

This module implements the Temporal Fusion Transformer, a state-of-the-art
architecture for multi-horizon time series forecasting that combines:

1. Variable Selection Networks - Automatic feature selection
2. Static Covariate Encoders - Handle time-invariant features  
3. Temporal Processing - LSTM for sequential modeling
4. Multi-Head Attention - Capture temporal dependencies
5. Quantile Forecasting - Uncertainty quantification
6. Interpretability - Attention visualization and variable importance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import json
import time
import math
from pathlib import Path
import warnings
from collections import OrderedDict

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class TFTConfig:
    """Configuration for Temporal Fusion Transformer."""
    
    # Data dimensions
    num_time_varying_features: int = 50
    num_static_features: int = 10
    num_categorical_features: int = 5
    categorical_cardinalities: List[int] = field(default_factory=lambda: [10, 5, 3, 7, 4])
    
    # Model architecture
    hidden_size: int = 256
    num_heads: int = 8
    num_lstm_layers: int = 2
    dropout: float = 0.1
    
    # Time series parameters
    input_window: int = 168  # 1 week of hourly data
    output_window: int = 24  # 1 day prediction
    max_encoder_length: int = 168
    max_prediction_length: int = 24
    
    # Variable selection
    variable_selection_dropout: float = 0.1
    variable_selection_hidden_size: int = 128
    
    # Quantile prediction
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 100
    patience: int = 10
    
    # Hardware
    device: str = 'auto'
    
    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for automatic feature selection."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.0):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Flattened gradients
        self.flattened_grn = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout
        )
        
        # Individual variable processing
        self.individual_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=1,
                hidden_size=hidden_size,
                output_size=output_size,
                dropout=dropout
            ) for _ in range(input_size)
        ])
        
        # Variable weights
        self.variable_weights = nn.Linear(input_size * output_size, input_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for variable selection.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            context: Optional context tensor
            
        Returns:
            Tuple of (selected_variables, variable_weights)
        """
        batch_size = x.size(0)
        
        # Process flattened inputs
        flattened_embedding = self.flattened_grn(x, context)
        
        # Process individual variables
        individual_embeddings = []
        for i, grn in enumerate(self.individual_grns):
            var_input = x[:, i:i+1]  # Single variable
            var_embedding = grn(var_input, context)
            individual_embeddings.append(var_embedding)
        
        # Stack individual embeddings
        individual_embeddings = torch.stack(individual_embeddings, dim=1)  # (batch_size, input_size, output_size)
        
        # Compute variable weights
        flat_embeddings = individual_embeddings.view(batch_size, -1)  # (batch_size, input_size * output_size)
        variable_weights = self.variable_weights(flat_embeddings)  # (batch_size, input_size)
        variable_weights = self.softmax(variable_weights)  # Normalize weights
        
        # Apply variable selection
        selected_variables = torch.sum(
            individual_embeddings * variable_weights.unsqueeze(-1), 
            dim=1
        )  # (batch_size, output_size)
        
        return selected_variables, variable_weights

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network building block."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 dropout: float = 0.0, context_size: Optional[int] = None):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        
        # Main network
        self.linear1 = nn.Linear(input_size, hidden_size)
        if context_size is not None:
            self.context_projection = nn.Linear(context_size, hidden_size, bias=False)
        
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        # Skip connection
        if input_size != output_size:
            self.skip_connection = nn.Linear(input_size, output_size)
        else:
            self.skip_connection = None
            
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through GRN."""
        
        # Main path
        hidden = self.linear1(x)
        
        # Add context if available
        if context is not None and self.context_size is not None:
            hidden = hidden + self.context_projection(context)
        
        hidden = self.elu(hidden)
        hidden = self.linear2(hidden)
        hidden = self.dropout(hidden)
        
        # Gating mechanism
        gate = self.sigmoid(self.gate(hidden))
        
        # Skip connection
        if self.skip_connection is not None:
            residual = self.skip_connection(x)
        else:
            residual = x
        
        # Combine with gating
        output = gate * hidden + residual
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output

class StaticCovariateEncoder(nn.Module):
    """Encoder for static (time-invariant) covariates."""
    
    def __init__(self, num_static_features: int, num_categorical_features: int,
                 categorical_cardinalities: List[int], hidden_size: int):
        super().__init__()
        
        self.num_static_features = num_static_features
        self.num_categorical_features = num_categorical_features
        
        # Categorical embeddings
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, hidden_size // 4)
            for cardinality in categorical_cardinalities
        ])
        
        # Continuous feature processing
        if num_static_features > 0:
            self.continuous_projection = nn.Linear(num_static_features, hidden_size // 2)
        
        # Combine all static features
        total_static_size = (
            hidden_size // 2 +  # Continuous features
            (hidden_size // 4) * num_categorical_features  # Categorical features
        )
        
        self.static_combine = nn.Linear(total_static_size, hidden_size)
        
    def forward(self, continuous_static: torch.Tensor, categorical_static: torch.Tensor) -> torch.Tensor:
        """Encode static covariates."""
        
        embeddings = []
        
        # Process continuous static features
        if self.num_static_features > 0:
            continuous_embed = self.continuous_projection(continuous_static)
            embeddings.append(continuous_embed)
        
        # Process categorical static features
        for i, embedding_layer in enumerate(self.categorical_embeddings):
            cat_embed = embedding_layer(categorical_static[:, i])
            embeddings.append(cat_embed)
        
        # Combine all embeddings
        combined = torch.cat(embeddings, dim=-1)
        static_encoding = self.static_combine(combined)
        
        return static_encoding

class TemporalFusionTransformer(nn.Module):
    """Complete Temporal Fusion Transformer model."""
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        
        # Static covariate encoder
        self.static_encoder = StaticCovariateEncoder(
            num_static_features=config.num_static_features,
            num_categorical_features=config.num_categorical_features,
            categorical_cardinalities=config.categorical_cardinalities,
            hidden_size=config.hidden_size
        )
        
        # Variable selection networks
        self.historical_vsn = VariableSelectionNetwork(
            input_size=config.num_time_varying_features,
            hidden_size=config.variable_selection_hidden_size,
            output_size=config.hidden_size,
            dropout=config.variable_selection_dropout
        )
        
        self.future_vsn = VariableSelectionNetwork(
            input_size=config.num_time_varying_features,
            hidden_size=config.variable_selection_hidden_size,
            output_size=config.hidden_size,
            dropout=config.variable_selection_dropout
        )
        
        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_lstm_layers,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Temporal self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Position encoding
        self.positional_encoding = PositionalEncoding(config.hidden_size, max_len=config.max_encoder_length)
        
        # Output layers for each quantile
        self.quantile_outputs = nn.ModuleList([
            nn.Linear(config.hidden_size, config.output_window)
            for _ in config.quantiles
        ])
        
        # Gated residual networks for processing
        self.encoder_grn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
            dropout=config.dropout,
            context_size=config.hidden_size
        )
        
        self.decoder_grn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
            dropout=config.dropout,
            context_size=config.hidden_size
        )
        
    def forward(self, historical_data: torch.Tensor, future_data: torch.Tensor,
                static_continuous: torch.Tensor, static_categorical: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TFT.
        
        Args:
            historical_data: Historical time-varying features (batch, seq_len, features)
            future_data: Future time-varying features (batch, pred_len, features)
            static_continuous: Static continuous features (batch, features)
            static_categorical: Static categorical features (batch, features)
            
        Returns:
            Dictionary with quantile predictions and attention weights
        """
        batch_size, seq_len, _ = historical_data.shape
        pred_len = future_data.shape[1]
        
        # Encode static covariates
        static_encoding = self.static_encoder(static_continuous, static_categorical)
        static_context = static_encoding.unsqueeze(1)  # (batch, 1, hidden_size)
        
        # Historical variable selection
        historical_selected = []
        historical_weights = []
        
        for t in range(seq_len):
            selected, weights = self.historical_vsn(historical_data[:, t, :], static_encoding)
            historical_selected.append(selected)
            historical_weights.append(weights)
        
        historical_selected = torch.stack(historical_selected, dim=1)  # (batch, seq_len, hidden_size)
        historical_weights = torch.stack(historical_weights, dim=1)    # (batch, seq_len, num_features)
        
        # Future variable selection
        future_selected = []
        future_weights = []
        
        for t in range(pred_len):
            selected, weights = self.future_vsn(future_data[:, t, :], static_encoding)
            future_selected.append(selected)
            future_weights.append(weights)
        
        future_selected = torch.stack(future_selected, dim=1)  # (batch, pred_len, hidden_size)
        future_weights = torch.stack(future_weights, dim=1)    # (batch, pred_len, num_features)
        
        # Add positional encoding
        historical_selected = self.positional_encoding(historical_selected)
        
        # Encoder processing
        encoder_inputs = self.encoder_grn(historical_selected, static_context.repeat(1, seq_len, 1))
        
        # LSTM processing
        lstm_output, (hidden, cell) = self.lstm(encoder_inputs)
        
        # Self-attention over historical sequence
        attended_output, attention_weights = self.self_attention(lstm_output, lstm_output, lstm_output)
        
        # Decoder processing
        decoder_inputs = self.decoder_grn(future_selected, static_context.repeat(1, pred_len, 1))
        
        # Use last LSTM state for initial decoder state
        decoder_output, _ = self.lstm(decoder_inputs, (hidden, cell))
        
        # Generate quantile predictions
        quantile_predictions = {}
        for i, quantile in enumerate(self.config.quantiles):
            pred = self.quantile_outputs[i](decoder_output)  # (batch, pred_len, 1)
            quantile_predictions[f'q{quantile}'] = pred.squeeze(-1)  # (batch, pred_len)
        
        return {
            'predictions': quantile_predictions,
            'attention_weights': attention_weights,
            'historical_variable_weights': historical_weights,
            'future_variable_weights': future_weights,
            'static_encoding': static_encoding
        }

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)

class QuantileLoss(nn.Module):
    """Quantile loss for probabilistic forecasting."""
    
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss."""
        
        total_loss = 0.0
        
        for quantile in self.quantiles:
            pred = predictions[f'q{quantile}']
            errors = targets - pred
            
            quantile_loss = torch.max(
                quantile * errors,
                (quantile - 1) * errors
            )
            
            total_loss += quantile_loss.mean()
        
        return total_loss / len(self.quantiles)

class TFTTrainer:
    """Trainer for Temporal Fusion Transformer."""
    
    def __init__(self, model: TemporalFusionTransformer, config: TFTConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = QuantileLoss(config.quantiles)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': []
        }
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            (historical_data, future_data, static_continuous, 
             static_categorical, targets) = [x.to(self.device) for x in batch]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(historical_data, future_data, static_continuous, static_categorical)
            
            # Compute loss
            loss = self.criterion(outputs['predictions'], targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model."""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                (historical_data, future_data, static_continuous, 
                 static_categorical, targets) = [x.to(self.device) for x in batch]
                
                # Forward pass
                outputs = self.model(historical_data, future_data, static_continuous, static_categorical)
                
                # Compute loss
                loss = self.criterion(outputs['predictions'], targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None) -> Dict[str, List[float]]:
        """Full training loop."""
        
        logger.info("Starting TFT training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.training_history['train_losses'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.training_history['val_losses'].append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.training_history['learning_rates'].append(current_lr)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                           f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                           f"LR: {current_lr:.2e}")
                
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - Train Loss: {train_loss:.6f}")
        
        logger.info("TFT training completed!")
        return self.training_history
    
    def predict(self, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        """Generate predictions."""
        
        self.model.eval()
        all_predictions = {f'q{q}': [] for q in self.config.quantiles}
        attention_weights = []
        variable_weights = []
        
        with torch.no_grad():
            for batch in data_loader:
                (historical_data, future_data, static_continuous, static_categorical) = batch[:4]
                batch_data = [x.to(self.device) for x in [historical_data, future_data, static_continuous, static_categorical]]
                
                # Forward pass
                outputs = self.model(*batch_data)
                
                # Store predictions
                for quantile in self.config.quantiles:
                    pred = outputs['predictions'][f'q{quantile}'].cpu().numpy()
                    all_predictions[f'q{quantile}'].append(pred)
                
                # Store attention weights for interpretability
                attention_weights.append(outputs['attention_weights'].cpu().numpy())
                variable_weights.append(outputs['historical_variable_weights'].cpu().numpy())
        
        # Concatenate all predictions
        for quantile in self.config.quantiles:
            all_predictions[f'q{quantile}'] = np.concatenate(all_predictions[f'q{quantile}'], axis=0)
        
        return {
            'predictions': all_predictions,
            'attention_weights': np.concatenate(attention_weights, axis=0),
            'variable_weights': np.concatenate(variable_weights, axis=0)
        }

class TFTDataPreprocessor:
    """Data preprocessor for TFT."""
    
    def __init__(self, config: TFTConfig):
        self.config = config
        
    def create_sequences(self, data: pd.DataFrame, target_col: str) -> Tuple[torch.Tensor, ...]:
        """Create sequences for TFT training."""
        
        sequences = []
        targets = []
        
        for i in range(len(data) - self.config.input_window - self.config.output_window + 1):
            # Historical sequence
            hist_start = i
            hist_end = i + self.config.input_window
            historical = data.iloc[hist_start:hist_end].drop(columns=[target_col]).values
            
            # Future sequence (assuming future features are available)
            fut_start = hist_end
            fut_end = fut_start + self.config.output_window
            future = data.iloc[fut_start:fut_end].drop(columns=[target_col]).values
            
            # Target sequence
            target = data.iloc[fut_start:fut_end][target_col].values
            
            sequences.append((historical, future))
            targets.append(target)
        
        # Convert to tensors
        historical_tensors = torch.stack([torch.tensor(seq[0], dtype=torch.float32) for seq in sequences])
        future_tensors = torch.stack([torch.tensor(seq[1], dtype=torch.float32) for seq in sequences])
        target_tensors = torch.stack([torch.tensor(target, dtype=torch.float32) for target in targets])
        
        return historical_tensors, future_tensors, target_tensors
    
    def prepare_static_features(self, data: pd.DataFrame, static_cols: List[str], 
                               categorical_cols: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare static features."""
        
        # Continuous static features
        continuous_static = data[static_cols].iloc[0].values  # Assuming static across time
        continuous_static = torch.tensor(continuous_static, dtype=torch.float32).unsqueeze(0)
        
        # Categorical static features
        categorical_static = data[categorical_cols].iloc[0].values
        categorical_static = torch.tensor(categorical_static, dtype=torch.long).unsqueeze(0)
        
        return continuous_static, categorical_static

# Factory functions
def create_tft_model(config: TFTConfig = None) -> TemporalFusionTransformer:
    """Create TFT model with default or custom configuration."""
    if config is None:
        config = TFTConfig()
    return TemporalFusionTransformer(config)

def create_tft_trainer(model: TemporalFusionTransformer, config: TFTConfig = None) -> TFTTrainer:
    """Create TFT trainer with default or custom configuration."""
    if config is None:
        config = TFTConfig()
    return TFTTrainer(model, config)

def quick_tft_training(historical_data: torch.Tensor, future_data: torch.Tensor,
                      static_continuous: torch.Tensor, static_categorical: torch.Tensor,
                      targets: torch.Tensor, validation_split: float = 0.2) -> TFTTrainer:
    """Quick TFT training with default settings."""
    
    # Create dataset
    dataset = TensorDataset(historical_data, future_data, static_continuous, static_categorical, targets)
    
    # Split data
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    config = TFTConfig()
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create and train model
    model = create_tft_model(config)
    trainer = create_tft_trainer(model, config)
    trainer.train(train_loader, val_loader)
    
    return trainer

# Global TFT instances
tft_model = None
tft_trainer = None

def get_tft_model(config: TFTConfig = None) -> TemporalFusionTransformer:
    """Get global TFT model instance."""
    global tft_model
    if tft_model is None:
        tft_model = create_tft_model(config)
    return tft_model

def get_tft_trainer(config: TFTConfig = None) -> TFTTrainer:
    """Get global TFT trainer instance."""
    global tft_trainer
    if tft_trainer is None:
        model = get_tft_model(config)
        tft_trainer = create_tft_trainer(model, config)
    return tft_trainer

