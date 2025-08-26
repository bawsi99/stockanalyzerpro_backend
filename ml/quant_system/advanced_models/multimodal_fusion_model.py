"""
Multi-Modal Fusion Model for Advanced Trading

This module implements a custom multi-modal fusion model that combines:
1. Price/Volume data (technical analysis)
2. News sentiment (fundamental analysis)
3. Social media sentiment (crowd psychology)
4. Market microstructure (order book dynamics)

The model uses advanced attention mechanisms and cross-domain insights
from weather forecasting, epidemiology, and neuroscience to achieve
superior prediction accuracy.
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
import math

logger = logging.getLogger(__name__)

@dataclass
class MultiModalConfig:
    """Configuration for multi-modal fusion model."""
    
    # Model architecture
    price_embedding_dim: int = 256
    text_embedding_dim: int = 512
    social_embedding_dim: int = 128
    fusion_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    
    # Data processing
    sequence_length: int = 252  # 1 year of daily data
    prediction_horizon: int = 30  # 30 days ahead
    num_classes: int = 3  # Buy, Hold, Sell
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Cross-domain features
    weather_forecasting_enabled: bool = True
    epidemiology_enabled: bool = True
    neuroscience_enabled: bool = True
    quantum_modeling_enabled: bool = True
    
    # Advanced feature parameters
    perturbation_strength: float = 0.05

class PriceEncoder(nn.Module):
    """Advanced price data encoder with technical indicators."""
    
    def __init__(self, config: MultiModalConfig):
        super(PriceEncoder, self).__init__()
        
        self.config = config
        
        # Technical indicators layers
        self.rsi_layer = nn.Linear(1, config.price_embedding_dim // 8)
        self.macd_layer = nn.Linear(1, config.price_embedding_dim // 8)
        self.bollinger_layer = nn.Linear(1, config.price_embedding_dim // 8)
        self.volume_layer = nn.Linear(1, config.price_embedding_dim // 8)
        self.momentum_layer = nn.Linear(1, config.price_embedding_dim // 8)
        self.volatility_layer = nn.Linear(1, config.price_embedding_dim // 8)
        self.trend_layer = nn.Linear(1, config.price_embedding_dim // 8)
        self.support_resistance_layer = nn.Linear(1, config.price_embedding_dim // 8)
        
        # Temporal convolution for pattern recognition
        self.temporal_conv = nn.Conv1d(config.price_embedding_dim, config.price_embedding_dim, 
                                      kernel_size=3, padding=1)
        
        # Attention mechanism for temporal patterns
        self.temporal_attention = nn.MultiheadAttention(config.price_embedding_dim, 
                                                       num_heads=4, dropout=config.dropout)
        
        # Output projection
        self.output_projection = nn.Linear(config.price_embedding_dim, config.price_embedding_dim)
        
    def forward(self, price_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through price encoder.
        
        Args:
            price_data: Tensor of shape (batch_size, sequence_length, num_features)
                       Features: [open, high, low, close, volume, rsi, macd, bollinger, etc.]
        
        Returns:
            Encoded price representation
        """
        batch_size, seq_len, num_features = price_data.shape
        
        # Extract and encode technical indicators
        rsi_encoded = self.rsi_layer(price_data[:, :, 5:6])  # RSI
        macd_encoded = self.macd_layer(price_data[:, :, 6:7])  # MACD
        bollinger_encoded = self.bollinger_layer(price_data[:, :, 7:8])  # Bollinger
        volume_encoded = self.volume_layer(price_data[:, :, 4:5])  # Volume
        momentum_encoded = self.momentum_layer(price_data[:, :, 8:9])  # Momentum
        volatility_encoded = self.volatility_layer(price_data[:, :, 9:10])  # Volatility
        trend_encoded = self.trend_layer(price_data[:, :, 10:11])  # Trend
        sr_encoded = self.support_resistance_layer(price_data[:, :, 11:12])  # Support/Resistance
        
        # Concatenate all encoded features
        encoded_features = torch.cat([
            rsi_encoded, macd_encoded, bollinger_encoded, volume_encoded,
            momentum_encoded, volatility_encoded, trend_encoded, sr_encoded
        ], dim=-1)
        
        # Apply temporal convolution
        encoded_features = encoded_features.transpose(1, 2)  # (batch, features, seq_len)
        conv_output = F.relu(self.temporal_conv(encoded_features))
        conv_output = conv_output.transpose(1, 2)  # (batch, seq_len, features)
        
        # Apply temporal attention
        attn_output, _ = self.temporal_attention(
            conv_output, conv_output, conv_output
        )
        
        # Residual connection and output projection
        output = self.output_projection(attn_output + conv_output)
        
        return output

class TextEncoder(nn.Module):
    """Advanced text encoder for news and social media sentiment."""
    
    def __init__(self, config: MultiModalConfig):
        super(TextEncoder, self).__init__()
        
        self.config = config
        
        # Word embedding layer
        self.word_embedding = nn.Embedding(50000, config.text_embedding_dim)
        
        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(config.text_embedding_dim, config.text_embedding_dim // 2,
                           bidirectional=True, batch_first=True, dropout=config.dropout)
        
        # Attention mechanism for important words/phrases
        self.attention = nn.MultiheadAttention(config.text_embedding_dim, 
                                             num_heads=4, dropout=config.dropout)
        
        # Sentiment analysis layers
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(config.text_embedding_dim, config.text_embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.text_embedding_dim // 2, 3)  # Positive, Neutral, Negative
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.text_embedding_dim, config.text_embedding_dim)
        
    def forward(self, text_data: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through text encoder.
        
        Args:
            text_data: Tensor of shape (batch_size, max_seq_len)
            text_lengths: Tensor of shape (batch_size,) with actual sequence lengths
        
        Returns:
            Encoded text representation
        """
        batch_size, max_seq_len = text_data.shape
        
        # Word embeddings
        embeddings = self.word_embedding(text_data)
        
        # Pack sequence for LSTM
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM processing
        lstm_output, (hidden, cell) = self.lstm(packed_embeddings)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        
        # Attention mechanism
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        
        # Sentiment classification
        sentiment_scores = self.sentiment_classifier(attn_output)
        
        # Global average pooling
        mask = torch.arange(max_seq_len).unsqueeze(0) < text_lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        
        masked_output = attn_output * mask
        pooled_output = masked_output.sum(dim=1) / text_lengths.unsqueeze(1).float()
        
        # Output projection
        output = self.output_projection(pooled_output)
        
        return output, sentiment_scores

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for fusing different data modalities."""
    
    def __init__(self, config: MultiModalConfig):
        super(CrossModalAttention, self).__init__()
        
        self.config = config
        
        # Cross-modal attention layers
        self.price_to_text_attention = nn.MultiheadAttention(
            config.price_embedding_dim, num_heads=4, dropout=config.dropout
        )
        self.text_to_price_attention = nn.MultiheadAttention(
            config.text_embedding_dim, num_heads=4, dropout=config.dropout
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.price_embedding_dim + config.text_embedding_dim, config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim, config.fusion_dim)
        )
        
        # Cross-domain feature extractors
        if config.weather_forecasting_enabled:
            self.weather_features = WeatherFeatureExtractor(config)
        
        if config.epidemiology_enabled:
            self.contagion_features = ContagionFeatureExtractor(config)
        
        if config.neuroscience_enabled:
            self.neuro_features = NeuroFeatureExtractor(config)
        
        if config.quantum_modeling_enabled:
            self.quantum_features = QuantumFeatureExtractor(config)
    
    def forward(self, price_encoded: torch.Tensor, text_encoded: torch.Tensor,
                social_encoded: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through cross-modal attention.
        
        Args:
            price_encoded: Encoded price data
            text_encoded: Encoded text data
            social_encoded: Encoded social media data
        
        Returns:
            Fused multi-modal representation
        """
        # Cross-modal attention
        price_attended, _ = self.price_to_text_attention(price_encoded, text_encoded, text_encoded)
        text_attended, _ = self.text_to_price_attention(text_encoded, price_encoded, price_encoded)
        
        # Concatenate attended representations
        fused_features = torch.cat([price_attended, text_attended], dim=-1)
        
        # Apply fusion layer
        fused_output = self.fusion_layer(fused_features)
        
        # Add cross-domain features if enabled
        if self.config.weather_forecasting_enabled:
            weather_features = self.weather_features(price_encoded)
            fused_output = fused_output + weather_features
        
        if self.config.epidemiology_enabled:
            contagion_features = self.contagion_features(price_encoded, social_encoded)
            fused_output = fused_output + contagion_features
        
        if self.config.neuroscience_enabled:
            neuro_features = self.neuro_features(price_encoded, social_encoded)
            fused_output = fused_output + neuro_features
        
        if self.config.quantum_modeling_enabled:
            quantum_features = self.quantum_features(price_encoded, text_encoded)
            fused_output = fused_output + quantum_features
        
        return fused_output

class WeatherFeatureExtractor(nn.Module):
    """Extracts weather forecasting-inspired features from market data."""
    
    def __init__(self, config: MultiModalConfig):
        super(WeatherFeatureExtractor, self).__init__()
        
        # Ensemble forecasting inspired features
        self.perturbation_generator = nn.Linear(config.price_embedding_dim, config.price_embedding_dim)
        self.ensemble_combiner = nn.Linear(config.price_embedding_dim * 3, config.fusion_dim)
        
    def forward(self, price_data: torch.Tensor) -> torch.Tensor:
        """Generate ensemble-like features inspired by weather forecasting."""
        # Create perturbed versions (like ensemble forecasting)
        # Use configurable perturbation strength instead of hardcoded 0.1
        perturbation_strength = getattr(self.config, 'perturbation_strength', 0.05)
        perturbation1 = self.perturbation_generator(price_data) + torch.randn_like(price_data) * perturbation_strength
        perturbation2 = self.perturbation_generator(price_data) + torch.randn_like(price_data) * perturbation_strength
        
        # Combine original and perturbed versions
        ensemble_features = torch.cat([price_data, perturbation1, perturbation2], dim=-1)
        
        # Generate ensemble output
        weather_features = self.ensemble_combiner(ensemble_features)
        
        return weather_features

class ContagionFeatureExtractor(nn.Module):
    """Extracts epidemiological-inspired features for market contagion modeling."""
    
    def __init__(self, config: MultiModalConfig):
        super(ContagionFeatureExtractor, self).__init__()
        
        # SIR model inspired features
        self.susceptible_detector = nn.Linear(config.price_embedding_dim, config.fusion_dim // 4)
        self.infected_detector = nn.Linear(config.price_embedding_dim, config.fusion_dim // 4)
        self.recovery_detector = nn.Linear(config.price_embedding_dim, config.fusion_dim // 4)
        self.contagion_rate = nn.Linear(config.fusion_dim // 4 * 3, config.fusion_dim // 4)
        
    def forward(self, price_data: torch.Tensor, social_data: torch.Tensor) -> torch.Tensor:
        """Generate contagion features inspired by epidemiological models."""
        # Detect susceptible, infected, and recovered market states
        susceptible = self.susceptible_detector(price_data)
        infected = self.infected_detector(price_data)
        recovery = self.recovery_detector(price_data)
        
        # Combine SIR features
        sir_features = torch.cat([susceptible, infected, recovery], dim=-1)
        
        # Calculate contagion rate
        contagion_features = self.contagion_rate(sir_features)
        
        return contagion_features

class NeuroFeatureExtractor(nn.Module):
    """Extracts neuroscience-inspired features for market behavior modeling."""
    
    def __init__(self, config: MultiModalConfig):
        super(NeuroFeatureExtractor, self).__init__()
        
        # Spike timing detection
        self.spike_detector = nn.Linear(config.price_embedding_dim, config.fusion_dim // 3)
        
        # Synaptic plasticity modeling
        self.plasticity_model = nn.Linear(config.price_embedding_dim, config.fusion_dim // 3)
        
        # Neural synchronization
        self.synchronization = nn.Linear(config.price_embedding_dim, config.fusion_dim // 3)
        
    def forward(self, price_data: torch.Tensor, social_data: torch.Tensor) -> torch.Tensor:
        """Generate neuroscience-inspired features."""
        # Detect market spikes (sudden movements)
        spikes = self.spike_detector(price_data)
        
        # Model synaptic plasticity (market adaptation)
        plasticity = self.plasticity_model(price_data)
        
        # Detect synchronization patterns
        synchronization = self.synchronization(price_data)
        
        # Combine neuro features
        neuro_features = torch.cat([spikes, plasticity, synchronization], dim=-1)
        
        return neuro_features

class QuantumFeatureExtractor(nn.Module):
    """Extracts quantum mechanics-inspired features for market uncertainty modeling."""
    
    def __init__(self, config: MultiModalConfig):
        super(QuantumFeatureExtractor, self).__init__()
        
        # Market superposition modeling
        self.superposition = nn.Linear(config.price_embedding_dim, config.fusion_dim // 3)
        
        # Entanglement detection
        self.entanglement = nn.Linear(config.price_embedding_dim, config.fusion_dim // 3)
        
        # Wave function collapse prediction
        self.collapse_predictor = nn.Linear(config.text_embedding_dim, config.fusion_dim // 3)
        
    def forward(self, price_data: torch.Tensor, text_data: torch.Tensor) -> torch.Tensor:
        """Generate quantum mechanics-inspired features."""
        # Model market superposition (multiple possible states)
        superposition = self.superposition(price_data)
        
        # Detect entanglement (correlated movements)
        entanglement = self.entanglement(price_data)
        
        # Predict wave function collapse (news impact)
        collapse = self.collapse_predictor(text_data)
        
        # Combine quantum features
        quantum_features = torch.cat([superposition, entanglement, collapse], dim=-1)
        
        return quantum_features

class MultiModalFusionModel(nn.Module):
    """Complete multi-modal fusion model for advanced trading."""
    
    def __init__(self, config: MultiModalConfig):
        super(MultiModalFusionModel, self).__init__()
        
        self.config = config
        
        # Encoders
        self.price_encoder = PriceEncoder(config)
        self.text_encoder = TextEncoder(config)
        
        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(config)
        
        # Prediction heads
        self.direction_head = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim // 2, config.num_classes)
        )
        
        self.magnitude_head = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim // 2, 1)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty quantification
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim // 2, 1),
            nn.Softplus()
        )
    
    def forward(self, price_data: torch.Tensor, text_data: torch.Tensor,
                text_lengths: torch.Tensor, social_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-modal fusion model.
        
        Args:
            price_data: Price/volume data
            text_data: News text data
            text_lengths: Text sequence lengths
            social_data: Social media data
        
        Returns:
            Dictionary with predictions and uncertainty
        """
        # Encode different modalities
        price_encoded = self.price_encoder(price_data)
        text_encoded, sentiment_scores = self.text_encoder(text_data, text_lengths)
        
        # Cross-modal fusion
        fused_features = self.cross_modal_attention(price_encoded, text_encoded, social_data)
        
        # Global average pooling
        fused_pooled = fused_features.mean(dim=1)
        
        # Generate predictions
        direction_logits = self.direction_head(fused_pooled)
        magnitude = self.magnitude_head(fused_pooled)
        confidence = self.confidence_head(fused_pooled)
        uncertainty = self.uncertainty_head(fused_pooled)
        
        return {
            'direction_logits': direction_logits,
            'direction_probs': F.softmax(direction_logits, dim=-1),
            'magnitude': magnitude,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'sentiment_scores': sentiment_scores
        }

class MultiModalTrainer:
    """Trainer for multi-modal fusion model."""
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.model = MultiModalFusionModel(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Loss functions
        self.direction_loss = nn.CrossEntropyLoss()
        self.magnitude_loss = nn.MSELoss()
        self.confidence_loss = nn.BCELoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'direction_accuracy': [],
            'magnitude_mae': [],
            'epochs': []
        }
    
    def train(self, train_loader, val_loader, verbose: bool = True) -> Dict[str, Any]:
        """Train the multi-modal fusion model."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss, train_metrics = self._train_epoch(train_loader)
            
            # Validation
            val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['direction_accuracy'].append(val_metrics['direction_accuracy'])
            self.training_history['magnitude_mae'].append(val_metrics['magnitude_mae'])
            self.training_history['epochs'].append(epoch)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_multimodal_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                print(f"Direction Accuracy = {val_metrics['direction_accuracy']:.4f}")
                print(f"Magnitude MAE = {val_metrics['magnitude_mae']:.6f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_multimodal_model.pth'))
        
        return {
            'training_history': self.training_history,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch
        }
    
    def _train_epoch(self, train_loader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_direction_correct = 0
        total_magnitude_error = 0.0
        num_batches = 0
        
        for batch in train_loader:
            price_data, text_data, text_lengths, social_data, direction_targets, magnitude_targets = batch
            
            # Move to device
            price_data = price_data.to(self.device)
            text_data = text_data.to(self.device)
            text_lengths = text_lengths.to(self.device)
            social_data = social_data.to(self.device)
            direction_targets = direction_targets.to(self.device)
            magnitude_targets = magnitude_targets.to(self.device)
            
            # Forward pass
            outputs = self.model(price_data, text_data, text_lengths, social_data)
            
            # Calculate losses
            direction_loss = self.direction_loss(outputs['direction_logits'], direction_targets)
            magnitude_loss = self.magnitude_loss(outputs['magnitude'].squeeze(), magnitude_targets)
            
            # Combined loss
            loss = direction_loss + 0.5 * magnitude_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            direction_preds = torch.argmax(outputs['direction_probs'], dim=1)
            total_direction_correct += (direction_preds == direction_targets).sum().item()
            total_magnitude_error += torch.abs(outputs['magnitude'].squeeze() - magnitude_targets).sum().item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        direction_accuracy = total_direction_correct / (num_batches * self.config.batch_size)
        magnitude_mae = total_magnitude_error / (num_batches * self.config.batch_size)
        
        return avg_loss, {
            'direction_accuracy': direction_accuracy,
            'magnitude_mae': magnitude_mae
        }
    
    def _validate_epoch(self, val_loader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_direction_correct = 0
        total_magnitude_error = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                price_data, text_data, text_lengths, social_data, direction_targets, magnitude_targets = batch
                
                # Move to device
                price_data = price_data.to(self.device)
                text_data = text_data.to(self.device)
                text_lengths = text_lengths.to(self.device)
                social_data = social_data.to(self.device)
                direction_targets = direction_targets.to(self.device)
                magnitude_targets = magnitude_targets.to(self.device)
                
                # Forward pass
                outputs = self.model(price_data, text_data, text_lengths, social_data)
                
                # Calculate losses
                direction_loss = self.direction_loss(outputs['direction_logits'], direction_targets)
                magnitude_loss = self.magnitude_loss(outputs['magnitude'].squeeze(), magnitude_targets)
                
                # Combined loss
                loss = direction_loss + 0.5 * magnitude_loss
                
                # Metrics
                total_loss += loss.item()
                direction_preds = torch.argmax(outputs['direction_probs'], dim=1)
                total_direction_correct += (direction_preds == direction_targets).sum().item()
                total_magnitude_error += torch.abs(outputs['magnitude'].squeeze() - magnitude_targets).sum().item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        direction_accuracy = total_direction_correct / (num_batches * self.config.batch_size)
        magnitude_mae = total_magnitude_error / (num_batches * self.config.batch_size)
        
        return avg_loss, {
            'direction_accuracy': direction_accuracy,
            'magnitude_mae': magnitude_mae
        }
    
    def predict(self, price_data: torch.Tensor, text_data: torch.Tensor,
                text_lengths: torch.Tensor, social_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions using the trained model."""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(price_data, text_data, text_lengths, social_data)
        
        return outputs

# Global instance for easy access
multimodal_trainer = None

def get_multimodal_trainer(config: Optional[MultiModalConfig] = None) -> MultiModalTrainer:
    """Get or create multi-modal trainer instance."""
    global multimodal_trainer
    if multimodal_trainer is None:
        config = config or MultiModalConfig()
        multimodal_trainer = MultiModalTrainer(config)
    return multimodal_trainer
