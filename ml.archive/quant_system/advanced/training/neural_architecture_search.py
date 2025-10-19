"""
Neural Architecture Search (NAS) for Advanced Trading System

This module implements automated neural architecture search to find optimal
model architectures for trading predictions. It includes:

1. Search Space Definition - Define possible model architectures
2. Architecture Evaluation - Evaluate candidate architectures
3. Search Strategy - Efficient search through architecture space
4. Progressive Search - Start simple and increase complexity
5. Multi-Objective Optimization - Balance accuracy, speed, and interpretability
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import json
import time
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search."""
    
    # Search space parameters
    max_layers: int = 8
    min_layers: int = 2
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    activation_functions: List[str] = field(default_factory=lambda: ['relu', 'gelu', 'swish', 'leaky_relu'])
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    normalization_types: List[str] = field(default_factory=lambda: ['batch_norm', 'layer_norm', 'none'])
    
    # Attention mechanisms
    attention_types: List[str] = field(default_factory=lambda: ['none', 'self_attention', 'multi_head', 'cross_attention'])
    num_attention_heads: List[int] = field(default_factory=lambda: [4, 8, 16])
    
    # Residual connections
    use_residual: List[bool] = field(default_factory=lambda: [True, False])
    residual_types: List[str] = field(default_factory=lambda: ['simple', 'dense', 'highway'])
    
    # Search strategy
    search_strategy: str = 'progressive'  # 'random', 'evolutionary', 'progressive', 'differentiable'
    max_search_time: int = 7200  # 2 hours in seconds
    max_architectures: int = 100
    population_size: int = 20  # for evolutionary search
    
    # Evaluation parameters
    train_epochs: int = 10
    validation_split: float = 0.2
    early_stopping_patience: int = 3
    
    # Multi-objective weights
    accuracy_weight: float = 0.6
    speed_weight: float = 0.2
    complexity_weight: float = 0.1
    interpretability_weight: float = 0.1
    
    # Hardware constraints
    max_memory_mb: int = 4096
    max_inference_time_ms: int = 100
    
    # Output
    save_top_k: int = 5
    results_dir: str = "nas_results"

class ArchitectureEncoder:
    """Encode and decode neural architectures."""
    
    def __init__(self, config: NASConfig):
        self.config = config
        
    def encode_architecture(self, architecture: Dict[str, Any]) -> List[int]:
        """Encode architecture as integer vector for search algorithms."""
        encoding = []
        
        # Number of layers
        encoding.append(architecture.get('num_layers', 3))
        
        # Layer configurations
        layers = architecture.get('layers', [])
        for i in range(self.config.max_layers):
            if i < len(layers):
                layer = layers[i]
                # Hidden size index
                hidden_size = layer.get('hidden_size', 128)
                hidden_idx = self.config.hidden_sizes.index(hidden_size) if hidden_size in self.config.hidden_sizes else 0
                encoding.append(hidden_idx)
                
                # Activation function index
                activation = layer.get('activation', 'relu')
                activation_idx = self.config.activation_functions.index(activation) if activation in self.config.activation_functions else 0
                encoding.append(activation_idx)
                
                # Dropout rate index
                dropout = layer.get('dropout', 0.0)
                dropout_idx = self.config.dropout_rates.index(dropout) if dropout in self.config.dropout_rates else 0
                encoding.append(dropout_idx)
                
                # Normalization type index
                norm_type = layer.get('normalization', 'none')
                norm_idx = self.config.normalization_types.index(norm_type) if norm_type in self.config.normalization_types else 0
                encoding.append(norm_idx)
                
                # Use residual
                use_residual = layer.get('use_residual', False)
                encoding.append(int(use_residual))
            else:
                # Padding for unused layers
                encoding.extend([0, 0, 0, 0, 0])
        
        # Global architecture properties
        attention_type = architecture.get('attention_type', 'none')
        attention_idx = self.config.attention_types.index(attention_type) if attention_type in self.config.attention_types else 0
        encoding.append(attention_idx)
        
        num_heads = architecture.get('num_attention_heads', 8)
        heads_idx = self.config.num_attention_heads.index(num_heads) if num_heads in self.config.num_attention_heads else 0
        encoding.append(heads_idx)
        
        return encoding
    
    def decode_architecture(self, encoding: List[int]) -> Dict[str, Any]:
        """Decode integer vector back to architecture specification."""
        architecture = {}
        
        # Number of layers
        num_layers = max(self.config.min_layers, min(encoding[0], self.config.max_layers))
        architecture['num_layers'] = num_layers
        
        # Layer configurations
        layers = []
        idx = 1
        for i in range(num_layers):
            layer = {}
            
            # Hidden size
            hidden_idx = encoding[idx] if idx < len(encoding) else 0
            hidden_idx = max(0, min(hidden_idx, len(self.config.hidden_sizes) - 1))
            layer['hidden_size'] = self.config.hidden_sizes[hidden_idx]
            idx += 1
            
            # Activation function
            activation_idx = encoding[idx] if idx < len(encoding) else 0
            activation_idx = max(0, min(activation_idx, len(self.config.activation_functions) - 1))
            layer['activation'] = self.config.activation_functions[activation_idx]
            idx += 1
            
            # Dropout rate
            dropout_idx = encoding[idx] if idx < len(encoding) else 0
            dropout_idx = max(0, min(dropout_idx, len(self.config.dropout_rates) - 1))
            layer['dropout'] = self.config.dropout_rates[dropout_idx]
            idx += 1
            
            # Normalization type
            norm_idx = encoding[idx] if idx < len(encoding) else 0
            norm_idx = max(0, min(norm_idx, len(self.config.normalization_types) - 1))
            layer['normalization'] = self.config.normalization_types[norm_idx]
            idx += 1
            
            # Use residual
            use_residual = bool(encoding[idx]) if idx < len(encoding) else False
            layer['use_residual'] = use_residual
            idx += 1
            
            layers.append(layer)
        
        architecture['layers'] = layers
        
        # Skip unused layer slots
        idx = 1 + self.config.max_layers * 5
        
        # Global attention properties
        if idx < len(encoding):
            attention_idx = max(0, min(encoding[idx], len(self.config.attention_types) - 1))
            architecture['attention_type'] = self.config.attention_types[attention_idx]
            idx += 1
        
        if idx < len(encoding):
            heads_idx = max(0, min(encoding[idx], len(self.config.num_attention_heads) - 1))
            architecture['num_attention_heads'] = self.config.num_attention_heads[heads_idx]
        
        return architecture

class DynamicNeuralNetwork(nn.Module):
    """Dynamic neural network that can be constructed from architecture specification."""
    
    def __init__(self, architecture: Dict[str, Any], input_size: int, output_size: int):
        super().__init__()
        self.architecture = architecture
        self.input_size = input_size
        self.output_size = output_size
        
        self.layers = nn.ModuleList()
        self.residual_connections = []
        
        # Build layers
        current_size = input_size
        for i, layer_config in enumerate(architecture['layers']):
            layer = self._build_layer(layer_config, current_size)
            self.layers.append(layer)
            
            # Track residual connections
            if layer_config.get('use_residual', False):
                self.residual_connections.append((i, current_size))
            
            current_size = layer_config['hidden_size']
        
        # Output layer
        self.output_layer = nn.Linear(current_size, output_size)
        
        # Attention mechanism
        attention_type = architecture.get('attention_type', 'none')
        if attention_type != 'none':
            self.attention = self._build_attention(attention_type, current_size)
        else:
            self.attention = None
    
    def _build_layer(self, layer_config: Dict[str, Any], input_size: int) -> nn.Module:
        """Build a single layer from configuration."""
        layers = []
        
        # Linear layer
        hidden_size = layer_config['hidden_size']
        layers.append(nn.Linear(input_size, hidden_size))
        
        # Normalization
        norm_type = layer_config.get('normalization', 'none')
        if norm_type == 'batch_norm':
            layers.append(nn.BatchNorm1d(hidden_size))
        elif norm_type == 'layer_norm':
            layers.append(nn.LayerNorm(hidden_size))
        
        # Activation
        activation = layer_config.get('activation', 'relu')
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'swish':
            layers.append(nn.SiLU())
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.1))
        
        # Dropout
        dropout_rate = layer_config.get('dropout', 0.0)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _build_attention(self, attention_type: str, hidden_size: int) -> nn.Module:
        """Build attention mechanism."""
        if attention_type == 'self_attention':
            return nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)
        elif attention_type == 'multi_head':
            num_heads = self.architecture.get('num_attention_heads', 8)
            return nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        else:
            return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dynamic network."""
        batch_size = x.size(0)
        
        # Store intermediate outputs for residual connections
        layer_outputs = [x]
        
        # Forward through layers
        current_x = x
        for i, layer in enumerate(self.layers):
            layer_output = layer(current_x)
            
            # Apply residual connection if configured
            use_residual = False
            residual_input_size = None
            for res_idx, res_input_size in self.residual_connections:
                if res_idx == i and res_input_size == layer_output.size(-1):
                    use_residual = True
                    residual_input_size = res_input_size
                    break
            
            if use_residual and len(layer_outputs) > 1:
                # Find compatible previous layer for residual connection
                for prev_output in reversed(layer_outputs[:-1]):
                    if prev_output.size(-1) == layer_output.size(-1):
                        layer_output = layer_output + prev_output
                        break
            
            layer_outputs.append(layer_output)
            current_x = layer_output
        
        # Apply attention if configured
        if self.attention is not None:
            # Reshape for attention (assuming sequence length of 1)
            current_x = current_x.unsqueeze(1)  # Add sequence dimension
            attended_x, _ = self.attention(current_x, current_x, current_x)
            current_x = attended_x.squeeze(1)  # Remove sequence dimension
        
        # Output layer
        output = self.output_layer(current_x)
        return output

class ArchitectureEvaluator:
    """Evaluate neural architectures using multiple criteria."""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def evaluate_architecture(self, architecture: Dict[str, Any], 
                            train_data: torch.Tensor, train_targets: torch.Tensor,
                            val_data: torch.Tensor, val_targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate a single architecture on multiple criteria."""
        
        try:
            # Build model
            input_size = train_data.size(-1)
            output_size = train_targets.size(-1) if len(train_targets.shape) > 1 else 1
            
            model = DynamicNeuralNetwork(architecture, input_size, output_size)
            model = model.to(self.device)
            
            # Evaluate accuracy
            accuracy_score = self._evaluate_accuracy(model, train_data, train_targets, val_data, val_targets)
            
            # Evaluate speed
            speed_score = self._evaluate_speed(model, val_data)
            
            # Evaluate complexity
            complexity_score = self._evaluate_complexity(model)
            
            # Evaluate interpretability
            interpretability_score = self._evaluate_interpretability(architecture)
            
            # Calculate weighted score
            weighted_score = (
                self.config.accuracy_weight * accuracy_score +
                self.config.speed_weight * speed_score +
                self.config.complexity_weight * complexity_score +
                self.config.interpretability_weight * interpretability_score
            )
            
            return {
                'accuracy': accuracy_score,
                'speed': speed_score,
                'complexity': complexity_score,
                'interpretability': interpretability_score,
                'weighted_score': weighted_score,
                'architecture': architecture
            }
            
        except Exception as e:
            logger.error(f"Error evaluating architecture: {e}")
            return {
                'accuracy': 0.0,
                'speed': 0.0,
                'complexity': 0.0,
                'interpretability': 0.0,
                'weighted_score': 0.0,
                'architecture': architecture,
                'error': str(e)
            }
    
    def _evaluate_accuracy(self, model: nn.Module, train_data: torch.Tensor, train_targets: torch.Tensor,
                          val_data: torch.Tensor, val_targets: torch.Tensor) -> float:
        """Evaluate model accuracy."""
        try:
            # Prepare data
            train_dataset = TensorDataset(train_data, train_targets)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            val_dataset = TensorDataset(val_data, val_targets)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Quick training
            model.train()
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.train_epochs):
                # Training
                train_loss = 0.0
                for batch_data, batch_targets in train_loader:
                    batch_data = batch_data.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_data)
                    
                    # Handle different output shapes
                    if len(outputs.shape) > len(batch_targets.shape):
                        outputs = outputs.squeeze()
                    elif len(outputs.shape) < len(batch_targets.shape):
                        outputs = outputs.unsqueeze(-1)
                    
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_data, batch_targets in val_loader:
                        batch_data = batch_data.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        
                        outputs = model(batch_data)
                        
                        # Handle different output shapes
                        if len(outputs.shape) > len(batch_targets.shape):
                            outputs = outputs.squeeze()
                        elif len(outputs.shape) < len(batch_targets.shape):
                            outputs = outputs.unsqueeze(-1)
                        
                        loss = criterion(outputs, batch_targets)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        break
                
                model.train()
            
            # Return accuracy score (inverse of loss, normalized)
            accuracy_score = 1.0 / (1.0 + best_val_loss)
            return min(1.0, max(0.0, accuracy_score))
            
        except Exception as e:
            logger.error(f"Error in accuracy evaluation: {e}")
            return 0.0
    
    def _evaluate_speed(self, model: nn.Module, test_data: torch.Tensor) -> float:
        """Evaluate model inference speed."""
        try:
            model.eval()
            
            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    _ = model(test_data[:1].to(self.device))
            
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = model(test_data[:1].to(self.device))
            
            avg_inference_time_ms = (time.time() - start_time) * 1000 / 100
            
            # Score: higher is better (faster models get higher scores)
            if avg_inference_time_ms <= self.config.max_inference_time_ms:
                speed_score = 1.0 - (avg_inference_time_ms / self.config.max_inference_time_ms)
            else:
                speed_score = 0.1  # Penalty for slow models
            
            return max(0.0, min(1.0, speed_score))
            
        except Exception as e:
            logger.error(f"Error in speed evaluation: {e}")
            return 0.0
    
    def _evaluate_complexity(self, model: nn.Module) -> float:
        """Evaluate model complexity (lower complexity is better)."""
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Count layers
            total_layers = len([m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d))])
            
            # Normalize complexity score (simpler models get higher scores)
            param_score = 1.0 / (1.0 + total_params / 10000)  # Normalize by 10k params
            layer_score = 1.0 / (1.0 + total_layers / 10)     # Normalize by 10 layers
            
            complexity_score = (param_score + layer_score) / 2
            return max(0.0, min(1.0, complexity_score))
            
        except Exception as e:
            logger.error(f"Error in complexity evaluation: {e}")
            return 0.0
    
    def _evaluate_interpretability(self, architecture: Dict[str, Any]) -> float:
        """Evaluate architecture interpretability."""
        try:
            score = 1.0
            
            # Penalty for too many layers
            num_layers = len(architecture.get('layers', []))
            if num_layers > 5:
                score *= 0.8
            
            # Bonus for attention mechanisms (more interpretable)
            attention_type = architecture.get('attention_type', 'none')
            if attention_type != 'none':
                score *= 1.2
            
            # Penalty for complex residual connections
            residual_count = sum(1 for layer in architecture.get('layers', []) if layer.get('use_residual', False))
            if residual_count > 2:
                score *= 0.9
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error in interpretability evaluation: {e}")
            return 0.0

class NeuralArchitectureSearch:
    """Main Neural Architecture Search engine."""
    
    def __init__(self, config: NASConfig = None):
        self.config = config or NASConfig()
        self.encoder = ArchitectureEncoder(self.config)
        self.evaluator = ArchitectureEvaluator(self.config)
        self.search_history = []
        self.best_architectures = []
        
        # Create results directory
        Path(self.config.results_dir).mkdir(exist_ok=True)
        
    def search(self, train_data: torch.Tensor, train_targets: torch.Tensor,
               val_data: torch.Tensor, val_targets: torch.Tensor) -> List[Dict[str, Any]]:
        """Run neural architecture search."""
        
        logger.info(f"Starting NAS with strategy: {self.config.search_strategy}")
        logger.info(f"Search space: {self._get_search_space_size()} possible architectures")
        
        start_time = time.time()
        
        if self.config.search_strategy == 'random':
            results = self._random_search(train_data, train_targets, val_data, val_targets, start_time)
        elif self.config.search_strategy == 'evolutionary':
            results = self._evolutionary_search(train_data, train_targets, val_data, val_targets, start_time)
        elif self.config.search_strategy == 'progressive':
            results = self._progressive_search(train_data, train_targets, val_data, val_targets, start_time)
        else:
            raise ValueError(f"Unknown search strategy: {self.config.search_strategy}")
        
        # Save results
        self._save_results(results)
        
        logger.info(f"NAS completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Evaluated {len(self.search_history)} architectures")
        logger.info(f"Best weighted score: {results[0]['weighted_score']:.4f}")
        
        return results
    
    def _random_search(self, train_data: torch.Tensor, train_targets: torch.Tensor,
                      val_data: torch.Tensor, val_targets: torch.Tensor, start_time: float) -> List[Dict[str, Any]]:
        """Random search strategy."""
        
        logger.info("Starting random search...")
        
        while (len(self.search_history) < self.config.max_architectures and 
               time.time() - start_time < self.config.max_search_time):
            
            # Generate random architecture
            architecture = self._generate_random_architecture()
            
            # Evaluate architecture
            result = self.evaluator.evaluate_architecture(
                architecture, train_data, train_targets, val_data, val_targets
            )
            
            self.search_history.append(result)
            
            logger.info(f"Evaluated {len(self.search_history)}/{self.config.max_architectures} "
                       f"architectures, best score: {max(self.search_history, key=lambda x: x['weighted_score'])['weighted_score']:.4f}")
        
        # Return top k results
        return sorted(self.search_history, key=lambda x: x['weighted_score'], reverse=True)[:self.config.save_top_k]
    
    def _progressive_search(self, train_data: torch.Tensor, train_targets: torch.Tensor,
                           val_data: torch.Tensor, val_targets: torch.Tensor, start_time: float) -> List[Dict[str, Any]]:
        """Progressive search strategy - start simple and increase complexity."""
        
        logger.info("Starting progressive search...")
        
        # Stage 1: Simple architectures (2-3 layers)
        logger.info("Stage 1: Evaluating simple architectures...")
        for _ in range(self.config.max_architectures // 3):
            if time.time() - start_time >= self.config.max_search_time:
                break
                
            architecture = self._generate_random_architecture(max_layers=3, complexity='simple')
            result = self.evaluator.evaluate_architecture(
                architecture, train_data, train_targets, val_data, val_targets
            )
            self.search_history.append(result)
        
        # Stage 2: Medium architectures (4-6 layers)
        logger.info("Stage 2: Evaluating medium architectures...")
        for _ in range(self.config.max_architectures // 3):
            if time.time() - start_time >= self.config.max_search_time:
                break
                
            architecture = self._generate_random_architecture(max_layers=6, complexity='medium')
            result = self.evaluator.evaluate_architecture(
                architecture, train_data, train_targets, val_data, val_targets
            )
            self.search_history.append(result)
        
        # Stage 3: Complex architectures (7-8 layers with attention)
        logger.info("Stage 3: Evaluating complex architectures...")
        for _ in range(self.config.max_architectures // 3):
            if time.time() - start_time >= self.config.max_search_time:
                break
                
            architecture = self._generate_random_architecture(max_layers=8, complexity='complex')
            result = self.evaluator.evaluate_architecture(
                architecture, train_data, train_targets, val_data, val_targets
            )
            self.search_history.append(result)
        
        # Return top k results
        return sorted(self.search_history, key=lambda x: x['weighted_score'], reverse=True)[:self.config.save_top_k]
    
    def _evolutionary_search(self, train_data: torch.Tensor, train_targets: torch.Tensor,
                            val_data: torch.Tensor, val_targets: torch.Tensor, start_time: float) -> List[Dict[str, Any]]:
        """Evolutionary search strategy."""
        
        logger.info("Starting evolutionary search...")
        
        # Initialize population
        population = []
        for _ in range(self.config.population_size):
            architecture = self._generate_random_architecture()
            result = self.evaluator.evaluate_architecture(
                architecture, train_data, train_targets, val_data, val_targets
            )
            population.append(result)
            self.search_history.append(result)
        
        generation = 0
        while (len(self.search_history) < self.config.max_architectures and 
               time.time() - start_time < self.config.max_search_time):
            
            generation += 1
            logger.info(f"Generation {generation}, population size: {len(population)}")
            
            # Selection: keep top 50%
            population = sorted(population, key=lambda x: x['weighted_score'], reverse=True)
            elite = population[:self.config.population_size // 2]
            
            # Reproduction: create offspring
            offspring = []
            for _ in range(self.config.population_size - len(elite)):
                if time.time() - start_time >= self.config.max_search_time:
                    break
                
                # Select parents
                parent1 = np.random.choice(elite)
                parent2 = np.random.choice(elite)
                
                # Crossover and mutation
                child_architecture = self._crossover_and_mutate(
                    parent1['architecture'], parent2['architecture']
                )
                
                # Evaluate child
                child_result = self.evaluator.evaluate_architecture(
                    child_architecture, train_data, train_targets, val_data, val_targets
                )
                
                offspring.append(child_result)
                self.search_history.append(child_result)
            
            # New population
            population = elite + offspring
        
        # Return top k results
        return sorted(self.search_history, key=lambda x: x['weighted_score'], reverse=True)[:self.config.save_top_k]
    
    def _generate_random_architecture(self, max_layers: int = None, complexity: str = 'random') -> Dict[str, Any]:
        """Generate a random architecture within constraints."""
        
        if max_layers is None:
            max_layers = self.config.max_layers
        
        num_layers = np.random.randint(self.config.min_layers, max_layers + 1)
        
        architecture = {
            'num_layers': num_layers,
            'layers': []
        }
        
        for i in range(num_layers):
            layer = {
                'hidden_size': np.random.choice(self.config.hidden_sizes),
                'activation': np.random.choice(self.config.activation_functions),
                'dropout': np.random.choice(self.config.dropout_rates),
                'normalization': np.random.choice(self.config.normalization_types),
                'use_residual': np.random.choice(self.config.use_residual)
            }
            
            # Adjust based on complexity
            if complexity == 'simple':
                layer['dropout'] = np.random.choice([0.0, 0.1, 0.2])
                layer['use_residual'] = False
                layer['normalization'] = np.random.choice(['none', 'batch_norm'])
            elif complexity == 'medium':
                layer['dropout'] = np.random.choice([0.1, 0.2, 0.3])
                layer['use_residual'] = np.random.choice([True, False])
            elif complexity == 'complex':
                layer['dropout'] = np.random.choice([0.2, 0.3, 0.4, 0.5])
                layer['use_residual'] = True
            
            architecture['layers'].append(layer)
        
        # Add attention mechanism
        if complexity == 'complex' or (complexity == 'random' and np.random.random() > 0.7):
            architecture['attention_type'] = np.random.choice(['self_attention', 'multi_head'])
            architecture['num_attention_heads'] = np.random.choice(self.config.num_attention_heads)
        else:
            architecture['attention_type'] = 'none'
            architecture['num_attention_heads'] = 8
        
        return architecture
    
    def _crossover_and_mutate(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Create offspring through crossover and mutation."""
        
        # Encode parents
        encoding1 = self.encoder.encode_architecture(parent1)
        encoding2 = self.encoder.encode_architecture(parent2)
        
        # Crossover: random selection from parents
        child_encoding = []
        for i in range(max(len(encoding1), len(encoding2))):
            if i < len(encoding1) and i < len(encoding2):
                child_encoding.append(encoding1[i] if np.random.random() < 0.5 else encoding2[i])
            elif i < len(encoding1):
                child_encoding.append(encoding1[i])
            else:
                child_encoding.append(encoding2[i])
        
        # Mutation: random changes with 10% probability
        for i in range(len(child_encoding)):
            if np.random.random() < 0.1:
                if i == 0:  # Number of layers
                    child_encoding[i] = np.random.randint(self.config.min_layers, self.config.max_layers + 1)
                else:
                    # Other parameters
                    child_encoding[i] = np.random.randint(0, 5)  # Approximate range
        
        # Decode child
        child_architecture = self.encoder.decode_architecture(child_encoding)
        
        return child_architecture
    
    def _get_search_space_size(self) -> int:
        """Estimate search space size."""
        layer_combinations = (
            len(self.config.hidden_sizes) *
            len(self.config.activation_functions) *
            len(self.config.dropout_rates) *
            len(self.config.normalization_types) *
            len(self.config.use_residual)
        )
        
        total_combinations = 1
        for num_layers in range(self.config.min_layers, self.config.max_layers + 1):
            total_combinations += layer_combinations ** num_layers
        
        return total_combinations * len(self.config.attention_types) * len(self.config.num_attention_heads)
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save search results to disk."""
        
        results_file = Path(self.config.results_dir) / f"nas_results_{int(time.time())}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_result = {
                'accuracy': float(result['accuracy']),
                'speed': float(result['speed']),
                'complexity': float(result['complexity']),
                'interpretability': float(result['interpretability']),
                'weighted_score': float(result['weighted_score']),
                'architecture': result['architecture']
            }
            if 'error' in result:
                serializable_result['error'] = result['error']
            
            serializable_results.append(serializable_result)
        
        # Save to JSON with proper serialization
        import json
        
        def convert_to_serializable(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
        
        # Convert results to fully serializable format
        fully_serializable_results = convert_to_serializable(serializable_results)
        
        # Save to JSON
        with open(results_file, 'w') as f:
            json.dump({
                'config': {
                    'search_strategy': self.config.search_strategy,
                    'max_architectures': self.config.max_architectures,
                    'accuracy_weight': self.config.accuracy_weight,
                    'speed_weight': self.config.speed_weight,
                    'complexity_weight': self.config.complexity_weight,
                    'interpretability_weight': self.config.interpretability_weight
                },
                'results': fully_serializable_results,
                'summary': {
                    'total_evaluated': len(self.search_history),
                    'best_score': float(results[0]['weighted_score']) if results else 0.0,
                    'search_time': time.time()
                }
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")

# Factory functions for easy integration
def create_nas_engine(config: NASConfig = None) -> NeuralArchitectureSearch:
    """Create a NAS engine with default or custom configuration."""
    return NeuralArchitectureSearch(config)

def quick_architecture_search(train_data: torch.Tensor, train_targets: torch.Tensor,
                             val_data: torch.Tensor, val_targets: torch.Tensor,
                             max_time_minutes: int = 30) -> List[Dict[str, Any]]:
    """Quick architecture search with sensible defaults."""
    
    config = NASConfig(
        max_search_time=max_time_minutes * 60,
        max_architectures=50,
        search_strategy='progressive'
    )
    
    nas = NeuralArchitectureSearch(config)
    return nas.search(train_data, train_targets, val_data, val_targets)

# Global NAS instance for easy access
nas_engine = None

def get_nas_engine() -> NeuralArchitectureSearch:
    """Get global NAS engine instance."""
    global nas_engine
    if nas_engine is None:
        nas_engine = create_nas_engine()
    return nas_engine
