"""
Meta-Learning Framework for Advanced Trading System

This module implements Model-Agnostic Meta-Learning (MAML) and other meta-learning
approaches for fast adaptation to new market conditions and instruments:

1. MAML (Model-Agnostic Meta-Learning) - Learn to learn new tasks quickly
2. Prototypical Networks - Learn market regime prototypes
3. Memory-Augmented Networks - Store and retrieve market patterns
4. Continual Learning - Adapt without forgetting previous knowledge
5. Few-Shot Learning - Learn from limited examples in new markets
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
from pathlib import Path
import warnings
from collections import OrderedDict, defaultdict
import copy

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class MetaLearningConfig:
    """Configuration for Meta-Learning Framework."""
    
    # MAML parameters
    meta_lr: float = 0.001  # Meta-learning rate
    task_lr: float = 0.01   # Task-specific learning rate
    meta_batch_size: int = 8  # Number of tasks per meta-batch
    num_inner_steps: int = 5  # Inner loop gradient steps
    num_meta_epochs: int = 100
    
    # Model architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = 'relu'
    dropout: float = 0.1
    
    # Task generation
    support_size: int = 50   # Number of samples for adaptation
    query_size: int = 20     # Number of samples for evaluation
    min_task_length: int = 100  # Minimum samples per task
    max_task_length: int = 500  # Maximum samples per task
    
    # Market regime clustering
    num_prototypes: int = 10  # Number of market regime prototypes
    prototype_dim: int = 128
    
    # Memory parameters
    memory_size: int = 1000
    memory_key_dim: int = 64
    memory_value_dim: int = 128
    
    # Continual learning
    ewc_lambda: float = 1000.0  # Elastic Weight Consolidation regularization
    replay_buffer_size: int = 5000
    
    # Training parameters
    patience: int = 10
    min_delta: float = 1e-4
    device: str = 'auto'
    
    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MetaLearner(nn.Module):
    """Base meta-learning model architecture."""
    
    def __init__(self, input_size: int, output_size: int, config: MetaLearningConfig):
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        
        # Build network layers
        layers = []
        current_size = input_size
        
        for hidden_size in config.hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            
            if config.activation == 'relu':
                layers.append(nn.ReLU())
            elif config.activation == 'gelu':
                layers.append(nn.GELU())
            elif config.activation == 'tanh':
                layers.append(nn.Tanh())
            
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            current_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def get_flat_params(self) -> torch.Tensor:
        """Get flattened model parameters."""
        return torch.cat([p.view(-1) for p in self.parameters()])
    
    def set_flat_params(self, flat_params: torch.Tensor):
        """Set model parameters from flattened tensor."""
        offset = 0
        for param in self.parameters():
            param_length = param.numel()
            param.data = flat_params[offset:offset + param_length].view(param.shape)
            offset += param_length
    
    def get_named_params(self) -> OrderedDict:
        """Get ordered dictionary of parameters."""
        return OrderedDict(self.named_parameters())
    
    def set_named_params(self, named_params: OrderedDict):
        """Set parameters from ordered dictionary."""
        for name, param in self.named_parameters():
            param.data = named_params[name].data

class MAML:
    """Model-Agnostic Meta-Learning implementation."""
    
    def __init__(self, model: MetaLearner, config: MetaLearningConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Meta-optimizer
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=config.meta_lr)
        
        # Training history
        self.meta_train_losses = []
        self.meta_val_losses = []
        
    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor, 
              num_steps: int = None) -> MetaLearner:
        """
        Adapt model to a new task using support data.
        
        Args:
            support_x: Support set features
            support_y: Support set targets
            num_steps: Number of gradient steps (default: config.num_inner_steps)
        
        Returns:
            Adapted model
        """
        if num_steps is None:
            num_steps = self.config.num_inner_steps
        
        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()
        
        # Task-specific optimizer
        task_optimizer = optim.SGD(adapted_model.parameters(), lr=self.config.task_lr)
        
        # Inner loop: adapt to task
        for step in range(num_steps):
            task_optimizer.zero_grad()
            
            # Forward pass
            predictions = adapted_model(support_x)
            
            # Compute loss
            loss = F.mse_loss(predictions, support_y)
            
            # Backward pass
            loss.backward()
            task_optimizer.step()
        
        return adapted_model
    
    def meta_update(self, tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
        """
        Perform one meta-update step.
        
        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples
        
        Returns:
            Meta-loss value
        """
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Move to device
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            # Adapt model to task
            adapted_model = self.adapt(support_x, support_y)
            
            # Evaluate on query set
            query_predictions = adapted_model(query_x)
            task_loss = F.mse_loss(query_predictions, query_y)
            
            meta_loss += task_loss
        
        # Average over tasks
        meta_loss = meta_loss / len(tasks)
        
        # Meta-gradient update
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def train(self, task_generator, num_epochs: int = None, validation_tasks: List = None) -> Dict[str, List[float]]:
        """
        Train the meta-learner.
        
        Args:
            task_generator: Generator yielding batches of tasks
            num_epochs: Number of meta-training epochs
            validation_tasks: Optional validation tasks
        
        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = self.config.num_meta_epochs
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting MAML training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for task_batch in task_generator:
                if len(task_batch) < self.config.meta_batch_size:
                    continue
                    
                batch_loss = self.meta_update(task_batch)
                epoch_loss += batch_loss
                num_batches += 1
                
                # Break after reasonable number of batches per epoch
                if num_batches >= 10:  # Adjust as needed
                    break
            
            avg_train_loss = epoch_loss / max(num_batches, 1)
            self.meta_train_losses.append(avg_train_loss)
            
            # Validation
            if validation_tasks:
                val_loss = self.evaluate(validation_tasks)
                self.meta_val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss - self.config.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {avg_train_loss:.6f}, "
                           f"Val Loss: {val_loss:.6f}")
                
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}")
        
        return {
            'train_losses': self.meta_train_losses,
            'val_losses': self.meta_val_losses
        }
    
    def evaluate(self, tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
        """Evaluate meta-learner on validation tasks."""
        self.model.eval()
        
        total_loss = 0.0
        
        with torch.no_grad():
            for support_x, support_y, query_x, query_y in tasks:
                # Move to device
                support_x = support_x.to(self.device)
                support_y = support_y.to(self.device)
                query_x = query_x.to(self.device)
                query_y = query_y.to(self.device)
                
                # Adapt model
                adapted_model = self.adapt(support_x, support_y)
                adapted_model.eval()
                
                # Evaluate
                query_predictions = adapted_model(query_x)
                loss = F.mse_loss(query_predictions, query_y)
                total_loss += loss.item()
        
        return total_loss / len(tasks)

class PrototypicalNetworks:
    """Prototypical Networks for market regime classification."""
    
    def __init__(self, feature_dim: int, config: MetaLearningConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Embedding network
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_sizes[0], config.prototype_dim)
        ).to(self.device)
        
        # Prototypes (will be computed from support set)
        self.prototypes = None
        self.regime_labels = None
        
        # Training components
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=config.meta_lr)
        
    def compute_prototypes(self, support_features: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """Compute prototype embeddings for each regime."""
        
        # Encode support features
        support_embeddings = self.encoder(support_features)
        
        # Compute prototypes as mean of embeddings per class
        unique_labels = torch.unique(support_labels)
        prototypes = []
        
        for label in unique_labels:
            mask = (support_labels == label)
            class_embeddings = support_embeddings[mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def predict(self, query_features: torch.Tensor, support_features: torch.Tensor, 
                support_labels: torch.Tensor) -> torch.Tensor:
        """Predict regime labels for query features."""
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_features, support_labels)
        
        # Encode query features
        query_embeddings = self.encoder(query_features)
        
        # Compute distances to prototypes
        distances = torch.cdist(query_embeddings, prototypes)
        
        # Predict closest prototype
        predictions = torch.argmin(distances, dim=1)
        
        return predictions
    
    def train_episode(self, support_features: torch.Tensor, support_labels: torch.Tensor,
                     query_features: torch.Tensor, query_labels: torch.Tensor) -> float:
        """Train on one episode."""
        
        self.optimizer.zero_grad()
        
        # Move to device
        support_features = support_features.to(self.device)
        support_labels = support_labels.to(self.device)
        query_features = query_features.to(self.device)
        query_labels = query_labels.to(self.device)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_features, support_labels)
        
        # Encode query features
        query_embeddings = self.encoder(query_features)
        
        # Compute log probabilities
        distances = torch.cdist(query_embeddings, prototypes)
        log_probs = -F.log_softmax(distances, dim=1)
        
        # Compute loss
        loss = F.nll_loss(log_probs, query_labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class MemoryAugmentedNetwork:
    """Memory-Augmented Network for storing and retrieving market patterns."""
    
    def __init__(self, input_dim: int, output_dim: int, config: MetaLearningConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Controller network
        self.controller = nn.Sequential(
            nn.Linear(input_dim, config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_sizes[0], output_dim)
        ).to(self.device)
        
        # Memory components
        self.memory_keys = nn.Parameter(torch.randn(config.memory_size, config.memory_key_dim))
        self.memory_values = nn.Parameter(torch.randn(config.memory_size, config.memory_value_dim))
        
        # Key and value generators
        self.key_generator = nn.Linear(input_dim, config.memory_key_dim).to(self.device)
        self.value_generator = nn.Linear(input_dim, config.memory_value_dim).to(self.device)
        
        # Output projection
        self.output_projection = nn.Linear(
            config.hidden_sizes[0] + config.memory_value_dim, output_dim
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.parameters(), lr=config.meta_lr)
    
    def parameters(self):
        """Get all trainable parameters."""
        for module in [self.controller, self.key_generator, self.value_generator, self.output_projection]:
            for param in module.parameters():
                yield param
        yield self.memory_keys
        yield self.memory_values
    
    def read_memory(self, query_key: torch.Tensor) -> torch.Tensor:
        """Read from memory using attention mechanism."""
        
        # Compute attention weights
        similarities = torch.matmul(query_key, self.memory_keys.T)
        attention_weights = F.softmax(similarities, dim=-1)
        
        # Weighted sum of memory values
        retrieved_values = torch.matmul(attention_weights, self.memory_values)
        
        return retrieved_values
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory augmentation."""
        
        # Generate query key
        query_key = self.key_generator(x)
        
        # Read from memory
        memory_output = self.read_memory(query_key)
        
        # Controller output
        controller_output = self.controller(x)
        
        # Combine controller and memory outputs
        combined = torch.cat([controller_output, memory_output], dim=-1)
        output = self.output_projection(combined)
        
        return output
    
    def update_memory(self, x: torch.Tensor, target: torch.Tensor):
        """Update memory with new patterns."""
        
        # Generate new key-value pairs
        new_key = self.key_generator(x)
        new_value = self.value_generator(x)
        
        # Find least used memory slot (simple strategy)
        # In practice, you might use more sophisticated memory management
        memory_usage = torch.sum(torch.abs(self.memory_values), dim=1)
        update_idx = torch.argmin(memory_usage)
        
        # Update memory
        self.memory_keys.data[update_idx] = new_key.mean(dim=0)
        self.memory_values.data[update_idx] = new_value.mean(dim=0)

class ContinualLearner:
    """Continual learning with Elastic Weight Consolidation (EWC)."""
    
    def __init__(self, model: MetaLearner, config: MetaLearningConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # EWC components
        self.fisher_information = {}
        self.optimal_params = {}
        self.task_id = 0
        
        # Replay buffer
        self.replay_buffer = []
        
        self.optimizer = optim.Adam(model.parameters(), lr=config.meta_lr)
    
    def compute_fisher_information(self, data_loader: DataLoader) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information Matrix for EWC."""
        
        fisher_info = {}
        
        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            fisher_info[name] = torch.zeros_like(param.data)
        
        self.model.eval()
        
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            output = self.model(data)
            loss = F.mse_loss(output, target)
            
            # Compute gradients
            self.optimizer.zero_grad()
            loss.backward()
            
            # Accumulate Fisher information
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
        
        # Normalize by number of samples
        num_samples = len(data_loader.dataset)
        for name in fisher_info:
            fisher_info[name] /= num_samples
        
        return fisher_info
    
    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        
        if not self.fisher_information:
            return torch.tensor(0.0, device=self.device)
        
        ewc_loss = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
        
        return self.config.ewc_lambda * ewc_loss
    
    def train_task(self, data_loader: DataLoader, num_epochs: int = 10):
        """Train on a new task with EWC regularization."""
        
        logger.info(f"Training task {self.task_id + 1} with EWC...")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                task_loss = F.mse_loss(output, target)
                
                # Add EWC regularization
                ewc_reg = self.ewc_loss()
                total_loss = task_loss + ewc_reg
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                
                # Add to replay buffer
                if len(self.replay_buffer) < self.config.replay_buffer_size:
                    self.replay_buffer.append((data.cpu(), target.cpu()))
                else:
                    # Replace random sample
                    idx = np.random.randint(len(self.replay_buffer))
                    self.replay_buffer[idx] = (data.cpu(), target.cpu())
            
            logger.info(f"Task {self.task_id + 1}, Epoch {epoch + 1}: Loss = {epoch_loss / len(data_loader):.6f}")
        
        # Update Fisher information and optimal parameters
        self.fisher_information = self.compute_fisher_information(data_loader)
        self.optimal_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        self.task_id += 1

class MarketTaskGenerator:
    """Generate tasks for meta-learning from market data."""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        
    def generate_tasks_from_data(self, market_data: pd.DataFrame, features: pd.DataFrame,
                                targets: pd.DataFrame, num_tasks: int = 100) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate tasks from market data."""
        
        tasks = []
        
        for _ in range(num_tasks):
            # Random time window
            start_idx = np.random.randint(0, len(market_data) - self.config.max_task_length)
            end_idx = start_idx + np.random.randint(self.config.min_task_length, self.config.max_task_length)
            
            # Extract task data
            task_features = features.iloc[start_idx:end_idx].values
            task_targets = targets.iloc[start_idx:end_idx].values
            
            # Split into support and query sets
            total_samples = len(task_features)
            support_indices = np.random.choice(total_samples, self.config.support_size, replace=False)
            query_indices = np.random.choice(total_samples, self.config.query_size, replace=False)
            
            support_x = torch.tensor(task_features[support_indices], dtype=torch.float32)
            support_y = torch.tensor(task_targets[support_indices], dtype=torch.float32)
            query_x = torch.tensor(task_features[query_indices], dtype=torch.float32)
            query_y = torch.tensor(task_targets[query_indices], dtype=torch.float32)
            
            # Ensure proper shapes
            if len(support_y.shape) == 1:
                support_y = support_y.unsqueeze(-1)
            if len(query_y.shape) == 1:
                query_y = query_y.unsqueeze(-1)
            
            tasks.append((support_x, support_y, query_x, query_y))
        
        return tasks
    
    def generate_regime_tasks(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate tasks for regime classification."""
        
        tasks = []
        unique_regimes = np.unique(regime_labels)
        
        for regime in unique_regimes:
            regime_indices = np.where(regime_labels == regime)[0]
            
            if len(regime_indices) < self.config.support_size + self.config.query_size:
                continue
            
            # Sample support and query sets
            selected_indices = np.random.choice(regime_indices, 
                                               self.config.support_size + self.config.query_size, 
                                               replace=False)
            
            support_indices = selected_indices[:self.config.support_size]
            query_indices = selected_indices[self.config.support_size:]
            
            support_x = torch.tensor(market_data.iloc[support_indices].values, dtype=torch.float32)
            support_y = torch.tensor(regime_labels[support_indices], dtype=torch.long)
            query_x = torch.tensor(market_data.iloc[query_indices].values, dtype=torch.float32)
            query_y = torch.tensor(regime_labels[query_indices], dtype=torch.long)
            
            tasks.append((support_x, support_y, query_x, query_y))
        
        return tasks

class MetaLearningFramework:
    """Main Meta-Learning Framework integrating all components."""
    
    def __init__(self, input_size: int, output_size: int, config: MetaLearningConfig = None):
        self.config = config or MetaLearningConfig()
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize components
        self.base_model = MetaLearner(input_size, output_size, self.config)
        self.maml = MAML(self.base_model, self.config)
        self.prototypical_nets = PrototypicalNetworks(input_size, self.config)
        self.memory_net = MemoryAugmentedNetwork(input_size, output_size, self.config)
        self.continual_learner = ContinualLearner(self.base_model, self.config)
        self.task_generator = MarketTaskGenerator(self.config)
        
        # Training history
        self.training_history = {}
        
    def quick_adapt(self, support_data: torch.Tensor, support_targets: torch.Tensor,
                   num_steps: int = 5) -> MetaLearner:
        """Quickly adapt to new market conditions."""
        
        logger.info("Performing quick adaptation to new market conditions...")
        return self.maml.adapt(support_data, support_targets, num_steps)
    
    def classify_market_regime(self, features: torch.Tensor, 
                             support_features: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """Classify current market regime using prototypical networks."""
        
        return self.prototypical_nets.predict(features, support_features, support_labels)
    
    def predict_with_memory(self, features: torch.Tensor) -> torch.Tensor:
        """Make predictions using memory-augmented network."""
        
        return self.memory_net.forward(features)
    
    def train_meta_learner(self, market_data: pd.DataFrame, features: pd.DataFrame, 
                          targets: pd.DataFrame, num_tasks: int = 1000) -> Dict[str, Any]:
        """Train the meta-learning framework."""
        
        logger.info("Training meta-learning framework...")
        
        # Generate tasks
        tasks = self.task_generator.generate_tasks_from_data(
            market_data, features, targets, num_tasks
        )
        
        # Split into train and validation
        split_idx = int(0.8 * len(tasks))
        train_tasks = tasks[:split_idx]
        val_tasks = tasks[split_idx:]
        
        # Create task generator for training
        def task_batch_generator():
            batch = []
            for task in train_tasks:
                batch.append(task)
                if len(batch) >= self.config.meta_batch_size:
                    yield batch
                    batch = []
            if batch:  # Yield remaining tasks
                yield batch
        
        # Train MAML
        maml_history = self.maml.train(task_batch_generator(), validation_tasks=val_tasks)
        
        self.training_history['maml'] = maml_history
        
        logger.info("Meta-learning training completed!")
        
        return self.training_history
    
    def save_framework(self, save_path: str):
        """Save the entire meta-learning framework."""
        
        checkpoint = {
            'config': self.config,
            'base_model_state': self.base_model.state_dict(),
            'prototypical_nets_state': self.prototypical_nets.encoder.state_dict(),
            'memory_net_keys': self.memory_net.memory_keys.data,
            'memory_net_values': self.memory_net.memory_values.data,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Meta-learning framework saved to {save_path}")
    
    def load_framework(self, load_path: str):
        """Load the meta-learning framework."""
        
        checkpoint = torch.load(load_path, map_location=self.config.device)
        
        self.base_model.load_state_dict(checkpoint['base_model_state'])
        self.prototypical_nets.encoder.load_state_dict(checkpoint['prototypical_nets_state'])
        self.memory_net.memory_keys.data = checkpoint['memory_net_keys']
        self.memory_net.memory_values.data = checkpoint['memory_net_values']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Meta-learning framework loaded from {load_path}")

# Factory functions
def create_meta_learning_framework(input_size: int, output_size: int, 
                                 config: MetaLearningConfig = None) -> MetaLearningFramework:
    """Create meta-learning framework with default or custom configuration."""
    return MetaLearningFramework(input_size, output_size, config)

def quick_market_adaptation(model: MetaLearner, support_data: torch.Tensor, 
                           support_targets: torch.Tensor, config: MetaLearningConfig = None) -> MetaLearner:
    """Quick adaptation to new market conditions."""
    if config is None:
        config = MetaLearningConfig()
    
    maml = MAML(model, config)
    return maml.adapt(support_data, support_targets)

# Global meta-learning framework instance
meta_framework = None

def get_meta_framework(input_size: int = None, output_size: int = None) -> MetaLearningFramework:
    """Get global meta-learning framework instance."""
    global meta_framework
    if meta_framework is None and input_size is not None and output_size is not None:
        meta_framework = create_meta_learning_framework(input_size, output_size)
    return meta_framework
