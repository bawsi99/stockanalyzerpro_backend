"""
Advanced Training Strategies for Trading Models

This module implements sophisticated training methodologies to improve
model robustness, generalization, and performance:

1. Curriculum Learning - Progressive difficulty training
2. Adversarial Training - Robustness against perturbations
3. Self-Supervised Learning - Learn from unlabeled data
4. Contrastive Learning - Learn discriminative features
5. Knowledge Distillation - Transfer knowledge between models
6. Regularization Techniques - Advanced regularization methods
7. Multi-Task Learning - Joint training on related tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import logging
import json
import time
import math
from pathlib import Path
import warnings
from collections import defaultdict
import copy

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class AdvancedTrainingConfig:
    """Configuration for advanced training strategies."""
    
    # Curriculum Learning
    curriculum_enabled: bool = True
    curriculum_strategy: str = 'difficulty_based'  # 'difficulty_based', 'temporal', 'volatility_based'
    curriculum_pace: str = 'linear'  # 'linear', 'exponential', 'step'
    curriculum_start_ratio: float = 0.3  # Start with 30% of easiest data
    curriculum_epochs_per_stage: int = 10
    
    # Adversarial Training
    adversarial_enabled: bool = True
    adversarial_method: str = 'fgsm'  # 'fgsm', 'pgd', 'gaussian_noise'
    adversarial_epsilon: float = 0.01
    adversarial_alpha: float = 0.005
    adversarial_steps: int = 7
    adversarial_ratio: float = 0.3  # Ratio of adversarial samples
    
    # Self-Supervised Learning
    ssl_enabled: bool = True
    ssl_method: str = 'masked_prediction'  # 'masked_prediction', 'future_prediction', 'contrastive'
    ssl_mask_ratio: float = 0.15
    ssl_weight: float = 0.3  # Weight for SSL loss
    
    # Contrastive Learning
    contrastive_enabled: bool = True
    temperature: float = 0.1
    contrastive_batch_size: int = 64
    num_negatives: int = 16
    
    # Knowledge Distillation
    distillation_enabled: bool = False
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.5  # Weight for distillation loss
    
    # Regularization
    mixup_enabled: bool = True
    mixup_alpha: float = 0.2
    cutmix_enabled: bool = True
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.1
    dropout_schedule: bool = True
    
    # Multi-Task Learning
    multitask_enabled: bool = True
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        'direction': 1.0,
        'magnitude': 0.8,
        'volatility': 0.6,
        'regime': 0.4
    })
    
    # Training parameters
    batch_size: int = 128
    learning_rate: float = 0.001
    num_epochs: int = 100
    warmup_epochs: int = 10
    patience: int = 15
    min_delta: float = 1e-5
    
    # Hardware
    device: str = 'auto'
    
    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CurriculumLearner:
    """Curriculum Learning implementation for progressive training."""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        
    def compute_sample_difficulty(self, features: torch.Tensor, targets: torch.Tensor, 
                                 model: nn.Module = None) -> torch.Tensor:
        """Compute difficulty score for each sample."""
        
        if self.config.curriculum_strategy == 'volatility_based':
            # Use price volatility as difficulty measure
            if features.shape[-1] >= 5:  # Assuming OHLCV data
                high_col = 1  # High price column
                low_col = 2   # Low price column
                volatility = (features[:, high_col] - features[:, low_col]) / features[:, low_col]
                difficulty = volatility
            else:
                # Fallback: use target variance
                if len(targets.shape) > 1:
                    difficulty = torch.var(targets, dim=1)
                else:
                    difficulty = torch.abs(targets - targets.mean())
        
        elif self.config.curriculum_strategy == 'temporal':
            # Older samples are easier
            difficulty = torch.arange(len(features), dtype=torch.float32) / len(features)
        
        elif self.config.curriculum_strategy == 'difficulty_based' and model is not None:
            # Use model prediction confidence as difficulty
            model.eval()
            with torch.no_grad():
                predictions = model(features)
                # Higher prediction variance = more difficult
                if len(predictions.shape) > 1:
                    difficulty = torch.var(predictions, dim=1)
                else:
                    difficulty = torch.abs(predictions - targets.squeeze())
        
        else:
            # Random difficulty as fallback
            difficulty = torch.rand(len(features))
        
        return difficulty
    
    def create_curriculum_schedule(self, total_samples: int, num_stages: int = None) -> List[int]:
        """Create curriculum schedule for progressive training."""
        
        if num_stages is None:
            num_stages = max(1, self.config.num_epochs // self.config.curriculum_epochs_per_stage)
        
        schedule = []
        start_samples = int(total_samples * self.config.curriculum_start_ratio)
        
        for stage in range(num_stages):
            if self.config.curriculum_pace == 'linear':
                progress = stage / (num_stages - 1) if num_stages > 1 else 1.0
                num_samples = start_samples + int((total_samples - start_samples) * progress)
            
            elif self.config.curriculum_pace == 'exponential':
                progress = stage / (num_stages - 1) if num_stages > 1 else 1.0
                exp_progress = (math.exp(progress) - 1) / (math.e - 1)
                num_samples = start_samples + int((total_samples - start_samples) * exp_progress)
            
            elif self.config.curriculum_pace == 'step':
                if stage < num_stages // 2:
                    num_samples = start_samples
                else:
                    num_samples = total_samples
            
            schedule.append(min(num_samples, total_samples))
        
        return schedule
    
    def get_curriculum_subset(self, dataset: TensorDataset, difficulty_scores: torch.Tensor,
                             num_samples: int) -> Subset:
        """Get subset of dataset based on curriculum schedule."""
        
        # Sort by difficulty (easiest first)
        sorted_indices = torch.argsort(difficulty_scores)
        selected_indices = sorted_indices[:num_samples].tolist()
        
        return Subset(dataset, selected_indices)

class AdversarialTrainer:
    """Adversarial Training for model robustness."""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        
    def fgsm_attack(self, model: nn.Module, features: torch.Tensor, targets: torch.Tensor,
                   epsilon: float = None) -> torch.Tensor:
        """Fast Gradient Sign Method attack."""
        
        if epsilon is None:
            epsilon = self.config.adversarial_epsilon
        
        features.requires_grad_(True)
        
        # Forward pass
        outputs = model(features)
        loss = F.mse_loss(outputs, targets)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        data_grad = features.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_features = features + epsilon * sign_data_grad
        
        return perturbed_features.detach()
    
    def pgd_attack(self, model: nn.Module, features: torch.Tensor, targets: torch.Tensor,
                  epsilon: float = None, alpha: float = None, num_steps: int = None) -> torch.Tensor:
        """Projected Gradient Descent attack."""
        
        if epsilon is None:
            epsilon = self.config.adversarial_epsilon
        if alpha is None:
            alpha = self.config.adversarial_alpha
        if num_steps is None:
            num_steps = self.config.adversarial_steps
        
        # Initialize perturbation
        delta = torch.zeros_like(features).uniform_(-epsilon, epsilon)
        delta.requires_grad_(True)
        
        for _ in range(num_steps):
            # Forward pass
            outputs = model(features + delta)
            loss = F.mse_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update perturbation
            grad = delta.grad.detach()
            delta.data = delta.data + alpha * grad.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.grad.zero_()
        
        return (features + delta).detach()
    
    def gaussian_noise_attack(self, features: torch.Tensor, epsilon: float = None) -> torch.Tensor:
        """Add Gaussian noise to features."""
        
        if epsilon is None:
            epsilon = self.config.adversarial_epsilon
        
        noise = torch.randn_like(features) * epsilon
        return features + noise
    
    def generate_adversarial_examples(self, model: nn.Module, features: torch.Tensor,
                                    targets: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using configured method."""
        
        if self.config.adversarial_method == 'fgsm':
            return self.fgsm_attack(model, features, targets)
        elif self.config.adversarial_method == 'pgd':
            return self.pgd_attack(model, features, targets)
        elif self.config.adversarial_method == 'gaussian_noise':
            return self.gaussian_noise_attack(features)
        else:
            raise ValueError(f"Unknown adversarial method: {self.config.adversarial_method}")

class SelfSupervisedLearner:
    """Self-Supervised Learning for representation learning."""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        
    def masked_prediction_task(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Masked feature prediction task."""
        
        batch_size, seq_len = features.shape
        
        # Create random mask
        mask_prob = self.config.ssl_mask_ratio
        mask = torch.rand(batch_size, seq_len) < mask_prob
        
        # Create masked features
        masked_features = features.clone()
        masked_features[mask] = 0.0  # Mask with zeros
        
        # Target is original features at masked positions
        targets = features[mask]
        
        return masked_features, targets, mask
    
    def future_prediction_task(self, features: torch.Tensor, horizon: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Future value prediction task."""
        
        # Input: current and past features
        input_features = features[:-horizon]
        
        # Target: future features
        future_targets = features[horizon:]
        
        return input_features, future_targets
    
    def contrastive_task(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Contrastive learning task."""
        
        batch_size = features.shape[0]
        
        # Create positive pairs (same sample with different augmentations)
        # For simplicity, use slight noise as augmentation
        augmented_features = features + torch.randn_like(features) * 0.01
        
        # Positive pairs
        positives = torch.cat([features, augmented_features], dim=0)
        
        # Create labels for contrastive loss
        labels = torch.arange(batch_size).repeat(2)
        
        return positives, labels, torch.tensor(list(range(batch_size)) + list(range(batch_size)))

class ContrastiveLearner:
    """Contrastive Learning implementation."""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        
    def nt_xent_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Normalized Temperature-Scaled Cross-Entropy Loss."""
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.config.temperature
        
        # Create mask for positive pairs
        batch_size = embeddings.shape[0]
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        
        # Remove self-similarities
        mask = mask.fill_diagonal_(False)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        
        log_prob = similarity_matrix - torch.log(sum_exp_sim)
        mean_log_prob_pos = torch.sum(mask * log_prob, dim=1) / torch.sum(mask, dim=1)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss

class KnowledgeDistiller:
    """Knowledge Distillation for model compression and transfer."""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        
    def distillation_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_outputs / self.config.distillation_temperature, dim=1)
        soft_predictions = F.log_softmax(student_outputs / self.config.distillation_temperature, dim=1)
        
        # Distillation loss
        distill_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean')
        distill_loss *= (self.config.distillation_temperature ** 2)
        
        # Hard target loss
        hard_loss = F.cross_entropy(student_outputs, targets)
        
        # Combined loss
        total_loss = (
            self.config.distillation_alpha * distill_loss +
            (1 - self.config.distillation_alpha) * hard_loss
        )
        
        return total_loss

class RegularizationTechniques:
    """Advanced regularization techniques."""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        
    def mixup(self, features: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """MixUp data augmentation."""
        
        if not self.config.mixup_enabled:
            return features, targets
        
        batch_size = features.shape[0]
        
        # Sample mixing coefficient
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha) if self.config.mixup_alpha > 0 else 1.0
        
        # Random permutation
        indices = torch.randperm(batch_size)
        
        # Mix features and targets
        mixed_features = lam * features + (1 - lam) * features[indices]
        mixed_targets = lam * targets + (1 - lam) * targets[indices]
        
        return mixed_features, mixed_targets
    
    def cutmix(self, features: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """CutMix data augmentation (adapted for financial data)."""
        
        if not self.config.cutmix_enabled:
            return features, targets
        
        batch_size = features.shape[0]
        
        # Sample mixing coefficient
        lam = np.random.beta(self.config.cutmix_alpha, self.config.cutmix_alpha) if self.config.cutmix_alpha > 0 else 1.0
        
        # Random permutation
        indices = torch.randperm(batch_size)
        
        # For 1D financial data, randomly mask some features
        num_features = features.shape[1]
        cut_size = int(num_features * (1 - lam))
        cut_start = np.random.randint(0, num_features - cut_size + 1)
        
        mixed_features = features.clone()
        mixed_features[:, cut_start:cut_start + cut_size] = features[indices, cut_start:cut_start + cut_size]
        
        # Adjust targets based on mix ratio
        mixed_targets = lam * targets + (1 - lam) * targets[indices]
        
        return mixed_features, mixed_targets
    
    def label_smoothing_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Label smoothing for regression (adapted)."""
        
        if self.config.label_smoothing <= 0:
            return F.mse_loss(outputs, targets)
        
        # For regression, add small noise to targets
        smoothed_targets = targets + torch.randn_like(targets) * self.config.label_smoothing
        
        return F.mse_loss(outputs, smoothed_targets)

class MultiTaskLearner:
    """Multi-Task Learning for joint training on related tasks."""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        
    def create_multitask_targets(self, price_data: torch.Tensor, returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create multiple task targets from price data."""
        
        tasks = {}
        
        # Direction prediction (binary classification)
        tasks['direction'] = (returns > 0).float()
        
        # Magnitude prediction (regression)
        tasks['magnitude'] = torch.abs(returns)
        
        # Volatility prediction (regression)
        if len(price_data.shape) > 1 and price_data.shape[1] >= 5:
            # Assuming OHLCV data
            high_prices = price_data[:, 1]
            low_prices = price_data[:, 2]
            close_prices = price_data[:, 3]
            volatility = (high_prices - low_prices) / close_prices
            tasks['volatility'] = volatility
        else:
            # Fallback: rolling standard deviation of returns
            window_size = min(20, len(returns))
            volatility = torch.zeros_like(returns)
            for i in range(window_size, len(returns)):
                volatility[i] = torch.std(returns[i-window_size:i])
            tasks['volatility'] = volatility
        
        # Market regime (classification - simplified)
        # Based on volatility levels
        vol_percentiles = torch.quantile(tasks['volatility'], torch.tensor([0.33, 0.67]))
        regime = torch.zeros_like(tasks['volatility'], dtype=torch.long)
        regime[tasks['volatility'] > vol_percentiles[1]] = 2  # High volatility
        regime[(tasks['volatility'] > vol_percentiles[0]) & (tasks['volatility'] <= vol_percentiles[1])] = 1  # Medium
        tasks['regime'] = regime
        
        return tasks
    
    def multitask_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted multi-task loss."""
        
        total_loss = 0.0
        
        for task_name in outputs:
            if task_name in targets and task_name in self.config.task_weights:
                weight = self.config.task_weights[task_name]
                
                if task_name in ['direction', 'regime']:
                    # Classification tasks
                    task_loss = F.cross_entropy(outputs[task_name], targets[task_name].long())
                else:
                    # Regression tasks
                    task_loss = F.mse_loss(outputs[task_name], targets[task_name])
                
                total_loss += weight * task_loss
        
        return total_loss

class AdvancedTrainer:
    """Main advanced trainer integrating all strategies."""
    
    def __init__(self, model: nn.Module, config: AdvancedTrainingConfig = None):
        self.model = model
        self.config = config or AdvancedTrainingConfig()
        self.device = torch.device(self.config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize strategy components
        self.curriculum_learner = CurriculumLearner(self.config)
        self.adversarial_trainer = AdversarialTrainer(self.config)
        self.ssl_learner = SelfSupervisedLearner(self.config)
        self.contrastive_learner = ContrastiveLearner(self.config)
        self.knowledge_distiller = KnowledgeDistiller(self.config)
        self.regularization = RegularizationTechniques(self.config)
        self.multitask_learner = MultiTaskLearner(self.config)
        
        # Training components
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs)
        
        # Training history
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'adversarial_losses': [],
            'ssl_losses': [],
            'contrastive_losses': []
        }
        
    def train_epoch(self, train_loader: DataLoader, epoch: int, teacher_model: nn.Module = None) -> Dict[str, float]:
        """Train for one epoch using all advanced strategies."""
        
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Apply data augmentation
            if self.config.mixup_enabled or self.config.cutmix_enabled:
                if np.random.random() < 0.5:
                    features, targets = self.regularization.mixup(features, targets)
                else:
                    features, targets = self.regularization.cutmix(features, targets)
            
            # Forward pass
            outputs = self.model(features)
            
            # Main supervised loss
            supervised_loss = self.regularization.label_smoothing_loss(outputs, targets)
            total_loss = supervised_loss
            epoch_metrics['supervised_loss'] += supervised_loss.item()
            
            # Adversarial training
            if self.config.adversarial_enabled and np.random.random() < self.config.adversarial_ratio:
                adv_features = self.adversarial_trainer.generate_adversarial_examples(
                    self.model, features, targets
                )
                adv_outputs = self.model(adv_features)
                adv_loss = F.mse_loss(adv_outputs, targets)
                total_loss += 0.5 * adv_loss
                epoch_metrics['adversarial_loss'] += adv_loss.item()
            
            # Self-supervised learning
            if self.config.ssl_enabled:
                if self.config.ssl_method == 'masked_prediction':
                    masked_features, ssl_targets, mask = self.ssl_learner.masked_prediction_task(features)
                    ssl_outputs = self.model(masked_features)
                    # Fix masking issue - ensure proper indexing
                    if mask.sum() > 0:  # Only if we have masked features
                        masked_outputs = ssl_outputs[mask]
                        if len(masked_outputs.shape) > 1:
                            masked_outputs = masked_outputs.squeeze()
                        if len(ssl_targets.shape) > 1:
                            ssl_targets = ssl_targets.squeeze()
                        ssl_loss = F.mse_loss(masked_outputs, ssl_targets)
                        total_loss += self.config.ssl_weight * ssl_loss
                        epoch_metrics['ssl_loss'] += ssl_loss.item()
            
            # Knowledge distillation
            if self.config.distillation_enabled and teacher_model is not None:
                teacher_model.eval()
                with torch.no_grad():
                    teacher_outputs = teacher_model(features)
                
                distill_loss = self.knowledge_distiller.distillation_loss(
                    outputs, teacher_outputs, targets
                )
                total_loss = distill_loss  # Replace main loss
                epoch_metrics['distillation_loss'] += distill_loss.item()
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_metrics['total_loss'] += total_loss.item()
            num_batches += 1
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return dict(epoch_metrics)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model performance."""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                
                outputs = self.model(features)
                loss = F.mse_loss(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_dataset: TensorDataset, val_dataset: TensorDataset = None,
             teacher_model: nn.Module = None) -> Dict[str, List[float]]:
        """Full training with all advanced strategies."""
        
        logger.info("Starting advanced training...")
        
        # Curriculum learning setup
        if self.config.curriculum_enabled:
            # Compute difficulty scores
            features = train_dataset.tensors[0]
            targets = train_dataset.tensors[1]
            difficulty_scores = self.curriculum_learner.compute_sample_difficulty(
                features, targets, self.model
            )
            
            # Create curriculum schedule
            curriculum_schedule = self.curriculum_learner.create_curriculum_schedule(
                len(train_dataset)
            )
        else:
            curriculum_schedule = [len(train_dataset)] * (self.config.num_epochs // 10)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Curriculum learning: adjust dataset size
            if self.config.curriculum_enabled:
                stage = epoch // self.config.curriculum_epochs_per_stage
                if stage < len(curriculum_schedule):
                    num_samples = curriculum_schedule[stage]
                    current_dataset = self.curriculum_learner.get_curriculum_subset(
                        train_dataset, difficulty_scores, num_samples
                    )
                else:
                    current_dataset = train_dataset
            else:
                current_dataset = train_dataset
            
            # Create data loader
            train_loader = DataLoader(
                current_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True
            )
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch, teacher_model)
            
            # Learning rate scheduling
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            # Validation
            if val_dataset is not None:
                val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
                val_loss = self.validate(val_loader)
                
                self.training_history['val_losses'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss - self.config.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                           f"Train Loss: {train_metrics['total_loss']:.6f}, "
                           f"Val Loss: {val_loss:.6f}")
                
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                           f"Train Loss: {train_metrics['total_loss']:.6f}")
            
            # Store training history
            self.training_history['train_losses'].append(train_metrics['total_loss'])
            if 'adversarial_loss' in train_metrics:
                self.training_history['adversarial_losses'].append(train_metrics['adversarial_loss'])
            if 'ssl_loss' in train_metrics:
                self.training_history['ssl_losses'].append(train_metrics['ssl_loss'])
        
        logger.info("Advanced training completed!")
        return self.training_history

# Factory functions
def create_advanced_trainer(model: nn.Module, config: AdvancedTrainingConfig = None) -> AdvancedTrainer:
    """Create advanced trainer with default or custom configuration."""
    return AdvancedTrainer(model, config)

def train_with_curriculum(model: nn.Module, train_data: torch.Tensor, train_targets: torch.Tensor,
                         val_data: torch.Tensor = None, val_targets: torch.Tensor = None) -> AdvancedTrainer:
    """Quick curriculum learning training."""
    
    config = AdvancedTrainingConfig(
        curriculum_enabled=True,
        adversarial_enabled=False,
        ssl_enabled=False,
        num_epochs=50
    )
    
    trainer = AdvancedTrainer(model, config)
    
    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets) if val_data is not None else None
    
    trainer.train(train_dataset, val_dataset)
    return trainer

def train_with_adversarial(model: nn.Module, train_data: torch.Tensor, train_targets: torch.Tensor,
                          val_data: torch.Tensor = None, val_targets: torch.Tensor = None) -> AdvancedTrainer:
    """Quick adversarial training."""
    
    config = AdvancedTrainingConfig(
        curriculum_enabled=False,
        adversarial_enabled=True,
        ssl_enabled=False,
        num_epochs=50
    )
    
    trainer = AdvancedTrainer(model, config)
    
    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets) if val_data is not None else None
    
    trainer.train(train_dataset, val_dataset)
    return trainer

# Global advanced trainer instance
advanced_trainer = None

def get_advanced_trainer(model: nn.Module = None, config: AdvancedTrainingConfig = None) -> AdvancedTrainer:
    """Get global advanced trainer instance."""
    global advanced_trainer
    if advanced_trainer is None and model is not None:
        advanced_trainer = create_advanced_trainer(model, config)
    return advanced_trainer
