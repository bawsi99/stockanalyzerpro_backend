"""
Model Registry for Quantitative Trading System

This module provides a centralized registry for managing all trained models,
scalers, and related metadata in the system.
"""

import logging
import json
import pickle
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    
    name: str
    model_type: str
    created_at: datetime
    last_updated: datetime
    version: str = "1.0.0"
    description: str = ""
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'name': self.name,
            'model_type': self.model_type,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'version': self.version,
            'description': self.description,
            'performance_metrics': self.performance_metrics,
            'training_config': self.training_config,
            'feature_columns': self.feature_columns,
            'dependencies': self.dependencies
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary."""
        return cls(
            name=data['name'],
            model_type=data['model_type'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_updated=datetime.fromisoformat(data['last_updated']),
            version=data.get('version', '1.0.0'),
            description=data.get('description', ''),
            performance_metrics=data.get('performance_metrics', {}),
            training_config=data.get('training_config', {}),
            feature_columns=data.get('feature_columns', []),
            dependencies=data.get('dependencies', [])
        )

class ModelRegistry:
    """Centralized registry for all models and related components."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        self.feature_columns: Dict[str, List[str]] = {}
        self.training_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Load existing registry if storage path exists
        if self.storage_path and self.storage_path.exists():
            self.load_registry()
        
        logger.info("Model registry initialized")
    
    def register_model(self, name: str, model: Any, 
                      scaler: Any = None,
                      feature_columns: List[str] = None,
                      metadata: Optional[ModelMetadata] = None,
                      training_info: Dict[str, Any] = None) -> bool:
        """Register a trained model with all related components."""
        try:
            # Store the model
            self.models[name] = model
            
            # Store scaler if provided
            if scaler is not None:
                self.scalers[name] = scaler
            
            # Store feature columns
            if feature_columns is not None:
                self.feature_columns[name] = feature_columns.copy()
            
            # Create or update metadata
            if metadata is None:
                metadata = ModelMetadata(
                    name=name,
                    model_type=type(model).__name__,
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    feature_columns=feature_columns or []
                )
            else:
                metadata.last_updated = datetime.now()
            
            self.metadata[name] = metadata
            
            # Store training history
            if training_info is not None:
                if name not in self.training_history:
                    self.training_history[name] = []
                self.training_history[name].append({
                    'timestamp': datetime.now().isoformat(),
                    'info': training_info
                })
            
            logger.info(f"Model '{name}' registered successfully")
            
            # Save registry if storage path is configured
            if self.storage_path:
                self.save_registry()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model '{name}': {e}")
            return False
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a registered model."""
        return self.models.get(name)
    
    def get_scaler(self, name: str) -> Optional[Any]:
        """Get a registered scaler."""
        return self.scalers.get(name)
    
    def get_metadata(self, name: str) -> Optional[ModelMetadata]:
        """Get model metadata."""
        return self.metadata.get(name)
    
    def get_feature_columns(self, name: str) -> Optional[List[str]]:
        """Get feature columns for a model."""
        return self.feature_columns.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())
    
    def list_models_by_type(self, model_type: str) -> List[str]:
        """List models of a specific type."""
        return [name for name, metadata in self.metadata.items() 
                if metadata.model_type == model_type]
    
    def remove_model(self, name: str) -> bool:
        """Remove a model and all its related data."""
        try:
            if name in self.models:
                del self.models[name]
            
            if name in self.scalers:
                del self.scalers[name]
            
            if name in self.metadata:
                del self.metadata[name]
            
            if name in self.feature_columns:
                del self.feature_columns[name]
            
            if name in self.training_history:
                del self.training_history[name]
            
            logger.info(f"Model '{name}' removed successfully")
            
            # Save registry if storage path is configured
            if self.storage_path:
                self.save_registry()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove model '{name}': {e}")
            return False
    
    def update_metadata(self, name: str, **kwargs) -> bool:
        """Update model metadata."""
        try:
            if name not in self.metadata:
                logger.error(f"Model '{name}' not found in registry")
                return False
            
            metadata = self.metadata[name]
            for key, value in kwargs.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
                else:
                    logger.warning(f"Unknown metadata field: {key}")
            
            metadata.last_updated = datetime.now()
            self.metadata[name] = metadata
            
            logger.info(f"Metadata updated for model '{name}'")
            
            # Save registry if storage path is configured
            if self.storage_path:
                self.save_registry()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update metadata for model '{name}': {e}")
            return False
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of the entire registry."""
        return {
            'total_models': len(self.models),
            'model_types': list(set(meta.model_type for meta in self.metadata.values())),
            'models': {
                name: {
                    'type': meta.model_type,
                    'created_at': meta.created_at.isoformat(),
                    'last_updated': meta.last_updated.isoformat(),
                    'version': meta.version,
                    'has_scaler': name in self.scalers,
                    'feature_count': len(self.feature_columns.get(name, [])),
                    'training_sessions': len(self.training_history.get(name, []))
                }
                for name, meta in self.metadata.items()
            }
        }
    
    def save_registry(self) -> bool:
        """Save registry to storage."""
        if not self.storage_path:
            logger.warning("No storage path configured for registry")
            return False
        
        try:
            # Create storage directory if it doesn't exist
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Save models
            models_path = self.storage_path / "models"
            models_path.mkdir(exist_ok=True)
            
            for name, model in self.models.items():
                model_path = models_path / f"{name}.joblib"
                joblib.dump(model, model_path)
            
            # Save scalers
            scalers_path = self.storage_path / "scalers"
            scalers_path.mkdir(exist_ok=True)
            
            for name, scaler in self.scalers.items():
                scaler_path = scalers_path / f"{name}.joblib"
                joblib.dump(scaler, scaler_path)
            
            # Save metadata
            metadata_path = self.storage_path / "metadata.json"
            metadata_dict = {name: meta.to_dict() for name, meta in self.metadata.items()}
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            # Save feature columns
            features_path = self.storage_path / "feature_columns.json"
            with open(features_path, 'w') as f:
                json.dump(self.feature_columns, f, indent=2)
            
            # Save training history
            history_path = self.storage_path / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            logger.info(f"Registry saved to {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            return False
    
    def load_registry(self) -> bool:
        """Load registry from storage."""
        if not self.storage_path or not self.storage_path.exists():
            logger.warning("No storage path or registry not found")
            return False
        
        try:
            # Load metadata
            metadata_path = self.storage_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                self.metadata = {
                    name: ModelMetadata.from_dict(data) 
                    for name, data in metadata_dict.items()
                }
            
            # Load feature columns
            features_path = self.storage_path / "feature_columns.json"
            if features_path.exists():
                with open(features_path, 'r') as f:
                    self.feature_columns = json.load(f)
            
            # Load training history
            history_path = self.storage_path / "training_history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    self.training_history = json.load(f)
            
            # Load models (lazy loading - only when requested)
            self._models_loaded = False
            self._scalers_loaded = False
            
            logger.info(f"Registry loaded from {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return False
    
    def _load_model(self, name: str) -> Optional[Any]:
        """Lazy load a model from storage."""
        if not self.storage_path:
            return None
        
        try:
            model_path = self.storage_path / "models" / f"{name}.joblib"
            if model_path.exists():
                return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model '{name}': {e}")
        return None
    
    def _load_scaler(self, name: str) -> Optional[Any]:
        """Lazy load a scaler from storage."""
        if not self.storage_path:
            return None
        
        try:
            scaler_path = self.storage_path / "scalers" / f"{name}.joblib"
            if scaler_path.exists():
                return joblib.load(scaler_path)
        except Exception as e:
            logger.error(f"Failed to load scaler '{name}': {e}")
        return None

# Global registry instance
global_registry = ModelRegistry()

# Backward compatibility
MLModelRegistry = ModelRegistry  # For existing code compatibility
