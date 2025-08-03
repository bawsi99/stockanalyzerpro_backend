"""
Storage Configuration for Deployment

This module provides centralized storage path management for different
deployment environments and storage backends.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from deployment_config import DeploymentConfig

class StorageConfig:
    """Centralized storage configuration for deployment environments."""
    
    # Environment detection
    @classmethod
    def get_environment(cls) -> str:
        """Get current environment."""
        return os.getenv("ENVIRONMENT", "development").lower()
    
    # Storage type detection
    @classmethod
    def get_storage_type(cls) -> str:
        """Get current storage type."""
        return os.getenv("STORAGE_TYPE", "local").lower()
    
    # Base storage configurations
    STORAGE_CONFIGS = {
        "development": {
            "type": "local",
            "base_path": "./output",
            "charts_path": "./output/charts",
            "analysis_path": "./output",
            "datasets_path": "./analysis_datasets",
            "enhanced_data_path": "./enhanced_sector_data",
            "logs_path": "./logs",
            "cache_path": "./cache"
        },
        "staging": {
            "type": "local",
            "base_path": "/app/data",
            "charts_path": "/app/data/charts",
            "analysis_path": "/app/data/analysis",
            "datasets_path": "/app/data/datasets",
            "enhanced_data_path": "/app/data/enhanced_sector_data",
            "logs_path": "/app/data/logs",
            "cache_path": "/app/data/cache"
        },
        "production": {
            "type": "local",  # Can be changed to "cloud" or "persistent"
            "base_path": "/app/data",
            "charts_path": "/app/data/charts",
            "analysis_path": "/app/data/analysis",
            "datasets_path": "/app/data/datasets",
            "enhanced_data_path": "/app/data/enhanced_sector_data",
            "logs_path": "/app/data/logs",
            "cache_path": "/app/data/cache"
        }
    }
    
    @classmethod
    def get_storage_config(cls) -> Dict[str, Any]:
        """Get storage configuration for current environment."""
        environment = cls.get_environment()
        config = cls.STORAGE_CONFIGS.get(environment, cls.STORAGE_CONFIGS["development"]).copy()
        
        # Override with environment variables if present
        config.update({
            "type": cls.get_storage_type(),
            "base_path": os.getenv("STORAGE_BASE_PATH", config["base_path"]),
            "charts_path": os.getenv("STORAGE_CHARTS_PATH", config["charts_path"]),
            "analysis_path": os.getenv("STORAGE_ANALYSIS_PATH", config["analysis_path"]),
            "datasets_path": os.getenv("STORAGE_DATASETS_PATH", config["datasets_path"]),
            "enhanced_data_path": os.getenv("STORAGE_ENHANCED_DATA_PATH", config["enhanced_data_path"]),
            "logs_path": os.getenv("STORAGE_LOGS_PATH", config["logs_path"]),
            "cache_path": os.getenv("STORAGE_CACHE_PATH", config["cache_path"])
        })
        
        return config
    
    @classmethod
    def get_charts_path(cls) -> str:
        """Get charts storage path."""
        config = cls.get_storage_config()
        return config["charts_path"]
    
    @classmethod
    def get_analysis_path(cls) -> str:
        """Get analysis results storage path."""
        config = cls.get_storage_config()
        return config["analysis_path"]
    
    @classmethod
    def get_datasets_path(cls) -> str:
        """Get datasets storage path."""
        config = cls.get_storage_config()
        return config["datasets_path"]
    
    @classmethod
    def get_enhanced_data_path(cls) -> str:
        """Get enhanced sector data storage path."""
        config = cls.get_storage_config()
        return config["enhanced_data_path"]
    
    @classmethod
    def get_logs_path(cls) -> str:
        """Get logs storage path."""
        config = cls.get_storage_config()
        return config["logs_path"]
    
    @classmethod
    def get_cache_path(cls) -> str:
        """Get cache storage path."""
        config = cls.get_storage_config()
        return config["cache_path"]
    
    @classmethod
    def ensure_directories_exist(cls) -> None:
        """Ensure all storage directories exist."""
        config = cls.get_storage_config()
        
        for key, path in config.items():
            if key.endswith("_path") and path:
                Path(path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_storage_info(cls) -> Dict[str, Any]:
        """Get comprehensive storage information."""
        config = cls.get_storage_config()
        
        # Check if directories exist and get sizes
        storage_info = {
            "environment": cls.get_environment(),
            "storage_type": config["type"],
            "paths": config,
            "directory_status": {}
        }
        
        for key, path in config.items():
            if key.endswith("_path") and path:
                path_obj = Path(path)
                storage_info["directory_status"][key] = {
                    "exists": path_obj.exists(),
                    "is_directory": path_obj.is_dir() if path_obj.exists() else False,
                    "size_mb": cls._get_directory_size_mb(path_obj) if path_obj.exists() else 0
                }
        
        return storage_info
    
    @classmethod
    def _get_directory_size_mb(cls, directory: Path) -> float:
        """Calculate directory size in MB."""
        try:
            total_size = 0
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return round(total_size / (1024 * 1024), 2)
        except Exception:
            return 0.0

# Storage path constants for easy access
STORAGE_PATHS = {
    "CHARTS": StorageConfig.get_charts_path,
    "ANALYSIS": StorageConfig.get_analysis_path,
    "DATASETS": StorageConfig.get_datasets_path,
    "ENHANCED_DATA": StorageConfig.get_enhanced_data_path,
    "LOGS": StorageConfig.get_logs_path,
    "CACHE": StorageConfig.get_cache_path
}

def get_storage_path(storage_type: str) -> str:
    """Get storage path by type."""
    if storage_type.upper() in STORAGE_PATHS:
        return STORAGE_PATHS[storage_type.upper()]()
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")

# Deployment storage recommendations
DEPLOYMENT_STORAGE_RECOMMENDATIONS = {
    "development": {
        "description": "Local development storage",
        "recommendations": [
            "Use relative paths for easy development",
            "Keep all data in project directory",
            "No special permissions required"
        ]
    },
    "staging": {
        "description": "Staging environment storage",
        "recommendations": [
            "Use absolute paths in container",
            "Mount persistent volumes",
            "Ensure proper permissions"
        ]
    },
    "production": {
        "description": "Production environment storage",
        "recommendations": [
            "Use persistent volumes or cloud storage",
            "Implement backup strategies",
            "Monitor storage usage",
            "Consider distributed storage for scaling"
        ]
    }
}

def get_storage_recommendations() -> Dict[str, Any]:
    """Get storage recommendations for current environment."""
    env = StorageConfig.get_environment()
    return {
        "current_environment": env,
        "recommendations": DEPLOYMENT_STORAGE_RECOMMENDATIONS.get(env, DEPLOYMENT_STORAGE_RECOMMENDATIONS["development"]),
        "config": StorageConfig.get_storage_config()
    } 