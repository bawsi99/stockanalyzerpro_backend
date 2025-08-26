"""
Deployment Configuration for Chart Management

This module provides configuration settings for chart storage and cleanup
in different deployment environments (development, staging, production).
"""

import os
from typing import Dict, Any

class DeploymentConfig:
    """Configuration settings for different deployment environments."""
    
    # Environment detection
    @classmethod
    def get_environment(cls) -> str:
        """Get current environment."""
        return os.getenv("ENVIRONMENT", "development").lower()
    
    # Chart storage configurations for different environments
    CHART_CONFIGS = {
        "development": {
            "max_age_hours": 24,
            "max_total_size_mb": 1000,
            "cleanup_interval_minutes": 60,
            "base_output_dir": "./output/charts",
            "enable_cleanup": True
        },
        "staging": {
            "max_age_hours": 12,
            "max_total_size_mb": 500,
            "cleanup_interval_minutes": 30,
            "base_output_dir": "/app/data/charts",
            "enable_cleanup": True
        },
        "production": {
            "max_age_hours": 6,
            "max_total_size_mb": 200,
            "cleanup_interval_minutes": 15,
            "base_output_dir": "/app/data/charts",
            "enable_cleanup": True
        }
    }
    
    # Redis image manager configurations for different environments
    REDIS_IMAGE_CONFIGS = {
        "development": {
            "max_age_hours": 24,
            "max_total_size_mb": 1000,
            "cleanup_interval_minutes": 60,
            "enable_cleanup": True,
            "image_quality": 85,
            "image_format": "PNG",
            "redis_url": "redis://localhost:6379/0"
        },
        "staging": {
            "max_age_hours": 12,
            "max_total_size_mb": 500,
            "cleanup_interval_minutes": 30,
            "enable_cleanup": True,
            "image_quality": 85,
            "image_format": "PNG",
            "redis_url": "redis://localhost:6379/0"
        },
        "production": {
            "max_age_hours": 6,
            "max_total_size_mb": 200,
            "cleanup_interval_minutes": 15,
            "enable_cleanup": True,
            "image_quality": 80,
            "image_format": "JPEG",
            "redis_url": "redis://localhost:6379/0"
        }
    }
    
    # Redis cache manager configurations for different environments
    REDIS_CACHE_CONFIGS = {
        "development": {
            "enable_compression": True,
            "enable_local_fallback": True,
            "local_cache_size": 1000,
            "cleanup_interval_minutes": 60,
            "redis_url": "redis://localhost:6379/0",
            "ttl_settings": {
                "stock_data": 300,      # 5 minutes
                "indicators": 600,      # 10 minutes
                "patterns": 1800,       # 30 minutes
                "sector_data": 3600,    # 1 hour
                "ml_predictions": 1800, # 30 minutes
                "api_responses": 300    # 5 minutes
            }
        },
        "staging": {
            "enable_compression": True,
            "enable_local_fallback": True,
            "local_cache_size": 500,
            "cleanup_interval_minutes": 30,
            "redis_url": "redis://localhost:6379/0",
            "ttl_settings": {
                "stock_data": 300,      # 5 minutes
                "indicators": 600,      # 10 minutes
                "patterns": 1800,       # 30 minutes
                "sector_data": 3600,    # 1 hour
                "ml_predictions": 1800, # 30 minutes
                "api_responses": 300    # 5 minutes
            }
        },
        "production": {
            "enable_compression": True,
            "enable_local_fallback": False,  # Disable local fallback in production
            "local_cache_size": 0,
            "cleanup_interval_minutes": 15,
            "redis_url": "redis://localhost:6379/0",
            "ttl_settings": {
                "stock_data": 180,      # 3 minutes
                "indicators": 300,      # 5 minutes
                "patterns": 900,        # 15 minutes
                "sector_data": 1800,    # 30 minutes
                "ml_predictions": 900,  # 15 minutes
                "api_responses": 180    # 3 minutes
            }
        }
    }
    
    @classmethod
    def get_chart_config(cls) -> Dict[str, Any]:
        """Get chart configuration for current environment."""
        environment = cls.get_environment()
        config = cls.CHART_CONFIGS.get(environment, cls.CHART_CONFIGS["development"]).copy()
        
        # Override with environment variables if present
        config.update({
            "max_age_hours": int(os.getenv("CHART_MAX_AGE_HOURS", config["max_age_hours"])),
            "max_total_size_mb": int(os.getenv("CHART_MAX_SIZE_MB", config["max_total_size_mb"])),
            "cleanup_interval_minutes": int(os.getenv("CHART_CLEANUP_INTERVAL_MINUTES", config["cleanup_interval_minutes"])),
            "base_output_dir": os.getenv("CHART_OUTPUT_DIR", config["base_output_dir"]),
            "enable_cleanup": os.getenv("CHART_ENABLE_CLEANUP", "true").lower() == "true"
        })
        
        return config
    
    @classmethod
    def get_redis_image_config(cls) -> Dict[str, Any]:
        """Get Redis image manager configuration for current environment."""
        environment = cls.get_environment()
        config = cls.REDIS_IMAGE_CONFIGS.get(environment, cls.REDIS_IMAGE_CONFIGS["development"]).copy()
        
        # Override with environment variables if present
        config.update({
            "max_age_hours": int(os.getenv("REDIS_IMAGE_MAX_AGE_HOURS", config["max_age_hours"])),
            "max_total_size_mb": int(os.getenv("REDIS_IMAGE_MAX_SIZE_MB", config["max_total_size_mb"])),
            "cleanup_interval_minutes": int(os.getenv("REDIS_IMAGE_CLEANUP_INTERVAL_MINUTES", config["cleanup_interval_minutes"])),
            "enable_cleanup": os.getenv("REDIS_IMAGE_ENABLE_CLEANUP", "true").lower() == "true",
            "image_quality": int(os.getenv("REDIS_IMAGE_QUALITY", config["image_quality"])),
            "image_format": os.getenv("REDIS_IMAGE_FORMAT", config["image_format"]),
            "redis_url": os.getenv("REDIS_URL", config["redis_url"])
        })
        
        return config
    
    @classmethod
    def get_redis_cache_config(cls) -> Dict[str, Any]:
        """Get Redis cache manager configuration for current environment."""
        environment = cls.get_environment()
        config = cls.REDIS_CACHE_CONFIGS.get(environment, cls.REDIS_CACHE_CONFIGS["development"]).copy()
        
        # Override with environment variables if present
        config.update({
            "enable_compression": os.getenv("REDIS_CACHE_ENABLE_COMPRESSION", "true").lower() == "true",
            "enable_local_fallback": os.getenv("REDIS_CACHE_ENABLE_LOCAL_FALLBACK", "true").lower() == "true",
            "local_cache_size": int(os.getenv("REDIS_CACHE_LOCAL_SIZE", config["local_cache_size"])),
            "cleanup_interval_minutes": int(os.getenv("REDIS_CACHE_CLEANUP_INTERVAL_MINUTES", config["cleanup_interval_minutes"])),
            "redis_url": os.getenv("REDIS_URL", config["redis_url"])
        })
        
        return config
    
    @classmethod
    def get_environment_info(cls) -> Dict[str, Any]:
        """Get current environment information."""
        return {
            "environment": cls.get_environment(),
            "chart_config": cls.get_chart_config(),
            "redis_image_config": cls.get_redis_image_config(),
            "redis_cache_config": cls.get_redis_cache_config(),
            "environment_variables": {
                "CHART_MAX_AGE_HOURS": os.getenv("CHART_MAX_AGE_HOURS"),
                "CHART_MAX_SIZE_MB": os.getenv("CHART_MAX_SIZE_MB"),
                "CHART_CLEANUP_INTERVAL_MINUTES": os.getenv("CHART_CLEANUP_INTERVAL_MINUTES"),
                "CHART_OUTPUT_DIR": os.getenv("CHART_OUTPUT_DIR"),
                "CHART_ENABLE_CLEANUP": os.getenv("CHART_ENABLE_CLEANUP"),
                "REDIS_IMAGE_MAX_AGE_HOURS": os.getenv("REDIS_IMAGE_MAX_AGE_HOURS"),
                "REDIS_IMAGE_MAX_SIZE_MB": os.getenv("REDIS_IMAGE_MAX_SIZE_MB"),
                "REDIS_IMAGE_CLEANUP_INTERVAL_MINUTES": os.getenv("REDIS_IMAGE_CLEANUP_INTERVAL_MINUTES"),
                "REDIS_IMAGE_ENABLE_CLEANUP": os.getenv("REDIS_IMAGE_ENABLE_CLEANUP"),
                "REDIS_IMAGE_QUALITY": os.getenv("REDIS_IMAGE_QUALITY"),
                "REDIS_IMAGE_FORMAT": os.getenv("REDIS_IMAGE_FORMAT"),
                "REDIS_CACHE_ENABLE_COMPRESSION": os.getenv("REDIS_CACHE_ENABLE_COMPRESSION"),
                "REDIS_CACHE_ENABLE_LOCAL_FALLBACK": os.getenv("REDIS_CACHE_ENABLE_LOCAL_FALLBACK"),
                "REDIS_CACHE_LOCAL_SIZE": os.getenv("REDIS_CACHE_LOCAL_SIZE"),
                "REDIS_CACHE_CLEANUP_INTERVAL_MINUTES": os.getenv("REDIS_CACHE_CLEANUP_INTERVAL_MINUTES"),
                "REDIS_URL": os.getenv("REDIS_URL")
            }
        }

# Deployment recommendations
DEPLOYMENT_RECOMMENDATIONS = {
    "development": {
        "description": "Local development environment",
        "recommendations": [
            "Use default settings for development",
            "Charts can be kept longer for debugging",
            "Larger storage limits for testing",
            "Use local Redis instance",
            "PNG format for best quality during development"
        ]
    },
    "staging": {
        "description": "Staging/testing environment",
        "recommendations": [
            "Moderate cleanup frequency",
            "Medium storage limits",
            "Good balance between performance and debugging",
            "Use local or cloud Redis instance",
            "PNG format for quality testing"
        ]
    },
    "production": {
        "description": "Production environment",
        "recommendations": [
            "Aggressive cleanup (6 hours max age)",
            "Small storage limits (200MB)",
            "Frequent cleanup (15 minutes)",
            "Monitor storage usage closely",
            "Use cloud Redis instance for scalability",
            "JPEG format for smaller file sizes",
            "Lower image quality (80) for better performance"
        ]
    }
}

def get_deployment_recommendations() -> Dict[str, Any]:
    """Get deployment recommendations for current environment."""
    env = DeploymentConfig.get_environment()
    return {
        "current_environment": env,
        "recommendations": DEPLOYMENT_RECOMMENDATIONS.get(env, DEPLOYMENT_RECOMMENDATIONS["development"]),
        "chart_config": DeploymentConfig.get_chart_config(),
        "redis_image_config": DeploymentConfig.get_redis_image_config()
    } 