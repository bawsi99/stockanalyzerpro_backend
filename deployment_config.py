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
    def get_environment_info(cls) -> Dict[str, Any]:
        """Get current environment information."""
        return {
            "environment": cls.get_environment(),
            "chart_config": cls.get_chart_config(),
            "environment_variables": {
                "CHART_MAX_AGE_HOURS": os.getenv("CHART_MAX_AGE_HOURS"),
                "CHART_MAX_SIZE_MB": os.getenv("CHART_MAX_SIZE_MB"),
                "CHART_CLEANUP_INTERVAL_MINUTES": os.getenv("CHART_CLEANUP_INTERVAL_MINUTES"),
                "CHART_OUTPUT_DIR": os.getenv("CHART_OUTPUT_DIR"),
                "CHART_ENABLE_CLEANUP": os.getenv("CHART_ENABLE_CLEANUP")
            }
        }

# Deployment recommendations
DEPLOYMENT_RECOMMENDATIONS = {
    "development": {
        "description": "Local development environment",
        "recommendations": [
            "Use default settings for development",
            "Charts can be kept longer for debugging",
            "Larger storage limits for testing"
        ]
    },
    "staging": {
        "description": "Staging/testing environment",
        "recommendations": [
            "Moderate cleanup frequency",
            "Medium storage limits",
            "Good balance between performance and debugging"
        ]
    },
    "production": {
        "description": "Production environment",
        "recommendations": [
            "Aggressive cleanup (6 hours max age)",
            "Small storage limits (200MB)",
            "Frequent cleanup (15 minutes)",
            "Monitor storage usage closely"
        ]
    }
}

def get_deployment_recommendations() -> Dict[str, Any]:
    """Get deployment recommendations for current environment."""
    env = DeploymentConfig.get_environment()
    return {
        "current_environment": env,
        "recommendations": DEPLOYMENT_RECOMMENDATIONS.get(env, DEPLOYMENT_RECOMMENDATIONS["development"]),
        "config": DeploymentConfig.get_chart_config()
    } 