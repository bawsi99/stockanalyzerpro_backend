"""
Configuration file for adjustable thresholds and parameters.
This centralizes all configurable values for easy maintenance and optimization.
"""

import os
from typing import Dict, Any

class Config:
    """Central configuration class for all adjustable parameters."""
    
    # === PERFORMANCE CONFIGURATION ===
    PERFORMANCE = {
        "enable_caching": True,
        "cache_ttl_seconds": 300,  # 5 minutes
        "monte_carlo_simulations": 1000,
        "monte_carlo_days": 252,
        "max_data_points": 10000,  # Limit for large datasets
        "parallel_processing": True,
        "chunk_size": 1000,  # For processing large datasets in chunks
    }
    
    # === VOLUME ANALYSIS CONFIGURATION ===
    VOLUME = {
        "anomaly_threshold": 2.0,  # Standard deviations for volume anomaly detection
        "high_anomaly_threshold": 3.0,  # For high strength anomalies
        "volume_ratio_thresholds": {
            "high": 1.5,
            "moderate": 1.2,
            "low": 0.8,
            "very_low": 0.5
        },
        "correlation_thresholds": {
            "strong": 0.5,
            "moderate": 0.3,
            "weak": 0.1
        },
        "strength_score_weights": {
            "volume_ratio_high": 30,
            "volume_ratio_moderate": 20,
            "volume_ratio_low": 10,
            "confirmation": 25,
            "obv_uptrend": 20,
            "strong_correlation": 15,
            "neutral_mfi": 10
        }
    }
    
    # === PATTERN DETECTION CONFIGURATION ===
    PATTERNS = {
        "head_and_shoulders": {
            "shoulder_tolerance": 0.02,  # 2% tolerance for shoulder similarity
            "head_prominence_threshold": 0.02,  # 2% minimum head prominence
            "quality_weights": {
                "head_prominence": 30,
                "shoulder_symmetry": 20,
                "volume_confirmation": 10,
                "completion": 20
            }
        },
        "cup_and_handle": {
            "min_cup_duration": 20,
            "max_cup_duration": 100,
            "handle_duration_ratio": 0.3,
            "depth_tolerance": 0.15,
            "min_depth": 0.05,
            "max_breakdown": 0.02
        },
        "triple_patterns": {
            "price_tolerance": 0.02,
            "min_spacing": 5,
            "min_valley_ratio": 0.03,
            "min_peak_ratio": 0.03
        },
        "wedge_patterns": {
            "min_points": 6,
            "min_duration": 20,
            "min_quality_score": 20
        },
        "channel_patterns": {
            "min_points": 4,
            "min_duration": 15,
            "parallelism_tolerance": 0.1
        }
    }
    
    # === RISK ANALYSIS CONFIGURATION ===
    RISK = {
        "var_confidence_levels": [0.95, 0.99],
        "stress_testing": {
            "worst_periods": [20, 60, 252],
            "volatility_percentiles": [0.95, 0.99],
            "drawdown_thresholds": {
                "significant": -0.10,
                "moderate": -0.05
            }
        },
        "scenario_analysis": {
            "bear_market_decline": -0.20,
            "crisis_decline": -0.40,
            "black_swan_decline": -0.60,
            "volatility_multipliers": [0.5, 1.0, 2.0, 3.0, 5.0]
        },
        "risk_score_weights": {
            "volatility": 0.30,
            "drawdown": 0.25,
            "tail_risk": 0.20,
            "liquidity": 0.15,
            "correlation": 0.10
        },
        "risk_levels": {
            "high": 70,
            "medium": 40,
            "low": 0
        }
    }
    
    # === MULTI-TIMEFRAME CONFIGURATION ===
    MULTI_TIMEFRAME = {
        "timeframes": {
            "short_term": {"periods": [5, 10, 20], "weight": 0.3},
            "medium_term": {"periods": [50, 100, 200], "weight": 0.4},
            "long_term": {"periods": [200, 365], "weight": 0.3}
        },
        "consensus_thresholds": {
            "strong": 10,
            "moderate": 5,
            "weak": 2
        }
    }
    
    # === MARKET DATA CONFIGURATION ===
    MARKET_DATA = {
        "default_beta": 1.0,
        "default_correlation": 0.75,
        "risk_free_rate": 0.02,  # 2% annual risk-free rate
        "market_index_symbol": "NIFTY50",  # Default market index
        "correlation_lookback": 252,  # Days for correlation calculation
        "beta_lookback": 252,  # Days for beta calculation
    }
    
    # === TECHNICAL INDICATORS CONFIGURATION ===
    TECHNICAL_INDICATORS = {
        "rsi": {
            "period": 14,
            "overbought": 70,
            "oversold": 30,
            "near_overbought": 60,
            "near_oversold": 40
        },
        "macd": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        },
        "bollinger_bands": {
            "period": 20,
            "std_dev": 2,
            "squeeze_threshold": 0.1
        },
        "moving_averages": {
            "short": 20,
            "medium": 50,
            "long": 200
        },
        "atr": {
            "period": 14
        },
        "volume_indicators": {
            "vwap_period": 20,
            "mfi_period": 14,
            "obv_smoothing": 20
        }
    }
    
    # === CACHING CONFIGURATION ===
    CACHE = {
        "enabled": True,
        "max_size": 1000,  # Maximum number of cached items
        "default_ttl": 300,  # 5 minutes default TTL
        "indicators_ttl": 600,  # 10 minutes for indicators
        "patterns_ttl": 1800,  # 30 minutes for patterns
        "risk_ttl": 3600,  # 1 hour for risk calculations
    }
    
    # === LOGGING CONFIGURATION ===
    LOGGING = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "traderpro.log",
        "max_size": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5
    }
    
    # === ENVIRONMENT CONFIGURATION ===
    ENVIRONMENT = {
        "debug": os.getenv("DEBUG", "False").lower() == "true",
        "production": os.getenv("PRODUCTION", "False").lower() == "true",
        "max_workers": int(os.getenv("MAX_WORKERS", "4")),
        "timeout": int(os.getenv("TIMEOUT", "30")),
    }
    
    @classmethod
    def get(cls, section: str, key: str = None, default: Any = None) -> Any:
        """Get configuration value by section and key."""
        if not hasattr(cls, section.upper()):
            return default
        
        section_config = getattr(cls, section.upper())
        
        if key is None:
            return section_config
        
        return section_config.get(key, default)
    
    @classmethod
    def set(cls, section: str, key: str, value: Any) -> None:
        """Set configuration value (for runtime adjustments)."""
        if not hasattr(cls, section.upper()):
            setattr(cls, section.upper(), {})
        
        section_config = getattr(cls, section.upper())
        section_config[key] = value
    
    @classmethod
    def update_section(cls, section: str, updates: Dict[str, Any]) -> None:
        """Update entire configuration section."""
        if not hasattr(cls, section.upper()):
            setattr(cls, section.upper(), {})
        
        section_config = getattr(cls, section.upper())
        section_config.update(updates)
    
    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        config = {}
        for attr_name in dir(cls):
            if attr_name.isupper() and not attr_name.startswith('_'):
                config[attr_name.lower()] = getattr(cls, attr_name)
        return config

# Global configuration instance
config = Config() 