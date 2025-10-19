"""
Quantitative Trading System - Reorganized Structure

This is the main entry point for the reorganized quantitative trading system.
The system is now organized into logical modules:

- core/: Core infrastructure (config, base models, registry, utils)
- engines/: ML engines (pattern ML, raw data ML, hybrid ML, unified manager)
- advanced/: Advanced ML components (models, training, integration)
- features/: Feature engineering (technical indicators, pattern features)
- data/: Data pipeline (processing, datasets, market data integration)
- trading/: Trading system (strategies, execution, backtesting)
- evaluation/: Evaluation and analysis tools
- utils/: Common utilities
- scripts/: Utility scripts and tools

Usage:
    from quant_system import config, engines, features, trading
    from quant_system.core import UnifiedConfig, BaseMLEngine
    from quant_system.engines import UnifiedMLManager
"""

from . import core
from . import engines
from . import advanced
from . import features
from . import data
from . import trading
from . import evaluation
from . import utils
from . import scripts

# Import main components for easy access
from .core import config, UnifiedConfig, BaseMLEngine, global_registry
from .engines import UnifiedMLManager

__version__ = "2.0.0"
__author__ = "Quantitative Trading System Team"
__description__ = "Reorganized Quantitative Trading System"

# Main exports
__all__ = [
    # Core modules
    'core',
    'engines', 
    'advanced',
    'features',
    'data',
    'trading',
    'evaluation',
    'utils',
    'scripts',
    
    # Main components
    'config',
    'UnifiedConfig',
    'BaseMLEngine',
    'global_registry',
    'UnifiedMLManager'
]
