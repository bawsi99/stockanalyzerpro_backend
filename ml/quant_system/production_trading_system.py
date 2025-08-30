#!/usr/bin/env python3
"""
Production Trading System - Real Market Data + Advanced Strategies

This module integrates the Phase 2 Advanced Trading System with:
1. Real market data from multiple sources
2. Advanced trading strategies with risk management
3. Live trading capabilities
4. Comprehensive monitoring and reporting

Production-ready system for live trading with real market data.
"""

import asyncio
import logging
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import sys
import os

# Import our custom modules
from production_market_data_integration import (
    ProductionMarketDataIntegrator, MarketDataConfig, create_production_market_data_integrator
)
from advanced_trading_strategies_with_risk_management import (
    AdvancedTradingStrategy, RiskManagementConfig, create_advanced_trading_strategy
)
from advanced_models.phase2_integration_manager import Phase2IntegrationManager, Phase2Config
from advanced_models.advanced_feature_engineer import AdvancedFeatureEngineer, FeatureConfig
from advanced_models.real_time_data_integrator import RealTimeDataIntegrator, RealTimeConfig

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ProductionTradingSystem:
    """Production-ready trading system with real market data and advanced strategies."""
    
    def __init__(self, 
                 symbols: List[str] = None,
                 initial_capital: float = 100000.0,
                 market_data_config: MarketDataConfig = None,
                 risk_config: RiskManagementConfig = None):
        
        self.symbols = symbols or ['RELIANCE', 'TCS', 'HDFC', 'INFY', 'ICICIBANK']
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Initialize configurations
        self.market_data_config = market_data_config or MarketDataConfig()
        self.risk_config = risk_config or RiskManagementConfig()
        
        # Initialize components
        self._initialize_components()
        
        # Trading state
        self.positions = {}
        self.trade_history = []
        self.performance_metrics = {}
        self.system_start_time = datetime.now()
        
        logger.info(f"Production Trading System initialized with {len(self.symbols)} symbols")
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
    
    def _initialize_components(self):
        """Initialize all system components."""
        
        try:
            # Initialize market data integrator
            self.market_data_integrator = create_production_market_data_integrator(self.market_data_config)
            logger.info("✅ Market data integrator initialized")
            
            # Initialize advanced trading strategy
            self.trading_strategy = create_advanced_trading_strategy(self.risk_config)
            self.trading_strategy.portfolio_value = self.current_capital
            logger.info("✅ Advanced trading strategy initialized")
            
            # Initialize Phase 2 ML components
            self._initialize_ml_components()
            logger.info("✅ ML components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _initialize_ml_components(self):
        """Initialize Phase 2 ML components."""
        
        try:
            # Feature engineering
            feature_config = FeatureConfig(
                rsi_period=14,
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
                bb_period=20,
                bb_std=2.0,
                atr_period=14,
                volatility_windows=[5, 10, 20, 60],
                correlation_window=60,
                beta_window=252,
                sector_correlation=True,
                order_flow_window=20,
                imbalance_threshold=0.1,
                sentiment_window=5,
                news_impact_window=3,
                missing_threshold=0.1
            )
            self.feature_engineer = AdvancedFeatureEngineer(feature_config)
            
            # Real-time data integrator
            realtime_config = RealTimeConfig(
                zerodha_enabled=True,
                news_api_enabled=False,  # Set to True if you have API keys
                social_api_enabled=False,  # Set to True if you have API keys
                market_data_frequency=1,
                news_update_frequency=300,
                social_update_frequency=60,
                max_data_points=10000,
                cache_duration=3600
            )
            self.realtime_integrator = RealTimeDataIntegrator(realtime_config)
            
            # Phase 2 integration manager
            phase2_config = Phase2Config(
                enable_nas=False,  # Disable for production stability
                enable_meta_learning=False,  # Disable for production stability
                enable_advanced_training=True,
                enable_tft=False,  # Disable for production stability
                model_selection_strategy='performance_based',
                model_update_frequency=24,
                performance_window=1000,
                adaptation_threshold=0.05,
                max_models=10,
                model_cache_size=50,
                auto_retrain=True
            )
            
            estimated_input_size = 75  # Based on feature engineering output
            self.integration_manager = Phase2IntegrationManager(
                input_size=estimated_input_size,
                output_size=1,
                config=phase2_config
            )
            
            # Register initial models
            self._register_initial_models()
            
        except Exception as e:
            logger.error(f"Error initializing ML components: {e}")
            raise
    
    def _register_initial_models(self):
        """Register initial models for ensemble predictions."""
        
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            # Create a simple placeholder for nn to avoid 'name nn is not defined' errors
            class PlaceholderModule:
                def __init__(self, *args, **kwargs):
                    pass
                def __call__(self, *args, **kwargs):
                    return self
            
            class PlaceholderNN:
                def __init__(self):
                    self.Module = PlaceholderModule
                    self.Linear = PlaceholderModule
                    self.Sequential = PlaceholderModule
                    self.ReLU = PlaceholderModule
                    self.Dropout = PlaceholderModule
            
            nn = PlaceholderNN()
            logger.warning("PyTorch not available. Using placeholder nn module.")
            
            # Simple linear model
            linear_model = nn.Sequential(
                nn.Linear(75, 50),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.Linear(25, 1)
            )
            
            # Deep model
            deep_model = nn.Sequential(
                nn.Linear(75, 100),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(100, 75),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(75, 50),
                nn.ReLU(),
                nn.Linear(50, 1)
            )
            
            # Wide model
            wide_model = nn.Sequential(
                nn.Linear(75, 150),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(150, 1)
            )
            
            # Register models
            self.integration_manager.model_registry.register_model("linear_model", linear_model, {"type": "linear", "description": "Simple linear model"})
            self.integration_manager.model_registry.register_model("deep_model", deep_model, {"type": "deep", "description": "Deep neural network"})
            self.integration_manager.model_registry.register_model("wide_model", wide_model, {"type": "wide", "description": "Wide neural network"})
            
            logger.info("✅ Initial models registered successfully")
            
        except Exception as e:
            logger.error(f"Error registering models: {e}")
            raise
    
    async def get_real_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get real market data for a symbol."""
        
        try:
            # Get comprehensive market data
            market_data = await self.market_data_integrator.get_comprehensive_market_data(symbol)
            
            if 'error' in market_data:
                logger.warning(f"Error getting market data for {symbol}: {market_data['error']}")
                return None
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting real market data for {symbol}: {e}")
            return None
    
    async def generate_ml_predictions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ML predictions using Phase 2 components."""
        
        try:
            symbol = market_data['symbol']
            price_data = market_data['price_data']
            
            if price_data.empty:
                return {'ensemble_prediction': 0, 'ensemble_confidence': 0}
            
            # Create features
            features = self.feature_engineer.create_all_features(price_data)
            
            if features is None or features.empty:
                return {'ensemble_prediction': 0, 'ensemble_confidence': 0}
            
            # Get latest features
            latest_features = features.iloc[-1].values
            
            # Generate ensemble prediction
            prediction_result = self.integration_manager.generate_ensemble_prediction(
                latest_features.reshape(1, -1)
            )
            
            if prediction_result is None:
                return {'ensemble_prediction': 0, 'ensemble_confidence': 0}
            
            return {
                'ensemble_prediction': prediction_result.get('prediction', 0),
                'ensemble_confidence': prediction_result.get('confidence', 0),
                'model_predictions': prediction_result.get('model_predictions', {}),
                'feature_importance': prediction_result.get('feature_importance', {})
            }
            
        except Exception as e:
            logger.error(f"Error generating ML predictions: {e}")
            return {'ensemble_prediction': 0, 'ensemble_confidence': 0}
    
    async def execute_trading_cycle(self, symbol: str) -> Dict[str, Any]:
        """Execute a complete trading cycle for a symbol."""
        
        try:
            logger.info(f"🔄 Starting trading cycle for {symbol}")
            
            # Step 1: Get real market data
            market_data = await self.get_real_market_data(symbol)
            if market_data is None:
                return {'status': 'error', 'reason': 'No market data available'}
            
            # Step 2: Generate ML predictions
            predictions = await self.generate_ml_predictions(market_data)
            
            # Step 3: Generate trading signal with risk management
            trading_signal = self.trading_strategy.generate_trading_signal(
                market_data, predictions
            )
            
            # Step 4: Execute trade
            trade_result = self.trading_strategy.execute_trade(symbol, trading_signal)
            
            # Step 5: Update system state
            self._update_system_state(symbol, trade_result)
            
            return {
                'symbol': symbol,
                'market_data_status': 'success' if market_data else 'error',
                'predictions': predictions,
                'trading_signal': trading_signal,
                'trade_result': trade_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in trading cycle for {symbol}: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def _update_system_state(self, symbol: str, trade_result: Dict[str, Any]):
        """Update system state after trade execution."""
        
        try:
            # Update positions
            if trade_result.get('action') in ['BUY', 'SELL']:
                position_size = trade_result.get('position_size', 0)
                if trade_result['action'] == 'BUY':
                    self.positions[symbol] = self.positions.get(symbol, 0) + position_size
                else:  # SELL
                    self.positions[symbol] = self.positions.get(symbol, 0) - position_size
            
            # Update trade history
            self.trade_history.append(trade_result)
            
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error updating system state: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        
        try:
            # Calculate basic metrics
            total_trades = len(self.trade_history)
            successful_trades = len([t for t in self.trade_history if t.get('action') in ['BUY', 'SELL']])
            
            # Calculate portfolio value
            portfolio_value = self.current_capital
            for symbol, position in self.positions.items():
                portfolio_value += position  # Simplified - should use current prices
            
            # Update metrics
            self.performance_metrics = {
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'success_rate': successful_trades / total_trades if total_trades > 0 else 0,
                'portfolio_value': portfolio_value,
                'total_return': (portfolio_value - self.initial_capital) / self.initial_capital,
                'active_positions': len([p for p in self.positions.values() if p != 0]),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def run_trading_session(self, duration_minutes: int = 60, cycle_interval: int = 60):
        """Run a complete trading session."""
        
        logger.info(f"🚀 Starting production trading session")
        logger.info(f"Duration: {duration_minutes} minutes")
        logger.info(f"Cycle interval: {cycle_interval} seconds")
        logger.info(f"Symbols: {self.symbols}")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        cycle_count = 0
        
        try:
            while datetime.now() < end_time:
                cycle_count += 1
                cycle_start = datetime.now()
                
                logger.info(f"\n📊 Trading Cycle {cycle_count}")
                logger.info(f"Time: {datetime.now().strftime('%H:%M:%S')}")
                
                # Execute trading cycles for all symbols
                cycle_results = []
                for symbol in self.symbols:
                    result = await self.execute_trading_cycle(symbol)
                    cycle_results.append(result)
                    
                    # Log results
                    if result.get('trade_result', {}).get('action') != 'HOLD':
                        logger.info(f"  {symbol}: {result['trade_result']['action']} "
                                  f"${result['trade_result'].get('position_size', 0):,.2f}")
                
                # Log portfolio status
                portfolio_status = self.trading_strategy.get_portfolio_status()
                logger.info(f"  Portfolio Value: ${portfolio_status['portfolio_value']:,.2f}")
                logger.info(f"  Active Positions: {portfolio_status['active_positions']}")
                
                # Calculate cycle duration
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                logger.info(f"  Cycle Duration: {cycle_duration:.2f}s")
                
                # Wait for next cycle
                if cycle_duration < cycle_interval:
                    wait_time = cycle_interval - cycle_duration
                    logger.info(f"  Waiting {wait_time:.2f}s for next cycle...")
                    await asyncio.sleep(wait_time)
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Trading session interrupted by user")
        except Exception as e:
            logger.error(f"Error in trading session: {e}")
        finally:
            # Final summary
            session_duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"\n🏁 Trading session completed")
            logger.info(f"Total cycles: {cycle_count}")
            logger.info(f"Session duration: {session_duration/60:.2f} minutes")
            
            # Final performance report
            self._generate_performance_report()
    
    def _generate_performance_report(self):
        """Generate comprehensive performance report."""
        
        try:
            portfolio_status = self.trading_strategy.get_portfolio_status()
            
            report = {
                'session_summary': {
                    'start_time': self.system_start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_minutes': (datetime.now() - self.system_start_time).total_seconds() / 60,
                    'symbols_traded': self.symbols,
                    'initial_capital': self.initial_capital
                },
                'portfolio_performance': {
                    'current_value': portfolio_status['portfolio_value'],
                    'total_return': (portfolio_status['portfolio_value'] - self.initial_capital) / self.initial_capital,
                    'max_drawdown': portfolio_status.get('drawdown', 0),
                    'total_trades': portfolio_status['total_trades'],
                    'active_positions': portfolio_status['active_positions']
                },
                'risk_metrics': {
                    'positions': self.positions,
                    'risk_config': {
                        'max_position_size': self.risk_config.max_position_size,
                        'stop_loss_pct': self.risk_config.stop_loss_pct,
                        'max_drawdown': self.risk_config.max_drawdown
                    }
                },
                'system_status': {
                    'market_data_sources': self.market_data_config.__dict__,
                    'ml_components': 'Phase 2 Advanced Trading System',
                    'risk_management': 'Advanced Risk Management Active'
                }
            }
            
            # Save report
            report_filename = f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📊 Performance report saved: {report_filename}")
            
            # Print summary
            print("\n" + "="*60)
            print("📊 PRODUCTION TRADING SYSTEM PERFORMANCE REPORT")
            print("="*60)
            print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
            print(f"💰 Current Portfolio: ${portfolio_status['portfolio_value']:,.2f}")
            print(f"📈 Total Return: {report['portfolio_performance']['total_return']*100:.2f}%")
            print(f"📊 Total Trades: {portfolio_status['total_trades']}")
            print(f"🎯 Active Positions: {portfolio_status['active_positions']}")
            print(f"⚠️  Max Drawdown: {report['portfolio_performance']['max_drawdown']*100:.2f}%")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        try:
            portfolio_status = self.trading_strategy.get_portfolio_status()
            
            return {
                'system_info': {
                    'name': 'Production Trading System',
                    'version': '2.0',
                    'start_time': self.system_start_time.isoformat(),
                    'uptime_minutes': (datetime.now() - self.system_start_time).total_seconds() / 60
                },
                'trading_config': {
                    'symbols': self.symbols,
                    'initial_capital': self.initial_capital,
                    'current_capital': self.current_capital
                },
                'portfolio_status': portfolio_status,
                'performance_metrics': self.performance_metrics,
                'component_status': {
                    'market_data_integrator': 'Active',
                    'trading_strategy': 'Active',
                    'ml_components': 'Phase 2 Active',
                    'risk_management': 'Active'
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

# Factory function
def create_production_trading_system(
    symbols: List[str] = None,
    initial_capital: float = 100000.0,
    market_data_config: MarketDataConfig = None,
    risk_config: RiskManagementConfig = None
) -> ProductionTradingSystem:
    """Create a production trading system."""
    return ProductionTradingSystem(
        symbols=symbols,
        initial_capital=initial_capital,
        market_data_config=market_data_config,
        risk_config=risk_config
    )

# Main execution
async def main():
    """Main execution function."""
    
    print("🚀 PRODUCTION TRADING SYSTEM")
    print("="*60)
    print("✅ Real market data integration")
    print("✅ Advanced trading strategies")
    print("✅ Comprehensive risk management")
    print("✅ Phase 2 ML components")
    print("✅ Live trading capabilities")
    print("="*60)
    
    # Configuration
    symbols = ['RELIANCE', 'TCS', 'HDFC', 'INFY', 'ICICIBANK']
    initial_capital = 100000.0
    
    # Market data configuration
    market_data_config = MarketDataConfig(
        enable_yahoo_finance=True,
        enable_alpha_vantage=False,  # Set to True if you have API key
        enable_news_api=False,  # Set to True if you have API key
        cache_duration=300
    )
    
    # Risk management configuration
    risk_config = RiskManagementConfig(
        max_position_size=0.1,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
        max_drawdown=0.20,
        kelly_fraction=0.25
    )
    
    # Create and run trading system
    trading_system = create_production_trading_system(
        symbols=symbols,
        initial_capital=initial_capital,
        market_data_config=market_data_config,
        risk_config=risk_config
    )
    
    # Run trading session
    await trading_system.run_trading_session(duration_minutes=30, cycle_interval=60)

if __name__ == "__main__":
    asyncio.run(main())
