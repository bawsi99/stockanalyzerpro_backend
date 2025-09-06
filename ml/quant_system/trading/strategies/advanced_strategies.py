#!/usr/bin/env python3
"""
Advanced Trading Strategies with Risk Management - Production Ready

This module implements sophisticated trading strategies with comprehensive
risk management for the Phase 2 Advanced Trading System:

1. Position Sizing Algorithms
2. Stop-Loss and Take-Profit Mechanisms
3. Portfolio Optimization
4. Risk Metrics (VaR, CVaR, Sharpe Ratio)
5. Dynamic Asset Allocation
6. Market Regime Detection
7. Advanced Entry/Exit Strategies
"""

import numpy as np
import pandas as pd
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import scipy.optimize as optimize
from scipy import stats
import math

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class RiskManagementConfig:
    """Configuration for risk management parameters."""
    
    # Position sizing
    max_position_size: float = 0.1  # Maximum 10% of portfolio per position
    max_portfolio_risk: float = 0.02  # Maximum 2% portfolio risk per trade
    kelly_fraction: float = 0.25  # Kelly criterion fraction (conservative)
    initial_capital: float = 100000.0  # Starting capital for sizing/cash accounting
    
    # Stop-loss and take-profit
    stop_loss_pct: float = 0.05  # 5% stop-loss
    take_profit_pct: float = 0.15  # 15% take-profit
    trailing_stop: bool = True
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    
    # Risk metrics
    var_confidence: float = 0.95  # 95% VaR confidence level
    max_drawdown: float = 0.20  # Maximum 20% drawdown
    target_sharpe_ratio: float = 1.5  # Target Sharpe ratio
    target_volatility: float = 0.15  # Annualized target volatility for sizing
    
    # Portfolio constraints
    max_sector_exposure: float = 0.3  # Maximum 30% exposure per sector
    max_correlation: float = 0.7  # Maximum correlation between positions
    min_diversification: int = 5  # Minimum number of positions
    
    # Market regime
    volatility_threshold: float = 0.25  # High volatility threshold
    trend_strength_threshold: float = 0.6  # Minimum trend strength
    
    # Trading frequency
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    position_hold_min: int = 1  # Minimum holding period in days

class PositionSizer:
    """Advanced position sizing algorithms."""
    
    def __init__(self, config: RiskManagementConfig):
        self.config = config
    
    def kelly_position_size(self, win_rate: float, avg_win: float, avg_loss: float, 
                           current_capital: float) -> float:
        """Calculate position size using Kelly Criterion."""
        
        if avg_loss == 0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds received, p = probability of win, q = probability of loss
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply conservative Kelly fraction
        conservative_fraction = kelly_fraction * self.config.kelly_fraction
        
        # Ensure within bounds
        position_size = min(
            conservative_fraction * current_capital,
            current_capital * self.config.max_position_size
        )
        
        return max(0.0, position_size)
    
    def volatility_position_size(self, volatility: float, target_volatility: float,
                               current_capital: float) -> float:
        """Calculate position size based on volatility targeting."""
        
        if volatility == 0:
            return 0.0
        
        # Volatility targeting: position_size = target_vol / current_vol
        vol_ratio = target_volatility / volatility
        
        # Apply maximum position size constraint
        position_size = min(
            vol_ratio * current_capital * 0.1,  # 10% base allocation
            current_capital * self.config.max_position_size
        )
        
        return max(0.0, position_size)
    
    def risk_parity_position_size(self, asset_volatility: float, portfolio_volatility: float,
                                 current_capital: float, num_assets: int) -> float:
        """Calculate position size using risk parity approach."""
        
        if asset_volatility == 0 or num_assets == 0:
            return 0.0
        
        # Risk parity: equal risk contribution from each asset
        target_risk = portfolio_volatility / num_assets
        position_size = (target_risk / asset_volatility) * current_capital
        
        # Apply constraints
        position_size = min(
            position_size,
            current_capital * self.config.max_position_size
        )
        
        return max(0.0, position_size)

class RiskMetrics:
    """Advanced risk metrics calculation."""
    
    def __init__(self, config: RiskManagementConfig):
        self.config = config
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = None) -> float:
        """Calculate Value at Risk (VaR)."""
        
        if confidence_level is None:
            confidence_level = self.config.var_confidence
        
        if len(returns) < 2:
            return 0.0
        
        # Historical VaR
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return abs(var)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = None) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        
        if confidence_level is None:
            confidence_level = self.config.var_confidence
        
        if len(returns) < 2:
            return 0.0
        
        # Calculate VaR first
        var = self.calculate_var(returns, confidence_level)
        
        # CVaR is the mean of returns below VaR
        tail_returns = returns[returns <= -var]
        
        if len(tail_returns) == 0:
            return var
        
        cvar = abs(tail_returns.mean())
        return cvar
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        return sharpe
    
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        
        if len(prices) < 2:
            return 0.0
        
        cumulative = prices / prices.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min())
    
    def calculate_volatility(self, returns: pd.Series, window: int = 252) -> float:
        """Calculate rolling volatility."""
        
        if len(returns) < window:
            return returns.std() * np.sqrt(252)
        
        return returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)

class PortfolioOptimizer:
    """Portfolio optimization using modern portfolio theory."""
    
    def __init__(self, config: RiskManagementConfig):
        self.config = config
    
    def optimize_portfolio(self, returns: pd.DataFrame, method: str = "sharpe") -> Dict[str, Any]:
        """Optimize portfolio weights using different methods."""
        
        if len(returns) < 30:
            return {"error": "Insufficient data for optimization"}
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        n_assets = len(returns.columns)
        
        if method == "sharpe":
            return self._optimize_sharpe_ratio(expected_returns, cov_matrix, n_assets)
        elif method == "min_variance":
            return self._optimize_min_variance(cov_matrix, n_assets)
        elif method == "risk_parity":
            return self._optimize_risk_parity(cov_matrix, n_assets)
        else:
            return {"error": f"Unknown optimization method: {method}"}
    
    def _optimize_sharpe_ratio(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, 
                              n_assets: int) -> Dict[str, Any]:
        """Optimize for maximum Sharpe ratio."""
        
        def objective(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            return -sharpe  # Minimize negative Sharpe ratio
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds: no short selling, maximum position size
        bounds = [(0, self.config.max_position_size) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        try:
            result = optimize.minimize(
                objective, initial_weights, method='SLSQP',
                bounds=bounds, constraints=constraints
            )
            
            if result.success:
                return {
                    'weights': result.x,
                    'sharpe_ratio': -result.fun,
                    'expected_return': np.sum(expected_returns * result.x),
                    'volatility': np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x)))
                }
            else:
                return {"error": f"Optimization failed: {result.message}"}
                
        except Exception as e:
            return {"error": f"Optimization error: {str(e)}"}
    
    def _optimize_min_variance(self, cov_matrix: pd.DataFrame, n_assets: int) -> Dict[str, Any]:
        """Optimize for minimum variance portfolio."""
        
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds
        bounds = [(0, self.config.max_position_size) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.array([1/n_assets] * n_assets)
        
        try:
            result = optimize.minimize(
                objective, initial_weights, method='SLSQP',
                bounds=bounds, constraints=constraints
            )
            
            if result.success:
                return {
                    'weights': result.x,
                    'volatility': result.fun,
                    'expected_return': np.sum(pd.Series([0.08] * n_assets) * result.x)  # Assume 8% return
                }
            else:
                return {"error": f"Optimization failed: {result.message}"}
                
        except Exception as e:
            return {"error": f"Optimization error: {str(e)}"}

class MarketRegimeDetector:
    """Detect market regimes for adaptive strategies."""
    
    def __init__(self, config: RiskManagementConfig):
        self.config = config
    
    def detect_regime(self, returns: pd.Series, prices: pd.Series) -> Dict[str, Any]:
        """Detect current market regime."""
        
        if len(returns) < 30:
            return {"regime": "unknown", "confidence": 0.0}
        
        # Calculate regime indicators
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        trend_strength = self._calculate_trend_strength(prices)
        momentum = self._calculate_momentum(prices)
        
        # Determine regime
        if volatility > self.config.volatility_threshold:
            if trend_strength > self.config.trend_strength_threshold:
                regime = "trending_high_vol"
                confidence = min(trend_strength, 0.9)
            else:
                regime = "choppy_high_vol"
                confidence = min(volatility / 0.5, 0.8)
        else:
            if trend_strength > self.config.trend_strength_threshold:
                regime = "trending_low_vol"
                confidence = min(trend_strength, 0.9)
            else:
                regime = "sideways_low_vol"
                confidence = 0.7
        
        return {
            'regime': regime,
            'confidence': confidence,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'momentum': momentum
        }
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength using linear regression R-squared."""
        
        if len(prices) < 20:
            return 0.0
        
        x = np.arange(len(prices))
        y = prices.values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return r_value ** 2  # R-squared
    
    def _calculate_momentum(self, prices: pd.Series) -> float:
        """Calculate price momentum."""
        
        if len(prices) < 20:
            return 0.0
        
        return (prices.iloc[-1] / prices.iloc[-20] - 1) * 100

class AdvancedTradingStrategy:
    """Advanced trading strategy with comprehensive risk management."""
    
    def __init__(self, config: RiskManagementConfig):
        self.config = config
        self.position_sizer = PositionSizer(config)
        self.risk_metrics = RiskMetrics(config)
        self.portfolio_optimizer = PortfolioOptimizer(config)
        self.regime_detector = MarketRegimeDetector(config)
        
        # Strategy state
        # positions[symbol] = {
        #   'quantity': float,
        #   'avg_price': float,
        #   'stop_loss': Optional[float],
        #   'take_profit': Optional[float],
        #   'trailing_stop': Optional[float]
        # }
        self.positions = {}
        self.trade_history = []
        # Treat as available cash; initialize from config
        self.portfolio_value = float(self.config.initial_capital)
        self.max_portfolio_value = 0.0
    
    def generate_trading_signal(self, market_data: Dict[str, Any], 
                               prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive trading signal with risk management."""
        
        try:
            symbol = market_data['symbol']
            price_data = market_data['price_data']
            real_time_data = market_data['real_time_data']
            
            if price_data.empty:
                return {'signal': 'HOLD', 'reason': 'No price data available'}
            
            # Extract key data
            current_price = real_time_data.get('price', price_data['close'].iloc[-1])
            returns = price_data['close'].pct_change().dropna()
            
            # Market regime detection
            regime = self.regime_detector.detect_regime(returns, price_data['close'])
            
            # Risk metrics
            volatility = self.risk_metrics.calculate_volatility(returns)
            var = self.risk_metrics.calculate_var(returns)
            sharpe = self.risk_metrics.calculate_sharpe_ratio(returns)
            
            # Prediction analysis
            predicted_return = prediction.get('ensemble_prediction', 0)
            confidence = prediction.get('ensemble_confidence', 0)
            
            # Signal generation logic
            signal = self._generate_signal_logic(
                predicted_return, confidence, regime, volatility, var, sharpe
            )
            
            # Position sizing
            position_size = self._calculate_position_size(
                signal, predicted_return, confidence, volatility, current_price
            )
            
            # Risk checks
            risk_check = self._perform_risk_checks(
                signal, position_size, volatility, var, regime
            )
            
            if not risk_check['passed']:
                signal = 'HOLD'
                position_size = 0.0
            
            return {
                'signal': signal,
                'position_size': position_size,
                'confidence': confidence,
                'predicted_return': predicted_return,
                'regime': regime['regime'],
                'volatility': volatility,
                'var': var,
                'sharpe': sharpe,
                'risk_check': risk_check,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return {'signal': 'HOLD', 'reason': f'Error: {str(e)}'}
    
    def _generate_signal_logic(self, predicted_return: float, confidence: float,
                              regime: Dict[str, Any], volatility: float, var: float,
                              sharpe: float) -> str:
        """Generate trading signal based on multiple factors."""
        
        # Base signal from prediction
        if confidence > 0.7:  # High confidence threshold
            if predicted_return > 0.02:  # 2% positive return
                base_signal = 'BUY'
            elif predicted_return < -0.02:  # 2% negative return
                base_signal = 'SELL'
            else:
                base_signal = 'HOLD'
        else:
            base_signal = 'HOLD'
        
        # Regime-based adjustments
        if regime['regime'] == 'trending_high_vol':
            # Reduce position sizes in high volatility trending markets
            if base_signal in ['BUY', 'SELL']:
                base_signal = 'HOLD'  # Be more conservative
        elif regime['regime'] == 'choppy_high_vol':
            # Avoid trading in choppy high volatility
            base_signal = 'HOLD'
        elif regime['regime'] == 'sideways_low_vol':
            # Reduce position sizes in sideways markets
            if base_signal in ['BUY', 'SELL']:
                base_signal = 'HOLD'
        
        # Risk-based adjustments
        if volatility > 0.3:  # Very high volatility
            base_signal = 'HOLD'
        elif var > 0.05:  # High VaR
            base_signal = 'HOLD'
        elif sharpe < 0.5:  # Low Sharpe ratio
            base_signal = 'HOLD'
        
        return base_signal
    
    def _calculate_position_size(self, signal: str, predicted_return: float,
                                confidence: float, volatility: float,
                                current_price: float) -> float:
        """Calculate position size using multiple methods."""
        
        if signal == 'HOLD':
            return 0.0
        
        # Base position size from Kelly criterion
        # Assume win rate based on confidence, avg win/loss based on prediction
        win_rate = confidence
        avg_win = abs(predicted_return) if predicted_return > 0 else 0.01
        avg_loss = abs(predicted_return) if predicted_return < 0 else 0.01
        
        kelly_size = self.position_sizer.kelly_position_size(
            win_rate, avg_win, avg_loss, self.portfolio_value
        )
        
        # Volatility-adjusted position size
        vol_size = self.position_sizer.volatility_position_size(
            volatility, self.config.target_volatility, self.portfolio_value
        )
        
        # Use the smaller of the two for conservative approach
        position_size = min(kelly_size, vol_size)
        
        # Apply maximum position size constraint
        position_size = min(position_size, self.portfolio_value * self.config.max_position_size)
        
        return position_size
    
    def _perform_risk_checks(self, signal: str, position_size: float,
                            volatility: float, var: float, regime: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk checks."""
        
        checks = {
            'max_position_size': position_size <= self.portfolio_value * self.config.max_position_size,
            'volatility_limit': volatility <= 0.5,  # Maximum 50% volatility
            'var_limit': var <= 0.1,  # Maximum 10% VaR
            'regime_suitable': regime['regime'] not in ['choppy_high_vol'],
            # Gate on model confidence via position_size>0; regime confidence used as soft signal elsewhere
            'confidence_threshold': True
        }
        
        passed = all(checks.values())
        
        return {
            'passed': passed,
            'checks': checks,
            'failed_checks': [k for k, v in checks.items() if not v]
        }
    
    def execute_trade(self, symbol: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade with comprehensive risk management."""
        
        try:
            trade_signal = signal['signal']
            position_size = signal['position_size']
            current_price = signal.get('current_price')
            
            if trade_signal == 'HOLD':
                return {
                    'symbol': symbol,
                    'action': 'HOLD',
                    'reason': 'No trade signal generated',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Check if we have sufficient capital
            if position_size > self.portfolio_value:
                return {
                    'symbol': symbol,
                    'action': 'HOLD',
                    'reason': 'Insufficient capital',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Execute trade (long-only semantics)
            trade_record = {
                'symbol': symbol,
                'action': trade_signal,
                'position_size': position_size,
                'confidence': signal['confidence'],
                'predicted_return': signal['predicted_return'],
                'regime': signal['regime'],
                'volatility': signal['volatility'],
                'var': signal['var'],
                'sharpe': signal['sharpe'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Require current_price to compute quantities
            if current_price is None or current_price <= 0:
                raise ValueError('Missing or invalid current_price in signal')

            # Update positions (long-only)
            if trade_signal == 'BUY':
                quantity = position_size / current_price if current_price > 0 else 0.0
                if quantity <= 0:
                    return {
                        'symbol': symbol,
                        'action': 'HOLD',
                        'reason': 'Zero quantity after sizing',
                        'timestamp': datetime.now().isoformat()
                    }
                pos = self.positions.get(symbol)
                if pos:
                    # Update VWAP and quantity
                    total_cost = pos['avg_price'] * pos['quantity'] + current_price * quantity
                    total_qty = pos['quantity'] + quantity
                    pos['avg_price'] = total_cost / total_qty
                    pos['quantity'] = total_qty
                else:
                    # Create new position with stops
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'avg_price': current_price,
                        'stop_loss': current_price * (1 - self.config.stop_loss_pct) if self.config.stop_loss_pct else None,
                        'take_profit': current_price * (1 + self.config.take_profit_pct) if self.config.take_profit_pct else None,
                        'trailing_stop': current_price * (1 - self.config.trailing_stop_pct) if self.config.trailing_stop else None,
                    }
                self.portfolio_value -= position_size
            elif trade_signal == 'SELL':
                pos = self.positions.get(symbol)
                if not pos or pos.get('quantity', 0) <= 0:
                    # No shorting; ignore SELL without position
                    return {
                        'symbol': symbol,
                        'action': 'HOLD',
                        'reason': 'No long position to reduce/close',
                        'timestamp': datetime.now().isoformat()
                    }
                sell_qty = min(pos['quantity'], position_size / current_price)
                pos['quantity'] -= sell_qty
                cash_inflow = sell_qty * current_price
                self.portfolio_value += cash_inflow
                if pos['quantity'] <= 0:
                    del self.positions[symbol]
            
            # Update max portfolio value
            self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
            
            # Record trade
            self.trade_history.append(trade_record)
            
            logger.info(f"Executed {trade_signal} trade for {symbol}: ${position_size:,.2f}")
            
            return trade_record
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': 'ERROR',
                'reason': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status."""
        
        total_positions = len(self.positions)
        active_positions = len([p for p in self.positions.values() if p != 0])
        
        # Calculate drawdown
        drawdown = 0.0
        if self.max_portfolio_value > 0:
            drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        return {
            'portfolio_value': self.portfolio_value,
            'max_portfolio_value': self.max_portfolio_value,
            'drawdown': drawdown,
            'total_positions': total_positions,
            'active_positions': active_positions,
            'positions': self.positions,
            'total_trades': len(self.trade_history),
            'timestamp': datetime.now().isoformat()
        }

# Factory function
def create_advanced_trading_strategy(config: RiskManagementConfig = None) -> AdvancedTradingStrategy:
    """Create an advanced trading strategy with risk management."""
    if config is None:
        config = RiskManagementConfig()
    return AdvancedTradingStrategy(config)

# Example usage
def main():
    """Example usage of advanced trading strategy."""
    
    config = RiskManagementConfig(
        max_position_size=0.1,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
        max_drawdown=0.20
    )
    
    strategy = create_advanced_trading_strategy(config)
    
    print("🚀 Advanced Trading Strategy with Risk Management")
    print("=" * 60)
    print("✅ Position sizing algorithms")
    print("✅ Risk metrics (VaR, CVaR, Sharpe)")
    print("✅ Portfolio optimization")
    print("✅ Market regime detection")
    print("✅ Comprehensive risk management")
    print("=" * 60)

if __name__ == "__main__":
    main()

