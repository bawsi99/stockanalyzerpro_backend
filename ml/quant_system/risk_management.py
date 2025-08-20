"""
Risk Management for Quantitative Trading System

This module provides comprehensive risk management capabilities for:
1. Position sizing and capital allocation
2. Stop-loss and take-profit calculation
3. Risk metrics calculation (VaR, CVaR, Sharpe ratio, etc.)
4. Portfolio risk management
5. Drawdown analysis and management
6. Risk-adjusted performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class RiskConfig:
    """Configuration for risk management."""
    # Capital and position sizing
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% of capital
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk per trade
    
    # Stop-loss and take-profit
    default_stop_loss: float = 0.02  # 2% stop loss
    default_take_profit: float = 0.04  # 4% take profit
    trailing_stop: bool = True
    trailing_stop_distance: float = 0.01  # 1% trailing stop
    
    # Risk metrics
    var_confidence: float = 0.95  # 95% VaR
    cvar_confidence: float = 0.95  # 95% CVaR
    risk_free_rate: float = 0.02  # 2% risk-free rate
    
    # Portfolio limits
    max_correlation: float = 0.7  # Maximum correlation between positions
    max_sector_exposure: float = 0.3  # Maximum sector exposure
    max_single_stock_exposure: float = 0.15  # Maximum single stock exposure

class PositionSizer:
    """Calculate optimal position sizes based on risk parameters."""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
    
    def calculate_position_size(self, capital: float, entry_price: float, 
                              stop_loss_price: float, risk_per_trade: float = None) -> Dict[str, float]:
        """Calculate position size based on risk per trade."""
        if risk_per_trade is None:
            risk_per_trade = self.config.max_portfolio_risk
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share <= 0:
            logger.warning("Invalid stop loss price")
            return {'position_size': 0, 'shares': 0, 'risk_amount': 0}
        
        # Calculate maximum risk amount
        max_risk_amount = capital * risk_per_trade
        
        # Calculate position size
        shares = max_risk_amount / risk_per_share
        
        # Apply position size limits
        max_shares = capital * self.config.max_position_size / entry_price
        shares = min(shares, max_shares)
        
        position_value = shares * entry_price
        actual_risk_amount = shares * risk_per_share
        
        return {
            'position_size': position_value,
            'shares': shares,
            'risk_amount': actual_risk_amount,
            'risk_percentage': actual_risk_amount / capital
        }
    
    def calculate_kelly_position_size(self, win_rate: float, avg_win: float, 
                                    avg_loss: float, capital: float) -> Dict[str, float]:
        """Calculate position size using Kelly Criterion."""
        if avg_loss == 0:
            logger.warning("Average loss cannot be zero for Kelly Criterion")
            return {'position_size': 0, 'kelly_fraction': 0}
        
        # Kelly fraction
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Apply constraints
        kelly_fraction = max(0, min(kelly_fraction, self.config.max_position_size))
        
        position_size = capital * kelly_fraction
        
        return {
            'position_size': position_size,
            'kelly_fraction': kelly_fraction
        }
    
    def calculate_volatility_position_size(self, capital: float, volatility: float, 
                                         target_volatility: float = 0.15) -> Dict[str, float]:
        """Calculate position size based on volatility targeting."""
        if volatility <= 0:
            logger.warning("Invalid volatility value")
            return {'position_size': 0, 'volatility_adjustment': 0}
        
        # Volatility adjustment factor
        vol_adjustment = target_volatility / volatility
        
        # Apply position size limits
        vol_adjustment = min(vol_adjustment, self.config.max_position_size)
        
        position_size = capital * vol_adjustment
        
        return {
            'position_size': position_size,
            'volatility_adjustment': vol_adjustment
        }

class StopLossManager:
    """Manage stop-loss and take-profit levels."""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
    
    def calculate_stop_loss(self, entry_price: float, direction: str = 'long', 
                          atr: float = None, custom_stop: float = None) -> float:
        """Calculate stop-loss level."""
        if custom_stop is not None:
            return custom_stop
        
        if direction.lower() == 'long':
            if atr is not None:
                # Use ATR-based stop loss
                stop_loss = entry_price - (2 * atr)
            else:
                # Use percentage-based stop loss
                stop_loss = entry_price * (1 - self.config.default_stop_loss)
        else:  # short
            if atr is not None:
                # Use ATR-based stop loss
                stop_loss = entry_price + (2 * atr)
            else:
                # Use percentage-based stop loss
                stop_loss = entry_price * (1 + self.config.default_stop_loss)
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, direction: str = 'long',
                            risk_reward_ratio: float = 2.0, custom_take_profit: float = None) -> float:
        """Calculate take-profit level."""
        if custom_take_profit is not None:
            return custom_take_profit
        
        if direction.lower() == 'long':
            take_profit = entry_price * (1 + self.config.default_take_profit)
        else:  # short
            take_profit = entry_price * (1 - self.config.default_take_profit)
        
        return take_profit
    
    def update_trailing_stop(self, current_price: float, highest_price: float, 
                           lowest_price: float, direction: str = 'long') -> float:
        """Update trailing stop level."""
        if not self.config.trailing_stop:
            return None
        
        if direction.lower() == 'long':
            # For long positions, trail below the highest price
            trailing_stop = highest_price * (1 - self.config.trailing_stop_distance)
        else:  # short
            # For short positions, trail above the lowest price
            trailing_stop = lowest_price * (1 + self.config.trailing_stop_distance)
        
        return trailing_stop
    
    def check_stop_loss_hit(self, current_price: float, stop_loss: float, 
                           direction: str = 'long') -> bool:
        """Check if stop loss has been hit."""
        if direction.lower() == 'long':
            return current_price <= stop_loss
        else:  # short
            return current_price >= stop_loss
    
    def check_take_profit_hit(self, current_price: float, take_profit: float, 
                             direction: str = 'long') -> bool:
        """Check if take profit has been hit."""
        if direction.lower() == 'long':
            return current_price >= take_profit
        else:  # short
            return current_price <= take_profit

class RiskMetrics:
    """Calculate various risk metrics."""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
    
    def calculate_var(self, returns: pd.Series, confidence: float = None) -> float:
        """Calculate Value at Risk (VaR)."""
        if confidence is None:
            confidence = self.config.var_confidence
        
        if len(returns) == 0:
            return 0.0
        
        # Historical VaR
        var = np.percentile(returns, (1 - confidence) * 100)
        return abs(var)
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = None) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        if confidence is None:
            confidence = self.config.cvar_confidence
        
        if len(returns) == 0:
            return 0.0
        
        # Historical CVaR
        var = self.calculate_var(returns, confidence)
        cvar = returns[returns <= -var].mean()
        return abs(cvar) if not np.isnan(cvar) else 0.0
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> float:
        """Calculate Sharpe ratio."""
        if risk_free_rate is None:
            risk_free_rate = self.config.risk_free_rate
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> float:
        """Calculate Sortino ratio."""
        if risk_free_rate is None:
            risk_free_rate = self.config.risk_free_rate
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        return sortino
    
    def calculate_max_drawdown(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate maximum drawdown."""
        if len(prices) == 0:
            return {'max_drawdown': 0.0, 'drawdown_duration': 0}
        
        # Calculate cumulative returns
        cumulative_returns = (prices / prices.iloc[0]) - 1
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = cumulative_returns - running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        drawdown_periods = (drawdown < 0).astype(int)
        drawdown_duration = drawdown_periods.sum()
        
        return {
            'max_drawdown': abs(max_drawdown),
            'drawdown_duration': drawdown_duration,
            'drawdown_series': drawdown
        }
    
    def calculate_volatility(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling volatility."""
        if len(returns) == 0:
            return pd.Series()
        
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility
    
    def calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market."""
        if len(returns) == 0 or len(market_returns) == 0:
            return 0.0
        
        # Align the series
        aligned_data = pd.concat([returns, market_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        returns_aligned = aligned_data.iloc[:, 0]
        market_aligned = aligned_data.iloc[:, 1]
        
        # Calculate covariance and variance
        covariance = np.cov(returns_aligned, market_aligned)[0, 1]
        market_variance = np.var(market_aligned)
        
        if market_variance == 0:
            return 0.0
        
        beta = covariance / market_variance
        return beta
    
    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for portfolio assets."""
        if returns_df.empty:
            return pd.DataFrame()
        
        correlation_matrix = returns_df.corr()
        return correlation_matrix
    
    def calculate_portfolio_risk(self, weights: np.ndarray, returns_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio risk metrics."""
        if len(weights) == 0 or returns_df.empty:
            return {'portfolio_volatility': 0.0, 'portfolio_var': 0.0}
        
        # Calculate covariance matrix
        covariance_matrix = returns_df.cov() * 252  # Annualized
        
        # Portfolio volatility
        portfolio_volatility = np.sqrt(weights.T @ covariance_matrix @ weights)
        
        # Portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Portfolio VaR
        portfolio_var = self.calculate_var(portfolio_returns)
        
        return {
            'portfolio_volatility': portfolio_volatility,
            'portfolio_var': portfolio_var,
            'portfolio_returns': portfolio_returns
        }

class PortfolioRiskManager:
    """Manage portfolio-level risk."""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.risk_metrics = RiskMetrics(config)
    
    def check_position_limits(self, current_positions: Dict[str, float], 
                            new_position: Dict[str, Any]) -> bool:
        """Check if new position violates position limits."""
        total_exposure = sum(current_positions.values()) + new_position.get('position_size', 0)
        
        # Check single position limit
        if new_position.get('position_size', 0) > self.config.max_single_stock_exposure:
            logger.warning("New position exceeds single stock exposure limit")
            return False
        
        # Check portfolio concentration
        if total_exposure > 1.0:  # 100% of capital
            logger.warning("New position would exceed portfolio limits")
            return False
        
        return True
    
    def calculate_portfolio_weights(self, positions: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio weights."""
        total_value = sum(positions.values())
        
        if total_value == 0:
            return {}
        
        weights = {symbol: value / total_value for symbol, value in positions.items()}
        return weights
    
    def optimize_portfolio_weights(self, returns_df: pd.DataFrame, 
                                 target_return: float = None) -> Dict[str, float]:
        """Optimize portfolio weights using risk-return optimization."""
        if returns_df.empty:
            return {}
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns_df.mean() * 252  # Annualized
        covariance_matrix = returns_df.cov() * 252  # Annualized
        
        # Simple equal-weight allocation (can be enhanced with optimization)
        n_assets = len(returns_df.columns)
        weights = {asset: 1.0 / n_assets for asset in returns_df.columns}
        
        return weights
    
    def calculate_portfolio_metrics(self, positions: Dict[str, float], 
                                  returns_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive portfolio risk metrics."""
        if not positions or returns_df.empty:
            return {}
        
        # Calculate weights
        weights = self.calculate_portfolio_weights(positions)
        weights_array = np.array(list(weights.values()))
        
        # Calculate portfolio risk
        portfolio_risk = self.risk_metrics.calculate_portfolio_risk(weights_array, returns_df)
        
        # Calculate individual metrics
        portfolio_returns = portfolio_risk['portfolio_returns']
        
        metrics = {
            'portfolio_volatility': portfolio_risk['portfolio_volatility'],
            'portfolio_var': portfolio_risk['portfolio_var'],
            'portfolio_cvar': self.risk_metrics.calculate_cvar(portfolio_returns),
            'sharpe_ratio': self.risk_metrics.calculate_sharpe_ratio(portfolio_returns),
            'sortino_ratio': self.risk_metrics.calculate_sortino_ratio(portfolio_returns),
            'max_drawdown': self.risk_metrics.calculate_max_drawdown(portfolio_returns.cumsum())['max_drawdown']
        }
        
        return metrics
    
    def generate_risk_report(self, positions: Dict[str, float], 
                           returns_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        if not positions or returns_df.empty:
            return {}
        
        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(positions, returns_df)
        
        # Calculate correlation matrix
        correlation_matrix = self.risk_metrics.calculate_correlation_matrix(returns_df)
        
        # Calculate individual asset metrics
        asset_metrics = {}
        for asset in returns_df.columns:
            asset_returns = returns_df[asset].dropna()
            if len(asset_returns) > 0:
                asset_metrics[asset] = {
                    'volatility': asset_returns.std() * np.sqrt(252),
                    'var': self.risk_metrics.calculate_var(asset_returns),
                    'sharpe_ratio': self.risk_metrics.calculate_sharpe_ratio(asset_returns),
                    'max_drawdown': self.risk_metrics.calculate_max_drawdown(asset_returns.cumsum())['max_drawdown']
                }
        
        # Risk report
        risk_report = {
            'portfolio_metrics': portfolio_metrics,
            'asset_metrics': asset_metrics,
            'correlation_matrix': correlation_matrix,
            'positions': positions,
            'total_exposure': sum(positions.values()),
            'number_of_positions': len(positions),
            'report_date': datetime.now().isoformat()
        }
        
        return risk_report

class RiskAdjustedPerformance:
    """Calculate risk-adjusted performance metrics."""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.risk_metrics = RiskMetrics(config)
    
    def calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio."""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align the series
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        returns_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]
        
        # Calculate excess returns
        excess_returns = returns_aligned - benchmark_aligned
        
        if excess_returns.std() == 0:
            return 0.0
        
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return information_ratio
    
    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        if len(returns) == 0:
            return 0.0
        
        # Calculate annualized return
        annual_return = returns.mean() * 252
        
        # Calculate maximum drawdown
        max_drawdown = self.risk_metrics.calculate_max_drawdown(returns.cumsum())['max_drawdown']
        
        if max_drawdown == 0:
            return 0.0
        
        calmar_ratio = annual_return / max_drawdown
        return calmar_ratio
    
    def calculate_treynor_ratio(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate Treynor ratio."""
        if len(returns) == 0 or len(market_returns) == 0:
            return 0.0
        
        # Calculate beta
        beta = self.risk_metrics.calculate_beta(returns, market_returns)
        
        if beta == 0:
            return 0.0
        
        # Calculate excess return
        excess_return = returns.mean() * 252 - self.config.risk_free_rate
        
        treynor_ratio = excess_return / beta
        return treynor_ratio
    
    def calculate_jensen_alpha(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate Jensen's alpha."""
        if len(returns) == 0 or len(market_returns) == 0:
            return 0.0
        
        # Calculate beta
        beta = self.risk_metrics.calculate_beta(returns, market_returns)
        
        # Calculate expected return
        expected_return = self.config.risk_free_rate + beta * (market_returns.mean() * 252 - self.config.risk_free_rate)
        
        # Calculate actual return
        actual_return = returns.mean() * 252
        
        # Calculate alpha
        alpha = actual_return - expected_return
        return alpha
