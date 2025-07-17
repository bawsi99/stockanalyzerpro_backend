"""
Market data integration for real market data calculations.
Handles beta calculation, correlation analysis, and market index data.
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from config import Config
from cache_manager import cached, monitor_performance

logger = logging.getLogger(__name__)

class MarketDataProvider:
    """Provides real market data for calculations."""
    
    def __init__(self):
        self.market_index_symbol = Config.get("market_data", "market_index_symbol", "NIFTY50")
        self.correlation_lookback = Config.get("market_data", "correlation_lookback", 252)
        self.beta_lookback = Config.get("market_data", "beta_lookback", 252)
        self.risk_free_rate = Config.get("market_data", "risk_free_rate", 0.02)
        
        # Cache for market data
        self._market_data_cache = {}
        self._last_update = None
    
    @cached(ttl=3600, key_prefix="market_data")  # Cache for 1 hour
    @monitor_performance("fetch_market_data")
    def fetch_market_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Fetch market data for a given symbol.
        
        Args:
            symbol: Stock or index symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No data found for symbol: {symbol}")
                return None
            
            # Clean and validate data
            data = data.dropna()
            if len(data) < 30:  # Minimum required data points
                logger.warning(f"Insufficient data for {symbol}: {len(data)} points")
                return None
            
            logger.info(f"Successfully fetched {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    @cached(ttl=1800, key_prefix="market_index")
    def get_market_index_data(self, symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get market index data for correlation and beta calculations.
        
        Args:
            symbol: Market index symbol (uses default if None)
            
        Returns:
            DataFrame with market index data
        """
        index_symbol = symbol or self.market_index_symbol
        
        # Try different index symbols if the default fails
        index_symbols = [index_symbol, "NIFTY50", "SENSEX", "^NSEI", "^BSESN"]
        
        for symbol in index_symbols:
            data = self.fetch_market_data(symbol, period="1y")
            if data is not None:
                logger.info(f"Using market index: {symbol}")
                return data
        
        logger.error("Failed to fetch any market index data")
        return None
    
    @monitor_performance("calculate_beta")
    def calculate_beta(self, stock_data: pd.DataFrame, market_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate beta for a stock relative to market index.
        
        Args:
            stock_data: Stock price data
            market_data: Market index data (fetched if None)
            
        Returns:
            Beta value
        """
        try:
            if market_data is None:
                market_data = self.get_market_index_data()
                if market_data is None:
                    logger.warning("Using default beta due to missing market data")
                    return Config.get("market_data", "default_beta", 1.0)
            
            # Align data by date
            stock_returns = stock_data['Close'].pct_change().dropna()
            market_returns = market_data['Close'].pct_change().dropna()
            
            # Align series
            common_dates = stock_returns.index.intersection(market_returns.index)
            if len(common_dates) < 30:
                logger.warning("Insufficient common data points for beta calculation")
                return Config.get("market_data", "default_beta", 1.0)
            
            stock_returns = stock_returns.loc[common_dates]
            market_returns = market_returns.loc[common_dates]
            
            # Calculate beta using covariance method
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            if market_variance == 0:
                logger.warning("Market variance is zero, using default beta")
                return Config.get("market_data", "default_beta", 1.0)
            
            beta = covariance / market_variance
            
            # Validate beta value
            if not np.isfinite(beta) or abs(beta) > 10:
                logger.warning(f"Invalid beta value: {beta}, using default")
                return Config.get("market_data", "default_beta", 1.0)
            
            logger.info(f"Calculated beta: {beta:.3f}")
            return float(beta)
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return Config.get("market_data", "default_beta", 1.0)
    
    @monitor_performance("calculate_correlation")
    def calculate_correlation(self, stock_data: pd.DataFrame, market_data: Optional[pd.DataFrame] = None, 
                            lookback: Optional[int] = None) -> float:
        """
        Calculate correlation between stock and market index.
        
        Args:
            stock_data: Stock price data
            market_data: Market index data (fetched if None)
            lookback: Number of days to look back (uses default if None)
            
        Returns:
            Correlation coefficient
        """
        try:
            if market_data is None:
                market_data = self.get_market_index_data()
                if market_data is None:
                    logger.warning("Using default correlation due to missing market data")
                    return Config.get("market_data", "default_correlation", 0.75)
            
            lookback_days = lookback or self.correlation_lookback
            
            # Get recent data
            stock_returns = stock_data['Close'].pct_change().dropna().tail(lookback_days)
            market_returns = market_data['Close'].pct_change().dropna().tail(lookback_days)
            
            # Align series
            common_dates = stock_returns.index.intersection(market_returns.index)
            if len(common_dates) < 30:
                logger.warning("Insufficient common data points for correlation calculation")
                return Config.get("market_data", "default_correlation", 0.75)
            
            stock_returns = stock_returns.loc[common_dates]
            market_returns = market_returns.loc[common_dates]
            
            # Calculate correlation
            correlation = stock_returns.corr(market_returns)
            
            # Validate correlation value
            if not np.isfinite(correlation) or abs(correlation) > 1:
                logger.warning(f"Invalid correlation value: {correlation}, using default")
                return Config.get("market_data", "default_correlation", 0.75)
            
            logger.info(f"Calculated correlation: {correlation:.3f}")
            return float(correlation)
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return Config.get("market_data", "default_correlation", 0.75)
    
    @monitor_performance("calculate_risk_free_rate")
    def get_risk_free_rate(self) -> float:
        """
        Get current risk-free rate.
        
        Returns:
            Annual risk-free rate
        """
        try:
            # Try to fetch current government bond yield
            # For now, return configured default
            return self.risk_free_rate
        except Exception as e:
            logger.error(f"Error fetching risk-free rate: {e}")
            return Config.get("market_data", "risk_free_rate", 0.02)
    
    @monitor_performance("calculate_market_metrics")
    def calculate_market_metrics(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive market metrics for a stock.
        
        Args:
            stock_data: Stock price data
            
        Returns:
            Dictionary with market metrics
        """
        try:
            market_data = self.get_market_index_data()
            
            # Calculate beta and correlation
            beta = self.calculate_beta(stock_data, market_data)
            correlation = self.calculate_correlation(stock_data, market_data)
            risk_free_rate = self.get_risk_free_rate()
            
            # Calculate market-adjusted metrics
            stock_returns = stock_data['Close'].pct_change().dropna()
            market_returns = market_data['Close'].pct_change().dropna() if market_data is not None else None
            
            # Align returns for calculations
            if market_returns is not None:
                common_dates = stock_returns.index.intersection(market_returns.index)
                stock_returns = stock_returns.loc[common_dates]
                market_returns = market_returns.loc[common_dates]
                
                # Calculate excess returns
                excess_returns = stock_returns - risk_free_rate / 252  # Daily risk-free rate
                market_excess_returns = market_returns - risk_free_rate / 252
                
                # Calculate CAPM metrics
                market_volatility = market_returns.std() * np.sqrt(252)
                stock_volatility = stock_returns.std() * np.sqrt(252)
                
                # Systematic and unsystematic risk
                systematic_risk = (beta * market_volatility) ** 2
                total_risk = stock_volatility ** 2
                unsystematic_risk = total_risk - systematic_risk
                
                # Information ratio (if we have enough data)
                if len(excess_returns) > 30:
                    information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                else:
                    information_ratio = 0.0
            else:
                # Fallback calculations without market data
                systematic_risk = (beta * 0.20) ** 2  # Assume 20% market volatility
                total_risk = stock_returns.std() ** 2 * 252
                unsystematic_risk = total_risk - systematic_risk
                information_ratio = 0.0
            
            metrics = {
                "beta": beta,
                "correlation": correlation,
                "risk_free_rate": risk_free_rate,
                "systematic_risk": float(systematic_risk),
                "unsystematic_risk": float(unsystematic_risk),
                "total_risk": float(total_risk),
                "information_ratio": float(information_ratio),
                "data_quality": "real" if market_data is not None else "estimated"
            }
            
            logger.info(f"Calculated market metrics: beta={beta:.3f}, correlation={correlation:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating market metrics: {e}")
            return {
                "beta": Config.get("market_data", "default_beta", 1.0),
                "correlation": Config.get("market_data", "default_correlation", 0.75),
                "risk_free_rate": Config.get("market_data", "risk_free_rate", 0.02),
                "systematic_risk": 0.04,  # 4% systematic risk
                "unsystematic_risk": 0.06,  # 6% unsystematic risk
                "total_risk": 0.10,  # 10% total risk
                "information_ratio": 0.0,
                "data_quality": "fallback"
            }

# Global market data provider instance
market_data_provider = MarketDataProvider()

def get_market_data_provider() -> MarketDataProvider:
    """Get the global market data provider instance."""
    return market_data_provider

def calculate_stock_beta(stock_data: pd.DataFrame) -> float:
    """Calculate beta for a stock."""
    return market_data_provider.calculate_beta(stock_data)

def calculate_stock_correlation(stock_data: pd.DataFrame) -> float:
    """Calculate correlation for a stock."""
    return market_data_provider.calculate_correlation(stock_data)

def get_market_metrics(stock_data: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive market metrics for a stock."""
    return market_data_provider.calculate_market_metrics(stock_data) 