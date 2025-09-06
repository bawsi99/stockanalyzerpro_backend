"""
Real-Time Data Integrator for Advanced Trading System

This module provides real-time data integration capabilities:
1. Live market data from Zerodha
2. Real-time news sentiment
3. Live social media sentiment
4. Portfolio data integration
5. Real-time risk metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Optional imports - will be None if not available
try:
    from zerodha_client import ZerodhaDataClient
    ZERODHA_AVAILABLE = True
except ImportError:
    ZerodhaDataClient = None
    ZERODHA_AVAILABLE = False

try:
    from zerodha_ws_client import ZerodhaWebSocketClient
    WEBSOCKET_AVAILABLE = True
except ImportError:
    ZerodhaWebSocketClient = None
    WEBSOCKET_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class RealTimeConfig:
    """Configuration for real-time data integration."""
    
    # Data sources
    zerodha_enabled: bool = True
    news_api_enabled: bool = True
    social_api_enabled: bool = True
    
    # Update frequencies
    market_data_frequency: int = 1  # seconds
    news_update_frequency: int = 300  # 5 minutes
    social_update_frequency: int = 60  # 1 minute
    
    # Data retention
    max_data_points: int = 10000
    cache_duration: int = 3600  # 1 hour
    
    # API configurations
    news_api_key: str = None
    social_api_key: str = None
    
    def __post_init__(self):
        # Load from environment variables if not provided
        if self.news_api_key is None:
            self.news_api_key = os.getenv('NEWS_API_KEY')
        if self.social_api_key is None:
            self.social_api_key = os.getenv('SOCIAL_API_KEY')

class RealTimeMarketData:
    """Real-time market data integration."""
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        if config.zerodha_enabled and ZERODHA_AVAILABLE:
            self.zerodha_client = ZerodhaDataClient()
        else:
            self.zerodha_client = None
            if config.zerodha_enabled and not ZERODHA_AVAILABLE:
                logger.warning("Zerodha client requested but not available")
        
        self.ws_client = None
        self.live_data_cache = {}
        self.last_update = {}
        
    async def initialize_websocket(self):
        """Initialize WebSocket connection for real-time data."""
        if not self.config.zerodha_enabled or not WEBSOCKET_AVAILABLE:
            return
        
        try:
            self.ws_client = ZerodhaWebSocketClient()
            await self.ws_client.connect()
            logger.info("WebSocket connection established for real-time data")
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket: {e}")
    
    def get_live_market_data(self, symbol: str, exchange: str = "NSE") -> pd.DataFrame:
        """Get live market data for a symbol."""
        if not self.config.zerodha_enabled:
            logger.warning("Zerodha client not enabled")
            return pd.DataFrame()
        
        try:
            # Get latest data from Zerodha
            current_time = datetime.now()
            end_date = current_time
            start_date = current_time - timedelta(days=1)  # Get last 24 hours
            
            data = self.zerodha_client.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                interval="1minute",
                from_date=start_date,
                to_date=end_date
            )
            
            if not data.empty:
                # Standardize column names
                data.columns = [col.lower() for col in data.columns]
                data.index.name = 'datetime'
                
                # Cache the data
                cache_key = f"{symbol}_{exchange}"
                self.live_data_cache[cache_key] = data
                self.last_update[cache_key] = current_time
                
                logger.info(f"Retrieved live data for {symbol}: {len(data)} records")
                return data
            else:
                logger.warning(f"No live data available for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting live market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_cached_data(self, symbol: str, exchange: str = "NSE") -> pd.DataFrame:
        """Get cached market data if recent, otherwise fetch live."""
        cache_key = f"{symbol}_{exchange}"
        
        if cache_key in self.live_data_cache:
            last_update = self.last_update.get(cache_key)
            if last_update and (datetime.now() - last_update).seconds < self.config.cache_duration:
                return self.live_data_cache[cache_key]
        
        # Fetch fresh data if cache is stale or missing
        return self.get_live_market_data(symbol, exchange)
    
    def get_multi_timeframe_data(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Get multi-timeframe data for a symbol."""
        multi_tf_data = {}
        
        for timeframe in timeframes:
            try:
                # Map timeframe to interval
                interval_map = {
                    "1min": "1minute",
                    "5min": "5minute", 
                    "15min": "15minute",
                    "30min": "30minute",
                    "1hour": "60minute",
                    "1day": "day"
                }
                
                interval = interval_map.get(timeframe, "1minute")
                
                # Get data for this timeframe
                current_time = datetime.now()
                end_date = current_time
                
                # Calculate start date based on timeframe
                if timeframe == "1min":
                    start_date = current_time - timedelta(days=1)
                elif timeframe == "5min":
                    start_date = current_time - timedelta(days=3)
                elif timeframe == "15min":
                    start_date = current_time - timedelta(days=7)
                elif timeframe == "30min":
                    start_date = current_time - timedelta(days=14)
                elif timeframe == "1hour":
                    start_date = current_time - timedelta(days=30)
                elif timeframe == "1day":
                    start_date = current_time - timedelta(days=252)
                
                data = self.zerodha_client.get_historical_data(
                    symbol=symbol,
                    exchange="NSE",
                    interval=interval,
                    from_date=start_date,
                    to_date=end_date
                )
                
                if not data.empty:
                    data.columns = [col.lower() for col in data.columns]
                    data.index.name = 'datetime'
                    multi_tf_data[timeframe] = data
                    
            except Exception as e:
                logger.error(f"Error getting {timeframe} data for {symbol}: {e}")
                continue
        
        return multi_tf_data

class RealTimeNewsData:
    """Real-time news sentiment integration."""
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.news_cache = {}
        self.last_update = {}
        
    def get_live_news_sentiment(self, symbol: str) -> pd.DataFrame:
        """Get live news sentiment for a symbol."""
        if not self.config.news_api_enabled or not self.config.news_api_key:
            logger.warning("News API not enabled or configured")
            return pd.DataFrame()
        
        try:
            # TODO: Implement actual news API integration
            # For now, return empty DataFrame to indicate no mock data
            logger.info(f"News sentiment requested for {symbol} - API integration pending")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return pd.DataFrame()

class RealTimeSocialData:
    """Real-time social media sentiment integration."""
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.social_cache = {}
        self.last_update = {}
        
    def get_live_social_sentiment(self, symbol: str) -> pd.DataFrame:
        """Get live social media sentiment for a symbol."""
        if not self.config.social_api_enabled or not self.config.social_api_key:
            logger.warning("Social API not enabled or configured")
            return pd.DataFrame()
        
        try:
            # TODO: Implement actual social media API integration
            # For now, return empty DataFrame to indicate no mock data
            logger.info(f"Social sentiment requested for {symbol} - API integration pending")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting social sentiment for {symbol}: {e}")
            return pd.DataFrame()

class PortfolioDataIntegrator:
    """Portfolio data integration for real-time capital and position tracking."""
    
    def __init__(self):
        self.portfolio_cache = {}
        self.last_update = {}
        
    def get_current_capital(self) -> float:
        """Get current available capital for trading."""
        try:
            # TODO: Implement actual portfolio manager integration
            # For now, return from environment variable
            capital = float(os.getenv('TRADING_CAPITAL', '100000.0'))
            logger.info(f"Current capital: {capital}")
            return capital
            
        except Exception as e:
            logger.error(f"Error getting current capital: {e}")
            return 0.0
    
    def get_current_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current trading positions."""
        try:
            # TODO: Implement actual position tracking
            # For now, return empty dict to indicate no mock data
            logger.info("Position data requested - portfolio integration pending")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return {}
    
    def get_portfolio_performance(self) -> Dict[str, float]:
        """Get current portfolio performance metrics."""
        try:
            # TODO: Implement actual performance calculation
            # For now, return empty dict to indicate no mock data
            logger.info("Portfolio performance requested - integration pending")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting portfolio performance: {e}")
            return {}

class RealTimeDataIntegrator:
    """Main real-time data integrator."""
    
    def __init__(self, config: RealTimeConfig = None):
        self.config = config or RealTimeConfig()
        self.market_data = RealTimeMarketData(self.config)
        self.news_data = RealTimeNewsData(self.config)
        self.social_data = RealTimeSocialData(self.config)
        self.portfolio = PortfolioDataIntegrator()
        
    async def initialize(self):
        """Initialize all real-time data connections."""
        await self.market_data.initialize_websocket()
        logger.info("Real-time data integrator initialized")
    
    def get_comprehensive_data(self, symbol: str, timeframes: List[str] = None) -> Dict[str, Any]:
        """Get comprehensive real-time data for a symbol."""
        if timeframes is None:
            timeframes = ["1min", "5min", "15min", "30min", "1hour", "1day"]
        
        try:
            # Get market data
            market_data = self.market_data.get_cached_data(symbol)
            multi_tf_data = self.market_data.get_multi_timeframe_data(symbol, timeframes)
            
            # Get news and social data
            news_data = self.news_data.get_live_news_sentiment(symbol)
            social_data = self.social_data.get_live_social_sentiment(symbol)
            
            # Get portfolio data
            current_capital = self.portfolio.get_current_capital()
            positions = self.portfolio.get_current_positions()
            performance = self.portfolio.get_portfolio_performance()
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'market_data': market_data,
                'multi_timeframe_data': multi_tf_data,
                'news_data': news_data,
                'social_data': social_data,
                'portfolio': {
                    'current_capital': current_capital,
                    'positions': positions,
                    'performance': performance
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive data for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def is_data_fresh(self, symbol: str, max_age_seconds: int = 60) -> bool:
        """Check if data is fresh (within max_age_seconds)."""
        cache_key = f"{symbol}_NSE"
        if cache_key in self.market_data.last_update:
            age = (datetime.now() - self.market_data.last_update[cache_key]).seconds
            return age <= max_age_seconds
        return False

# Global instance for easy access
# Global instance removed - instantiate locally as needed
