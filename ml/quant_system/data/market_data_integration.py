import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import asyncio
import aiohttp
from dataclasses import dataclass, field

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from backend directory (one level up from current file)
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(env_path)
    print(f"Loaded .env file from: {env_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"ZERODHA_API_KEY: {os.getenv('ZERODHA_API_KEY', 'Not set')[:8]}...")
    print(f"ZERODHA_API_SECRET: {os.getenv('ZERODHA_API_SECRET', 'Not set')[:8]}...")
except ImportError:
    print("dotenv package not available, using system environment variables")
except Exception as e:
    print(f"Error loading .env file: {e}")

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MarketDataConfig:
    """Configuration for market data sources."""
    zerodha_api_key: str = ""
    zerodha_api_secret: str = ""
    news_api_key: str = ""
    twitter_api_key: str = ""
    
    enable_zerodha: bool = True  # ONLY SOURCE - Indian markets
    enable_yahoo_finance: bool = False
    enable_news_api: bool = False
    enable_social_api: bool = False
    
    default_timeframe: str = "5minute"
    max_retries: int = 3
    request_timeout: int = 30
    cache_duration: int = 300
    
    requests_per_minute: int = 60
    requests_per_second: int = 10
    
    def __post_init__(self):
        self.zerodha_api_key = os.getenv('ZERODHA_API_KEY', self.zerodha_api_key)
        self.zerodha_api_secret = os.getenv('ZERODHA_API_SECRET', self.zerodha_api_secret)
        self.news_api_key = os.getenv('NEWS_API_KEY', self.news_api_key)
        self.twitter_api_key = os.getenv('TWITTER_API_KEY', self.twitter_api_key)

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, requests_per_minute: int = 60, requests_per_second: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self.minute_requests = []
        self.second_requests = []
    
    async def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        now = time.time()
        
        # Clean old requests
        self.minute_requests = [req_time for req_time in self.minute_requests if now - req_time < 60]
        self.second_requests = [req_time for req_time in self.second_requests if now - req_time < 1]
        
        # Check limits
        if len(self.minute_requests) >= self.requests_per_minute:
            wait_time = 60 - (now - self.minute_requests[0])
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        if len(self.second_requests) >= self.requests_per_second:
            wait_time = 1 - (now - self.second_requests[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.minute_requests.append(now)
        self.second_requests.append(now)

class ZerodhaDataProvider:
    """Zerodha data provider using existing data service."""
    
    def __init__(self, config: MarketDataConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.requests_per_minute, config.requests_per_second)
        self.base_url = "http://localhost:8000"  # Data service endpoint
        
        # Check if data service is available
        self.data_service_available = True
        logger.info("Zerodha data provider initialized using data service")
    
    async def get_historical_data(self, symbol: str, exchange: str = "NSE", 
                                interval: str = "5minute", period: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data from data service."""
        if not self.data_service_available:
            logger.error("Data service not available")
            return None
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            # Map interval to data service format
            interval_mapping = {
                '5minute': '5min',
                '15minute': '15min',
                '30minute': '30min',
                '60minute': '1h',
                'day': '1day'
            }
            service_interval = interval_mapping.get(interval, '5min')
            
            # Call data service endpoint
            url = f"{self.base_url}/stock/{symbol}/history"
            params = {
                'interval': service_interval,
                'exchange': exchange,
                'limit': period
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('success') and data.get('candles'):
                            # Convert candles to DataFrame
                            candles = data['candles']
                            df_data = []
                            for candle in candles:
                                df_data.append({
                                    'date': pd.to_datetime(candle['time'], unit='s'),
                                    'open': candle['open'],
                                    'high': candle['high'],
                                    'low': candle['low'],
                                    'close': candle['close'],
                                    'volume': candle['volume']
                                })
                            
                            df = pd.DataFrame(df_data)
                            df.set_index('date', inplace=True)
                            
                            logger.info(f"Retrieved {len(df)} records for {symbol} from data service")
                            return df
                        else:
                            logger.warning(f"No data received from data service for {symbol}")
                            return None
                    else:
                        logger.error(f"Data service error for {symbol}: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error getting data from data service for {symbol}: {e}")
            return None
    
    async def get_current_quote(self, symbol: str, exchange: str = "NSE") -> Optional[Dict]:
        """Get current quote from data service."""
        if not self.data_service_available:
            return None
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            # Call data service quote endpoint
            url = f"{self.base_url}/stock/{symbol}/info"
            params = {'exchange': exchange}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('success') and data.get('quote'):
                            logger.info(f"Retrieved quote for {symbol} from data service")
                            return data['quote']
                        else:
                            logger.warning(f"No quote received from data service for {symbol}")
                            return None
                    else:
                        logger.error(f"Data service error for quote {symbol}: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error getting quote from data service for {symbol}: {e}")
            return None

class NewsDataProvider:
    """News data provider."""
    
    def __init__(self, config: MarketDataConfig):
        self.config = config
        self.api_key = config.news_api_key
        self.enabled = config.enable_news_api
    
    async def get_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Get news for a symbol."""
        if not self.enabled or not self.api_key:
            return []
        
        try:
            # Simulate news data for now
            return [
                {
                    "title": f"Market update for {symbol}",
                    "summary": f"Recent developments in {symbol} stock",
                    "date": datetime.now().isoformat(),
                    "sentiment": "neutral"
                }
            ]
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {e}")
            return []

class ProductionMarketDataIntegrator:
    """Production-ready market data integrator with Zerodha as the only source."""
    
    def __init__(self, config: MarketDataConfig = None):
        self.config = config or MarketDataConfig()
        self.cache = {}
        self.cache_timestamps = {}
        
        self.zerodha_provider = ZerodhaDataProvider(self.config)
        self.news_provider = NewsDataProvider(self.config)
        
        logger.info("Production Market Data Integrator initialized")
        logger.info("Data source: Data Service (Zerodha backend)")
    
    async def get_comprehensive_market_data(self, symbol: str, exchange: str = "NSE", 
                                          interval: str = "5minute", period: int = 30) -> Dict[str, Any]:
        """Get comprehensive market data for a symbol."""
        try:
            # Get price data from Zerodha
            price_data = await self.zerodha_provider.get_historical_data(
                symbol, exchange, interval, period
            )
            
            if price_data is None or price_data.empty:
                return {
                    "success": False,
                    "error": f"No data received from Zerodha for {symbol}",
                    "symbol": symbol,
                    "exchange": exchange,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Get current quote
            current_quote = await self.zerodha_provider.get_current_quote(symbol, exchange)
            
            # Get news data
            news_data = await self.news_provider.get_news(symbol)
            
            # Build comprehensive response
            result = {
                "success": True,
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": datetime.now().isoformat(),
                "data_source": "Zerodha",
                "price_data": {
                    "records": len(price_data),
                    "start_date": price_data.index[0].isoformat() if not price_data.empty else None,
                    "end_date": price_data.index[-1].isoformat() if not price_data.empty else None,
                    "latest_price": float(price_data['close'].iloc[-1]) if not price_data.empty else None,
                    "price_change": float(price_data['close'].iloc[-1] - price_data['close'].iloc[-2]) if len(price_data) > 1 else 0,
                    "volume": float(price_data['volume'].iloc[-1]) if not price_data.empty else None
                },
                "current_quote": current_quote,
                "news_data": news_data,
                "technical_indicators": self._calculate_basic_indicators(price_data),
                "market_status": "live" if self._is_market_open() else "closed"
            }
            
            # Cache the result
            self._cache_result(symbol, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting comprehensive market data for {symbol}: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_basic_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic technical indicators."""
        if data.empty:
            return {}
        
        try:
            close_prices = data['close']
            volume = data['volume']
            
            # Simple moving averages
            sma_20 = close_prices.rolling(window=20).mean().iloc[-1] if len(close_prices) >= 20 else None
            sma_50 = close_prices.rolling(window=50).mean().iloc[-1] if len(close_prices) >= 50 else None
            
            # RSI (simplified)
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None
            
            # Volume analysis
            avg_volume = volume.rolling(window=20).mean().iloc[-1] if len(volume) >= 20 else None
            current_volume = volume.iloc[-1] if not volume.empty else None
            volume_ratio = current_volume / avg_volume if avg_volume and avg_volume > 0 else None
            
            return {
                "sma_20": float(sma_20) if sma_20 is not None else None,
                "sma_50": float(sma_50) if sma_50 is not None else None,
                "rsi_14": float(current_rsi) if current_rsi is not None else None,
                "volume_ratio": float(volume_ratio) if volume_ratio is not None else None,
                "current_volume": float(current_volume) if current_volume is not None else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def _is_market_open(self) -> bool:
        """Check if Indian market is open."""
        now = datetime.now()
        ist_time = now + timedelta(hours=5, minutes=30)  # Convert to IST
        
        # Check if it's a weekday
        if ist_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's market hours (9:15 AM to 3:30 PM IST)
        market_open = ist_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = ist_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= ist_time <= market_close
    
    def _cache_result(self, symbol: str, result: Dict[str, Any]):
        """Cache the result."""
        self.cache[symbol] = result
        self.cache_timestamps[symbol] = datetime.now()
        
        # Clean old cache entries
        cutoff_time = datetime.now() - timedelta(seconds=self.config.cache_duration)
        expired_symbols = [
            symbol for symbol, timestamp in self.cache_timestamps.items()
            if timestamp < cutoff_time
        ]
        
        for expired_symbol in expired_symbols:
            del self.cache[expired_symbol]
            del self.cache_timestamps[expired_symbol]
    
    def get_cached_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached data if still valid."""
        if symbol in self.cache:
            timestamp = self.cache_timestamps.get(symbol)
            if timestamp and (datetime.now() - timestamp).total_seconds() < self.config.cache_duration:
                return self.cache[symbol]
        return None

# Example usage
if __name__ == "__main__":
    async def main():
        config = MarketDataConfig()
        integrator = ProductionMarketDataIntegrator(config)
        
        # Test with a stock
        result = await integrator.get_comprehensive_market_data("RELIANCE")
        print(json.dumps(result, indent=2, default=str))
    
    asyncio.run(main())
