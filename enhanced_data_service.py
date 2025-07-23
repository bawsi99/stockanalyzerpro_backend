"""
enhanced_data_service.py

Enhanced data service that provides cost-efficient data fetching based on market status.
Integrates with market hours manager to optimize API usage and storage costs.
"""
import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from dataclasses import dataclass

from market_hours_manager import market_hours_manager, MarketStatus
from zerodha_client import ZerodhaDataClient
from zerodha_ws_client import zerodha_ws_client

logger = logging.getLogger(__name__)

@dataclass
class DataRequest:
    """Represents a data request with optimization parameters."""
    symbol: str
    exchange: str = "NSE"
    interval: str = "1d"
    period: int = 365
    force_live: bool = False
    cache_duration: Optional[int] = None

@dataclass
class DataResponse:
    """Represents a data response with metadata."""
    data: pd.DataFrame
    data_freshness: str
    market_status: str
    source: str
    cache_until: Optional[datetime] = None
    cost_estimate: float = 0.0
    optimization_applied: bool = False

class EnhancedDataService:
    """
    Enhanced data service that optimizes data fetching based on market status.
    Provides cost-efficient data handling for both live and historical data.
    """
    
    def __init__(self):
        self.zerodha_client = ZerodhaDataClient()
        self.market_hours_manager = market_hours_manager
        self.data_cache = {}
        self.cache_metadata = {}
        
        # Cost tracking
        self.total_cost = 0.0
        self.request_count = 0
        self.optimization_savings = 0.0
        
        logger.info("EnhancedDataService initialized")
    
    def get_optimal_data(self, request: DataRequest) -> DataResponse:
        """
        Get optimal data based on market status and cost efficiency.
        
        Args:
            request: DataRequest object with parameters
            
        Returns:
            DataResponse with data and metadata
        """
        self.request_count += 1
        
        # Get optimal strategy
        strategy = self.market_hours_manager.get_optimal_data_strategy(
            request.symbol, request.interval
        )
        
        # Check cache first
        cache_key = f"{request.symbol}_{request.exchange}_{request.interval}_{request.period}"
        if cache_key in self.data_cache:
            cached_data, metadata = self.data_cache[cache_key]
            if self._is_cache_valid(metadata, strategy["cache_duration"]):
                logger.info(f"Using cached data for {request.symbol}")
                return DataResponse(
                    data=cached_data,
                    data_freshness=metadata["data_freshness"],
                    market_status=metadata["market_status"],
                    source="cache",
                    cache_until=metadata["cache_until"],
                    cost_estimate=0.0,
                    optimization_applied=True
                )
        
        # Determine data source based on strategy
        if strategy["recommended_approach"] == "live" and not request.force_live:
            data, source = self._get_live_data(request)
        else:
            data, source = self._get_historical_data(request)
        
        if data is None or data.empty:
            logger.error(f"Failed to get data for {request.symbol}")
            return DataResponse(
                data=pd.DataFrame(),
                data_freshness="error",
                market_status=strategy["market_status"],
                source="error",
                cost_estimate=0.0,
                optimization_applied=False
            )
        
        # Calculate cost
        cost = self.market_hours_manager.estimate_data_cost(
            strategy["recommended_approach"], 0.1  # Assume 6 minutes of usage
        )
        self.total_cost += cost
        
        # Create response
        response = DataResponse(
            data=data,
            data_freshness=strategy["data_freshness"],
            market_status=strategy["market_status"],
            source=source,
            cache_until=datetime.now() + timedelta(seconds=strategy["cache_duration"]),
            cost_estimate=cost,
            optimization_applied=True
        )
        
        # Cache the data
        self._cache_data(cache_key, data, response)
        
        return response
    
    def _get_live_data(self, request: DataRequest) -> Tuple[Optional[pd.DataFrame], str]:
        """Get live data from WebSocket or real-time API."""
        try:
            # Get token for the symbol
            token = self.zerodha_client.get_instrument_token(request.symbol, request.exchange)
            if token is None:
                logger.warning(f"Token not found for {request.symbol}, falling back to historical")
                return self._get_historical_data(request)
            
            # Check if we have recent WebSocket data
            if request.interval in ["1m", "5m", "15m"]:
                candle = zerodha_ws_client.get_latest_candle(token, request.interval)
                if candle:
                    # Convert to DataFrame
                    df = pd.DataFrame([candle])
                    df['datetime'] = pd.to_datetime(df['start'], unit='s')
                    df = df.set_index('datetime')
                    return df, "websocket"
            
            # Fallback to real-time quote
            quote = self.zerodha_client.get_quote(request.symbol, request.exchange)
            if quote:
                # Create minimal DataFrame from quote
                df = pd.DataFrame([{
                    'open': quote.get('open', 0),
                    'high': quote.get('high', 0),
                    'low': quote.get('low', 0),
                    'close': quote.get('last_price', 0),
                    'volume': quote.get('volume', 0)
                }])
                df.index = [datetime.now()]
                return df, "realtime_quote"
            
        except Exception as e:
            logger.error(f"Error getting live data for {request.symbol}: {e}")
        
        return None, "error"
    
    def _get_historical_data(self, request: DataRequest) -> Tuple[Optional[pd.DataFrame], str]:
        """Get historical data from Zerodha API."""
        try:
            # Map interval format to Zerodha format
            interval_mapping = {
                "1m": "minute",
                "5m": "5minute", 
                "15m": "15minute",
                "1h": "60minute",
                "1d": "day"
            }
            
            zerodha_interval = interval_mapping.get(request.interval, request.interval)
            
            data = self.zerodha_client.get_historical_data(
                symbol=request.symbol,
                exchange=request.exchange,
                interval=zerodha_interval,
                period=request.period
            )
            
            if data is not None and not data.empty:
                return data, "historical_api"
            
        except Exception as e:
            logger.error(f"Error getting historical data for {request.symbol}: {e}")
        
        return None, "error"
    
    def _is_cache_valid(self, metadata: Dict[str, Any], cache_duration: int) -> bool:
        """Check if cached data is still valid."""
        if "cache_until" not in metadata:
            return False
        
        cache_until = metadata["cache_until"]
        if isinstance(cache_until, str):
            cache_until = datetime.fromisoformat(cache_until)
        
        return datetime.now() < cache_until
    
    def _cache_data(self, key: str, data: pd.DataFrame, response: DataResponse) -> None:
        """Cache data with metadata."""
        metadata = {
            "data_freshness": response.data_freshness,
            "market_status": response.market_status,
            "source": response.source,
            "cache_until": response.cache_until.isoformat() if response.cache_until else None,
            "cached_at": datetime.now().isoformat()
        }
        
        self.data_cache[key] = (data, metadata)
        
        # Limit cache size
        if len(self.data_cache) > 100:
            # Remove oldest entries
            oldest_key = min(self.data_cache.keys(), 
                           key=lambda k: self.data_cache[k][1].get("cached_at", ""))
            del self.data_cache[oldest_key]
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status and optimization information."""
        return self.market_hours_manager.get_market_info()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "total_requests": self.request_count,
            "total_cost": self.total_cost,
            "average_cost_per_request": self.total_cost / max(self.request_count, 1),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "market_status": self.market_hours_manager.get_market_status().value,
            "websocket_stats": zerodha_ws_client.get_optimization_stats()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        # This is a simplified calculation
        # In a real implementation, you'd track cache hits vs misses
        return 0.0
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.data_cache.clear()
        self.cache_metadata.clear()
        logger.info("Data cache cleared")
    
    def get_cost_analysis(self) -> Dict[str, Any]:
        """Get detailed cost analysis."""
        strategy = self.market_hours_manager.get_optimal_data_strategy("RELIANCE", "1d")
        
        return {
            "current_strategy": strategy,
            "cost_comparison": {
                "live_data": self.market_hours_manager.estimate_data_cost("live", 1.0),
                "historical_data": self.market_hours_manager.estimate_data_cost("historical", 1.0),
                "websocket": self.market_hours_manager.estimate_data_cost("websocket", 1.0)
            },
            "recommendations": self._get_cost_recommendations()
        }
    
    def _get_cost_recommendations(self) -> List[str]:
        """Get cost optimization recommendations."""
        recommendations = []
        market_status = self.market_hours_manager.get_market_status()
        
        if market_status in [MarketStatus.CLOSED, MarketStatus.WEEKEND, MarketStatus.HOLIDAY]:
            recommendations.append("Use historical data during market closed hours")
            recommendations.append("Disable WebSocket connections to save costs")
            recommendations.append("Increase cache duration for closed market periods")
        
        if market_status == MarketStatus.OPEN:
            recommendations.append("Use WebSocket for short intervals (1m, 5m, 15m)")
            recommendations.append("Use historical API for longer intervals")
            recommendations.append("Cache data for 1-5 minutes during market hours")
        
        return recommendations

# Global instance
enhanced_data_service = EnhancedDataService() 