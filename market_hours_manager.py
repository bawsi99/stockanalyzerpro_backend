"""
market_hours_manager.py

Manages market hours, trading sessions, and optimizes data fetching based on market status.
Provides cost-efficient data handling for both live and historical data.
"""
import os
import logging
from datetime import datetime, time, timedelta
from typing import Dict, Any, Optional, Tuple
import pytz
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MarketStatus(Enum):
    """Market status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    POST_MARKET = "post_market"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"

@dataclass
class MarketSession:
    """Represents a market trading session."""
    start_time: time
    end_time: time
    name: str
    is_active: bool = True

@dataclass
class MarketHours:
    """Market hours configuration for Indian markets."""
    timezone: str = "Asia/Kolkata"
    regular_session: MarketSession = None
    pre_market: MarketSession = None
    post_market: MarketSession = None
    
    def __post_init__(self):
        if self.regular_session is None:
            self.regular_session = MarketSession(
                start_time=time(9, 15),  # 9:15 AM IST
                end_time=time(15, 30),   # 3:30 PM IST
                name="Regular Trading"
            )
        if self.pre_market is None:
            self.pre_market = MarketSession(
                start_time=time(9, 0),   # 9:00 AM IST
                end_time=time(9, 15),    # 9:15 AM IST
                name="Pre-Market",
                is_active=False  # Disabled by default
            )
        if self.post_market is None:
            self.post_market = MarketSession(
                start_time=time(15, 30), # 3:30 PM IST
                end_time=time(16, 0),    # 4:00 PM IST
                name="Post-Market",
                is_active=False  # Disabled by default
            )

class MarketHoursManager:
    """
    Manages market hours and provides cost-efficient data fetching strategies.
    """
    
    def __init__(self, market_hours: MarketHours = None):
        self.market_hours = market_hours or MarketHours()
        self.ist_tz = pytz.timezone(self.market_hours.timezone)
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)
        
        # Cost optimization settings
        self.live_data_cost_per_minute = 1.0  # Relative cost units
        self.historical_data_cost_per_request = 10.0  # Relative cost units
        self.websocket_cost_per_hour = 5.0  # Relative cost units
        
        # Data freshness thresholds
        self.live_data_threshold = timedelta(minutes=1)
        self.historical_data_threshold = timedelta(hours=1)
        
        logger.info("MarketHoursManager initialized")
    
    def get_current_ist_time(self) -> datetime:
        """Get current time in IST."""
        utc_now = datetime.now(pytz.UTC)
        return utc_now.astimezone(self.ist_tz)
    
    def is_weekend(self, dt: datetime = None) -> bool:
        """Check if the given date is a weekend."""
        if dt is None:
            dt = self.get_current_ist_time()
        return dt.weekday() >= 5  # Saturday = 5, Sunday = 6
    
    def is_market_holiday(self, dt: datetime = None) -> bool:
        """Check if the given date is a market holiday."""
        if dt is None:
            dt = self.get_current_ist_time()
        
        # TODO: Implement holiday calendar integration
        # For now, return False (assume no holidays)
        return False
    
    def get_market_status(self, dt: datetime = None) -> MarketStatus:
        """
        Get current market status.
        
        Returns:
            MarketStatus: Current market status
        """
        if dt is None:
            dt = self.get_current_ist_time()
        
        # Check cache first
        cache_key = dt.strftime('%Y%m%d_%H%M')
        if cache_key in self.cache:
            cached_time, cached_status = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return cached_status
        
        current_time = dt.time()
        
        # Check for weekend
        if self.is_weekend(dt):
            status = MarketStatus.WEEKEND
        # Check for holiday
        elif self.is_market_holiday(dt):
            status = MarketStatus.HOLIDAY
        # Check pre-market
        elif (self.market_hours.pre_market.is_active and 
              self.market_hours.pre_market.start_time <= current_time < self.market_hours.pre_market.end_time):
            status = MarketStatus.PRE_MARKET
        # Check regular market hours
        elif (self.market_hours.regular_session.start_time <= current_time <= self.market_hours.regular_session.end_time):
            status = MarketStatus.OPEN
        # Check post-market
        elif (self.market_hours.post_market.is_active and 
              self.market_hours.post_market.start_time <= current_time < self.market_hours.post_market.end_time):
            status = MarketStatus.POST_MARKET
        else:
            status = MarketStatus.CLOSED
        
        # Cache the result
        self.cache[cache_key] = (datetime.now(), status)
        
        return status
    
    def is_market_open(self, dt: datetime = None) -> bool:
        """Check if market is currently open for trading."""
        status = self.get_market_status(dt)
        return status == MarketStatus.OPEN
    
    def get_next_market_open(self, dt: datetime = None) -> datetime:
        """Get the next market opening time."""
        if dt is None:
            dt = self.get_current_ist_time()
        
        # If it's weekend, find next Monday
        while self.is_weekend(dt) or self.is_market_holiday(dt):
            dt += timedelta(days=1)
        
        # Set to market open time
        next_open = dt.replace(
            hour=self.market_hours.regular_session.start_time.hour,
            minute=self.market_hours.regular_session.start_time.minute,
            second=0,
            microsecond=0
        )
        
        # If today's market open has passed, get next day
        if next_open <= dt:
            next_open += timedelta(days=1)
            # Skip weekends and holidays
            while self.is_weekend(next_open) or self.is_market_holiday(next_open):
                next_open += timedelta(days=1)
        
        return next_open
    
    def get_last_market_close(self, dt: datetime = None) -> datetime:
        """Get the last market closing time."""
        if dt is None:
            dt = self.get_current_ist_time()
        
        # Set to market close time
        last_close = dt.replace(
            hour=self.market_hours.regular_session.end_time.hour,
            minute=self.market_hours.regular_session.end_time.minute,
            second=0,
            microsecond=0
        )
        
        # If today's market close hasn't happened yet, get previous day
        if last_close > dt:
            last_close -= timedelta(days=1)
            # Skip weekends and holidays
            while self.is_weekend(last_close) or self.is_market_holiday(last_close):
                last_close -= timedelta(days=1)
        
        return last_close
    
    def get_optimal_data_strategy(self, symbol: str, interval: str = "1d") -> Dict[str, Any]:
        """
        Get optimal data fetching strategy based on market status and cost efficiency.
        
        Args:
            symbol: Stock symbol
            interval: Data interval (1m, 5m, 15m, 1h, 1d)
            
        Returns:
            Dict containing optimal strategy
        """
        current_status = self.get_market_status()
        current_time = self.get_current_ist_time()
        
        strategy = {
            "market_status": current_status.value,
            "current_time": current_time.isoformat(),
            "recommended_approach": "historical",
            "reason": "",
            "cost_efficiency": "high",
            "data_freshness": "last_close",
            "websocket_recommended": False,
            "cache_duration": 3600,  # 1 hour default
            "next_update": None
        }
        
        # Determine optimal strategy based on market status and interval
        if current_status == MarketStatus.OPEN:
            if interval in ["1m", "5m", "15m"]:
                # Short intervals during market hours - use live data
                strategy.update({
                    "recommended_approach": "live",
                    "data_freshness": "real_time",
                    "websocket_recommended": True,
                    "cache_duration": 60,  # 1 minute
                    "cost_efficiency": "medium"
                })
            else:
                # Longer intervals - use historical with periodic updates
                strategy.update({
                    "recommended_approach": "historical",
                    "data_freshness": "near_real_time",
                    "cache_duration": 300,  # 5 minutes
                    "cost_efficiency": "high"
                })
        else:
            # Market closed - use historical data
            strategy.update({
                "recommended_approach": "historical",
                "data_freshness": "last_close",
                "cache_duration": 3600,  # 1 hour
                "cost_efficiency": "high",
                "reason": f"Market is {current_status.value}"
            })
            
            # Set next update time
            if current_status == MarketStatus.CLOSED:
                strategy["next_update"] = self.get_next_market_open().isoformat()
        
        return strategy
    
    def should_use_websocket(self, symbol: str, interval: str = "1d") -> bool:
        """Determine if WebSocket should be used for the given symbol and interval."""
        strategy = self.get_optimal_data_strategy(symbol, interval)
        return strategy["websocket_recommended"]
    
    def get_cache_duration(self, symbol: str, interval: str = "1d") -> int:
        """Get recommended cache duration for the given symbol and interval."""
        strategy = self.get_optimal_data_strategy(symbol, interval)
        return strategy["cache_duration"]
    
    def estimate_data_cost(self, approach: str, duration_hours: float = 1.0) -> float:
        """
        Estimate the cost of data fetching for a given approach and duration.
        
        Args:
            approach: "live", "historical", or "websocket"
            duration_hours: Duration in hours
            
        Returns:
            Estimated cost in relative units
        """
        if approach == "live":
            return self.live_data_cost_per_minute * duration_hours * 60
        elif approach == "historical":
            return self.historical_data_cost_per_request
        elif approach == "websocket":
            return self.websocket_cost_per_hour * duration_hours
        else:
            return 0.0
    
    def get_market_info(self) -> Dict[str, Any]:
        """Get comprehensive market information."""
        current_time = self.get_current_ist_time()
        status = self.get_market_status()
        
        info = {
            "current_time": current_time.isoformat(),
            "timezone": self.market_hours.timezone,
            "market_status": status.value,
            "is_weekend": self.is_weekend(),
            "is_holiday": self.is_market_holiday(),
            "regular_session": {
                "start": self.market_hours.regular_session.start_time.isoformat(),
                "end": self.market_hours.regular_session.end_time.isoformat(),
                "name": self.market_hours.regular_session.name
            }
        }
        
        if status == MarketStatus.CLOSED:
            info["next_market_open"] = self.get_next_market_open().isoformat()
        elif status == MarketStatus.OPEN:
            info["market_close"] = current_time.replace(
                hour=self.market_hours.regular_session.end_time.hour,
                minute=self.market_hours.regular_session.end_time.minute,
                second=0,
                microsecond=0
            ).isoformat()
        
        return info

# Global instance
market_hours_manager = MarketHoursManager() 