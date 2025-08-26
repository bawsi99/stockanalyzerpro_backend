#!/usr/bin/env python3
"""
Central Data Provider Module

This module implements a centralized data provider that caches data fetches and shares
them across different components in the system. It helps to reduce redundant data fetches
and improve performance.
"""

import os
import logging
import time
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# Library modules should not configure global logging. Use module logger only.
logger = logging.getLogger(__name__)

class CentralDataProvider:
    """
    Centralized data provider that caches and shares data across components.
    Implements singleton pattern to ensure only one instance exists.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(CentralDataProvider, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the data provider with caches."""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        # Try to initialize Redis cache manager
        self.redis_cache_manager = None
        try:
            from redis_cache_manager import get_redis_cache_manager
            self.redis_cache_manager = get_redis_cache_manager()
            logger.info("Using Redis cache manager")
        except Exception as e:
            logger.warning(f"Redis cache manager not available: {e}")
            logger.info("Falling back to local cache")
        
        # Initialize local caches as fallback
        self.stock_data_cache = {}  # {(symbol, exchange, interval, period): (data, timestamp)}
        self.index_data_cache = {}  # {(index, interval, period): (data, timestamp)}
        self.sector_data_cache = {}  # {sector: (data, timestamp)}
        self.indicator_cache = {}   # {(symbol, interval): (indicators, timestamp)}
        self.pattern_cache = {}     # {(symbol, interval): (patterns, timestamp)}
        
        # Cache expiration settings (in seconds)
        self.cache_expiration = {
            'stock_data': 300,      # 5 minutes
            'index_data': 600,      # 10 minutes
            'sector_data': 3600,    # 1 hour
            'indicator_data': 600,  # 10 minutes
            'pattern_data': 1800    # 30 minutes
        }
        
        self._initialized = True
        logger.info("CentralDataProvider initialized")

    # -------------------- ASYNC WRAPPERS (non-blocking) --------------------
    async def get_stock_data_async(self, symbol: str, exchange: str, interval: str, period: int,
                                   force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Async wrapper to fetch stock data without blocking the event loop."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_stock_data(symbol, exchange, interval, period, force_refresh)
        )

    async def get_patterns_async(self, symbol: str, exchange: str, interval: str,
                                 data: Optional[pd.DataFrame] = None,
                                 force_refresh: bool = False) -> Optional[Dict]:
        """Async wrapper to detect patterns without blocking the event loop."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_patterns(symbol, exchange, interval, data, force_refresh)
        )
    
    def get_stock_data(self, symbol: str, exchange: str, interval: str, period: int, 
                      force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get stock data from cache or fetch it if not available.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            interval: Data interval (e.g., 'day', 'minute', etc.)
            period: Historical period in days
            force_refresh: Force a refresh even if cached data exists
            
        Returns:
            DataFrame with stock data or None if failed
        """
        # Try Redis cache first
        if self.redis_cache_manager and not force_refresh:
            cached_data = self.redis_cache_manager.get('stock_data', symbol, exchange, interval, period)
            if cached_data is not None:
                logger.debug(f"Redis cache hit for {symbol} {interval} data")
                return cached_data
        
        # Fallback to local cache
        cache_key = (symbol, exchange, interval, period)
        if not force_refresh and cache_key in self.stock_data_cache:
            data, timestamp = self.stock_data_cache[cache_key]
            if time.time() - timestamp < self.cache_expiration['stock_data']:
                logger.debug(f"Local cache hit for {symbol} {interval} data")
                return data
        
        # If not in cache or expired, fetch new data
        try:
            # Import ZerodhaDataClient here to avoid circular imports
            from zerodha_client import ZerodhaDataClient
            client = ZerodhaDataClient()
            
            # Ensure authenticated (idempotent call)
            if not client.authenticate():
                logger.error(f"Authentication failed when fetching {symbol} data")
                return None
            
            # Fetch data
            data = client.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                period=period
            )
            
            # Cache the result
            if data is not None and not data.empty:
                # Cache in Redis if available
                if self.redis_cache_manager:
                    self.redis_cache_manager.set('stock_data', data, self.cache_expiration['stock_data'], 
                                               symbol, exchange, interval, period)
                
                # Also cache locally as fallback
                self.stock_data_cache[cache_key] = (data, time.time())
                return data
            else:
                logger.error(f"Fetched empty data for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return None
    
    def get_nifty50_data(self, interval: str = "day", period: int = 365, 
                       force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Specialized method to get NIFTY 50 data, optimized for reuse across components.
        This is particularly important as NIFTY 50 data is frequently used as a benchmark.
        
        Args:
            interval: Data interval (e.g., 'day', 'minute', etc.)
            period: Historical period in days
            force_refresh: Force a refresh even if cached data exists
            
        Returns:
            DataFrame with NIFTY 50 data or None if failed
        """
        return self.get_index_data("NIFTY 50", interval, period, force_refresh)
    
    def get_index_data(self, index: str, interval: str, period: int, 
                      force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get index data from cache or fetch it if not available.
        
        Args:
            index: Index symbol
            interval: Data interval (e.g., 'day', 'minute', etc.)
            period: Historical period in days
            force_refresh: Force a refresh even if cached data exists
            
        Returns:
            DataFrame with index data or None if failed
        """
        cache_key = (index, interval, period)
        
        # Check if we have non-expired data in cache
        if not force_refresh and cache_key in self.index_data_cache:
            data, timestamp = self.index_data_cache[cache_key]
            if time.time() - timestamp < self.cache_expiration['index_data']:
                logger.debug(f"Cache hit for {index} {interval} data")
                return data
        
        # If not in cache or expired, fetch new data
        try:
            # Import ZerodhaDataClient here to avoid circular imports
            from zerodha_client import ZerodhaDataClient
            client = ZerodhaDataClient()
            
            # Ensure authenticated (idempotent call)
            if not client.authenticate():
                logger.error(f"Authentication failed when fetching {index} data")
                return None
            
            # Fetch data
            data = client.get_historical_data(
                symbol=index,
                exchange="NSE",  # Indices are typically on NSE
                interval=interval,
                period=period
            )
            
            # Cache the result
            if data is not None and not data.empty:
                self.index_data_cache[cache_key] = (data, time.time())
                return data
            else:
                logger.error(f"Fetched empty data for {index}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching index data for {index}: {e}")
            return None
    
    def get_sector_data(self, sector: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get sector data from cache or fetch it if not available.
        
        Args:
            sector: Sector name
            force_refresh: Force a refresh even if cached data exists
            
        Returns:
            Dict with sector data or None if failed
        """
        # Check if we have non-expired data in cache
        if not force_refresh and sector in self.sector_data_cache:
            data, timestamp = self.sector_data_cache[sector]
            if time.time() - timestamp < self.cache_expiration['sector_data']:
                logger.debug(f"Cache hit for {sector} sector data")
                return data
        
        # If not in cache or expired, fetch new data
        try:
            # Import SectorClassifier here to avoid circular imports
            from sector_classifier import sector_classifier
            
            # Get sector stocks and index
            stocks = sector_classifier.get_sector_stocks(sector)
            index = sector_classifier.get_primary_sector_index(sector)
            display_name = sector_classifier.get_sector_display_name(sector)
            
            sector_data = {
                "sector": sector,
                "display_name": display_name,
                "primary_index": index,
                "stocks": stocks,
                "stock_count": len(stocks)
            }
            
            # Cache the result
            self.sector_data_cache[sector] = (sector_data, time.time())
            return sector_data
                
        except Exception as e:
            logger.error(f"Error fetching sector data for {sector}: {e}")
            return None
    
    def get_technical_indicators(self, symbol: str, exchange: str, interval: str,
                               data: Optional[pd.DataFrame] = None,
                               force_refresh: bool = False) -> Optional[Dict]:
        """
        Get technical indicators from cache or calculate them if not available.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            interval: Data interval
            data: Optional pre-fetched data
            force_refresh: Force a refresh even if cached data exists
            
        Returns:
            Dict with technical indicators or None if failed
        """
        cache_key = (symbol, exchange, interval)
        
        # Check if we have non-expired data in cache
        if not force_refresh and cache_key in self.indicator_cache:
            indicators, timestamp = self.indicator_cache[cache_key]
            if time.time() - timestamp < self.cache_expiration['indicator_data']:
                logger.debug(f"Cache hit for {symbol} indicators")
                return indicators
        
        # If not in cache or expired, calculate indicators
        try:
            # Get data if not provided
            if data is None:
                period = 365  # Default to 1 year of data
                data = self.get_stock_data(symbol, exchange, interval, period)
                
            if data is None or data.empty:
                logger.error(f"No data available for {symbol} indicators calculation")
                return None
            
            # Import TechnicalIndicators here to avoid circular imports
            from technical_indicators import TechnicalIndicators
            
            # Calculate indicators
            indicators = TechnicalIndicators.calculate_all_indicators_optimized(data, symbol)
            
            # Cache the result
            self.indicator_cache[cache_key] = (indicators, time.time())
            return indicators
                
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None
    
    def get_patterns(self, symbol: str, exchange: str, interval: str,
                    data: Optional[pd.DataFrame] = None,
                    force_refresh: bool = False) -> Optional[Dict]:
        """
        Get pattern recognition results from cache or calculate them if not available.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            interval: Data interval
            data: Optional pre-fetched data
            force_refresh: Force a refresh even if cached data exists
            
        Returns:
            Dict with pattern data or None if failed
        """
        cache_key = (symbol, exchange, interval)
        
        # Check if we have non-expired data in cache
        if not force_refresh and cache_key in self.pattern_cache:
            patterns, timestamp = self.pattern_cache[cache_key]
            if time.time() - timestamp < self.cache_expiration['pattern_data']:
                logger.debug(f"Cache hit for {symbol} patterns")
                return patterns
        
        # If not in cache or expired, detect patterns
        try:
            # Get data if not provided
            if data is None:
                period = 365  # Default to 1 year of data
                data = self.get_stock_data(symbol, exchange, interval, period)
                
            if data is None or data.empty:
                logger.error(f"No data available for {symbol} pattern detection")
                return None
            
            # Import PatternRecognition here to avoid circular imports
            from patterns.recognition import PatternRecognition
            
            # Detect various patterns
            patterns = {}
            
            # Candlestick patterns
            patterns['candlestick_patterns'] = PatternRecognition.detect_candlestick_patterns(data)
            
            # Chart patterns
            patterns['double_tops'] = PatternRecognition.detect_double_top(data['close'])
            patterns['double_bottoms'] = PatternRecognition.detect_double_bottom(data['close'])
            patterns['head_and_shoulders'] = PatternRecognition.detect_head_and_shoulders(data['close'])
            patterns['triangles'] = PatternRecognition.detect_triangle(data['close'])
            
            # Advanced patterns (guard each call for API drift)
            adv = {}
            try:
                adv['head_and_shoulders'] = PatternRecognition.detect_head_and_shoulders(data['close'])
            except Exception:
                adv['head_and_shoulders'] = []
            try:
                adv['inverse_head_and_shoulders'] = PatternRecognition.detect_inverse_head_and_shoulders(data['close'])
            except Exception:
                adv['inverse_head_and_shoulders'] = []
            try:
                adv['cup_and_handle'] = PatternRecognition.detect_cup_and_handle(data['close'])
            except Exception:
                adv['cup_and_handle'] = []
            try:
                adv['triple_tops'] = PatternRecognition.detect_triple_top(data['close'])
            except Exception:
                adv['triple_tops'] = []
            try:
                adv['triple_bottoms'] = PatternRecognition.detect_triple_bottom(data['close'])
            except Exception:
                adv['triple_bottoms'] = []
            # Wedge and Channel methods may differ across versions; try alternative names
            try:
                adv['wedge_patterns'] = PatternRecognition.detect_wedge_patterns(data['close'])
            except Exception:
                try:
                    adv['wedge_patterns'] = PatternRecognition.detect_wedge(data['close'])
                except Exception:
                    adv['wedge_patterns'] = []
            try:
                adv['channel_patterns'] = PatternRecognition.detect_channel_patterns(data['close'])
            except Exception:
                adv['channel_patterns'] = []

            patterns['advanced_patterns'] = adv
            
            # Cache the result
            self.pattern_cache[cache_key] = (patterns, time.time())
            return patterns
                
        except Exception as e:
            logger.error(f"Error detecting patterns for {symbol}: {e}")
            return None
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear specified cache or all caches.
        
        Args:
            cache_type: Type of cache to clear ('stock_data', 'index_data', 'sector_data', 
                      'indicator_data', 'pattern_data') or None to clear all
        """
        if cache_type == 'stock_data' or cache_type is None:
            self.stock_data_cache = {}
        if cache_type == 'index_data' or cache_type is None:
            self.index_data_cache = {}
        if cache_type == 'sector_data' or cache_type is None:
            self.sector_data_cache = {}
        if cache_type == 'indicator_data' or cache_type is None:
            self.indicator_cache = {}
        if cache_type == 'pattern_data' or cache_type is None:
            self.pattern_cache = {}
            
        logger.info(f"Cleared {'all caches' if cache_type is None else cache_type + ' cache'}")

    # -------------------- Cache Mutation APIs --------------------
    def set_patterns_cache(self, symbol: str, exchange: str, interval: str, patterns: Dict[str, Any]) -> None:
        """Persist externally computed patterns into the cache for reuse."""
        try:
            key = (symbol, exchange, interval)
            self.pattern_cache[key] = (patterns or {}, time.time())
        except Exception:
            # Non-fatal: avoid raising from cache setter
            pass

# Create a singleton instance
central_data_provider = CentralDataProvider()
