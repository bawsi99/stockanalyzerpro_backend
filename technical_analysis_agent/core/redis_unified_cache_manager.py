"""
Unified Redis Cache Manager

This module provides a comprehensive Redis-based caching system that replaces all local caching:
- Stock data (historical prices, indicators)
- Technical analysis results
- Pattern recognition results
- Sector analysis data
- ML model predictions
- API responses
- Historical data with market-aware TTL
- In-memory LRU cache replacement
- File-based cache replacement

Features:
- Automatic expiration and cleanup
- Configurable TTL for different data types
- Compression for large datasets
- Performance monitoring and statistics
- Market-aware caching (different TTL for market open/closed)
- LRU-like behavior using Redis sorted sets
"""

import os
import json
import time
import gzip
import pickle
import hashlib
import logging
import threading
from typing import Any, Dict, Optional, List, Union, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import redis

logger = logging.getLogger(__name__)

class RedisUnifiedCacheManager:
    """
    Unified Redis-based cache manager that replaces all local caching systems.
    
    Provides intelligent caching with:
    - Automatic compression for large datasets
    - Configurable TTL for different data types
    - Performance monitoring
    - Automatic cleanup
    - Market-aware TTL (longer TTL when market is closed)
    - LRU-like behavior for historical data
    """
    
    def __init__(self, 
                 redis_url: str = None,
                 enable_compression: bool = True,
                 cleanup_interval_minutes: int = 60,
                 ttl_settings: Dict[str, int] = None,
                 lru_capacity: int = 128):
        """
        Initialize unified Redis cache manager.
        
        Args:
            redis_url: Redis connection URL
            enable_compression: Whether to compress large datasets
            cleanup_interval_minutes: How often to run cleanup
            ttl_settings: TTL settings for different data types
            lru_capacity: Maximum number of items in LRU-like cache
        """
        # Prioritize cloud Redis, only fallback to localhost if explicitly configured
        self.redis_url = redis_url or os.getenv('REDIS_URL')
        if not self.redis_url:
            logger.warning("No REDIS_URL provided, please set REDIS_URL environment variable")
            raise ValueError("REDIS_URL environment variable is required")
        self.enable_compression = enable_compression
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.lru_capacity = lru_capacity
        
        # TTL settings for different data types
        self.ttl_settings = ttl_settings or {
            "stock_data": 300,      # 5 minutes
            "indicators": 600,      # 10 minutes
            "patterns": 1800,       # 30 minutes
            "sector_data": 3600,    # 1 hour
            "ml_predictions": 1800, # 30 minutes
            "api_responses": 300,   # 5 minutes
            "historical_data": 3600, # 1 hour (market-aware)
            "instruments": 86400,   # 24 hours
            "live_data": 60,        # 1 minute
            "enhanced_data": 300    # 5 minutes
        }
        
        # Initialize Redis connection
        self._init_redis_connection()
        
        # Performance monitoring
        self.stats = {
            'redis_hits': 0,
            'redis_misses': 0,
            'compression_savings': 0,
            'errors': 0,
            'lru_evictions': 0
        }
        self.stats_lock = threading.RLock()
        
        # Redis cleanup thread removed - Redis handles expiration automatically
        
        logger.info(f"RedisUnifiedCacheManager initialized: {self.redis_url}")
    
    def _init_redis_connection(self):
        """Initialize Redis connection with error handling."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info(f"✅ Redis connection established: {self.redis_url}")
        except (redis.exceptions.AuthenticationError, redis.exceptions.ConnectionError, Exception) as e:
            logger.error(f"❌ Redis connection failed: {e}. Redis is required for caching.")
            self.redis_available = False
            self.redis_client = None
            raise RuntimeError(f"Redis connection failed: {e}. Redis is required for caching.")
    
    def _is_market_closed(self) -> bool:
        """Check if the market is currently closed (after 3:30 PM IST or before 9:15 AM IST)."""
        now = datetime.now()
        ist_time = now + timedelta(hours=5, minutes=30)  # Convert to IST
        market_open = datetime.strptime("09:15", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()
        
        return ist_time.time() < market_open or ist_time.time() > market_close
    
    def _get_market_aware_ttl(self, data_type: str, base_ttl: int = None) -> int:
        """Get TTL considering market status - longer TTL when market is closed."""
        if base_ttl is None:
            base_ttl = self.ttl_settings.get(data_type, 300)
        
        # For historical data, extend TTL when market is closed
        if data_type == "historical_data" and self._is_market_closed():
            return base_ttl * 4  # 4x longer TTL when market is closed
        
        return base_ttl
    
    def _generate_cache_key(self, data_type: str, *args, **kwargs) -> str:
        """Generate a unique cache key."""
        # Create a string representation of arguments
        key_data = {
            "type": data_type,
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        
        # Convert to JSON string and hash
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return f"cache:{data_type}:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage, with optional compression."""
        try:
            # Handle pandas DataFrames specially
            if isinstance(data, pd.DataFrame):
                # Convert to dict for better compression
                data_dict = {
                    'type': 'dataframe',
                    'data': data.to_dict('records'),
                    'index': data.index.tolist(),
                    'columns': data.columns.tolist(),
                    'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
                }
                serialized = pickle.dumps(data_dict)
            elif isinstance(data, np.ndarray):
                # Handle numpy arrays
                data_dict = {
                    'type': 'numpy_array',
                    'data': data.tolist(),
                    'shape': data.shape,
                    'dtype': str(data.dtype)
                }
                serialized = pickle.dumps(data_dict)
            else:
                # Handle other data types
                serialized = pickle.dumps(data)
            
            # Compress if enabled and data is large enough
            if self.enable_compression and len(serialized) > 1024:  # 1KB threshold
                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized):
                    self._update_stats('compression_savings', len(serialized) - len(compressed))
                    return compressed
            
            return serialized
            
        except Exception as e:
            logger.error(f"Error serializing data: {e}")
            raise
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from storage."""
        try:
            # Try to decompress first
            try:
                decompressed = gzip.decompress(data)
                data = decompressed
            except:
                pass  # Not compressed, use as-is
            
            # Deserialize
            deserialized = pickle.loads(data)
            
            # Reconstruct pandas DataFrame if needed
            if isinstance(deserialized, dict) and deserialized.get('type') == 'dataframe':
                df = pd.DataFrame(deserialized['data'], 
                                index=deserialized['index'], 
                                columns=deserialized['columns'])
                # Restore dtypes
                for col, dtype_str in deserialized['dtypes'].items():
                    if col in df.columns:
                        try:
                            df[col] = df[col].astype(dtype_str)
                        except:
                            pass  # Keep original dtype if conversion fails
                return df
            
            # Reconstruct numpy array if needed
            elif isinstance(deserialized, dict) and deserialized.get('type') == 'numpy_array':
                return np.array(deserialized['data'], dtype=deserialized['dtype'])
            
            return deserialized
            
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            raise
    
    def _update_stats(self, stat_name: str, increment: int = 1):
        """Update statistics thread-safely."""
        with self.stats_lock:
            if stat_name in self.stats:
                self.stats[stat_name] += increment
    
    # ==================== UNIFIED CACHE METHODS ====================
    
    def get(self, data_type: str, *args, **kwargs) -> Optional[Any]:
        """
        Retrieve data from cache.
        
        Args:
            data_type: Type of data
            *args, **kwargs: Arguments to generate cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        if not self.redis_available:
            logger.warning("Redis not available, cannot retrieve from cache")
            return None
        
        cache_key = self._generate_cache_key(data_type, *args, **kwargs)
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data is not None:
                print(f"✅ [REDIS DEBUG] Cache HIT in Redis: {cache_key}")
                self._update_stats('redis_hits')
                
                # Update access time for LRU-like behavior
                if data_type == "historical_data":
                    self._update_lru_access(cache_key)
                
                return self._deserialize_data(cached_data)
            else:
                print(f"❌ [REDIS DEBUG] Cache MISS in Redis: {cache_key}")
                self._update_stats('redis_misses')
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            self._update_stats('errors')
        
        return None
    
    def set(self, data_type: str, data: Any, ttl_seconds: int = None, *args, **kwargs) -> bool:
        """
        Store data in cache.
        
        Args:
            data_type: Type of data
            data: Data to cache
            ttl_seconds: Time to live in seconds (uses market-aware TTL if None)
            *args, **kwargs: Arguments to generate cache key
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not self.redis_available:
            logger.warning("Redis not available, cannot store in cache")
            return False
        
        # Reject None data
        if data is None:
            logger.warning("Cannot cache None data")
            return False
        
        # Use market-aware TTL if not provided
        if ttl_seconds is None:
            ttl_seconds = self._get_market_aware_ttl(data_type)
        
        cache_key = self._generate_cache_key(data_type, *args, **kwargs)
        serialized_data = self._serialize_data(data)
        
        try:
            print(f"💾 [REDIS DEBUG] Storing cache: {cache_key} (TTL: {ttl_seconds}s, size: {len(serialized_data)} bytes)")
            
            # Store the data
            self.redis_client.setex(cache_key, ttl_seconds, serialized_data)
            
            # For historical data, maintain LRU-like behavior
            if data_type == "historical_data":
                self._add_to_lru(cache_key, ttl_seconds)
            
            return True
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            self._update_stats('errors')
            return False
    
    # ==================== LRU-LIKE BEHAVIOR FOR HISTORICAL DATA ====================
    
    def _add_to_lru(self, cache_key: str, ttl_seconds: int):
        """Add key to LRU-like tracking using Redis sorted set."""
        try:
            # Use sorted set with timestamp as score for LRU behavior
            lru_key = "lru:historical_data"
            timestamp = time.time()
            
            # Add to sorted set
            self.redis_client.zadd(lru_key, {cache_key: timestamp})
            
            # Maintain capacity limit
            current_count = self.redis_client.zcard(lru_key)
            if current_count > self.lru_capacity:
                # Remove oldest entries
                removed = self.redis_client.zremrangebyrank(lru_key, 0, current_count - self.lru_capacity - 1)
                self._update_stats('lru_evictions', removed)
                logger.debug(f"LRU evicted {removed} old entries")
                
        except Exception as e:
            logger.warning(f"Error updating LRU: {e}")
    
    def _update_lru_access(self, cache_key: str):
        """Update access time for LRU-like behavior."""
        try:
            lru_key = "lru:historical_data"
            timestamp = time.time()
            
            # Update timestamp (this moves it to the end of the sorted set)
            self.redis_client.zadd(lru_key, {cache_key: timestamp})
            
        except Exception as e:
            logger.warning(f"Error updating LRU access: {e}")
    
    # ==================== SPECIALIZED CACHE METHODS ====================
    
    def cache_historical_data(self, symbol: str, exchange: str, interval: str, 
                            from_date: datetime, to_date: datetime, data: pd.DataFrame,
                            ttl_seconds: int = None) -> bool:
        """Cache historical data with market-aware TTL."""
        if data is None or data.empty:
            return False
        
        # Store with metadata
        metadata = {
            'symbol': symbol,
            'exchange': exchange,
            'interval': interval,
            'from_date': from_date.isoformat(),
            'to_date': to_date.isoformat(),
            'cached_at': datetime.now().isoformat(),
            'market_closed': self._is_market_closed()
        }
        
        # Store data and metadata separately using consistent key generation
        data_success = self.set('historical_data', data, ttl_seconds, symbol, exchange, interval, from_date, to_date)
        metadata_success = self.set('historical_metadata', metadata, ttl_seconds, symbol, exchange, interval, from_date, to_date)
        
        return data_success and metadata_success
    
    def get_cached_historical_data(self, symbol: str, exchange: str, interval: str,
                                 from_date: datetime, to_date: datetime) -> Optional[pd.DataFrame]:
        """Get cached historical data if available."""
        return self.get('historical_data', symbol, exchange, interval, from_date, to_date)
    
    def cache_instruments(self, instruments: List[Dict], ttl_seconds: int = None) -> bool:
        """Cache instruments list."""
        if ttl_seconds is None:
            ttl_seconds = self.ttl_settings.get('instruments', 86400)
        
        return self.set('instruments', instruments, ttl_seconds)
    
    def get_cached_instruments(self) -> Optional[List[Dict]]:
        """Get cached instruments list."""
        return self.get('instruments')
    
    def cache_live_data(self, symbol: str, exchange: str, data: Any, ttl_seconds: int = None) -> bool:
        """Cache live data with short TTL."""
        if ttl_seconds is None:
            ttl_seconds = self.ttl_settings.get('live_data', 60)
        
        return self.set('live_data', data, ttl_seconds, symbol, exchange)
    
    def get_cached_live_data(self, symbol: str, exchange: str) -> Optional[Any]:
        """Get cached live data."""
        return self.get('live_data', symbol, exchange)
    
    # ==================== ENHANCED DATA SERVICE CACHE METHODS ====================
    
    def cache_enhanced_data(self, symbol: str, exchange: str, interval: str, period: int,
                           data: pd.DataFrame, metadata: Dict, ttl_seconds: int = None) -> bool:
        """Cache enhanced data service data."""
        if ttl_seconds is None:
            ttl_seconds = self.ttl_settings.get('enhanced_data', 300)
        
        # Store data and metadata
        data_success = self.set('enhanced_data', data, ttl_seconds, symbol, exchange, interval, period)
        metadata_success = self.set('enhanced_metadata', metadata, ttl_seconds, symbol, exchange, interval, period)
        
        return data_success and metadata_success
    
    def get_cached_enhanced_data(self, symbol: str, exchange: str, interval: str, period: int) -> Optional[tuple]:
        """Get cached enhanced data and metadata."""
        data = self.get('enhanced_data', symbol, exchange, interval, period)
        metadata = self.get('enhanced_metadata', symbol, exchange, interval, period)
        
        if data is not None and metadata is not None:
            return data, metadata
        return None
    
    # ==================== UTILITY METHODS ====================
    
    def delete(self, data_type: str, *args, **kwargs) -> bool:
        """Delete data from cache."""
        if not self.redis_available:
            logger.warning("Redis not available, cannot delete from cache")
            return False
        
        cache_key = self._generate_cache_key(data_type, *args, **kwargs)
        
        try:
            result = self.redis_client.delete(cache_key)
            return result > 0
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            self._update_stats('errors')
            return False
    
    def clear(self, data_type: str = None) -> Dict[str, int]:
        """
        Clear cache data.
        
        Args:
            data_type: Specific data type to clear, or None for all
            
        Returns:
            Dictionary with deletion counts
        """
        if not self.redis_available:
            logger.warning("Redis not available, cannot clear cache")
            return {'deleted': 0}
        
        try:
            if data_type:
                # Clear specific data type
                pattern = f"cache:{data_type}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    logger.info(f"Cleared {deleted} keys for data type: {data_type}")
                    return {'deleted': deleted}
                return {'deleted': 0}
            else:
                # Clear all cache
                pattern = "cache:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    logger.info(f"Cleared all cache: {deleted} keys")
                    return {'deleted': deleted}
                return {'deleted': 0}
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return {'deleted': 0, 'error': str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.stats_lock:
            stats = self.stats.copy()
        
        # Add Redis info if available
        if self.redis_available:
            try:
                info = self.redis_client.info()
                stats.update({
                    'redis_version': info.get('redis_version'),
                    'used_memory_human': info.get('used_memory_human'),
                    'connected_clients': info.get('connected_clients'),
                    'total_commands_processed': info.get('total_commands_processed'),
                    'keyspace_hits': info.get('keyspace_hits'),
                    'keyspace_misses': info.get('keyspace_misses')
                })
                
                # Add LRU stats
                lru_key = "lru:historical_data"
                stats['lru_count'] = self.redis_client.zcard(lru_key)
                stats['lru_capacity'] = self.lru_capacity
                
            except Exception as e:
                logger.warning(f"Could not get Redis info: {e}")
        
        stats['redis_available'] = self.redis_available
        stats['redis_url'] = self.redis_url
        stats['enable_compression'] = self.enable_compression
        stats['market_closed'] = self._is_market_closed()
        
        return stats
    
    def _start_cleanup_thread(self):
        """Start the cleanup thread."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval_minutes * 60)
                    # Redis handles expiration automatically, so this is mainly for monitoring
                    logger.debug("Redis cache cleanup check completed")
                except Exception as e:
                    logger.error(f"Error in cleanup worker: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

# Global unified Redis cache manager instance
_unified_redis_cache_manager = None

def get_unified_redis_cache_manager() -> RedisUnifiedCacheManager:
    """Get the global unified Redis cache manager instance."""
    global _unified_redis_cache_manager
    if _unified_redis_cache_manager is None:
        _unified_redis_cache_manager = RedisUnifiedCacheManager()
    return _unified_redis_cache_manager

def initialize_unified_redis_cache_manager(**kwargs) -> RedisUnifiedCacheManager:
    """Initialize the global unified Redis cache manager with custom settings."""
    global _unified_redis_cache_manager
    _unified_redis_cache_manager = RedisUnifiedCacheManager(**kwargs)
    return _unified_redis_cache_manager

# Convenience functions for common cache operations
def cache_stock_data(symbol: str, exchange: str, interval: str, period: int, data: pd.DataFrame, ttl_seconds: int = 300) -> bool:
    """Cache stock data."""
    return get_unified_redis_cache_manager().set('stock_data', data, ttl_seconds, symbol, exchange, interval, period)

def get_cached_stock_data(symbol: str, exchange: str, interval: str, period: int) -> Optional[pd.DataFrame]:
    """Get cached stock data."""
    return get_unified_redis_cache_manager().get('stock_data', symbol, exchange, interval, period)

def cache_indicators(symbol: str, exchange: str, interval: str, indicators: Dict, ttl_seconds: int = 600) -> bool:
    """Cache technical indicators."""
    return get_unified_redis_cache_manager().set('indicators', indicators, ttl_seconds, symbol, exchange, interval)

def get_cached_indicators(symbol: str, exchange: str, interval: str) -> Optional[Dict]:
    """Get cached technical indicators."""
    return get_unified_redis_cache_manager().get('indicators', symbol, exchange, interval)

def cache_patterns(symbol: str, exchange: str, interval: str, patterns: Dict, ttl_seconds: int = 1800) -> bool:
    """Cache pattern recognition results."""
    return get_unified_redis_cache_manager().set('patterns', patterns, ttl_seconds, symbol, exchange, interval)

def get_cached_patterns(symbol: str, exchange: str, interval: str) -> Optional[Dict]:
    """Get cached pattern recognition results."""
    return get_unified_redis_cache_manager().get('patterns', symbol, exchange, interval)

def cache_sector_data(sector: str, period: int, data: Dict, ttl_seconds: int = 3600) -> bool:
    """Cache sector analysis data."""
    return get_unified_redis_cache_manager().set('sector_data', data, ttl_seconds, sector, period)

def get_cached_sector_data(sector: str, period: int) -> Optional[Dict]:
    """Get cached sector analysis data."""
    return get_unified_redis_cache_manager().get('sector_data', sector, period)

def clear_stock_cache(symbol: str = None, exchange: str = None) -> Dict[str, int]:
    """Clear stock data cache."""
    if symbol and exchange:
        # Clear specific symbol
        return get_unified_redis_cache_manager().clear('stock_data')
    else:
        # Clear all stock data
        return get_unified_redis_cache_manager().clear('stock_data')
