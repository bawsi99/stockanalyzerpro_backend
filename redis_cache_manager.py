"""
Redis Cache Manager for Stock Data and Analysis

This module provides a comprehensive Redis-based caching system for:
- Stock data (historical prices, indicators)
- Technical analysis results
- Pattern recognition results
- Sector analysis data
- ML model predictions
- API responses

Features:
- Automatic expiration and cleanup
- Configurable TTL for different data types
- Compression for large datasets
- Fallback to local cache if Redis unavailable
- Performance monitoring and statistics
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
from collections import OrderedDict

logger = logging.getLogger(__name__)

class RedisCacheManager:
    """
    Redis-based cache manager for stock data and analysis results.
    
    Provides intelligent caching with:
    - Automatic compression for large datasets
    - Configurable TTL for different data types
    - Fallback to local cache if Redis unavailable
    - Performance monitoring
    - Automatic cleanup
    """
    
    def __init__(self, 
                 redis_url: str = None,
                 enable_compression: bool = True,
                 enable_local_fallback: bool = True,
                 local_cache_size: int = 1000,
                 cleanup_interval_minutes: int = 60,
                 ttl_settings: Dict[str, int] = None):
        """
        Initialize Redis cache manager.
        
        Args:
            redis_url: Redis connection URL
            enable_compression: Whether to compress large datasets
            enable_local_fallback: Whether to use local cache as fallback
            local_cache_size: Size of local fallback cache
            cleanup_interval_minutes: How often to run cleanup
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.enable_compression = enable_compression
        self.enable_local_fallback = enable_local_fallback
        self.cleanup_interval_minutes = cleanup_interval_minutes
        
        # TTL settings for different data types
        self.ttl_settings = ttl_settings or {
            "stock_data": 300,      # 5 minutes
            "indicators": 600,      # 10 minutes
            "patterns": 1800,       # 30 minutes
            "sector_data": 3600,    # 1 hour
            "ml_predictions": 1800, # 30 minutes
            "api_responses": 300    # 5 minutes
        }
        
        # Initialize Redis connection
        self._init_redis_connection()
        
        # Local fallback cache
        if self.enable_local_fallback:
            self.local_cache = OrderedDict()
            self.local_cache_size = local_cache_size
            self.local_cache_lock = threading.RLock()
        
        # Performance monitoring
        self.stats = {
            'redis_hits': 0,
            'redis_misses': 0,
            'local_hits': 0,
            'local_misses': 0,
            'compression_savings': 0,
            'errors': 0
        }
        self.stats_lock = threading.RLock()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info(f"RedisCacheManager initialized: {self.redis_url}")
    
    def _init_redis_connection(self):
        """Initialize Redis connection with error handling."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info(f"✅ Redis connection established: {self.redis_url}")
        except Exception as e:
            logger.warning(f"❌ Failed to connect to Redis at {self.redis_url}: {e}")
            self.redis_available = False
            self.redis_client = None
    
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
                # Not compressed, use as-is
                pass
            
            # Deserialize
            deserialized = pickle.loads(data)
            
            # Handle special types
            if isinstance(deserialized, dict):
                if deserialized.get('type') == 'dataframe':
                    # Reconstruct pandas DataFrame
                    df = pd.DataFrame(deserialized['data'])
                    df.index = deserialized['index']
                    df.columns = deserialized['columns']
                    return df
                elif deserialized.get('type') == 'numpy_array':
                    # Reconstruct numpy array
                    return np.array(deserialized['data'], dtype=deserialized['dtype'])
            
            return deserialized
            
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            raise
    
    def _update_stats(self, stat_name: str, value: int = 1):
        """Update statistics."""
        with self.stats_lock:
            if stat_name in self.stats:
                self.stats[stat_name] += value
    
    def get(self, data_type: str, *args, **kwargs) -> Optional[Any]:
        """
        Get data from cache.
        
        Args:
            data_type: Type of data (e.g., 'stock_data', 'indicators', 'patterns')
            *args, **kwargs: Arguments to generate cache key
            
        Returns:
            Cached data or None if not found
        """
        cache_key = self._generate_cache_key(data_type, *args, **kwargs)
        
        # Try Redis first
        if self.redis_available:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    print(f"🎯 [REDIS DEBUG] Cache HIT in Redis: {cache_key}")
                    self._update_stats('redis_hits')
                    return self._deserialize_data(cached_data)
                else:
                    print(f"❌ [REDIS DEBUG] Cache MISS in Redis: {cache_key}")
                    self._update_stats('redis_misses')
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
                self._update_stats('errors')
        
        # Try local fallback cache
        if self.enable_local_fallback:
            with self.local_cache_lock:
                if cache_key in self.local_cache:
                    item = self.local_cache[cache_key]
                    if not item['expired']():
                        # Move to end (LRU)
                        self.local_cache.move_to_end(cache_key)
                        print(f"📱 [REDIS DEBUG] Cache HIT in local cache: {cache_key}")
                        self._update_stats('local_hits')
                        return item['data']
                    else:
                        # Remove expired item
                        print(f"🗑️  [REDIS DEBUG] Removing expired item from local cache: {cache_key}")
                        del self.local_cache[cache_key]
                
                print(f"❌ [REDIS DEBUG] Cache MISS in local cache: {cache_key}")
                self._update_stats('local_misses')
        
        return None
    
    def set(self, data_type: str, data: Any, ttl_seconds: int = None, *args, **kwargs) -> bool:
        """
        Store data in cache.
        
        Args:
            data_type: Type of data
            data: Data to cache
            ttl_seconds: Time to live in seconds (uses default from ttl_settings if None)
            *args, **kwargs: Arguments to generate cache key
            
        Returns:
            True if successfully cached, False otherwise
        """
        # Use TTL from settings if not provided
        if ttl_seconds is None:
            ttl_seconds = self.ttl_settings.get(data_type, 300)
        cache_key = self._generate_cache_key(data_type, *args, **kwargs)
        serialized_data = self._serialize_data(data)
        
        success = False
        
        # Try Redis first
        if self.redis_available:
            try:
                print(f"💾 [REDIS DEBUG] Storing cache: {cache_key} (TTL: {ttl_seconds}s, size: {len(serialized_data)} bytes)")
                self.redis_client.setex(cache_key, ttl_seconds, serialized_data)
                success = True
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
                self._update_stats('errors')
        
        # Always store in local fallback cache
        if self.enable_local_fallback:
            with self.local_cache_lock:
                # Remove if already exists
                if cache_key in self.local_cache:
                    del self.local_cache[cache_key]
                
                # Add new item
                self.local_cache[cache_key] = {
                    'data': data,
                    'created_at': time.time(),
                    'ttl': ttl_seconds,
                    'expired': lambda: time.time() - self.local_cache[cache_key]['created_at'] > ttl_seconds
                }
                
                print(f"📱 [REDIS DEBUG] Stored in local cache: {cache_key}")
                
                # Maintain cache size
                if len(self.local_cache) > self.local_cache_size:
                    removed_key = self.local_cache.popitem(last=False)[0]
                    print(f"🗑️  [REDIS DEBUG] Removed from local cache (size limit): {removed_key}")
                
                success = True
        
        return success
    
    def delete(self, data_type: str, *args, **kwargs) -> bool:
        """Delete data from cache."""
        cache_key = self._generate_cache_key(data_type, *args, **kwargs)
        success = False
        
        # Delete from Redis
        if self.redis_available:
            try:
                result = self.redis_client.delete(cache_key)
                success = result > 0
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
                self._update_stats('errors')
        
        # Delete from local cache
        if self.enable_local_fallback:
            with self.local_cache_lock:
                if cache_key in self.local_cache:
                    del self.local_cache[cache_key]
                    success = True
        
        return success
    
    def clear(self, data_type: str = None) -> Dict[str, int]:
        """Clear cache entries."""
        deleted_counts = {'redis': 0, 'local': 0}
        
        # Clear from Redis
        if self.redis_available:
            try:
                if data_type:
                    pattern = f"cache:{data_type}:*"
                else:
                    pattern = "cache:*"
                
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted_counts['redis'] = self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
                self._update_stats('errors')
        
        # Clear from local cache
        if self.enable_local_fallback:
            with self.local_cache_lock:
                if data_type:
                    # Delete specific type
                    keys_to_delete = [
                        key for key in self.local_cache.keys()
                        if key.startswith(f"cache:{data_type}:")
                    ]
                    for key in keys_to_delete:
                        del self.local_cache[key]
                    deleted_counts['local'] = len(keys_to_delete)
                else:
                    # Clear all
                    deleted_counts['local'] = len(self.local_cache)
                    self.local_cache.clear()
        
        return deleted_counts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.stats_lock:
            stats = self.stats.copy()
        
        # Add current cache sizes
        stats['redis_available'] = self.redis_available
        stats['local_cache_size'] = len(self.local_cache) if self.enable_local_fallback else 0
        stats['local_cache_max_size'] = self.local_cache_size if self.enable_local_fallback else 0
        
        # Calculate hit rates
        total_redis_requests = stats['redis_hits'] + stats['redis_misses']
        total_local_requests = stats['local_hits'] + stats['local_misses']
        
        stats['redis_hit_rate'] = (stats['redis_hits'] / total_redis_requests * 100) if total_redis_requests > 0 else 0
        stats['local_hit_rate'] = (stats['local_hits'] / total_local_requests * 100) if total_local_requests > 0 else 0
        
        return stats
    
    def _cleanup_expired_local_cache(self):
        """Remove expired items from local cache."""
        if not self.enable_local_fallback:
            return
        
        with self.local_cache_lock:
            expired_keys = [
                key for key, item in self.local_cache.items()
                if item['expired']()
            ]
            
            for key in expired_keys:
                del self.local_cache[key]
                print(f"🗑️  [REDIS DEBUG] Removed expired from local cache: {key}")
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired local cache items")
                print(f"🧹 [REDIS DEBUG] Local cache cleanup: {len(expired_keys)} expired items removed")
            else:
                print(f"ℹ️  [REDIS DEBUG] Local cache cleanup: no expired items found")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval_minutes * 60)
                    self._cleanup_expired_local_cache()
                except Exception as e:
                    logger.error(f"Error in cache cleanup worker: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("Cache cleanup thread started")

# Cache decorator for functions
def redis_cached(data_type: str, ttl_seconds: int = 300):
    """
    Decorator for caching function results in Redis.
    
    Args:
        data_type: Type of data being cached
        ttl_seconds: Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Get cache manager instance
            cache_manager = get_redis_cache_manager()
            
            # Try to get from cache
            cached_result = cache_manager.get(data_type, *args, **kwargs)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Calculate result
            logger.debug(f"Cache miss for {func.__name__}, calculating...")
            result = func(*args, **kwargs)
            
            # Cache the result
            cache_manager.set(data_type, result, ttl_seconds, *args, **kwargs)
            return result
        
        return wrapper
    return decorator

# Global cache manager instance
_redis_cache_manager: Optional[RedisCacheManager] = None

def get_redis_cache_manager() -> RedisCacheManager:
    """Get or create the global Redis cache manager instance."""
    global _redis_cache_manager
    if _redis_cache_manager is None:
        _redis_cache_manager = RedisCacheManager()
    return _redis_cache_manager

def initialize_redis_cache_manager(**kwargs) -> RedisCacheManager:
    """Initialize the global Redis cache manager with custom settings."""
    global _redis_cache_manager
    _redis_cache_manager = RedisCacheManager(**kwargs)
    return _redis_cache_manager

# Utility functions for common cache operations
def cache_stock_data(symbol: str, exchange: str, interval: str, period: int, data: pd.DataFrame, ttl_seconds: int = 300) -> bool:
    """Cache stock data."""
    cache_manager = get_redis_cache_manager()
    return cache_manager.set('stock_data', data, ttl_seconds, symbol, exchange, interval, period)

def get_cached_stock_data(symbol: str, exchange: str, interval: str, period: int) -> Optional[pd.DataFrame]:
    """Get cached stock data."""
    cache_manager = get_redis_cache_manager()
    return cache_manager.get('stock_data', symbol, exchange, interval, period)

def cache_indicators(symbol: str, exchange: str, interval: str, indicators: Dict, ttl_seconds: int = 600) -> bool:
    """Cache technical indicators."""
    cache_manager = get_redis_cache_manager()
    return cache_manager.set('indicators', indicators, ttl_seconds, symbol, exchange, interval)

def get_cached_indicators(symbol: str, exchange: str, interval: str) -> Optional[Dict]:
    """Get cached technical indicators."""
    cache_manager = get_redis_cache_manager()
    return cache_manager.get('indicators', symbol, exchange, interval)

def cache_patterns(symbol: str, exchange: str, interval: str, patterns: Dict, ttl_seconds: int = 1800) -> bool:
    """Cache pattern recognition results."""
    cache_manager = get_redis_cache_manager()
    return cache_manager.set('patterns', patterns, ttl_seconds, symbol, exchange, interval)

def get_cached_patterns(symbol: str, exchange: str, interval: str) -> Optional[Dict]:
    """Get cached pattern recognition results."""
    cache_manager = get_redis_cache_manager()
    return cache_manager.get('patterns', symbol, exchange, interval)

def cache_sector_data(sector: str, period: int, data: Dict, ttl_seconds: int = 3600) -> bool:
    """Cache sector analysis data."""
    cache_manager = get_redis_cache_manager()
    return cache_manager.set('sector_data', data, ttl_seconds, sector, period)

def get_cached_sector_data(sector: str, period: int) -> Optional[Dict]:
    """Get cached sector analysis data."""
    cache_manager = get_redis_cache_manager()
    return cache_manager.get('sector_data', sector, period)

def clear_stock_cache(symbol: str = None, exchange: str = None) -> Dict[str, int]:
    """Clear stock-related cache entries."""
    cache_manager = get_redis_cache_manager()
    if symbol:
        # Clear specific symbol
        cache_manager.clear('stock_data')
        cache_manager.clear('indicators')
        cache_manager.clear('patterns')
    else:
        # Clear all stock data
        cache_manager.clear('stock_data')
        cache_manager.clear('indicators')
        cache_manager.clear('patterns')
    return cache_manager.get_stats()
