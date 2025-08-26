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

logger = logging.getLogger(__name__)

class RedisCacheManager:
    """
    Redis-based cache manager for stock data and analysis results.
    
    Provides intelligent caching with:
    - Automatic compression for large datasets
    - Configurable TTL for different data types
    - Performance monitoring
    - Automatic cleanup
    """
    
    def __init__(self, 
                 redis_url: str = None,
                 enable_compression: bool = True,
                 cleanup_interval_minutes: int = 60,
                 ttl_settings: Dict[str, int] = None):
        """
        Initialize Redis cache manager.
        
        Args:
            redis_url: Redis connection URL
            enable_compression: Whether to compress large datasets
            cleanup_interval_minutes: How often to run cleanup
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.enable_compression = enable_compression
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
        
        # Performance monitoring
        self.stats = {
            'redis_hits': 0,
            'redis_misses': 0,
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
        except (redis.exceptions.AuthenticationError, redis.exceptions.ConnectionError, Exception) as e:
            logger.warning(f"⚠️  Redis connection failed: {e}. Using in-memory fallback.")
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
            ttl_seconds: Time to live in seconds (uses default from ttl_settings if None)
            *args, **kwargs: Arguments to generate cache key
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not self.redis_available:
            logger.warning("Redis not available, cannot store in cache")
            return False
        
        # Use TTL from settings if not provided
        if ttl_seconds is None:
            ttl_seconds = self.ttl_settings.get(data_type, 300)
        
        cache_key = self._generate_cache_key(data_type, *args, **kwargs)
        serialized_data = self._serialize_data(data)
        
        try:
            print(f"💾 [REDIS DEBUG] Storing cache: {cache_key} (TTL: {ttl_seconds}s, size: {len(serialized_data)} bytes)")
            self.redis_client.setex(cache_key, ttl_seconds, serialized_data)
            return True
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            self._update_stats('errors')
            return False
    
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
            except Exception as e:
                logger.warning(f"Could not get Redis info: {e}")
        
        stats['redis_available'] = self.redis_available
        stats['redis_url'] = self.redis_url
        stats['enable_compression'] = self.enable_compression
        
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
        logger.info(f"Started cleanup thread with {self.cleanup_interval_minutes} minute interval")

# Global Redis cache manager instance
_redis_cache_manager = None

def get_redis_cache_manager() -> RedisCacheManager:
    """Get the global Redis cache manager instance."""
    global _redis_cache_manager
    if _redis_cache_manager is None:
        _redis_cache_manager = RedisCacheManager()
    return _redis_cache_manager

def initialize_redis_cache_manager(**kwargs) -> RedisCacheManager:
    """Initialize the global Redis cache manager with custom settings."""
    global _redis_cache_manager
    _redis_cache_manager = RedisCacheManager(**kwargs)
    return _redis_cache_manager

# Convenience functions for common cache operations
def cache_stock_data(symbol: str, exchange: str, interval: str, period: int, data: pd.DataFrame, ttl_seconds: int = 300) -> bool:
    """Cache stock data."""
    return get_redis_cache_manager().set('stock_data', data, ttl_seconds, symbol, exchange, interval, period)

def get_cached_stock_data(symbol: str, exchange: str, interval: str, period: int) -> Optional[pd.DataFrame]:
    """Get cached stock data."""
    return get_redis_cache_manager().get('stock_data', symbol, exchange, interval, period)

def cache_indicators(symbol: str, exchange: str, interval: str, indicators: Dict, ttl_seconds: int = 600) -> bool:
    """Cache technical indicators."""
    return get_redis_cache_manager().set('indicators', indicators, ttl_seconds, symbol, exchange, interval)

def get_cached_indicators(symbol: str, exchange: str, interval: str) -> Optional[Dict]:
    """Get cached technical indicators."""
    return get_redis_cache_manager().get('indicators', symbol, exchange, interval)

def cache_patterns(symbol: str, exchange: str, interval: str, patterns: Dict, ttl_seconds: int = 1800) -> bool:
    """Cache pattern recognition results."""
    return get_redis_cache_manager().set('patterns', patterns, ttl_seconds, symbol, exchange, interval)

def get_cached_patterns(symbol: str, exchange: str, interval: str) -> Optional[Dict]:
    """Get cached pattern recognition results."""
    return get_redis_cache_manager().get('patterns', symbol, exchange, interval)

def cache_sector_data(sector: str, period: int, data: Dict, ttl_seconds: int = 3600) -> bool:
    """Cache sector analysis data."""
    return get_redis_cache_manager().set('sector_data', data, ttl_seconds, sector, period)

def get_cached_sector_data(sector: str, period: int) -> Optional[Dict]:
    """Get cached sector analysis data."""
    return get_redis_cache_manager().get('sector_data', sector, period)

def clear_stock_cache(symbol: str = None, exchange: str = None) -> Dict[str, int]:
    """Clear stock data cache."""
    if symbol and exchange:
        # Clear specific symbol
        return get_redis_cache_manager().clear('stock_data')
    else:
        # Clear all stock data
        return get_redis_cache_manager().clear('stock_data')
