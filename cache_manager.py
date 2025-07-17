"""
Cache manager for optimizing performance of expensive calculations.
Provides intelligent caching with TTL, size limits, and automatic cleanup.
"""

import time
import hashlib
import json
import threading
from typing import Any, Dict, Optional, Callable
from collections import OrderedDict
import logging
from config import Config

logger = logging.getLogger(__name__)

class CacheItem:
    """Represents a cached item with metadata."""
    
    def __init__(self, value: Any, ttl: int = 300):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = time.time()
    
    def is_expired(self) -> bool:
        """Check if the cache item has expired."""
        return time.time() - self.created_at > self.ttl
    
    def access(self):
        """Mark the item as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def get_age(self) -> float:
        """Get the age of the cache item in seconds."""
        return time.time() - self.created_at

class CacheManager:
    """Intelligent cache manager with TTL, size limits, and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheItem] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from function arguments."""
        # Create a string representation of arguments
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        
        # Generate hash for consistent key length
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        with self.lock:
            if key in self.cache:
                item = self.cache[key]
                
                if item.is_expired():
                    # Remove expired item
                    del self.cache[key]
                    self.stats["expirations"] += 1
                    self.stats["misses"] += 1
                    return None
                
                # Mark as accessed and move to end (LRU)
                item.access()
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                return item.value
            
            self.stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache."""
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]
            
            # Check if we need to evict items
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new item
            cache_ttl = ttl if ttl is not None else self.default_ttl
            self.cache[key] = CacheItem(value, cache_ttl)
    
    def _evict_lru(self) -> None:
        """Evict least recently used items."""
        while len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats["evictions"] += 1
    
    def _cleanup_worker(self) -> None:
        """Background thread to clean up expired items."""
        while True:
            try:
                time.sleep(60)  # Run every minute
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cache cleanup worker: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired items from cache."""
        with self.lock:
            expired_keys = [
                key for key, item in self.cache.items()
                if item.is_expired()
            ]
            
            for key in expired_keys:
                del self.cache[key]
                self.stats["expirations"] += 1
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                **self.stats,
                "hit_rate": hit_rate,
                "size": len(self.cache),
                "max_size": self.max_size,
                "utilization": (len(self.cache) / self.max_size * 100) if self.max_size > 0 else 0
            }
    
    def get_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        with self.lock:
            items_info = []
            for key, item in self.cache.items():
                items_info.append({
                    "key": key[:16] + "...",  # Truncate long keys
                    "age": item.get_age(),
                    "ttl": item.ttl,
                    "access_count": item.access_count,
                    "last_accessed": item.last_accessed
                })
            
            return {
                "stats": self.get_stats(),
                "items": items_info[:10]  # Show first 10 items
            }

# Global cache instance
cache_manager = CacheManager(
    max_size=Config.get("cache", "max_size", 1000),
    default_ttl=Config.get("cache", "default_ttl", 300)
)

def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds (uses default if None)
        key_prefix: Prefix for cache key
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Skip caching if disabled
            if not Config.get("cache", "enabled", True):
                return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{cache_manager._generate_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Calculate result
            logger.debug(f"Cache miss for {func.__name__}, calculating...")
            result = func(*args, **kwargs)
            
            # Cache the result
            cache_ttl = ttl if ttl is not None else Config.get("cache", f"{func.__name__}_ttl", None)
            if cache_ttl is None:
                cache_ttl = Config.get("cache", "default_ttl", 300)
            
            cache_manager.set(cache_key, result, cache_ttl)
            return result
        
        return wrapper
    return decorator

def clear_cache() -> None:
    """Clear all cached data."""
    cache_manager.clear()
    logger.info("Cache cleared")

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return cache_manager.get_stats()

def get_cache_info() -> Dict[str, Any]:
    """Get detailed cache information."""
    return cache_manager.get_info()

# Performance monitoring
class PerformanceMonitor:
    """Monitor performance of expensive operations."""
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.RLock()
    
    def start_timer(self, operation: str) -> float:
        """Start timing an operation."""
        return time.time()
    
    def end_timer(self, operation: str, start_time: float) -> float:
        """End timing an operation and record metrics."""
        duration = time.time() - start_time
        
        with self.lock:
            if operation not in self.metrics:
                self.metrics[operation] = {
                    "count": 0,
                    "total_time": 0,
                    "avg_time": 0,
                    "min_time": float('inf'),
                    "max_time": 0
                }
            
            metric = self.metrics[operation]
            metric["count"] += 1
            metric["total_time"] += duration
            metric["avg_time"] = metric["total_time"] / metric["count"]
            metric["min_time"] = min(metric["min_time"], duration)
            metric["max_time"] = max(metric["max_time"], duration)
        
        logger.debug(f"{operation} completed in {duration:.3f}s")
        return duration
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        with self.lock:
            return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        with self.lock:
            self.metrics.clear()

# Global performance monitor
performance_monitor = PerformanceMonitor()

def monitor_performance(operation: str):
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = performance_monitor.start_timer(operation)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                performance_monitor.end_timer(operation, start_time)
        return wrapper
    return decorator 