import os
import base64
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import io
from PIL import Image
import redis
from pathlib import Path

logger = logging.getLogger(__name__)

class RedisImageManager:
    """
    Manages chart image storage in Redis with automatic cleanup and expiration.
    
    Features:
    - Store images as base64 encoded strings in Redis
    - Automatic cleanup of old images
    - Configurable retention policies
    - Memory-efficient image handling
    - Support for both local and cloud Redis instances
    """
    
    def __init__(self, 
                 redis_url: str = None,
                 max_age_hours: int = 24,
                 max_total_size_mb: int = 1000,
                 cleanup_interval_minutes: int = 60,
                 enable_cleanup: bool = True,
                 image_quality: int = 85,
                 image_format: str = 'PNG'):
        """
        Initialize Redis image manager.
        
        Args:
            redis_url: Redis connection URL (defaults to REDIS_URL env var)
            max_age_hours: Maximum age of images before cleanup (hours)
            max_total_size_mb: Maximum total size of images in Redis (MB)
            cleanup_interval_minutes: How often to run cleanup (minutes)
            enable_cleanup: Whether to enable automatic cleanup
            image_quality: JPEG quality (1-100) for compression
            image_format: Image format ('PNG', 'JPEG', 'WEBP')
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.max_age_hours = max_age_hours
        self.max_total_size_mb = max_total_size_mb
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.enable_cleanup = enable_cleanup
        self.image_quality = image_quality
        self.image_format = image_format.upper()
        
        # Initialize Redis connection
        self._init_redis_connection()
        
        # Start cleanup thread only if enabled
        if self.enable_cleanup:
            self._start_cleanup_thread()
            logger.info(f"RedisImageManager initialized: max_age={max_age_hours}h, max_size={max_total_size_mb}MB, cleanup=enabled")
        else:
            logger.info(f"RedisImageManager initialized: max_age={max_age_hours}h, max_size={max_total_size_mb}MB, cleanup=disabled")
    
    def _init_redis_connection(self):
        """Initialize Redis connection with error handling."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Redis connection established: {self.redis_url}")
        except (redis.exceptions.AuthenticationError, redis.exceptions.ConnectionError, Exception) as e:
            logger.error(f"❌ Redis connection failed: {e}. Redis is required for chart storage.")
            self.redis_client = None
            self.redis_available = False
            raise RuntimeError(f"Redis connection failed: {e}. Redis is required for chart storage.")
    
    def _generate_image_key(self, symbol: str, interval: str, chart_type: str) -> str:
        """Generate a unique Redis key for an image."""
        timestamp = int(time.time())
        return f"chart:{symbol}:{interval}:{chart_type}:{timestamp}"
    
    def _generate_metadata_key(self, image_key: str) -> str:
        """Generate metadata key for an image."""
        return f"metadata:{image_key}"
    
    def store_image(self, 
                   image_data: Union[bytes, str, Image.Image], 
                   symbol: str, 
                   interval: str, 
                   chart_type: str,
                   metadata: Dict[str, Any] = None) -> str:
        """
        Store an image in Redis.
        
        Args:
            image_data: Image data (bytes, base64 string, or PIL Image)
            symbol: Stock symbol
            interval: Time interval
            chart_type: Type of chart
            metadata: Additional metadata to store
            
        Returns:
            Redis key of the stored image
        """
        try:
            # Generate unique key
            image_key = self._generate_image_key(symbol, interval, chart_type)
            metadata_key = self._generate_metadata_key(image_key)
            
            # Convert image data to base64
            if isinstance(image_data, Image.Image):
                # Convert PIL Image to base64
                img_buffer = io.BytesIO()
                if self.image_format == 'JPEG':
                    image_data.save(img_buffer, format='JPEG', quality=self.image_quality, optimize=True)
                elif self.image_format == 'WEBP':
                    image_data.save(img_buffer, format='WEBP', quality=self.image_quality)
                else:
                    image_data.save(img_buffer, format='PNG', optimize=True)
                img_buffer.seek(0)
                base64_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            elif isinstance(image_data, bytes):
                # Convert bytes to base64
                base64_data = base64.b64encode(image_data).decode('utf-8')
            elif isinstance(image_data, str):
                # Assume it's already base64
                if image_data.startswith('data:image'):
                    # Remove data URL prefix
                    base64_data = image_data.split(',')[1]
                else:
                    base64_data = image_data
            else:
                raise ValueError(f"Unsupported image data type: {type(image_data)}")
            
            # Store image data
            print(f"🖼️  [REDIS DEBUG] Storing image: {image_key} ({len(base64_data)} chars)")
            self.redis_client.set(image_key, base64_data)
            
            # Store metadata
            metadata = metadata or {}
            metadata.update({
                'symbol': symbol,
                'interval': interval,
                'chart_type': chart_type,
                'created_at': time.time(),
                'size_bytes': len(base64_data.encode('utf-8')),
                'format': self.image_format,
                'quality': self.image_quality
            })
            print(f"📋 [REDIS DEBUG] Storing metadata: {metadata_key}")
            self.redis_client.hmset(metadata_key, metadata)
            
            # Set expiration (max_age_hours)
            expiration_seconds = int(self.max_age_hours * 3600)
            print(f"⏰ [REDIS DEBUG] Setting expiration: {expiration_seconds}s for {image_key}")
            self.redis_client.expire(image_key, expiration_seconds)
            self.redis_client.expire(metadata_key, expiration_seconds)
            
            logger.debug(f"Stored image: {image_key} ({len(base64_data)} chars)")
            return image_key
            
        except Exception as e:
            logger.error(f"Error storing image for {symbol}_{interval}_{chart_type}: {e}")
            raise
    
    def get_image(self, image_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an image from Redis.
        
        Args:
            image_key: Redis key of the image
            
        Returns:
            Dictionary with image data and metadata, or None if not found
        """
        try:
            # Get image data
            image_data = self.redis_client.get(image_key)
            if not image_data:
                return None
            
            # Get metadata
            metadata_key = self._generate_metadata_key(image_key)
            metadata = self.redis_client.hgetall(metadata_key)
            
            # Convert metadata values
            metadata = {k.decode('utf-8'): v.decode('utf-8') if isinstance(v, bytes) else v 
                       for k, v in metadata.items()}
            
            return {
                'data': f"data:image/{metadata.get('format', 'png').lower()};base64,{image_data.decode('utf-8')}",
                'metadata': metadata,
                'key': image_key
            }
            
        except Exception as e:
            logger.error(f"Error retrieving image {image_key}: {e}")
            return None
    
    def get_images_by_symbol(self, symbol: str, interval: str = None) -> List[Dict[str, Any]]:
        """
        Get all images for a specific symbol.
        
        Args:
            symbol: Stock symbol
            interval: Optional time interval filter
            
        Returns:
            List of image dictionaries
        """
        try:
            pattern = f"chart:{symbol}:*"
            if interval:
                pattern = f"chart:{symbol}:{interval}:*"
            
            keys = self.redis_client.keys(pattern)
            images = []
            
            for key in keys:
                image_data = self.get_image(key.decode('utf-8'))
                if image_data:
                    images.append(image_data)
            
            return images
            
        except Exception as e:
            logger.error(f"Error retrieving images for {symbol}: {e}")
            return []
    
    def delete_image(self, image_key: str) -> bool:
        """
        Delete an image from Redis.
        
        Args:
            image_key: Redis key of the image
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            metadata_key = self._generate_metadata_key(image_key)
            
            # Delete both image and metadata
            result1 = self.redis_client.delete(image_key)
            result2 = self.redis_client.delete(metadata_key)
            
            success = result1 > 0 or result2 > 0
            if success:
                logger.debug(f"Deleted image: {image_key}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting image {image_key}: {e}")
            return False
    
    def cleanup_old_images(self) -> Dict[str, int]:
        """
        Clean up old images based on age and size limits.
        
        Returns:
            Dict with cleanup statistics
        """
        print(f"🧹 [REDIS DEBUG] Starting image cleanup...")
        stats = {
            'images_removed': 0,
            'bytes_freed': 0,
            'errors': 0
        }
        
        if not self.enable_cleanup:
            logger.info("Image cleanup is disabled")
            return stats
        
        try:
            current_time = time.time()
            max_age_seconds = self.max_age_hours * 3600
            
            # Get all image keys
            image_keys = self.redis_client.keys("chart:*")
            
            for key in image_keys:
                key_str = key.decode('utf-8')
                metadata_key = self._generate_metadata_key(key_str)
                
                try:
                    # Get metadata
                    metadata = self.redis_client.hgetall(metadata_key)
                    if not metadata:
                        # No metadata, delete the image
                        self.delete_image(key_str)
                        stats['images_removed'] += 1
                        continue
                    
                    # Check age
                    created_at = float(metadata.get(b'created_at', current_time))
                    if isinstance(created_at, bytes):
                        created_at = float(created_at.decode('utf-8'))
                    
                    age_seconds = current_time - created_at
                    if age_seconds > max_age_seconds:
                        # Get size before deletion
                        size_bytes = int(metadata.get(b'size_bytes', 0))
                        if isinstance(size_bytes, bytes):
                            size_bytes = int(size_bytes.decode('utf-8'))
                        
                        print(f"🗑️  [REDIS DEBUG] Deleting old image: {key_str} (age: {age_seconds/3600:.1f}h)")
                        if self.delete_image(key_str):
                            stats['images_removed'] += 1
                            stats['bytes_freed'] += size_bytes
                
                except Exception as e:
                    stats['errors'] += 1
                    logger.warning(f"Error processing image {key_str}: {e}")
            
            # Clean up by size if needed
            total_size = self._get_total_size()
            if total_size > (self.max_total_size_mb * 1024 * 1024):
                self._cleanup_by_size(stats)
            
            if stats['images_removed'] > 0:
                logger.info(f"Image cleanup completed: {stats}")
                print(f"✅ [REDIS DEBUG] Image cleanup completed: {stats['images_removed']} images removed, {stats['bytes_freed']} bytes freed")
            else:
                print(f"ℹ️  [REDIS DEBUG] No images cleaned up (all within age limit)")
            
        except Exception as e:
            logger.error(f"Error during image cleanup: {e}")
            stats['errors'] += 1
        
        return stats
    
    def _cleanup_by_size(self, stats: Dict[str, int]) -> None:
        """Clean up images by removing oldest when size limit is exceeded."""
        try:
            # Get all images with their creation times
            image_keys = self.redis_client.keys("chart:*")
            images_with_time = []
            
            for key in image_keys:
                key_str = key.decode('utf-8')
                metadata_key = self._generate_metadata_key(key_str)
                
                try:
                    metadata = self.redis_client.hgetall(metadata_key)
                    if metadata:
                        created_at = float(metadata.get(b'created_at', 0))
                        if isinstance(created_at, bytes):
                            created_at = float(created_at.decode('utf-8'))
                        
                        size_bytes = int(metadata.get(b'size_bytes', 0))
                        if isinstance(size_bytes, bytes):
                            size_bytes = int(size_bytes.decode('utf-8'))
                        
                        images_with_time.append((key_str, created_at, size_bytes))
                except Exception:
                    continue
            
            # Sort by creation time (oldest first)
            images_with_time.sort(key=lambda x: x[1])
            
            # Remove oldest images until under size limit
            target_size = self.max_total_size_mb * 1024 * 1024
            current_size = sum(size for _, _, size in images_with_time)
            
            for key_str, _, size in images_with_time:
                if current_size <= target_size:
                    break
                
                if self.delete_image(key_str):
                    current_size -= size
                    stats['images_removed'] += 1
                    stats['bytes_freed'] += size
                    logger.debug(f"Removed image for size cleanup: {key_str}")
        
        except Exception as e:
            logger.error(f"Error during size-based cleanup: {e}")
    
    def _get_total_size(self) -> int:
        """Get total size of all images in Redis."""
        try:
            image_keys = self.redis_client.keys("chart:*")
            total_size = 0
            
            for key in image_keys:
                key_str = key.decode('utf-8')
                metadata_key = self._generate_metadata_key(key_str)
                
                try:
                    metadata = self.redis_client.hgetall(metadata_key)
                    if metadata and b'size_bytes' in metadata:
                        size_bytes = int(metadata[b'size_bytes'])
                        total_size += size_bytes
                except Exception:
                    continue
            
            return total_size
            
        except Exception as e:
            logger.error(f"Error calculating total size: {e}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get current storage statistics."""
        try:
            image_keys = self.redis_client.keys("chart:*")
            total_images = len(image_keys)
            total_size = self._get_total_size()
            
            # Calculate age statistics
            ages = []
            for key in image_keys:
                key_str = key.decode('utf-8')
                metadata_key = self._generate_metadata_key(key_str)
                
                try:
                    metadata = self.redis_client.hgetall(metadata_key)
                    if metadata and b'created_at' in metadata:
                        created_at = float(metadata[b'created_at'])
                        age_hours = (time.time() - created_at) / 3600
                        ages.append(age_hours)
                except Exception:
                    continue
            
            return {
                'total_images': total_images,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'oldest_image_age_hours': round(max(ages), 2) if ages else None,
                'newest_image_age_hours': round(min(ages), 2) if ages else None,
                'average_age_hours': round(sum(ages) / len(ages), 2) if ages else None,
                'max_age_hours': self.max_age_hours,
                'max_size_mb': self.max_total_size_mb,
                'redis_url': self.redis_url
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {'error': str(e)}
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval_minutes * 60)
                    self.cleanup_old_images()
                except Exception as e:
                    logger.error(f"Error in cleanup worker: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("Image cleanup thread started")
    
    def clear_all_images(self) -> Dict[str, int]:
        """Clear all images from Redis."""
        stats = {'images_removed': 0, 'bytes_freed': 0, 'errors': 0}
        
        try:
            image_keys = self.redis_client.keys("chart:*")
            metadata_keys = self.redis_client.keys("metadata:chart:*")
            
            # Delete all images and metadata
            if image_keys:
                self.redis_client.delete(*image_keys)
                stats['images_removed'] = len(image_keys)
            
            if metadata_keys:
                self.redis_client.delete(*metadata_keys)
            
            logger.info(f"Cleared all images: {stats['images_removed']} images removed")
            
        except Exception as e:
            logger.error(f"Error clearing all images: {e}")
            stats['errors'] += 1
        
        return stats

# Global Redis image manager instance
_redis_image_manager: Optional[RedisImageManager] = None

def get_redis_image_manager() -> RedisImageManager:
    """Get or create the global Redis image manager instance."""
    global _redis_image_manager
    if _redis_image_manager is None:
        _redis_image_manager = RedisImageManager()
    return _redis_image_manager

def initialize_redis_image_manager(**kwargs) -> RedisImageManager:
    """Initialize the global Redis image manager with custom settings."""
    global _redis_image_manager
    _redis_image_manager = RedisImageManager(**kwargs)
    return _redis_image_manager
