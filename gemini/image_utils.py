import io
import os
import base64
from PIL import Image
import matplotlib.pyplot as plt

class ImageUtils:
    @staticmethod
    def load_image_bytes(image_path: str) -> bytes:
        """Load image bytes from file path."""
        with open(image_path, 'rb') as f:
            return f.read()
    
    @staticmethod
    def load_image_from_redis_or_file(image_path: str, symbol: str = None, interval: str = None, chart_type: str = None) -> bytes:
        """
        Load image bytes from Redis if available, otherwise from file.
        
        Args:
            image_path: Redis key or local file path
            symbol: Stock symbol for Redis lookup (used if image_path is not a Redis key)
            interval: Time interval for Redis lookup (used if image_path is not a Redis key)
            chart_type: Chart type for Redis lookup (used if image_path is not a Redis key)
            
        Returns:
            Image bytes
        """
        # Check if image_path is a Redis key
        if isinstance(image_path, str) and image_path.startswith('chart:'):
            try:
                from redis_image_manager import get_redis_image_manager
                redis_manager = get_redis_image_manager()
                
                # Load image directly from Redis key
                image_data = redis_manager.get_image(image_path)
                if image_data:
                    # Convert base64 data back to bytes
                    base64_data = image_data['data'].split(',')[1]  # Remove data URL prefix
                    return base64.b64decode(base64_data)
                else:
                    print(f"Image not found in Redis with key: {image_path}")
                    raise ValueError(f"Image not found in Redis with key: {image_path}")
            except Exception as e:
                print(f"Redis lookup failed for key {image_path}: {e}")
                raise ValueError(f"Failed to load image from Redis: {e}")
        
        # Try Redis lookup by symbol/interval/chart_type if we have the required parameters
        if symbol and interval and chart_type:
            try:
                from redis_image_manager import get_redis_image_manager
                redis_manager = get_redis_image_manager()
                
                # Get images for this symbol and interval
                images = redis_manager.get_images_by_symbol(symbol, interval)
                
                # Find the matching chart type
                for image_data in images:
                    metadata = image_data.get('metadata', {})
                    if metadata.get('chart_type') == chart_type:
                        # Convert base64 data back to bytes
                        base64_data = image_data['data'].split(',')[1]  # Remove data URL prefix
                        return base64.b64decode(base64_data)
                
                print(f"Image not found in Redis for {symbol}_{interval}_{chart_type}, falling back to file")
            except Exception as e:
                print(f"Redis lookup failed: {e}, falling back to file")
        
        # Fallback to file
        return ImageUtils.load_image_bytes(image_path)
    
    @staticmethod
    def load_image_from_redis_key(redis_key: str) -> bytes:
        """
        Load image bytes from Redis using a specific key.
        
        Args:
            redis_key: Redis key for the image
            
        Returns:
            Image bytes
        """
        try:
            from redis_image_manager import get_redis_image_manager
            redis_manager = get_redis_image_manager()
            
            image_data = redis_manager.get_image(redis_key)
            if image_data:
                # Convert base64 data back to bytes
                base64_data = image_data['data'].split(',')[1]  # Remove data URL prefix
                return base64.b64decode(base64_data)
            else:
                raise ValueError(f"Image not found in Redis with key: {redis_key}")
        except Exception as e:
            raise ValueError(f"Failed to load image from Redis: {e}")

    @staticmethod
    def bytes_to_image(image_data: bytes) -> Image.Image:
        """Convert bytes to PIL Image."""
        return Image.open(io.BytesIO(image_data))

    @staticmethod
    def figure_to_image(figure) -> Image.Image:
        """Convert matplotlib figure to PIL Image."""
        buf = io.BytesIO()
        figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        plt.close(figure)
        return image 