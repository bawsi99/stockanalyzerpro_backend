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
        Load image bytes from file path.
        
        Args:
            image_path: Local file path
            symbol: Stock symbol (kept for compatibility, no longer used)
            interval: Time interval (kept for compatibility, no longer used)
            chart_type: Chart type (kept for compatibility, no longer used)
            
        Returns:
            Image bytes
        """
        # Redis image storage has been removed - charts are now generated in-memory
        # Always load from file path
        return ImageUtils.load_image_bytes(image_path)
    
    @staticmethod
    def load_image_from_redis_key(redis_key: str) -> bytes:
        """
        Load image bytes from Redis key (deprecated - Redis image storage removed).
        
        Args:
            redis_key: Redis key for the image (no longer used)
            
        Returns:
            Image bytes (always raises error as Redis image storage is removed)
        """
        raise ValueError("Redis image storage has been removed - charts are now generated in-memory")

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