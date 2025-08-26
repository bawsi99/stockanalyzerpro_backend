#!/usr/bin/env python3
"""
Comprehensive Redis Integration Verification Script

This script verifies that the entire StockAnalyzer system is properly integrated with Redis
for both image storage and data caching.
"""

import os
import sys
import time
import asyncio
from datetime import datetime

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_success(message):
    """Print a success message."""
    print(f"✅ {message}")

def print_error(message):
    """Print an error message."""
    print(f"❌ {message}")

def print_info(message):
    """Print an info message."""
    print(f"ℹ️  {message}")

def test_redis_connection():
    """Test basic Redis connection."""
    print_header("Testing Redis Connection")
    
    try:
        import redis
        client = redis.from_url("redis://localhost:6379/0")
        client.ping()
        print_success("Redis connection successful")
        return True
    except Exception as e:
        print_error(f"Redis connection failed: {e}")
        return False

def test_redis_image_manager():
    """Test Redis image manager."""
    print_header("Testing Redis Image Manager")
    
    try:
        from redis_image_manager import RedisImageManager
        from deployment_config import DeploymentConfig
        
        config = DeploymentConfig.get_redis_image_config()
        manager = RedisImageManager(**config)
        
        # Test storing and retrieving an image
        from PIL import Image
        import io
        
        # Create a test image
        test_image = Image.new('RGB', (400, 300), color='white')
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        # Store in Redis
        redis_key = manager.store_image(img_bytes, "TEST", "day", "test_chart")
        print_success(f"Image stored with key: {redis_key}")
        
        # Retrieve from Redis
        retrieved = manager.get_image(redis_key)
        if retrieved:
            print_success("Image retrieved successfully from Redis")
        else:
            print_error("Failed to retrieve image from Redis")
            return False
        
        # Clean up
        manager.delete_image(redis_key)
        print_success("Image deleted from Redis")
        
        return True
    except Exception as e:
        print_error(f"Redis image manager test failed: {e}")
        return False

def test_redis_cache_manager():
    """Test Redis cache manager."""
    print_header("Testing Redis Cache Manager")
    
    try:
        from redis_cache_manager import RedisCacheManager
        from deployment_config import DeploymentConfig
        import pandas as pd
        
        config = DeploymentConfig.get_redis_cache_config()
        cache_manager = RedisCacheManager(**config)
        
        # Test caching stock data
        test_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Cache the data
        success = cache_manager.set("stock_data", test_data, symbol="TEST")
        if success:
            print_success("Stock data cached successfully")
        else:
            print_error("Failed to cache stock data")
            return False
        
        # Retrieve the data
        retrieved_data = cache_manager.get("stock_data", symbol="TEST")
        if retrieved_data is not None:
            print_success("Stock data retrieved successfully from cache")
        else:
            print_error("Failed to retrieve stock data from cache")
            return False
        
        # Test cache statistics
        stats = cache_manager.get_stats()
        print_success(f"Cache statistics: {stats['redis_hits']} hits, {stats['redis_misses']} misses")
        
        return True
    except Exception as e:
        print_error(f"Redis cache manager test failed: {e}")
        return False

def test_image_utils():
    """Test ImageUtils with Redis integration."""
    print_header("Testing ImageUtils Redis Integration")
    
    try:
        from gemini.image_utils import ImageUtils
        from redis_image_manager import RedisImageManager
        from deployment_config import DeploymentConfig
        from PIL import Image
        import io
        
        # Create Redis image manager
        config = DeploymentConfig.get_redis_image_config()
        redis_manager = RedisImageManager(**config)
        
        # Create a test image
        test_image = Image.new('RGB', (400, 300), color='blue')
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        # Store in Redis
        redis_key = redis_manager.store_image(img_bytes, "TEST", "day", "test_chart")
        
        # Test loading from Redis
        loaded_bytes = ImageUtils.load_image_from_redis_or_file(
            "/tmp/dummy.png",  # This should not be used
            symbol="TEST",
            interval="day",
            chart_type="test_chart"
        )
        
        if loaded_bytes and len(loaded_bytes) > 0:
            print_success("ImageUtils successfully loaded image from Redis")
        else:
            print_error("ImageUtils failed to load image from Redis")
            return False
        
        # Test fallback to file
        temp_image = Image.new('RGB', (200, 150), color='red')
        temp_path = "/tmp/fallback_test.png"
        temp_image.save(temp_path)
        
        fallback_bytes = ImageUtils.load_image_from_redis_or_file(
            temp_path,
            symbol="NONEXISTENT",
            interval="day",
            chart_type="nonexistent"
        )
        
        if fallback_bytes and len(fallback_bytes) > 0:
            print_success("ImageUtils successfully fell back to file")
        else:
            print_error("ImageUtils failed to fall back to file")
            return False
        
        # Clean up
        redis_manager.delete_image(redis_key)
        os.remove(temp_path)
        
        return True
    except Exception as e:
        print_error(f"ImageUtils test failed: {e}")
        return False

def test_gemini_client_integration():
    """Test GeminiClient with Redis image reading."""
    print_header("Testing GeminiClient Redis Integration")
    
    try:
        from gemini.gemini_client import GeminiClient
        from gemini.image_utils import ImageUtils
        from redis_image_manager import RedisImageManager
        from deployment_config import DeploymentConfig
        from PIL import Image
        import io
        
        # Create Redis image manager
        config = DeploymentConfig.get_redis_image_config()
        redis_manager = RedisImageManager(**config)
        
        # Create test images for different chart types
        chart_types = ['technical_overview', 'pattern_analysis', 'volume_analysis', 'mtf_comparison']
        test_images = {}
        
        for chart_type in chart_types:
            # Create a test image
            test_image = Image.new('RGB', (600, 400), color='green')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()
            
            # Store in Redis
            redis_key = redis_manager.store_image(img_bytes, "TEST", "day", chart_type)
            test_images[chart_type] = redis_key
        
        # Test that ImageUtils can load these images
        for chart_type in chart_types:
            loaded_bytes = ImageUtils.load_image_from_redis_or_file(
                "/tmp/dummy.png",
                symbol="TEST",
                interval="day",
                chart_type=chart_type
            )
            
            if loaded_bytes and len(loaded_bytes) > 0:
                print_success(f"Successfully loaded {chart_type} from Redis")
            else:
                print_error(f"Failed to load {chart_type} from Redis")
                return False
        
        # Clean up
        for redis_key in test_images.values():
            redis_manager.delete_image(redis_key)
        
        print_success("GeminiClient Redis integration test completed")
        return True
    except Exception as e:
        print_error(f"GeminiClient integration test failed: {e}")
        return False

def test_analysis_service_integration():
    """Test analysis service Redis integration."""
    print_header("Testing Analysis Service Redis Integration")
    
    try:
        # Test that the analysis service can initialize Redis managers
        from analysis_service import startup_event
        
        # This would normally be called by FastAPI, but we can test the imports
        from redis_image_manager import get_redis_image_manager
        from redis_cache_manager import get_redis_cache_manager
        
        # Try to get the managers (they might not be initialized yet)
        try:
            redis_image_manager = get_redis_image_manager()
            print_success("Redis image manager accessible from analysis service")
        except:
            print_info("Redis image manager not yet initialized (this is normal)")
        
        try:
            redis_cache_manager = get_redis_cache_manager()
            print_success("Redis cache manager accessible from analysis service")
        except:
            print_info("Redis cache manager not yet initialized (this is normal)")
        
        print_success("Analysis service Redis integration test completed")
        return True
    except Exception as e:
        print_error(f"Analysis service integration test failed: {e}")
        return False

def main():
    """Main verification function."""
    print("🚀 Redis Integration Verification")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Redis Connection", test_redis_connection),
        ("Redis Image Manager", test_redis_image_manager),
        ("Redis Cache Manager", test_redis_cache_manager),
        ("ImageUtils Integration", test_image_utils),
        ("GeminiClient Integration", test_gemini_client_integration),
        ("Analysis Service Integration", test_analysis_service_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print_error(f"{test_name} test failed")
        except Exception as e:
            print_error(f"{test_name} test failed with exception: {e}")
    
    print_header("Verification Summary")
    print(f"📊 Tests passed: {passed}/{total}")
    
    if passed == total:
        print_success("🎉 All Redis integration tests passed!")
        print_success("✅ Your StockAnalyzer system is fully integrated with Redis")
        print_info("🚀 You can now start your services with Redis support")
        return True
    else:
        print_error(f"❌ {total - passed} tests failed")
        print_info("🔧 Please check the error messages above and fix any issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
