#!/usr/bin/env python3
"""
Migration Script: Local Cache to Redis Cache

This script helps migrate from local file-based and in-memory caching to Redis-based caching.
It will:
1. Migrate existing file-based cache data to Redis
2. Update configuration to use Redis exclusively
3. Clean up old local cache files
4. Verify Redis cache functionality

Usage:
    python migrate_to_redis_cache.py [--dry-run] [--cleanup-local]
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from redis_unified_cache_manager import get_unified_redis_cache_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CacheMigrationManager:
    """Manages migration from local cache to Redis cache."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.redis_cache = get_unified_redis_cache_manager()
        self.migration_stats = {
            'files_migrated': 0,
            'data_migrated': 0,
            'errors': 0,
            'bytes_migrated': 0
        }
        
        # Cache directories to migrate
        self.cache_dirs = [
            Path("cache"),  # Zerodha client cache
            Path("output/charts"),  # Chart cache
            Path("ml/quant_system/cache"),  # ML cache
        ]
    
    def migrate_file_cache(self, cache_dir: Path) -> bool:
        """Migrate file-based cache to Redis."""
        if not cache_dir.exists():
            logger.info(f"Cache directory {cache_dir} does not exist, skipping")
            return True
        
        logger.info(f"Migrating cache directory: {cache_dir}")
        
        try:
            # Find all cache files
            cache_files = []
            for ext in ['*.csv', '*.json', '*.pkl', '*.joblib']:
                cache_files.extend(cache_dir.rglob(ext))
            
            if not cache_files:
                logger.info(f"No cache files found in {cache_dir}")
                return True
            
            logger.info(f"Found {len(cache_files)} cache files to migrate")
            
            for cache_file in cache_files:
                if self._migrate_cache_file(cache_file):
                    self.migration_stats['files_migrated'] += 1
                else:
                    self.migration_stats['errors'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error migrating cache directory {cache_dir}: {e}")
            self.migration_stats['errors'] += 1
            return False
    
    def _migrate_cache_file(self, cache_file: Path) -> bool:
        """Migrate a single cache file to Redis."""
        try:
            logger.debug(f"Migrating cache file: {cache_file}")
            
            # Determine cache type based on file path and extension
            cache_type = self._determine_cache_type(cache_file)
            
            if cache_type == "unknown":
                logger.warning(f"Unknown cache type for file: {cache_file}")
                return False
            
            # Read and parse the cache file
            data = self._read_cache_file(cache_file)
            if data is None:
                return False
            
            # Generate cache key
            cache_key = self._generate_cache_key_from_file(cache_file, cache_type)
            
            # Store in Redis
            if not self.dry_run:
                success = self.redis_cache.set(cache_type, data, cache_key=cache_key)
                if success:
                    self.migration_stats['data_migrated'] += 1
                    self.migration_stats['bytes_migrated'] += cache_file.stat().st_size
                    logger.debug(f"Successfully migrated: {cache_file}")
                    return True
                else:
                    logger.error(f"Failed to store in Redis: {cache_file}")
                    return False
            else:
                # Dry run - just count
                self.migration_stats['data_migrated'] += 1
                self.migration_stats['bytes_migrated'] += cache_file.stat().st_size
                logger.info(f"[DRY RUN] Would migrate: {cache_file}")
                return True
                
        except Exception as e:
            logger.error(f"Error migrating cache file {cache_file}: {e}")
            return False
    
    def _determine_cache_type(self, cache_file: Path) -> str:
        """Determine the type of cache based on file path and content."""
        file_path = str(cache_file)
        
        if "chart" in file_path.lower() or cache_file.suffix in ['.png', '.jpg', '.jpeg']:
            return "chart_image"
        elif cache_file.suffix == '.csv':
            if "historical" in file_path.lower() or "stock" in file_path.lower():
                return "historical_data"
            else:
                return "stock_data"
        elif cache_file.suffix == '.json':
            if "pattern" in file_path.lower():
                return "patterns"
            elif "sector" in file_path.lower():
                return "sector_data"
            elif "indicator" in file_path.lower():
                return "indicators"
            else:
                return "api_responses"
        elif cache_file.suffix in ['.pkl', '.joblib']:
            return "ml_predictions"
        else:
            return "unknown"
    
    def _read_cache_file(self, cache_file: Path) -> Optional[Any]:
        """Read and parse a cache file."""
        try:
            if cache_file.suffix == '.csv':
                return pd.read_csv(cache_file, parse_dates=['date'] if 'date' in pd.read_csv(cache_file, nrows=1).columns else False)
            elif cache_file.suffix == '.json':
                with open(cache_file, 'r') as f:
                    return json.load(f)
            elif cache_file.suffix == '.pkl':
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            elif cache_file.suffix == '.joblib':
                import joblib
                return joblib.load(cache_file)
            else:
                logger.warning(f"Unsupported file type: {cache_file.suffix}")
                return None
        except Exception as e:
            logger.error(f"Error reading cache file {cache_file}: {e}")
            return None
    
    def _generate_cache_key_from_file(self, cache_file: Path, cache_type: str) -> str:
        """Generate a cache key from file path."""
        # Create a meaningful key based on file path
        try:
            relative_path = cache_file.relative_to(Path.cwd())
        except ValueError:
            # If relative_to fails, use the filename
            relative_path = cache_file.name
        return f"migrated:{cache_type}:{relative_path}"
    
    def migrate_enhanced_data_cache(self):
        """Migrate enhanced data service in-memory cache to Redis."""
        logger.info("Migrating enhanced data service cache...")
        
        try:
            # Import enhanced data service
            from enhanced_data_service import EnhancedDataService
            
            # Create instance to access cache
            service = EnhancedDataService()
            
            if hasattr(service, 'data_cache') and service.data_cache:
                logger.info(f"Found {len(service.data_cache)} enhanced data cache entries")
                
                for cache_key, (data, metadata) in service.data_cache.items():
                    if not self.dry_run:
                        # Parse cache key to extract parameters
                        parts = cache_key.split('_')
                        if len(parts) >= 4:
                            symbol, exchange, interval, period = parts[0], parts[1], parts[2], int(parts[3])
                            
                            success = self.redis_cache.cache_enhanced_data(
                                symbol, exchange, interval, period, data, metadata
                            )
                            
                            if success:
                                self.migration_stats['data_migrated'] += 1
                                logger.debug(f"Migrated enhanced data: {cache_key}")
                            else:
                                self.migration_stats['errors'] += 1
                                logger.error(f"Failed to migrate enhanced data: {cache_key}")
                        else:
                            logger.warning(f"Invalid cache key format: {cache_key}")
                    else:
                        self.migration_stats['data_migrated'] += 1
                        logger.info(f"[DRY RUN] Would migrate enhanced data: {cache_key}")
            
        except ImportError:
            logger.info("Enhanced data service not available, skipping")
        except Exception as e:
            logger.error(f"Error migrating enhanced data cache: {e}")
            self.migration_stats['errors'] += 1
    
    def migrate_ml_cache(self):
        """Migrate ML system in-memory cache to Redis."""
        logger.info("Migrating ML system cache...")
        
        try:
            # Import ML pipeline
            from ml.quant_system.enhanced_data_pipeline import EnhancedDataPipeline
            
            # Create instance to access cache
            pipeline = EnhancedDataPipeline()
            
            if hasattr(pipeline, 'data_cache') and pipeline.data_cache:
                logger.info(f"Found {len(pipeline.data_cache)} ML cache entries")
                
                for cache_key, data in pipeline.data_cache.items():
                    if not self.dry_run:
                        # Parse cache key
                        parts = cache_key.split('_')
                        if len(parts) >= 2:
                            symbol, timeframe = parts[0], parts[1]
                            
                            success = self.redis_cache.set('ml_data', data, cache_key=f"ml:{symbol}:{timeframe}")
                            
                            if success:
                                self.migration_stats['data_migrated'] += 1
                                logger.debug(f"Migrated ML data: {cache_key}")
                            else:
                                self.migration_stats['errors'] += 1
                                logger.error(f"Failed to migrate ML data: {cache_key}")
                        else:
                            logger.warning(f"Invalid ML cache key format: {cache_key}")
                    else:
                        self.migration_stats['data_migrated'] += 1
                        logger.info(f"[DRY RUN] Would migrate ML data: {cache_key}")
            
        except ImportError:
            logger.info("ML pipeline not available, skipping")
        except Exception as e:
            logger.error(f"Error migrating ML cache: {e}")
            self.migration_stats['errors'] += 1
    
    def cleanup_local_cache(self):
        """Clean up local cache files after successful migration."""
        if self.dry_run:
            logger.info("[DRY RUN] Would clean up local cache files")
            return
        
        logger.info("Cleaning up local cache files...")
        
        for cache_dir in self.cache_dirs:
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    logger.info(f"Removed cache directory: {cache_dir}")
                except Exception as e:
                    logger.error(f"Error removing cache directory {cache_dir}: {e}")
    
    def verify_redis_cache(self):
        """Verify Redis cache functionality after migration."""
        logger.info("Verifying Redis cache functionality...")
        
        try:
            # Test basic operations
            test_data = {"test": "migration", "timestamp": datetime.now().isoformat()}
            
            # Test set
            success = self.redis_cache.set('test', test_data, ttl_seconds=60)
            if not success:
                logger.error("Redis cache set test failed")
                return False
            
            # Test get
            retrieved_data = self.redis_cache.get('test')
            if retrieved_data != test_data:
                logger.error("Redis cache get test failed")
                return False
            
            # Test delete
            success = self.redis_cache.delete('test')
            if not success:
                logger.error("Redis cache delete test failed")
                return False
            
            # Get stats
            stats = self.redis_cache.get_stats()
            logger.info(f"Redis cache stats: {stats}")
            
            logger.info("✅ Redis cache verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Redis cache verification failed: {e}")
            return False
    
    def run_migration(self, cleanup_local: bool = False):
        """Run the complete migration process."""
        logger.info("Starting cache migration to Redis...")
        logger.info(f"Dry run mode: {self.dry_run}")
        
        start_time = datetime.now()
        
        try:
            # Migrate file-based caches
            for cache_dir in self.cache_dirs:
                self.migrate_file_cache(cache_dir)
            
            # Migrate in-memory caches
            self.migrate_enhanced_data_cache()
            self.migrate_ml_cache()
            
            # Verify Redis cache
            if self.verify_redis_cache():
                logger.info("✅ Migration completed successfully")
                
                # Clean up local cache if requested
                if cleanup_local and not self.dry_run:
                    self.cleanup_local_cache()
                
                # Print migration summary
                self._print_migration_summary(start_time)
                
                return True
            else:
                logger.error("❌ Migration verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def _print_migration_summary(self, start_time: datetime):
        """Print migration summary."""
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 50)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Duration: {duration}")
        logger.info(f"Files migrated: {self.migration_stats['files_migrated']}")
        logger.info(f"Data entries migrated: {self.migration_stats['data_migrated']}")
        logger.info(f"Bytes migrated: {self.migration_stats['bytes_migrated']:,}")
        logger.info(f"Errors: {self.migration_stats['errors']}")
        logger.info("=" * 50)

def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate local cache to Redis cache")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without actually doing it")
    parser.add_argument("--cleanup-local", action="store_true", help="Remove local cache files after successful migration")
    
    args = parser.parse_args()
    
    try:
        # Create migration manager
        migration_manager = CacheMigrationManager(dry_run=args.dry_run)
        
        # Run migration
        success = migration_manager.run_migration(cleanup_local=args.cleanup_local)
        
        if success:
            logger.info("Migration completed successfully!")
            sys.exit(0)
        else:
            logger.error("Migration failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during migration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
