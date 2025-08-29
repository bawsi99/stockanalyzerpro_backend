#!/usr/bin/env python3
"""
Configuration Update Script: Switch to Unified Redis Cache

This script updates the system configuration to use the unified Redis cache manager
instead of local caching systems.

It will:
1. Update deployment configuration to use unified Redis cache
2. Update environment variables
3. Create backup of current configuration
4. Verify the new configuration

Usage:
    python update_cache_config.py [--dry-run] [--backup]
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CacheConfigUpdater:
    """Updates cache configuration to use unified Redis cache."""
    
    def __init__(self, dry_run: bool = False, backup: bool = False):
        self.dry_run = dry_run
        self.backup = backup
        self.backup_dir = Path(f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
    def backup_configuration(self):
        """Create backup of current configuration files."""
        if not self.backup:
            logger.info("Backup not requested, skipping")
            return True
        
        logger.info(f"Creating backup in: {self.backup_dir}")
        
        try:
            self.backup_dir.mkdir(exist_ok=True)
            
            # Files to backup
            config_files = [
                "deployment_config.py",
                "config.py",
                ".env",
                "requirements.txt"
            ]
            
            for config_file in config_files:
                if Path(config_file).exists():
                    backup_path = self.backup_dir / config_file
                    shutil.copy2(config_file, backup_path)
                    logger.info(f"Backed up: {config_file}")
            
            # Create backup info
            backup_info = {
                "backup_time": datetime.now().isoformat(),
                "files_backed_up": [f for f in config_files if Path(f).exists()],
                "original_config": "Cache configuration before Redis unification"
            }
            
            with open(self.backup_dir / "backup_info.json", "w") as f:
                json.dump(backup_info, f, indent=2)
            
            logger.info("✅ Configuration backup completed")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def update_deployment_config(self):
        """Update deployment configuration to use unified Redis cache."""
        logger.info("Updating deployment configuration...")
        
        config_file = "deployment_config.py"
        if not Path(config_file).exists():
            logger.warning(f"Configuration file {config_file} not found, skipping")
            return True
        
        try:
            # Read current configuration
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Add unified Redis cache configuration
            unified_cache_config = '''
    # Unified Redis cache manager configurations for different environments
    UNIFIED_REDIS_CACHE_CONFIGS = {
        "development": {
            "enable_compression": True,
            "cleanup_interval_minutes": 60,
            "redis_url": "redis://localhost:6379/0",
            "lru_capacity": 128,
            "ttl_settings": {
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
        },
        "staging": {
            "enable_compression": True,
            "cleanup_interval_minutes": 30,
            "redis_url": "redis://localhost:6379/0",
            "lru_capacity": 256,
            "ttl_settings": {
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
        },
        "production": {
            "enable_compression": True,
            "cleanup_interval_minutes": 15,
            "redis_url": "redis://localhost:6379/0",
            "lru_capacity": 512,
            "ttl_settings": {
                "stock_data": 180,      # 3 minutes
                "indicators": 300,      # 5 minutes
                "patterns": 900,        # 15 minutes
                "sector_data": 1800,    # 30 minutes
                "ml_predictions": 900,  # 15 minutes
                "api_responses": 180,   # 3 minutes
                "historical_data": 1800, # 30 minutes (market-aware)
                "instruments": 86400,   # 24 hours
                "live_data": 30,        # 30 seconds
                "enhanced_data": 180    # 3 minutes
            }
        }
    }
'''
            
            # Add getter method for unified cache config
            getter_method = '''
    @classmethod
    def get_unified_redis_cache_config(cls) -> Dict[str, Any]:
        """Get unified Redis cache manager configuration for current environment."""
        environment = cls.get_environment()
        config = cls.UNIFIED_REDIS_CACHE_CONFIGS.get(environment, cls.UNIFIED_REDIS_CACHE_CONFIGS["development"]).copy()
        
        # Override with environment variables if present
        config.update({
            "enable_compression": os.getenv("REDIS_CACHE_ENABLE_COMPRESSION", "true").lower() == "true",
            "cleanup_interval_minutes": int(os.getenv("REDIS_CACHE_CLEANUP_INTERVAL_MINUTES", config["cleanup_interval_minutes"])),
            "redis_url": os.getenv("REDIS_URL", config["redis_url"]),
            "lru_capacity": int(os.getenv("REDIS_CACHE_LRU_CAPACITY", config["lru_capacity"]))
        })
        
        return config
'''
            
            # Find the right place to insert the configuration
            if 'UNIFIED_REDIS_CACHE_CONFIGS' not in content:
                # Insert after existing REDIS_CACHE_CONFIGS
                if 'REDIS_CACHE_CONFIGS' in content:
                    insert_point = content.find('REDIS_CACHE_CONFIGS')
                    end_point = content.find('}', insert_point) + 1
                    
                    new_content = (
                        content[:end_point] + 
                        unified_cache_config + 
                        content[end_point:]
                    )
                else:
                    # Insert before the first @classmethod
                    insert_point = content.find('@classmethod')
                    new_content = (
                        content[:insert_point] + 
                        unified_cache_config + 
                        content[insert_point:]
                    )
                
                # Add getter method
                if 'get_unified_redis_cache_config' not in new_content:
                    # Insert before the last @classmethod
                    last_classmethod = new_content.rfind('@classmethod')
                    if last_classmethod != -1:
                        new_content = (
                            new_content[:last_classmethod] + 
                            getter_method + 
                            '\n    ' + 
                            new_content[last_classmethod:]
                        )
                
                if not self.dry_run:
                    # Write updated configuration
                    with open(config_file, 'w') as f:
                        f.write(new_content)
                    logger.info(f"✅ Updated {config_file}")
                else:
                    logger.info(f"[DRY RUN] Would update {config_file}")
                
                return True
            else:
                logger.info(f"Unified Redis cache configuration already exists in {config_file}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating deployment configuration: {e}")
            return False
    
    def update_environment_variables(self):
        """Update environment variables for unified Redis cache."""
        logger.info("Updating environment variables...")
        
        env_file = ".env"
        if not Path(env_file).exists():
            logger.info(f"Environment file {env_file} not found, creating new one")
            env_content = self._create_env_content()
        else:
            # Read existing environment file
            with open(env_file, 'r') as f:
                existing_content = f.read()
            
            # Update Redis-related variables
            env_content = self._update_env_content(existing_content)
        
        if not self.dry_run:
            # Write updated environment file
            with open(env_file, 'w') as f:
                f.write(env_content)
            logger.info(f"✅ Updated {env_file}")
        else:
            logger.info(f"[DRY RUN] Would update {env_file}")
        
        return True
    
    def _create_env_content(self) -> str:
        """Create new environment file content."""
        return f"""# Redis Configuration for Unified Cache
REDIS_URL=redis://localhost:6379/0

# Unified Redis Cache Manager Configuration
REDIS_CACHE_ENABLE_COMPRESSION=true
REDIS_CACHE_CLEANUP_INTERVAL_MINUTES=60
REDIS_CACHE_LRU_CAPACITY=128

# Redis Image Manager Configuration
REDIS_IMAGE_MAX_AGE_HOURS=24
REDIS_IMAGE_MAX_SIZE_MB=1000
REDIS_IMAGE_CLEANUP_INTERVAL_MINUTES=60
REDIS_IMAGE_ENABLE_CLEANUP=true
REDIS_IMAGE_QUALITY=85
REDIS_IMAGE_FORMAT=PNG

# Environment Configuration
ENVIRONMENT=development

# Other configurations...
# Add your existing environment variables here
"""
    
    def _update_env_content(self, existing_content: str) -> str:
        """Update existing environment file content."""
        # Remove old Redis cache variables
        lines = existing_content.split('\n')
        updated_lines = []
        
        # Variables to remove/replace
        old_vars = [
            'REDIS_CACHE_ENABLE_LOCAL_FALLBACK',
            'REDIS_CACHE_LOCAL_SIZE'
        ]
        
        # Variables to add/update
        new_vars = {
            'REDIS_CACHE_ENABLE_COMPRESSION': 'true',
            'REDIS_CACHE_CLEANUP_INTERVAL_MINUTES': '60',
            'REDIS_CACHE_LRU_CAPACITY': '128'
        }
        
        for line in lines:
            # Skip old variables
            if any(line.strip().startswith(f"{var}=") for var in old_vars):
                continue
            
            # Add new variables if not present
            for var, value in new_vars.items():
                if line.strip().startswith(f"{var}="):
                    line = f"{var}={value}"
                    new_vars.pop(var)  # Mark as added
                    break
            
            updated_lines.append(line)
        
        # Add any remaining new variables
        for var, value in new_vars.items():
            updated_lines.append(f"{var}={value}")
        
        return '\n'.join(updated_lines)
    
    def update_requirements(self):
        """Update requirements.txt to ensure Redis dependency is present."""
        logger.info("Checking requirements.txt...")
        
        requirements_file = "requirements.txt"
        if not Path(requirements_file).exists():
            logger.warning(f"Requirements file {requirements_file} not found, skipping")
            return True
        
        try:
            with open(requirements_file, 'r') as f:
                content = f.read()
            
            # Check if Redis is already in requirements
            if 'redis' not in content.lower():
                if not self.dry_run:
                    # Add Redis dependency
                    with open(requirements_file, 'a') as f:
                        f.write('\n# Redis for unified caching\nredis>=5.2.1\n')
                    logger.info(f"✅ Added Redis dependency to {requirements_file}")
                else:
                    logger.info(f"[DRY RUN] Would add Redis dependency to {requirements_file}")
            else:
                logger.info(f"Redis dependency already present in {requirements_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating requirements: {e}")
            return False
    
    def create_migration_guide(self):
        """Create a migration guide for developers."""
        guide_content = f"""# Cache Migration Guide

## Overview
This system has been migrated from local caching to unified Redis caching.

## What Changed
- **Local file cache** → Redis cache
- **In-memory cache** → Redis cache  
- **Separate cache managers** → Unified Redis cache manager

## New Configuration
- Use `redis_unified_cache_manager.py` for all caching operations
- Configuration in `deployment_config.py` under `UNIFIED_REDIS_CACHE_CONFIGS`
- Environment variables prefixed with `REDIS_CACHE_`

## Migration Steps
1. Run: `python migrate_to_redis_cache.py --dry-run` to see what will be migrated
2. Run: `python migrate_to_redis_cache.py` to perform migration
3. Run: `python migrate_to_redis_cache.py --cleanup-local` to remove old cache files

## Usage Examples
```python
from redis_unified_cache_manager import get_unified_redis_cache_manager

# Get cache manager
cache = get_unified_redis_cache_manager()

# Cache data
cache.set('stock_data', data, symbol='RELIANCE', exchange='NSE', interval='day', period=365)

# Retrieve data
data = cache.get('stock_data', symbol='RELIANCE', exchange='NSE', interval='day', period=365)
```

## Benefits
- **Centralized caching**: All cache data in one place
- **Better performance**: Redis is faster than file I/O
- **Automatic cleanup**: TTL-based expiration
- **Market-aware TTL**: Longer cache duration when market is closed
- **LRU behavior**: Automatic eviction of least used data

## Monitoring
- Check cache stats: `cache.get_stats()`
- Monitor Redis memory usage
- Use Redis CLI: `redis-cli info memory`

Generated on: {datetime.now().isoformat()}
"""
        
        guide_file = "CACHE_MIGRATION_GUIDE.md"
        if not self.dry_run:
            with open(guide_file, 'w') as f:
                f.write(guide_content)
            logger.info(f"✅ Created migration guide: {guide_file}")
        else:
            logger.info(f"[DRY RUN] Would create migration guide: {guide_file}")
        
        return True
    
    def verify_configuration(self):
        """Verify the new configuration works correctly."""
        logger.info("Verifying configuration...")
        
        try:
            # Test importing the unified cache manager
            from redis_unified_cache_manager import get_unified_redis_cache_manager
            
            # Test getting cache manager
            cache_manager = get_unified_redis_cache_manager()
            
            # Test basic functionality
            test_data = {"test": "config", "timestamp": datetime.now().isoformat()}
            
            # Test set
            success = cache_manager.set('test', test_data, ttl_seconds=60)
            if not success:
                logger.error("Configuration verification failed: Redis set test failed")
                return False
            
            # Test get
            retrieved_data = cache_manager.get('test')
            if retrieved_data != test_data:
                logger.error("Configuration verification failed: Redis get test failed")
                return False
            
            # Clean up test data
            cache_manager.delete('test')
            
            logger.info("✅ Configuration verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Configuration verification failed: {e}")
            return False
    
    def run_update(self):
        """Run the complete configuration update process."""
        logger.info("Starting cache configuration update...")
        logger.info(f"Dry run mode: {self.dry_run}")
        logger.info(f"Backup mode: {self.backup}")
        
        try:
            # Create backup if requested
            if not self.backup_configuration():
                logger.error("Backup failed, aborting update")
                return False
            
            # Update configuration files
            if not self.update_deployment_config():
                logger.error("Failed to update deployment configuration")
                return False
            
            if not self.update_environment_variables():
                logger.error("Failed to update environment variables")
                return False
            
            if not self.update_requirements():
                logger.error("Failed to update requirements")
                return False
            
            # Create migration guide
            if not self.create_migration_guide():
                logger.error("Failed to create migration guide")
                return False
            
            # Verify configuration
            if not self.verify_configuration():
                logger.error("Configuration verification failed")
                return False
            
            logger.info("✅ Configuration update completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return False

def main():
    """Main configuration update function."""
    parser = argparse.ArgumentParser(description="Update cache configuration to use unified Redis cache")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without actually doing it")
    parser.add_argument("--backup", action="store_true", help="Create backup of current configuration")
    
    args = parser.parse_args()
    
    try:
        # Create configuration updater
        updater = CacheConfigUpdater(dry_run=args.dry_run, backup=args.backup)
        
        # Run update
        success = updater.run_update()
        
        if success:
            logger.info("Configuration update completed successfully!")
            if args.dry_run:
                logger.info("This was a dry run. Run without --dry-run to apply changes.")
            sys.exit(0)
        else:
            logger.error("Configuration update failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Configuration update interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during configuration update: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
