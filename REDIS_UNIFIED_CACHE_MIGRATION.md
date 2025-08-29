# Redis Unified Cache Migration Guide

## 🎯 Overview

This guide helps you migrate your StockAnalyzer Pro system from local caching (file-based and in-memory) to a **unified Redis-based caching system**. This migration provides better performance, centralized management, and automatic cleanup.

## 🚀 What You'll Get

### ✅ **Benefits of Unified Redis Caching**
- **Centralized Storage**: All cache data in one Redis instance
- **Better Performance**: Redis is faster than file I/O operations
- **Automatic Cleanup**: TTL-based expiration and LRU eviction
- **Market-Aware TTL**: Longer cache duration when market is closed
- **Compression**: Automatic compression for large datasets
- **Monitoring**: Built-in statistics and performance metrics
- **Scalability**: Easy to scale across multiple instances

### 🔄 **What Gets Migrated**
- **File-based cache** (`cache/` directory) → Redis cache
- **In-memory cache** (Enhanced Data Service) → Redis cache
- **ML system cache** → Redis cache
- **Chart image cache** → Redis (already implemented)
- **Technical indicators cache** → Redis (already implemented)

## 📋 Prerequisites

### 1. **Redis Installation**
Make sure Redis is running on your system:

```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG

# If Redis is not installed:
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Docker
docker run -d --name redis -p 6379:6379 redis:alpine
```

### 2. **Python Dependencies**
Ensure Redis Python package is installed:

```bash
pip install redis>=5.2.1
```

## 🛠️ Migration Process

### **Step 1: Configuration Update (Recommended First)**

Update your system configuration to use the unified Redis cache:

```bash
# Dry run to see what will be changed
python update_cache_config.py --dry-run

# Apply changes with backup
python update_cache_config.py --backup

# Apply changes without backup
python update_cache_config.py
```

This will:
- Update `deployment_config.py` with unified Redis cache settings
- Update environment variables
- Ensure Redis dependency is in requirements.txt
- Create a migration guide

### **Step 2: Data Migration**

Migrate existing cache data to Redis:

```bash
# See what will be migrated (dry run)
python migrate_to_redis_cache.py --dry-run

# Perform migration
python migrate_to_redis_cache.py

# Perform migration and clean up old cache files
python migrate_to_redis_cache.py --cleanup-local
```

### **Step 3: Verification**

Verify the migration was successful:

```bash
# Test Redis cache functionality
python -c "
from redis_unified_cache_manager import get_unified_redis_cache_manager
cache = get_unified_redis_cache_manager()
print('✅ Redis cache working:', cache.redis_available)
print('📊 Cache stats:', cache.get_stats())
"
```

## 🔧 Configuration

### **Environment Variables**

Add these to your `.env` file:

```bash
# Redis Connection
REDIS_URL=redis://localhost:6379/0

# Unified Redis Cache Manager Configuration
REDIS_CACHE_ENABLE_COMPRESSION=true
REDIS_CACHE_CLEANUP_INTERVAL_MINUTES=60
REDIS_CACHE_LRU_CAPACITY=128

# Redis Image Manager Configuration removed - charts are now generated in-memory
```

### **Deployment Configuration**

The system automatically detects your environment and applies appropriate settings:

- **Development**: Longer TTL, larger storage limits
- **Staging**: Moderate TTL, medium storage limits  
- **Production**: Aggressive TTL, small storage limits

## 📚 Usage Examples

### **Basic Cache Operations**

```python
from redis_unified_cache_manager import get_unified_redis_cache_manager

# Get cache manager
cache = get_unified_redis_cache_manager()

# Cache stock data
cache.set('stock_data', stock_df, symbol='RELIANCE', exchange='NSE', interval='day', period=365)

# Retrieve stock data
data = cache.get('stock_data', symbol='RELIANCE', exchange='NSE', interval='day', period=365)

# Cache technical indicators
cache.set('indicators', indicators_dict, symbol='RELIANCE', exchange='NSE', interval='day')

# Cache patterns
cache.set('patterns', patterns_dict, symbol='RELIANCE', exchange='NSE', interval='day')
```

### **Specialized Cache Methods**

```python
# Cache historical data with market-aware TTL
cache.cache_historical_data(
    symbol='RELIANCE', 
    exchange='NSE', 
    interval='day',
    from_date=datetime(2024, 1, 1),
    to_date=datetime(2024, 12, 31),
    data=historical_df
)

# Cache instruments list
cache.cache_instruments(instruments_list)

# Cache live data with short TTL
cache.cache_live_data('RELIANCE', 'NSE', live_data)

# Cache enhanced data service data
cache.cache_enhanced_data(
    symbol='RELIANCE',
    exchange='NSE', 
    interval='day',
    period=365,
    data=data_df,
    metadata=metadata_dict
)
```

### **Cache Management**

```python
# Get cache statistics
stats = cache.get_stats()
print(f"Redis hits: {stats['redis_hits']}")
print(f"Redis misses: {stats['redis_misses']}")
print(f"Compression savings: {stats['compression_savings']} bytes")

# Clear specific cache type
cache.clear('stock_data')

# Clear all cache
cache.clear()

# Delete specific item
cache.delete('stock_data', symbol='RELIANCE', exchange='NSE', interval='day', period=365)
```

## 🔍 Monitoring and Debugging

### **Cache Statistics**

```python
# Get comprehensive cache stats
stats = cache.get_stats()
print(json.dumps(stats, indent=2))

# Output includes:
# - Redis hits/misses
# - Compression savings
# - LRU eviction count
# - Redis server info
# - Market status
```

### **Redis CLI Monitoring**

```bash
# Monitor Redis in real-time
redis-cli monitor

# Check Redis memory usage
redis-cli info memory

# Check Redis keys
redis-cli keys "cache:*"

# Check specific cache type
redis-cli keys "cache:stock_data:*"
```

### **Debug Mode**

Enable debug logging to see cache operations:

```python
import logging
logging.getLogger('redis_unified_cache_manager').setLevel(logging.DEBUG)
```

## 🚨 Troubleshooting

### **Common Issues**

#### 1. **Redis Connection Failed**
```bash
# Check if Redis is running
redis-cli ping

# Check Redis URL in .env
echo $REDIS_URL

# Test Redis connection
redis-cli -u redis://localhost:6379/0 ping
```

#### 2. **Cache Data Not Found**
```python
# Check if data exists
keys = cache.redis_client.keys("cache:stock_data:*")
print(f"Found keys: {keys}")

# Check TTL
ttl = cache.redis_client.ttl("cache:stock_data:key")
print(f"TTL: {ttl} seconds")
```

#### 3. **Memory Issues**
```bash
# Check Redis memory usage
redis-cli info memory

# Clear old cache data
redis-cli flushdb

# Adjust TTL settings in deployment_config.py
```

### **Performance Optimization**

#### 1. **Adjust TTL Settings**
```python
# Shorter TTL for frequently changing data
cache.set('live_data', data, ttl_seconds=30)

# Longer TTL for stable data
cache.set('sector_data', data, ttl_seconds=3600)
```

#### 2. **Enable Compression**
```python
# Compression is enabled by default
# Large datasets (>1KB) are automatically compressed
# Monitor compression savings in stats
```

#### 3. **LRU Capacity**
```python
# Adjust LRU capacity based on available memory
# Development: 128 items
# Staging: 256 items  
# Production: 512 items
```

## 🔄 Rollback Plan

If you need to rollback to local caching:

### **1. Restore Configuration Backup**
```bash
# Find backup directory
ls -la config_backup_*

# Restore files
cp config_backup_YYYYMMDD_HHMMSS/deployment_config.py .
cp config_backup_YYYYMMDD_HHMMSS/.env .
```

### **2. Revert Code Changes**
```python
# Import old cache managers
from redis_cache_manager import get_redis_cache_manager
# Redis image manager removed - charts are now generated in-memory
```

### **3. Restart Services**
```bash
# Restart your application services
python start_all_services.py
```

## 📊 Migration Checklist

- [ ] Redis is running and accessible
- [ ] Configuration updated (`update_cache_config.py`)
- [ ] Data migrated (`migrate_to_redis_cache.py`)
- [ ] Services updated to use unified cache
- [ ] Cache functionality verified
- [ ] Performance monitoring enabled
- [ ] Backup created (if needed)
- [ ] Team notified of changes

## 🆘 Support

### **Getting Help**
1. Check the troubleshooting section above
2. Review Redis logs: `redis-cli info server`
3. Check application logs for cache errors
4. Verify Redis connection: `redis-cli ping`

### **Useful Commands**
```bash
# Check Redis status
redis-cli info

# Monitor cache operations
redis-cli monitor

# Check cache keys
redis-cli keys "cache:*"

# Clear all cache (emergency)
redis-cli flushdb
```

## 🎉 Migration Complete!

After successful migration, you'll have:
- ✅ **Centralized Redis caching** for all data types
- ✅ **Automatic cleanup** and expiration
- ✅ **Better performance** and scalability
- ✅ **Comprehensive monitoring** and statistics
- ✅ **Market-aware caching** strategies

Your system is now using a modern, scalable caching architecture that will improve performance and reduce maintenance overhead!

---

**Generated on**: {{ datetime.now().isoformat() }}  
**Version**: 1.0  
**Compatibility**: StockAnalyzer Pro v3.0+
