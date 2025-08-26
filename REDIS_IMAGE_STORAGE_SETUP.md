# Redis Image Storage Setup Guide

This guide explains how to set up and configure Redis for storing generated chart images instead of using the local file system.

## Overview

The Redis Image Manager provides:
- **Centralized Storage**: Store images in Redis instead of local files
- **Automatic Cleanup**: Configurable expiration and size-based cleanup
- **Scalability**: Support for both local and cloud Redis instances
- **Fallback Support**: Graceful fallback to file-based storage if Redis is unavailable

## Configuration

### Environment Variables

Add these variables to your `.env` file:

```bash
# Redis Connection
REDIS_URL=redis://localhost:6379/0

# Redis Image Manager Settings
REDIS_IMAGE_MAX_AGE_HOURS=24
REDIS_IMAGE_MAX_SIZE_MB=1000
REDIS_IMAGE_CLEANUP_INTERVAL_MINUTES=60
REDIS_IMAGE_ENABLE_CLEANUP=true
REDIS_IMAGE_QUALITY=85
REDIS_IMAGE_FORMAT=PNG
```

### Redis URL Examples

#### Local Redis
```bash
REDIS_URL=redis://localhost:6379/0
```

#### Redis Cloud
```bash
REDIS_URL=redis://username:password@host:port
```

#### AWS ElastiCache
```bash
REDIS_URL=redis://your-elasticache-endpoint:6379/0
```

#### Google Cloud Memorystore
```bash
REDIS_URL=redis://your-memorystore-ip:6379/0
```

## Installation

### 1. Install Redis

#### Local Installation (macOS)
```bash
brew install redis
brew services start redis
```

#### Local Installation (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### Docker
```bash
docker run -d --name redis -p 6379:6379 redis:alpine
```

### 2. Install Python Dependencies

The Redis dependency is already included in `requirements.txt`:
```bash
pip install redis==5.2.1
```

## Usage

### Automatic Integration

The Redis Image Manager is automatically integrated into the analysis service. When you run stock analysis:

1. **Image Generation**: Charts are generated as usual
2. **Redis Storage**: Images are automatically stored in Redis with metadata
3. **Base64 Response**: Images are converted to base64 for immediate frontend display
4. **Automatic Cleanup**: Old images are automatically cleaned up based on configuration

### Manual Management

#### Check Redis Image Stats
```bash
curl http://localhost:8001/redis/images/stats
```

#### Manual Cleanup
```bash
curl -X POST http://localhost:8001/redis/images/cleanup
```

#### Get Images for a Symbol
```bash
curl http://localhost:8001/redis/images/RELIANCE
```

#### Cleanup Images for a Symbol
```bash
curl -X DELETE http://localhost:8001/redis/images/RELIANCE
```

#### Clear All Images
```bash
curl -X DELETE http://localhost:8001/redis/images
```

## Configuration by Environment

### Development
```bash
REDIS_IMAGE_MAX_AGE_HOURS=24
REDIS_IMAGE_MAX_SIZE_MB=1000
REDIS_IMAGE_CLEANUP_INTERVAL_MINUTES=60
REDIS_IMAGE_QUALITY=85
REDIS_IMAGE_FORMAT=PNG
```

### Staging
```bash
REDIS_IMAGE_MAX_AGE_HOURS=12
REDIS_IMAGE_MAX_SIZE_MB=500
REDIS_IMAGE_CLEANUP_INTERVAL_MINUTES=30
REDIS_IMAGE_QUALITY=85
REDIS_IMAGE_FORMAT=PNG
```

### Production
```bash
REDIS_IMAGE_MAX_AGE_HOURS=6
REDIS_IMAGE_MAX_SIZE_MB=200
REDIS_IMAGE_CLEANUP_INTERVAL_MINUTES=15
REDIS_IMAGE_QUALITY=80
REDIS_IMAGE_FORMAT=JPEG
```

## Monitoring

### Storage Statistics
The system provides comprehensive storage statistics:

```json
{
  "success": true,
  "stats": {
    "total_images": 150,
    "total_size_mb": 45.2,
    "oldest_image_age_hours": 23.5,
    "newest_image_age_hours": 0.1,
    "average_age_hours": 12.3,
    "max_age_hours": 24,
    "max_size_mb": 1000,
    "redis_url": "redis://localhost:6379/0"
  }
}
```

### Combined Stats
Get both file-based and Redis storage stats:
```bash
curl http://localhost:8001/charts/storage/stats
```

## Troubleshooting

### Redis Connection Issues

1. **Check Redis Status**
   ```bash
   redis-cli ping
   ```

2. **Check Redis Logs**
   ```bash
   # macOS
   tail -f /usr/local/var/log/redis.log
   
   # Ubuntu
   sudo journalctl -u redis-server -f
   ```

3. **Test Connection**
   ```bash
   redis-cli
   > SET test "Hello Redis"
   > GET test
   ```

### Fallback Behavior

If Redis is unavailable, the system automatically falls back to file-based storage:
- Images are still generated and served
- Cleanup continues to work with local files
- No functionality is lost

### Performance Optimization

1. **Image Quality**: Lower quality for smaller file sizes
   ```bash
   REDIS_IMAGE_QUALITY=80  # Instead of 85
   ```

2. **Image Format**: Use JPEG for smaller files
   ```bash
   REDIS_IMAGE_FORMAT=JPEG  # Instead of PNG
   ```

3. **Cleanup Frequency**: More frequent cleanup for production
   ```bash
   REDIS_IMAGE_CLEANUP_INTERVAL_MINUTES=15  # Instead of 60
   ```

## Security Considerations

1. **Redis Authentication**: Use password authentication for production
   ```bash
   REDIS_URL=redis://:password@localhost:6379/0
   ```

2. **Network Security**: Use SSL/TLS for cloud Redis instances
   ```bash
   REDIS_URL=rediss://username:password@host:port
   ```

3. **Access Control**: Restrict Redis access to application servers only

## Migration from File Storage

The system automatically handles migration:
1. New images are stored in Redis
2. Old file-based images continue to work
3. File cleanup continues to run
4. No manual migration required

## API Endpoints

### Redis Image Management
- `GET /redis/images/stats` - Get storage statistics
- `POST /redis/images/cleanup` - Manual cleanup
- `GET /redis/images/{symbol}` - Get images for symbol
- `DELETE /redis/images/{symbol}` - Cleanup images for symbol
- `DELETE /redis/images` - Clear all images

### Combined Chart Management
- `GET /charts/storage/stats` - Get both file and Redis stats
- `POST /charts/cleanup` - Cleanup both file and Redis storage
- `DELETE /charts/{symbol}/{interval}` - Cleanup both for symbol/interval

## Example .env Configuration

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Redis Image Manager
REDIS_IMAGE_MAX_AGE_HOURS=24
REDIS_IMAGE_MAX_SIZE_MB=1000
REDIS_IMAGE_CLEANUP_INTERVAL_MINUTES=60
REDIS_IMAGE_ENABLE_CLEANUP=true
REDIS_IMAGE_QUALITY=85
REDIS_IMAGE_FORMAT=PNG

# Environment
ENVIRONMENT=development

# Other configurations...
ZERODHA_API_KEY=your_api_key
ZERODHA_ACCESS_TOKEN=your_access_token
```
