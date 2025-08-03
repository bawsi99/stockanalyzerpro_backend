# Chart Management Deployment Guide

## Overview

This guide explains how to manage chart storage and cleanup in different deployment environments. The chart management system automatically handles temporary chart files to prevent disk space issues in production deployments.

## Current Chart Storage Issue

### Problem
- Charts are generated for LLM analysis and stored in `./output/charts/`
- Each analysis creates 4 charts (~300-800KB each)
- Charts accumulate indefinitely without cleanup
- Can cause disk space issues in production

### Solution
- **ChartManager**: Automatic lifecycle management
- **Configurable retention**: Environment-specific settings
- **Background cleanup**: Automatic removal of old charts
- **Size limits**: Prevent disk space exhaustion

## Chart Management Features

### 1. Automatic Cleanup
- **Age-based**: Remove charts older than configured hours
- **Size-based**: Remove oldest charts when size limit exceeded
- **Background thread**: Runs cleanup automatically
- **Configurable intervals**: How often cleanup runs

### 2. Environment-Specific Settings

#### Development Environment
```bash
ENVIRONMENT=development
CHART_MAX_AGE_HOURS=24
CHART_MAX_SIZE_MB=1000
CHART_CLEANUP_INTERVAL_MINUTES=60
```

#### Staging Environment
```bash
ENVIRONMENT=staging
CHART_MAX_AGE_HOURS=12
CHART_MAX_SIZE_MB=500
CHART_CLEANUP_INTERVAL_MINUTES=30
```

#### Production Environment
```bash
ENVIRONMENT=production
CHART_MAX_AGE_HOURS=6
CHART_MAX_SIZE_MB=200
CHART_CLEANUP_INTERVAL_MINUTES=15
```

## Environment Variables

| Variable | Description | Default | Production |
|----------|-------------|---------|------------|
| `ENVIRONMENT` | Deployment environment | `development` | `production` |
| `CHART_MAX_AGE_HOURS` | Max age of charts (hours) | `24` | `6` |
| `CHART_MAX_SIZE_MB` | Max total chart size (MB) | `1000` | `200` |
| `CHART_CLEANUP_INTERVAL_MINUTES` | Cleanup frequency (minutes) | `60` | `15` |
| `CHART_OUTPUT_DIR` | Chart storage directory | `./output/charts` | `./output/charts` |
| `CHART_ENABLE_CLEANUP` | Enable automatic cleanup | `true` | `true` |

## API Endpoints

### Chart Storage Statistics
```bash
GET /charts/storage/stats
```
Returns current storage usage and statistics.

### Manual Cleanup
```bash
POST /charts/cleanup
```
Manually trigger chart cleanup.

### Cleanup Specific Charts
```bash
DELETE /charts/{symbol}/{interval}
```
Clean up charts for a specific symbol and interval.

### Cleanup All Charts
```bash
DELETE /charts/all
```
Clean up all charts (use with caution).

## Deployment Configurations

### Docker Deployment

#### Dockerfile
```dockerfile
# Add environment variables
ENV ENVIRONMENT=production
ENV CHART_MAX_AGE_HOURS=6
ENV CHART_MAX_SIZE_MB=200
ENV CHART_CLEANUP_INTERVAL_MINUTES=15
ENV CHART_ENABLE_CLEANUP=true
```

#### Docker Compose
```yaml
version: '3.8'
services:
  analysis-service:
    build: .
    environment:
      - ENVIRONMENT=production
      - CHART_MAX_AGE_HOURS=6
      - CHART_MAX_SIZE_MB=200
      - CHART_CLEANUP_INTERVAL_MINUTES=15
      - CHART_ENABLE_CLEANUP=true
    volumes:
      - ./output:/app/output
```

### Kubernetes Deployment

#### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: chart-management-config
data:
  ENVIRONMENT: "production"
  CHART_MAX_AGE_HOURS: "6"
  CHART_MAX_SIZE_MB: "200"
  CHART_CLEANUP_INTERVAL_MINUTES: "15"
  CHART_ENABLE_CLEANUP: "true"
```

#### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: analysis-service
spec:
  template:
    spec:
      containers:
      - name: analysis-service
        envFrom:
        - configMapRef:
            name: chart-management-config
        volumeMounts:
        - name: chart-storage
          mountPath: /app/output
      volumes:
      - name: chart-storage
        emptyDir: {}
```

### Cloud Deployment (AWS/GCP/Azure)

#### Environment Variables
```bash
# Set in your deployment platform
ENVIRONMENT=production
CHART_MAX_AGE_HOURS=6
CHART_MAX_SIZE_MB=200
CHART_CLEANUP_INTERVAL_MINUTES=15
CHART_ENABLE_CLEANUP=true
```

#### Persistent Storage
- Use cloud storage (S3, GCS, Azure Blob) for persistent chart storage
- Configure `CHART_OUTPUT_DIR` to point to cloud storage mount
- Ensure proper IAM permissions for read/write access

## Monitoring and Alerts

### Storage Monitoring
Monitor chart storage usage:
```bash
curl http://your-service/charts/storage/stats
```

### Alert Thresholds
- **Warning**: Storage > 80% of max size
- **Critical**: Storage > 95% of max size
- **Cleanup failures**: More than 5 errors in cleanup

### Log Monitoring
Monitor these log patterns:
- `Chart cleanup completed`
- `Error during chart cleanup`
- `Chart manager initialized`

## Best Practices

### 1. Production Settings
- Use aggressive cleanup (6 hours max age)
- Small storage limits (200MB)
- Frequent cleanup (15 minutes)
- Monitor storage usage closely

### 2. Development Settings
- Longer retention for debugging
- Larger storage limits
- Less frequent cleanup

### 3. Monitoring
- Set up alerts for storage usage
- Monitor cleanup success rates
- Track chart generation frequency

### 4. Backup Considerations
- Charts are temporary and don't need backup
- Analysis results are stored in database
- Focus backup on database and configuration

## Troubleshooting

### High Storage Usage
1. Check current storage stats:
   ```bash
   curl http://your-service/charts/storage/stats
   ```

2. Manually trigger cleanup:
   ```bash
   curl -X POST http://your-service/charts/cleanup
   ```

3. Check cleanup logs for errors

### Cleanup Not Working
1. Verify `CHART_ENABLE_CLEANUP=true`
2. Check cleanup thread is running
3. Verify file permissions on output directory
4. Check for disk space issues

### Performance Issues
1. Reduce cleanup frequency
2. Increase max age hours
3. Increase max size limits
4. Monitor system resources

## Migration from Current System

### Current State
- Charts stored in `./output/charts/`
- No automatic cleanup
- Manual cleanup required

### Migration Steps
1. Deploy new chart manager
2. Set appropriate environment variables
3. Monitor initial cleanup
4. Verify storage usage reduction

### Rollback Plan
1. Set `CHART_ENABLE_CLEANUP=false`
2. Revert to manual cleanup
3. Monitor storage usage

## Security Considerations

### File Permissions
- Ensure output directory has proper permissions
- Restrict access to chart files
- Use dedicated user for service

### Cleanup Safety
- Cleanup only affects chart files
- Analysis results are preserved in database
- No impact on application data

### Monitoring
- Log all cleanup activities
- Monitor for unusual cleanup patterns
- Alert on cleanup failures 