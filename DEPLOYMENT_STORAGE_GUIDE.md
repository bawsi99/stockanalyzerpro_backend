# Deployment Storage Guide

## Overview

This guide provides comprehensive information about storage configuration for different deployment environments. The application uses multiple storage paths that need to be properly configured for production deployment.

## Current Storage Paths Analysis

### 📁 **Storage Paths Identified**

| Storage Type | Development Path | Production Path | Purpose |
|--------------|------------------|-----------------|---------|
| **Charts** | `./output/charts/` | `/app/data/charts/` | Temporary chart images for LLM analysis |
| **Analysis Results** | `./output/` | `/app/data/analysis/` | Analysis results and JSON files |
| **Datasets** | `./analysis_datasets/` | `/app/data/datasets/` | Sector data and analysis datasets |
| **Enhanced Data** | `./enhanced_sector_data/` | `/app/data/enhanced_sector_data/` | Filtered equity stocks, sector performance |
| **Logs** | `./logs/` | `/app/data/logs/` | Application logs |
| **Cache** | `./cache/` | `/app/data/cache/` | Application cache |

### 🚨 **Deployment Issues with Current Paths**

1. **Relative Paths**: `./output/` is relative to working directory
2. **Container Volatility**: Docker containers lose data on restart
3. **Permission Issues**: Container users may lack write permissions
4. **Scaling Issues**: Multiple instances can't share local paths
5. **Backup Challenges**: Local storage is harder to backup

## Storage Configuration

### Environment Variables

| Variable | Description | Development | Staging | Production |
|----------|-------------|-------------|---------|------------|
| `ENVIRONMENT` | Deployment environment | `development` | `staging` | `production` |
| `STORAGE_TYPE` | Storage backend type | `local` | `local` | `local/cloud` |
| `STORAGE_BASE_PATH` | Base storage directory | `./output` | `/app/data` | `/app/data` |
| `STORAGE_CHARTS_PATH` | Charts storage path | `./output/charts` | `/app/data/charts` | `/app/data/charts` |
| `STORAGE_ANALYSIS_PATH` | Analysis results path | `./output` | `/app/data/analysis` | `/app/data/analysis` |
| `STORAGE_DATASETS_PATH` | Datasets path | `./analysis_datasets` | `/app/data/datasets` | `/app/data/datasets` |
| `STORAGE_ENHANCED_DATA_PATH` | Enhanced data path | `./enhanced_sector_data` | `/app/data/enhanced_sector_data` | `/app/data/enhanced_sector_data` |
| `STORAGE_LOGS_PATH` | Logs path | `./logs` | `/app/data/logs` | `/app/data/logs` |
| `STORAGE_CACHE_PATH` | Cache path | `./cache` | `/app/data/cache` | `/app/data/cache` |

## Deployment Configurations

### 1. Docker Deployment

#### Dockerfile
```dockerfile
# Set environment variables
ENV ENVIRONMENT=production
ENV STORAGE_TYPE=local
ENV STORAGE_BASE_PATH=/app/data
ENV STORAGE_CHARTS_PATH=/app/data/charts
ENV STORAGE_ANALYSIS_PATH=/app/data/analysis
ENV STORAGE_DATASETS_PATH=/app/data/datasets
ENV STORAGE_ENHANCED_DATA_PATH=/app/data/enhanced_sector_data
ENV STORAGE_LOGS_PATH=/app/data/logs
ENV STORAGE_CACHE_PATH=/app/data/cache

# Chart management settings
ENV CHART_MAX_AGE_HOURS=6
ENV CHART_MAX_SIZE_MB=200
ENV CHART_CLEANUP_INTERVAL_MINUTES=15
ENV CHART_ENABLE_CLEANUP=true

# Create data directory
RUN mkdir -p /app/data/{charts,analysis,datasets,enhanced_sector_data,logs,cache}
RUN chown -R app:app /app/data
```

#### Docker Compose
```yaml
version: '3.8'
services:
  analysis-service:
    build: .
    environment:
      - ENVIRONMENT=production
      - STORAGE_TYPE=local
      - STORAGE_BASE_PATH=/app/data
      - STORAGE_CHARTS_PATH=/app/data/charts
      - STORAGE_ANALYSIS_PATH=/app/data/analysis
      - STORAGE_DATASETS_PATH=/app/data/datasets
      - STORAGE_ENHANCED_DATA_PATH=/app/data/enhanced_sector_data
      - STORAGE_LOGS_PATH=/app/data/logs
      - STORAGE_CACHE_PATH=/app/data/cache
      - CHART_MAX_AGE_HOURS=6
      - CHART_MAX_SIZE_MB=200
      - CHART_CLEANUP_INTERVAL_MINUTES=15
      - CHART_ENABLE_CLEANUP=true
    volumes:
      - ./data:/app/data
      - ./logs:/app/data/logs
    user: "1000:1000"  # Use non-root user
```

### 2. Kubernetes Deployment

#### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: storage-config
data:
  ENVIRONMENT: "production"
  STORAGE_TYPE: "local"
  STORAGE_BASE_PATH: "/app/data"
  STORAGE_CHARTS_PATH: "/app/data/charts"
  STORAGE_ANALYSIS_PATH: "/app/data/analysis"
  STORAGE_DATASETS_PATH: "/app/data/datasets"
  STORAGE_ENHANCED_DATA_PATH: "/app/data/enhanced_sector_data"
  STORAGE_LOGS_PATH: "/app/data/logs"
  STORAGE_CACHE_PATH: "/app/data/cache"
  CHART_MAX_AGE_HOURS: "6"
  CHART_MAX_SIZE_MB: "200"
  CHART_CLEANUP_INTERVAL_MINUTES: "15"
  CHART_ENABLE_CLEANUP: "true"
```

#### Persistent Volume Claim
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: analysis-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
```

#### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: analysis-service
spec:
  replicas: 1
  selector:
    matchLabels:
      app: analysis-service
  template:
    metadata:
      labels:
        app: analysis-service
    spec:
      containers:
      - name: analysis-service
        image: analysis-service:latest
        envFrom:
        - configMapRef:
            name: storage-config
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/data/logs
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: analysis-data-pvc
      - name: logs-volume
        emptyDir: {}
```

### 3. Cloud Deployment (AWS/GCP/Azure)

#### Environment Variables
```bash
# Set in your deployment platform
ENVIRONMENT=production
STORAGE_TYPE=local
STORAGE_BASE_PATH=/app/data
STORAGE_CHARTS_PATH=/app/data/charts
STORAGE_ANALYSIS_PATH=/app/data/analysis
STORAGE_DATASETS_PATH=/app/data/datasets
STORAGE_ENHANCED_DATA_PATH=/app/data/enhanced_sector_data
STORAGE_LOGS_PATH=/app/data/logs
STORAGE_CACHE_PATH=/app/data/cache
CHART_MAX_AGE_HOURS=6
CHART_MAX_SIZE_MB=200
CHART_CLEANUP_INTERVAL_MINUTES=15
CHART_ENABLE_CLEANUP=true
```

#### Cloud Storage Integration
For persistent storage, consider mounting cloud storage:

**AWS EFS:**
```bash
# Mount EFS to /app/data
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 fs-12345678.efs.us-west-2.amazonaws.com:/ /app/data
```

**GCP Filestore:**
```bash
# Mount Filestore to /app/data
sudo mount -t nfs 10.0.0.1:/vol1 /app/data
```

**Azure Files:**
```bash
# Mount Azure Files to /app/data
sudo mount -t cifs //storageaccount.file.core.windows.net/share /app/data -o vers=3.0,credentials=/etc/smbcredentials/storageaccount.cred,dir_mode=0777,file_mode=0777,serverino
```

## Storage Monitoring

### API Endpoints

#### Storage Information
```bash
GET /storage/info
```
Returns comprehensive storage information including:
- Environment and storage type
- All storage paths
- Directory existence and sizes
- Storage usage statistics

#### Storage Recommendations
```bash
GET /storage/recommendations
```
Returns environment-specific storage recommendations.

### Monitoring Scripts

#### Storage Health Check
```bash
#!/bin/bash
# Check storage health
curl -s http://your-service/storage/info | jq '.storage_info.directory_status'
```

#### Storage Usage Alert
```bash
#!/bin/bash
# Alert if storage usage is high
USAGE=$(curl -s http://your-service/storage/info | jq -r '.storage_info.directory_status.charts_path.size_mb')
if (( $(echo "$USAGE > 150" | bc -l) )); then
    echo "WARNING: Chart storage usage is high: ${USAGE}MB"
fi
```

## Best Practices

### 1. Production Storage Setup

#### Directory Structure
```bash
/app/data/
├── charts/           # Temporary chart images (auto-cleanup)
├── analysis/         # Analysis results (persistent)
├── datasets/         # Sector datasets (persistent)
├── enhanced_sector_data/  # Enhanced data (persistent)
├── logs/            # Application logs (rotated)
└── cache/           # Application cache (auto-cleanup)
```

#### Permissions
```bash
# Set proper permissions
chown -R app:app /app/data
chmod -R 755 /app/data
chmod -R 777 /app/data/charts  # Allow chart cleanup
```

### 2. Backup Strategy

#### Critical Data (Backup Required)
- `/app/data/analysis/` - Analysis results
- `/app/data/datasets/` - Sector datasets
- `/app/data/enhanced_sector_data/` - Enhanced data

#### Temporary Data (No Backup Needed)
- `/app/data/charts/` - Auto-cleanup charts
- `/app/data/cache/` - Application cache
- `/app/data/logs/` - Application logs

### 3. Scaling Considerations

#### Single Instance
- Use local persistent volumes
- All storage paths on same volume

#### Multiple Instances
- Use shared storage (NFS, cloud storage)
- Implement distributed locking for cleanup
- Consider separate storage for each instance

### 4. Security Considerations

#### File Permissions
- Restrict access to storage directories
- Use dedicated user for application
- Implement proper file ownership

#### Data Protection
- Encrypt sensitive data at rest
- Implement access controls
- Regular security audits

## Troubleshooting

### Common Issues

#### Permission Denied
```bash
# Check permissions
ls -la /app/data/
# Fix permissions
chown -R app:app /app/data
chmod -R 755 /app/data
```

#### Storage Full
```bash
# Check storage usage
df -h /app/data
# Trigger manual cleanup
curl -X POST http://your-service/charts/cleanup
```

#### Directory Not Found
```bash
# Check if directories exist
ls -la /app/data/
# Create missing directories
mkdir -p /app/data/{charts,analysis,datasets,enhanced_sector_data,logs,cache}
```

### Debug Commands

#### Check Storage Configuration
```bash
curl -s http://your-service/storage/info | jq '.'
```

#### Check Chart Storage
```bash
curl -s http://your-service/charts/storage/stats | jq '.'
```

#### Manual Cleanup
```bash
curl -X POST http://your-service/charts/cleanup
```

## Migration Guide

### From Development to Production

1. **Update Environment Variables**
   ```bash
   ENVIRONMENT=production
   STORAGE_BASE_PATH=/app/data
   ```

2. **Create Production Directories**
   ```bash
   mkdir -p /app/data/{charts,analysis,datasets,enhanced_sector_data,logs,cache}
   ```

3. **Set Permissions**
   ```bash
   chown -R app:app /app/data
   chmod -R 755 /app/data
   ```

4. **Deploy Application**
   ```bash
   docker-compose up -d
   ```

5. **Verify Storage**
   ```bash
   curl http://your-service/storage/info
   ```

### Data Migration

If migrating existing data:

```bash
# Copy development data to production
cp -r ./output/* /app/data/analysis/
cp -r ./analysis_datasets/* /app/data/datasets/
cp -r ./enhanced_sector_data/* /app/data/enhanced_sector_data/
```

## Conclusion

Proper storage configuration is crucial for production deployment. The provided configuration ensures:

- **Persistent Data**: Critical data survives container restarts
- **Automatic Cleanup**: Temporary charts are automatically removed
- **Scalability**: Storage can be shared across multiple instances
- **Monitoring**: Comprehensive storage monitoring and alerting
- **Security**: Proper permissions and access controls

Follow the environment-specific configurations and best practices to ensure reliable storage management in production. 