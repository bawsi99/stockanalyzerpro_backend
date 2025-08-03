# Chart Memory Issue - Solution Summary

## Problem Identified

### Current Chart Storage Issue
- **Location**: Charts stored in `./output/charts/{symbol}_{interval}/`
- **Volume**: 4 charts per analysis (~300-800KB each)
- **Accumulation**: Charts remain indefinitely without cleanup
- **Impact**: Disk space exhaustion in production deployments

### Analysis of Current System
```bash
# Current output directory structure
./output/charts/
├── RELIANCE_1day/
│   ├── technical_overview.png
│   ├── pattern_analysis.png
│   ├── volume_analysis.png
│   └── mtf_comparison.png
├── INFY_1day/
│   └── ...
└── ... (accumulating indefinitely)
```

## Solution Implemented

### 1. ChartManager Class (`chart_manager.py`)
**Features:**
- Automatic lifecycle management
- Age-based cleanup (configurable retention)
- Size-based cleanup (configurable limits)
- Background cleanup thread
- Storage statistics and monitoring

**Key Methods:**
```python
# Initialize with custom settings
chart_manager = ChartManager(
    max_age_hours=24,
    max_total_size_mb=1000,
    cleanup_interval_minutes=60
)

# Create chart directories
chart_dir = chart_manager.create_chart_directory(symbol, interval)

# Automatic cleanup
stats = chart_manager.cleanup_old_charts()

# Storage monitoring
stats = chart_manager.get_storage_stats()
```

### 2. Deployment Configuration (`deployment_config.py`)
**Environment-Specific Settings:**

| Environment | Max Age | Max Size | Cleanup Interval |
|-------------|---------|----------|------------------|
| Development | 24h     | 1000MB   | 60min            |
| Staging     | 12h     | 500MB    | 30min            |
| Production  | 6h      | 200MB    | 15min            |

**Environment Variables:**
```bash
ENVIRONMENT=production
CHART_MAX_AGE_HOURS=6
CHART_MAX_SIZE_MB=200
CHART_CLEANUP_INTERVAL_MINUTES=15
CHART_ENABLE_CLEANUP=true
```

### 3. API Integration (`analysis_service.py`)
**Updated Chart Generation:**
```python
# Before: Manual directory creation
output_dir = f"./output/charts/{symbol}_{interval}"
os.makedirs(output_dir, exist_ok=True)

# After: Managed directory creation
chart_manager = get_chart_manager()
chart_dir = chart_manager.create_chart_directory(symbol, interval)
output_dir = str(chart_dir)
```

**New API Endpoints:**
```bash
GET /charts/storage/stats          # Storage statistics
POST /charts/cleanup              # Manual cleanup trigger
DELETE /charts/{symbol}/{interval} # Cleanup specific charts
DELETE /charts/all                # Cleanup all charts
```

## Implementation Details

### Automatic Cleanup Process
1. **Background Thread**: Runs cleanup every configured interval
2. **Age Check**: Removes files older than max_age_hours
3. **Size Check**: Removes oldest files when size limit exceeded
4. **Empty Directory Cleanup**: Removes empty directories
5. **Error Handling**: Logs errors but continues operation

### Storage Monitoring
```python
# Example storage stats response
{
    "total_files": 45,
    "total_size_mb": 23.5,
    "oldest_file_age_hours": 18.2,
    "newest_file_age_hours": 0.1,
    "max_age_hours": 24,
    "max_size_mb": 1000
}
```

### Safety Features
- **Non-destructive**: Only removes chart files, not analysis data
- **Database preservation**: Analysis results stored separately
- **Error recovery**: Continues operation even if cleanup fails
- **Logging**: Comprehensive logging of all operations

## Deployment Recommendations

### Production Environment
```bash
# Aggressive cleanup for production
ENVIRONMENT=production
CHART_MAX_AGE_HOURS=6
CHART_MAX_SIZE_MB=200
CHART_CLEANUP_INTERVAL_MINUTES=15
CHART_ENABLE_CLEANUP=true
```

### Docker Deployment
```dockerfile
ENV ENVIRONMENT=production
ENV CHART_MAX_AGE_HOURS=6
ENV CHART_MAX_SIZE_MB=200
ENV CHART_CLEANUP_INTERVAL_MINUTES=15
ENV CHART_ENABLE_CLEANUP=true
```

### Kubernetes Deployment
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

## Testing

### Test Script (`test_chart_manager.py`)
**Test Coverage:**
- Basic functionality (directory creation, file management)
- Cleanup functionality (age and size-based)
- Deployment configuration
- Global chart manager instance
- API endpoint simulation

**Run Tests:**
```bash
cd backend
python test_chart_manager.py
```

## Migration Strategy

### Phase 1: Implementation
1. ✅ Deploy ChartManager class
2. ✅ Update analysis service integration
3. ✅ Add API endpoints
4. ✅ Create deployment configuration

### Phase 2: Deployment
1. Set environment variables for target environment
2. Deploy updated service
3. Monitor initial cleanup operation
4. Verify storage usage reduction

### Phase 3: Monitoring
1. Set up storage monitoring alerts
2. Monitor cleanup success rates
3. Track chart generation frequency
4. Optimize settings based on usage patterns

## Benefits

### Immediate Benefits
- **Disk Space Management**: Automatic cleanup prevents space exhaustion
- **Performance**: Reduced disk I/O from fewer files
- **Reliability**: No manual cleanup required
- **Monitoring**: Real-time storage statistics

### Long-term Benefits
- **Scalability**: Handles increased chart generation
- **Cost Reduction**: Lower storage requirements
- **Maintenance**: Reduced operational overhead
- **Compliance**: Configurable retention policies

## Monitoring and Alerts

### Key Metrics to Monitor
- Chart storage usage (MB)
- Number of chart files
- Cleanup success rate
- Oldest file age
- Cleanup errors

### Alert Thresholds
- **Warning**: Storage > 80% of max size
- **Critical**: Storage > 95% of max size
- **Cleanup failures**: > 5 errors in cleanup cycle

### Log Patterns
```
Chart cleanup completed: {'files_removed': 5, 'bytes_freed': 2048000}
Chart manager initialized: ./output/charts, max_age=6h, max_size=200MB
Error during chart cleanup: Permission denied
```

## Future Enhancements

### Potential Improvements
1. **Cloud Storage Integration**: Store charts in S3/GCS for persistence
2. **Compression**: Compress charts to reduce storage
3. **Caching**: Cache frequently accessed charts
4. **Analytics**: Track chart usage patterns
5. **Retention Policies**: User-specific retention settings

### Advanced Features
1. **Chart Versioning**: Keep multiple versions of charts
2. **Selective Cleanup**: Keep charts for specific symbols longer
3. **Backup Integration**: Backup important charts before cleanup
4. **Performance Optimization**: Parallel cleanup for large directories

## Conclusion

The implemented ChartManager solution provides a comprehensive approach to managing chart storage in deployment environments. It addresses the immediate memory issue while providing flexibility for different deployment scenarios and future enhancements.

**Key Success Metrics:**
- Reduced disk space usage
- Automatic cleanup without manual intervention
- Configurable retention policies
- Comprehensive monitoring and alerting
- Zero impact on analysis functionality 