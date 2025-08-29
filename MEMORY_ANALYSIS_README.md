# Memory Analysis Tools for Stock Analyzer Pro

This directory contains comprehensive tools to analyze memory usage of the Stock Analyzer Pro service under different load conditions. These tools help estimate cloud deployment requirements and identify optimization opportunities.

## 🚀 Quick Start

### Option 1: Automated Analysis (Recommended)
```bash
# From the backend directory
./run_memory_analysis.sh
```

### Option 2: Manual Step-by-Step
```bash
# 1. Install dependencies
pip install -r memory_analysis_requirements.txt

# 2. Run comprehensive analysis
python memory_analysis_runner.py

# 3. Or run individual components
python memory_monitor.py --pid <PID> --duration 60
python load_tester.py --users 20 --requests 15
```

## 📁 Files Overview

### Core Tools
- **`memory_analysis_runner.py`** - Main orchestrator that runs complete analysis
- **`memory_monitor.py`** - Real-time memory monitoring and profiling
- **`load_tester.py`** - Simulates multiple concurrent users
- **`run_memory_analysis.sh`** - Automated shell script for easy execution

### Configuration & Dependencies
- **`memory_analysis_requirements.txt`** - Python package dependencies
- **`MEMORY_ANALYSIS_README.md`** - This documentation file

## 🔍 What the Analysis Measures

### 1. Baseline Memory Usage
- **Duration**: 30 seconds
- **Purpose**: Establish normal memory consumption without load
- **Metrics**: System memory %, Process memory (MB), Python GC stats

### 2. Load Testing Phase
- **Duration**: 120 seconds (2 minutes)
- **Purpose**: Stress test with 20 concurrent users making 15 requests each
- **Endpoints Tested**:
  - **Light**: `/health`, `/`, `/data/market-status` (60% of requests)
  - **Medium**: Stock history endpoints (30% of requests)
  - **Heavy**: Analysis endpoints, pattern detection (10% of requests)

### 3. Cooldown Monitoring
- **Duration**: 30 seconds
- **Purpose**: Observe memory recovery after load test
- **Metrics**: Recovery rate, memory cleanup patterns

## 📊 Analysis Output

### Real-time Monitoring
- Memory usage alerts at 80% (warning) and 90% (critical)
- Process-specific memory breakdown
- Python garbage collector statistics
- System-wide memory trends

### Comprehensive Reports
- **Memory Growth Patterns**: Baseline → Load → Cooldown transitions
- **Peak Usage Analysis**: Maximum memory consumption under stress
- **Recovery Patterns**: How well memory is freed after load
- **Performance Metrics**: Response times, success rates, throughput
- **Cloud Deployment Recommendations**: Memory requirements, optimization suggestions

### Generated Files
1. **`comprehensive_memory_analysis.json`** - Complete analysis results
2. **`load_test_during_analysis.json`** - Load testing performance data
3. **`memory_analysis.json`** - Raw memory monitoring data

## 🎯 Understanding the Results

### Memory Requirements for Cloud Deployment

#### Safe Deployment
- **Peak System Memory**: < 80%
- **Memory Recovery**: > 80% after load
- **Recommendation**: Deploy with 1.5x peak memory

#### Warning Level
- **Peak System Memory**: 80-90%
- **Memory Recovery**: 60-80%
- **Recommendation**: Monitor closely, consider optimization

#### Critical Level
- **Peak System Memory**: > 90%
- **Memory Recovery**: < 60%
- **Recommendation**: Optimize before deployment

### Service Component Memory Breakdown

The analysis provides detailed breakdowns of memory usage by:

1. **FastAPI Application**: Core service overhead
2. **Data Service**: Stock data processing, WebSocket connections
3. **Analysis Service**: ML models, pattern recognition, chart generation
4. **Zerodha Integration**: Market data streaming, API connections
5. **Database Connections**: Supabase connections, query caching
6. **Python Runtime**: Garbage collection, object memory

## 🔧 Customization Options

### Memory Monitoring
```bash
# Monitor specific process
python memory_monitor.py --pid 12345 --interval 0.5 --duration 300

# System-wide monitoring
python memory_monitor.py --interval 1.0 --duration 600
```

### Load Testing
```bash
# Light load test
python load_tester.py --users 5 --requests 10 --duration 60

# Heavy load test
python load_tester.py --users 50 --requests 30 --duration 300

# Custom endpoint testing
python load_tester.py --url "http://your-service.com" --users 25 --requests 20
```

### Analysis Configuration
```bash
# Quick analysis (1 minute total)
python memory_analysis_runner.py --baseline 15 --load-test 30 --cooldown 15

# Extended analysis (5 minutes total)
python memory_analysis_runner.py --baseline 60 --load-test 180 --cooldown 60

# High-load analysis
python memory_analysis_runner.py --users 50 --requests 25 --interval 0.25
```

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check if port 8000 is available
lsof -i :8000

# Check Python dependencies
pip list | grep -E "(fastapi|uvicorn|psutil|aiohttp)"
```

#### Memory Monitoring Errors
```bash
# Check psutil installation
python -c "import psutil; print(psutil.__version__)"

# Check process permissions
ps aux | grep start_with_cors_fix
```

#### Load Testing Failures
```bash
# Check service health
curl http://localhost:8000/health

# Check network connectivity
ping localhost
```

### Performance Tuning

#### For High-Load Scenarios
- Increase monitoring interval: `--interval 1.0`
- Reduce concurrent users: `--users 10`
- Monitor specific endpoints only

#### For Detailed Analysis
- Decrease monitoring interval: `--interval 0.1`
- Increase test duration: `--load-test 300`
- Add more baseline monitoring: `--baseline 60`

## 📈 Interpreting Results for Cloud Deployment

### Memory Requirements Calculation

#### Minimum Requirements
```
Min Memory = Peak Process Memory × 1.5 (safety margin)
```

#### Recommended Requirements
```
Recommended Memory = Peak Process Memory × 2.0 (comfortable margin)
```

#### Scaling Considerations
- **Horizontal Scaling**: If memory growth is linear with users
- **Vertical Scaling**: If memory growth is exponential
- **Memory Optimization**: If recovery patterns are poor

### Cloud Platform Recommendations

#### Render
- **Free Tier**: 512 MB RAM (for development only)
- **Starter**: 1 GB RAM (for light production)
- **Standard**: 2 GB RAM (recommended for production)
- **Pro**: 4 GB RAM (for high-traffic scenarios)

#### AWS/GCP/Azure
- **Small Instance**: 1-2 GB RAM
- **Medium Instance**: 4 GB RAM
- **Large Instance**: 8 GB RAM
- **Auto-scaling**: Based on memory thresholds

## 🔄 Continuous Monitoring

### Production Monitoring
```bash
# Monitor production service
python memory_monitor.py --pid <PROD_PID> --interval 5.0 --duration 86400

# Save to production monitoring file
python memory_monitor.py --pid <PROD_PID> --output "prod_memory_$(date +%Y%m%d).json"
```

### Alerting Setup
- Set up alerts for memory > 80%
- Monitor recovery patterns after traffic spikes
- Track memory growth trends over time

## 📚 Additional Resources

### Memory Optimization
- [Python Memory Management](https://docs.python.org/3/c-api/memory.html)
- [FastAPI Performance](https://fastapi.tiangolo.com/tutorial/performance/)
- [psutil Documentation](https://psutil.readthedocs.io/)

### Load Testing
- [aiohttp Best Practices](https://docs.aiohttp.org/en/stable/client_quickstart.html)
- [Async Python Performance](https://docs.python.org/3/library/asyncio.html)

### Cloud Deployment
- [Render Documentation](https://render.com/docs)
- [AWS Lambda Memory](https://docs.aws.amazon.com/lambda/latest/dg/configuration-memory.html)
- [Docker Memory Limits](https://docs.docker.com/config/containers/resource_constraints/)

## 🤝 Contributing

To improve these tools:
1. Test with different service configurations
2. Add new monitoring metrics
3. Improve load testing scenarios
4. Enhance analysis algorithms
5. Add visualization capabilities

## 📄 License

These tools are part of the Stock Analyzer Pro project and follow the same license terms.
