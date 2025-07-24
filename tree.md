# Call Tree and Architecture Overview

## System Architecture

The system is built with a **split backend architecture** consisting of two independent services:

### 1. Data Service (Port 8000)
**Purpose**: Handles all data fetching, WebSocket connections, and real-time data streaming.

**Core Components**:
- `data_service.py` - Main FastAPI application for data operations
- `zerodha_ws_client.py` - WebSocket client for real-time data
- `enhanced_data_service.py` - Cost-efficient data fetching
- `market_hours_manager.py` - Market hours and optimization
- `zerodha_client.py` - Zerodha API client

### 2. Analysis Service (Port 8001)
**Purpose**: Handles all analysis, AI processing, and chart generation.

**Core Components**:
- `analysis_service.py` - Main FastAPI application for analysis
- `agent_capabilities.py` - Analysis orchestrator
- `technical_indicators.py` - Technical analysis calculations
- `sector_benchmarking.py` - Sector analysis and benchmarking
- `gemini/` - AI analysis modules

---

## Service Orchestration

### Service Management (`run_services.py`)
- **start_services()**: Starts both data and analysis services
- **monitor_processes()**: Monitors service health and output
- **stop_processes()**: Graceful shutdown of all services

### Service Startup Scripts
- `start_data_service.py` - Data service startup (Port 8000)
- `start_analysis_service.py` - Analysis service startup (Port 8001)

---

## Data Service Architecture (Port 8000)

### Main Application (`data_service.py`)

#### FastAPI Endpoints
- **GET /health** - Service health check
- **GET /ws/health** - WebSocket health status
- **GET /ws/test** - WebSocket test endpoint
- **GET /ws/connections** - Active WebSocket connections
- **GET /stock/{symbol}/history** - Historical OHLCV data
- **POST /data/optimized** - Optimized data fetching
- **GET /stock/{symbol}/info** - Stock information
- **GET /market/status** - Market status
- **GET /mapping/token-to-symbol** - Token to symbol mapping
- **GET /mapping/symbol-to-token** - Symbol to token mapping
- **WebSocket /ws/stream** - Real-time data streaming

#### Core Classes
- **LiveDataPubSub**: Real-time data pub/sub system
- **AlertManager**: Alert management and evaluation
- **HistoricalDataRequest**: Data request models
- **OptimizedDataRequest**: Optimized data request models

### WebSocket Client (`zerodha_ws_client.py`)

#### ZerodhaWSClient
- **connect()**: Establishes WebSocket connection
- **subscribe()**: Subscribe to instrument tokens
- **on_ticks()**: Process incoming tick data
- **parse_binary_message()**: Parse binary market data
- **get_latest_tick()**: Get latest tick for token
- **get_latest_candle()**: Get latest candle for timeframe

#### CandleAggregator
- **process_tick()**: Aggregate ticks into candles
- **get_latest_candle()**: Retrieve latest candle data
- **register_callback()**: Register candle callbacks

### Enhanced Data Service (`enhanced_data_service.py`)

#### EnhancedDataService
- **get_optimal_data()**: Get optimal data based on market status
- **get_market_status()**: Current market status
- **get_optimization_stats()**: Optimization statistics
- **get_cost_analysis()**: Cost analysis and recommendations

#### Data Models
- **DataRequest**: Data request with optimization parameters
- **DataResponse**: Data response with metadata

### Market Hours Manager (`market_hours_manager.py`)

#### MarketHoursManager
- **get_market_status()**: Current market status
- **is_market_open()**: Check if market is open
- **get_optimal_data_strategy()**: Optimal data fetching strategy
- **should_use_websocket()**: Determine WebSocket usage
- **estimate_data_cost()**: Estimate data fetching costs

#### Market Models
- **MarketStatus**: Market status enumeration
- **MarketSession**: Trading session configuration
- **MarketHours**: Market hours configuration

---

## Analysis Service Architecture (Port 8001)

### Main Application (`analysis_service.py`)

#### FastAPI Endpoints
- **GET /health** - Service health check
- **POST /analyze** - Comprehensive stock analysis
- **POST /analyze/enhanced** - Enhanced analysis with code execution
- **GET /stock/{symbol}/indicators** - Technical indicators
- **GET /patterns/{symbol}** - Pattern recognition
- **GET /charts/{symbol}** - Chart generation
- **GET /sector/list** - Sector information
- **POST /sector/benchmark** - Sector benchmarking
- **GET /sector/{sector_name}/stocks** - Sector stocks
- **GET /sector/{sector_name}/performance** - Sector performance
- **POST /sector/compare** - Compare multiple sectors

#### Request Models
- **AnalysisRequest**: Basic analysis request
- **EnhancedAnalysisRequest**: Enhanced analysis with code execution
- **SectorAnalysisRequest**: Sector analysis request
- **SectorComparisonRequest**: Sector comparison request
- **IndicatorsRequest**: Technical indicators request

### Analysis Orchestrator (`agent_capabilities.py`)

#### StockAnalysisOrchestrator
- **__init__**: Initialize data client, Gemini client, indicators, visualizer
- **authenticate()**: Authenticate with Zerodha API
- **retrieve_stock_data()**: Fetch historical/real-time stock data
- **calculate_indicators()**: Calculate all technical indicators
- **create_visualizations()**: Generate chart visualizations
- **orchestrate_llm_analysis()**: Coordinate AI analysis
- **analyze_with_ai()**: Perform AI-powered analysis
- **analyze_stock()**: Main analysis orchestration method
- **enhanced_analyze_stock()**: Enhanced analysis with code execution

#### AnalysisState (dataclass)
- **symbol, exchange**: Stock identification
- **indicators, patterns, analysis_results**: Cached analysis data
- **last_updated**: Timestamp for cache validation
- **is_valid()**: Check if cached data is still valid
- **update()**: Update state with new data

### Technical Analysis (`technical_indicators.py`)

#### TechnicalIndicators
- **calculate_all_indicators()**: Calculate all technical indicators
- **calculate_sma/ema/wma()**: Moving averages
- **calculate_macd()**: MACD indicator
- **calculate_rsi()**: RSI indicator
- **calculate_bollinger_bands()**: Bollinger Bands
- **calculate_adx()**: ADX indicator
- **calculate_enhanced_volume_analysis()**: Volume analysis
- **calculate_multi_timeframe_analysis()**: Multi-timeframe analysis
- **get_market_metrics()**: Market-specific metrics

#### DataCollector
- **collect_all_data()**: Collect and organize all technical data

#### IndianMarketMetricsProvider
- **get_sector_index_data()**: Retrieve sector index data
- **get_basic_market_metrics()**: Calculate basic market metrics
- **get_enhanced_market_metrics()**: Calculate enhanced market metrics
- **get_nifty_50_data()**: Get NIFTY 50 data
- **get_india_vix_data()**: Get India VIX data
- **calculate_beta()**: Calculate beta coefficient
- **calculate_correlation()**: Calculate correlation coefficient

### AI Analysis (`gemini/`)

#### GeminiClient (`gemini_client.py`)
- **build_indicators_summary()**: Create indicator summary for AI
- **analyze_stock()**: Comprehensive AI analysis (single source of truth)
- **analyze_comprehensive_overview()**: Analyze comprehensive chart overview
- **analyze_volume_comprehensive()**: Analyze volume patterns
- **analyze_reversal_patterns()**: Analyze reversal patterns
- **analyze_continuation_levels()**: Analyze continuation patterns
- **analyze_stock_with_enhanced_calculations()**: Enhanced analysis with code execution

#### Supporting Modules
- **gemini_core.py**: Core Gemini API integration
- **prompt_manager.py**: Prompt management and formatting
- **image_utils.py**: Image processing utilities
- **error_utils.py**: Error handling utilities

### Sector Analysis (`sector_benchmarking.py`)

#### SectorBenchmarkingProvider
- **get_comprehensive_benchmarking()**: Comprehensive sector benchmarking
- **get_sector_rotation_analysis()**: Sector rotation analysis (cached)
- **get_sector_correlation_analysis()**: Sector correlation analysis
- **get_stock_specific_benchmarking()**: Stock-specific benchmarking
- **get_optimized_sector_rotation()**: Optimized sector rotation
- **get_comprehensive_sector_analysis()**: Comprehensive sector analysis
- **get_hybrid_stock_analysis()**: Hybrid stock analysis

#### Sector Classification
- **sector_classifier.py**: Basic sector classification
- **enhanced_sector_classifier.py**: Enhanced sector classification
- **sector_manager.py**: Sector management utilities

### Pattern Recognition (`patterns/`)

#### PatternRecognition (`recognition.py`)
- **identify_peaks_lows()**: Identify local peaks and lows
- **detect_divergence()**: Detect price-indicator divergences
- **detect_volume_anomalies()**: Detect unusual volume patterns
- **detect_double_top/bottom()**: Detect double top/bottom patterns
- **detect_triangle()**: Detect triangle patterns
- **detect_flag()**: Detect flag patterns
- **detect_head_and_shoulders()**: Detect head and shoulders patterns
- **detect_cup_and_handle()**: Detect cup and handle patterns
- **detect_wedge_patterns()**: Detect wedge patterns
- **detect_channel_patterns()**: Detect channel patterns
- **detect_candlestick_patterns()**: Detect candlestick patterns
- **backtest_pattern()**: Backtest pattern reliability

#### PatternVisualizer (`visualization.py`)
- **plot_price_with_peaks_lows()**: Visualize peaks and lows
- **plot_divergences()**: Visualize divergences
- **plot_double_tops_bottoms()**: Visualize double tops/bottoms
- **plot_triangles_flags()**: Visualize triangles and flags
- **plot_support_resistance()**: Visualize support and resistance

#### ChartVisualizer
- **plot_comparison_chart()**: Create comprehensive comparison charts
- **plot_volume_analysis()**: Create volume analysis charts
- **plot_pattern_charts()**: Create pattern-specific charts
- **plot_multi_timeframe_analysis()**: Multi-timeframe charts

---

## Performance and Caching

### Cache Manager (`cache_manager.py`)

#### CacheManager
- **get()**: Retrieve cached data
- **set()**: Store data in cache
- **clear()**: Clear all cached data
- **get_stats()**: Cache performance statistics
- **get_info()**: Cache information and metrics

#### Performance Monitoring
- **PerformanceMonitor**: Monitor operation performance
- **monitor_performance()**: Performance monitoring decorator
- **cached()**: Caching decorator with TTL

---

## Frontend Architecture

### Service Integration (`services/`)

#### ApiService (`api.ts`)
- **Data Service Endpoints** (Port 8000):
  - `getHistoricalData()`: Historical OHLCV data
  - `getStockInfo()`: Stock information
  - `getMarketStatus()`: Market status
  - `getOptimizedData()`: Optimized data fetching
  - WebSocket connection management

- **Analysis Service Endpoints** (Port 8001):
  - `analyzeStock()`: Comprehensive stock analysis
  - `enhancedAnalyzeStock()`: Enhanced analysis
  - `getIndicators()`: Technical indicators
  - `getPatterns()`: Pattern recognition
  - `getCharts()`: Chart generation
  - `getSectors()`: Sector information
  - `getSectorBenchmark()`: Sector benchmarking

#### Live Data Service (`liveDataService.ts`)
- **WebSocket Management**: Real-time data streaming
- **Data Processing**: Process incoming market data
- **Connection Management**: Handle WebSocket connections

### Configuration (`config.ts`)
- **DATA_SERVICE_URL**: Data service endpoint (Port 8000)
- **ANALYSIS_SERVICE_URL**: Analysis service endpoint (Port 8001)
- **WEBSOCKET_URL**: WebSocket endpoint
- **ENDPOINTS**: Service endpoint mappings

---

## Call Flow Architecture

### Data Service Flow
1. **WebSocket Connection** (`zerodha_ws_client.py`)
   ├─ connect() → subscribe() → on_ticks()
   ├─ parse_binary_message() → process_tick()
   └─ publish() → LiveDataPubSub

2. **Historical Data Request** (`data_service.py`)
   ├─ get_stock_history() → ZerodhaDataClient
   ├─ enhanced_data_service.get_optimal_data()
   └─ market_hours_manager.get_optimal_data_strategy()

3. **Real-time Streaming** (`data_service.py`)
   ├─ ws_stream() → authenticate_websocket()
   ├─ LiveDataPubSub.subscribe()
   └─ AlertManager.evaluate_alert()

### Analysis Service Flow
1. **Stock Analysis Request** (`analysis_service.py`)
   ├─ analyze() → StockAnalysisOrchestrator.analyze_stock()
   ├─ retrieve_stock_data() → Data Service (Port 8000)
   ├─ calculate_indicators() → TechnicalIndicators
   ├─ create_visualizations() → PatternVisualizer
   ├─ analyze_with_ai() → GeminiClient
   └─ sector_benchmarking() → SectorBenchmarkingProvider

2. **Enhanced Analysis** (`analysis_service.py`)
   ├─ enhanced_analyze() → StockAnalysisOrchestrator.enhanced_analyze_stock()
   ├─ enhanced_analyze_with_ai() → GeminiClient with code execution
   └─ mathematical validation and enhanced calculations

3. **Technical Indicators** (`analysis_service.py`)
   ├─ get_stock_indicators() → TechnicalIndicators.calculate_all_indicators()
   └─ CacheManager for performance optimization

4. **Pattern Recognition** (`analysis_service.py`)
   ├─ get_patterns() → PatternRecognition.detect_*()
   └─ PatternVisualizer for chart generation

### AI Analysis Flow
1. **GeminiClient.analyze_stock()**
   ├─ build_indicators_summary()
   ├─ analyze_comprehensive_overview()
   ├─ analyze_volume_comprehensive()
   ├─ analyze_reversal_patterns()
   └─ analyze_continuation_levels()

2. **Enhanced AI Analysis**
   ├─ analyze_stock_with_enhanced_calculations()
   ├─ code execution for mathematical validation
   └─ enhanced calculations and risk assessment

### Sector Analysis Flow
1. **SectorBenchmarkingProvider**
   ├─ get_comprehensive_benchmarking()
   ├─ get_sector_rotation_analysis()
   ├─ get_sector_correlation_analysis()
   └─ get_hybrid_stock_analysis()

2. **Sector Classification**
   ├─ sector_classifier.classify_stock()
   ├─ enhanced_sector_classifier.classify_stock()
   └─ sector_manager utilities

---

## Data Flow Architecture

### Input Data Sources
- **Zerodha API**: Historical OHLCV data
- **WebSocket Stream**: Real-time tick data
- **Sector Indices**: NIFTY sector indices
- **Market Data**: NIFTY 50, India VIX

### Processing Pipeline
1. **Data Retrieval**: Fetch historical/real-time data
2. **Market Hours Optimization**: Optimize based on market status
3. **Technical Analysis**: Calculate all indicators
4. **Pattern Recognition**: Detect chart patterns
5. **AI Analysis**: Gemini LLM processing
6. **Sector Analysis**: Sector benchmarking and context
7. **Visualization**: Generate charts and overlays
8. **Caching**: Performance optimization

### Output Data
- **AI Analysis**: Comprehensive analysis with confidence levels
- **Technical Indicators**: All calculated indicators and values
- **Pattern Recognition**: Detected patterns and reliability scores
- **Trading Strategies**: Recommendations and risk management
- **Sector Benchmarking**: Sector performance and correlation data
- **Chart Visualizations**: Generated chart images and overlays

---

## Service Communication

### Frontend Integration
- **Data Service** (Port 8000): Real-time data, historical data, WebSocket
- **Analysis Service** (Port 8001): Analysis, indicators, patterns, charts

### Service-to-Service Communication
- **HTTP Requests**: Cross-service API calls
- **Shared Data Models**: Consistent data structures
- **Error Handling**: Graceful service failure handling

### WebSocket Architecture
- **Real-time Streaming**: Live market data
- **Pub/Sub System**: Efficient multi-client streaming
- **Alert Management**: Real-time alert evaluation
- **Connection Management**: Health monitoring and recovery

---

## Performance Optimizations

### Caching Strategy
- **Technical Indicators**: 10-minute TTL with LRU eviction
- **Sector Data**: 15-minute TTL for performance data
- **Comprehensive Analysis**: 1-hour TTL for heavy computations
- **Market Data**: 5-minute TTL for real-time data

### Market Hours Optimization
- **Live Data**: During market hours for supported timeframes
- **Historical Data**: Outside market hours or for longer timeframes
- **Cost Optimization**: Minimize API calls and data costs
- **Cache Strategy**: Intelligent cache duration based on market status

### Code Execution Enhancement
- **Mathematical Validation**: Enhanced calculations with code execution
- **Risk Assessment**: Advanced risk metrics and stress testing
- **Scenario Analysis**: Multiple market scenario evaluations
- **Performance Monitoring**: Real-time performance tracking

---

## Security and Authentication

### JWT Authentication
- **Token Creation**: Secure JWT token generation
- **Token Verification**: JWT token validation
- **API Key Support**: Alternative API key authentication
- **WebSocket Auth**: Secure WebSocket connections

### Service Security
- **CORS Configuration**: Cross-origin resource sharing
- **Input Validation**: Request parameter validation
- **Error Handling**: Secure error responses
- **Rate Limiting**: API rate limiting (configurable)

---

## Monitoring and Health Checks

### Health Endpoints
- **Data Service**: `GET /health` - Service status and metrics
- **Analysis Service**: `GET /health` - Service status and metrics
- **WebSocket Health**: `GET /ws/health` - WebSocket status
- **Connection Monitoring**: `GET /ws/connections` - Active connections

### Performance Metrics
- **Cache Hit Rates**: Cache performance statistics
- **Response Times**: API response time monitoring
- **Error Rates**: Error tracking and reporting
- **Resource Usage**: Memory and CPU monitoring

---

## Deployment and Configuration

### Environment Variables
- **Zerodha API**: API key and access token
- **JWT Secret**: Authentication secret key
- **Service Ports**: Configurable service ports
- **Cache Settings**: Cache size and TTL configuration
- **Performance Settings**: Monitoring and optimization parameters

### Service Dependencies
- **Data Service**: zerodha_client, zerodha_ws_client, enhanced_data_service
- **Analysis Service**: agent_capabilities, technical_indicators, sector_benchmarking
- **Shared Dependencies**: cache_manager, market_hours_manager, patterns

### Development Tools
- **Service Startup Scripts**: Easy service management
- **Health Monitoring**: Real-time service health
- **Logging**: Comprehensive logging and debugging
- **Testing**: Unit and integration test support

This architecture provides a robust, scalable, and maintainable foundation for the trading platform with clear separation of concerns, optimized performance, and comprehensive analysis capabilities.