# Call Tree and Architecture Overview

## Main Orchestrator (agent_capabilities.py)

### StockAnalysisOrchestrator
- **__init__**: Sets up ZerodhaDataClient, GeminiClient, TechnicalIndicators, PatternVisualizer, DataCollector, and state cache.
- **authenticate**: Authenticates with Zerodha API.
- **retrieve_stock_data**: Fetches historical stock data with market timing awareness.
- **calculate_indicators**: Calculates all technical indicators using TechnicalIndicators class.
- **create_visualizations**: Generates chart visualizations using PatternVisualizer.
- **serialize_indicators**: Converts indicators to JSON-serializable format.
- **orchestrate_llm_analysis**: Coordinates AI analysis using GeminiClient.
- **analyze_with_ai**: Performs AI-powered analysis with sector context.
- **analyze_stock**: Main analysis method that orchestrates the entire workflow.

### AnalysisState (dataclass)
- Caches indicators, analysis_results, last_updated.
- **is_valid**: Checks if cached data is still valid.
- **update**: Updates state with new data.

## Technical Analysis (technical_indicators.py)

### TechnicalIndicators
- **calculate_all_indicators**: Main method that calculates all technical indicators.
- **calculate_sma/ema/wma**: Moving average calculations.
- **calculate_macd**: MACD indicator calculation.
- **calculate_rsi**: RSI indicator calculation.
- **calculate_bollinger_bands**: Bollinger Bands calculation.
- **calculate_adx**: ADX trend strength calculation.
- **calculate_enhanced_volume_analysis**: Advanced volume analysis.
- **calculate_multi_timeframe_analysis**: Multi-timeframe analysis.
- **get_market_metrics**: Market-specific metrics calculation.

### DataCollector
- **collect_all_data**: Collects and organizes all technical data.

### IndianMarketMetricsProvider
- **get_sector_index_data**: Retrieves sector index data.
- **get_basic_market_metrics**: Calculates basic market metrics.
- **get_enhanced_market_metrics**: Calculates enhanced market metrics.

## Pattern Recognition (patterns/recognition.py)

### PatternRecognition
- **detect_triangle**: Detects triangle patterns.
- **detect_flag**: Detects flag patterns.
- **detect_double_top/bottom**: Detects double top/bottom patterns.
- **detect_head_and_shoulders**: Detects head and shoulders patterns.
- **detect_divergence**: Detects price-indicator divergences.
- **detect_volume_anomalies**: Detects unusual volume patterns.

## Visualization (patterns/visualization.py)

### PatternVisualizer
- **plot_triangle_pattern**: Visualizes triangle patterns.
- **plot_flag_pattern**: Visualizes flag patterns.
- **plot_double_top_pattern**: Visualizes double top patterns.
- **plot_head_and_shoulders_pattern**: Visualizes head and shoulders patterns.

### ChartVisualizer
- **plot_comparison_chart**: Creates comprehensive comparison charts.
- **plot_volume_analysis**: Creates volume analysis charts.
- **plot_pattern_charts**: Creates pattern-specific charts.

## AI Analysis (gemini/gemini_client.py)

### GeminiClient
- **build_indicators_summary**: Creates indicator summary for AI analysis.
- **analyze_stock**: Performs comprehensive AI analysis.
- **analyze_comprehensive_overview**: Analyzes comprehensive chart overview.
- **analyze_volume_comprehensive**: Analyzes volume patterns.
- **analyze_reversal_patterns**: Analyzes reversal patterns.
- **analyze_continuation_levels**: Analyzes continuation patterns.

## Sector Analysis (sector_benchmarking.py)

### sector_benchmarking_provider
- **get_comprehensive_benchmarking**: Provides comprehensive sector benchmarking.
- **get_sector_rotation_analysis**: Analyzes sector rotation patterns.
- **get_sector_correlation_analysis**: Analyzes sector correlations.

## Data Client (zerodha_client.py)

### ZerodhaDataClient
- **authenticate**: Authenticates with Zerodha API.
- **get_historical_data**: Retrieves historical stock data.
- **get_instruments**: Retrieves instrument information.

## API Server (api.py)

### FastAPI Endpoints
- **POST /analyze**: Main analysis endpoint.
- **POST /sector/benchmark**: Sector benchmarking endpoint.
- **GET /sector/list**: List available sectors.
- **GET /sector/{sector_name}/stocks**: Get stocks in a sector.
- **GET /sector/{sector_name}/performance**: Get sector performance.
- **POST /sector/compare**: Compare multiple sectors.
- **GET /stock/{symbol}/sector**: Get stock's sector.
- **GET /health**: Health check endpoint.
- **GET /stock/{symbol}/info**: Get stock information.

## Call Flow

### Main Analysis Flow
1. **StockAnalysisOrchestrator.analyze_stock()**
   ├─ retrieve_stock_data()
   ├─ calculate_indicators()
   ├─ create_visualizations()
   ├─ analyze_with_ai()
   │  ├─ build_indicators_summary()
   │  ├─ analyze_comprehensive_overview()
   │  ├─ analyze_volume_comprehensive()
   │  ├─ analyze_reversal_patterns()
   │  └─ analyze_continuation_levels()
   └─ _create_overlays()

### Technical Analysis Flow
1. **TechnicalIndicators.calculate_all_indicators()**
   ├─ calculate_sma/ema/wma()
   ├─ calculate_macd()
   ├─ calculate_rsi()
   ├─ calculate_bollinger_bands()
   ├─ calculate_adx()
   ├─ calculate_enhanced_volume_analysis()
   └─ calculate_multi_timeframe_analysis()

### Pattern Recognition Flow
1. **PatternRecognition.detect_*()**
   ├─ detect_triangle()
   ├─ detect_flag()
   ├─ detect_double_top/bottom()
   ├─ detect_head_and_shoulders()
   ├─ detect_divergence()
   └─ detect_volume_anomalies()

### AI Analysis Flow
1. **GeminiClient.analyze_stock()**
   ├─ build_indicators_summary()
   ├─ analyze_comprehensive_overview()
   ├─ analyze_volume_comprehensive()
   ├─ analyze_reversal_patterns()
   └─ analyze_continuation_levels()

## Data Flow

### Input Data
- Stock symbol and exchange
- Analysis period and interval
- Optional sector information

### Processing Steps
1. **Data Retrieval**: Fetch historical OHLCV data
2. **Technical Analysis**: Calculate all technical indicators
3. **Pattern Recognition**: Detect chart patterns
4. **Visualization**: Generate chart images
5. **AI Analysis**: Perform AI-powered analysis
6. **Sector Analysis**: Apply sector-specific context
7. **Results Assembly**: Compile comprehensive results

### Output Data
- AI analysis with confidence levels
- Technical indicators and values
- Chart patterns and overlays
- Trading strategies and recommendations
- Risk management guidance
- Sector benchmarking data