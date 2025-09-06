# Stock Analysis System

A state-of-the-art, AI-powered stock analysis backend providing technical analysis, pattern recognition, sector benchmarking, and real-time trading insights.

## 🚀 Key Features

- **AI-Only Analysis**: Pure AI-powered analysis using Google's Gemini LLM—no rule-based consensus, no conflicting signals.
- **25+ Technical Indicators**: SMA, EMA, MACD, RSI, Bollinger Bands, ADX, Stochastic, OBV, Ichimoku, Fibonacci, and more.
- **Advanced Pattern Recognition**: Detects triangles, flags, double tops/bottoms, head & shoulders, divergences, and volume anomalies.
- **Sector Benchmarking**: Hybrid approach for sector-specific and market-wide benchmarking, with sector rotation and correlation analysis.
- **Real-Time Data**: Live market data and streaming via Zerodha API and WebSocket with multi-service architecture.
- **Advanced ML System**: Reorganized quantitative trading system with pattern ML, raw data ML, and hybrid approaches.
- **Microservices Architecture**: Scalable service-oriented architecture with consolidated and individual service deployment options.
- **Enhanced Visualization**: Chart generation with pattern overlays, multi-pane charts, and real-time updates.
- **Comprehensive Backtesting**: Advanced backtesting framework with realistic constraints and performance analysis.
- **Robust Error Handling**: Comprehensive validation, error reporting, and recovery across all services.
- **Performance Optimizations**: Intelligent caching, rate limiting, parallel processing, and memory optimization.

---

## 🏗️ Architecture

### Microservices Architecture

The system is built as a collection of microservices that can be deployed independently or as a consolidated service:

#### Core Services

- **Consolidated Service** (`services/consolidated_service.py`): Main FastAPI application that combines all services for deployment
- **Analysis Service** (`services/analysis_service.py`): Handles AI analysis, technical indicators, chart generation, and pattern recognition
- **Data Service** (`services/data_service.py`): Manages historical data retrieval, real-time streaming, and market data caching
- **Database Service** (`services/database_service.py`): Handles data persistence and retrieval operations
- **WebSocket Service** (`services/websocket_service.py`): Dedicated real-time data streaming service
- **Enhanced Data Service** (`services/enhanced_data_service.py`): Optimized data fetching with advanced caching

#### Core Components

- **StockAnalysisOrchestrator** (`analysis/orchestrator.py`): Main orchestrator for the entire analysis workflow. Handles authentication, data retrieval, indicator calculation, pattern recognition, AI analysis, visualization, and sector benchmarking.
- **TechnicalIndicators** (`ml/indicators/technical_indicators.py`): Calculates 25+ technical indicators and market metrics.
- **PatternRecognition** (`patterns/recognition.py`): Detects advanced chart patterns and anomalies.
- **PatternVisualizer/ChartVisualizer** (`patterns/visualization.py`): Generates pattern and comparison charts.
- **GeminiClient** (`gemini/gemini_client.py`): Interfaces with Gemini LLM for AI-powered analysis and summary generation.
- **SectorBenchmarkingProvider** (`ml/sector/benchmarking.py`): Provides sector benchmarking, rotation, and correlation analysis.
- **ZerodhaDataClient** (`zerodha/client.py`): Handles all data retrieval from Zerodha APIs.
- **SectorClassifier/EnhancedSectorClassifier** (`ml/sector/`): Classifies stocks into sectors using JSON-driven mappings and advanced filtering.

#### ML & Quantitative System

- **Quantitative Trading System** (`ml/quant_system/`): Reorganized ML system with:
  - **Core Infrastructure** (`core/`): Configuration, base models, registry, utilities
  - **ML Engines** (`engines/`): Pattern ML, raw data ML, hybrid ML, unified manager
  - **Advanced ML** (`advanced/`): N-BEATS, TFT, multimodal fusion, meta-learning
  - **Feature Engineering** (`features/`): Technical indicators, pattern features
  - **Trading System** (`trading/`): Strategies, execution, backtesting
  - **Evaluation** (`evaluation/`): Performance analysis and model evaluation

### Analysis Flow

1. **Data Retrieval**: Fetch historical OHLCV data from Zerodha.
2. **Technical Analysis**: Calculate all technical indicators.
3. **Pattern Recognition**: Detect chart and volume patterns.
4. **AI Analysis**: Generate insights and trading recommendations using Gemini LLM.
5. **Sector Context**: Apply sector benchmarking, rotation, and correlation context.
6. **Results Assembly**: Compile and serialize results for API response.

---

## 📊 API Endpoints

All endpoints are served via FastAPI through the consolidated service or individual microservices.

### Service Architecture

The system can be deployed as:
- **Consolidated Service**: Single FastAPI app with all endpoints
- **Microservices**: Individual services on different ports
  - Analysis Service: Port 8001
  - Data Service: Port 8000
  - Database Service: Port 8003
  - WebSocket Service: Port 8081

### Main Analysis Endpoints

- `POST /analysis/full` — Run full AI-powered analysis for a stock.
  - **Request:** `{ "symbol": "RELIANCE", "exchange": "NSE", "period": 365, "interval": "day" }`
  - **Response:** Complete analysis results with AI insights

- `POST /analysis/enhanced` — Enhanced analysis with advanced ML features
- `POST /analysis/risk` — Risk assessment for a stock
- `POST /analysis/backtest` — Backtesting analysis

### Data Service Endpoints

- `POST /data/fetch` — Fetch historical stock data
- `GET /data/stock-info/{symbol}` — Get basic stock information
- `GET /data/market-status` — Get current market status
- `GET /data/token-mapping` — Get token mapping for symbols

### Technical Analysis Endpoints

- `POST /technical/indicators` — Calculate technical indicators
- `POST /technical/market-metrics` — Get market metrics
- `POST /technical/enhanced-metrics` — Get enhanced market metrics

### Pattern Recognition Endpoints

- `POST /patterns/detect` — Detect chart patterns
- `GET /patterns/candlestick` — Get candlestick patterns
- `GET /patterns/chart` — Get chart patterns
- `GET /patterns/volume` — Get volume patterns

### Sector Analysis Endpoints

- `GET /sectors/info` — Get sector information
- `POST /sectors/benchmarking` — Get sector benchmarking
- `GET /sectors/rotation` — Get sector rotation data
- `GET /sectors/correlation` — Get sector correlation analysis
- `GET /sectors/performance` — Get sector performance metrics

### System Endpoints

- `GET /health` — Comprehensive health check for all services
- `GET /status` — Detailed status of all service components
- `GET /ws/stream` — WebSocket endpoint for real-time data streaming

### WebSocket Historical Replay (history action)

- **Action:** `history`
- **Description:** Request recent N candles for a given token and timeframe via WebSocket.
- **Request:**
  ```json
  {
    "action": "history",
    "token": "<INSTRUMENT_TOKEN>",
    "timeframe": "1m", // or 5m, 1d, etc.
    "count": 10 // optional, default 50, max 500
  }
  ```
- **Response:**
  ```json
  {
    "type": "history",
    "token": "<INSTRUMENT_TOKEN>",
    "timeframe": "1m",
    "count": 10,
    "candles": [
      { "open": ..., "high": ..., "low": ..., "close": ..., "volume": ..., "start": ..., "end": ... },
      // ... up to N candles ...
    ]
  }
  ```
- **Example Usage:**
  - Connect to `ws://<host>:<port>/ws/stream`
  - Send the above request as JSON
  - Receive the response with the most recent candles

### WebSocket Custom Alerts

- **Action:** `register_alert`
- **Description:** Register a custom alert for a token/timeframe (e.g., price crosses, volume spikes).
- **Request:**
  ```json
  {
    "action": "register_alert",
    "alert": {
      "type": "price_cross", // or "volume_spike"
      "token": "<INSTRUMENT_TOKEN>",
      "timeframe": "1m", // optional, for candle-based alerts
      "params": {
        "threshold": 2500,
        "direction": "above" // or "below" (for price_cross)
      }
    }
  }
  ```
- **Response:**
  ```json
  { "type": "alert_registered", "alert_id": "<ALERT_ID>" }
  ```
- **Alert Event:**
  When the alert triggers, the client receives:
  ```json
  {
    "type": "alert",
    "alert_id": "<ALERT_ID>",
    "alert_type": "price_cross",
    "token": "<INSTRUMENT_TOKEN>",
    "timeframe": "1m",
    "data": { /* tick or candle data */ },
    "params": { /* alert params */ }
  }
  ```
- **Remove Alert:**
  ```json
  { "action": "remove_alert", "alert_id": "<ALERT_ID>" }
  ```
  Response:
  ```json
  { "type": "alert_removed", "alert_id": "<ALERT_ID>" }
  ```
- **Supported Alert Types:**
  - `price_cross`: Triggers when price crosses a threshold (params: threshold, direction)
  - `volume_spike`: Triggers when volume exceeds a threshold (params: threshold)

- **Example Usage:**
  1. Connect to `ws://<host>:<port>/ws/stream`
  2. Register an alert as above
  3. Wait for alert events
  4. Remove alert when no longer needed

### Example Analysis Response

```json
{
  "success": true,
  "stock_symbol": "RELIANCE",
  "exchange": "NSE",
  "analysis_period": "365 days",
  "interval": "day",
  "timestamp": "2024-01-15T10:30:00",
  "message": "AI analysis completed for RELIANCE. Signal: Bullish (Confidence: 85%)",
  "results": {
    "ai_analysis": {
      "trend": "Bullish",
      "confidence_pct": 85,
      "short_term": {
        "entry_range": [2500, 2550],
        "stop_loss": 2450,
        "targets": [2600, 2650],
        "rationale": "Strong momentum with volume confirmation"
      }
    },
    "summary": {
      "overall_signal": "Bullish",
      "confidence": 85,
      "analysis_method": "AI-Powered Analysis",
      "risk_level": "Low",
      "recommendation": "Strong Buy"
    },
    "trading_guidance": {
      "short_term": {},
      "medium_term": {},
      "long_term": {},
      "risk_management": [],
      "key_levels": []
    },
    "indicators": {},
    "overlays": {},
    "indicator_summary_md": "",
    "chart_insights": "",
    "sector_benchmarking": {},
    "metadata": {}
  }
}
```

---

## 🧠 AI-Only System: No Consensus, No Conflicts

- **Single Source of Truth**: All recommendations and signals are generated by Gemini LLM.
- **Confidence Scoring**: Every signal and recommendation includes a confidence percentage.
- **Risk Management**: Built-in risk assessment and actionable trading guidance.
- **No Rule-Based Consensus**: All legacy consensus/conflict resolution code has been removed.

## 🤖 Advanced ML & Quantitative System

### Reorganized ML Architecture

The system now features a completely reorganized quantitative trading system (`ml/quant_system/`) with:

#### Core ML Infrastructure
- **Unified Configuration**: Centralized config management across all ML components
- **Model Registry**: Centralized model management and versioning
- **Base Models**: Abstract base classes for consistent ML implementations
- **Performance Tracking**: Comprehensive metrics and evaluation framework

#### ML Engines (Phase 1)
- **Pattern ML Engine**: CatBoost-based pattern success modeling
- **Raw Data ML Engine**: LSTM and Random Forest for direct OHLCV analysis
- **Hybrid ML Engine**: Combined approach for enhanced predictions
- **Unified ML Manager**: Orchestrates all ML engines with intelligent routing

#### Advanced ML (Phase 2)
- **N-BEATS Model**: Neural basis expansion for time series forecasting
- **Temporal Fusion Transformer**: Advanced transformer for multivariate time series
- **Multimodal Fusion**: Combines text, charts, and numerical data
- **Meta-Learning Framework**: Adaptive learning across different market conditions
- **Neural Architecture Search**: Automated model architecture optimization

#### Feature Engineering
- **Technical Indicators**: 25+ technical indicators with optimized calculations
- **Pattern Features**: Advanced pattern recognition features
- **Market Regime Features**: Market condition detection and classification
- **Cross-Asset Features**: Multi-asset correlation and momentum features

#### Trading System
- **Strategy Framework**: Modular trading strategy implementation
- **Execution Engine**: Order management and execution
- **Backtesting**: Comprehensive backtesting with realistic constraints
- **Risk Management**: Position sizing and risk controls

#### Evaluation & Analysis
- **Performance Analysis**: Comprehensive model performance evaluation
- **Price Analysis**: Advanced price prediction analysis
- **Benchmark Comparison**: Performance against market benchmarks
- **Feature Importance**: Model interpretability and feature analysis

---

## 🏭 Hybrid Sector Analysis

- **Stock-Specific Benchmarking**: Only fetches data for the stock's sector and NIFTY 50, minimizing API calls.
- **Comprehensive Sector Context**: Cached sector rotation and correlation analysis for all sectors, updated hourly.
- **Hybrid Results**: Combines stock-specific and market-wide sector context for robust benchmarking.

---

## ⚡ Real-Time Data & Streaming

- **WebSocket Streaming**: Real-time tick and candle data via `/ws/stream` endpoint.
- **Live Pub/Sub**: Efficient, filterable pub/sub system for streaming to multiple clients.
- **Multi-Service Architecture**: Dedicated WebSocket service for optimal performance.
- **Enhanced Data Service**: Optimized data fetching with intelligent caching.

## 🚀 Service Deployment Options

### Consolidated Service (Recommended for Production)
- **Single Application**: All services combined into one FastAPI app
- **Simplified Deployment**: Single port, single process
- **Resource Efficient**: Shared resources and optimized memory usage
- **Easy Scaling**: Horizontal scaling with load balancers

### Microservices Architecture (Recommended for Development)
- **Data Service** (Port 8000): Historical data, real-time streaming, market status
- **Analysis Service** (Port 8001): AI analysis, technical indicators, chart generation
- **Database Service** (Port 8003): Data persistence, caching, retrieval operations
- **WebSocket Service** (Port 8081): Dedicated real-time streaming service

### Service Communication
- **HTTP APIs**: RESTful communication between services
- **WebSocket**: Real-time data streaming
- **Redis**: Shared caching and pub/sub messaging
- **Supabase**: Centralized data persistence and authentication

### Deployment Configurations
- **Development**: Local services with hot reloading
- **Staging**: Containerized services with monitoring
- **Production**: Kubernetes/Docker deployment with auto-scaling

---

## 🔒 Security & Performance

- **API Rate Limiting**: Built-in rate limiting for all external API calls.
- **Caching**: Intelligent caching for repeated queries and sector data.
- **Parallel Processing**: Optimized for large datasets and fast response times.
- **Robust Error Handling**: All endpoints return clear error messages and validation.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Zerodha API credentials
- Google Gemini API key
- Redis (for caching and real-time features)
- Supabase account (for authentication and data persistence)

### Installation

#### Option 1: Consolidated Service (Recommended)
```bash
# Install all dependencies
pip install -r config/requirements.txt

# Run consolidated service
python services/consolidated_service.py
```

#### Option 2: Individual Microservices
```bash
# Install dependencies
pip install -r config/requirements.txt

# Run individual services
python services/data_service.py      # Port 8000
python services/analysis_service.py  # Port 8001
python services/database_service.py  # Port 8003
python services/websocket_service.py # Port 8081
```

#### Option 3: Development Setup
```bash
# Install development dependencies
pip install -r config/requirements-dev.txt

# Run with development configuration
python services/consolidated_service.py --dev
```

### Configuration
1. Set up Zerodha API credentials in environment variables
2. Configure Google Gemini API key
3. Set up Redis connection
4. Configure Supabase credentials
5. Set up sector classification data in `ml/sector/`

### Environment Variables

Create a `.env` file in the backend directory:

```bash
# Zerodha API Configuration
ZERODHA_API_KEY=your_api_key
ZERODHA_ACCESS_TOKEN=your_access_token

# Google Gemini API
GOOGLE_GEMINI_API_KEY=your_gemini_api_key

# Database Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Redis Configuration
REDIS_URL=redis://localhost:6379

# JWT Configuration
JWT_SECRET=your_jwt_secret

# Service URLs (for microservices deployment)
DATABASE_SERVICE_URL=http://localhost:8003
ANALYSIS_SERVICE_URL=http://localhost:8001
DATA_SERVICE_URL=http://localhost:8000

# Test Mode Configuration (Optional)
MARKET_HOURS_TEST_MODE=false
FORCE_MARKET_OPEN=false

# Deployment Configuration
ENVIRONMENT=development  # development, staging, production
```

### Test Mode Configuration

For testing purposes, you can disable market hours restrictions to test frontend live data functionality anytime.

#### Enable Test Mode
```bash
# Add to your .env file
MARKET_HOURS_TEST_MODE=true
FORCE_MARKET_OPEN=true
```

#### Test Mode Behavior
When `MARKET_HOURS_TEST_MODE=true`:
- **Market Status**: Always returns `OPEN` status
- **Weekends**: Treated as regular trading days
- **Holidays**: Treated as regular trading days
- **WebSocket**: Always recommended for all intervals
- **Cache Duration**: Reduced to 1 minute for live testing
- **Cost Estimation**: Returns 0 cost for testing
- **Tick Processing**: All ticks are processed regardless of market status

#### Usage Examples
```bash
# For Frontend Testing
MARKET_HOURS_TEST_MODE=true
FORCE_MARKET_OPEN=true

# For Production
MARKET_HOURS_TEST_MODE=false
FORCE_MARKET_OPEN=false
```

#### API Response Examples
Market Status (with test mode):
```json
{
  "current_time": "2024-01-15T14:30:00+05:30",
  "timezone": "Asia/Kolkata",
  "market_status": "open",
  "is_weekend": false,
  "is_holiday": false,
  "test_mode_enabled": true,
  "force_market_open": true
}
```

WebSocket Messages (with test mode):
```json
{
  "type": "tick",
  "token": "256265",
  "price": 2450.50,
  "timestamp": 1705312200,
  "volume_traded": 1000,
  "market_status": "open",
  "data_freshness": "real_time",
  "test_mode": true,
  "force_market_open": true
}
```

#### Logging
When test mode is enabled, you'll see:
```
🔧 MARKET HOURS TEST MODE ENABLED - Market hours restrictions disabled
🔧 FORCE MARKET OPEN ENABLED - Market will always appear as OPEN
[TEST MODE] Processed tick for token: 256265, price: 2450.50
```

**Security Note**: Test mode should only be enabled in development/testing environments, never in production.

---

## 📝 Documentation & Contribution

- **API Documentation**: See this README and code comments for endpoint details.
- **Analysis Guides**: See `AI_ONLY_ANALYSIS_GUIDE.md` and `HYBRID_SECTOR_ANALYSIS_APPROACH.md` for advanced usage.
- **Contributing**: Fork, branch, make changes, add tests, and submit a pull request.
- **License**: MIT License (see LICENSE file).


