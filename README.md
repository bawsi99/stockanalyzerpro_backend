# Stock Analysis System

A state-of-the-art, AI-powered stock analysis backend providing technical analysis, pattern recognition, sector benchmarking, and real-time trading insights.

## 🚀 Key Features

- **AI-Only Analysis**: Pure AI-powered analysis using Google's Gemini LLM—no rule-based consensus, no conflicting signals.
- **Comprehensive Technical Indicators**: SMA, EMA, MACD, RSI, Bollinger Bands, ADX, multi-timeframe, and more.
- **Advanced Pattern Recognition**: Detects triangles, flags, double tops/bottoms, head & shoulders, divergences, and volume anomalies.
- **Sector Benchmarking**: Hybrid approach for sector-specific and market-wide benchmarking, with sector rotation and correlation analysis.
- **Real-Time Data**: Live market data and streaming via Zerodha API and WebSocket.
- **Visualization**: Chart generation and pattern overlays for actionable insights.
- **Robust Error Handling**: Comprehensive validation, error reporting, and recovery.
- **Performance Optimizations**: Caching, rate limiting, and parallel processing.

---

## 🏗️ Architecture

### Core Components

- **StockAnalysisOrchestrator** (`agent_capabilities.py`): Main orchestrator for the entire analysis workflow. Handles authentication, data retrieval, indicator calculation, pattern recognition, AI analysis, visualization, and sector benchmarking.
- **TechnicalIndicators** (`technical_indicators.py`): Calculates all technical indicators and market metrics.
- **PatternRecognition** (`patterns/recognition.py`): Detects advanced chart patterns and anomalies.
- **PatternVisualizer/ChartVisualizer** (`patterns/visualization.py`): Generates pattern and comparison charts.
- **GeminiClient** (`gemini/gemini_client.py`): Interfaces with Gemini LLM for AI-powered analysis and summary generation.
- **SectorBenchmarkingProvider** (`sector_benchmarking.py`): Provides sector benchmarking, rotation, and correlation analysis.
- **ZerodhaDataClient** (`zerodha_client.py`): Handles all data retrieval from Zerodha APIs.
- **SectorClassifier/EnhancedSectorClassifier**: Classifies stocks into sectors using JSON-driven mappings and advanced filtering.
- **LiveDataPubSub** (`api.py`): Real-time data pub/sub for WebSocket streaming.

### Analysis Flow

1. **Data Retrieval**: Fetch historical OHLCV data from Zerodha.
2. **Technical Analysis**: Calculate all technical indicators.
3. **Pattern Recognition**: Detect chart and volume patterns.
4. **AI Analysis**: Generate insights and trading recommendations using Gemini LLM.
5. **Sector Context**: Apply sector benchmarking, rotation, and correlation context.
6. **Results Assembly**: Compile and serialize results for API response.

---

## 📊 API Endpoints

All endpoints are served via FastAPI.

### Main Endpoints

- `POST /analyze` — Run full AI-powered analysis for a stock.
  - **Request:** `{ "stock": "RELIANCE", "exchange": "NSE", "period": 365, "interval": "day", "sector": "energy" }`
  - **Response:** Full analysis results (see below).

- `POST /sector/benchmark` — Get sector benchmarking for a stock.
  - **Request:** `{ "stock": "RELIANCE", "sector": "energy" }`

- `POST /sector/compare` — Compare multiple sectors.
  - **Request:** `{ "sectors": ["energy", "technology", "banking"] }`

- `GET /sector/list` — List all available sectors.
- `GET /sector/{sector_name}/stocks` — List all stocks in a sector.
- `GET /sector/{sector_name}/performance` — Get sector performance metrics.
- `GET /stock/{symbol}/sector` — Get sector info for a stock.
- `GET /stock/{symbol}/info` — Get basic stock info and quote.
- `GET /health` — Health check endpoint.
- `GET /ws/stream` — WebSocket endpoint for real-time data streaming.

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

---

## 🏭 Hybrid Sector Analysis

- **Stock-Specific Benchmarking**: Only fetches data for the stock's sector and NIFTY 50, minimizing API calls.
- **Comprehensive Sector Context**: Cached sector rotation and correlation analysis for all sectors, updated hourly.
- **Hybrid Results**: Combines stock-specific and market-wide sector context for robust benchmarking.

---

## ⚡ Real-Time Data & Streaming

- **WebSocket Streaming**: Real-time tick and candle data via `/ws/stream` endpoint.
- **Live Pub/Sub**: Efficient, filterable pub/sub system for streaming to multiple clients.

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

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
1. Set up Zerodha API credentials in `config.py` or environment variables.
2. Configure Google Gemini API key.
3. Set up sector classification data in `sector_category/`.

### Environment Variables

Create a `.env` file in the backend directory:

```bash
# Zerodha API Configuration
ZERODHA_API_KEY=your_api_key
ZERODHA_ACCESS_TOKEN=your_access_token

# JWT Configuration
JWT_SECRET=your_jwt_secret

# Test Mode Configuration (Optional)
MARKET_HOURS_TEST_MODE=false
FORCE_MARKET_OPEN=false

# Google Gemini API
GOOGLE_GEMINI_API_KEY=your_gemini_api_key
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


