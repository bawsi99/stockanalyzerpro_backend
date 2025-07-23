# 🚀 Real-Time Analysis System Guide

## Overview

The real-time analysis system provides **live, AI-powered stock analysis** using websocket streaming data. It automatically triggers full analysis (indicators, patterns, charts, LLM) on new candles and notifies users in real-time.

## ✨ Key Features

### ✅ **Full AI Analysis**
- **LLM Integration**: Every new candle triggers complete AI analysis using Gemini
- **Real-time Charts**: Automatic chart generation for technical analysis
- **Multi-timeframe**: Analyzes multiple timeframes simultaneously
- **Sector Context**: Includes sector benchmarking and rotation analysis

### ✅ **Result Storage & Retrieval**
- **In-Memory Storage**: Latest analysis results stored for each token/timeframe
- **Analysis History**: Access to recent analysis results
- **Chart Caching**: Generated charts cached for quick access

### ✅ **User Notifications**
- **WebSocket Notifications**: Real-time push notifications to subscribed users
- **Subscription Management**: Subscribe/unsubscribe to specific stocks/timeframes
- **User-Specific Queues**: Each user gets their own notification stream

### ✅ **Rate Limiting & Optimization**
- **Smart Rate Limiting**: 30-second minimum between analyses per token/timeframe
- **LLM Call Optimization**: Prevents excessive API usage
- **Resource Management**: Efficient memory and CPU usage

---

## 🏗️ Architecture

### **Core Components**

1. **RealTimeAnalysisManager** (`api.py`)
   - Manages analysis results storage
   - Handles user subscriptions
   - Coordinates notifications

2. **Enhanced Real-Time Callback** (`api.py`)
   - Triggers on every new candle
   - Runs full analysis pipeline
   - Stores results and notifies users

3. **WebSocket Endpoints**
   - `/ws/stream`: Raw data streaming
   - `/ws/realtime-analysis`: Analysis notifications

4. **REST API Endpoints**
   - Real-time analysis retrieval
   - Subscription management
   - System status monitoring

---

## 📡 API Endpoints

### **Real-Time Analysis Endpoints**

#### `GET /realtime/analysis/{token}/{timeframe}`
Get the latest real-time analysis for a specific token/timeframe.

**Response:**
```json
{
  "success": true,
  "token": "256265",
  "timeframe": "1d",
  "analysis": {
    "symbol": "RELIANCE",
    "ai_analysis": {
      "trend": "Bullish",
      "confidence_pct": 85,
      "short_term": { ... },
      "medium_term": { ... },
      "long_term": { ... }
    },
    "indicators": { ... },
    "signal_confluence": {
      "score": 0.75,
      "high_confidence": true
    },
    "chart_paths": { ... },
    "analysis_metadata": {
      "analysis_timestamp": "2024-01-15T10:30:00",
      "data_freshness": "real_time"
    }
  }
}
```

#### `POST /realtime/subscribe`
Subscribe a user to real-time analysis updates.

**Request:**
```json
{
  "user_id": "user123",
  "token": "256265",
  "timeframe": "1d"
}
```

#### `POST /realtime/unsubscribe`
Unsubscribe a user from real-time analysis updates.

#### `GET /realtime/subscriptions/{user_id}`
Get all subscriptions for a user.

#### `GET /realtime/analysis-history/{token}/{timeframe}`
Get analysis history for a token/timeframe.

#### `GET /realtime/status`
Get system status and statistics.

---

## 🔌 WebSocket Endpoints

### **`/ws/realtime-analysis`**

Real-time analysis notifications via WebSocket.

#### **Connection**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime-analysis');
```

#### **Subscribe to Analysis**
```javascript
ws.send(JSON.stringify({
  action: 'subscribe',
  token: '256265',
  timeframe: '1d'
}));
```

#### **Unsubscribe from Analysis**
```javascript
ws.send(JSON.stringify({
  action: 'unsubscribe',
  token: '256265',
  timeframe: '1d'
}));
```

#### **Get Current Subscriptions**
```javascript
ws.send(JSON.stringify({
  action: 'get_subscriptions'
}));
```

#### **Get Latest Analysis**
```javascript
ws.send(JSON.stringify({
  action: 'get_latest_analysis',
  token: '256265',
  timeframe: '1d'
}));
```

#### **Received Notifications**
```javascript
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  if (data.type === 'analysis_update') {
    console.log('New analysis:', data.analysis);
    // Handle new analysis
  }
};
```

---

## 🔄 Analysis Pipeline

### **1. Trigger Conditions**
- New candle completion (every 1m, 5m, 15m, 1h, 1d)
- Rate limiting: minimum 30 seconds between analyses
- Minimum 20 candles required for analysis

### **2. Analysis Steps**
1. **Data Collection**: Fetch recent candles from websocket
2. **Technical Indicators**: Calculate all indicators
3. **Pattern Recognition**: Detect candlestick patterns
4. **Multi-timeframe**: Analyze higher timeframes
5. **Sector Analysis**: Get sector context and benchmarking
6. **Chart Generation**: Create real-time charts
7. **AI Analysis**: Call Gemini LLM for insights
8. **Result Storage**: Store analysis with timestamp
9. **User Notifications**: Notify subscribed users

### **3. Analysis Components**
- **AI Analysis**: Trend, confidence, trading strategies
- **Signal Confluence**: Multi-factor confirmation scoring
- **Risk Management**: Dynamic stops and position sizing
- **Pattern Backtesting**: Historical pattern performance
- **Sector Context**: Relative strength and rotation

---

## 📊 Analysis Results Structure

```json
{
  "symbol": "RELIANCE",
  "token": "256265",
  "timeframe": "1d",
  "last_candle": { ... },
  "indicators": {
    "sma": { ... },
    "ema": { ... },
    "rsi": { ... },
    "macd": { ... },
    "bollinger_bands": { ... }
  },
  "candlestick_patterns": [ ... ],
  "volume_confirmation": "bullish",
  "multi_timeframe_confirmations": {
    "1h": { ... },
    "4h": { ... }
  },
  "confirmation_status": "confirmed_bullish",
  "sector_analysis": {
    "sector": "OIL_GAS",
    "rs_sector": 1.2,
    "rs_index": 1.1,
    "sector_trend": "bullish"
  },
  "risk_management": {
    "dynamic_risk": { ... },
    "pattern_stats": { ... },
    "invalidation_trigger": "..."
  },
  "signal_confluence": {
    "signals": { ... },
    "score": 0.75,
    "rationale": { ... },
    "high_confidence": true
  },
  "ai_analysis": {
    "trend": "Bullish",
    "confidence_pct": 85,
    "short_term": {
      "entry_range": [100, 105],
      "stop_loss": 95,
      "targets": [110, 115]
    },
    "medium_term": { ... },
    "long_term": { ... }
  },
  "chart_paths": {
    "comparison_chart": "./output/realtime/RELIANCE_1d/comparison_chart.png",
    "volume_analysis": "./output/realtime/RELIANCE_1d/volume_analysis.png"
  },
  "analysis_metadata": {
    "candles_analyzed": 50,
    "analysis_timestamp": "2024-01-15T10:30:00",
    "data_freshness": "real_time"
  }
}
```

---

## 🚀 Usage Examples

### **Python Client Example**

```python
import asyncio
import websockets
import json

async def real_time_analysis_client():
    uri = "ws://localhost:8000/ws/realtime-analysis"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe to RELIANCE daily analysis
        await websocket.send(json.dumps({
            "action": "subscribe",
            "token": "256265",
            "timeframe": "1d"
        }))
        
        # Listen for analysis updates
        async for message in websocket:
            data = json.loads(message)
            
            if data["type"] == "analysis_update":
                analysis = data["analysis"]
                print(f"New analysis for {analysis['symbol']}:")
                print(f"AI Trend: {analysis['ai_analysis']['trend']}")
                print(f"Confidence: {analysis['ai_analysis']['confidence_pct']}%")
                print(f"Confluence Score: {analysis['signal_confluence']['score']}")

# Run the client
asyncio.run(real_time_analysis_client())
```

### **JavaScript Client Example**

```javascript
class RealTimeAnalysisClient {
    constructor() {
        this.ws = new WebSocket('ws://localhost:8000/ws/realtime-analysis');
        this.setupEventHandlers();
    }
    
    setupEventHandlers() {
        this.ws.onopen = () => {
            console.log('Connected to real-time analysis');
            this.subscribe('256265', '1d'); // RELIANCE daily
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
    }
    
    subscribe(token, timeframe) {
        this.ws.send(JSON.stringify({
            action: 'subscribe',
            token: token,
            timeframe: timeframe
        }));
    }
    
    handleMessage(data) {
        if (data.type === 'analysis_update') {
            const analysis = data.analysis;
            console.log(`New analysis for ${analysis.symbol}:`);
            console.log(`AI Trend: ${analysis.ai_analysis.trend}`);
            console.log(`Confidence: ${analysis.ai_analysis.confidence_pct}%`);
            
            // Update UI with new analysis
            this.updateUI(analysis);
        }
    }
    
    updateUI(analysis) {
        // Update your UI components here
        document.getElementById('trend').textContent = analysis.ai_analysis.trend;
        document.getElementById('confidence').textContent = analysis.ai_analysis.confidence_pct + '%';
    }
}

// Initialize client
const client = new RealTimeAnalysisClient();
```

---

## 📦 Standardized WebSocket Message Schemas

### Tick Message
```json
{
  "type": "tick",
  "token": <int>,
  "price": <float>,
  "timestamp": <float>,
  "volume_traded": <float>,
  // ...additional fields as needed
}
```

### Candle Message
```json
{
  "type": "candle",
  "token": <str>,
  "timeframe": <str>,
  "data": {
    "open": <float>,
    "high": <float>,
    "low": <float>,
    "close": <float>,
    "volume": <float>,
    "start": <float>,
    "end": <float>
  },
  "timestamp": <float>
}
```

- All timestamps are in UNIX epoch seconds (UTC).
- All price/volume fields are floats.
- The `token` is the instrument token (int for tick, str for candle for compatibility).
- The `type` field is always present and is either 'tick' or 'candle'.
- Additional fields may be added as needed, but these are the required minimums.

---

## ⚙️ Configuration

### **Rate Limiting**
- **Analysis Interval**: 30 seconds minimum between analyses per token/timeframe
- **LLM Calls**: Rate limited to prevent excessive API usage
- **Chart Generation**: Optimized for real-time performance

### **Data Requirements**
- **Minimum Candles**: 20 candles required for analysis
- **Supported Timeframes**: 1m, 5m, 15m, 1h, 1d
- **Market Hours**: Real-time data only during market hours (9:15 AM - 3:30 PM IST)

### **Storage**
- **Analysis Results**: In-memory storage (latest per token/timeframe)
- **Chart Cache**: File-based storage in `./output/realtime/`
- **User Subscriptions**: In-memory with automatic cleanup

---

## 🔧 Troubleshooting

### **Common Issues**

1. **No Analysis Results**
   - Check if websocket is connected
   - Verify token is subscribed to
   - Ensure market is open for real-time data

2. **High CPU Usage**
   - Reduce analysis frequency
   - Limit number of subscriptions
   - Check for memory leaks

3. **LLM API Errors**
   - Verify API key is valid
   - Check rate limits
   - Monitor API usage

### **Monitoring**

Use the status endpoint to monitor system health:
```bash
curl http://localhost:8000/realtime/status
```

---

## 🎯 Next Steps

1. **Database Integration**: Store analysis history in database
2. **Advanced Filtering**: Filter analyses by confidence, trend, etc.
3. **Alert System**: Custom alerts based on analysis results
4. **Backtesting**: Historical analysis replay
5. **Performance Optimization**: Parallel processing for multiple analyses

---

## 📝 Notes

- **Memory Usage**: Analysis results are stored in memory, consider database for production
- **API Limits**: Monitor LLM API usage to avoid rate limits
- **Market Hours**: Real-time analysis only works during market hours
- **Error Handling**: All errors are logged and handled gracefully
- **Scalability**: System designed for multiple concurrent users 