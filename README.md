# Stock Analysis System

A comprehensive stock analysis system that provides AI-powered technical analysis, pattern recognition, and trading insights.

## 🚀 **Key Features**

- **AI-Powered Analysis**: Advanced analysis using Google's Gemini LLM
- **Technical Indicators**: Comprehensive technical indicator calculations
- **Pattern Recognition**: Advanced chart pattern detection
- **Sector Analysis**: Sector-specific benchmarking and analysis
- **Real-time Data**: Live market data integration via Zerodha API
- **Visualization**: Advanced chart generation and pattern visualization

## 🏗️ **Architecture**

### **Core Components**

- **StockAnalysisOrchestrator**: Main orchestrator for analysis workflow
- **TechnicalIndicators**: Technical indicator calculations
- **GeminiClient**: AI-powered analysis using Google's Gemini LLM
- **PatternRecognition**: Advanced pattern detection algorithms
- **SectorBenchmarking**: Sector-specific analysis and benchmarking

### **Analysis Flow**

1. **Data Retrieval**: Fetch historical data from Zerodha API
2. **Technical Analysis**: Calculate comprehensive technical indicators
3. **Pattern Recognition**: Detect chart patterns and formations
4. **AI Analysis**: Generate AI-powered insights and trading recommendations
5. **Sector Context**: Apply sector-specific analysis and benchmarking
6. **Results Assembly**: Compile comprehensive analysis results

## 📊 **API Response Structure**

The system returns comprehensive analysis results including:

- `ai_analysis`: AI-powered analysis with confidence levels and trading strategies
- `indicators`: Technical indicator calculations and values
- `overlays`: Chart patterns and technical overlays
- `indicator_summary_md`: Markdown summary of technical indicators
- `chart_insights`: AI-generated chart pattern insights
- `summary`: Overall analysis summary with signals and recommendations
- `trading_guidance`: Specific trading strategies and risk management
- `sector_benchmarking`: Sector-specific analysis and comparisons
- `metadata`: Analysis metadata and timestamps

### **Example Response Structure**
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
      "short_term": { /* trading strategy */ },
      "medium_term": { /* trading strategy */ },
      "long_term": { /* trading strategy */ },
      "risk_management": [ /* risk factors */ ],
      "key_levels": [ /* important price levels */ ]
    }
  }
}
```

## 🔧 **Installation & Setup**

### **Prerequisites**
- Python 3.8+
- Zerodha API credentials
- Google Gemini API key

### **Installation**
```bash
pip install -r requirements.txt
```

### **Configuration**
1. Set up Zerodha API credentials in `config.py`
2. Configure Google Gemini API key
3. Set up sector classification data

## 🚀 **Usage**

### **API Endpoints**

#### **Analyze Stock**
```bash
POST /analyze
{
  "stock": "RELIANCE",
  "exchange": "NSE",
  "period": 365,
  "interval": "day",
  "sector": "energy"
}
```

#### **Sector Benchmarking**
```bash
POST /sector/benchmark
{
  "stock": "RELIANCE",
  "sector": "energy"
}
```

#### **Sector Comparison**
```bash
POST /sector/compare
{
  "sectors": ["energy", "technology", "banking"]
}
```

## 🎯 **AI-Powered Analysis**

The system uses Google's Gemini LLM for sophisticated analysis:

- **Multi-modal Analysis**: Combines technical indicators, chart patterns, and market context
- **Confidence Scoring**: Provides confidence levels for all recommendations
- **Conflict Resolution**: Intelligent handling of conflicting signals
- **Market Context**: Considers broader market conditions and sector dynamics
- **Risk Management**: Built-in risk assessment and management recommendations

## 📈 **Technical Indicators**

Comprehensive technical analysis including:

- **Moving Averages**: SMA, EMA, WMA with multiple timeframes
- **Momentum Indicators**: RSI, MACD, Stochastic, Williams %R
- **Volatility Indicators**: Bollinger Bands, ATR, Keltner Channels
- **Volume Analysis**: OBV, Volume Profile, Volume Ratios
- **Trend Indicators**: ADX, Ichimoku, Parabolic SAR
- **Support/Resistance**: Dynamic level detection and analysis

## 🎨 **Pattern Recognition**

Advanced pattern detection algorithms:

- **Reversal Patterns**: Head & Shoulders, Double Tops/Bottoms, Triple Tops/Bottoms
- **Continuation Patterns**: Triangles, Flags, Pennants, Wedges
- **Candlestick Patterns**: Doji, Hammer, Shooting Star, Engulfing
- **Divergence Detection**: Price-Volume and Price-Indicator divergences
- **Volume Anomalies**: Unusual volume patterns and spikes

## 🏭 **Sector Analysis**

Comprehensive sector-specific analysis:

- **Sector Benchmarking**: Performance comparison within sectors
- **Sector Rotation**: Analysis of sector rotation patterns
- **Correlation Analysis**: Inter-sector correlation insights
- **Sector-Specific Metrics**: Tailored analysis for different sectors

## 🔒 **Security & Performance**

- **API Rate Limiting**: Built-in rate limiting for external APIs
- **Caching**: Intelligent caching for performance optimization
- **Error Handling**: Comprehensive error handling and recovery
- **Data Validation**: Robust data validation and sanitization

## 📝 **Documentation**

- **API Documentation**: Comprehensive API endpoint documentation
- **Analysis Guides**: Detailed guides for interpreting analysis results
- **Best Practices**: Trading and analysis best practices
- **Troubleshooting**: Common issues and solutions

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.


