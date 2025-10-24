# 📊 **StockAnalyzer Pro - ML System Documentation**

## 🎯 **Overview**

StockAnalyzer Pro uses machine learning to predict profitable trading opportunities in Indian stock markets (NSE). The system combines traditional technical analysis with machine learning to automate trading decisions, optimizing for actual returns rather than just prediction accuracy.

---

## 🧠 **Model Architecture**

### **Model Type**: Binary Classification
- **Algorithm**: Logistic Regression with L2 regularization
- **Framework**: scikit-learn
- **Objective**: Predict probability of profitable trades after transaction costs

### **Why Classification?**
- **Training Target**: Binary labels (profitable=1, unprofitable=0)
- **Output**: Probabilities between 0-1
- **Decision**: Probability threshold determines buy/hold signals
- **Optimization**: Threshold optimized for maximum returns, not accuracy

---

## 📈 **What We're Predicting**

### **Core Question**
*"If I buy this stock right NOW, will I make money after transaction costs within the next X time periods?"*

### **Multi-Timeframe Predictions**
| Timeframe | Horizon | Transaction Cost | Use Case |
|-----------|---------|------------------|----------|
| 5-minute | 12 bars (1 hour) | 8 bps (0.08%) | High-frequency trading |
| 15-minute | 8 bars (2 hours) | 7 bps (0.07%) | Intraday swing trading |
| 1-hour | 12 bars (12 hours) | 6 bps (0.06%) | End-of-day trading |
| Daily | 5 bars (5 days) | 5 bps (0.05%) | Swing/position trading |

### **Current Universe**
- **Primary**: RELIANCE (NSE)
- **Exchange**: National Stock Exchange (NSE)
- **Expandable**: Architecture supports multiple stocks

---

## 🛠 **Features (Input Variables)**

The model uses **40+ technical indicators** across multiple categories:

### **1. Volatility Features (3)**
- `atr_14_pct`: Average True Range as % of price
- `atr_vol_20`: 20-period volatility of ATR
- `range_pct`: Daily high-low range as % of close

### **2. Trend & Moving Averages (2)**
- `dist_sma50_pct`: Distance from 50-period moving average
- `macd_hist`: MACD histogram (momentum indicator)

### **3. Bollinger Bands (1)**
- `bb_bw_20`: Bollinger Band bandwidth

### **4. Volume Features (5)**
- `vol_ratio_20`: Current volume vs 20-day average
- `vol_cv_20`: Volume coefficient of variation
- `cmf_20`: Chaikin Money Flow
- `up_down_vol_ratio_20`: Volume on up days vs down days
- `ret_vol_corr_20`: Return-volume correlation

### **5. Price Position Features (4)**
- `pct_dist_to_20_high`: Distance from 20-day high
- `breakout_up_20`: Binary flag for upward breakouts
- `breakout_down_20`: Binary flag for downward breakouts
- `vwap_dist`: Distance from VWAP

### **6. VWAP Features (2)**
- `vwap_dist`: Distance from VWAP
- `vwap_slope_5`: 5-period slope of VWAP

### **7. Candlestick Patterns (6)**
- `wick_to_body_ratio`: Ratio of wicks to candle body
- `inside_bar`: Inside bar pattern
- `engulfing`: Engulfing pattern
- `gap_pct`: Gap percentage between sessions
- `up_streak`: Consecutive up days
- `down_streak`: Consecutive down days

### **8. Statistical Features (2)**
- `ret_skew_20`: 20-period return skewness
- `ret_kurt_20`: 20-period return kurtosis

### **9. Calendar Features (8)**
- `dow`: Day of week (0-6)
- `dow_sin`, `dow_cos`: Cyclical day-of-week encoding
- `hour`: Hour of day (for intraday)
- `hour_sin`, `hour_cos`: Cyclical hour encoding

---

## 🏷️ **Labels (Target Variables)**

### **Dual Label System**

#### **y_reg (Continuous Return)**
```python
# Calculation
close = current_closing_price
fwd_price = closing_price_after_N_bars
log_ret = ln(fwd_price / close)
cost = transaction_cost_bps / 10000.0
y_reg = log_ret - cost  # Net return after costs
```

**Examples**:
- `y_reg = +0.0234` → 2.34% profit after costs
- `y_reg = -0.0087` → 0.87% loss after costs
- `y_reg = +0.0003` → 0.03% tiny profit

#### **y_cls (Binary Classification)**
```python
y_cls = 1 if y_reg > 0 else 0  # 1 = profitable, 0 = unprofitable
```

### **Key Insight**: 
- **Model trains on** `y_cls` (binary)
- **Threshold optimization uses** `y_reg` (continuous)
- **Model never predicts** `y_reg` - only probability of `y_cls`

---

## ⚖️ **Threshold Optimization**

### **The Innovation**
Instead of using default 50% probability threshold, we find the optimal threshold that maximizes historical returns.

### **Process**
```python
def select_best_threshold(probabilities, y_reg_values):
    # Test 50 thresholds from 50th to 99th percentile
    for threshold in threshold_grid:
        # Simulate trading with this threshold
        trades = probabilities >= threshold
        avg_return = y_reg_values[trades].mean()
        # Keep threshold with highest average return
```

### **Example Results**
| Threshold | Trades | Coverage | Avg Return | Total Return |
|-----------|--------|----------|------------|--------------|
| 0.50 | 45/50 | 90% | +0.5% | +22.5% |
| 0.60 | 38/50 | 76% | +0.8% | +30.4% |
| 0.70 | 25/50 | 50% | +1.5% | +37.5% |
| **0.72** | **18/50** | **36%** | **+2.1%** | **+37.8%** ← **OPTIMAL** |
| 0.80 | 8/50 | 16% | +3.5% | +28.0% |

### **Trade-off**
- **Lower thresholds**: More trades, lower average profit
- **Higher thresholds**: Fewer trades, higher average profit
- **Optimal**: Maximum total portfolio return

---

## 📊 **Model Training Configuration**

### **Algorithm Settings**
```python
LogisticRegression(
    penalty="l2",              # L2 regularization
    class_weight="balanced",   # Handle imbalanced classes
    max_iter=2000,            # Convergence iterations
    solver="lbfgs"            # Optimization algorithm
)
```

### **Data Splits**
- **Training**: 70% (temporal split)
- **Validation**: 15% (for threshold optimization)
- **Test**: 15% (final evaluation)

### **Feature Scaling**
- **Method**: Z-score standardization
- **Formula**: `(feature - mean) / std`
- **Binary features**: Not scaled (kept as 0/1)

---

## 🔄 **Data Pipeline**

### **Directory Structure**
```
backend/agents/ml/
├── data_processing/          # Data pipeline & feature engineering
│   ├── data_extractor.py    # Raw data from Zerodha API
│   ├── build_dataset.py     # Feature engineering (40+ indicators)
│   ├── build_labels.py      # y_reg & y_cls generation
│   ├── clean_dataset.py     # Data cleaning & validation
│   ├── cap_features.py      # Outlier handling
│   ├── qc_dataset.py        # Quality control
│   ├── split_dataset.py     # Train/val/test splitting
│   ├── split_qc.py          # Split quality & drift analysis
│   └── standardize_train.py # Feature standardization
├── training/                 # Model training
│   └── train_logistic.py    # Main training script
├── config/                   # Configuration
│   └── config.py            # Timeframe specs & settings
├── models/                   # Trained models
│   └── model_YYYYMMDD_HHMMSS/
│       ├── logreg.joblib    # Trained model
│       ├── metrics.json     # Performance metrics
│       ├── roc_curves.png   # ROC curve visualization
│       └── test_predictions.csv
└── data/                     # Data storage
    ├── raw/                  # Raw OHLCV data
    └── processed/            # Processed features & labels
```

### **Pipeline Steps**
1. **Extract** raw OHLCV data from Zerodha API
2. **Engineer** 40+ technical indicators
3. **Generate** forward-looking labels (y_reg, y_cls)
4. **Clean** and validate data quality
5. **Split** into train/validation/test sets
6. **Standardize** features using training statistics
7. **Train** logistic regression model
8. **Optimize** probability threshold for maximum returns
9. **Evaluate** performance on test set

---

## 📈 **Model Outputs**

### **Training Artifacts**
```
models/model_20241021_143022/
├── logreg.joblib              # Serialized model + threshold + features
├── metrics.json               # Performance metrics
├── roc_curves.png            # AUC-ROC visualization
└── test_predictions.csv      # Detailed test predictions
```

### **Prediction Output**
```python
{
    "probability": 0.734,          # P(profitable trade)
    "threshold": 0.682,           # Optimal threshold
    "decision": "BUY",            # BUY/HOLD recommendation
    "confidence": 0.052,          # |probability - threshold|
    "expected_return": 0.021      # Historical avg return at this prob level
}
```

### **Performance Metrics**
```json
{
    "auc": {"train": 0.724, "val": 0.698, "test": 0.681},
    "threshold": 0.682,
    "test": {
        "coverage": 0.234,        # Trade 23.4% of time
        "avg_y_reg": 0.0167,      # 1.67% average profit
        "cum_y_reg": 0.245        # 24.5% total return
    },
    "features": ["atr_14_pct", "vol_ratio_20", ...]
}
```

---

## 🚀 **Production Deployment**

### **Real-Time Prediction Flow**
```python
# 1. Fetch current market data
current_ohlcv = get_live_data("RELIANCE")

# 2. Calculate features  
features = calculate_features(current_ohlcv)

# 3. Standardize using training parameters
standardized = apply_scaler(features, saved_scaler)

# 4. Model prediction
probability = model.predict_proba(standardized)[0][1]

# 5. Apply optimized threshold
decision = "BUY" if probability > optimal_threshold else "HOLD"
```

### **Production Challenges & Solutions**

#### **1. Threshold Drift**
**Problem**: Market conditions change, optimal threshold becomes outdated

**Solutions**:
- **Regular Retraining**: Monthly full model + threshold re-optimization
- **Adaptive Thresholds**: Daily micro-adjustments based on recent performance
- **Performance Monitoring**: Trigger retraining if performance degrades >20%

#### **2. Model Staleness**
**Problem**: Model trained on old data becomes less effective

**Solutions**:
- **Rolling Window Training**: Always use last 365 days of data
- **Incremental Updates**: Weekly feature importance analysis
- **Regime Detection**: Monitor for structural market changes

#### **3. Production Architecture**
```python
class ProductionTradingSystem:
    def __init__(self):
        self.base_threshold = 0.68        # From training
        self.adaptive_modifier = 0.0      # Dynamic adjustment ±0.05
        self.performance_tracker = []     # Last 200 trades
        self.last_retrain = datetime.now()
        
    def get_current_threshold(self):
        return np.clip(
            self.base_threshold + self.adaptive_modifier,
            0.55, 0.90  # Safety bounds
        )
    
    def update_performance(self, prediction, actual_return):
        self.performance_tracker.append((prediction, actual_return))
        
        # Adaptive adjustment
        if len(self.performance_tracker) >= 50:
            recent_perf = analyze_recent_performance()
            if recent_perf < historical_benchmark * 0.8:
                self.adaptive_modifier += 0.01  # Be more selective
            elif recent_perf > historical_benchmark * 1.2:
                self.adaptive_modifier -= 0.01  # Be less selective
                
        # Trigger retraining if needed
        if (datetime.now() - self.last_retrain).days > 30:
            self.retrain_if_needed()
```

### **Production Timeline**
- **Real-time**: Make trading decisions using current threshold
- **Daily**: Monitor performance, update adaptive modifiers
- **Weekly**: Analyze performance metrics, adjust risk parameters
- **Monthly**: Full model retraining with threshold re-optimization
- **Quarterly**: Strategy review and methodology validation

---

## 📊 **Performance Monitoring**

### **Key Metrics**
```python
production_metrics = {
    # Trading Performance
    "win_rate": 0.62,                    # 62% of trades profitable
    "average_return": 0.0167,            # 1.67% per trade
    "sharpe_ratio": 1.34,               # Risk-adjusted returns
    "max_drawdown": 0.087,              # Worst losing streak
    
    # Model Performance  
    "auc_rolling_30d": 0.673,           # Recent classification accuracy
    "threshold_stability": 0.015,       # Threshold variation
    "feature_drift": 0.023,             # Feature distribution drift
    
    # Operational Metrics
    "prediction_latency": "23ms",        # Time to make decision
    "uptime": 0.997,                    # System availability
    "data_freshness": "45s"             # Latest data age
}
```

### **Alert Conditions**
- **Performance Degradation**: Average return drops below 1.0%
- **High Volatility**: Sharpe ratio falls below 1.0
- **Model Drift**: AUC drops below 0.60
- **Threshold Instability**: Daily threshold changes > 5%

---

## 🧪 **Usage Examples**

### **Training a New Model**
```bash
# 1. Extract fresh data
cd backend/agents/ml/data_processing
python data_extractor.py

# 2. Build features  
python build_dataset.py data/raw/symbol=RELIANCE/timeframe=1d/bars.csv

# 3. Generate labels
python build_labels.py data/processed/symbol=RELIANCE/timeframe=1d/features.csv

# 4. Data quality checks
python qc_dataset.py data/processed/symbol=RELIANCE/timeframe=1d/labels.csv

# 5. Create train/val/test splits
python split_dataset.py data/processed/symbol=RELIANCE/timeframe=1d/labels.csv

# 6. Standardize training data
python standardize_train.py data/processed/symbol=RELIANCE/timeframe=1d/splits/train.csv

# 7. Train model with threshold optimization
cd ../training
python train_logistic.py --splits_dir ../data_processing/splits/
```

### **Making Predictions**
```python
import joblib
import numpy as np
import pandas as pd

# Load trained model
model_data = joblib.load('models/model_20241021_143022/logreg.joblib')
model = model_data['model']
features = model_data['features']  
threshold = model_data['threshold']

# Prepare current market data
current_features = extract_current_features("RELIANCE")
standardized = standardize_features(current_features)

# Predict
probability = model.predict_proba(standardized.reshape(1, -1))[0][1]
decision = "BUY" if probability > threshold else "HOLD"

print(f"Probability: {probability:.3f}")
print(f"Threshold: {threshold:.3f}")
print(f"Decision: {decision}")
```

---

## 🔧 **Configuration**

### **Timeframe Specifications**
```python
# config/config.py
DEFAULT_TIMEFRAMES = {
    "5m": TimeframeSpec(
        backfill_days=180,
        horizon_bars=12,        # 1 hour ahead
        est_cost_bps=8.0,       # 0.08% transaction cost
        interval="5min"
    ),
    "1d": TimeframeSpec(
        backfill_days=2000,
        horizon_bars=5,         # 5 days ahead  
        est_cost_bps=5.0,       # 0.05% transaction cost
        interval="day"
    )
}
```

### **Model Parameters**
```python
# Training configuration
TRAINING_CONFIG = {
    "test_size": 0.15,
    "val_size": 0.15,
    "random_state": 42,
    "class_weight": "balanced",
    "max_iter": 2000,
    "solver": "lbfgs",
    "penalty": "l2"
}

# Threshold optimization
THRESHOLD_CONFIG = {
    "min_percentile": 0.50,     # Start at 50th percentile
    "max_percentile": 0.99,     # End at 99th percentile  
    "n_thresholds": 50,         # Test 50 different values
    "min_trades": 5             # Minimum trades per threshold
}
```

---

## 🚨 **Important Notes**

### **Model Limitations**
1. **Single Stock**: Currently trained only on RELIANCE
2. **Technical Only**: No fundamental analysis or news sentiment
3. **Market Hours**: Designed for NSE trading hours (9:15 AM - 3:30 PM)
4. **Historical Bias**: Performance based on past patterns
5. **Transaction Costs**: Estimates may not reflect actual brokerage

### **Risk Warnings**
- **Past performance doesn't guarantee future results**
- **Model can have losing streaks**
- **Always use proper position sizing**
- **Monitor performance continuously**
- **Have stop-loss mechanisms**

### **Regulatory Compliance**
- **Model is interpretable** (logistic regression coefficients)
- **All decisions are auditable** (probability scores saved)
- **No insider information used** (only public market data)
- **Transaction costs included** in all calculations

---

## 📚 **References & Further Reading**

### **Technical Analysis**
- Average True Range (ATR) for volatility measurement
- MACD for momentum analysis  
- Bollinger Bands for volatility expansion/contraction
- Volume analysis for market participation

### **Machine Learning**
- Logistic Regression for binary classification
- ROC-AUC for model evaluation
- Cross-validation for model selection
- Feature scaling and normalization

### **Quantitative Finance**
- Log returns for statistical stability
- Transaction cost modeling
- Sharpe ratio for risk-adjusted returns
- Maximum drawdown for risk assessment

---

## 🔄 **Version History**

### **v1.0** (Current)
- ✅ Single stock (RELIANCE) support
- ✅ 40+ technical indicators
- ✅ Multi-timeframe predictions
- ✅ Threshold optimization
- ✅ Production monitoring framework

### **Planned v1.1**
- 🔄 Multi-stock universe expansion
- 🔄 Real-time feature drift detection
- 🔄 Advanced ensemble methods
- 🔄 Sector-relative analysis

### **Future Roadmap**
- 📋 Fundamental data integration
- 📋 News sentiment analysis
- 📋 Market regime detection
- 📋 Portfolio optimization
- 📋 Options strategy integration

---

*📊 **StockAnalyzer Pro ML System** - Turning market data into profitable trading decisions through intelligent threshold optimization.*