# 📊 ML Data Processing Pipeline

Complete guide to the machine learning data processing system for StockAnalyzer Pro.

---

## 🎯 Overview

This directory contains a complete data processing pipeline that transforms raw OHLCV market data into ML-ready datasets with 40+ engineered features, forward-looking labels, and comprehensive quality control.

### Pipeline Architecture

```
Raw Data → Feature Engineering → Label Generation → QC → Consolidation → Splitting → Standardization → Ready for Training
```

---

## 📁 Directory Structure

```
data_processing/
├── build_dataset.py              # Core: 40+ technical indicator engineering
├── build_labels.py               # Label generation with forward returns
├── combine_processed.py          # Multi-stock dataset consolidation
├── multi_stock_processor.py      # Unified end-to-end pipeline (RECOMMENDED)
├── split_and_standardize.py      # Train/val/test splitting + normalization
├── qc_dataset.py                 # Quality control and validation
├── clean_dataset.py              # Data cleaning utilities
├── corr_matrix.py                # Feature correlation analysis
├── data_extractor.py             # Raw data extraction from APIs
├── build_full_dataset.py         # Full pipeline for single stock
├── run_build_full_on_dir.py      # Batch processing for directories
├── run_qc_on_dir.py              # Batch QC for directories
├── split_qc.py                   # Split quality analysis
└── standardize_train.py          # Training data standardization
```

---

## 🚀 Quick Start

### Option 1: Multi-Stock Unified Pipeline (Recommended)

**Use this for production workflows** - processes multiple stocks and timeframes in one unified pipeline.

```bash
cd backend/agents/ml/data_processing

python multi_stock_processor.py \
  --base_dir ../data/raw \
  --output_dir ../data/processed/multi_stock \
  --symbols RELIANCE TCS INFY HDFCBANK ICICIBANK \
  --timeframes 5m 15m 1h 1d \
  --train_pct 0.6 \
  --val_pct 0.2 \
  --test_pct 0.2
```

**Output:**
```
backend/agents/ml/data/processed/multi_stock/run_20241022_124619/
├── train.csv                      # Training split (standardized)
├── val.csv                        # Validation split (standardized)
├── test.csv                       # Test split (standardized)
├── scaler.json                    # Standardization parameters
└── processing_metadata.json       # Complete processing info
```

### Option 2: Step-by-Step Individual Pipeline

**Use this for debugging or custom workflows** - gives you control over each step.

```bash
cd backend/agents/ml/data_processing

# Step 1: Feature Engineering
python build_dataset.py \
  ../data/raw/symbol=RELIANCE/timeframe=1d/bars.csv \
  --output_csv ../data/processed/symbol=RELIANCE/timeframe=1d/features.csv

# Step 2: Label Generation
python build_labels.py \
  ../data/processed/symbol=RELIANCE/timeframe=1d/features.csv \
  --timeframe 1d

# Step 3: Quality Control
python qc_dataset.py \
  ../data/processed/symbol=RELIANCE/timeframe=1d/labels.csv

# Step 4: Combine Multiple Stocks (optional)
python combine_processed.py \
  --processed_dir ../data/processed \
  --symbols RELIANCE TCS INFY \
  --timeframes 1d 1h 15m

# Step 5: Split and Standardize
python split_and_standardize.py \
  --input_csv ../data/data_20241022_120000/combined_raw.csv \
  --train_pct 0.7 \
  --val_pct 0.15 \
  --test_pct 0.15
```

---

## 📋 Component Documentation

### 1. `build_dataset.py` - Feature Engineering

**Purpose:** Transforms raw OHLCV data into 40+ technical indicators.

**Features Generated:**

#### Volatility Features (3)
- `atr_14_pct`: Average True Range as % of price
- `atr_vol_20`: 20-period volatility of ATR
- `range_pct`: Daily high-low range as % of close

#### Trend & Moving Averages (2)
- `dist_sma50_pct`: Distance from 50-period SMA
- `macd_hist`: MACD histogram

#### Bollinger Bands (1)
- `bb_bw_20`: Bollinger Band bandwidth

#### Volume Features (5)
- `vol_ratio_20`: Current volume vs 20-day average
- `vol_cv_20`: Volume coefficient of variation
- `cmf_20`: Chaikin Money Flow
- `up_down_vol_ratio_20`: Volume on up days vs down days
- `ret_vol_corr_20`: Return-volume correlation

#### Price Position Features (4)
- `pct_dist_to_20_high`: Distance from 20-day high
- `breakout_up_20`: Binary flag for upward breakouts
- `breakout_down_20`: Binary flag for downward breakouts
- `vwap_dist`: Distance from VWAP

#### VWAP Features (2)
- `vwap_dist`: Distance from VWAP
- `vwap_slope_5`: 5-period slope of VWAP

#### Candlestick Patterns (8)
- `wick_to_body_ratio`: Ratio of wicks to candle body
- `inside_bar`: Inside bar pattern
- `engulfing`: Engulfing pattern
- `gap_pct`: Gap percentage between sessions
- `up_streak`: Consecutive up days
- `down_streak`: Consecutive down days
- `wick_up_streak_3`: Wick-up streak (3-period)
- `wick_down_streak_3`: Wick-down streak (3-period)

#### Statistical Features (2)
- `ret_skew_20`: 20-period return skewness
- `ret_kurt_20`: 20-period return kurtosis

#### Calendar Features (8)
- `dow`: Day of week (0-6)
- `dow_sin`, `dow_cos`: Cyclical day-of-week encoding
- `hour`: Hour of day (0-23)
- `hour_sin`, `hour_cos`: Cyclical hour encoding

**Usage:**
```bash
python build_dataset.py <input_csv> [--output_csv OUTPUT] [--drop_warmup]

# Auto-detects structure: data/raw/symbol=X/timeframe=Y/bars.csv
# Auto-outputs to:       data/processed/symbol=X/timeframe=Y/features.csv
```

**Key Parameters:**
- `input_csv`: Path to raw OHLCV CSV with columns: open, high, low, close, volume
- `--output_csv`: Optional custom output path
- `--drop_warmup`: Drop initial rows with NaN due to indicator warmup (recommended)

---

### 2. `build_labels.py` - Label Generation

**Purpose:** Creates forward-looking binary (y_cls) and continuous (y_reg) labels.

**Label Calculation:**
```python
# Step 1: Calculate forward return
close_now = close[t]
close_future = close[t + horizon_bars]
log_return = ln(close_future / close_now)

# Step 2: Subtract transaction costs
cost = est_cost_bps / 10000.0  # Convert basis points to decimal
y_reg = log_return - cost

# Step 3: Binary classification
y_cls = 1 if y_reg > 0 else 0  # 1 = profitable, 0 = not profitable
```

**Timeframe Configurations:**
| Timeframe | Horizon Bars | Transaction Cost | Forward Period |
|-----------|--------------|------------------|----------------|
| 5m        | 12           | 8 bps (0.08%)    | 1 hour         |
| 15m       | 8            | 7 bps (0.07%)    | 2 hours        |
| 1h        | 12           | 6 bps (0.06%)    | 12 hours       |
| 1d        | 5            | 5 bps (0.05%)    | 5 days         |

**Usage:**
```bash
python build_labels.py <features_csv> [--timeframe TIMEFRAME] [--output_csv OUTPUT]

# Auto-detects timeframe from path pattern: timeframe=XXX
```

**Key Points:**
- Removes last `horizon_bars` rows (no future data available)
- Transaction costs are **realistic estimates** based on typical slippage + fees
- Uses **log returns** for statistical stability

---

### 3. `combine_processed.py` - Dataset Consolidation

**Purpose:** Combines processed data from multiple symbols/timeframes into a single unified dataset.

**What it does:**
1. Discovers all available label files in processed directory
2. Filters by specified symbols and timeframes
3. Adds `symbol` and `timeframe` columns for identification
4. Concatenates all data into `combined_raw.csv`
5. Generates metadata with coverage statistics

**Usage:**
```bash
python combine_processed.py \
  --processed_dir ../data/processed \
  --symbols RELIANCE TCS INFY HDFCBANK \
  --timeframes 5m 15m 1h 1d \
  --out_root ../data \
  --run_dir data_20241022_120000
```

**Output:**
```
backend/agents/ml/data/data_20241022_120000/
├── combined_raw.csv              # All data consolidated
└── combine_metadata.json         # Coverage and metadata
```

**Metadata includes:**
- Per-symbol/timeframe row counts
- Date ranges for each combination
- Column inventory
- File paths and processing errors

---

### 4. `multi_stock_processor.py` - Unified Pipeline (RECOMMENDED)

**Purpose:** End-to-end pipeline from raw data to train/val/test splits in one command.

**Complete Workflow:**
```
1. Consolidate raw OHLCV data
   ↓
2. Apply feature engineering uniformly
   ↓
3. Generate labels with transaction costs
   ↓
4. Quality control (outliers, missing data)
   ↓
5. Temporal train/val/test splits
   ↓
6. Standardize using training statistics
   ↓
7. Save ready-to-train datasets
```

**Usage:**
```bash
python multi_stock_processor.py \
  --base_dir ../data/raw \
  --output_dir ../data/processed/multi_stock \
  --symbols RELIANCE TCS INFY HDFCBANK ICICIBANK \
  --timeframes 5m 15m 1h 1d \
  --train_pct 0.6 \
  --val_pct 0.2 \
  --test_pct 0.2 \
  --label_horizon 12 \
  --label_threshold 0.03 \
  --suffix "_experiment_1"
```

**Key Features:**
- **Uniform processing**: All stocks/timeframes use identical feature engineering
- **Temporal splits**: Maintains chronological order within each symbol/timeframe
- **Automatic QC**: Removes outliers, validates data quality
- **Standardization**: Z-score normalization using training set statistics
- **Metadata tracking**: Complete provenance and processing statistics

**Parameters:**
- `--base_dir`: Root directory containing raw data (symbol=*/timeframe=*/)
- `--output_dir`: Where to save processed datasets
- `--symbols`: List of stock symbols to process
- `--timeframes`: List of timeframes (5m, 15m, 1h, 1d)
- `--train_pct`: Training split percentage (default: 0.6)
- `--val_pct`: Validation split percentage (default: 0.2)
- `--test_pct`: Test split percentage (default: 0.2)
- `--label_horizon`: Custom horizon bars (overrides timeframe defaults)
- `--label_threshold`: Custom threshold for binary labels
- `--suffix`: Optional suffix for output directory name

---

### 5. `split_and_standardize.py` - Data Splitting

**Purpose:** Split combined dataset into train/val/test and standardize features.

**Splitting Strategy:**
- **Temporal splits**: Maintains time order within each symbol/timeframe group
- **No data leakage**: Validation/test never come before training
- **Group-aware**: Each symbol/timeframe is split independently

**Standardization:**
```python
# Compute from training data only
mean = training_data[feature].mean()
std = training_data[feature].std()

# Apply to all splits
standardized = (data[feature] - mean) / std
```

**Excluded from standardization:**
- Categorical: `symbol`, `timeframe`, `dow`, `hour`
- Targets: `y_cls`, `y_reg`
- Raw OHLCV: `open`, `high`, `low`, `close`, `volume`
- Binary flags: Any 0/1 indicator features

**Usage:**
```bash
python split_and_standardize.py \
  --input_csv ../data/data_20241022/combined_raw.csv \
  --train_pct 0.7 \
  --val_pct 0.15 \
  --test_pct 0.15
```

**Output:**
- `train.csv`: Raw training split
- `val.csv`: Raw validation split
- `test.csv`: Raw test split
- `train_standardized.csv`: Standardized training data
- `scaler.json`: Standardization parameters (for production inference)
- `split_metadata.json`: Split statistics

---

### 6. `qc_dataset.py` - Quality Control

**Purpose:** Validate data quality and remove problematic samples.

**Checks Performed:**

1. **Missing Data**
   - Max 10% NaN features per row
   - Removes rows exceeding threshold

2. **Outlier Detection**
   - IQR method with 3.0x factor
   - Per-feature outlier removal
   - Option for z-score method

3. **Minimum Sample Size**
   - Requires 100+ samples per symbol/timeframe group
   - Removes groups with insufficient data

4. **Feature Distribution**
   - Checks for inf, -inf values
   - Validates feature ranges
   - Reports distribution statistics

5. **Label Balance**
   - Reports positive/negative class distribution
   - Warns if severely imbalanced

**Usage:**
```bash
python qc_dataset.py <labels_csv> [--output_csv OUTPUT]

# Auto-outputs to: labels_capped_cleaned.csv
```

**Quality Report:**
```json
{
  "input_rows": 50000,
  "output_rows": 48234,
  "removed": 1766,
  "removed_pct": 3.5,
  "features_checked": 42,
  "outliers_removed": 1234,
  "na_removed": 532,
  "label_distribution": {
    "y_cls=0": 0.543,
    "y_cls=1": 0.457
  }
}
```

---

## 📊 Dataset Formats

### Raw Data Format

**Expected structure:**
```
backend/agents/ml/data/raw/
├── symbol=RELIANCE/
│   ├── timeframe=5m/
│   │   └── bars.csv          # or bars.parquet
│   ├── timeframe=15m/
│   │   └── bars.csv
│   ├── timeframe=1h/
│   │   └── bars.csv
│   └── timeframe=1d/
│       └── bars.csv
├── symbol=TCS/
│   └── ...
```

**Required columns:**
- `datetime` (index): Timestamp in UTC
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

**Supported formats:**
- CSV with datetime index
- Parquet with datetime index

---

### Processed Data Format

**After feature engineering:**
```
backend/agents/ml/data/processed/
├── symbol=RELIANCE/
│   └── timeframe=1d/
│       ├── features.csv                    # 40+ engineered features
│       ├── labels.csv                      # features + y_cls + y_reg
│       └── labels_capped_cleaned.csv       # QC-applied version
```

**Column groups:**
- **OHLCV**: `open`, `high`, `low`, `close`, `volume`
- **Features**: All 40+ technical indicators
- **Metadata**: `symbol`, `timeframe` (if combined)
- **Targets**: `y_cls` (binary), `y_reg` (continuous)

---

### Combined Dataset Format

**After consolidation:**
```
backend/agents/ml/data/data_20241022_120000/
├── combined_raw.csv                   # All symbols/timeframes
├── combine_metadata.json              # Coverage info
├── train.csv                          # Training split
├── val.csv                            # Validation split
├── test.csv                           # Test split
├── train_standardized.csv             # Standardized training
├── scaler.json                        # Standardization params
└── split_metadata.json                # Split statistics
```

**Additional columns in combined data:**
- `symbol`: Stock symbol identifier
- `timeframe`: Timeframe identifier
- `seq_id`: Sequence ID within each symbol/timeframe

---

## 🔧 Advanced Usage

### Batch Processing

**Process all stocks in a directory:**
```bash
python run_build_full_on_dir.py \
  --base_dir ../data/raw \
  --output_dir ../data/processed
```

**Batch QC on all label files:**
```bash
python run_qc_on_dir.py \
  --processed_dir ../data/processed
```

### Feature Correlation Analysis

**Generate correlation matrix:**
```bash
python corr_matrix.py \
  --labels_csv ../data/processed/symbol=RELIANCE/timeframe=1d/labels.csv \
  --output_png correlation_matrix.png \
  --threshold 0.95
```

**Output:**
- Visual correlation heatmap
- List of highly correlated features (>threshold)
- Recommendations for feature removal

### Custom Label Configurations

**Override default timeframe settings:**
```python
# In multi_stock_processor.py
label_config = {
    "5m": {"horizon": 24, "threshold": 0.015, "method": "fixed_threshold"},
    "15m": {"horizon": 16, "threshold": 0.020, "method": "fixed_threshold"},
    "1h": {"horizon": 18, "threshold": 0.025, "method": "fixed_threshold"},
    "1d": {"horizon": 10, "threshold": 0.040, "method": "fixed_threshold"}
}
```

---

## 🎯 Best Practices

### 1. Data Collection
- **Minimum history**: 180 days for 5m, 2000 days for 1d
- **Data quality**: Validate no missing trading sessions
- **Timezone**: Always use UTC for consistency

### 2. Feature Engineering
- **Warmup period**: Always use `--drop_warmup` to remove NaN-filled initial rows
- **Indicator stability**: Check for inf/-inf in volatile markets
- **Consistent calculations**: Use same parameters across all stocks

### 3. Label Generation
- **Transaction costs**: Update `est_cost_bps` based on actual brokerage
- **Horizon selection**: Balance between predictability and actionability
- **Label distribution**: Aim for 40-60% positive class balance

### 4. Quality Control
- **Outlier threshold**: Use 3.0x IQR for conservative removal
- **Missing data**: Max 10% per row, but preferably <5%
- **Minimum samples**: At least 100 periods, preferably 200+

### 5. Data Splitting
- **Temporal splits**: NEVER shuffle - maintain time order
- **Split ratios**: 60/20/20 or 70/15/15 are standard
- **Group awareness**: Split within each symbol/timeframe independently

### 6. Standardization
- **Training only**: Compute statistics from training set ONLY
- **Apply uniformly**: Use same scaler for val/test/production
- **Binary exclusion**: Don't standardize 0/1 flags
- **Save scaler**: Store scaler.json for production deployment

---

## 🐛 Troubleshooting

### Issue: "No data found for symbol/timeframe"
**Solution:** Check raw data directory structure matches `symbol=X/timeframe=Y/bars.csv`

### Issue: "Too many NaN values after feature engineering"
**Solution:** 
- Check input data quality
- Use `--drop_warmup` flag
- Reduce minimum required period for indicators

### Issue: "All samples removed during QC"
**Solution:**
- Increase outlier factor (e.g., 5.0 instead of 3.0)
- Check for data corruption in input
- Validate date ranges and missing data

### Issue: "Label imbalance >80%"
**Solution:**
- Adjust `est_cost_bps` (may be too high/low)
- Check horizon bars (may be too short/long)
- Verify label calculation logic

### Issue: "Standardization produces inf/-inf"
**Solution:**
- Check for zero-variance features in training set
- Remove constant features before standardization
- Validate no corrupted rows in training data

---

## 📚 Related Documentation

- **Main ML README**: `../README.md` - Overview of entire ML system
- **Training Guide**: `../training/README.md` - Model training workflows (if exists)
- **Config Reference**: `../config/config.py` - Timeframe specs and defaults
- **API Docs**: `../../README.md` - Data extraction from APIs

---

## 🔄 Workflow Summary

**Production Pipeline (Recommended):**
```bash
# 1. Multi-stock unified processing
python multi_stock_processor.py \
  --symbols RELIANCE TCS INFY HDFCBANK ICICIBANK \
  --timeframes 5m 15m 1h 1d

# Output: train.csv, val.csv, test.csv, scaler.json

# 2. Train models (see ../training/)
cd ../training
python train_multi_models.py --splits_dir [output_from_step_1]
```

**Development/Debug Pipeline:**
```bash
# 1. Feature engineering
python build_dataset.py <raw_csv> --drop_warmup

# 2. Label generation  
python build_labels.py <features_csv> --timeframe 1d

# 3. Quality control
python qc_dataset.py <labels_csv>

# 4. Consolidate multiple stocks
python combine_processed.py --symbols RELIANCE TCS --timeframes 1d

# 5. Split and standardize
python split_and_standardize.py --input_csv <combined_csv>

# 6. Train models
cd ../training
python train_multi_models.py --splits_dir [splits_dir]
```

---

## 📝 Additional Utility Files

The following utility scripts are available but not covered in detail above:

### `data_extractor.py`
Extracts raw OHLCV data from Zerodha KiteConnect API and saves in structured format.

### `build_full_dataset.py`
Combines feature engineering + label generation in a single script for individual stocks.

### `run_build_full_on_dir.py`
Batch processing wrapper that runs `build_full_dataset.py` on all stocks in a directory.

### `run_qc_on_dir.py`
Batch quality control that processes all label files in a directory structure.

### `qc_combined.py`
Quality control specifically designed for combined multi-stock datasets.

### `split_dataset.py`
Standalone temporal splitting (without standardization) for combined datasets.

### `split_qc.py`
Analyzes train/val/test splits for data drift, distribution shifts, and quality issues.

### `standardize_train.py`
Standalone feature standardization for training data only.

### `clean_dataset.py`
Data cleaning utilities for handling missing values, duplicates, and data type issues.

### `corr_matrix.py`
Generates feature correlation matrices and identifies highly correlated feature pairs.

**Note:** For production workflows, prefer using `multi_stock_processor.py` which orchestrates these components in an optimized pipeline. The individual utilities are best used for debugging, experimentation, or custom workflows.

---

**Questions or Issues?** Check the main project README or create an issue in the repository.
