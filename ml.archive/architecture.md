
1) Pattern-based event classifier (primary)
•  Goal: For detected chart patterns (double top/bottom, H&S, triangle, flag, etc.), predict the probability that a take-profit (TP) is hit before stop-loss (SL) within a fixed horizon.
•  Data/labels: dataset_builder detects patterns on OHLCV series, then labels each event with y_success via first_hit_outcome over horizon_days using TP/SL thresholds (tp_pct/sl_pct).
•  Features: At event time, basic features include duration, volume_ratio20 (volume confirmation), trend_alignment (sign of prior ret_20), completion_status, and pattern_type (categorical). There’s room to add richer, strictly past-looking features.
•  Model: CatBoostClassifier with pattern_type as categorical, class weighting, optional calibration, and time-based evaluation in robust_evaluation (time split and walk-forward). Output is a calibrated probability of pattern success used to rank/filter trades.

2) Raw OHLCV predictive model (complementary)
•  Goal: Predict short-horizon price direction (classification) and magnitude (regression), plus volatility and market regime.
•  Features: Many engineered technical features (MAs/EMAs, RSI, stochastic, volatility ratios, Bollinger band metrics, ADX proxy, volume features, time features).
•  Models: RandomForestClassifier for direction (with isotonic calibration and a tuned threshold), GradientBoostingRegressor for magnitude; simple volatility forecasting and market regime classification from returns/volatility structure.

3) Hybrid decision layer
•  HybridMLEngine combines the pattern success probability with raw-model direction/magnitude, volatility outlook, and market regime to produce a consensus signal, combined confidence, and risk score for trading/execution.