#!/usr/bin/env python3
"""
Test Hybrid ML Engine

This script demonstrates the benefits of combining pattern-based ML with raw data ML
for comprehensive quantitative analysis.
"""

import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import logging

# Import ML engines
from ml.hybrid_ml_engine import hybrid_ml_engine
from ml.raw_data_engine import raw_data_ml_engine
from ml.inference import predict_probability

# Import data client
from zerodha_client import ZerodhaDataClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(days: int = 100) -> pd.DataFrame:
    """Generate sample stock data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # Generate realistic price data with trends and volatility
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))  # Starting at 100
    
    # Generate OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        volatility = 0.02
        high = price * (1 + abs(np.random.normal(0, volatility)))
        low = price * (1 - abs(np.random.normal(0, volatility)))
        open_price = price * (1 + np.random.normal(0, volatility * 0.5))
        volume = int(np.random.uniform(1000000, 5000000))
        
        data.append({
            'date': date,
            'open': open_price,
            'high': max(high, open_price, price),
            'low': min(low, open_price, price),
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df

def test_pattern_only_analysis():
    """Test pattern-based ML only."""
    print("🔍 Testing Pattern-Based ML Only")
    print("=" * 50)
    
    # Sample pattern features
    pattern_features = {
        "duration": 15.0,
        "volume_ratio": 1.2,
        "trend_alignment": 0.8,
        "completion": 0.9
    }
    
    # Get pattern prediction
    pattern_prob = predict_probability(pattern_features, "ascending_triangle")
    
    print(f"Pattern Success Probability: {pattern_prob:.2%}")
    print(f"Pattern Signal: {'Buy' if pattern_prob > 0.6 else 'Sell' if pattern_prob < 0.4 else 'Hold'}")
    print(f"Confidence: {pattern_prob:.2%}")
    print()

def test_raw_data_only_analysis():
    """Test raw data ML only."""
    print("📊 Testing Raw Data ML Only")
    print("=" * 50)
    
    # Generate sample data
    stock_data = generate_sample_data(100)
    
    # Train model
    print("Training raw data model...")
    success = raw_data_ml_engine.train_price_prediction_model(stock_data)
    
    if success:
        # Get predictions
        price_pred = raw_data_ml_engine.predict_price_movement(stock_data)
        volatility_pred = raw_data_ml_engine.predict_volatility(stock_data)
        market_regime = raw_data_ml_engine.classify_market_regime(stock_data)
        
        print(f"Price Direction: {price_pred.direction}")
        print(f"Price Magnitude: {price_pred.magnitude:.2%}")
        print(f"Price Confidence: {price_pred.confidence:.2%}")
        print(f"Volatility Regime: {volatility_pred.volatility_regime}")
        print(f"Market Regime: {market_regime.regime}")
        print(f"Market Strength: {market_regime.strength:.2%}")
    else:
        print("❌ Raw data model training failed")
    print()

def test_hybrid_analysis():
    """Test hybrid ML combining both approaches."""
    print("🚀 Testing Hybrid ML (Pattern + Raw Data)")
    print("=" * 50)
    
    # Generate sample data
    stock_data = generate_sample_data(100)
    
    # Sample pattern features
    pattern_features = {
        "duration": 15.0,
        "volume_ratio": 1.2,
        "trend_alignment": 0.8,
        "completion": 0.9
    }
    
    # Train models
    print("Training hybrid models...")
    success = hybrid_ml_engine.train_models(stock_data)
    
    if success:
        # Get comprehensive analysis
        analysis = hybrid_ml_engine.get_comprehensive_analysis(
            stock_data, pattern_features, "ascending_triangle"
        )
        
        print("📈 HYBRID PREDICTION RESULTS:")
        print(f"Consensus Signal: {analysis['hybrid_prediction']['consensus_signal']}")
        print(f"Combined Confidence: {analysis['hybrid_prediction']['combined_confidence']:.2%}")
        print(f"Risk Score: {analysis['hybrid_prediction']['risk_score']:.1f}")
        print(f"Recommendation: {analysis['hybrid_prediction']['recommendation']}")
        print()
        
        print("🔍 PATTERN ANALYSIS:")
        print(f"Success Probability: {analysis['pattern_analysis']['success_probability']:.2%}")
        print(f"Pattern Signal: {analysis['pattern_analysis']['signal']}")
        print()
        
        print("📊 PRICE ANALYSIS:")
        print(f"Direction: {analysis['price_analysis']['direction']}")
        print(f"Expected Move: {analysis['price_analysis']['expected_move']}")
        print(f"Confidence: {analysis['price_analysis']['confidence']:.2%}")
        print()
        
        print("🌍 MARKET CONTEXT:")
        print(f"Volatility Regime: {analysis['market_context']['volatility']['regime']}")
        print(f"Market Regime: {analysis['market_context']['market_regime']['regime']}")
        print(f"Regime Strength: {analysis['market_context']['market_regime']['strength']:.2%}")
        print()
        
        print("⚠️ RISK ASSESSMENT:")
        print(f"Risk Level: {analysis['risk_assessment']['risk_level']}")
        print(f"Position Size: {analysis['risk_assessment']['position_sizing']:.2%}")
    else:
        print("❌ Hybrid model training failed")
    print()

def compare_approaches():
    """Compare the three approaches."""
    print("📊 COMPARISON: Pattern vs Raw Data vs Hybrid")
    print("=" * 60)
    
    # Generate consistent test data
    stock_data = generate_sample_data(100)
    pattern_features = {
        "duration": 15.0,
        "volume_ratio": 1.2,
        "trend_alignment": 0.8,
        "completion": 0.9
    }
    
    # Train models
    hybrid_ml_engine.train_models(stock_data)
    
    # Get predictions from all approaches
    pattern_prob = predict_probability(pattern_features, "ascending_triangle")
    price_pred = raw_data_ml_engine.predict_price_movement(stock_data)
    hybrid_analysis = hybrid_ml_engine.get_comprehensive_analysis(
        stock_data, pattern_features, "ascending_triangle"
    )
    
    # Create comparison table
    comparison = {
        "Approach": ["Pattern-Only", "Raw Data-Only", "Hybrid"],
        "Signal": [
            "Buy" if pattern_prob > 0.6 else "Sell" if pattern_prob < 0.4 else "Hold",
            "Buy" if price_pred.direction == "up" else "Sell" if price_pred.direction == "down" else "Hold",
            hybrid_analysis['hybrid_prediction']['consensus_signal']
        ],
        "Confidence": [
            f"{pattern_prob:.1%}",
            f"{price_pred.confidence:.1%}",
            f"{hybrid_analysis['hybrid_prediction']['combined_confidence']:.1%}"
        ],
        "Risk Assessment": [
            "Medium",
            "Medium", 
            hybrid_analysis['risk_assessment']['risk_level']
        ],
        "Position Sizing": [
            "N/A",
            "N/A",
            f"{hybrid_analysis['risk_assessment']['position_sizing']:.1%}"
        ],
        "Market Context": [
            "Limited",
            "Basic",
            "Comprehensive"
        ]
    }
    
    # Print comparison
    df = pd.DataFrame(comparison)
    print(df.to_string(index=False))
    print()
    
    print("🎯 KEY BENEFITS OF HYBRID APPROACH:")
    print("✅ Combines pattern success probability with price movement prediction")
    print("✅ Provides market regime classification and volatility forecasting")
    print("✅ Offers comprehensive risk assessment and position sizing")
    print("✅ Creates consensus signals from multiple data sources")
    print("✅ Adapts to different market conditions")

def main():
    """Run all tests."""
    print("🧪 HYBRID ML ENGINE TESTING")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test individual approaches
    test_pattern_only_analysis()
    test_raw_data_only_analysis()
    test_hybrid_analysis()
    
    # Compare approaches
    compare_approaches()
    
    print("\n" + "=" * 60)
    print("✅ HYBRID ML ENGINE TESTING COMPLETED")
    print("The hybrid approach provides the most comprehensive quantitative analysis!")

if __name__ == "__main__":
    main()
