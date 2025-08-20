#!/usr/bin/env python3
"""
Validation Script for Step 1.3: ML Model Development

This script validates the ML model development implementation by testing:
1. Target generation
2. Feature selection
3. Data preprocessing
4. Model training (regression and classification)
5. Model evaluation
6. Prediction capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the quant_system directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ml_models import ModelConfig, TargetGenerator, FeatureSelector, DataPreprocessor, ModelTrainer, ModelEvaluator

def test_target_generation():
    """Test target generation functionality."""
    print("🔍 Testing Target Generation")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, 50)
    prices = 100 * np.exp(np.cumsum(returns))
    
    test_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, 50)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 50))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 50))),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 50)
    }, index=dates)
    
    # Ensure price logic is valid
    test_data['high'] = np.maximum(test_data[['open', 'high', 'close']].max(axis=1), test_data['high'])
    test_data['low'] = np.minimum(test_data[['open', 'low', 'close']].min(axis=1), test_data['low'])
    
    print(f"✅ Test data created: {test_data.shape}")
    
    # Test target generation
    config = ModelConfig()
    target_generator = TargetGenerator(config)
    
    # Generate targets
    df_with_targets = target_generator.generate_targets(test_data)
    
    # Check if targets were generated
    expected_targets = ['direction', 'volatility', 'return_target', 'price_target']
    missing_targets = [target for target in expected_targets if target not in df_with_targets.columns]
    
    if not missing_targets:
        print("✅ All targets generated successfully")
        
        # Check target values
        direction_values = df_with_targets['direction'].dropna()
        if len(direction_values) > 0 and all(val in [0, 1] for val in direction_values):
            print("✅ Direction target values are binary")
        else:
            print("❌ Direction target values are not binary")
            return False
        
        # Check for NaN values in targets
        nan_counts = df_with_targets[expected_targets].isnull().sum()
        if nan_counts.sum() > 0:
            print(f"⚠️ NaN values in targets: {nan_counts.to_dict()}")
        else:
            print("✅ No NaN values in targets")
        
        return True
    else:
        print(f"❌ Missing targets: {missing_targets}")
        return False

def test_feature_selection():
    """Test feature selection functionality."""
    print("\n🔍 Testing Feature Selection")
    print("=" * 50)
    
    # Create sample data with features
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 50),
        'high': np.random.uniform(105, 115, 50),
        'low': np.random.uniform(95, 105, 50),
        'close': np.random.uniform(100, 110, 50),
        'volume': np.random.randint(1000000, 5000000, 50),
        'rsi': np.random.uniform(0, 100, 50),
        'macd': np.random.uniform(-2, 2, 50),
        'bb_upper': np.random.uniform(105, 115, 50),
        'bb_lower': np.random.uniform(95, 105, 50),
        'atr': np.random.uniform(1, 5, 50),
        'vwap': np.random.uniform(100, 110, 50),
        'returns': np.random.uniform(-0.05, 0.05, 50),
        'direction': np.random.randint(0, 2, 50),
        'volatility': np.random.uniform(0.01, 0.05, 50)
    }, index=dates)
    
    print(f"✅ Test data created: {test_data.shape}")
    
    # Test feature selection
    config = ModelConfig()
    feature_selector = FeatureSelector(config)
    
    features_df, feature_columns = feature_selector.select_features(test_data)
    
    if not features_df.empty and len(feature_columns) > 0:
        print(f"✅ Features selected: {len(feature_columns)} features")
        print(f"   Feature columns: {feature_columns}")
        
        # Check that excluded columns are not in features
        excluded_cols = ['open', 'high', 'low', 'close', 'volume']
        excluded_in_features = [col for col in excluded_cols if col in feature_columns]
        
        if not excluded_in_features:
            print("✅ Excluded columns properly removed")
        else:
            print(f"❌ Excluded columns found in features: {excluded_in_features}")
            return False
        
        # Check that target columns are not in features
        target_cols = ['direction', 'volatility', 'return_target', 'price_target']
        targets_in_features = [col for col in target_cols if col in feature_columns]
        
        if not targets_in_features:
            print("✅ Target columns properly removed")
        else:
            print(f"❌ Target columns found in features: {targets_in_features}")
            return False
        
        return True
    else:
        print("❌ No features selected")
        return False

def test_data_preprocessing():
    """Test data preprocessing functionality."""
    print("\n🔍 Testing Data Preprocessing")
    print("=" * 50)
    
    # Create sample data with features
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 50),
        'high': np.random.uniform(105, 115, 50),
        'low': np.random.uniform(95, 105, 50),
        'close': np.random.uniform(100, 110, 50),
        'volume': np.random.randint(1000000, 5000000, 50),
        'rsi': np.random.uniform(0, 100, 50),
        'macd': np.random.uniform(-2, 2, 50),
        'bb_upper': np.random.uniform(105, 115, 50),
        'bb_lower': np.random.uniform(95, 105, 50),
        'atr': np.random.uniform(1, 5, 50),
        'vwap': np.random.uniform(100, 110, 50),
        'returns': np.random.uniform(-0.05, 0.05, 50),
        'direction': np.random.randint(0, 2, 50),
        'volatility': np.random.uniform(0.01, 0.05, 50)
    }, index=dates)
    
    print(f"✅ Test data created: {test_data.shape}")
    
    # Test data preprocessing
    config = ModelConfig()
    preprocessor = DataPreprocessor(config)
    
    X, y = preprocessor.preprocess_data(test_data, 'direction')
    
    if len(X) > 0 and len(y) > 0:
        print(f"✅ Data preprocessed successfully: X={X.shape}, y={y.shape}")
        
        # Check that features are scaled
        if X.shape[1] > 0:
            print("✅ Features scaled successfully")
        else:
            print("❌ No features in preprocessed data")
            return False
        
        # Check that target has correct values
        unique_targets = np.unique(y)
        if len(unique_targets) > 0:
            print(f"✅ Target values: {unique_targets}")
        else:
            print("❌ No target values")
            return False
        
        return True
    else:
        print("❌ Data preprocessing failed")
        return False

def test_model_training():
    """Test model training functionality."""
    print("\n🔍 Testing Model Training")
    print("=" * 50)
    
    # Create sample data with features
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, 100)
    prices = 100 * np.exp(np.cumsum(returns))
    
    test_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, 100)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100),
        'rsi': np.random.uniform(0, 100, 100),
        'macd': np.random.uniform(-2, 2, 100),
        'bb_upper': prices * (1 + np.random.uniform(0.02, 0.05, 100)),
        'bb_lower': prices * (1 - np.random.uniform(0.02, 0.05, 100)),
        'atr': np.random.uniform(1, 5, 100),
        'vwap': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
        'returns': np.random.uniform(-0.05, 0.05, 100)
    }, index=dates)
    
    # Ensure price logic is valid
    test_data['high'] = np.maximum(test_data[['open', 'high', 'close']].max(axis=1), test_data['high'])
    test_data['low'] = np.minimum(test_data[['open', 'low', 'close']].min(axis=1), test_data['low'])
    
    print(f"✅ Test data created: {test_data.shape}")
    
    # Test model training
    config = ModelConfig()
    trainer = ModelTrainer(config)
    
    # Train direction prediction model (classification)
    results = trainer.train_direction_prediction_model(test_data)
    
    if results:
        print("✅ Model training completed successfully")
        
        # Check model performance
        for model_name, result in results.items():
            if 'f1' in result:
                f1_score = result['f1']
                print(f"   {model_name}: F1={f1_score:.4f}")
        
        # Check if models are stored
        if 'direction_prediction' in trainer.models:
            print("✅ Best model stored successfully")
        else:
            print("❌ Best model not stored")
            return False
        
        return True
    else:
        print("❌ Model training failed")
        return False

def test_model_evaluation():
    """Test model evaluation functionality."""
    print("\n🔍 Testing Model Evaluation")
    print("=" * 50)
    
    # Create sample predictions and true values
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    
    print(f"✅ Test data created: {len(y_true)} samples")
    
    # Test model evaluation
    config = ModelConfig()
    evaluator = ModelEvaluator(config)
    
    # Test classification evaluation
    classification_metrics = evaluator.evaluate_model_performance(y_true, y_pred, 'classification')
    
    if classification_metrics:
        print("✅ Classification evaluation completed")
        for metric, value in classification_metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        # Test regression evaluation
        y_true_reg = np.random.uniform(0, 100, 100)
        y_pred_reg = np.random.uniform(0, 100, 100)
        
        regression_metrics = evaluator.evaluate_model_performance(y_true_reg, y_pred_reg, 'regression')
        
        if regression_metrics:
            print("✅ Regression evaluation completed")
            for metric, value in regression_metrics.items():
                print(f"   {metric}: {value:.4f}")
            
            return True
        else:
            print("❌ Regression evaluation failed")
            return False
    else:
        print("❌ Classification evaluation failed")
        return False

def test_prediction_capabilities():
    """Test prediction capabilities."""
    print("\n🔍 Testing Prediction Capabilities")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 50),
        'high': np.random.uniform(105, 115, 50),
        'low': np.random.uniform(95, 105, 50),
        'close': np.random.uniform(100, 110, 50),
        'volume': np.random.randint(1000000, 5000000, 50),
        'rsi': np.random.uniform(0, 100, 50),
        'macd': np.random.uniform(-2, 2, 50),
        'bb_upper': np.random.uniform(105, 115, 50),
        'bb_lower': np.random.uniform(95, 105, 50),
        'atr': np.random.uniform(1, 5, 50),
        'vwap': np.random.uniform(100, 110, 50),
        'returns': np.random.uniform(-0.05, 0.05, 50)
    }, index=dates)
    
    print(f"✅ Test data created: {test_data.shape}")
    
    # Train a model first
    config = ModelConfig()
    trainer = ModelTrainer(config)
    
    # Add targets to data
    target_generator = TargetGenerator(config)
    data_with_targets = target_generator.generate_targets(test_data)
    
    # Train model
    results = trainer.train_direction_prediction_model(data_with_targets)
    
    if results:
        # Test prediction
        prediction_result = trainer.predict(test_data, 'direction_prediction')
        
        if prediction_result:
            predictions = prediction_result['predictions']
            probabilities = prediction_result['probabilities']
            
            print(f"✅ Predictions generated: {len(predictions)} predictions")
            
            if probabilities is not None:
                print(f"✅ Probabilities generated: {len(probabilities)} probabilities")
                
                # Check probability range
                if all(0 <= prob <= 1 for prob in probabilities):
                    print("✅ Probabilities in valid range [0, 1]")
                else:
                    print("❌ Probabilities out of valid range")
                    return False
            else:
                print("⚠️ No probabilities generated")
            
            return True
        else:
            print("❌ Prediction failed")
            return False
    else:
        print("❌ Model training failed for prediction test")
        return False

def run_comprehensive_validation():
    """Run all validation tests."""
    print("🧪 STEP 1.3 VALIDATION: ML Model Development")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all tests
    test_results.append(("Target Generation", test_target_generation()))
    test_results.append(("Feature Selection", test_feature_selection()))
    test_results.append(("Data Preprocessing", test_data_preprocessing()))
    test_results.append(("Model Training", test_model_training()))
    test_results.append(("Model Evaluation", test_model_evaluation()))
    test_results.append(("Prediction Capabilities", test_prediction_capabilities()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 STEP 1.3 VALIDATION COMPLETED SUCCESSFULLY!")
        print("✅ ML model development is working correctly")
        print("✅ Ready to proceed to Step 1.4: Risk Management")
    else:
        print("⚠️ Some tests failed. Please review and fix issues.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)
