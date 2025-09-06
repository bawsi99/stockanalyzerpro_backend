#!/usr/bin/env python3
"""
Price ML Model Analysis

This script provides a comprehensive analysis of the price prediction ML model,
including training status, performance evaluation, and comparison with benchmarks.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from ..engines.raw_data_ml import raw_data_ml_engine, PricePrediction, VolatilityPrediction, MarketRegime, RawDataMLEngine
from ..engines.unified_manager import UnifiedMLManager
from zerodha_client import ZerodhaDataClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PriceMLAnalyzer:
    """Comprehensive analyzer for price ML model."""
    
    def __init__(self):
        self.zerodha_client = ZerodhaDataClient()
        self.ml_manager = UnifiedMLManager()
        
    def analyze_current_status(self) -> Dict[str, Any]:
        """Analyze current status of price ML model."""
        print("🔍 ANALYZING PRICE ML MODEL STATUS")
        print("=" * 50)
        
        status = {
            'raw_data_ml_trained': raw_data_ml_engine.is_trained,
            'direction_model_exists': raw_data_ml_engine.direction_model is not None,
            'magnitude_model_exists': raw_data_ml_engine.magnitude_model is not None,
            'feature_columns_count': len(raw_data_ml_engine.feature_columns),
            'unified_manager_status': self.ml_manager.engine_status.get('raw_data_ml', False)
        }
        
        print(f"Raw Data ML Trained: {'✅' if status['raw_data_ml_trained'] else '❌'}")
        print(f"Direction Model: {'✅' if status['direction_model_exists'] else '❌'}")
        print(f"Magnitude Model: {'✅' if status['magnitude_model_exists'] else '❌'}")
        print(f"Feature Columns: {status['feature_columns_count']}")
        print(f"Unified Manager Status: {'✅' if status['unified_manager_status'] else '❌'}")
        
        return status
    
    def fetch_training_data(self, symbols: List[str] = None, days: int = 730) -> pd.DataFrame:
        """Fetch historical data for training."""
        if symbols is None:
            symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'AXISBANK']
        
        print(f"\n📊 FETCHING TRAINING DATA")
        print(f"Symbols: {symbols}")
        print(f"Days: {days}")
        
        all_data = []
        
        for symbol in symbols:
            try:
                print(f"Fetching {symbol}...")
                data = self.zerodha_client.get_historical_data(
                    symbol=symbol,
                    exchange='NSE',
                    interval='day',
                    period=days
                )
                
                if data is not None and not data.empty:
                    # Ensure 'date' is a column
                    if 'date' not in data.columns:
                        # If index looks like dates, reset to a column
                        try:
                            if isinstance(data.index, pd.DatetimeIndex) or str(data.index.name).lower() == 'date':
                                data = data.reset_index()
                                # Guarantee column name is 'date'
                                if data.columns[0] != 'date':
                                    data = data.rename(columns={data.columns[0]: 'date'})
                        except Exception:
                            pass
                    # Drop unnamed columns
                    unnamed_cols = [c for c in data.columns if 'Unnamed' in c]
                    if unnamed_cols:
                        data = data.drop(columns=unnamed_cols)
                    # Coerce date column to datetime if present
                    if 'date' in data.columns:
                        data['date'] = pd.to_datetime(data['date'])
                    data['symbol'] = symbol
                    all_data.append(data)
                    print(f"✅ {symbol}: {len(data)} records")
                else:
                    print(f"❌ {symbol}: No data")
                    
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"\n📈 Total Training Data: {len(combined_data)} records")
            return combined_data
        else:
            print("❌ No training data available")
            return pd.DataFrame()
    
    def train_price_model(self, data: pd.DataFrame) -> bool:
        """Train the price prediction model."""
        if data.empty:
            print("❌ No data available for training")
            return False
        
        print(f"\n🤖 TRAINING PRICE PREDICTION MODEL")
        print(f"Training Data Size: {len(data)} records")
        
        try:
            # Group by symbol and train on each
            symbols = data['symbol'].unique()
            success_count = 0
            
            for symbol in symbols:
                symbol_data = data[data['symbol'] == symbol].copy()
                # Ensure date column exists
                if 'date' not in symbol_data.columns:
                    if isinstance(symbol_data.index, pd.DatetimeIndex):
                        symbol_data = symbol_data.reset_index().rename(columns={symbol_data.columns[0]: 'date'})
                # Coerce date
                if 'date' in symbol_data.columns:
                    symbol_data['date'] = pd.to_datetime(symbol_data['date'])
                    symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
                
                # Remove any unnamed columns that might cause issues
                unnamed_cols = [col for col in symbol_data.columns if 'Unnamed' in col]
                if unnamed_cols:
                    symbol_data = symbol_data.drop(columns=unnamed_cols)
                
                print(f"\nTraining on {symbol} ({len(symbol_data)} records)...")
                
                # Train the model
                success = raw_data_ml_engine.train(symbol_data, target_horizon=1)
                
                if success:
                    success_count += 1
                    print(f"✅ {symbol}: Training successful")
                else:
                    print(f"❌ {symbol}: Training failed")
            
            print(f"\n🎯 Training Summary: {success_count}/{len(symbols)} symbols successful")
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def evaluate_model_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance using walk-forward analysis."""
        print(f"\n📊 EVALUATING MODEL PERFORMANCE (WALK-FORWARD)")
        
        results = {}
        symbols = data['symbol'].unique()
        
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol].copy()
            # Ensure date column exists
            if 'date' not in symbol_data.columns:
                if isinstance(symbol_data.index, pd.DatetimeIndex):
                    symbol_data = symbol_data.reset_index().rename(columns={symbol_data.columns[0]: 'date'})
            if 'date' in symbol_data.columns:
                symbol_data['date'] = pd.to_datetime(symbol_data['date'])
                symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
            
            print(f"\nEvaluating {symbol}...")
            
            try:
                # Walk-forward evaluation with expanding windows
                total_samples = len(symbol_data)
                min_train_size = 100  # Minimum training data
                step_size = 20  # Days to move forward each iteration
                
                if total_samples < min_train_size + 50:
                    print(f"⚠️ {symbol}: Insufficient data (total={total_samples}, need={min_train_size + 50})")
                    continue
                
                all_predictions = []
                fold_metrics = []
                
                # Walk-forward evaluation
                for start_idx in range(min_train_size, total_samples - 30, step_size):
                    # Training data: from start to current position
                    train_data = symbol_data.iloc[:start_idx]
                    # Test data: next 30 days (or remaining)
                    test_end = min(start_idx + 30, total_samples)
                    test_data = symbol_data.iloc[start_idx:test_end]
                    
                    if len(test_data) < 10:
                        continue
                    
                    # Train a fresh engine on the training split
                    engine = RawDataMLEngine()
                    train_success = engine.train(train_data, target_horizon=1)
                    if not train_success:
                        continue
                    
                    # Make predictions on test data
                    fold_predictions = []
                    
                    for i in range(len(test_data) - 1):  # Predict next day
                        historical_data = test_data.iloc[:i+1]
                        if len(historical_data) < 30:
                            continue
                        
                        try:
                            # Make prediction
                            price_pred = engine.predict(historical_data)
                            volatility_pred = engine.predict_volatility(historical_data)
                            market_regime = engine.classify_market_regime(historical_data)
                            
                            # Get actual result
                            actual_return = (test_data.iloc[i+1]['close'] / test_data.iloc[i]['close']) - 1
                            actual_direction = 1 if actual_return > 0 else 0
                            
                            fold_predictions.append({
                                'predicted_direction': price_pred.direction,
                                'predicted_confidence': price_pred.confidence,
                                'predicted_magnitude': price_pred.magnitude,
                                'actual_direction': actual_direction,
                                'actual_return': actual_return,
                                'volatility_regime': volatility_pred.volatility_regime,
                                'market_regime': market_regime.regime,
                                'fold': len(fold_metrics)
                            })
                            
                        except Exception as e:
                            continue
                    
                    if fold_predictions:
                        # Calculate fold metrics
                        fold_accuracy = self._calculate_accuracy(fold_predictions)
                        fold_f1 = self._calculate_f1_score(fold_predictions)
                        fold_auc = self._calculate_auc(fold_predictions)
                        
                        fold_metrics.append({
                            'fold': len(fold_metrics),
                            'accuracy': fold_accuracy,
                            'f1_score': fold_f1,
                            'auc': fold_auc,
                            'predictions_count': len(fold_predictions)
                        })
                        
                        all_predictions.extend(fold_predictions)
                
                if all_predictions:
                    # Calculate overall metrics
                    overall_accuracy = self._calculate_accuracy(all_predictions)
                    overall_f1 = self._calculate_f1_score(all_predictions)
                    overall_auc = self._calculate_auc(all_predictions)
                    avg_confidence = np.mean([p['predicted_confidence'] for p in all_predictions])
                    avg_magnitude = np.mean([p['predicted_magnitude'] for p in all_predictions])
                    
                    # Calculate stability metrics
                    accuracy_std = np.std([m['accuracy'] for m in fold_metrics])
                    f1_std = np.std([m['f1_score'] for m in fold_metrics])
                    
                    results[symbol] = {
                        'accuracy': overall_accuracy,
                        'f1_score': overall_f1,
                        'auc': overall_auc,
                        'avg_confidence': avg_confidence,
                        'avg_magnitude': avg_magnitude,
                        'predictions_count': len(all_predictions),
                        'folds_count': len(fold_metrics),
                        'accuracy_std': accuracy_std,
                        'f1_std': f1_std,
                        'fold_metrics': fold_metrics,
                        'predictions': all_predictions
                    }
                    
                    print(f"✅ {symbol}: Accuracy={overall_accuracy:.2%}±{accuracy_std:.2%}, F1={overall_f1:.2%}±{f1_std:.2%}, AUC={overall_auc:.2%}")
                else:
                    print(f"⚠️ {symbol}: No predictions generated")
                    
            except Exception as e:
                print(f"❌ {symbol}: Evaluation failed - {e}")
        
        return results
    
    def _calculate_accuracy(self, predictions: List[Dict]) -> float:
        """Calculate accuracy from predictions."""
        if not predictions:
            return 0.0
        
        correct = sum(1 for p in predictions if 
            (p['predicted_direction'] == 'up' and p['actual_direction'] == 1) or
            (p['predicted_direction'] == 'down' and p['actual_direction'] == 0))
        
        return correct / len(predictions)
    
    def _calculate_f1_score(self, predictions: List[Dict]) -> float:
        """Calculate F1 score from predictions."""
        if not predictions:
            return 0.0
        
        try:
            y_true = [p['actual_direction'] for p in predictions]
            y_pred = [1 if p['predicted_direction'] == 'up' else 0 for p in predictions]
            
            from sklearn.metrics import f1_score
            return f1_score(y_true, y_pred, average='weighted')
        except:
            return 0.0
    
    def _calculate_auc(self, predictions: List[Dict]) -> float:
        """Calculate AUC from predictions."""
        if not predictions:
            return 0.0
        
        try:
            y_true = [p['actual_direction'] for p in predictions]
            y_score = [p['predicted_confidence'] if p['predicted_direction'] == 'up' else 1 - p['predicted_confidence'] for p in predictions]
            
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_score)
        except:
            return 0.0
    
    def compare_with_benchmarks(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare model performance with benchmarks."""
        print(f"\n🏆 COMPARING WITH BENCHMARKS")
        
        if not results:
            print("❌ No results to compare")
            return {}
        
        # Calculate overall metrics
        accuracies = [r['accuracy'] for r in results.values()]
        f1_scores = [r['f1_score'] for r in results.values()]
        auc_scores = [r['auc'] for r in results.values()]
        confidences = [r['avg_confidence'] for r in results.values()]
        
        overall_accuracy = np.mean(accuracies)
        overall_f1 = np.mean(f1_scores)
        overall_auc = np.mean(auc_scores)
        overall_confidence = np.mean(confidences)
        
        # Calculate stability metrics
        accuracy_std = np.mean([r['accuracy_std'] for r in results.values()])
        f1_std = np.mean([r['f1_std'] for r in results.values()])
        
        print(f"Overall Accuracy: {overall_accuracy:.2%} ± {accuracy_std:.2%}")
        print(f"Overall F1-Score: {overall_f1:.2%} ± {f1_std:.2%}")
        print(f"Overall AUC: {overall_auc:.2%}")
        print(f"Overall Confidence: {overall_confidence:.2%}")
        
        # Benchmark comparison for accuracy
        benchmarks = {
            'random_guess': 0.50,
            'buy_hold': 0.55,  # Assuming slight upward bias
            'moving_average': 0.60,
            'rsi_strategy': 0.65,
            'professional_trader': 0.70,
            'quant_fund': 0.75,
            'our_model': overall_accuracy
        }
        
        print(f"\n📊 ACCURACY BENCHMARK COMPARISON:")
        for benchmark, accuracy in benchmarks.items():
            status = "✅" if overall_accuracy >= accuracy else "❌"
            print(f"{status} {benchmark.replace('_', ' ').title()}: {accuracy:.1%}")
        
        # Performance ranking
        sorted_benchmarks = sorted(benchmarks.items(), key=lambda x: x[1], reverse=True)
        our_rank = next(i for i, (name, _) in enumerate(sorted_benchmarks) if name == 'our_model') + 1
        
        print(f"\n🏅 ACCURACY RANKING: {our_rank}/{len(benchmarks)}")
        
        # F1 and AUC analysis
        print(f"\n📈 DETAILED METRICS ANALYSIS:")
        print(f"F1-Score: {overall_f1:.2%} (Good: >0.60, Excellent: >0.70)")
        print(f"AUC: {overall_auc:.2%} (Good: >0.65, Excellent: >0.75)")
        
        # Model stability analysis
        print(f"\n🔒 MODEL STABILITY:")
        print(f"Accuracy Stability: {accuracy_std:.2%} (Lower is better)")
        print(f"F1 Stability: {f1_std:.2%} (Lower is better)")
        
        return {
            'overall_accuracy': overall_accuracy,
            'overall_f1': overall_f1,
            'overall_auc': overall_auc,
            'overall_confidence': overall_confidence,
            'accuracy_std': accuracy_std,
            'f1_std': f1_std,
            'benchmarks': benchmarks,
            'ranking': our_rank
        }
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance if available."""
        print(f"\n🔍 ANALYZING FEATURE IMPORTANCE")
        
        if not raw_data_ml_engine.is_trained or raw_data_ml_engine.direction_model is None:
            print("❌ Model not trained or no feature importance available")
            return {}
        
        try:
            # Get feature importance from Random Forest
            if hasattr(raw_data_ml_engine.direction_model, 'feature_importances_'):
                importances = raw_data_ml_engine.direction_model.feature_importances_
                feature_names = raw_data_ml_engine.feature_columns
                
                # Create feature importance dataframe
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print(f"\n🏆 TOP 10 FEATURES:")
                for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                    print(f"{i+1:2d}. {row['feature']}: {row['importance']:.4f}")
                
                return {
                    'feature_importance': feature_importance.to_dict('records'),
                    'top_features': feature_importance.head(10).to_dict('records')
                }
            else:
                print("⚠️ Feature importance not available for this model type")
                return {}
                
        except Exception as e:
            print(f"❌ Feature importance analysis failed: {e}")
            return {}
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on comprehensive metrics."""
        print(f"\n💡 GENERATING RECOMMENDATIONS")
        
        recommendations = []
        
        if not results:
            recommendations.append("Train the model with sufficient historical data")
            return recommendations
        
        overall_accuracy = results.get('overall_accuracy', 0)
        overall_f1 = results.get('overall_f1', 0)
        overall_auc = results.get('overall_auc', 0)
        accuracy_std = results.get('accuracy_std', 0)
        
        # Accuracy-based recommendations
        if overall_accuracy < 0.55:
            recommendations.append("Accuracy below 55% - fundamental feature engineering needed")
            recommendations.append("Consider multi-timeframe analysis (5min, 1hr, daily)")
            recommendations.append("Add market index relative features (beta, correlation)")
        elif overall_accuracy < 0.65:
            recommendations.append("Accuracy below 65% - implement ensemble methods")
            recommendations.append("Add market regime awareness and regime-specific models")
            recommendations.append("Consider alternative targets (3-5 day horizon, magnitude thresholds)")
        
        # F1-score recommendations
        if overall_f1 < 0.50:
            recommendations.append("F1-score below 50% - class imbalance or poor probability calibration")
            recommendations.append("Implement SMOTE or class weights for balanced training")
            recommendations.append("Fine-tune classification thresholds per market regime")
        elif overall_f1 < 0.60:
            recommendations.append("F1-score below 60% - improve feature selection and engineering")
            recommendations.append("Add cross-asset correlation features")
            recommendations.append("Implement recursive feature elimination")
        
        # AUC recommendations
        if overall_auc < 0.60:
            recommendations.append("AUC below 60% - poor probability estimates")
            recommendations.append("Implement better probability calibration methods")
            recommendations.append("Consider alternative model architectures (LightGBM, XGBoost)")
        elif overall_auc < 0.70:
            recommendations.append("AUC below 70% - moderate discriminative power")
            recommendations.append("Add regime-specific feature engineering")
            recommendations.append("Implement feature interaction terms")
        
        # Stability recommendations
        if accuracy_std > 0.10:
            recommendations.append("High accuracy variance - implement more robust validation")
            recommendations.append("Increase minimum training data size")
            recommendations.append("Add regularization to prevent overfitting")
        
        # General recommendations
        recommendations.append("Implement walk-forward analysis with multiple timeframes")
        recommendations.append("Add transaction costs and slippage to performance calculation")
        recommendations.append("Consider ensemble of multiple models (voting/stacking)")
        recommendations.append("Implement model retraining schedule based on regime changes")
        recommendations.append("Add risk management rules (position sizing, stop-loss)")
        
        # Advanced recommendations
        if overall_accuracy >= 0.65 and overall_f1 >= 0.60:
            recommendations.append("Good performance! Consider production deployment")
            recommendations.append("Implement real-time prediction pipeline")
            recommendations.append("Add portfolio-level risk management")
        
        print(f"\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec}")
        
        return recommendations
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis of the price ML model."""
        print("🚀 PRICE ML MODEL COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        
        analysis_results = {}
        
        # 1. Current Status
        status = self.analyze_current_status()
        analysis_results['status'] = status
        
        # 2. Fetch Training Data
        training_data = self.fetch_training_data()
        analysis_results['training_data_size'] = len(training_data)
        
        if training_data.empty:
            print("❌ No training data available. Analysis complete.")
            return analysis_results
        
        # 3. Train Model (if needed)
        if not status['raw_data_ml_trained']:
            training_success = self.train_price_model(training_data)
            analysis_results['training_success'] = training_success
            
            if not training_success:
                print("❌ Model training failed. Analysis complete.")
                return analysis_results
        
        # 4. Evaluate Performance
        performance_results = self.evaluate_model_performance(training_data)
        analysis_results['performance'] = performance_results
        
        # 5. Compare with Benchmarks
        benchmark_results = self.compare_with_benchmarks(performance_results)
        analysis_results['benchmarks'] = benchmark_results
        
        # 6. Feature Importance
        feature_importance = self.analyze_feature_importance()
        analysis_results['feature_importance'] = feature_importance
        
        # 7. Generate Recommendations
        recommendations = self.generate_recommendations(benchmark_results)
        analysis_results['recommendations'] = recommendations
        
        # 8. Summary
        print(f"\n📋 ANALYSIS SUMMARY")
        print("=" * 40)
        print(f"Model Trained: {'✅' if status['raw_data_ml_trained'] else '❌'}")
        print(f"Training Data: {len(training_data)} records")
        print(f"Symbols Analyzed: {len(performance_results)}")
        
        if benchmark_results:
            print(f"Overall Accuracy: {benchmark_results['overall_accuracy']:.2%} ± {benchmark_results['accuracy_std']:.2%}")
            print(f"Overall F1-Score: {benchmark_results['overall_f1']:.2%} ± {benchmark_results['f1_std']:.2%}")
            print(f"Overall AUC: {benchmark_results['overall_auc']:.2%}")
            print(f"Performance Ranking: {benchmark_results['ranking']}")
        
        print(f"Recommendations: {len(recommendations)}")
        
        # Additional insights
        if performance_results:
            total_predictions = sum(r['predictions_count'] for r in performance_results.values())
            total_folds = sum(r['folds_count'] for r in performance_results.values())
            print(f"Total Predictions: {total_predictions}")
            print(f"Total Folds: {total_folds}")
            
            # Best and worst performers
            best_symbol = max(performance_results.items(), key=lambda x: x[1]['accuracy'])
            worst_symbol = min(performance_results.items(), key=lambda x: x[1]['accuracy'])
            print(f"Best Performer: {best_symbol[0]} ({best_symbol[1]['accuracy']:.2%})")
            print(f"Worst Performer: {worst_symbol[0]} ({worst_symbol[1]['accuracy']:.2%})")
        
        return analysis_results

def main():
    """Main function to run the analysis."""
    analyzer = PriceMLAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"price_ml_analysis_{timestamp}.json"
    
    try:
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    return 0

if __name__ == "__main__":
    exit(main())
