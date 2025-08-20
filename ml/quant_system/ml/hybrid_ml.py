"""
Hybrid ML Module

This module integrates both pattern-based and raw data ML approaches:
1. Pattern-based ML (CatBoost) - predicts pattern success probability
2. Raw data ML - predicts price movements directly from OHLCV data

This creates a comprehensive quantitative analysis system.
Adapted from backend/ml/hybrid_ml_engine.py for unified integration.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .core import BaseMLEngine, UnifiedMLConfig
from .pattern_ml import pattern_ml_engine
from .raw_data_ml import raw_data_ml_engine, PricePrediction, VolatilityPrediction, MarketRegime

logger = logging.getLogger(__name__)

@dataclass
class HybridPrediction:
    """Combined prediction from both ML approaches."""
    # Pattern-based predictions
    pattern_success_probability: float
    pattern_confidence: float
    
    # Raw data predictions
    price_direction: str
    price_magnitude: float
    price_confidence: float
    
    # Combined metrics
    combined_confidence: float
    consensus_signal: str  # "strong_buy", "buy", "hold", "sell", "strong_sell"
    risk_score: float
    
    # Market context
    volatility_prediction: VolatilityPrediction
    market_regime: MarketRegime

class HybridMLEngine(BaseMLEngine):
    """Hybrid ML engine combining pattern-based and raw data analysis."""
    
    def __init__(self, config: UnifiedMLConfig = None):
        super().__init__(config)
        self.pattern_engine = pattern_ml_engine
        self.raw_engine = raw_data_ml_engine
        
    def train(self, stock_data: pd.DataFrame, pattern_data: Optional[Dict] = None) -> bool:
        """Train both pattern-based and raw data models."""
        success = True
        
        # Train raw data model
        logger.info("Training raw data ML model...")
        if not self.raw_engine.train(stock_data):
            logger.warning("Raw data model training failed")
            success = False
        
        # Pattern-based model training (if data provided)
        if pattern_data:
            logger.info("Training pattern-based ML model...")
            if not self.pattern_engine.train():
                logger.warning("Pattern-based model training failed")
                success = False
        
        self.is_trained = success
        return success
    
    def predict(self, stock_data: pd.DataFrame, pattern_features: Optional[Dict] = None, 
                pattern_type: Optional[str] = None) -> HybridPrediction:
        """Generate hybrid prediction combining both approaches."""
        
        # 1. Pattern-based prediction
        if pattern_features and pattern_type:
            pattern_prob = self.pattern_engine.predict(pattern_features, pattern_type)
            pattern_confidence = pattern_prob
        else:
            pattern_prob = 0.5
            pattern_confidence = 0.5
        
        # 2. Raw data prediction
        price_pred = self.raw_engine.predict(stock_data)
        volatility_pred = self.raw_engine.predict_volatility(stock_data)
        market_regime = self.raw_engine.classify_market_regime(stock_data)
        
        # 3. Combine predictions
        combined_confidence = self._calculate_combined_confidence(
            pattern_confidence, price_pred.confidence, market_regime.confidence
        )
        
        consensus_signal = self._determine_consensus_signal(
            pattern_prob, price_pred.direction, price_pred.confidence
        )
        
        risk_score = self._calculate_risk_score(
            pattern_prob, price_pred.magnitude, volatility_pred.predicted_volatility
        )
        
        return HybridPrediction(
            pattern_success_probability=pattern_prob,
            pattern_confidence=pattern_confidence,
            price_direction=price_pred.direction,
            price_magnitude=price_pred.magnitude,
            price_confidence=price_pred.confidence,
            combined_confidence=combined_confidence,
            consensus_signal=consensus_signal,
            risk_score=risk_score,
            volatility_prediction=volatility_pred,
            market_regime=market_regime
        )
    
    def _calculate_combined_confidence(self, pattern_conf: float, price_conf: float, 
                                     regime_conf: float) -> float:
        """Calculate combined confidence from all models."""
        # Weighted average with pattern-based ML having higher weight
        weights = [0.4, 0.4, 0.2]  # pattern, price, regime
        combined = (pattern_conf * weights[0] + 
                   price_conf * weights[1] + 
                   regime_conf * weights[2])
        return min(1.0, max(0.0, combined))
    
    def _determine_consensus_signal(self, pattern_prob: float, price_direction: str, 
                                  price_conf: float) -> str:
        """Determine consensus trading signal."""
        
        # Pattern signal
        if pattern_prob > 0.7:
            pattern_signal = "buy"
        elif pattern_prob < 0.3:
            pattern_signal = "sell"
        else:
            pattern_signal = "hold"
        
        # Price signal
        if price_direction == "up" and price_conf > 0.6:
            price_signal = "buy"
        elif price_direction == "down" and price_conf > 0.6:
            price_signal = "sell"
        else:
            price_signal = "hold"
        
        # Consensus logic
        if pattern_signal == "buy" and price_signal == "buy":
            return "strong_buy"
        elif pattern_signal == "sell" and price_signal == "sell":
            return "strong_sell"
        elif pattern_signal == "buy" or price_signal == "buy":
            return "buy"
        elif pattern_signal == "sell" or price_signal == "sell":
            return "sell"
        else:
            return "hold"
    
    def _calculate_risk_score(self, pattern_prob: float, price_magnitude: float, 
                            volatility: float) -> float:
        """Calculate overall risk score."""
        # Base risk from pattern probability
        pattern_risk = 1.0 - pattern_prob
        
        # Risk from price movement magnitude
        magnitude_risk = min(price_magnitude * 10, 1.0)
        
        # Risk from volatility
        volatility_risk = min(volatility * 20, 1.0)
        
        # Combined risk score (0-100)
        combined_risk = (pattern_risk * 0.4 + magnitude_risk * 0.3 + volatility_risk * 0.3) * 100
        return min(100.0, max(0.0, combined_risk))
    
    def get_comprehensive_analysis(self, stock_data: pd.DataFrame, 
                                 pattern_features: Optional[Dict] = None,
                                 pattern_type: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive analysis combining all ML approaches."""
        
        hybrid_pred = self.predict(stock_data, pattern_features, pattern_type)
        
        return {
            "hybrid_prediction": {
                "consensus_signal": hybrid_pred.consensus_signal,
                "combined_confidence": hybrid_pred.combined_confidence,
                "risk_score": hybrid_pred.risk_score,
                "recommendation": self._generate_recommendation(hybrid_pred)
            },
            "pattern_analysis": {
                "success_probability": hybrid_pred.pattern_success_probability,
                "confidence": hybrid_pred.pattern_confidence,
                "signal": "buy" if hybrid_pred.pattern_success_probability > 0.6 else "sell"
            },
            "price_analysis": {
                "direction": hybrid_pred.price_direction,
                "magnitude": hybrid_pred.price_magnitude,
                "confidence": hybrid_pred.price_confidence,
                "expected_move": f"{hybrid_pred.price_magnitude * 100:.2f}%"
            },
            "market_context": {
                "volatility": {
                    "current": hybrid_pred.volatility_prediction.current_volatility,
                    "predicted": hybrid_pred.volatility_prediction.predicted_volatility,
                    "regime": hybrid_pred.volatility_prediction.volatility_regime
                },
                "market_regime": {
                    "regime": hybrid_pred.market_regime.regime,
                    "strength": hybrid_pred.market_regime.strength,
                    "duration": hybrid_pred.market_regime.duration
                }
            },
            "risk_assessment": {
                "overall_risk": hybrid_pred.risk_score,
                "risk_level": self._classify_risk_level(hybrid_pred.risk_score),
                "position_sizing": self._calculate_position_size(hybrid_pred)
            }
        }
    
    def _generate_recommendation(self, pred: HybridPrediction) -> str:
        """Generate trading recommendation."""
        if pred.consensus_signal == "strong_buy" and pred.combined_confidence > 0.7:
            return "Strong Buy - High confidence pattern with bullish price prediction"
        elif pred.consensus_signal == "buy" and pred.combined_confidence > 0.6:
            return "Buy - Positive signals from both pattern and price analysis"
        elif pred.consensus_signal == "strong_sell" and pred.combined_confidence > 0.7:
            return "Strong Sell - High confidence bearish signals"
        elif pred.consensus_signal == "sell" and pred.combined_confidence > 0.6:
            return "Sell - Negative signals from both pattern and price analysis"
        else:
            return "Hold - Mixed signals or low confidence, wait for clearer setup"
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk level based on score."""
        if risk_score < 30:
            return "Low"
        elif risk_score < 60:
            return "Medium"
        else:
            return "High"
    
    def _calculate_position_size(self, pred: HybridPrediction) -> float:
        """Calculate recommended position size."""
        # Base position size from confidence and risk
        base_size = pred.combined_confidence * (1 - pred.risk_score / 100)
        
        # Adjust for market regime
        if pred.market_regime.regime == "volatile":
            base_size *= 0.5  # Reduce position in volatile markets
        elif pred.market_regime.regime in ["trending_bull", "trending_bear"]:
            base_size *= 1.2  # Increase position in trending markets
        
        # Cap at reasonable levels
        return min(0.1, max(0.01, base_size))  # 1-10% of portfolio
    
    def evaluate(self, stock_data: pd.DataFrame, pattern_features: Optional[Dict] = None,
                pattern_type: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Evaluate hybrid ML performance."""
        try:
            # Get comprehensive analysis
            analysis = self.get_comprehensive_analysis(stock_data, pattern_features, pattern_type)
            
            # Add evaluation metrics
            evaluation = {
                "analysis": analysis,
                "model_status": {
                    "pattern_ml_trained": self.pattern_engine.is_trained,
                    "raw_data_ml_trained": self.raw_engine.is_trained,
                    "hybrid_ml_trained": self.is_trained
                },
                "prediction_quality": {
                    "combined_confidence": analysis["hybrid_prediction"]["combined_confidence"],
                    "risk_score": analysis["hybrid_prediction"]["risk_score"],
                    "consensus_signal": analysis["hybrid_prediction"]["consensus_signal"]
                }
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}
    
    def save_model(self, path: str) -> bool:
        """Save the hybrid ML models."""
        try:
            import joblib
            
            # Save both engines
            model_data = {
                'pattern_engine': self.pattern_engine,
                'raw_engine': self.raw_engine,
                'config': self.config
            }
            
            joblib.dump(model_data, path)
            logger.info(f"Hybrid ML models saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save hybrid ML models: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load the hybrid ML models."""
        try:
            import joblib
            
            model_data = joblib.load(path)
            
            # Load both engines
            self.pattern_engine = model_data['pattern_engine']
            self.raw_engine = model_data['raw_engine']
            self.config = model_data['config']
            self.is_trained = True
            
            logger.info(f"Hybrid ML models loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load hybrid ML models: {e}")
            return False

# Global instance
hybrid_ml_engine = HybridMLEngine()
