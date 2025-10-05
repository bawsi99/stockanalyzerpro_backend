#!/usr/bin/env python3
"""
Quantitative Risk Processor

Comprehensive quantitative risk analysis with:
- Advanced Risk Metrics (VaR, Expected Shortfall, Sharpe ratios, etc.)
- Stress Testing (Historical, Monte Carlo, Sector-specific, Market crash scenarios)
- Scenario Analysis (Bull/Bear/Sideways/Volatility spike scenarios)

This module contains the sophisticated risk calculations that were previously 
part of the advanced analysis system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import asyncio

logger = logging.getLogger(__name__)


class QuantitativeRiskProcessor:
    """Comprehensive quantitative risk analysis processor with advanced calculations."""
    
    def __init__(self):
        self.agent_name = "quantitative_risk"
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.confidence_levels = [0.95, 0.99]  # VaR confidence levels
        
    async def analyze_async(self, stock_data: pd.DataFrame, indicators: Dict, context: str = "") -> Dict[str, Any]:
        """
        Perform comprehensive quantitative risk analysis asynchronously.
        Offloaded to a background thread to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.analyze_sync(stock_data, indicators, context)
        )

    def analyze_sync(self, stock_data: pd.DataFrame, indicators: Dict, context: str = "") -> Dict[str, Any]:
        """
        Synchronous implementation of quantitative risk analysis.
        """
        try:
            # Calculate returns for analysis (handle small/empty data safely)
            if stock_data is None or stock_data.empty or 'close' not in stock_data:
                returns = pd.Series(dtype=float)
            else:
                returns = stock_data['close'].pct_change().dropna()
            
            # Generate all quantitative risk components
            advanced_risk = self._calculate_advanced_risk_metrics(returns, stock_data, indicators)
            stress_testing = self._perform_stress_testing(returns, stock_data)
            scenario_analysis = self._perform_scenario_analysis(returns, stock_data, indicators)
            
            return {
                "agent_name": self.agent_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "context": context,
                "advanced_risk_metrics": advanced_risk,
                "stress_testing": stress_testing,
                "scenario_analysis": scenario_analysis,
                "overall_risk_assessment": self._generate_overall_assessment(advanced_risk, stress_testing, scenario_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error in quantitative risk analysis: {e}")
            return {
                "agent_name": self.agent_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "context": context,
                "error": str(e),
                "advanced_risk_metrics": {"error": str(e)},
                "stress_testing": {"error": str(e)},
                "scenario_analysis": {"error": str(e)}
            }
    
    def _calculate_advanced_risk_metrics(self, returns: pd.Series, 
                                       stock_data: pd.DataFrame, 
                                       indicators: dict) -> Dict[str, Any]:
        """Calculate advanced risk metrics for risk assessment."""
        try:
            # Basic volatility calculations
            volatility = returns.std() if len(returns) > 0 else 0.0
            annualized_volatility = volatility * np.sqrt(252)
            
            # Mean return calculations
            mean_return = returns.mean()
            annualized_return = mean_return * 252
            
            # Value at Risk (VaR) calculations
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0
            var_99 = np.percentile(returns, 1) if len(returns) > 0 else 0.0
            
            # Expected Shortfall (Conditional VaR)
            es_95 = returns[returns <= var_95].mean() if len(returns) > 0 else 0.0
            es_99 = returns[returns <= var_99].mean() if len(returns) > 0 else 0.0
            
            # Maximum Drawdown calculation
            if len(returns) > 0:
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()
            else:
                drawdown = pd.Series(dtype=float)
                max_drawdown = 0.0
            
            # Current drawdown
            current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0.0
            
            # Drawdown duration
            drawdown_duration = self._calculate_drawdown_duration(drawdown)
            
            # Risk-adjusted returns
            sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
            sortino_ratio = self._calculate_sortino_ratio(returns, annualized_return)
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Higher moments
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Tail risk analysis
            tail_frequency = self._calculate_tail_frequency(returns)
            
            # Risk score calculation
            risk_score = self._calculate_risk_score(volatility, max_drawdown, var_95, sharpe_ratio)
            risk_level = self._determine_risk_level(risk_score)
            
            # Risk components - calculate based on actual data
            risk_components = {
                "volatility_risk": "High" if annualized_volatility > 0.3 else "Medium" if annualized_volatility > 0.2 else "Low",
                "drawdown_risk": "High" if abs(max_drawdown) > 0.3 else "Medium" if abs(max_drawdown) > 0.2 else "Low",
                "tail_risk": "High" if abs(var_99) > 0.05 else "Medium" if abs(var_99) > 0.03 else "Low",
                "liquidity_risk": self._calculate_liquidity_risk(stock_data),
                "sector_risk": self._calculate_sector_risk(returns, indicators)
            }
            
            # Mitigation strategies
            mitigation_strategies = self._generate_mitigation_strategies(risk_components, risk_score)
            
            return {
                "volatility": float(volatility),
                "annualized_volatility": float(annualized_volatility),
                "mean_return": float(mean_return),
                "annualized_return": float(annualized_return),
                "var_95": float(var_95),
                "var_99": float(var_99),
                "expected_shortfall_95": float(es_95),
                "expected_shortfall_99": float(es_99),
                "max_drawdown": float(max_drawdown),
                "current_drawdown": float(current_drawdown),
                "drawdown_duration": int(drawdown_duration),
                "sharpe_ratio": float(sharpe_ratio),
                "sortino_ratio": float(sortino_ratio),
                "calmar_ratio": float(calmar_ratio),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "tail_frequency": float(tail_frequency),
                "risk_score": int(risk_score),
                "risk_level": risk_level,
                "risk_components": risk_components,
                "mitigation_strategies": mitigation_strategies,
                "calculation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating advanced risk metrics: {e}")
            return {"error": str(e)}
    
    def _perform_stress_testing(self, returns: pd.Series, 
                              stock_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform stress testing scenarios."""
        try:
            # Historical stress scenarios
            historical_stress = self._analyze_historical_stress(returns) if len(returns) > 0 else {
                "worst_20_day_period": -0.05,
                "second_worst_period": -0.04,
                "third_worst_period": -0.03,
                "stress_frequency": 0.0,
            }
            
            # Monte Carlo stress testing
            monte_carlo_stress = self._monte_carlo_stress_test(returns) if len(returns) > 0 else {
                "worst_case": -0.1,
                "fifth_percentile": -0.08,
                "tenth_percentile": -0.06,
                "expected_loss": -0.03,
                "probability_of_loss": 0.5,
            }
            
            # Sector-specific stress scenarios
            sector_stress = self._sector_specific_stress_scenarios(returns)
            
            # Market crash scenarios
            crash_scenarios = self._market_crash_scenarios(returns, stock_data)
            
            # Stress score calculation
            stress_score = self._calculate_stress_score(historical_stress, monte_carlo_stress)
            stress_level = self._determine_stress_level(stress_score)
            
            # Risk mitigation recommendations
            risk_mitigation = self._generate_stress_mitigation_recommendations(stress_score, stress_level)
            
            return {
                "stress_scenarios": {
                    "historical_stress": historical_stress,
                    "monte_carlo_stress": monte_carlo_stress,
                    "sector_stress": sector_stress,
                    "crash_scenarios": crash_scenarios
                },
                "stress_level": stress_level,
                "stress_score": int(stress_score),
                "worst_case_scenario": float(monte_carlo_stress.get("worst_case", -0.2)),
                "risk_mitigation_recommendations": risk_mitigation,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error performing stress testing: {e}")
            return {"error": str(e)}
    
    def _perform_scenario_analysis(self, returns: pd.Series, 
                                 stock_data: pd.DataFrame, 
                                 indicators: dict) -> Dict[str, Any]:
        """Perform scenario analysis with best/worst case outcomes."""
        try:
            # Bull market scenario
            bull_scenario = self._bull_market_scenario(returns, stock_data, indicators)
            
            # Bear market scenario
            bear_scenario = self._bear_market_scenario(returns, stock_data, indicators)
            
            # Sideways market scenario
            sideways_scenario = self._sideways_market_scenario(returns, stock_data, indicators)
            
            # Volatility spike scenario
            volatility_scenario = self._volatility_spike_scenario(returns, stock_data)
            
            # Confidence levels for each scenario
            confidence_levels = self._calculate_scenario_confidence_levels(
                bull_scenario, bear_scenario, sideways_scenario, volatility_scenario
            )
            
            # Probability scores
            probability_scores = self._calculate_scenario_probabilities(
                returns, indicators, confidence_levels
            )
            
            # Impact scores
            impact_scores = self._calculate_scenario_impact_scores(
                bull_scenario, bear_scenario, sideways_scenario, volatility_scenario
            )
            
            return {
                "best_case": bull_scenario,
                "worst_case": bear_scenario,
                "expected_outcomes": {
                    "bull_scenario": bull_scenario,
                    "bear_scenario": bear_scenario,
                    "sideways_scenario": sideways_scenario,
                    "volatility_scenario": volatility_scenario
                },
                "confidence_levels": confidence_levels,
                "probability_scores": probability_scores,
                "impact_scores": impact_scores,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error performing scenario analysis: {e}")
            return {"error": str(e)}
    
    def _generate_overall_assessment(self, advanced_risk: Dict, stress_testing: Dict, scenario_analysis: Dict) -> Dict[str, Any]:
        """Generate overall risk assessment from all components."""
        try:
            # Extract key metrics
            risk_score = advanced_risk.get('risk_score', 50)
            risk_level = advanced_risk.get('risk_level', 'Medium')
            stress_level = stress_testing.get('stress_level', 'Medium')
            worst_case = stress_testing.get('worst_case_scenario', -0.1)
            
            # Calculate combined risk score
            combined_risk_score = (risk_score * 0.4 + 
                                 (70 if stress_level == 'High' else 40 if stress_level == 'Medium' else 20) * 0.3 +
                                 abs(worst_case) * 100 * 0.3)
            
            # Determine overall risk level
            if combined_risk_score >= 70:
                overall_risk_level = "High"
            elif combined_risk_score >= 40:
                overall_risk_level = "Medium"
            else:
                overall_risk_level = "Low"
            
            # Key risk factors
            key_risk_factors = []
            if risk_level == "High":
                key_risk_factors.append("High volatility and drawdown risk")
            if stress_level == "High":
                key_risk_factors.append("Elevated stress testing scenarios")
            if abs(worst_case) > 0.15:
                key_risk_factors.append("Significant downside potential")
            
            return {
                "overall_risk_level": overall_risk_level,
                "combined_risk_score": int(combined_risk_score),
                "key_risk_factors": key_risk_factors,
                "confidence_score": self._calculate_overall_confidence(advanced_risk, stress_testing, scenario_analysis),
                "assessment_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating overall assessment: {e}")
            return {
                "overall_risk_level": "Medium",
                "combined_risk_score": 50,
                "key_risk_factors": ["Assessment calculation error"],
                "confidence_score": 0.5,
                "error": str(e)
            }
    
    def _calculate_overall_confidence(self, advanced_risk: Dict, stress_testing: Dict, scenario_analysis: Dict) -> float:
        """Calculate overall confidence score for the risk assessment."""
        confidence_scores = []
        
        # Add confidence from each component
        if 'calculation_timestamp' in advanced_risk and 'error' not in advanced_risk:
            confidence_scores.append(0.8)  # High confidence in quantitative metrics
        if 'analysis_timestamp' in stress_testing and 'error' not in stress_testing:
            confidence_scores.append(0.7)  # Good confidence in stress testing
        if 'analysis_timestamp' in scenario_analysis and 'error' not in scenario_analysis:
            confidence_scores.append(0.6)  # Moderate confidence in scenario analysis
        
        return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5

    # Helper methods for risk metrics
    def _calculate_sortino_ratio(self, returns: pd.Series, annualized_return: float) -> float:
        """Calculate Sortino ratio."""
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.01
        return (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate current drawdown duration in days."""
        if len(drawdown) == 0:
            return 0
        
        current_dd = drawdown.iloc[-1]
        if current_dd >= 0:
            return 0
        
        # Count consecutive negative drawdown days
        duration = 0
        for i in range(len(drawdown) - 1, -1, -1):
            if drawdown.iloc[i] < 0:
                duration += 1
            else:
                break
        return duration
    
    def _calculate_tail_frequency(self, returns: pd.Series) -> float:
        """Calculate frequency of tail events."""
        threshold = returns.std() * 2  # 2 standard deviations
        tail_events = len(returns[abs(returns) > threshold])
        return tail_events / len(returns) if len(returns) > 0 else 0
    
    def _calculate_liquidity_risk(self, stock_data: pd.DataFrame) -> str:
        """Calculate liquidity risk based on volume and price data."""
        try:
            if stock_data.empty:
                return "Medium"
            
            # Calculate average daily volume
            avg_volume = stock_data['volume'].mean()
            current_volume = stock_data['volume'].iloc[-1]
            current_price = stock_data['close'].iloc[-1]
            
            # Calculate volume-weighted average price (VWAP) with divide-by-zero safety
            vol_sum = float(stock_data['volume'].sum())
            vwap = float((stock_data['close'] * stock_data['volume']).sum() / vol_sum) if vol_sum > 0 else float(stock_data['close'].iloc[-1])
            
            # Calculate volume volatility with zero-mean safety
            vol_mean = float(stock_data['volume'].mean())
            volume_volatility = float((stock_data['volume'].std() / vol_mean) if vol_mean > 0 else 0.0)
            
            # Calculate bid-ask spread proxy (using high-low spread) with zero-price safety
            denom = stock_data['close'].replace(0, pd.NA)
            spread_series = ((stock_data['high'] - stock_data['low']) / denom).replace([np.inf, -np.inf], pd.NA).dropna()
            avg_spread = float(spread_series.mean()) if len(spread_series) > 0 else 0.02
            
            # Determine liquidity risk
            if (current_volume < avg_volume * 0.5 or 
                volume_volatility > 1.0 or 
                avg_spread > 0.02):
                return "High"
            elif (current_volume < avg_volume * 0.8 or 
                  volume_volatility > 0.7 or 
                  avg_spread > 0.01):
                return "Medium"
            else:
                return "Low"
                
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {e}")
            return "Medium"
    
    def _calculate_sector_risk(self, returns: pd.Series, indicators: dict) -> str:
        """Calculate sector-specific risk based on technical indicators."""
        try:
            # Calculate sector risk based on technical indicators
            rsi_values = indicators.get('rsi_14', [50])
            macd_values = indicators.get('macd_line', [0])
            
            # Get latest values
            current_rsi = rsi_values[-1] if rsi_values else 50
            current_macd = macd_values[-1] if macd_values else 0
            
            # Calculate sector risk factors
            rsi_risk = 0
            if current_rsi > 80:
                rsi_risk = 2  # Overbought
            elif current_rsi < 20:
                rsi_risk = 2  # Oversold
            elif current_rsi > 70 or current_rsi < 30:
                rsi_risk = 1  # Near extremes
            
            macd_risk = 0
            if abs(current_macd) > 2:  # Strong momentum
                macd_risk = 1
            
            # Calculate volatility-based sector risk
            volatility = returns.std()
            vol_risk = 0
            if volatility > 0.03:  # High volatility
                vol_risk = 2
            elif volatility > 0.02:  # Medium volatility
                vol_risk = 1
            
            # Total sector risk score
            total_risk = rsi_risk + macd_risk + vol_risk
            
            if total_risk >= 4:
                return "High"
            elif total_risk >= 2:
                return "Medium"
            else:
                return "Low"
                
        except Exception as e:
            logger.error(f"Error calculating sector risk: {e}")
            return "Medium"
    
    def _calculate_risk_score(self, volatility: float, max_drawdown: float, 
                            var_95: float, sharpe_ratio: float) -> int:
        """Calculate overall risk score (0-100)."""
        # Normalize components to 0-100 scale
        vol_score = min(100, (volatility * 1000))  # Scale volatility
        dd_score = min(100, abs(max_drawdown) * 100)  # Scale drawdown
        var_score = min(100, abs(var_95) * 1000)  # Scale VaR
        sharpe_score = max(0, min(100, (1 - sharpe_ratio) * 50))  # Invert Sharpe
        
        # Weighted average
        risk_score = (vol_score * 0.3 + dd_score * 0.3 + var_score * 0.2 + sharpe_score * 0.2)
        return int(risk_score)
    
    def _determine_risk_level(self, risk_score: int) -> str:
        """Determine risk level based on score."""
        if risk_score >= 70:
            return "High"
        elif risk_score >= 40:
            return "Medium"
        else:
            return "Low"
    
    def _generate_mitigation_strategies(self, risk_components: dict, risk_score: int) -> List[str]:
        """Generate risk mitigation strategies."""
        strategies = []
        
        if risk_score > 70:
            strategies.extend([
                "Consider reducing position size",
                "Implement strict stop-loss orders",
                "Diversify across uncorrelated assets",
                "Use options for downside protection"
            ])
        elif risk_score > 40:
            strategies.extend([
                "Monitor position closely",
                "Set reasonable stop-loss levels",
                "Consider partial profit taking",
                "Review portfolio allocation"
            ])
        else:
            strategies.extend([
                "Maintain current position",
                "Regular monitoring recommended",
                "Consider gradual position building"
            ])
        
        return strategies

    # Helper methods for stress testing
    def _analyze_historical_stress(self, returns: pd.Series) -> Dict:
        """Analyze historical stress periods."""
        # Find worst historical periods
        worst_periods = returns.rolling(window=20).sum().nsmallest(3)
        
        return {
            "worst_20_day_period": float(worst_periods.iloc[0]) if len(worst_periods) > 0 else -0.1,
            "second_worst_period": float(worst_periods.iloc[1]) if len(worst_periods) > 1 else -0.08,
            "third_worst_period": float(worst_periods.iloc[2]) if len(worst_periods) > 2 else -0.06,
            "stress_frequency": len(returns[returns < -0.05]) / len(returns) if len(returns) > 0 else 0
        }
    
    def _monte_carlo_stress_test(self, returns: pd.Series) -> Dict:
        """Perform Monte Carlo stress testing."""
        # Simplified Monte Carlo simulation
        np.random.seed(42)
        simulations = 1000
        
        # Generate random scenarios based on historical distribution
        simulated_returns = np.random.choice(returns.values, size=(simulations, 20), replace=True)
        cumulative_returns = np.prod(1 + simulated_returns, axis=1) - 1
        
        return {
            "worst_case": float(np.percentile(cumulative_returns, 1)),
            "fifth_percentile": float(np.percentile(cumulative_returns, 5)),
            "tenth_percentile": float(np.percentile(cumulative_returns, 10)),
            "expected_loss": float(np.mean(cumulative_returns[cumulative_returns < 0])),
            "probability_of_loss": float(np.mean(cumulative_returns < 0))
        }
    
    def _sector_specific_stress_scenarios(self, returns: pd.Series) -> Dict:
        """Generate sector-specific stress scenarios based on historical data."""
        try:
            # Calculate historical stress scenarios based on actual return data
            volatility = returns.std()
            mean_return = returns.mean()
            
            # Calculate stress scenarios based on historical volatility and returns
            sector_rotation_stress = -volatility * 2.5  # 2.5x volatility
            regulatory_stress = -volatility * 2.0       # 2.0x volatility
            commodity_price_stress = -volatility * 1.8  # 1.8x volatility
            currency_stress = -volatility * 1.5         # 1.5x volatility
            interest_rate_stress = -volatility * 1.2    # 1.2x volatility
            
            # Ensure scenarios are reasonable (not too extreme)
            max_stress = -0.25  # Maximum 25% stress
            sector_rotation_stress = max(sector_rotation_stress, max_stress)
            regulatory_stress = max(regulatory_stress, max_stress)
            commodity_price_stress = max(commodity_price_stress, max_stress)
            currency_stress = max(currency_stress, max_stress)
            interest_rate_stress = max(interest_rate_stress, max_stress)
            
            return {
                "sector_rotation_stress": float(sector_rotation_stress),
                "regulatory_stress": float(regulatory_stress),
                "commodity_price_stress": float(commodity_price_stress),
                "currency_stress": float(currency_stress),
                "interest_rate_stress": float(interest_rate_stress)
            }
        except Exception as e:
            logger.error(f"Error calculating sector stress scenarios: {e}")
            # Fallback to conservative estimates
            try:
                volatility = returns.std() if len(returns) > 0 else 0.02
                base_stress = -volatility * 2.0
                return {
                    "sector_rotation_stress": float(base_stress * 1.2),
                    "regulatory_stress": float(base_stress * 1.0),
                    "commodity_price_stress": float(base_stress * 0.8),
                    "currency_stress": float(base_stress * 0.6),
                    "interest_rate_stress": float(base_stress * 0.4)
                }
            except:
                # Ultimate fallback
                return {
                    "sector_rotation_stress": -0.10,
                    "regulatory_stress": -0.08,
                    "commodity_price_stress": -0.07,
                    "currency_stress": -0.06,
                    "interest_rate_stress": -0.05
                }
    
    def _market_crash_scenarios(self, returns: pd.Series, stock_data: pd.DataFrame) -> Dict:
        """Generate market crash scenarios based on historical data."""
        try:
            # Calculate crash scenarios based on historical volatility and extreme events
            volatility = returns.std()
            max_daily_loss = returns.min()  # Worst single day return
            
            # Calculate crash scenarios based on historical data
            black_swan_event = max_daily_loss * 3  # 3x worst daily loss
            systemic_crisis = max_daily_loss * 2.5  # 2.5x worst daily loss
            bubble_burst = max_daily_loss * 2.0    # 2.0x worst daily loss
            geopolitical_crisis = max_daily_loss * 1.8  # 1.8x worst daily loss
            economic_recession = max_daily_loss * 1.5   # 1.5x worst daily loss
            
            # Ensure scenarios are reasonable (not too extreme)
            max_crash = -0.30  # Maximum 30% crash
            black_swan_event = max(black_swan_event, max_crash)
            systemic_crisis = max(systemic_crisis, max_crash)
            bubble_burst = max(bubble_burst, max_crash)
            geopolitical_crisis = max(geopolitical_crisis, max_crash)
            economic_recession = max(economic_recession, max_crash)
            
            return {
                "black_swan_event": float(black_swan_event),
                "systemic_crisis": float(systemic_crisis),
                "bubble_burst": float(bubble_burst),
                "geopolitical_crisis": float(geopolitical_crisis),
                "economic_recession": float(economic_recession)
            }
        except Exception as e:
            logger.error(f"Error calculating market crash scenarios: {e}")
            # Fallback to conservative estimates
            try:
                volatility = returns.std() if len(returns) > 0 else 0.02
                max_daily_loss = returns.min() if len(returns) > 0 else -0.05
                base_crash = max_daily_loss * 2.0
                return {
                    "black_swan_event": float(max(base_crash * 1.5, -0.25)),
                    "systemic_crisis": float(max(base_crash * 1.25, -0.20)),
                    "bubble_burst": float(max(base_crash * 1.1, -0.18)),
                    "geopolitical_crisis": float(max(base_crash * 0.9, -0.15)),
                    "economic_recession": float(max(base_crash * 0.7, -0.12))
                }
            except:
                # Ultimate fallback
                return {
                    "black_swan_event": -0.20,
                    "systemic_crisis": -0.15,
                    "bubble_burst": -0.12,
                    "geopolitical_crisis": -0.10,
                    "economic_recession": -0.08
                }
    
    def _calculate_stress_score(self, historical_stress: Dict, monte_carlo_stress: Dict) -> int:
        """Calculate overall stress score."""
        # Combine various stress indicators
        worst_historical = abs(historical_stress.get("worst_20_day_period", -0.1))
        worst_monte_carlo = abs(monte_carlo_stress.get("worst_case", -0.2))
        stress_frequency = historical_stress.get("stress_frequency", 0)
        
        # Calculate weighted stress score
        stress_score = (worst_historical * 40 + worst_monte_carlo * 40 + stress_frequency * 20)
        return int(min(100, stress_score * 100))
    
    def _determine_stress_level(self, stress_score: int) -> str:
        """Determine stress level based on score."""
        if stress_score >= 70:
            return "High"
        elif stress_score >= 40:
            return "Medium"
        else:
            return "Low"
    
    def _generate_stress_mitigation_recommendations(self, stress_score: int, stress_level: str) -> List[str]:
        """Generate stress mitigation recommendations."""
        recommendations = []
        
        if stress_level == "High":
            recommendations.extend([
                "Implement strict risk management protocols",
                "Consider hedging strategies",
                "Reduce position sizes significantly",
                "Monitor market conditions closely",
                "Prepare contingency plans"
            ])
        elif stress_level == "Medium":
            recommendations.extend([
                "Review risk management procedures",
                "Consider partial hedging",
                "Monitor position sizes",
                "Stay informed about market developments"
            ])
        else:
            recommendations.extend([
                "Maintain current risk management",
                "Regular monitoring recommended",
                "Stay prepared for potential stress events"
            ])
        
        return recommendations

    # Helper methods for scenario analysis
    def _bull_market_scenario(self, returns: pd.Series, stock_data: pd.DataFrame, indicators: dict) -> Dict:
        """Generate bull market scenario based on actual data."""
        try:
            current_price = stock_data['close'].iloc[-1] if not stock_data.empty else 100
            
            # Calculate probability based on current trend and indicators
            positive_returns = len(returns[returns > 0])
            total_returns = len(returns)
            bull_probability = positive_returns / total_returns if total_returns > 0 else 0.25
            
            # Calculate return expectation based on historical data
            avg_positive_return = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.02
            bull_return_expectation = min(avg_positive_return * 252 * 0.5, 0.30)  # Annualized, capped at 30%
            
            # Calculate price target based on historical volatility
            volatility = returns.std()
            price_target = current_price * (1 + bull_return_expectation)
            
            # Determine confidence level based on trend strength
            rsi_values = indicators.get('rsi_14', [50])
            current_rsi = rsi_values[-1] if rsi_values else 50
            confidence_level = 0.5 + (current_rsi - 50) / 100  # Higher RSI = higher confidence
            confidence_level = max(0.3, min(0.9, confidence_level))  # Clamp between 0.3 and 0.9
            
            # Determine key drivers based on indicators
            key_drivers = []
            if current_rsi > 60:
                key_drivers.append("Strong momentum")
            if volatility < 0.02:
                key_drivers.append("Low volatility environment")
            if bull_probability > 0.6:
                key_drivers.append("Consistent positive returns")
            if len(key_drivers) == 0:
                key_drivers = ["Market recovery", "Sector rotation", "Technical breakout"]
            
            return {
                "scenario": "bull_market",
                "probability": float(bull_probability),
                "timeframe": "6-12 months",
                "price_target": float(price_target),
                "return_expectation": float(bull_return_expectation),
                "key_drivers": key_drivers,
                "confidence_level": float(confidence_level)
            }
        except Exception as e:
            logger.error(f"Error calculating bull market scenario: {e}")
            # Fallback to conservative estimates
            try:
                current_price = stock_data['close'].iloc[-1] if not stock_data.empty else 100
                return {
                    "scenario": "bull_market",
                    "probability": 0.25,
                    "timeframe": "6-12 months",
                    "price_target": current_price * 1.08,
                    "return_expectation": 0.08,
                    "key_drivers": ["Market recovery", "Sector rotation", "Technical breakout"],
                    "confidence_level": 0.6
                }
            except:
                return {
                    "scenario": "bull_market",
                    "probability": 0.25,
                    "timeframe": "6-12 months",
                    "price_target": 100.0,
                    "return_expectation": 0.08,
                    "key_drivers": ["Market recovery"],
                    "confidence_level": 0.5
                }
    
    def _bear_market_scenario(self, returns: pd.Series, stock_data: pd.DataFrame, indicators: dict) -> Dict:
        """Generate bear market scenario based on actual data."""
        try:
            current_price = stock_data['close'].iloc[-1] if not stock_data.empty else 100
            
            # Calculate probability based on current trend and indicators
            negative_returns = len(returns[returns < 0])
            total_returns = len(returns)
            bear_probability = negative_returns / total_returns if total_returns > 0 else 0.20
            
            # Calculate return expectation based on historical data
            avg_negative_return = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else -0.02
            bear_return_expectation = max(avg_negative_return * 252 * 0.5, -0.25)  # Annualized, capped at -25%
            
            # Calculate price target based on historical volatility
            volatility = returns.std()
            price_target = current_price * (1 + bear_return_expectation)
            
            # Determine confidence level based on trend strength
            rsi_values = indicators.get('rsi_14', [50])
            current_rsi = rsi_values[-1] if rsi_values else 50
            confidence_level = 0.5 - (current_rsi - 50) / 100  # Lower RSI = higher bear confidence
            confidence_level = max(0.3, min(0.9, confidence_level))  # Clamp between 0.3 and 0.9
            
            # Determine key drivers based on indicators
            key_drivers = []
            if current_rsi < 40:
                key_drivers.append("Weak momentum")
            if volatility > 0.03:
                key_drivers.append("High volatility environment")
            if bear_probability > 0.6:
                key_drivers.append("Consistent negative returns")
            if len(key_drivers) == 0:
                key_drivers = ["Market correction", "Economic concerns", "Technical breakdown"]
            
            return {
                "scenario": "bear_market",
                "probability": float(bear_probability),
                "timeframe": "3-6 months",
                "price_target": float(price_target),
                "return_expectation": float(bear_return_expectation),
                "key_drivers": key_drivers,
                "confidence_level": float(confidence_level)
            }
        except Exception as e:
            logger.error(f"Error calculating bear market scenario: {e}")
            # Fallback
            try:
                current_price = stock_data['close'].iloc[-1] if not stock_data.empty else 100
                return {
                    "scenario": "bear_market",
                    "probability": 0.20,
                    "timeframe": "3-6 months",
                    "price_target": current_price * 0.80,
                    "return_expectation": -0.20,
                    "key_drivers": ["Economic slowdown", "Rising interest rates", "Market correction"],
                    "confidence_level": 0.6
                }
            except:
                return {
                    "scenario": "bear_market",
                    "probability": 0.20,
                    "timeframe": "3-6 months",
                    "price_target": 80.0,
                    "return_expectation": -0.20,
                    "key_drivers": ["Market correction"],
                    "confidence_level": 0.5
                }
    
    def _sideways_market_scenario(self, returns: pd.Series, stock_data: pd.DataFrame, indicators: dict) -> Dict:
        """Generate sideways market scenario based on actual data."""
        try:
            current_price = stock_data['close'].iloc[-1] if not stock_data.empty else 100
            
            # Calculate probability based on volatility and trend consistency
            volatility = returns.std()
            mean_return = abs(returns.mean())
            
            # Sideways probability is higher when volatility is low and mean return is small
            sideways_probability = max(0.2, min(0.5, 1 - volatility * 10 - mean_return * 10))
            
            # Calculate return expectation for sideways market
            sideways_return_expectation = mean_return * 252 * 0.3  # Small annualized return
            sideways_return_expectation = max(-0.05, min(0.10, sideways_return_expectation))  # Clamp between -5% and 10%
            
            # Calculate price target
            price_target = current_price * (1 + sideways_return_expectation)
            
            # Determine confidence level based on volatility
            confidence_level = 0.8 - volatility * 10  # Lower volatility = higher confidence
            confidence_level = max(0.5, min(0.9, confidence_level))  # Clamp between 0.5 and 0.9
            
            # Determine key drivers based on market conditions
            key_drivers = []
            if volatility < 0.015:
                key_drivers.append("Low volatility environment")
            if mean_return < 0.005:
                key_drivers.append("Minimal trend direction")
            if len(key_drivers) == 0:
                key_drivers = ["Range-bound trading", "Consolidation", "Mixed signals"]
            
            return {
                "scenario": "sideways_market",
                "probability": float(sideways_probability),
                "timeframe": "3-9 months",
                "price_target": float(price_target),
                "return_expectation": float(sideways_return_expectation),
                "key_drivers": key_drivers,
                "confidence_level": float(confidence_level)
            }
        except Exception as e:
            logger.error(f"Error calculating sideways market scenario: {e}")
            # Fallback
            try:
                current_price = stock_data['close'].iloc[-1] if not stock_data.empty else 100
                return {
                    "scenario": "sideways_market",
                    "probability": 0.35,
                    "timeframe": "3-9 months",
                    "price_target": current_price * 1.05,
                    "return_expectation": 0.05,
                    "key_drivers": ["Range-bound trading", "Consolidation", "Mixed signals"],
                    "confidence_level": 0.8
                }
            except:
                return {
                    "scenario": "sideways_market",
                    "probability": 0.35,
                    "timeframe": "3-9 months",
                    "price_target": 105.0,
                    "return_expectation": 0.05,
                    "key_drivers": ["Range-bound trading"],
                    "confidence_level": 0.7
                }
    
    def _volatility_spike_scenario(self, returns: pd.Series, stock_data: pd.DataFrame) -> Dict:
        """Generate volatility spike scenario based on actual data."""
        try:
            current_price = stock_data['close'].iloc[-1] if not stock_data.empty else 100
            
            # Calculate volatility spike probability based on current volatility vs historical
            current_volatility = returns.std()
            historical_volatility = returns.rolling(window=20).std().mean()
            
            # Volatility spike probability is higher when current volatility is low relative to historical
            volatility_ratio = current_volatility / historical_volatility if historical_volatility > 0 else 1
            spike_probability = max(0.1, min(0.4, 1 - volatility_ratio * 0.5))
            
            # Calculate return expectation for volatility spike
            spike_return_expectation = -current_volatility * 2  # Negative return during volatility spike
            spike_return_expectation = max(-0.20, min(-0.05, spike_return_expectation))  # Clamp between -20% and -5%
            
            # Calculate price target
            price_target = current_price * (1 + spike_return_expectation)
            
            # Determine confidence level based on volatility trend
            confidence_level = 0.5 - (volatility_ratio - 1) * 0.2  # Lower confidence when volatility is already high
            confidence_level = max(0.3, min(0.7, confidence_level))  # Clamp between 0.3 and 0.7
            
            # Determine key drivers based on market conditions
            key_drivers = []
            if current_volatility < historical_volatility * 0.8:
                key_drivers.append("Low current volatility")
            if len(returns) > 20:
                recent_volatility = returns.tail(20).std()
                if recent_volatility < current_volatility:
                    key_drivers.append("Volatility compression")
            if len(key_drivers) == 0:
                key_drivers = ["Market uncertainty", "Event-driven volatility", "Liquidity concerns"]
            
            return {
                "scenario": "volatility_spike",
                "probability": float(spike_probability),
                "timeframe": "1-3 months",
                "price_target": float(price_target),
                "return_expectation": float(spike_return_expectation),
                "key_drivers": key_drivers,
                "confidence_level": float(confidence_level)
            }
        except Exception as e:
            logger.error(f"Error calculating volatility spike scenario: {e}")
            # Fallback
            try:
                current_price = stock_data['close'].iloc[-1] if not stock_data.empty else 100
                return {
                    "scenario": "volatility_spike",
                    "probability": 0.20,
                    "timeframe": "1-3 months",
                    "price_target": current_price * 0.90,
                    "return_expectation": -0.10,
                    "key_drivers": ["Market uncertainty", "Event-driven volatility", "Liquidity concerns"],
                    "confidence_level": 0.5
                }
            except:
                return {
                    "scenario": "volatility_spike",
                    "probability": 0.20,
                    "timeframe": "1-3 months",
                    "price_target": 90.0,
                    "return_expectation": -0.10,
                    "key_drivers": ["Market uncertainty"],
                    "confidence_level": 0.5
                }
    
    def _calculate_scenario_confidence_levels(self, bull_scenario: Dict, bear_scenario: Dict, 
                                            sideways_scenario: Dict, volatility_scenario: Dict) -> Dict:
        """Calculate confidence levels for each scenario."""
        return {
            "bull_scenario_confidence": bull_scenario.get("confidence_level", 0.7),
            "bear_scenario_confidence": bear_scenario.get("confidence_level", 0.6),
            "sideways_scenario_confidence": sideways_scenario.get("confidence_level", 0.8),
            "volatility_scenario_confidence": volatility_scenario.get("confidence_level", 0.5),
            "overall_confidence": 0.65
        }
    
    def _calculate_scenario_probabilities(self, returns: pd.Series, indicators: dict, 
                                        confidence_levels: Dict) -> Dict:
        """Calculate probability scores for scenarios based on actual data."""
        try:
            # Calculate probabilities based on actual market conditions
            current_volatility = returns.std()
            current_trend = returns.mean()
            positive_returns = len(returns[returns > 0])
            total_returns = len(returns)
            
            # Base probabilities from actual return distribution
            bull_prob = positive_returns / total_returns if total_returns > 0 else 0.25
            bear_prob = (total_returns - positive_returns) / total_returns if total_returns > 0 else 0.20
            
            # Adjust based on volatility
            if current_volatility > 0.03:  # High volatility
                volatility_adjustment = 0.1
                bull_prob = max(0.1, bull_prob - volatility_adjustment)
                bear_prob = min(0.4, bear_prob + volatility_adjustment)
            elif current_volatility < 0.015:  # Low volatility
                volatility_adjustment = 0.05
                bull_prob = min(0.5, bull_prob + volatility_adjustment)
                bear_prob = max(0.1, bear_prob - volatility_adjustment)
            
            # Adjust based on trend strength
            if abs(current_trend) > 0.001:  # Strong trend
                trend_adjustment = 0.05
                if current_trend > 0:
                    bull_prob = min(0.6, bull_prob + trend_adjustment)
                    bear_prob = max(0.1, bear_prob - trend_adjustment)
                else:
                    bear_prob = min(0.5, bear_prob + trend_adjustment)
                    bull_prob = max(0.1, bull_prob - trend_adjustment)
            
            # Calculate sideways probability as residual
            sideways_prob = max(0.1, 1 - bull_prob - bear_prob - 0.1)  # Leave 10% for volatility spike
            
            # Volatility spike probability based on volatility regime
            volatility_spike_prob = 0.1  # Base 10%
            if current_volatility < returns.rolling(window=20).std().mean() * 0.8:
                volatility_spike_prob = 0.15  # Higher probability when volatility is compressed
            
            # Normalize probabilities to sum to 1
            total_prob = bull_prob + bear_prob + sideways_prob + volatility_spike_prob
            if total_prob > 0:
                bull_prob = bull_prob / total_prob
                bear_prob = bear_prob / total_prob
                sideways_prob = sideways_prob / total_prob
                volatility_spike_prob = volatility_spike_prob / total_prob
            
            return {
                "bull": float(bull_prob),
                "bear": float(bear_prob),
                "sideways": float(sideways_prob),
                "volatility": float(volatility_spike_prob)
            }
        except Exception as e:
            logger.error(f"Error calculating scenario probabilities: {e}")
            # Ultimate fallback
            return {"bull": 0.25, "bear": 0.20, "sideways": 0.35, "volatility": 0.20}
    
    def _calculate_scenario_impact_scores(self, bull_scenario: Dict, bear_scenario: Dict, 
                                        sideways_scenario: Dict, volatility_scenario: Dict) -> Dict:
        """Calculate impact scores for scenarios based on actual return expectations."""
        try:
            # Calculate impact scores based on actual return expectations
            bull_impact = int(bull_scenario.get("return_expectation", 0.25) * 100)
            bear_impact = int(abs(bear_scenario.get("return_expectation", -0.20)) * 100)
            sideways_impact = int(sideways_scenario.get("return_expectation", 0.05) * 100)
            volatility_impact = int(abs(volatility_scenario.get("return_expectation", -0.10)) * 100)
            
            # Calculate overall impact score as weighted average
            bull_prob = bull_scenario.get("probability", 0.25)
            bear_prob = bear_scenario.get("probability", 0.20)
            sideways_prob = sideways_scenario.get("probability", 0.35)
            volatility_prob = volatility_scenario.get("probability", 0.20)
            
            overall_impact = (
                bull_impact * bull_prob +
                bear_impact * bear_prob +
                sideways_impact * sideways_prob +
                volatility_impact * volatility_prob
            )
            
            return {
                "bull_scenario_impact": bull_impact,
                "bear_scenario_impact": bear_impact,
                "sideways_scenario_impact": sideways_impact,
                "volatility_scenario_impact": volatility_impact,
                "overall_impact_score": int(overall_impact)
            }
        except Exception as e:
            logger.error(f"Error calculating scenario impact scores: {e}")
            # Ultimate fallback
            return {
                "bull_scenario_impact": 25,
                "bear_scenario_impact": 20,
                "sideways_scenario_impact": 5,
                "volatility_scenario_impact": 10,
                "overall_impact_score": 15
            }