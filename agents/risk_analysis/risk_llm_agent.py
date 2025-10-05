#!/usr/bin/env python3
"""
Risk LLM Agent

This agent takes raw risk analysis results from the risk orchestrator and sends them 
to the LLM for interpretation. It provides clear, concise risk insights for the 
final decision agent, similar to how indicator agents and volume agents work.

The agent:
- Receives comprehensive risk analysis data from RiskAgentsOrchestrator
- Formats it into a structured prompt for LLM consumption
- Calls Gemini to generate clear risk insights
- Returns structured risk analysis for the final decision LLM
"""

import logging
import asyncio
import time
import json
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class RiskLLMAgent:
    """
    Risk LLM Agent that analyzes comprehensive risk data using Gemini.
    
    This agent bridges the gap between technical risk analysis and clear, 
    actionable risk insights for trading decisions.
    """
    
    def __init__(self):
        self.agent_name = "risk_llm_agent"
        # Import here to avoid circular dependencies
        from gemini.gemini_client import GeminiClient
        
        # Load the prompt template from file
        self.prompt_template = self._load_prompt_template()
        
        # Use rotating API key with fallback
        try:
            from gemini.api_key_manager import get_api_key_manager
            key_manager = get_api_key_manager()
            api_key = key_manager.get_key("risk_llm")
            self.gemini_client = GeminiClient(api_key=api_key, agent_name="risk_llm")
            logger.info(f"[{self.agent_name.upper()}] Initialized with key rotation")
        except Exception as e:
            # Fallback to default initialization
            logger.warning(f"[{self.agent_name.upper()}] Key manager unavailable, using default: {e}")
            self.gemini_client = GeminiClient()
            logger.info(f"[{self.agent_name.upper()}] Initialized with default key")
    
    async def analyze_risk_with_llm(
        self, 
        symbol: str,
        risk_analysis_result,  # RiskAnalysisResult from orchestrator
        context: str = ""
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze comprehensive risk data using LLM to generate clear risk insights.
        
        Args:
            symbol: Stock symbol
            risk_analysis_result: RiskAnalysisResult from RiskAgentsOrchestrator
            context: Additional context string
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (success, risk_llm_analysis)
        """
        start_time = time.time()
        logger.info(f"[{self.agent_name.upper()}] Starting LLM analysis for {symbol}")
        print(f"[RISK_LLM_AGENT_DEBUG] Starting risk LLM analysis for {symbol}...")
        
        try:
            # Step 1: Build the prompt from risk analysis data
            prompt = self._build_risk_prompt(symbol, risk_analysis_result, context)
            
            if not prompt:
                logger.error(f"[{self.agent_name.upper()}] Failed to build prompt for {symbol}")
                return False, {"error": "Failed to build risk prompt", "agent": self.agent_name}
            
            logger.info(f"[{self.agent_name.upper()}] Built prompt ({len(prompt)} chars) for {symbol}")
            
            # Step 2: Call LLM with the prompt
            print(f"[RISK_LLM_AGENT_DEBUG] Sending request to Gemini API for {symbol}...")
            llm_start = time.time()
            
            try:
                # Use the core LLM call with code execution capability
                text_response, code_results, exec_results = await self.gemini_client.core.call_llm_with_code_execution(
                    prompt=prompt
                )
                
                llm_time = time.time() - llm_start
                print(f"[RISK_LLM_AGENT_DEBUG] Received response from Gemini API for {symbol} in {llm_time:.2f}s")
                logger.info(f"[{self.agent_name.upper()}] LLM response received in {llm_time:.2f}s for {symbol}")
                
            except Exception as llm_error:
                logger.error(f"[{self.agent_name.upper()}] LLM call failed for {symbol}: {llm_error}")
                print(f"[RISK_LLM_AGENT_DEBUG] LLM call failed for {symbol}: {llm_error}")
                return False, {
                    "error": f"LLM call failed: {str(llm_error)}",
                    "agent": self.agent_name
                }
            
            # Step 3: Structure the response
            processing_time = time.time() - start_time
            
            # Build structured response (current integration uses dict format)
            advanced_metrics = risk_analysis_result.get('advanced_risk_metrics', {}) if isinstance(risk_analysis_result, dict) else {}
            structured_analysis = {
                "success": True,
                "agent": self.agent_name,
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "llm_processing_time": llm_time,
                
                # LLM Analysis - This is the key output for final decision
                "risk_bullets": text_response or "",
                "code_executions": len(code_results) if code_results else 0,
                "calculation_results": len(exec_results) if exec_results else 0,
                
                # Risk Summary (from quantitative analysis)
                "overall_risk_score": advanced_metrics.get('risk_score', 0),
                "overall_confidence": 0.85,  # Default confidence for quantitative analysis
                "successful_agents": 1,
                "failed_agents": 0,
                
                # Risk analysis summary
                "risk_summary": {
                    'overall_level': advanced_metrics.get('risk_level', 'Medium'),
                    'risk_score': advanced_metrics.get('risk_score', 0),
                    'key_risk_factors': risk_analysis_result.get('overall_risk_assessment', {}).get('key_risk_factors', []) if isinstance(risk_analysis_result, dict) else []
                },
                "risk_breakdown": advanced_metrics.get('risk_components', {}),
                
                # Metadata
                "prompt_length": len(prompt),
                "response_length": len(text_response) if text_response else 0,
                "confidence": 0.85  # Fixed confidence for quantitative analysis
            }
            
            logger.info(f"[{self.agent_name.upper()}] Successfully completed analysis for {symbol} in {processing_time:.2f}s")
            print(f"[RISK_LLM_AGENT_DEBUG] Finished risk LLM analysis for {symbol}. Success: True")
            
            return True, structured_analysis
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Risk LLM analysis failed: {str(e)}"
            logger.error(f"[{self.agent_name.upper()}] {error_msg} for {symbol}")
            print(f"[RISK_LLM_AGENT_DEBUG] Error in risk LLM analysis for {symbol}: {e}")
            
            return False, {
                "success": False,
                "error": error_msg,
                "agent": self.agent_name,
                "symbol": symbol,
                "processing_time": processing_time
            }
    
    def _load_prompt_template(self) -> str:
        """
        Load the risk analysis prompt template from the external text file.
        
        Returns:
            str: The prompt template content, or empty string if loading fails
        """
        try:
            # Get the path to the prompts directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(os.path.dirname(current_dir))
            prompt_file_path = os.path.join(backend_dir, 'prompts', 'risk_analysis_prompt.txt')
            
            # Load the template from file
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            logger.info(f"[{self.agent_name.upper()}] Successfully loaded prompt template from {prompt_file_path}")
            return template
            
        except Exception as e:
            logger.error(f"[{self.agent_name.upper()}] Failed to load prompt template: {e}")
            # Return a basic fallback template
            return """# Risk Analysis Template Loading Failed
Please analyze the risk data provided and generate risk assessment bullets.
Data: {symbol} - {company} ({sector})
Context: {context}

Provide comprehensive risk analysis considering all quantitative metrics."""
    
    def _build_risk_prompt(self, symbol: str, risk_analysis_result, context: str) -> str:
        """
        Build enhanced quantitative risk analysis prompt for the final decision agent.
        
        Uses comprehensive quantitative metrics, stress testing, and scenario analysis 
        to provide multi-timeframe risk assessment for trading decisions.
        """
        try:
            # Only use enhanced quantitative format (all current integrations use dict format)
            if isinstance(risk_analysis_result, dict):
                return self._build_enhanced_quantitative_prompt(symbol, risk_analysis_result, context)
            else:
                # Fallback for any unexpected formats
                logger.warning(f"[{self.agent_name.upper()}] Unexpected risk analysis format for {symbol}")
                return self._build_enhanced_quantitative_prompt(symbol, {}, context)
            
        except Exception as e:
            logger.error(f"[{self.agent_name.upper()}] Error building risk prompt: {e}")
            return ""
    
    def _build_enhanced_quantitative_prompt(self, symbol: str, risk_analysis_result: Dict, context: str) -> str:
        """
        Build enhanced quantitative risk analysis prompt with comprehensive data utilization.
        
        Uses all available quantitative metrics, stress testing, and scenario analysis 
        to provide multi-timeframe risk assessment for trading decisions.
        """
        try:
            # Extract comprehensive data from enhanced quantitative risk analysis
            advanced_metrics = risk_analysis_result.get('advanced_risk_metrics', {})
            stress_testing = risk_analysis_result.get('stress_testing', {})
            scenario_analysis = risk_analysis_result.get('scenario_analysis', {})
            overall_assessment = risk_analysis_result.get('overall_risk_assessment', {})
            
            # Extract metadata
            timestamp = risk_analysis_result.get('timestamp', datetime.now().isoformat())
            company = risk_analysis_result.get('company', 'Unknown Company')
            sector = risk_analysis_result.get('sector', 'Unknown Sector')
            
            # Advanced risk metrics
            risk_score = advanced_metrics.get('risk_score', 0)
            risk_level = advanced_metrics.get('risk_level', 'Medium')
            sharpe_ratio = advanced_metrics.get('sharpe_ratio', 0)
            sortino_ratio = advanced_metrics.get('sortino_ratio', 0)
            calmar_ratio = advanced_metrics.get('calmar_ratio', 0)
            max_drawdown = advanced_metrics.get('max_drawdown', 0)
            current_drawdown = advanced_metrics.get('current_drawdown', 0)
            drawdown_duration = advanced_metrics.get('drawdown_duration', 0)
            var_95 = advanced_metrics.get('var_95', 0)
            var_99 = advanced_metrics.get('var_99', 0)
            expected_shortfall_95 = advanced_metrics.get('expected_shortfall_95', 0)
            annualized_volatility = advanced_metrics.get('annualized_volatility', 0)
            skewness = advanced_metrics.get('skewness', 0)
            kurtosis = advanced_metrics.get('kurtosis', 0)
            tail_frequency = advanced_metrics.get('tail_frequency', 0)
            
            # Risk components
            risk_components = advanced_metrics.get('risk_components', {})
            volatility_risk = risk_components.get('volatility_risk', 'Unknown')
            drawdown_risk = risk_components.get('drawdown_risk', 'Unknown')
            tail_risk = risk_components.get('tail_risk', 'Unknown')
            liquidity_risk = risk_components.get('liquidity_risk', 'Unknown')
            sector_risk = risk_components.get('sector_risk', 'Unknown')
            
            # Stress testing data
            stress_scenarios = stress_testing.get('stress_scenarios', {})
            historical_stress = stress_scenarios.get('historical_stress', {})
            monte_carlo_stress = stress_scenarios.get('monte_carlo_stress', {})
            sector_stress = stress_scenarios.get('sector_stress', {})
            crash_scenarios = stress_scenarios.get('crash_scenarios', {})
            
            # Historical stress metrics
            worst_20_day_period = historical_stress.get('worst_20_day_period', 0)
            second_worst_period = historical_stress.get('second_worst_period', 0)
            stress_frequency = historical_stress.get('stress_frequency', 0)
            
            # Monte Carlo stress metrics
            worst_case = monte_carlo_stress.get('worst_case', 0)
            fifth_percentile = monte_carlo_stress.get('fifth_percentile', 0)
            expected_loss = monte_carlo_stress.get('expected_loss', 0)
            probability_of_loss = monte_carlo_stress.get('probability_of_loss', 0)
            
            # Sector stress metrics
            sector_rotation_stress = sector_stress.get('sector_rotation_stress', 0)
            regulatory_stress = sector_stress.get('regulatory_stress', 0)
            economic_recession = crash_scenarios.get('economic_recession', 0)
            
            # Market crash scenarios
            black_swan_event = crash_scenarios.get('black_swan_event', 0)
            systemic_crisis = crash_scenarios.get('systemic_crisis', 0)
            geopolitical_crisis = crash_scenarios.get('geopolitical_crisis', 0)
            
            # Scenario analysis data
            expected_outcomes = scenario_analysis.get('expected_outcomes', {})
            probability_scores = scenario_analysis.get('probability_scores', {})
            
            # Bull scenario
            bull_scenario = expected_outcomes.get('bull_scenario', {})
            bull_probability = probability_scores.get('bull', 0)  # Already in 0-1 range
            bull_timeframe = bull_scenario.get('timeframe', '6-12 months')
            bull_price_target = bull_scenario.get('price_target', 0)
            bull_return_expectation = bull_scenario.get('return_expectation', 0)  # Already in decimal format
            bull_key_drivers = ', '.join(bull_scenario.get('key_drivers', ['Market recovery']))
            bull_confidence = bull_scenario.get('confidence_level', 0.5)  # Keep as decimal
            
            # Bear scenario
            bear_scenario = expected_outcomes.get('bear_scenario', {})
            bear_probability = probability_scores.get('bear', 0)  # Already in 0-1 range
            bear_timeframe = bear_scenario.get('timeframe', '3-6 months')
            bear_price_target = bear_scenario.get('price_target', 0)
            bear_return_expectation = bear_scenario.get('return_expectation', 0)  # Already in decimal format
            bear_key_drivers = ', '.join(bear_scenario.get('key_drivers', ['Market correction']))
            bear_confidence = bear_scenario.get('confidence_level', 0.5)  # Keep as decimal
            
            # Sideways scenario
            sideways_scenario = expected_outcomes.get('sideways_scenario', {})
            sideways_probability = probability_scores.get('sideways', 0)  # Already in 0-1 range
            sideways_timeframe = sideways_scenario.get('timeframe', '3-9 months')
            sideways_price_target = sideways_scenario.get('price_target', 0)
            sideways_return_expectation = sideways_scenario.get('return_expectation', 0)  # Already in decimal format
            sideways_key_drivers = ', '.join(sideways_scenario.get('key_drivers', ['Range-bound trading']))
            
            # Volatility scenario
            volatility_scenario = expected_outcomes.get('volatility_scenario', {})
            volatility_probability = probability_scores.get('volatility', 0)  # Already in 0-1 range
            volatility_timeframe = volatility_scenario.get('timeframe', '1-3 months')
            volatility_return_expectation = volatility_scenario.get('return_expectation', 0)  # Already in decimal format
            volatility_key_drivers = ', '.join(volatility_scenario.get('key_drivers', ['Market uncertainty']))
            
            # Build the enhanced prompt using the loaded template
            prompt = self.prompt_template.format(
                timestamp=timestamp,
                symbol=symbol,
                company=company,
                sector=sector,
                context=context,
                risk_level=risk_level,
                risk_score=risk_score,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall_95=expected_shortfall_95,
                annualized_volatility=annualized_volatility,
                skewness=skewness,
                kurtosis=kurtosis,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                drawdown_duration=drawdown_duration,
                volatility_risk=volatility_risk,
                drawdown_risk=drawdown_risk,
                tail_risk=tail_risk,
                tail_frequency=tail_frequency,
                liquidity_risk=liquidity_risk,
                sector_risk=sector_risk,
                worst_20_day_period=worst_20_day_period,
                second_worst_period=second_worst_period,
                stress_frequency=stress_frequency,
                worst_case=worst_case,
                fifth_percentile=fifth_percentile,
                expected_loss=expected_loss,
                probability_of_loss=probability_of_loss,
                sector_rotation_stress=sector_rotation_stress,
                regulatory_stress=regulatory_stress,
                economic_recession=economic_recession,
                black_swan_event=black_swan_event,
                systemic_crisis=systemic_crisis,
                geopolitical_crisis=geopolitical_crisis,
                bull_probability=bull_probability,
                bull_timeframe=bull_timeframe,
                bull_price_target=bull_price_target,
                bull_return_expectation=bull_return_expectation,
                bull_key_drivers=bull_key_drivers,
                bull_confidence=bull_confidence,
                bear_probability=bear_probability,
                bear_timeframe=bear_timeframe,
                bear_price_target=bear_price_target,
                bear_return_expectation=bear_return_expectation,
                bear_key_drivers=bear_key_drivers,
                bear_confidence=bear_confidence,
                sideways_probability=sideways_probability,
                sideways_timeframe=sideways_timeframe,
                sideways_price_target=sideways_price_target,
                sideways_return_expectation=sideways_return_expectation,
                sideways_key_drivers=sideways_key_drivers,
                volatility_probability=volatility_probability,
                volatility_timeframe=volatility_timeframe,
                volatility_return_expectation=volatility_return_expectation,
                volatility_key_drivers=volatility_key_drivers
            )
            
            return prompt
            
        except Exception as e:
            logger.error(f"[{self.agent_name.upper()}] Error building enhanced quantitative prompt: {e}")
            return ""
    
    


# Global instance for use by the analysis service
risk_llm_agent = RiskLLMAgent()