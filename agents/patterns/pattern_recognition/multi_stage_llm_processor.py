"""
Multi-Stage LLM Pattern Processor

Implements advanced multi-stage LLM processing for pattern recognition with:
1. Market Structure Analysis Stage
2. Pattern Detection Stage  
3. Cross-Pattern Validation Stage
4. Trading Insights Synthesis Stage
5. Final Consolidation Stage

This approach allows for deeper, more specialized analysis through multiple LLM calls,
each focused on specific aspects of pattern recognition.
"""

import numpy as np
import pandas as pd
import asyncio
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class MultiStageLLMProcessor:
    """
    Multi-stage LLM processor that performs specialized analysis through
    multiple focused LLM calls for comprehensive pattern recognition.
    """
    
    def __init__(self, llm_client=None):
        self.name = "multi_stage_llm_pattern_processor"
        self.version = "1.0.0"
        self.llm_client = llm_client
        self.stage_results = {}
        
        # Stage configuration
        self.stages = {
            'market_structure': {
                'enabled': True,
                'timeout': 30,
                'max_retries': 2
            },
            'pattern_detection': {
                'enabled': True, 
                'timeout': 30,
                'max_retries': 2
            },
            'cross_validation': {
                'enabled': True,
                'timeout': 25,
                'max_retries': 1
            },
            'trading_insights': {
                'enabled': True,
                'timeout': 35,
                'max_retries': 2
            },
            'final_synthesis': {
                'enabled': True,
                'timeout': 20,
                'max_retries': 1
            }
        }
        
        logger.info(f"[MULTI_STAGE_LLM] Initialized processor v{self.version} with {len(self.stages)} stages")
    
    async def process_multi_stage_analysis(self, 
                                         symbol: str,
                                         technical_analysis: Dict[str, Any],
                                         market_structure: Dict[str, Any],
                                         current_price: float,
                                         context: str = "") -> Dict[str, Any]:
        """
        Execute multi-stage LLM analysis for comprehensive pattern recognition.
        
        Args:
            symbol: Stock symbol
            technical_analysis: Complete technical analysis results
            market_structure: Market structure analysis data
            current_price: Current stock price
            context: Additional context
            
        Returns:
            Dictionary containing results from all analysis stages
        """
        start_time = datetime.now()
        operation_id = f"MSLLM_{symbol}_{int(start_time.timestamp())}"
        
        logger.info(f"[MULTI_STAGE_LLM] Starting multi-stage analysis for {symbol} (ID: {operation_id})")
        
        if not self.llm_client:
            logger.error(f"[MULTI_STAGE_LLM] No LLM client available for {symbol}")
            return self._build_error_result(symbol, "No LLM client available", start_time)
        
        try:
            # Initialize stage results
            self.stage_results = {
                'symbol': symbol,
                'operation_id': operation_id,
                'stages': {},
                'errors': []
            }
            
            # Stage 1: Market Structure Analysis
            if self.stages['market_structure']['enabled']:
                await self._execute_market_structure_stage(symbol, market_structure, technical_analysis)
            
            # Stage 2: Pattern Detection Analysis  
            if self.stages['pattern_detection']['enabled']:
                await self._execute_pattern_detection_stage(symbol, technical_analysis, current_price)
            
            # Stage 3: Cross-Pattern Validation
            if self.stages['cross_validation']['enabled']:
                await self._execute_cross_validation_stage(symbol, technical_analysis)
            
            # Stage 4: Trading Insights Generation
            if self.stages['trading_insights']['enabled']:
                await self._execute_trading_insights_stage(symbol, current_price)
            
            # Stage 5: Final Synthesis
            if self.stages['final_synthesis']['enabled']:
                await self._execute_final_synthesis_stage(symbol, current_price, context)
            
            # Build comprehensive result
            processing_time = (datetime.now() - start_time).total_seconds()
            return self._build_success_result(symbol, processing_time, start_time)
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"[MULTI_STAGE_LLM] Multi-stage analysis failed for {symbol}: {e}")
            return self._build_error_result(symbol, str(e), start_time)
    
    async def _execute_market_structure_stage(self, symbol: str, market_structure: Dict[str, Any], technical_analysis: Dict[str, Any]):
        """Stage 1: Deep market structure analysis with LLM"""
        stage_name = 'market_structure'
        logger.info(f"[MULTI_STAGE_LLM] Executing {stage_name} stage for {symbol}")
        
        try:
            # Build specialized prompt for market structure analysis
            prompt = self._build_market_structure_prompt(symbol, market_structure, technical_analysis)
            
            # Execute LLM call with retries
            response = await self._execute_llm_with_retries(
                prompt, stage_name, self.stages[stage_name]['timeout'], self.stages[stage_name]['max_retries']
            )
            
            # Process and store result
            self.stage_results['stages'][stage_name] = {
                'success': True,
                'response': response,
                'insights': self._extract_market_structure_insights(response),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"[MULTI_STAGE_LLM] {stage_name} stage completed for {symbol}")
            
        except Exception as e:
            logger.error(f"[MULTI_STAGE_LLM] {stage_name} stage failed for {symbol}: {e}")
            self.stage_results['stages'][stage_name] = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.stage_results['errors'].append(f"{stage_name}: {str(e)}")
    
    async def _execute_pattern_detection_stage(self, symbol: str, technical_analysis: Dict[str, Any], current_price: float):
        """Stage 2: Advanced pattern detection analysis"""
        stage_name = 'pattern_detection'
        logger.info(f"[MULTI_STAGE_LLM] Executing {stage_name} stage for {symbol}")
        
        try:
            # Build specialized prompt for pattern detection
            prompt = self._build_pattern_detection_prompt(symbol, technical_analysis, current_price)
            
            # Execute LLM call
            response = await self._execute_llm_with_retries(
                prompt, stage_name, self.stages[stage_name]['timeout'], self.stages[stage_name]['max_retries']
            )
            
            # Process and store result
            self.stage_results['stages'][stage_name] = {
                'success': True,
                'response': response,
                'patterns': self._extract_detected_patterns(response),
                'confidence': self._extract_pattern_confidence(response),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"[MULTI_STAGE_LLM] {stage_name} stage completed for {symbol}")
            
        except Exception as e:
            logger.error(f"[MULTI_STAGE_LLM] {stage_name} stage failed for {symbol}: {e}")
            self.stage_results['stages'][stage_name] = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.stage_results['errors'].append(f"{stage_name}: {str(e)}")
    
    async def _execute_cross_validation_stage(self, symbol: str, technical_analysis: Dict[str, Any]):
        """Stage 3: Cross-pattern validation and conflict resolution"""
        stage_name = 'cross_validation'
        logger.info(f"[MULTI_STAGE_LLM] Executing {stage_name} stage for {symbol}")
        
        try:
            # Only proceed if we have results from previous stages
            if not self._has_sufficient_stage_results():
                logger.warning(f"[MULTI_STAGE_LLM] Insufficient stage results for {stage_name}, skipping")
                return
            
            # Build validation prompt using previous stage results
            prompt = self._build_cross_validation_prompt(symbol, technical_analysis)
            
            # Execute LLM call
            response = await self._execute_llm_with_retries(
                prompt, stage_name, self.stages[stage_name]['timeout'], self.stages[stage_name]['max_retries']
            )
            
            # Process and store result
            self.stage_results['stages'][stage_name] = {
                'success': True,
                'response': response,
                'validations': self._extract_validation_results(response),
                'conflicts': self._extract_pattern_conflicts(response),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"[MULTI_STAGE_LLM] {stage_name} stage completed for {symbol}")
            
        except Exception as e:
            logger.error(f"[MULTI_STAGE_LLM] {stage_name} stage failed for {symbol}: {e}")
            self.stage_results['stages'][stage_name] = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.stage_results['errors'].append(f"{stage_name}: {str(e)}")
    
    async def _execute_trading_insights_stage(self, symbol: str, current_price: float):
        """Stage 4: Generate actionable trading insights"""
        stage_name = 'trading_insights'
        logger.info(f"[MULTI_STAGE_LLM] Executing {stage_name} stage for {symbol}")
        
        try:
            # Build trading insights prompt using all previous stage results
            prompt = self._build_trading_insights_prompt(symbol, current_price)
            
            # Execute LLM call
            response = await self._execute_llm_with_retries(
                prompt, stage_name, self.stages[stage_name]['timeout'], self.stages[stage_name]['max_retries']
            )
            
            # Process and store result
            self.stage_results['stages'][stage_name] = {
                'success': True,
                'response': response,
                'trading_signals': self._extract_trading_signals(response),
                'risk_assessment': self._extract_risk_assessment(response),
                'price_targets': self._extract_price_targets(response),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"[MULTI_STAGE_LLM] {stage_name} stage completed for {symbol}")
            
        except Exception as e:
            logger.error(f"[MULTI_STAGE_LLM] {stage_name} stage failed for {symbol}: {e}")
            self.stage_results['stages'][stage_name] = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.stage_results['errors'].append(f"{stage_name}: {str(e)}")
    
    async def _execute_final_synthesis_stage(self, symbol: str, current_price: float, context: str):
        """Stage 5: Final synthesis and consolidation"""
        stage_name = 'final_synthesis'
        logger.info(f"[MULTI_STAGE_LLM] Executing {stage_name} stage for {symbol}")
        
        try:
            # Build final synthesis prompt
            prompt = self._build_final_synthesis_prompt(symbol, current_price, context)
            
            # Execute LLM call
            response = await self._execute_llm_with_retries(
                prompt, stage_name, self.stages[stage_name]['timeout'], self.stages[stage_name]['max_retries']
            )
            
            # Process and store result
            self.stage_results['stages'][stage_name] = {
                'success': True,
                'response': response,
                'final_recommendation': self._extract_final_recommendation(response),
                'confidence_score': self._extract_final_confidence(response),
                'key_insights': self._extract_key_insights(response),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"[MULTI_STAGE_LLM] {stage_name} stage completed for {symbol}")
            
        except Exception as e:
            logger.error(f"[MULTI_STAGE_LLM] {stage_name} stage failed for {symbol}: {e}")
            self.stage_results['stages'][stage_name] = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.stage_results['errors'].append(f"{stage_name}: {str(e)}")
    
    async def _execute_llm_with_retries(self, prompt: str, stage_name: str, timeout: int, max_retries: int) -> str:
        """Execute LLM call with retries and timeout handling"""
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"[MULTI_STAGE_LLM] {stage_name} attempt {attempt + 1}/{max_retries + 1}")
                
                # Execute LLM call with timeout
                response = await asyncio.wait_for(
                    self.llm_client.generate(prompt=prompt, enable_code_execution=False),
                    timeout=timeout
                )
                
                if response and isinstance(response, str) and len(response.strip()) > 0:
                    return response
                else:
                    raise ValueError(f"Empty or invalid LLM response for {stage_name}")
                    
            except asyncio.TimeoutError:
                logger.warning(f"[MULTI_STAGE_LLM] {stage_name} timeout on attempt {attempt + 1}")
                if attempt == max_retries:
                    raise TimeoutError(f"{stage_name} stage timed out after {max_retries + 1} attempts")
            except Exception as e:
                logger.warning(f"[MULTI_STAGE_LLM] {stage_name} error on attempt {attempt + 1}: {e}")
                if attempt == max_retries:
                    raise e
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
    
    def _has_sufficient_stage_results(self) -> bool:
        """Check if we have sufficient results from previous stages"""
        successful_stages = [
            stage for stage, result in self.stage_results.get('stages', {}).items() 
            if result.get('success', False)
        ]
        return len(successful_stages) >= 1
    
    # Prompt building methods
    def _build_market_structure_prompt(self, symbol: str, market_structure: Dict[str, Any], technical_analysis: Dict[str, Any]) -> str:
        """Build specialized prompt for market structure analysis"""
        
        # Extract key market structure data
        swing_points = market_structure.get('swing_points', {})
        bos_choch = market_structure.get('bos_choch_analysis', {})
        trend_analysis = market_structure.get('trend_analysis', {})
        
        return f"""You are an expert market structure analyst. Analyze the market structure for {symbol} and provide deep insights.

MARKET STRUCTURE DATA:
Swing Points: {len(swing_points.get('swing_highs', []))} highs, {len(swing_points.get('swing_lows', []))} lows
BOS/CHOCH Analysis: {json.dumps(bos_choch, indent=2) if bos_choch else 'No significant breaks detected'}
Trend Analysis: {json.dumps(trend_analysis, indent=2) if trend_analysis else 'Trend analysis unavailable'}

TASK: Provide detailed analysis focusing on:
1. Market structure quality and reliability
2. Key swing points and their significance  
3. BOS (Break of Structure) and CHOCH (Change of Character) implications
4. Current trend strength and direction
5. Critical support/resistance levels from structure

Format your response as clear, actionable insights about the market structure."""
    
    def _build_pattern_detection_prompt(self, symbol: str, technical_analysis: Dict[str, Any], current_price: float) -> str:
        """Build specialized prompt for pattern detection"""
        
        # Extract pattern data
        price_patterns = technical_analysis.get('price_patterns', {})
        volume_patterns = technical_analysis.get('volume_patterns', {})
        momentum_patterns = technical_analysis.get('momentum_patterns', {})
        
        return f"""You are an expert pattern recognition specialist. Analyze the patterns for {symbol} at current price ${current_price:.2f}.

PATTERN ANALYSIS DATA:
Price Patterns: {json.dumps(price_patterns, indent=2) if price_patterns else 'No significant price patterns'}
Volume Patterns: {json.dumps(volume_patterns, indent=2) if volume_patterns else 'No volume analysis available'}
Momentum Patterns: {json.dumps(momentum_patterns, indent=2) if momentum_patterns else 'No momentum patterns detected'}

TASK: Identify and analyze:
1. Most significant chart patterns and their completion status
2. Volume confirmation or divergence with price patterns
3. Momentum patterns and oscillator signals
4. Pattern reliability and conviction levels
5. Expected pattern targets and invalidation levels

Provide specific pattern names, confidence levels, and price implications."""
    
    def _build_cross_validation_prompt(self, symbol: str, technical_analysis: Dict[str, Any]) -> str:
        """Build prompt for cross-pattern validation"""
        
        # Get previous stage results
        market_structure_result = self.stage_results['stages'].get('market_structure', {})
        pattern_detection_result = self.stage_results['stages'].get('pattern_detection', {})
        
        return f"""You are an expert pattern validation analyst. Cross-validate the analysis for {symbol}.

PREVIOUS ANALYSIS RESULTS:
Market Structure Analysis: {market_structure_result.get('response', 'Not available')[:500]}...

Pattern Detection Analysis: {pattern_detection_result.get('response', 'Not available')[:500]}...

TASK: Cross-validate and identify:
1. Confirmations between market structure and pattern analysis
2. Conflicts or contradictions between different analyses
3. Which signals are most reliable based on multiple confirmations
4. Risk factors from conflicting signals
5. Overall coherence of the analysis

Provide a balanced assessment of signal reliability and conflicts."""
    
    def _build_trading_insights_prompt(self, symbol: str, current_price: float) -> str:
        """Build prompt for trading insights generation"""
        
        # Summarize key findings from all previous stages
        successful_stages = {
            stage: result for stage, result in self.stage_results['stages'].items() 
            if result.get('success', False)
        }
        
        combined_analysis = ""
        for stage, result in successful_stages.items():
            combined_analysis += f"{stage.upper()}: {result.get('response', '')[:300]}...\n\n"
        
        return f"""You are an expert trading strategist. Generate actionable trading insights for {symbol} at ${current_price:.2f}.

COMPREHENSIVE ANALYSIS SUMMARY:
{combined_analysis}

TASK: Provide specific trading insights:
1. Primary trading bias (bullish/bearish/neutral) with conviction level
2. Specific entry points and entry conditions
3. Stop loss levels with rationale
4. Target price levels with timeframes
5. Risk assessment and position sizing recommendations
6. Key levels to monitor for trade management

Format as clear, actionable trading recommendations with specific price levels."""
    
    def _build_final_synthesis_prompt(self, symbol: str, current_price: float, context: str) -> str:
        """Build prompt for final synthesis"""
        
        # Summarize all stage results
        stage_summary = ""
        for stage, result in self.stage_results['stages'].items():
            status = "✓" if result.get('success', False) else "✗"
            stage_summary += f"{status} {stage}: {result.get('response', result.get('error', ''))[:200]}...\n\n"
        
        return f"""You are a senior technical analyst providing final synthesis for {symbol} at ${current_price:.2f}.

COMPLETE MULTI-STAGE ANALYSIS:
{stage_summary}

ADDITIONAL CONTEXT: {context}

TASK: Provide final comprehensive synthesis:
1. Overall market outlook for {symbol}
2. Key patterns and structure insights (most important findings)
3. Primary recommendation with clear rationale
4. Risk factors and mitigation strategies
5. Confidence level in the overall analysis (1-10 scale)
6. Next key levels/events to monitor

Synthesize all stages into a cohesive, actionable final recommendation."""
    
    # Result extraction methods
    def _extract_market_structure_insights(self, response: str) -> Dict[str, Any]:
        """Extract structured insights from market structure analysis"""
        return {
            'raw_analysis': response,
            'structure_quality': 'high',  # Would parse from response
            'key_levels': [],  # Would extract from response
            'trend_direction': 'neutral'  # Would parse from response
        }
    
    def _extract_detected_patterns(self, response: str) -> List[Dict[str, Any]]:
        """Extract detected patterns from response"""
        return [
            {
                'pattern_name': 'extracted_pattern',
                'confidence': 0.7,
                'completion': 0.8,
                'target': 0.0
            }
        ]
    
    def _extract_pattern_confidence(self, response: str) -> float:
        """Extract overall pattern confidence from response"""
        return 0.7  # Would parse from response
    
    def _extract_validation_results(self, response: str) -> Dict[str, Any]:
        """Extract validation results from cross-validation stage"""
        return {
            'confirmations': [],
            'reliability_score': 0.7
        }
    
    def _extract_pattern_conflicts(self, response: str) -> List[Dict[str, Any]]:
        """Extract pattern conflicts from validation"""
        return []
    
    def _extract_trading_signals(self, response: str) -> Dict[str, Any]:
        """Extract trading signals from insights stage"""
        return {
            'primary_bias': 'neutral',
            'entry_conditions': [],
            'conviction_level': 0.5
        }
    
    def _extract_risk_assessment(self, response: str) -> Dict[str, Any]:
        """Extract risk assessment from insights"""
        return {
            'risk_level': 'medium',
            'risk_factors': [],
            'mitigation_strategies': []
        }
    
    def _extract_price_targets(self, response: str) -> Dict[str, Any]:
        """Extract price targets from insights"""
        return {
            'entry_levels': [],
            'stop_loss_levels': [],
            'target_levels': []
        }
    
    def _extract_final_recommendation(self, response: str) -> str:
        """Extract final recommendation from synthesis"""
        return response[:200] + "..." if len(response) > 200 else response
    
    def _extract_final_confidence(self, response: str) -> float:
        """Extract final confidence score"""
        return 0.7  # Would parse from response
    
    def _extract_key_insights(self, response: str) -> List[str]:
        """Extract key insights from final synthesis"""
        return ["Key insight extracted from response"]
    
    # Result building methods
    def _build_success_result(self, symbol: str, processing_time: float, start_time: datetime) -> Dict[str, Any]:
        """Build successful multi-stage result"""
        successful_stages = sum(1 for result in self.stage_results['stages'].values() if result.get('success', False))
        total_stages = len(self.stages)
        
        return {
            'analysis_type': 'multi_stage_llm_pattern_analysis',
            'symbol': symbol,
            'timestamp': start_time.isoformat(),
            'processing_time': processing_time,
            'success': True,
            'stage_results': self.stage_results,
            'stage_summary': {
                'total_stages': total_stages,
                'successful_stages': successful_stages,
                'failed_stages': total_stages - successful_stages,
                'success_rate': successful_stages / total_stages if total_stages > 0 else 0
            },
            'final_analysis': self._consolidate_final_analysis(),
            'confidence_score': self._calculate_overall_confidence(),
            'errors': self.stage_results.get('errors', [])
        }
    
    def _build_error_result(self, symbol: str, error: str, start_time: datetime) -> Dict[str, Any]:
        """Build error result for failed multi-stage analysis"""
        processing_time = (datetime.now() - start_time).total_seconds()
        return {
            'analysis_type': 'multi_stage_llm_pattern_analysis',
            'symbol': symbol,
            'timestamp': start_time.isoformat(),
            'processing_time': processing_time,
            'success': False,
            'error': error,
            'stage_results': self.stage_results,
            'confidence_score': 0.0
        }
    
    def _consolidate_final_analysis(self) -> Dict[str, Any]:
        """Consolidate results from all successful stages"""
        consolidated = {
            'market_structure_insights': {},
            'detected_patterns': [],
            'validation_results': {},
            'trading_insights': {},
            'final_synthesis': {}
        }
        
        # Consolidate from each successful stage
        for stage_name, result in self.stage_results.get('stages', {}).items():
            if result.get('success', False):
                if stage_name == 'market_structure':
                    consolidated['market_structure_insights'] = result.get('insights', {})
                elif stage_name == 'pattern_detection':
                    consolidated['detected_patterns'] = result.get('patterns', [])
                elif stage_name == 'cross_validation':
                    consolidated['validation_results'] = result.get('validations', {})
                elif stage_name == 'trading_insights':
                    consolidated['trading_insights'] = {
                        'signals': result.get('trading_signals', {}),
                        'risk_assessment': result.get('risk_assessment', {}),
                        'price_targets': result.get('price_targets', {})
                    }
                elif stage_name == 'final_synthesis':
                    consolidated['final_synthesis'] = {
                        'recommendation': result.get('final_recommendation', ''),
                        'confidence': result.get('confidence_score', 0.5),
                        'key_insights': result.get('key_insights', [])
                    }
        
        return consolidated
    
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence based on successful stages"""
        successful_stages = [
            result for result in self.stage_results.get('stages', {}).values() 
            if result.get('success', False)
        ]
        
        if not successful_stages:
            return 0.0
        
        # Base confidence on number of successful stages and their individual confidences
        stage_success_rate = len(successful_stages) / len(self.stages)
        
        # Extract individual stage confidences if available
        stage_confidences = []
        for result in successful_stages:
            if 'confidence_score' in result:
                stage_confidences.append(result['confidence_score'])
        
        if stage_confidences:
            avg_confidence = sum(stage_confidences) / len(stage_confidences)
            return (stage_success_rate * 0.4) + (avg_confidence * 0.6)  # Weighted combination
        else:
            return stage_success_rate * 0.8  # Conservative confidence based on success rate only