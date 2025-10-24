#!/usr/bin/env python3
"""
Volume Momentum Agent - Master Agent Module

Coordinates technical analysis, chart generation, and LLM-powered insights
for volume-based momentum analysis using distributed architecture.
"""

import pandas as pd
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .processor import VolumeTrendMomentumProcessor
from .charts import VolumeTrendMomentumCharts
from .llm_agent import VolumeMomentumLLMAgent

logger = logging.getLogger(__name__)

class VolumeMomentumAgent:
    """
    Master Volume Momentum Agent
    
    Coordinates all components: technical analysis, chart generation, and LLM analysis
    for comprehensive volume momentum and trend analysis.
    """
    
    def __init__(self):
        self.agent_name = "volume_momentum"
        self.agent_version = "2.0.0"
        self.description = "Analyzes volume momentum and trend sustainability"
        
        # Initialize components
        self.processor = VolumeTrendMomentumProcessor()
        self.chart_generator = VolumeTrendMomentumCharts()
        self.llm_agent = VolumeMomentumLLMAgent()
        
        # Agent capabilities
        self.capabilities = {
            "volume_trend_analysis": True,
            "momentum_acceleration_detection": True,
            "volume_momentum_indicators": True,
            "trend_sustainability_analysis": True,
            "llm_enhanced_insights": True,
            "multi_modal_analysis": True
        }
        
        logger.info(f"Volume Momentum Agent v{self.agent_version} initialized")

    async def analyze_complete(self, 
                              stock_data: pd.DataFrame, 
                              symbol: str,
                              context: str = "") -> Dict[str, Any]:
        """
        Complete volume momentum analysis pipeline.
        
        This is the main method that orchestrates:
        1. Technical analysis
        2. Chart generation  
        3. LLM analysis
        
        Args:
            stock_data: DataFrame with OHLCV data
            symbol: Stock symbol
            context: Additional context for analysis
            
        Returns:
            Complete analysis results with LLM insights
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Technical Analysis
            logger.info(f"Starting technical analysis for {symbol}")
            technical_analysis = self.processor.process_volume_trend_momentum_data(stock_data)
            
            if 'error' in technical_analysis:
                return self._format_error_result(technical_analysis['error'], symbol)
            
            logger.info(f"Technical analysis completed for {symbol}")
            
            # Step 2: Chart Generation  
            logger.info(f"Generating chart for {symbol}")
            chart_image = None
            try:
                chart_image = await asyncio.to_thread(
                    self.chart_generator.generate_volume_momentum_chart,
                    stock_data, technical_analysis, symbol
                )
                logger.info(f"Chart generated for {symbol}: {len(chart_image) if chart_image else 0} bytes")
            except Exception as chart_error:
                logger.warning(f"Chart generation failed for {symbol}: {chart_error}")
            
            # Step 3: LLM Analysis (if chart is available)
            llm_analysis = None
            if chart_image:
                try:
                    logger.info(f"Starting LLM analysis for {symbol}")
                    llm_analysis = await self.llm_agent.analyze_with_chart(
                        analysis_data=technical_analysis,
                        symbol=symbol,
                        chart_image=chart_image,
                        context=context
                    )
                    logger.info(f"LLM analysis completed for {symbol}")
                except Exception as llm_error:
                    logger.warning(f"LLM analysis failed for {symbol}: {llm_error}")
            else:
                logger.info(f"Skipping LLM analysis for {symbol}: no chart image")
            
            # Step 4: Combine Results
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'agent_name': self.agent_name,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'processing_time': processing_time,
                'technical_analysis': technical_analysis,
                'chart_image': chart_image,
                'llm_analysis': llm_analysis,
                'has_llm_analysis': llm_analysis is not None,
                'agent_info': {
                    'agent_name': self.agent_name,
                    'agent_version': self.agent_version,
                    'description': self.description,
                    'capabilities': list(self.capabilities.keys())
                },
                'confidence_score': self._calculate_confidence_score(technical_analysis, llm_analysis)
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Volume momentum analysis failed for {symbol}: {e}")
            return self._format_error_result(str(e), symbol, processing_time)
    
    def get_technical_analysis_only(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get only technical analysis without LLM processing.
        
        Useful for quick analysis or when LLM is not needed.
        """
        try:
            return self.processor.process_volume_trend_momentum_data(stock_data)
        except Exception as e:
            return {'error': f'Technical analysis failed: {str(e)}'}
    
    async def generate_chart_only(self, 
                                 stock_data: pd.DataFrame, 
                                 symbol: str,
                                 analysis_data: Optional[Dict[str, Any]] = None) -> Optional[bytes]:
        """
        Generate chart without full analysis.
        
        Args:
            stock_data: DataFrame with OHLCV data
            symbol: Stock symbol
            analysis_data: Pre-computed analysis data (optional)
            
        Returns:
            Chart image bytes or None if failed
        """
        try:
            if analysis_data is None:
                analysis_data = self.get_technical_analysis_only(stock_data)
                
            if 'error' in analysis_data:
                return None
                
            return await asyncio.to_thread(
                self.chart_generator.generate_volume_momentum_chart,
                stock_data, analysis_data, symbol
            )
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return None
    
    def get_key_insights(self, analysis_result: Dict[str, Any]) -> list:
        """
        Extract key insights from complete analysis result.
        
        Args:
            analysis_result: Result from analyze_complete()
            
        Returns:
            List of key insight strings
        """
        insights = []
        
        if not analysis_result.get('success', False):
            return [f"Analysis failed: {analysis_result.get('error', 'Unknown error')}"]
        
        technical = analysis_result.get('technical_analysis', {})
        
        # Volume trend insight
        volume_trend = technical.get('volume_trend', {})
        if volume_trend:
            trend_direction = volume_trend.get('trend_direction', 'unknown')
            trend_strength = volume_trend.get('trend_strength', 'unknown')
            if trend_direction != 'unknown':
                insights.append(f"Volume trend: {trend_direction.replace('_', ' ').title()} ({trend_strength} strength)")
        
        # Momentum signals
        momentum_signals = technical.get('momentum_signals', {})
        if momentum_signals:
            acceleration = momentum_signals.get('acceleration_signal', 'neutral')
            if acceleration != 'neutral':
                insights.append(f"Momentum {acceleration}: {momentum_signals.get('signal_strength', 'unknown')} strength")
        
        # Volume momentum indicators
        momentum_indicators = technical.get('volume_momentum_indicators', {})
        if momentum_indicators:
            current_momentum = momentum_indicators.get('current_momentum_score', 0)
            if current_momentum != 0:
                momentum_level = 'strong' if abs(current_momentum) > 70 else 'moderate' if abs(current_momentum) > 40 else 'weak'
                direction = 'bullish' if current_momentum > 0 else 'bearish'
                insights.append(f"Current momentum: {momentum_level} {direction} ({current_momentum:.1f})")
        
        # Trend continuation probability
        trend_continuation = technical.get('trend_continuation', {})
        if trend_continuation:
            probability = trend_continuation.get('continuation_probability', 0)
            if probability > 0:
                confidence_level = 'high' if probability > 70 else 'medium' if probability > 50 else 'low'
                insights.append(f"Trend continuation probability: {probability:.0f}% ({confidence_level} confidence)")
        
        # LLM insights
        if analysis_result.get('has_llm_analysis', False):
            insights.append("Enhanced with AI-powered momentum analysis")
        
        return insights
    
    def _calculate_confidence_score(self, 
                                   technical_analysis: Dict[str, Any], 
                                   llm_analysis: Optional[str]) -> int:
        """Calculate confidence score based on analysis quality."""
        base_score = 50
        
        # Technical analysis quality
        if 'error' not in technical_analysis:
            base_score += 20
            
            # Volume trend detected
            volume_trend = technical_analysis.get('volume_trend', {})
            if volume_trend.get('trend_direction') != 'unknown':
                base_score += 10
            
            # Momentum signals quality
            momentum_signals = technical_analysis.get('momentum_signals', {})
            if momentum_signals.get('signal_strength') in ['strong', 'very_strong']:
                base_score += 10
            
            # Momentum indicators availability
            momentum_indicators = technical_analysis.get('volume_momentum_indicators', {})
            if momentum_indicators.get('current_momentum_score', 0) != 0:
                base_score += 5
            
            # Trend continuation analysis
            trend_continuation = technical_analysis.get('trend_continuation', {})
            if trend_continuation.get('continuation_probability', 0) > 0:
                base_score += 5
        
        # LLM analysis bonus
        if llm_analysis:
            base_score += 10
        
        return min(base_score, 100)
    
    def _format_error_result(self, 
                            error_message: str, 
                            symbol: str, 
                            processing_time: float = 0.0) -> Dict[str, Any]:
        """Format error result in standard format."""
        return {
            'success': False,
            'agent_name': self.agent_name,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'error': error_message,
            'technical_analysis': {'error': error_message},
            'chart_image': None,
            'llm_analysis': None,
            'has_llm_analysis': False,
            'confidence_score': 0,
            'agent_info': {
                'agent_name': self.agent_name,
                'agent_version': self.agent_version
            }
        }

    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information and capabilities."""
        return {
            'agent_name': self.agent_name,
            'agent_version': self.agent_version,
            'description': self.description,
            'capabilities': self.capabilities,
            'components': {
                'processor': 'VolumeTrendMomentumProcessor',
                'chart_generator': 'VolumeTrendMomentumCharts', 
                'llm_agent': 'VolumeMomentumLLMAgent'
            },
            'llm_framework': 'backend/llm',
            'migration_status': 'fully_distributed'
        }
