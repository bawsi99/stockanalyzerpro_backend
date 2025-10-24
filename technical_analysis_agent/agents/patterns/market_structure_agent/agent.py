#!/usr/bin/env python3
"""
Market Structure Agent - Master Agent Module

Coordinates technical analysis, chart generation, and LLM-powered insights
for market structure analysis using distributed architecture.
"""

import pandas as pd
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .processor import MarketStructureProcessor
from .charts import MarketStructureCharts
from .llm_agent import MarketStructureLLMAgent

logger = logging.getLogger(__name__)

class MarketStructureAgent:
    """
    Master Market Structure Agent
    
    Coordinates all components: technical analysis, chart generation, and LLM analysis
    for comprehensive market structure analysis including swing points, BOS/CHOCH events,
    trend structure analysis, and trading insights.
    """
    
    def __init__(self):
        self.agent_name = "market_structure"
        self.agent_version = "1.0.0"
        self.description = "Analyzes market structure, swing points, BOS/CHOCH events, and trend structure"
        
        # Initialize components
        self.processor = MarketStructureProcessor()
        self.chart_generator = MarketStructureCharts()
        self.llm_agent = MarketStructureLLMAgent()
        
        # Agent capabilities
        self.capabilities = {
            "swing_point_analysis": True,
            "bos_choch_detection": True,
            "trend_structure_analysis": True,
            "support_resistance_from_structure": True,
            "fractal_analysis": True,
            "llm_enhanced_insights": True,
            "multi_modal_analysis": True
        }
        
        logger.info(f"Market Structure Agent v{self.agent_version} initialized")
    
    async def analyze_complete(self, 
                              stock_data: pd.DataFrame, 
                              symbol: str,
                              context: str = "") -> Dict[str, Any]:
        """
        Complete market structure analysis pipeline.
        
        This is the main method that orchestrates:
        1. Technical analysis (swing points, BOS/CHOCH, trend structure)
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
            logger.info(f"Starting market structure technical analysis for {symbol}")
            technical_analysis = self.processor.process_market_structure_data(stock_data)
            
            if 'error' in technical_analysis:
                return self._format_error_result(technical_analysis['error'], symbol)
            
            logger.info(f"Technical analysis completed for {symbol}")
            
            # Step 2: Chart Generation  
            logger.info(f"Generating market structure chart for {symbol}")
            chart_image = None
            try:
                chart_image = await asyncio.to_thread(
                    self.chart_generator.generate_market_structure_chart,
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
                    llm_analysis = await self.llm_agent.analyze_market_structure(
                        chart_image=chart_image,
                        analysis_data=technical_analysis,
                        symbol=symbol
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
            logger.error(f"Market structure analysis failed for {symbol}: {e}")
            return self._format_error_result(str(e), symbol, processing_time)
    
    def get_technical_analysis_only(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get only technical analysis without LLM processing.
        
        Useful for quick analysis or when LLM is not needed.
        """
        try:
            return self.processor.process_market_structure_data(stock_data)
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
                self.chart_generator.generate_market_structure_chart,
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
        
        # Structure quality insight
        structure_quality = technical.get('structure_quality', {})
        if structure_quality:
            quality_rating = structure_quality.get('quality_rating', 'unknown')
            quality_score = structure_quality.get('quality_score', 0)
            insights.append(f"Structure quality: {quality_rating} ({quality_score}/100)")
        
        # Swing points insight
        swing_points = technical.get('swing_points', {})
        if swing_points:
            total_swings = swing_points.get('total_swings', 0)
            swing_density = swing_points.get('swing_density', 0)
            insights.append(f"Identified {total_swings} swing points (density: {swing_density:.3f})")
        
        # Trend insight
        trend_analysis = technical.get('trend_analysis', {})
        if trend_analysis:
            trend_direction = trend_analysis.get('trend_direction', 'unknown')
            trend_strength = trend_analysis.get('trend_strength', 'unknown') 
            trend_quality = trend_analysis.get('trend_quality', 'unknown')
            insights.append(f"Trend: {trend_direction} ({trend_strength} strength, {trend_quality} quality)")
        
        # BOS/CHOCH insight
        bos_choch = technical.get('bos_choch_analysis', {})
        if bos_choch:
            structural_bias = bos_choch.get('structural_bias', 'unknown')
            total_bos = bos_choch.get('total_bos_events', 0)
            total_choch = bos_choch.get('total_choch_events', 0)
            insights.append(f"Structural bias: {structural_bias} ({total_bos} BOS, {total_choch} CHoCH events)")
        
        # Current state insight
        current_state = technical.get('current_state', {})
        if current_state:
            structure_state = current_state.get('structure_state', 'unknown').replace('_', ' ')
            price_position = current_state.get('price_position_description', 'unknown').replace('_', ' ')
            insights.append(f"Current state: {structure_state}, price position: {price_position}")
        
        # Key levels insight
        key_levels = technical.get('key_levels', {})
        if key_levels:
            support_count = len(key_levels.get('support_levels', []))
            resistance_count = len(key_levels.get('resistance_levels', []))
            insights.append(f"Key levels: {support_count} support, {resistance_count} resistance")
        
        # Fractal analysis insight
        fractal_analysis = technical.get('fractal_analysis', {})
        if fractal_analysis:
            alignment = fractal_analysis.get('timeframe_alignment', 'unknown')
            consensus = fractal_analysis.get('trend_consensus', 'unknown')
            insights.append(f"Multi-timeframe: {alignment} alignment, {consensus} consensus")
        
        # LLM insights
        if analysis_result.get('has_llm_analysis', False):
            insights.append("Enhanced with AI-powered structural analysis")
        
        return insights
    
    def _calculate_confidence_score(self, 
                                   technical_analysis: Dict[str, Any], 
                                   llm_analysis: Optional[str]) -> int:
        """Calculate confidence score based on analysis quality."""
        base_score = 50
        
        # Technical analysis quality
        if 'error' not in technical_analysis:
            base_score += 20
            
            # Structure quality assessment
            structure_quality = technical_analysis.get('structure_quality', {})
            if structure_quality:
                quality_score = structure_quality.get('quality_score', 50)
                base_score += int(quality_score * 0.2)  # Up to 20 points
            
            # Swing point analysis quality
            swing_points = technical_analysis.get('swing_points', {})
            total_swings = swing_points.get('total_swings', 0)
            if total_swings >= 6:
                base_score += 10
            elif total_swings >= 4:
                base_score += 5
            
            # BOS/CHOCH events quality
            bos_choch = technical_analysis.get('bos_choch_analysis', {})
            if bos_choch:
                total_events = bos_choch.get('total_bos_events', 0) + bos_choch.get('total_choch_events', 0)
                if total_events > 0:
                    base_score += min(total_events * 3, 10)  # Up to 10 points for structural events
        
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
                'processor': 'MarketStructureProcessor',
                'chart_generator': 'MarketStructureCharts', 
                'llm_agent': 'MarketStructureLLMAgent'
            },
            'llm_framework': 'backend/llm',
            'migration_status': 'fully_distributed'
        }