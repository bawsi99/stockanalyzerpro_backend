#!/usr/bin/env python3
"""
Institutional Activity Agent - Master Agent Module

Coordinates technical analysis, chart generation, and LLM-powered insights
for institutional trading activity detection.
"""

import pandas as pd
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .processor import InstitutionalActivityProcessor
from .charts import InstitutionalActivityChartGenerator
from .llm_agent import InstitutionalActivityLLMAgent

logger = logging.getLogger(__name__)

class InstitutionalActivityAgent:
    """
    Master Institutional Activity Agent
    
    Coordinates all components: technical analysis, chart generation, and LLM analysis
    for comprehensive institutional activity detection and insights.
    """
    
    def __init__(self):
        self.agent_name = "institutional_activity"
        self.agent_version = "2.0.0"
        self.description = "Detects institutional trading activity through volume and pattern analysis"
        
        # Initialize components
        self.processor = InstitutionalActivityProcessor()
        self.chart_generator = InstitutionalActivityChartGenerator()
        self.llm_agent = InstitutionalActivityLLMAgent()
        
        # Agent capabilities
        self.capabilities = {
            "large_block_detection": True,
            "institutional_pattern_analysis": True,
            "volume_profile_analysis": True,
            "smart_money_flow_detection": True,
            "llm_enhanced_insights": True,
            "multi_modal_analysis": True
        }
        
        logger.info(f"Institutional Activity Agent v{self.agent_version} initialized")
    
    async def analyze_complete(self, 
                              stock_data: pd.DataFrame, 
                              symbol: str,
                              context: str = "") -> Dict[str, Any]:
        """
        Complete institutional activity analysis pipeline.
        
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
            technical_analysis = self.processor.process_institutional_activity_data(stock_data)
            
            if 'error' in technical_analysis:
                return self._format_error_result(technical_analysis['error'], symbol)
            
            logger.info(f"Technical analysis completed for {symbol}")
            
            # Step 2: Chart Generation  
            logger.info(f"Generating chart for {symbol}")
            chart_image = None
            try:
                chart_image = await asyncio.to_thread(
                    self.chart_generator.generate_institutional_activity_chart,
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
            logger.error(f"Institutional activity analysis failed for {symbol}: {e}")
            return self._format_error_result(str(e), symbol, processing_time)
    
    def get_technical_analysis_only(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get only technical analysis without LLM processing.
        
        Useful for quick analysis or when LLM is not needed.
        """
        try:
            return self.processor.process_institutional_activity_data(stock_data)
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
                self.chart_generator.generate_institutional_activity_chart,
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
        
        # Activity level insight
        activity_level = technical.get('institutional_activity_level', 'unknown')
        if activity_level != 'unknown':
            insights.append(f"Institutional activity level: {activity_level.replace('_', ' ').title()}")
        
        # Primary activity type
        primary_activity = technical.get('primary_activity', 'unknown')
        if primary_activity != 'unknown':
            insights.append(f"Primary activity type: {primary_activity.replace('_', ' ').title()}")
        
        # Large block analysis
        large_blocks = technical.get('large_block_analysis', {})
        total_blocks = large_blocks.get('total_large_blocks', 0)
        if total_blocks > 0:
            insights.append(f"Detected {total_blocks} large block transactions")
        
        # Volume profile insight
        volume_profile = technical.get('volume_profile', {})
        if volume_profile and 'error' not in volume_profile:
            highest_vol = volume_profile.get('highest_volume_level', {})
            if highest_vol:
                price_level = highest_vol.get('price_level', 0)
                if price_level > 0:
                    insights.append(f"Highest volume node at ${price_level:.2f}")
        
        # LLM insights
        if analysis_result.get('has_llm_analysis', False):
            insights.append("Enhanced with AI-powered pattern analysis")
        
        return insights
    
    def _calculate_confidence_score(self, 
                                   technical_analysis: Dict[str, Any], 
                                   llm_analysis: Optional[str]) -> int:
        """Calculate confidence score based on analysis quality."""
        base_score = 50
        
        # Technical analysis quality
        if 'error' not in technical_analysis:
            base_score += 20
            
            # Activity level detected
            if technical_analysis.get('institutional_activity_level') != 'unknown':
                base_score += 15
            
            # Large blocks detected
            large_blocks = technical_analysis.get('large_block_analysis', {})
            if large_blocks.get('total_large_blocks', 0) > 0:
                base_score += 10
            
            # Volume profile quality
            volume_profile = technical_analysis.get('volume_profile', {})
            if volume_profile and 'error' not in volume_profile:
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
                'processor': 'InstitutionalActivityProcessor',
                'chart_generator': 'InstitutionalActivityChartGenerator', 
                'llm_agent': 'InstitutionalActivityLLMAgent'
            },
            'llm_framework': 'backend/llm',
            'migration_status': 'fully_distributed'
        }