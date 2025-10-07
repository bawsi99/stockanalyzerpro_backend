#!/usr/bin/env python3
"""
Volume Anomaly Agent - Master Agent Module

Coordinates technical analysis, chart generation, and LLM-powered insights
for volume anomaly detection using distributed architecture.
"""

import pandas as pd
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .processor import VolumeAnomalyProcessor
from .charts import VolumeAnomalyCharts
from .llm_agent import VolumeAnomalyLLMAgent

logger = logging.getLogger(__name__)

class VolumeAnomalyAgent:
    """
    Master Volume Anomaly Agent
    
    Coordinates all components: technical analysis, chart generation, and LLM analysis
    for comprehensive volume anomaly detection and classification.
    """
    
    def __init__(self):
        self.agent_name = "volume_anomaly"
        self.agent_version = "2.0.0"
        self.description = "Detects and classifies statistical volume anomalies and outliers"
        
        # Initialize components
        self.processor = VolumeAnomalyProcessor()
        self.chart_generator = VolumeAnomalyCharts()
        self.llm_agent = VolumeAnomalyLLMAgent()
        
        # Agent capabilities
        self.capabilities = {
            "statistical_outlier_detection": True,
            "volume_spike_classification": True,
            "anomaly_pattern_analysis": True,
            "volume_distribution_analysis": True,
            "llm_enhanced_insights": True,
            "multi_modal_analysis": True
        }
        
        logger.info(f"Volume Anomaly Agent v{self.agent_version} initialized")
    
    async def analyze_complete(self, 
                              stock_data: pd.DataFrame, 
                              symbol: str,
                              context: str = "") -> Dict[str, Any]:
        """
        Complete volume anomaly analysis pipeline.
        
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
            technical_analysis = self.processor.process_volume_anomaly_data(stock_data)
            
            if 'error' in technical_analysis:
                return self._format_error_result(technical_analysis['error'], symbol)
            
            logger.info(f"Technical analysis completed for {symbol}")
            
            # Step 2: Chart Generation  
            logger.info(f"Generating chart for {symbol}")
            chart_image = None
            try:
                chart_image = await asyncio.to_thread(
                    self.chart_generator.generate_volume_anomaly_chart,
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
                    llm_analysis = await self.llm_agent.analyze_volume_anomaly(
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
            logger.error(f"Volume anomaly analysis failed for {symbol}: {e}")
            return self._format_error_result(str(e), symbol, processing_time)
    
    def get_technical_analysis_only(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get only technical analysis without LLM processing.
        
        Useful for quick analysis or when LLM is not needed.
        """
        try:
            return self.processor.process_volume_anomaly_data(stock_data)
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
                self.chart_generator.generate_volume_anomaly_chart,
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
        
        # Anomaly count insight
        anomalies = technical.get('significant_anomalies', [])
        if len(anomalies) > 0:
            insights.append(f"Detected {len(anomalies)} significant volume anomalies")
            
            # Categorize anomalies
            extreme = [a for a in anomalies if a.get('anomaly_type') == 'extreme_outlier']
            if extreme:
                insights.append(f"{len(extreme)} extreme volume outliers detected")
        else:
            insights.append("No significant volume anomalies detected")
        
        # Current volume status
        current_status = technical.get('current_volume_status', {})
        if current_status:
            status = current_status.get('current_status', 'unknown').replace('_', ' ').title()
            percentile = current_status.get('volume_percentile', 0)
            if percentile > 0:
                insights.append(f"Current volume status: {status} ({percentile}th percentile)")
        
        # Volume statistics insight
        volume_stats = technical.get('volume_statistics', {})
        if volume_stats:
            cv = volume_stats.get('volume_cv', 0)
            if cv > 0:
                volatility = 'high' if cv > 0.5 else 'moderate' if cv > 0.3 else 'low'
                insights.append(f"Volume volatility: {volatility} (CV: {cv:.2f})")
        
        # Anomaly patterns
        anomaly_patterns = technical.get('anomaly_patterns', {})
        if anomaly_patterns:
            frequency = anomaly_patterns.get('anomaly_frequency', 'unknown')
            pattern = anomaly_patterns.get('anomaly_pattern', 'unknown')
            if frequency != 'unknown':
                insights.append(f"Anomaly frequency: {frequency}, pattern: {pattern}")
        
        # LLM insights
        if analysis_result.get('has_llm_analysis', False):
            insights.append("Enhanced with AI-powered anomaly classification")
        
        return insights
    
    def _calculate_confidence_score(self, 
                                   technical_analysis: Dict[str, Any], 
                                   llm_analysis: Optional[str]) -> int:
        """Calculate confidence score based on analysis quality."""
        base_score = 50
        
        # Technical analysis quality
        if 'error' not in technical_analysis:
            base_score += 20
            
            # Quality assessment score
            quality = technical_analysis.get('quality_assessment', {})
            if quality:
                overall_score = quality.get('overall_score', 50)
                base_score += int(overall_score * 0.2)  # Up to 20 points
            
            # Anomaly detection quality
            anomalies = technical_analysis.get('significant_anomalies', [])
            if len(anomalies) > 0:
                base_score += min(len(anomalies) * 2, 10)  # Up to 10 points for anomalies
        
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
                'processor': 'VolumeAnomalyProcessor',
                'chart_generator': 'VolumeAnomalyCharts', 
                'llm_agent': 'VolumeAnomalyLLMAgent'
            },
            'llm_framework': 'backend/llm',
            'migration_status': 'fully_distributed'
        }