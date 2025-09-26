#!/usr/bin/env python3
"""
Volume-Based Support/Resistance Agent - Main Agent Module

Agent wrapper that integrates the Support/Resistance Processor with the agent system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .processor import SupportResistanceProcessor

class SupportResistanceAgent:
    """
    Volume-Based Support/Resistance Agent
    
    Identifies key price levels backed by volume analysis and historical testing
    """
    
    def __init__(self):
        self.agent_name = "Support/Resistance Agent"
        self.agent_version = "1.0.0"
        self.description = "Identifies volume-validated support and resistance levels"
        self.processor = SupportResistanceProcessor()
        
        # Agent capabilities
        self.capabilities = {
            "volume_at_price_analysis": True,
            "swing_level_detection": True,
            "level_validation": True,
            "strength_rating": True,
            "position_analysis": True,
            "trading_implications": True
        }
        
        # Minimum data requirements
        self.min_data_points = 90  # 90 days minimum for good level validation
        self.preferred_data_points = 180  # 6 months preferred
    
    def analyze(self, data: pd.DataFrame, symbol: str = None, **kwargs) -> Dict[str, Any]:
        """
        Main analysis method for support/resistance analysis
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol (optional)
            **kwargs: Additional parameters
            
        Returns:
            Dict containing comprehensive support/resistance analysis
        """
        try:
            # Validate input data
            validation_result = self._validate_input_data(data)
            if validation_result['status'] == 'error':
                return self._format_error_response(validation_result['message'])
            
            # Add metadata
            analysis_metadata = {
                'agent_name': self.agent_name,
                'agent_version': self.agent_version,
                'symbol': symbol or 'UNKNOWN',
                'analysis_timestamp': datetime.now().isoformat(),
                'data_period': {
                    'start_date': data.index[0].isoformat() if len(data) > 0 else None,
                    'end_date': data.index[-1].isoformat() if len(data) > 0 else None,
                    'total_days': len(data)
                },
                'data_quality': validation_result.get('quality', 'unknown')
            }
            
            # Process the data
            analysis_results = self.processor.process_support_resistance_data(data)
            
            if 'error' in analysis_results:
                return self._format_error_response(analysis_results['error'], analysis_metadata)
            
            # Format results for agent system
            formatted_results = self._format_analysis_results(analysis_results, analysis_metadata)
            
            # Add agent-specific context and insights
            formatted_results['agent_insights'] = self._generate_agent_insights(analysis_results, data)
            
            # Add actionable recommendations
            formatted_results['recommendations'] = self._generate_recommendations(analysis_results, data)
            
            return formatted_results
            
        except Exception as e:
            return self._format_error_response(f"Analysis failed: {str(e)}")
    
    def get_key_levels(self, data: pd.DataFrame, level_type: str = 'both') -> List[Dict[str, Any]]:
        """
        Get key support/resistance levels only
        
        Args:
            data: DataFrame with OHLCV data
            level_type: 'support', 'resistance', or 'both'
            
        Returns:
            List of key levels
        """
        try:
            analysis_results = self.processor.process_support_resistance_data(data)
            
            if 'error' in analysis_results:
                return []
            
            validated_levels = analysis_results.get('validated_levels', [])
            
            if level_type == 'support':
                return [level for level in validated_levels if level['type'] in ['support', 'both']]
            elif level_type == 'resistance':
                return [level for level in validated_levels if level['type'] in ['resistance', 'both']]
            else:
                return validated_levels
                
        except Exception as e:
            return []
    
    def get_current_position_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get current position analysis relative to key levels
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dict with current position analysis
        """
        try:
            analysis_results = self.processor.process_support_resistance_data(data)
            
            if 'error' in analysis_results:
                return {'error': analysis_results['error']}
            
            return analysis_results.get('current_position_analysis', {})
            
        except Exception as e:
            return {'error': f"Position analysis failed: {str(e)}"}
    
    def get_trading_implications(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get trading implications based on support/resistance analysis
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dict with trading implications
        """
        try:
            analysis_results = self.processor.process_support_resistance_data(data)
            
            if 'error' in analysis_results:
                return {'error': analysis_results['error']}
            
            return analysis_results.get('trading_implications', {})
            
        except Exception as e:
            return {'error': f"Trading implications analysis failed: {str(e)}"}
    
    def _validate_input_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate input data quality and completeness"""
        try:
            if data is None or data.empty:
                return {'status': 'error', 'message': 'No data provided'}
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                return {
                    'status': 'error', 
                    'message': f'Missing required columns: {missing_columns}'
                }
            
            # Check data length
            data_length = len(data)
            if data_length < self.min_data_points:
                return {
                    'status': 'error',
                    'message': f'Insufficient data: {data_length} days (minimum {self.min_data_points} required)'
                }
            
            # Assess data quality
            quality = self._assess_data_quality(data)
            
            return {
                'status': 'success',
                'quality': quality,
                'data_points': data_length,
                'quality_score': self._calculate_quality_score(data, quality)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Data validation failed: {str(e)}'}
    
    def _assess_data_quality(self, data: pd.DataFrame) -> str:
        """Assess overall data quality"""
        try:
            # Check for missing values
            missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
            
            # Check for zero volume days
            zero_volume_pct = (data['volume'] == 0).sum() / len(data) * 100
            
            # Check for unrealistic price movements (>50% daily change)
            price_changes = data['close'].pct_change().abs()
            extreme_moves_pct = (price_changes > 0.5).sum() / len(data) * 100
            
            # Determine quality level
            quality_issues = 0
            if missing_pct > 5:
                quality_issues += 1
            if zero_volume_pct > 10:
                quality_issues += 1
            if extreme_moves_pct > 1:
                quality_issues += 1
            
            if quality_issues == 0:
                return 'high'
            elif quality_issues == 1:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'unknown'
    
    def _calculate_quality_score(self, data: pd.DataFrame, quality: str) -> int:
        """Calculate numerical quality score"""
        base_scores = {'high': 90, 'medium': 70, 'low': 50, 'unknown': 30}
        base_score = base_scores.get(quality, 30)
        
        # Bonus for sufficient data length
        if len(data) >= self.preferred_data_points:
            base_score += 10
        elif len(data) >= self.min_data_points * 1.5:
            base_score += 5
        
        return min(base_score, 100)
    
    def _format_analysis_results(self, analysis_results: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Format analysis results for agent system"""
        
        # Core analysis data
        formatted_results = {
            'metadata': metadata,
            'analysis_summary': self._create_analysis_summary(analysis_results),
            'key_findings': self._extract_key_findings(analysis_results),
            'support_levels': analysis_results.get('volume_based_support_levels', []),
            'resistance_levels': analysis_results.get('volume_based_resistance_levels', []),
            'current_position': analysis_results.get('current_position_analysis', {}),
            'trading_implications': analysis_results.get('trading_implications', {}),
            'quality_assessment': analysis_results.get('quality_assessment', {}),
            'detailed_analysis': {
                'volume_profile': analysis_results.get('volume_at_price_analysis', {}),
                'level_ratings': analysis_results.get('level_ratings', {}),
                'validated_levels': analysis_results.get('validated_levels', [])
            }
        }
        
        return formatted_results
    
    def _create_analysis_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create high-level analysis summary"""
        try:
            validated_levels = analysis_results.get('validated_levels', [])
            current_analysis = analysis_results.get('current_position_analysis', {})
            quality = analysis_results.get('quality_assessment', {})
            
            # Count levels by type
            support_count = len([l for l in validated_levels if l['type'] in ['support', 'both']])
            resistance_count = len([l for l in validated_levels if l['type'] in ['resistance', 'both']])
            
            # Get strongest levels
            level_ratings = analysis_results.get('level_ratings', {})
            strongest_support = level_ratings.get('strongest_support')
            strongest_resistance = level_ratings.get('strongest_resistance')
            
            return {
                'total_validated_levels': len(validated_levels),
                'support_levels_found': support_count,
                'resistance_levels_found': resistance_count,
                'analysis_quality_score': quality.get('overall_score', 0),
                'quality_rating': quality.get('reliability_rating', 'unknown'),
                'current_price': current_analysis.get('current_price'),
                'range_position': current_analysis.get('range_position_classification', 'unknown'),
                'strongest_support_price': strongest_support['price'] if strongest_support else None,
                'strongest_resistance_price': strongest_resistance['price'] if strongest_resistance else None,
                'average_level_strength': level_ratings.get('average_strength', 0)
            }
            
        except Exception as e:
            return {'error': f'Summary creation failed: {str(e)}'}
    
    def _extract_key_findings(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis"""
        findings = []
        
        try:
            validated_levels = analysis_results.get('validated_levels', [])
            current_analysis = analysis_results.get('current_position_analysis', {})
            quality = analysis_results.get('quality_assessment', {})
            
            # Quality finding
            quality_score = quality.get('overall_score', 0)
            if quality_score >= 80:
                findings.append(f"High-quality analysis with {quality_score}/100 confidence score")
            elif quality_score >= 60:
                findings.append(f"Good analysis quality with {quality_score}/100 confidence score")
            else:
                findings.append(f"Moderate analysis quality with {quality_score}/100 confidence score")
            
            # Levels finding
            if len(validated_levels) >= 5:
                findings.append(f"Strong level structure identified with {len(validated_levels)} validated levels")
            elif len(validated_levels) >= 3:
                findings.append(f"Adequate level structure with {len(validated_levels)} validated levels")
            else:
                findings.append(f"Limited level structure with only {len(validated_levels)} validated levels")
            
            # Current position finding
            range_position = current_analysis.get('range_position_classification', 'unknown')
            current_price = current_analysis.get('current_price')
            
            if current_price and range_position != 'unknown':
                if range_position == 'near_support':
                    findings.append("Price currently near key support level - potential bounce opportunity")
                elif range_position == 'near_resistance':
                    findings.append("Price currently near key resistance level - potential reversal risk")
                elif range_position == 'middle_range':
                    findings.append("Price in middle of trading range - direction unclear")
            
            # Distance to levels
            support_distance_pct = current_analysis.get('support_distance_percentage', float('inf'))
            resistance_distance_pct = current_analysis.get('resistance_distance_percentage', float('inf'))
            
            if support_distance_pct < 3:
                findings.append(f"Very close to support ({support_distance_pct:.1f}% away)")
            elif resistance_distance_pct < 3:
                findings.append(f"Very close to resistance ({resistance_distance_pct:.1f}% away)")
            
            # High reliability levels
            high_reliability_levels = [l for l in validated_levels if l['reliability'] in ['high', 'very_high']]
            if len(high_reliability_levels) > 0:
                findings.append(f"{len(high_reliability_levels)} high-reliability levels identified")
            
        except Exception as e:
            findings.append(f"Error extracting findings: {str(e)}")
        
        return findings
    
    def _generate_agent_insights(self, analysis_results: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Generate agent-specific insights"""
        insights = {
            'market_structure': {},
            'volume_insights': {},
            'level_insights': {},
            'risk_insights': {}
        }
        
        try:
            validated_levels = analysis_results.get('validated_levels', [])
            vap_analysis = analysis_results.get('volume_at_price_analysis', {})
            current_analysis = analysis_results.get('current_position_analysis', {})
            
            # Market structure insights
            support_levels = [l for l in validated_levels if l['type'] in ['support', 'both']]
            resistance_levels = [l for l in validated_levels if l['type'] in ['resistance', 'both']]
            
            if len(support_levels) > 0 and len(resistance_levels) > 0:
                insights['market_structure']['pattern'] = 'range_bound'
                insights['market_structure']['structure_strength'] = 'strong' if len(validated_levels) >= 4 else 'moderate'
            elif len(support_levels) > len(resistance_levels):
                insights['market_structure']['pattern'] = 'support_heavy'
            elif len(resistance_levels) > len(support_levels):
                insights['market_structure']['pattern'] = 'resistance_heavy'
            else:
                insights['market_structure']['pattern'] = 'unclear'
            
            # Volume insights
            if 'error' not in vap_analysis:
                highest_volume_level = vap_analysis.get('highest_volume_level')
                if highest_volume_level:
                    current_price = current_analysis.get('current_price', 0)
                    if current_price > 0:
                        distance_to_hvn = abs(highest_volume_level['price_level'] - current_price) / current_price * 100
                        insights['volume_insights']['highest_volume_node_distance'] = distance_to_hvn
                        
                        if distance_to_hvn < 2:
                            insights['volume_insights']['proximity_to_hvn'] = 'very_close'
                        elif distance_to_hvn < 5:
                            insights['volume_insights']['proximity_to_hvn'] = 'close'
                        else:
                            insights['volume_insights']['proximity_to_hvn'] = 'distant'
            
            # Level insights
            if validated_levels:
                avg_success_rate = sum(l['success_rate'] for l in validated_levels) / len(validated_levels)
                insights['level_insights']['average_success_rate'] = avg_success_rate
                insights['level_insights']['reliability_assessment'] = (
                    'high' if avg_success_rate > 0.7 else 'medium' if avg_success_rate > 0.5 else 'low'
                )
                
                # Test frequency insight
                avg_tests = sum(l['total_tests'] for l in validated_levels) / len(validated_levels)
                insights['level_insights']['level_maturity'] = (
                    'well_tested' if avg_tests >= 5 else 'moderately_tested' if avg_tests >= 3 else 'lightly_tested'
                )
            
            # Risk insights
            trading_implications = analysis_results.get('trading_implications', {})
            risk_reward_ratio = trading_implications.get('risk_reward_ratio')
            
            if risk_reward_ratio is not None:
                if risk_reward_ratio > 2:
                    insights['risk_insights']['risk_reward_assessment'] = 'favorable'
                elif risk_reward_ratio > 1:
                    insights['risk_insights']['risk_reward_assessment'] = 'acceptable'
                else:
                    insights['risk_insights']['risk_reward_assessment'] = 'unfavorable'
                
                insights['risk_insights']['risk_reward_ratio'] = risk_reward_ratio
            
        except Exception as e:
            insights['error'] = f"Insight generation failed: {str(e)}"
        
        return insights
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any], data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            current_analysis = analysis_results.get('current_position_analysis', {})
            trading_implications = analysis_results.get('trading_implications', {})
            validated_levels = analysis_results.get('validated_levels', [])
            
            range_position = current_analysis.get('range_position_classification', 'unknown')
            nearest_support = current_analysis.get('nearest_support')
            nearest_resistance = current_analysis.get('nearest_resistance')
            
            # Position-based recommendations
            if range_position == 'near_support' and nearest_support:
                recommendations.append({
                    'type': 'entry_opportunity',
                    'priority': 'high',
                    'action': 'consider_buy',
                    'reason': f"Price near strong support at ${nearest_support['price']:.2f}",
                    'stop_loss_suggestion': trading_implications.get('risk_levels', {}).get('support_break', {}).get('price'),
                    'target_suggestion': trading_implications.get('target_levels', {}).get('resistance_target', {}).get('price')
                })
            
            elif range_position == 'near_resistance' and nearest_resistance:
                recommendations.append({
                    'type': 'exit_opportunity',
                    'priority': 'high',
                    'action': 'consider_sell',
                    'reason': f"Price near strong resistance at ${nearest_resistance['price']:.2f}",
                    'risk_management': 'Consider taking profits or tightening stops'
                })
            
            elif range_position == 'middle_range':
                recommendations.append({
                    'type': 'wait_signal',
                    'priority': 'medium',
                    'action': 'wait_for_direction',
                    'reason': 'Price in middle of range - wait for clear directional move',
                    'watch_levels': {
                        'support': nearest_support['price'] if nearest_support else None,
                        'resistance': nearest_resistance['price'] if nearest_resistance else None
                    }
                })
            
            # Risk management recommendations
            risk_reward_ratio = trading_implications.get('risk_reward_ratio')
            if risk_reward_ratio is not None:
                if risk_reward_ratio < 1:
                    recommendations.append({
                        'type': 'risk_warning',
                        'priority': 'high',
                        'action': 'avoid_trade',
                        'reason': f'Unfavorable risk/reward ratio: {risk_reward_ratio:.2f}',
                        'suggestion': 'Wait for better entry opportunity'
                    })
            
            # Volume-based recommendations
            quality_assessment = analysis_results.get('quality_assessment', {})
            quality_score = quality_assessment.get('overall_score', 0)
            
            if quality_score < 60:
                recommendations.append({
                    'type': 'data_quality',
                    'priority': 'medium',
                    'action': 'use_caution',
                    'reason': f'Analysis quality score is {quality_score}/100',
                    'suggestion': 'Consider using additional analysis methods'
                })
            
            # Level strength recommendations
            high_strength_levels = [l for l in validated_levels if l['reliability'] in ['high', 'very_high']]
            if len(high_strength_levels) >= 3:
                recommendations.append({
                    'type': 'confidence_boost',
                    'priority': 'low',
                    'action': 'trust_analysis',
                    'reason': f'{len(high_strength_levels)} high-reliability levels identified',
                    'suggestion': 'Analysis has strong foundation for decision making'
                })
            
        except Exception as e:
            recommendations.append({
                'type': 'error',
                'priority': 'high',
                'action': 'manual_review',
                'reason': f'Recommendation generation failed: {str(e)}',
                'suggestion': 'Manual analysis recommended'
            })
        
        return recommendations
    
    def _format_error_response(self, error_message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format error response"""
        base_metadata = {
            'agent_name': self.agent_name,
            'agent_version': self.agent_version,
            'analysis_timestamp': datetime.now().isoformat(),
            'status': 'error'
        }
        
        if metadata:
            base_metadata.update(metadata)
        
        return {
            'metadata': base_metadata,
            'error': error_message,
            'analysis_summary': {'error': 'Analysis failed'},
            'key_findings': [f'Error: {error_message}'],
            'support_levels': [],
            'resistance_levels': [],
            'current_position': {},
            'trading_implications': {},
            'recommendations': [{
                'type': 'error',
                'priority': 'high',
                'action': 'fix_issue',
                'reason': error_message,
                'suggestion': 'Resolve data or configuration issues and retry'
            }]
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information and capabilities"""
        return {
            'name': self.agent_name,
            'version': self.agent_version,
            'description': self.description,
            'capabilities': self.capabilities,
            'data_requirements': {
                'minimum_days': self.min_data_points,
                'preferred_days': self.preferred_data_points,
                'required_columns': ['open', 'high', 'low', 'close', 'volume']
            },
            'output_format': {
                'support_levels': 'List of validated support levels with strength ratings',
                'resistance_levels': 'List of validated resistance levels with strength ratings',
                'current_position': 'Analysis of current price relative to key levels',
                'trading_implications': 'Risk levels, targets, and strategy suggestions',
                'recommendations': 'Actionable recommendations based on analysis'
            }
        }