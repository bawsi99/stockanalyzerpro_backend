#!/usr/bin/env python3
"""
Cross-Validation Charts - Visualization Module

This module generates advanced charts for cross-validation analysis including:
- Validation method comparison charts
- Pattern confidence scoring visualization
- Statistical validation heatmaps
- Volume confirmation analysis charts
- Alternative method validation displays
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

class CrossValidationChartGenerator:
    """
    Advanced chart generator for cross-validation analysis.
    
    Generates comprehensive visualizations including:
    - Validation method comparison and scoring
    - Pattern confidence assessment charts
    - Statistical validation visualizations
    - Volume and alternative method analysis
    """
    
    def __init__(self):
        self.name = "cross_validation_charts"
        self.version = "1.0.0"
    
    def generate_cross_validation_charts(
        self, 
        stock_data: pd.DataFrame, 
        validation_data: Dict[str, Any],
        symbol: str = "STOCK",
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive cross-validation charts.
        
        Args:
            stock_data: DataFrame with OHLCV data
            validation_data: Cross-validation analysis results
            symbol: Stock symbol for chart titles
            save_path: Optional path to save charts
            
        Returns:
            Dictionary containing chart data and metadata
        """
        try:
            logger.info(f"[CROSS_VALIDATION_CHARTS] Generating charts for {symbol}")
            
            if stock_data is None or stock_data.empty:
                return self._build_error_result("No stock data provided for charting")
            
            if not validation_data.get('success', False):
                return self._build_error_result("Cross-validation analysis failed")
            
            charts_data = {}
            
            # 1. Validation Methods Comparison Chart
            methods_chart = self._create_validation_methods_chart(validation_data, symbol)
            charts_data['validation_methods_chart'] = methods_chart
            
            # 2. Pattern Confidence Scoring Chart
            confidence_chart = self._create_pattern_confidence_chart(validation_data, symbol)
            charts_data['pattern_confidence_chart'] = confidence_chart
            
            # 3. Statistical Validation Heatmap
            statistical_heatmap = self._create_statistical_validation_heatmap(validation_data, symbol)
            charts_data['statistical_heatmap'] = statistical_heatmap
            
            # 4. Volume Confirmation Analysis
            volume_chart = self._create_volume_confirmation_chart(validation_data, symbol)
            charts_data['volume_confirmation_chart'] = volume_chart
            
            # 5. Pattern Consistency Analysis
            consistency_chart = self._create_consistency_analysis_chart(validation_data, symbol)
            charts_data['consistency_chart'] = consistency_chart
            
            # 6. Alternative Methods Radar Chart
            radar_chart = self._create_alternative_methods_radar(validation_data, symbol)
            charts_data['alternative_methods_radar'] = radar_chart
            
            # 7. Comprehensive Validation Dashboard
            dashboard = self._create_validation_dashboard(validation_data, symbol)
            charts_data['validation_dashboard'] = dashboard
            
            # Save charts if path provided
            if save_path and os.path.exists(os.path.dirname(save_path)):
                self._save_charts(charts_data, save_path, symbol)
            
            result = {
                'success': True,
                'agent_name': self.name,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'charts_generated': len(charts_data),
                'charts_data': charts_data,
                'validation_methods_visualized': validation_data.get('validation_summary', {}).get('validation_methods_used', 0)
            }
            
            logger.info(f"[CROSS_VALIDATION_CHARTS] Generated {len(charts_data)} charts for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_CHARTS] Chart generation failed: {e}")
            return self._build_error_result(str(e))
    
    def _create_validation_methods_chart(self, validation_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Create chart comparing validation methods"""
        try:
            validation_scores = validation_data.get('validation_scores', {})
            method_scores = validation_scores.get('method_scores', {})
            
            if not method_scores:
                return {'error': 'No method scores available'}
            
            # Create bar chart comparing methods
            methods = list(method_scores.keys())
            scores = [method_scores[method] for method in methods]
            
            # Color code based on score
            colors = []
            for score in scores:
                if score >= 0.8:
                    colors.append('green')
                elif score >= 0.6:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            fig = go.Figure(data=[
                go.Bar(
                    x=methods,
                    y=scores,
                    marker_color=colors,
                    text=[f'{score:.2f}' for score in scores],
                    textposition='inside'
                )
            ])
            
            fig.update_layout(
                title=f'{symbol} - Cross-Validation Methods Comparison',
                xaxis_title='Validation Methods',
                yaxis_title='Validation Score',
                yaxis=dict(range=[0, 1]),
                height=500
            )
            
            # Add horizontal line for overall score
            overall_score = validation_scores.get('overall_score', 0)
            fig.add_hline(
                y=overall_score,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Overall: {overall_score:.2f}"
            )
            
            return {
                'chart_type': 'validation_methods',
                'title': f'{symbol} Validation Methods Comparison',
                'data': fig.to_json(),
                'methods_compared': len(methods),
                'overall_score': overall_score
            }
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_CHARTS] Validation methods chart failed: {e}")
            return {'error': str(e)}
    
    def _create_pattern_confidence_chart(self, validation_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Create pattern confidence scoring chart"""
        try:
            pattern_details = validation_data.get('pattern_validation_details', [])
            
            if not pattern_details:
                return {'error': 'No pattern details available'}
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Original vs Validated Confidence', 'Validation Method Scores',
                    'Pattern Reliability Distribution', 'Confidence Categories'
                ),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "pie"}, {"type": "bar"}]]
            )
            
            # 1. Original vs Validated Confidence Scatter
            pattern_names = [p['pattern_name'] for p in pattern_details]
            original_reliability = []
            validated_confidence = []
            
            for pattern in pattern_details:
                orig_rel = pattern.get('original_reliability', 'unknown')
                orig_score = {'high': 0.8, 'medium': 0.6, 'low': 0.4}.get(orig_rel, 0.5)
                original_reliability.append(orig_score)
                
                # Calculate average validation score
                validation_results = pattern.get('validation_results', {})
                val_scores = []
                for method_data in validation_results.values():
                    if isinstance(method_data, dict):
                        for key, value in method_data.items():
                            if 'score' in key and isinstance(value, (int, float)):
                                val_scores.append(value)
                
                avg_val_score = np.mean(val_scores) if val_scores else 0.5
                validated_confidence.append(avg_val_score)
            
            fig.add_trace(
                go.Scatter(
                    x=original_reliability,
                    y=validated_confidence,
                    mode='markers+text',
                    text=pattern_names,
                    textposition="top center",
                    marker=dict(size=12, color='blue', opacity=0.7),
                    name="Patterns"
                ),
                row=1, col=1
            )
            
            # Add diagonal line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 2. Validation Method Scores per Pattern
            if pattern_details:
                method_names = set()
                for pattern in pattern_details:
                    method_names.update(pattern.get('validation_results', {}).keys())
                
                method_names = list(method_names)
                for i, method in enumerate(method_names):
                    method_scores = []
                    for pattern in pattern_details:
                        method_data = pattern.get('validation_results', {}).get(method, {})
                        score = 0.5  # Default
                        for key, value in method_data.items():
                            if 'score' in key and isinstance(value, (int, float)):
                                score = value
                                break
                        method_scores.append(score)
                    
                    fig.add_trace(
                        go.Bar(
                            x=pattern_names,
                            y=method_scores,
                            name=method,
                            offsetgroup=i
                        ),
                        row=1, col=2
                    )
            
            # 3. Pattern Reliability Distribution
            reliability_counts = {}
            for pattern in pattern_details:
                rel = pattern.get('original_reliability', 'unknown')
                reliability_counts[rel] = reliability_counts.get(rel, 0) + 1
            
            fig.add_trace(
                go.Pie(
                    labels=list(reliability_counts.keys()),
                    values=list(reliability_counts.values()),
                    name="Reliability"
                ),
                row=2, col=1
            )
            
            # 4. Confidence Categories
            final_assessment = validation_data.get('final_confidence_assessment', {})
            confidence_level = final_assessment.get('confidence_level', 'unknown')
            
            # Create categories based on final confidence
            confidence_categories = ['very_low', 'low', 'medium', 'high', 'very_high']
            category_values = [1 if cat == confidence_level else 0 for cat in confidence_categories]
            
            fig.add_trace(
                go.Bar(
                    x=confidence_categories,
                    y=category_values,
                    marker_color=['red' if v == 1 else 'lightgray' for v in category_values],
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'{symbol} - Pattern Confidence Analysis',
                height=800,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Original Reliability", row=1, col=1)
            fig.update_yaxes(title_text="Validated Confidence", row=1, col=1)
            fig.update_xaxes(title_text="Patterns", row=1, col=2)
            fig.update_yaxes(title_text="Method Scores", row=1, col=2)
            fig.update_xaxes(title_text="Confidence Level", row=2, col=2)
            fig.update_yaxes(title_text="Current Level", row=2, col=2)
            
            return {
                'chart_type': 'pattern_confidence',
                'title': f'{symbol} Pattern Confidence Analysis',
                'data': fig.to_json(),
                'patterns_analyzed': len(pattern_details)
            }
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_CHARTS] Pattern confidence chart failed: {e}")
            return {'error': str(e)}
    
    def _create_statistical_validation_heatmap(self, validation_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Create statistical validation heatmap"""
        try:
            statistical_validation = validation_data.get('statistical_validation', {})
            validation_results = statistical_validation.get('validation_results', [])
            
            if not validation_results:
                return {'error': 'No statistical validation results available'}
            
            # Prepare data for heatmap
            patterns = [result['pattern_name'] for result in validation_results]
            test_names = []
            if validation_results:
                test_names = list(validation_results[0].get('statistical_tests', {}).keys())
            
            # Create matrix
            heatmap_data = []
            for result in validation_results:
                tests = result.get('statistical_tests', {})
                row = [tests.get(test_name, 0) for test_name in test_names]
                heatmap_data.append(row)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=test_names,
                y=patterns,
                colorscale='RdYlGn',
                zmid=0.5,
                text=[[f'{val:.2f}' if isinstance(val, (int, float)) else str(val) 
                       for val in row] for row in heatmap_data],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Test Score")
            ))
            
            fig.update_layout(
                title=f'{symbol} - Statistical Validation Tests Heatmap',
                xaxis_title='Statistical Tests',
                yaxis_title='Detected Patterns',
                height=400 + len(patterns) * 30
            )
            
            return {
                'chart_type': 'statistical_heatmap',
                'title': f'{symbol} Statistical Validation Heatmap',
                'data': fig.to_json(),
                'patterns_tested': len(patterns),
                'tests_performed': len(test_names)
            }
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_CHARTS] Statistical heatmap failed: {e}")
            return {'error': str(e)}
    
    def _create_volume_confirmation_chart(self, validation_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Create volume confirmation analysis chart"""
        try:
            volume_confirmation = validation_data.get('volume_confirmation', {})
            
            if volume_confirmation.get('error'):
                return {'error': 'Volume confirmation data not available'}
            
            confirmation_results = volume_confirmation.get('confirmation_results', [])
            
            if not confirmation_results:
                return {'error': 'No volume confirmation results'}
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Volume Confirmation Scores', 'Volume Strength Categories',
                    'Volume Analysis Components', 'Pattern Volume Distribution'
                ),
                specs=[[{"type": "bar"}, {"type": "pie"}],
                       [{"type": "bar"}, {"type": "histogram"}]]
            )
            
            # 1. Volume Confirmation Scores
            patterns = [result['pattern_name'] for result in confirmation_results]
            vol_scores = [result['volume_confirmation_score'] for result in confirmation_results]
            
            colors = ['green' if score >= 0.7 else 'orange' if score >= 0.5 else 'red' 
                     for score in vol_scores]
            
            fig.add_trace(
                go.Bar(
                    x=patterns,
                    y=vol_scores,
                    marker_color=colors,
                    text=[f'{score:.2f}' for score in vol_scores],
                    textposition='inside',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 2. Volume Strength Categories
            strength_counts = {}
            for result in confirmation_results:
                strength = result.get('volume_strength', 'unknown')
                strength_counts[strength] = strength_counts.get(strength, 0) + 1
            
            fig.add_trace(
                go.Pie(
                    labels=list(strength_counts.keys()),
                    values=list(strength_counts.values()),
                    name="Volume Strength"
                ),
                row=1, col=2
            )
            
            # 3. Volume Analysis Components
            if confirmation_results:
                component_names = list(confirmation_results[0].get('volume_analysis', {}).keys())
                
                for component in component_names:
                    component_scores = []
                    for result in confirmation_results:
                        score = result.get('volume_analysis', {}).get(component, 0.5)
                        component_scores.append(score if isinstance(score, (int, float)) else 0.5)
                    
                    fig.add_trace(
                        go.Bar(
                            x=patterns,
                            y=component_scores,
                            name=component.replace('_', ' ').title()
                        ),
                        row=2, col=1
                    )
            
            # 4. Volume Score Distribution
            fig.add_trace(
                go.Histogram(
                    x=vol_scores,
                    nbinsx=10,
                    name="Score Distribution",
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'{symbol} - Volume Confirmation Analysis',
                height=800,
                showlegend=True
            )
            
            # Update axes
            fig.update_yaxes(title_text="Confirmation Score", row=1, col=1)
            fig.update_yaxes(title_text="Component Score", row=2, col=1)
            fig.update_xaxes(title_text="Volume Score", row=2, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=2)
            
            return {
                'chart_type': 'volume_confirmation',
                'title': f'{symbol} Volume Confirmation Analysis',
                'data': fig.to_json(),
                'patterns_analyzed': len(confirmation_results)
            }
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_CHARTS] Volume confirmation chart failed: {e}")
            return {'error': str(e)}
    
    def _create_consistency_analysis_chart(self, validation_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Create pattern consistency analysis chart"""
        try:
            consistency_analysis = validation_data.get('consistency_analysis', {})
            
            if consistency_analysis.get('error'):
                return {'error': 'Consistency analysis data not available'}
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Pattern Conflicts', 'Pattern Reinforcements',
                    'Bias Consistency', 'Overall Consistency Score'
                ),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "pie"}, {"type": "indicator"}]]
            )
            
            # 1. Pattern Conflicts
            conflicts = consistency_analysis.get('pattern_conflicts', [])
            if conflicts:
                conflict_types = [c['conflict_type'] for c in conflicts]
                conflict_counts = {}
                for ctype in conflict_types:
                    conflict_counts[ctype] = conflict_counts.get(ctype, 0) + 1
                
                fig.add_trace(
                    go.Bar(
                        x=list(conflict_counts.keys()),
                        y=list(conflict_counts.values()),
                        marker_color='red',
                        name='Conflicts',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=['No Conflicts'],
                        y=[1],
                        marker_color='green',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # 2. Pattern Reinforcements
            reinforcements = consistency_analysis.get('pattern_reinforcements', [])
            if reinforcements:
                reinf_types = [r['reinforcement_type'] for r in reinforcements]
                reinf_counts = {}
                for rtype in reinf_types:
                    reinf_counts[rtype] = reinf_counts.get(rtype, 0) + 1
                
                fig.add_trace(
                    go.Bar(
                        x=list(reinf_counts.keys()),
                        y=list(reinf_counts.values()),
                        marker_color='green',
                        name='Reinforcements',
                        showlegend=False
                    ),
                    row=1, col=2
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=['No Reinforcements'],
                        y=[1],
                        marker_color='orange',
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # 3. Bias Consistency
            bias_consistency = consistency_analysis.get('bias_consistency', {})
            if not bias_consistency.get('error'):
                bias_dist = bias_consistency.get('bias_distribution', {})
                if bias_dist:
                    fig.add_trace(
                        go.Pie(
                            labels=list(bias_dist.keys()),
                            values=[bias_dist[k] for k in bias_dist.keys()],
                            name="Bias Distribution"
                        ),
                        row=2, col=1
                    )
            
            # 4. Overall Consistency Score
            consistency_score = consistency_analysis.get('consistency_score', 0.5)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=consistency_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Consistency %"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'{symbol} - Pattern Consistency Analysis',
                height=700
            )
            
            return {
                'chart_type': 'consistency_analysis',
                'title': f'{symbol} Pattern Consistency Analysis',
                'data': fig.to_json(),
                'conflicts_detected': len(conflicts),
                'reinforcements_detected': len(reinforcements)
            }
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_CHARTS] Consistency analysis chart failed: {e}")
            return {'error': str(e)}
    
    def _create_alternative_methods_radar(self, validation_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Create radar chart for alternative validation methods"""
        try:
            alternative_validation = validation_data.get('alternative_validation', {})
            alternative_results = alternative_validation.get('alternative_results', [])
            
            if not alternative_results:
                return {'error': 'No alternative validation results available'}
            
            # Create radar chart for each pattern
            fig = go.Figure()
            
            for i, pattern_result in enumerate(alternative_results):
                pattern_name = pattern_result['pattern_name']
                methods = pattern_result.get('alternative_methods', {})
                
                method_names = list(methods.keys())
                method_scores = [methods[method] if isinstance(methods[method], (int, float)) else 0.5 
                               for method in method_names]
                
                # Close the radar chart
                method_names_closed = method_names + [method_names[0]] if method_names else []
                method_scores_closed = method_scores + [method_scores[0]] if method_scores else []
                
                fig.add_trace(go.Scatterpolar(
                    r=method_scores_closed,
                    theta=method_names_closed,
                    fill='toself',
                    name=pattern_name,
                    opacity=0.6
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title=f'{symbol} - Alternative Validation Methods Radar',
                showlegend=True
            )
            
            return {
                'chart_type': 'alternative_methods_radar',
                'title': f'{symbol} Alternative Methods Radar',
                'data': fig.to_json(),
                'patterns_analyzed': len(alternative_results)
            }
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_CHARTS] Alternative methods radar failed: {e}")
            return {'error': str(e)}
    
    def _create_validation_dashboard(self, validation_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Create comprehensive validation dashboard"""
        try:
            validation_summary = validation_data.get('validation_summary', {})
            validation_scores = validation_data.get('validation_scores', {})
            final_confidence = validation_data.get('final_confidence_assessment', {})
            
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=(
                    'Overall Validation Score', 'Method Completeness', 'Final Confidence',
                    'Patterns Validated', 'Methods Used', 'Data Quality',
                    'Validation Quality', 'Processing Time', 'Recommendation'
                ),
                specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}, {"type": "table"}]]
            )
            
            # Row 1
            # Overall Validation Score
            overall_score = validation_scores.get('overall_score', 0.5)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=overall_score * 100,
                    title={'text': "Overall Score %"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 80], 'color': "yellow"},
                                   {'range': [80, 100], 'color': "green"}]}
                ),
                row=1, col=1
            )
            
            # Method Completeness
            completeness = validation_scores.get('validation_completeness', 0.5)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=completeness * 100,
                    title={'text': "Completeness %"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "purple"}}
                ),
                row=1, col=2
            )
            
            # Final Confidence
            final_conf = final_confidence.get('overall_confidence', 0.5)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=final_conf * 100,
                    title={'text': "Confidence %"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "green"}}
                ),
                row=1, col=3
            )
            
            # Row 2
            # Patterns Validated
            patterns_validated = validation_summary.get('patterns_validated', 0)
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=patterns_validated,
                    title={'text': "Patterns Validated"}
                ),
                row=2, col=1
            )
            
            # Methods Used
            methods_used = validation_summary.get('validation_methods_used', 0)
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=methods_used,
                    title={'text': "Methods Used"}
                ),
                row=2, col=2
            )
            
            # Data Quality
            data_quality = validation_data.get('data_quality', {})
            quality_score = data_quality.get('overall_quality_score', 50)
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=quality_score,
                    title={'text': "Data Quality"}
                ),
                row=2, col=3
            )
            
            # Row 3
            # Validation Quality
            validation_quality = validation_scores.get('validation_quality', 'unknown')
            quality_score = {'very_high': 95, 'high': 85, 'medium': 70, 'low': 50, 'very_low': 30}.get(validation_quality, 60)
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=quality_score,
                    title={'text': "Quality Score"}
                ),
                row=3, col=1
            )
            
            # Processing Time
            processing_time = validation_data.get('processing_time', 0)
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=processing_time,
                    title={'text': "Time (s)"},
                    number={'suffix': "s"}
                ),
                row=3, col=2
            )
            
            # Recommendations Table
            recommendation = final_confidence.get('recommendation', 'No recommendation available')
            confidence_level = final_confidence.get('confidence_level', 'unknown')
            confidence_category = final_confidence.get('confidence_category', 'unknown')
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value']),
                    cells=dict(values=[
                        ['Confidence Level', 'Category', 'Recommendation'],
                        [confidence_level, confidence_category, recommendation[:50] + '...' if len(recommendation) > 50 else recommendation]
                    ])
                ),
                row=3, col=3
            )
            
            fig.update_layout(
                title=f'{symbol} - Cross-Validation Dashboard',
                height=900,
                showlegend=False
            )
            
            return {
                'chart_type': 'validation_dashboard',
                'title': f'{symbol} Validation Dashboard',
                'data': fig.to_json(),
                'overall_score': overall_score,
                'final_confidence': final_conf
            }
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_CHARTS] Validation dashboard failed: {e}")
            return {'error': str(e)}
    
    def _save_charts(self, charts_data: Dict, save_path: str, symbol: str):
        """Save charts to files"""
        try:
            base_path = os.path.splitext(save_path)[0]
            
            for chart_name, chart_info in charts_data.items():
                if 'data' in chart_info and not chart_info.get('error'):
                    chart_path = f"{base_path}_{chart_name}_{symbol}.html"
                    
                    # Convert JSON back to figure and save
                    import plotly.io as pio
                    fig = pio.from_json(chart_info['data'])
                    fig.write_html(chart_path)
                    
            logger.info(f"[CROSS_VALIDATION_CHARTS] Charts saved to {base_path}_*_{symbol}.html")
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_CHARTS] Chart saving failed: {e}")
    
    def _build_error_result(self, error_message: str) -> Dict[str, Any]:
        """Build error result dictionary"""
        return {
            'success': False,
            'agent_name': self.name,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'charts_generated': 0
        }