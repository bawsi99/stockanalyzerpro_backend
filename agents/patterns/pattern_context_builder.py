"""
Pattern Context Builder

Transforms pattern analysis results from PatternAgentsOrchestrator into 
comprehensive LLM-friendly text context for analysis and synthesis.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class PatternContextBuilder:
    """
    Transforms pattern analysis results into comprehensive LLM context.
    
    Takes the AggregatedPatternAnalysis output from PatternAgentsOrchestrator
    and builds structured text context suitable for LLM analysis.
    """
    
    def __init__(self):
        self.name = "pattern_context_builder"
        self.version = "1.0.0"
    
    def build_comprehensive_pattern_context(
        self, 
        aggregated_analysis: Dict[str, Any],
        symbol: str,
        current_price: float = None
    ) -> str:
        """
        Build comprehensive context from all pattern agent results.
        
        Args:
            aggregated_analysis: Results from PatternAgentsOrchestrator.analyze_patterns_comprehensive()
            symbol: Stock symbol being analyzed
            current_price: Current stock price for context
            
        Returns:
            Formatted text context for LLM analysis
        """
        try:
            # Build main context header
            context_parts = []
            context_parts.append(self._build_header(symbol, current_price, aggregated_analysis))
            
            # Add individual agent results
            context_parts.append(self._format_individual_results(aggregated_analysis))
            
            # Add unified analysis
            context_parts.append(self._format_unified_analysis(aggregated_analysis))
            
            # Add consensus and conflicts
            context_parts.append(self._format_consensus_analysis(aggregated_analysis))
            
            # Add confidence and reliability metrics
            context_parts.append(self._format_confidence_metrics(aggregated_analysis))
            
            # Join all parts
            full_context = "\n\n".join([part for part in context_parts if part.strip()])
            
            logger.info(f"[PATTERN_CONTEXT] Built comprehensive context for {symbol} ({len(full_context)} chars)")
            return full_context
            
        except Exception as e:
            logger.error(f"[PATTERN_CONTEXT] Error building context for {symbol}: {e}")
            # Return fallback context
            return self._build_fallback_context(symbol, aggregated_analysis, str(e))
    
    def _build_header(self, symbol: str, current_price: Optional[float], analysis: Dict[str, Any]) -> str:
        """Build context header with basic information."""
        
        header = f"""COMPREHENSIVE PATTERN ANALYSIS FOR {symbol}
Analysis Timestamp: {datetime.now().isoformat()}"""
        
        if current_price:
            header += f"\nCurrent Price: ${current_price:.2f}"
        
        # Add analysis metadata
        if 'total_processing_time' in analysis:
            header += f"\nProcessing Time: {analysis['total_processing_time']:.2f}s"
        
        if 'successful_agents' in analysis and 'failed_agents' in analysis:
            successful = analysis['successful_agents']
            total = successful + analysis['failed_agents']
            header += f"\nPattern Agents: {successful}/{total} successful"
        
        if 'overall_confidence' in analysis:
            header += f"\nOverall Confidence: {analysis['overall_confidence']:.2f}"
        
        return header
    
    def _format_individual_results(self, analysis: Dict[str, Any]) -> str:
        """Format individual pattern agent results."""
        
        individual_results = analysis.get('individual_results', {})
        if not individual_results:
            return "INDIVIDUAL AGENT RESULTS:\nNo individual results available."
        
        sections = ["INDIVIDUAL AGENT RESULTS:"]
        
        for agent_name, result in individual_results.items():
            if not isinstance(result, dict):
                continue
                
            sections.append(f"\n{agent_name.upper().replace('_', ' ')} AGENT:")
            sections.append(f"- Success: {result.get('success', False)}")
            sections.append(f"- Processing Time: {result.get('processing_time', 0):.2f}s")
            sections.append(f"- Confidence: {result.get('confidence_score', 0):.2f}")
            
            if result.get('error_message'):
                sections.append(f"- Error: {result['error_message']}")
            elif result.get('analysis_data'):
                # Format analysis data based on agent type
                formatted_data = self._format_agent_specific_data(agent_name, result['analysis_data'])
                if formatted_data:
                    sections.append(formatted_data)
        
        return "\n".join(sections)
    
    def _format_agent_specific_data(self, agent_name: str, data: Dict[str, Any]) -> str:
        """Format agent-specific analysis data."""
        
        try:
            if agent_name == 'reversal':
                return self._format_reversal_data(data)
            elif agent_name == 'continuation':
                return self._format_continuation_data(data)
            elif agent_name == 'pattern_recognition':
                return self._format_pattern_recognition_data(data)
            elif agent_name == 'technical_overview':
                return self._format_technical_overview_data(data)
            else:
                # Generic formatting for unknown agent types
                return f"- Analysis Data: {json.dumps(data, indent=2)}"
        except Exception as e:
            logger.warning(f"[PATTERN_CONTEXT] Error formatting {agent_name} data: {e}")
            return f"- Analysis Data: [Error formatting data]"
    
    def _format_reversal_data(self, data: Dict[str, Any]) -> str:
        """Format reversal pattern agent data."""
        
        sections = []
        
        # Primary signal
        if 'primary_signal' in data:
            sections.append(f"- Primary Signal: {data['primary_signal']}")
        
        # Signal strength
        if 'signal_strength' in data:
            sections.append(f"- Signal Strength: {data['signal_strength']}")
        
        # Reversal patterns
        reversal_patterns = data.get('reversal_patterns', {})
        if reversal_patterns:
            sections.append("- Detected Patterns:")
            
            # Divergences
            divergences = reversal_patterns.get('divergences', [])
            if divergences:
                sections.append(f"  * Divergences: {len(divergences)} detected")
                for i, div in enumerate(divergences[:3]):  # Show top 3
                    if isinstance(div, dict):
                        sections.append(f"    - {div.get('type', 'unknown')}: confidence {div.get('confidence', 0):.2f}")
            
            # Double patterns
            double_patterns = reversal_patterns.get('double_patterns', [])
            if double_patterns:
                sections.append(f"  * Double Patterns: {len(double_patterns)} detected")
            
            # Other reversals
            other_reversals = reversal_patterns.get('other_reversals', [])
            if other_reversals:
                sections.append(f"  * Other Reversals: {len(other_reversals)} detected")
        
        # Entry/exit levels
        if 'entry_points' in data:
            entry_points = data['entry_points']
            if entry_points:
                sections.append(f"- Entry Points: {', '.join([f'${ep:.2f}' for ep in entry_points])}")
        
        if 'stop_loss_levels' in data:
            stop_levels = data['stop_loss_levels']
            if stop_levels:
                sections.append(f"- Stop Loss Levels: {', '.join([f'${sl:.2f}' for sl in stop_levels])}")
        
        if 'target_levels' in data:
            targets = data['target_levels']
            if targets:
                sections.append(f"- Target Levels: {', '.join([f'${tl:.2f}' for tl in targets])}")
        
        return "\n".join(sections) if sections else "- No reversal patterns detected"
    
    def _format_continuation_data(self, data: Dict[str, Any]) -> str:
        """Format continuation pattern agent data."""
        
        sections = []
        
        # Primary signal
        if 'primary_signal' in data:
            sections.append(f"- Primary Signal: {data['primary_signal']}")
        
        # Continuation patterns
        continuation_patterns = data.get('continuation_patterns', {})
        if continuation_patterns:
            sections.append("- Detected Patterns:")
            
            # Triangles
            triangles = continuation_patterns.get('triangles', [])
            if triangles:
                sections.append(f"  * Triangles: {len(triangles)} detected")
            
            # Flags/Pennants
            flags_pennants = continuation_patterns.get('flags_pennants', [])
            if flags_pennants:
                sections.append(f"  * Flags/Pennants: {len(flags_pennants)} detected")
            
            # Channels
            channels = continuation_patterns.get('channels', [])
            if channels:
                sections.append(f"  * Channels: {len(channels)} detected")
        
        # Breakout analysis
        if 'breakout_potential' in data:
            sections.append(f"- Breakout Potential: {data['breakout_potential']}")
        
        # Key levels
        key_levels = data.get('key_levels', {})
        if key_levels:
            sections.append("- Key Levels:")
            if 'resistance_levels' in key_levels:
                resistance = key_levels['resistance_levels']
                if resistance:
                    sections.append(f"  * Resistance: {', '.join([f'${r:.2f}' for r in resistance])}")
            if 'support_levels' in key_levels:
                support = key_levels['support_levels']
                if support:
                    sections.append(f"  * Support: {', '.join([f'${s:.2f}' for s in support])}")
        
        return "\n".join(sections) if sections else "- No continuation patterns detected"
    
    def _format_pattern_recognition_data(self, data: Dict[str, Any]) -> str:
        """Format pattern recognition agent data."""
        
        sections = []
        
        # Market structure
        market_structure = data.get('market_structure', {})
        if market_structure:
            trend_analysis = market_structure.get('trend_analysis', {})
            if trend_analysis:
                sections.append("- Market Structure:")
                sections.append(f"  * Short-term Trend: {trend_analysis.get('short_term_trend', 'neutral')}")
                sections.append(f"  * Medium-term Trend: {trend_analysis.get('medium_term_trend', 'neutral')}")
                sections.append(f"  * Trend Strength: {trend_analysis.get('trend_strength', 0):.2f}")
        
        # Price patterns
        price_patterns = data.get('price_patterns', {})
        if price_patterns:
            chart_patterns = price_patterns.get('chart_patterns', [])
            if chart_patterns:
                sections.append(f"- Chart Patterns: {len(chart_patterns)} detected")
        
        # Volume patterns
        volume_patterns = data.get('volume_patterns', {})
        if volume_patterns:
            volume_trend = volume_patterns.get('volume_trend', {})
            if volume_trend:
                sections.append(f"- Volume Trend: {volume_trend.get('trend_direction', 'neutral')}")
        
        return "\n".join(sections) if sections else "- Pattern recognition analysis completed"
    
    def _format_technical_overview_data(self, data: Dict[str, Any]) -> str:
        """Format technical overview agent data."""
        
        sections = []
        
        # Just show key summary information to avoid duplication
        if 'summary' in data:
            summary = data['summary']
            sections.append(f"- Technical Overview: {json.dumps(summary, indent=2)}")
        else:
            sections.append("- Technical overview analysis completed")
        
        return "\n".join(sections)
    
    def _format_unified_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format unified analysis section."""
        
        unified = analysis.get('unified_analysis', {})
        if not unified:
            return "UNIFIED ANALYSIS:\nNo unified analysis available."
        
        sections = ["UNIFIED ANALYSIS:"]
        
        # Pattern summary
        pattern_summary = unified.get('pattern_summary', {})
        if pattern_summary:
            total_patterns = pattern_summary.get('total_patterns_identified', 0)
            pattern_types = pattern_summary.get('pattern_types', [])
            high_confidence = pattern_summary.get('high_confidence_patterns', [])
            
            sections.append(f"- Total Patterns Identified: {total_patterns}")
            sections.append(f"- Pattern Types: {', '.join(pattern_types)}")
            sections.append(f"- High Confidence Patterns: {len(high_confidence)}")
        
        # Key levels
        key_levels = unified.get('key_levels', {})
        if key_levels:
            sections.append("- Key Levels Summary:")
            sections.append(f"  {json.dumps(key_levels, indent=2)}")
        
        # Trading recommendations
        trading_recs = unified.get('trading_recommendations', {})
        if trading_recs:
            sections.append("- Trading Recommendations:")
            sections.append(f"  {json.dumps(trading_recs, indent=2)}")
        
        return "\n".join(sections)
    
    def _format_consensus_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format consensus and conflict analysis."""
        
        sections = ["CONSENSUS & CONFLICTS ANALYSIS:"]
        
        # Consensus signals
        consensus = analysis.get('consensus_signals', {})
        if consensus:
            sections.append("- Consensus Signals:")
            for signal, data in consensus.items():
                if isinstance(data, dict):
                    support_count = data.get('support_count', 0)
                    strength = data.get('consensus_strength', 0)
                    sections.append(f"  * {signal}: {support_count} agents support (strength: {strength:.2f})")
        
        # Conflicting signals
        conflicts = analysis.get('conflicting_signals', [])
        if conflicts:
            sections.append(f"- Conflicting Signals: {len(conflicts)} detected")
            for conflict in conflicts:
                if isinstance(conflict, dict):
                    conflict_type = conflict.get('conflict_type', 'unknown')
                    severity = conflict.get('severity', 'unknown')
                    sections.append(f"  * {conflict_type} (severity: {severity})")
        
        return "\n".join(sections)
    
    def _format_confidence_metrics(self, analysis: Dict[str, Any]) -> str:
        """Format confidence and reliability metrics."""
        
        sections = ["CONFIDENCE & RELIABILITY METRICS:"]
        
        # Overall confidence
        overall_confidence = analysis.get('overall_confidence', 0)
        sections.append(f"- Overall Confidence Score: {overall_confidence:.2f}")
        
        # Successful vs failed agents
        successful = analysis.get('successful_agents', 0)
        failed = analysis.get('failed_agents', 0)
        total = successful + failed
        
        if total > 0:
            success_rate = successful / total
            sections.append(f"- Agent Success Rate: {success_rate:.2%} ({successful}/{total})")
        
        # Processing time
        processing_time = analysis.get('total_processing_time', 0)
        sections.append(f"- Total Processing Time: {processing_time:.2f}s")
        
        return "\n".join(sections)
    
    def _build_fallback_context(self, symbol: str, analysis: Dict[str, Any], error: str) -> str:
        """Build fallback context when main building fails."""
        
        return f"""PATTERN ANALYSIS FOR {symbol}
Analysis Timestamp: {datetime.now().isoformat()}

ERROR: Failed to build comprehensive context - {error}

RAW ANALYSIS DATA:
{json.dumps(analysis, indent=2, default=str)}

Please analyze this pattern data despite the formatting error."""


# Test function for the context builder
def test_pattern_context_builder():
    """Test the Pattern Context Builder with sample data."""
    
    print("🧪 Testing Pattern Context Builder")
    print("=" * 50)
    
    try:
        # Create context builder
        builder = PatternContextBuilder()
        print("✅ PatternContextBuilder created successfully")
        
        # Sample aggregated analysis data (mimicking PatternAgentsOrchestrator output)
        sample_analysis = {
            'total_processing_time': 12.45,
            'successful_agents': 3,
            'failed_agents': 1,
            'overall_confidence': 0.78,
            'individual_results': {
                'reversal': {
                    'success': True,
                    'processing_time': 3.2,
                    'confidence_score': 0.85,
                    'analysis_data': {
                        'primary_signal': 'bullish_reversal',
                        'signal_strength': 'strong',
                        'reversal_patterns': {
                            'divergences': [
                                {'type': 'bullish_divergence', 'confidence': 0.8},
                                {'type': 'rsi_divergence', 'confidence': 0.7}
                            ],
                            'double_patterns': [
                                {'type': 'double_bottom', 'completion': 85.0}
                            ]
                        },
                        'entry_points': [2450.0, 2465.0],
                        'stop_loss_levels': [2420.0],
                        'target_levels': [2520.0, 2580.0]
                    }
                },
                'continuation': {
                    'success': True,
                    'processing_time': 2.8,
                    'confidence_score': 0.72,
                    'analysis_data': {
                        'primary_signal': 'neutral',
                        'continuation_patterns': {
                            'triangles': [{'type': 'ascending_triangle'}],
                            'flags_pennants': [],
                            'channels': [{'type': 'horizontal_channel'}]
                        },
                        'breakout_potential': 'medium',
                        'key_levels': {
                            'resistance_levels': [2500.0, 2550.0],
                            'support_levels': [2400.0, 2380.0]
                        }
                    }
                }
            },
            'unified_analysis': {
                'pattern_summary': {
                    'total_patterns_identified': 4,
                    'pattern_types': ['bullish_divergence', 'double_bottom', 'ascending_triangle'],
                    'high_confidence_patterns': [
                        {'type': 'bullish_divergence', 'confidence': 0.8}
                    ]
                },
                'key_levels': {
                    'major_resistance': 2550.0,
                    'major_support': 2380.0
                }
            },
            'consensus_signals': {
                'bullish_reversal': {
                    'support_count': 2,
                    'consensus_strength': 0.75
                }
            },
            'conflicting_signals': [
                {
                    'conflict_type': 'directional_conflict',
                    'severity': 'low'
                }
            ]
        }
        
        # Test context building
        context = builder.build_comprehensive_pattern_context(
            sample_analysis, 
            "TEST_STOCK", 
            2455.50
        )
        
        print("✅ Context built successfully")
        print(f"   Context length: {len(context)} characters")
        print(f"   Contains header: {'COMPREHENSIVE PATTERN ANALYSIS' in context}")
        print(f"   Contains individual results: {'INDIVIDUAL AGENT RESULTS' in context}")
        print(f"   Contains unified analysis: {'UNIFIED ANALYSIS' in context}")
        print(f"   Contains confidence metrics: {'CONFIDENCE & RELIABILITY' in context}")
        
        # Print first 500 characters for preview
        print(f"\n📋 Context Preview (first 500 chars):")
        print("-" * 50)
        print(context[:500] + "..." if len(context) > 500 else context)
        
        print(f"\n✅ Pattern Context Builder test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Pattern Context Builder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_pattern_context_builder()