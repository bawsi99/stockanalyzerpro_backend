#!/usr/bin/env python3
"""
Pattern Conflict Detector - Pattern Signal Conflict Analysis

This module detects and analyzes conflicts between different pattern signals,
providing conflict resolution recommendations for the cross-validation system.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PatternConflictDetector:
    """
    Detector for conflicts between different pattern signals.
    
    This detector specializes in:
    - Directional conflict detection (continuation vs reversal)
    - Timeline conflict analysis
    - Pattern priority assessment
    - Conflict resolution recommendations
    """
    
    def __init__(self):
        self.name = "pattern_conflict_detector"
        self.description = "Detects and resolves conflicts between pattern signals"
        self.version = "1.0.0"
    
    def detect_pattern_conflicts(self, patterns: List[Dict[str, Any]], validation_scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect conflicts between detected patterns.
        
        Args:
            patterns: List of detected patterns
            validation_scores: Validation scores from cross-validation
            
        Returns:
            Dictionary containing conflict analysis results
        """
        try:
            logger.info(f"[CONFLICT_DETECTOR] Analyzing conflicts for {len(patterns)} patterns")
            
            if len(patterns) < 2:
                return self._build_no_conflicts_result(len(patterns))
            
            conflicts = []
            
            # 1. Directional Conflicts (continuation vs reversal)
            directional_conflicts = self._detect_directional_conflicts(patterns)
            conflicts.extend(directional_conflicts)
            
            # 2. Timeline Conflicts (overlapping patterns)
            timeline_conflicts = self._detect_timeline_conflicts(patterns)
            conflicts.extend(timeline_conflicts)
            
            # 3. Pattern Type Conflicts (multiple patterns of same type)
            type_conflicts = self._detect_pattern_type_conflicts(patterns)
            conflicts.extend(type_conflicts)
            
            # 4. Calculate overall conflict severity
            conflict_severity = self._calculate_conflict_severity(conflicts, patterns)
            
            # 5. Generate conflict resolution strategy
            resolution_strategy = self._generate_resolution_strategy(conflicts, patterns, validation_scores)
            
            result = {
                'total_conflicts': len(conflicts),
                'conflict_types': {
                    'directional': len(directional_conflicts),
                    'timeline': len(timeline_conflicts),
                    'pattern_type': len(type_conflicts)
                },
                'conflicts': conflicts,
                'conflict_severity': conflict_severity,
                'pattern_coherence': self._assess_pattern_coherence(conflict_severity),
                'resolution_strategy': resolution_strategy,
                'conflict_summary': self._generate_conflict_summary(conflicts, patterns)
            }
            
            logger.info(f"[CONFLICT_DETECTOR] Found {len(conflicts)} conflicts with {conflict_severity:.2f} severity")
            return result
            
        except Exception as e:
            logger.error(f"[CONFLICT_DETECTOR] Conflict detection failed: {e}")
            return {'error': str(e)}
    
    def _detect_directional_conflicts(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect conflicts between continuation and reversal patterns"""
        try:
            conflicts = []
            
            # Separate patterns by type
            continuations = [p for p in patterns if p.get('pattern_type') == 'continuation']
            reversals = [p for p in patterns if p.get('pattern_type') == 'reversal']
            consolidations = [p for p in patterns if p.get('pattern_type') == 'consolidation']
            
            # Check for continuation vs reversal conflicts
            if continuations and reversals:
                # Calculate strength of each side
                continuation_strength = self._calculate_pattern_group_strength(continuations)
                reversal_strength = self._calculate_pattern_group_strength(reversals)
                
                conflict = {
                    'conflict_type': 'directional_conflict',
                    'severity': 'high',
                    'description': f'{len(continuations)} continuation vs {len(reversals)} reversal patterns',
                    'conflicting_groups': {
                        'continuations': [p['pattern_name'] for p in continuations],
                        'reversals': [p['pattern_name'] for p in reversals]
                    },
                    'group_strengths': {
                        'continuation_strength': continuation_strength,
                        'reversal_strength': reversal_strength
                    },
                    'resolution_suggestion': 'favor_stronger_group',
                    'dominant_signal': 'continuation' if continuation_strength > reversal_strength else 'reversal'
                }
                conflicts.append(conflict)
            
            # Check for consolidation with strong directional signals
            if consolidations and (continuations or reversals):
                directional_count = len(continuations) + len(reversals)
                if directional_count >= 2:
                    conflict = {
                        'conflict_type': 'consolidation_vs_directional',
                        'severity': 'medium',
                        'description': f'{len(consolidations)} consolidation vs {directional_count} directional patterns',
                        'conflicting_patterns': {
                            'consolidations': [p['pattern_name'] for p in consolidations],
                            'directional': [p['pattern_name'] for p in continuations + reversals]
                        },
                        'resolution_suggestion': 'evaluate_timeframes',
                        'interpretation': 'Possible breakout preparation vs continuation'
                    }
                    conflicts.append(conflict)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"[CONFLICT_DETECTOR] Directional conflict detection failed: {e}")
            return []
    
    def _detect_timeline_conflicts(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect timeline conflicts between overlapping patterns"""
        try:
            conflicts = []
            
            # Group patterns by approximate timeline
            for i, pattern1 in enumerate(patterns):
                for j, pattern2 in enumerate(patterns[i+1:], i+1):
                    
                    # Check if patterns might overlap in time
                    timeline_conflict = self._check_timeline_overlap(pattern1, pattern2)
                    
                    if timeline_conflict:
                        # Check if they give conflicting signals
                        signal_conflict = self._check_signal_conflict(pattern1, pattern2)
                        
                        if signal_conflict:
                            conflict = {
                                'conflict_type': 'timeline_conflict',
                                'severity': signal_conflict['severity'],
                                'description': f"Overlapping {pattern1['pattern_name']} and {pattern2['pattern_name']}",
                                'conflicting_patterns': [pattern1['pattern_name'], pattern2['pattern_name']],
                                'pattern_details': {
                                    'pattern1': {
                                        'name': pattern1['pattern_name'],
                                        'type': pattern1.get('pattern_type', 'unknown'),
                                        'completion': pattern1.get('completion_percentage', 0),
                                        'reliability': pattern1.get('reliability', 'unknown')
                                    },
                                    'pattern2': {
                                        'name': pattern2['pattern_name'],
                                        'type': pattern2.get('pattern_type', 'unknown'),
                                        'completion': pattern2.get('completion_percentage', 0),
                                        'reliability': pattern2.get('reliability', 'unknown')
                                    }
                                },
                                'resolution_suggestion': signal_conflict['resolution'],
                                'priority_pattern': signal_conflict['priority_pattern']
                            }
                            conflicts.append(conflict)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"[CONFLICT_DETECTOR] Timeline conflict detection failed: {e}")
            return []
    
    def _detect_pattern_type_conflicts(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect conflicts from multiple patterns of same type"""
        try:
            conflicts = []
            
            # Group patterns by name
            pattern_groups = {}
            for pattern in patterns:
                name = pattern.get('pattern_name', 'unknown')
                if name not in pattern_groups:
                    pattern_groups[name] = []
                pattern_groups[name].append(pattern)
            
            # Check for multiple instances of same pattern type
            for pattern_name, pattern_list in pattern_groups.items():
                if len(pattern_list) > 2:  # More than 2 of same pattern is suspicious
                    # Calculate quality variations
                    completions = [p.get('completion_percentage', 0) for p in pattern_list]
                    reliabilities = [p.get('reliability', 'unknown') for p in pattern_list]
                    
                    completion_variation = np.std(completions) if len(completions) > 1 else 0
                    
                    if completion_variation > 15:  # High variation in completion
                        conflict = {
                            'conflict_type': 'pattern_duplication',
                            'severity': 'medium',
                            'description': f'{len(pattern_list)} instances of {pattern_name} with high variation',
                            'pattern_instances': len(pattern_list),
                            'completion_range': f"{min(completions):.1f}% - {max(completions):.1f}%",
                            'reliability_mix': list(set(reliabilities)),
                            'resolution_suggestion': 'select_highest_quality',
                            'quality_variation': completion_variation
                        }
                        conflicts.append(conflict)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"[CONFLICT_DETECTOR] Pattern type conflict detection failed: {e}")
            return []
    
    def _calculate_pattern_group_strength(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate overall strength of a pattern group"""
        try:
            if not patterns:
                return 0.0
            
            total_strength = 0.0
            for pattern in patterns:
                # Factors that contribute to pattern strength
                completion = pattern.get('completion_percentage', 0) / 100.0
                reliability_map = {'high': 1.0, 'medium': 0.7, 'low': 0.4, 'unknown': 0.3}
                reliability = reliability_map.get(pattern.get('reliability', 'unknown'), 0.3)
                
                pattern_strength = (completion * 0.6 + reliability * 0.4)
                total_strength += pattern_strength
            
            return total_strength / len(patterns)
            
        except Exception:
            return 0.5
    
    def _check_timeline_overlap(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> bool:
        """Check if two patterns might overlap in timeline"""
        try:
            # For now, assume patterns detected in same analysis might overlap
            # In a full implementation, this would check actual start/end dates
            start1 = pattern1.get('start_date', '')
            start2 = pattern2.get('start_date', '')
            
            # Simple heuristic: if start dates are within 7 days, consider potential overlap
            if start1 and start2:
                # This is a simplified check - real implementation would parse dates
                return abs(hash(start1) - hash(start2)) % 30 < 7
            
            return True  # Assume potential overlap if we can't determine
            
        except Exception:
            return True
    
    def _check_signal_conflict(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if two patterns give conflicting signals"""
        try:
            type1 = pattern1.get('pattern_type', 'unknown')
            type2 = pattern2.get('pattern_type', 'unknown')
            
            # Direct type conflicts
            if (type1 == 'continuation' and type2 == 'reversal') or (type1 == 'reversal' and type2 == 'continuation'):
                # Determine priority based on reliability and completion
                strength1 = self._calculate_pattern_strength(pattern1)
                strength2 = self._calculate_pattern_strength(pattern2)
                
                return {
                    'severity': 'high',
                    'resolution': 'favor_stronger_pattern',
                    'priority_pattern': pattern1['pattern_name'] if strength1 > strength2 else pattern2['pattern_name']
                }
            
            # Same type but different implications (e.g., different triangle types)
            if type1 == type2 and pattern1.get('pattern_name') != pattern2.get('pattern_name'):
                return {
                    'severity': 'medium',
                    'resolution': 'evaluate_quality_metrics',
                    'priority_pattern': None
                }
            
            return None
            
        except Exception:
            return None
    
    def _calculate_pattern_strength(self, pattern: Dict[str, Any]) -> float:
        """Calculate individual pattern strength"""
        try:
            completion = pattern.get('completion_percentage', 0) / 100.0
            reliability_map = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
            reliability = reliability_map.get(pattern.get('reliability', 'unknown'), 0.3)
            
            return completion * 0.6 + reliability * 0.4
            
        except Exception:
            return 0.3
    
    def _calculate_conflict_severity(self, conflicts: List[Dict[str, Any]], patterns: List[Dict[str, Any]]) -> float:
        """Calculate overall conflict severity score"""
        try:
            if not conflicts:
                return 0.0
            
            severity_weights = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
            total_severity = 0.0
            
            for conflict in conflicts:
                severity = conflict.get('severity', 'low')
                weight = severity_weights.get(severity, 0.3)
                total_severity += weight
            
            # Normalize by number of patterns
            max_possible_severity = len(patterns) * 1.0
            normalized_severity = min(1.0, total_severity / max_possible_severity) if max_possible_severity > 0 else 0.0
            
            return normalized_severity
            
        except Exception:
            return 0.5
    
    def _assess_pattern_coherence(self, conflict_severity: float) -> str:
        """Assess overall pattern coherence based on conflict severity"""
        if conflict_severity <= 0.2:
            return 'high'
        elif conflict_severity <= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _generate_resolution_strategy(self, conflicts: List[Dict[str, Any]], patterns: List[Dict[str, Any]], validation_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy for resolving pattern conflicts"""
        try:
            if not conflicts:
                return {'strategy': 'no_conflicts', 'action': 'proceed_with_all_patterns'}
            
            high_conflicts = [c for c in conflicts if c.get('severity') == 'high']
            medium_conflicts = [c for c in conflicts if c.get('severity') == 'medium']
            
            strategy = {
                'primary_strategy': 'unknown',
                'confidence_adjustment': 0.0,
                'recommended_actions': [],
                'pattern_priorities': {},
                'warnings': []
            }
            
            if high_conflicts:
                # High conflicts require significant confidence reduction
                strategy['primary_strategy'] = 'resolve_high_conflicts'
                strategy['confidence_adjustment'] = -0.2
                strategy['recommended_actions'].append('Prioritize higher quality patterns')
                strategy['recommended_actions'].append('Reduce overall confidence due to signal conflicts')
                
                # For directional conflicts, recommend the stronger side
                for conflict in high_conflicts:
                    if conflict.get('conflict_type') == 'directional_conflict':
                        dominant = conflict.get('dominant_signal', 'unknown')
                        strategy['recommended_actions'].append(f'Favor {dominant} patterns based on strength analysis')
            
            elif medium_conflicts:
                # Medium conflicts require moderate confidence reduction
                strategy['primary_strategy'] = 'monitor_medium_conflicts'
                strategy['confidence_adjustment'] = -0.1
                strategy['recommended_actions'].append('Monitor conflicting patterns closely')
                strategy['recommended_actions'].append('Seek additional confirmation signals')
            
            else:
                # Low conflicts
                strategy['primary_strategy'] = 'minimal_adjustment'
                strategy['confidence_adjustment'] = -0.05
                strategy['recommended_actions'].append('Minor confidence adjustment for low-level conflicts')
            
            # Add pattern-specific priorities
            for pattern in patterns:
                strength = self._calculate_pattern_strength(pattern)
                strategy['pattern_priorities'][pattern['pattern_name']] = strength
            
            return strategy
            
        except Exception as e:
            logger.error(f"[CONFLICT_DETECTOR] Resolution strategy generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_conflict_summary(self, conflicts: List[Dict[str, Any]], patterns: List[Dict[str, Any]]) -> str:
        """Generate human-readable conflict summary"""
        try:
            if not conflicts:
                return f"No conflicts detected among {len(patterns)} patterns. Patterns show good coherence."
            
            high_conflicts = len([c for c in conflicts if c.get('severity') == 'high'])
            medium_conflicts = len([c for c in conflicts if c.get('severity') == 'medium'])
            low_conflicts = len([c for c in conflicts if c.get('severity') == 'low'])
            
            summary = f"Detected {len(conflicts)} conflicts among {len(patterns)} patterns: "
            
            conflict_parts = []
            if high_conflicts > 0:
                conflict_parts.append(f"{high_conflicts} high severity")
            if medium_conflicts > 0:
                conflict_parts.append(f"{medium_conflicts} medium severity")
            if low_conflicts > 0:
                conflict_parts.append(f"{low_conflicts} low severity")
            
            summary += ", ".join(conflict_parts) + "."
            
            # Add specific conflict types
            directional = len([c for c in conflicts if c.get('conflict_type') == 'directional_conflict'])
            if directional > 0:
                summary += f" Includes {directional} directional conflict(s) requiring resolution."
            
            return summary
            
        except Exception:
            return f"Conflict analysis completed for {len(patterns)} patterns with {len(conflicts)} issues identified."
    
    def _build_no_conflicts_result(self, pattern_count: int) -> Dict[str, Any]:
        """Build result for cases with no conflicts"""
        return {
            'total_conflicts': 0,
            'conflict_types': {'directional': 0, 'timeline': 0, 'pattern_type': 0},
            'conflicts': [],
            'conflict_severity': 0.0,
            'pattern_coherence': 'high',
            'resolution_strategy': {
                'strategy': 'no_conflicts',
                'action': 'proceed_with_all_patterns',
                'confidence_adjustment': 0.0
            },
            'conflict_summary': f"No conflicts detected among {pattern_count} pattern(s). Patterns show good coherence."
        }