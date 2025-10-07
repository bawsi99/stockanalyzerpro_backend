#!/usr/bin/env python3
"""
Parallel Pattern Detection Module

This module provides functionality to execute pattern detection in parallel with
other asynchronous tasks in the ASYNC-OPTIMIZED-ENHANCED framework.
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelPatternDetection:
    """
    Handles parallel pattern detection as part of the enhanced analysis pipeline.
    Can be integrated with the existing async task framework.
    """
    
    @staticmethod
    async def detect_patterns_async(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Async wrapper for pattern detection to run in parallel with other tasks.
        
        Args:
            data: Price data as pandas DataFrame
            
        Returns:
            Dict containing detected patterns
        """
        # Run the CPU-bound pattern detection in a separate thread pool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, ParallelPatternDetection._detect_patterns_sync, data
        )
    
    @staticmethod
    def _detect_patterns_sync(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Synchronous pattern detection function that runs in a thread pool.
        
        Args:
            data: Price data as pandas DataFrame
            
        Returns:
            Dict containing detected patterns
        """
        try:
            start_time = time.time()
            logger.debug(f"Starting pattern detection for {len(data)} data points")
            
            # Import pattern recognition
            from patterns.recognition import PatternRecognition
            
            # Dictionary to store all detected patterns
            patterns = {}
            advanced_patterns = {}
            
            # Head and Shoulders patterns
            hs_patterns = PatternRecognition.detect_head_and_shoulders(data['close'])
            logger.debug(f"Head and Shoulders patterns detected: {len(hs_patterns)}")
            advanced_patterns["head_and_shoulders"] = []
            for pattern in hs_patterns:
                if pattern.get("head") is not None:
                    # Extract indices from the nested structure
                    left_shoulder_idx = pattern["left_shoulder"]["index"]
                    head_idx = pattern["head"]["index"]
                    right_shoulder_idx = pattern["right_shoulder"]["index"]
                    neckline_idx = pattern["neckline"]["index"]
                    
                    # Calculate start and end indices
                    start_idx = min(left_shoulder_idx, head_idx, right_shoulder_idx)
                    end_idx = max(left_shoulder_idx, head_idx, right_shoulder_idx)
                    
                    advanced_patterns["head_and_shoulders"].append({
                        "pattern_type": "head_and_shoulders",
                        "start_index": start_idx,
                        "end_index": end_idx,
                        "head_index": head_idx,
                        "left_shoulder_index": left_shoulder_idx,
                        "right_shoulder_index": right_shoulder_idx,
                        "neckline": {
                            "level": float(pattern["neckline"]["level"])
                        },
                        "target": float(pattern.get("target", 0.0)),
                        "completion": float(pattern.get("completion_status", 0.0) == "completed"),
                        "quality_score": float(pattern.get("quality_score", 0.0)),
                        "confidence": float(pattern.get("quality_score", 0.0) / 100.0)  # Convert quality score to confidence
                    })
            
            # Inverse Head and Shoulders patterns
            ihs_patterns = PatternRecognition.detect_inverse_head_and_shoulders(data['close'])
            logger.debug(f"Inverse Head and Shoulders patterns detected: {len(ihs_patterns)}")
            advanced_patterns["inverse_head_and_shoulders"] = []
            for pattern in ihs_patterns:
                if pattern.get("head") is not None:
                    # Extract indices from the nested structure
                    left_shoulder_idx = pattern["left_shoulder"]["index"]
                    head_idx = pattern["head"]["index"]
                    right_shoulder_idx = pattern["right_shoulder"]["index"]
                    neckline_idx = pattern["neckline"]["index"]
                    
                    # Calculate start and end indices
                    start_idx = min(left_shoulder_idx, head_idx, right_shoulder_idx)
                    end_idx = max(left_shoulder_idx, head_idx, right_shoulder_idx)
                    
                    advanced_patterns["inverse_head_and_shoulders"].append({
                        "pattern_type": "inverse_head_and_shoulders",
                        "start_index": start_idx,
                        "end_index": end_idx,
                        "head_index": head_idx,
                        "left_shoulder_index": left_shoulder_idx,
                        "right_shoulder_index": right_shoulder_idx,
                        "neckline": {
                            "level": float(pattern["neckline"]["level"])
                        },
                        "target": float(pattern.get("target", 0.0)),
                        "completion": float(pattern.get("completion_status", 0.0) == "completed"),
                        "quality_score": float(pattern.get("quality_score", 0.0)),
                        "confidence": float(pattern.get("quality_score", 0.0) / 100.0)  # Convert quality score to confidence
                    })
            
            # Cup and Handle patterns
            ch_patterns = PatternRecognition.detect_cup_and_handle(data['close'])
            logger.debug(f"Cup and Handle patterns detected: {len(ch_patterns)}")
            advanced_patterns["cup_and_handle"] = []
            for pattern in ch_patterns:
                # Extract indices from the nested structure
                cup_start_idx = pattern["cup"]["start_index"]
                cup_end_idx = pattern["cup"]["end_index"]
                handle_start_idx = pattern["handle"]["start_index"]
                handle_end_idx = pattern["handle"]["end_index"]
                cup_bottom_idx = pattern["cup"]["low_index"]
                handle_bottom_idx = pattern["handle"]["start_index"]  # Handle start is usually the handle bottom
                
                # Calculate overall start and end indices
                start_idx = min(cup_start_idx, handle_start_idx)
                end_idx = max(cup_end_idx, handle_end_idx)
                
                advanced_patterns["cup_and_handle"].append({
                    "pattern_type": "cup_and_handle",
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "left_rim_index": cup_start_idx,
                    "right_rim_index": cup_end_idx,
                    "cup_bottom_index": cup_bottom_idx,
                    "handle_bottom_index": handle_bottom_idx,
                    "rim_level": float(pattern["cup"]["start_price"]),  # Use cup start price as rim level
                    "target": float(pattern.get("target", 0.0)),
                    "completion": float(pattern.get("completion_status", 0.0) == "completed"),
                    "quality_score": float(pattern.get("quality_score", 0.0)),
                    "confidence": float(pattern.get("quality_score", 0.0) / 100.0)  # Convert quality score to confidence
                })
            
            # Triple Tops patterns
            tt_patterns = PatternRecognition.detect_triple_top(data['close'])
            logger.debug(f"Triple Tops patterns detected: {len(tt_patterns)}")
            advanced_patterns["triple_tops"] = []
            for pattern in tt_patterns:
                if pattern.get("lows") is not None:
                    # Extract indices from the nested structure
                    lows = pattern["lows"]
                    if len(lows) >= 3:
                        start_idx = min(low["index"] for low in lows)
                        end_idx = max(low["index"] for low in lows)
                        
                        advanced_patterns["triple_tops"].append({
                            "pattern_type": "triple_top",
                            "start_index": start_idx,
                            "end_index": end_idx,
                            "peaks": [{"index": low["index"], "price": float(low["price"])} for low in lows],
                            "support_level": float(pattern.get("support_level", 0.0)),
                            "target": float(pattern.get("target", 0.0)),
                            "completion": float(pattern.get("completion_status", 0.0) == "completed"),
                            "quality_score": float(pattern.get("quality_score", 0.0)),
                            "confidence": float(pattern.get("quality_score", 0.0) / 100.0)  # Convert quality score to confidence
                        })
            
            # Triple Bottoms patterns
            tb_patterns = PatternRecognition.detect_triple_bottom(data['close'])
            logger.debug(f"Triple Bottoms patterns detected: {len(tb_patterns)}")
            advanced_patterns["triple_bottoms"] = []
            for pattern in tb_patterns:
                if pattern.get("lows") is not None:
                    # Extract indices from the nested structure
                    lows = pattern["lows"]
                    if len(lows) >= 3:
                        start_idx = min(low["index"] for low in lows)
                        end_idx = max(low["index"] for low in lows)
                        
                        advanced_patterns["triple_bottoms"].append({
                            "pattern_type": "triple_bottom",
                            "start_index": start_idx,
                            "end_index": end_idx,
                            "lows": [{"index": low["index"], "price": float(low["price"])} for low in lows],
                            "resistance_level": float(pattern.get("resistance_level", 0.0)),
                            "target": float(pattern.get("target", 0.0)),
                            "completion": float(pattern.get("completion_status", 0.0) == "completed"),
                            "quality_score": float(pattern.get("quality_score", 0.0)),
                            "confidence": float(pattern.get("quality_score", 0.0) / 100.0)  # Convert quality score to confidence
                        })
            
            # Wedge patterns
            wedge_patterns = PatternRecognition.detect_wedge_patterns(data['close'])
            logger.debug(f"Wedge patterns detected: {len(wedge_patterns)}")
            advanced_patterns["wedge_patterns"] = []
            for pattern in wedge_patterns:
                advanced_patterns["wedge_patterns"].append({
                    "pattern_type": "wedge",
                    "start_index": pattern["start_index"],
                    "end_index": pattern["end_index"],
                    "type": pattern.get("type", "rising"),
                    "upper_line": {
                        "slope": float(pattern.get("slope_highs", 0.0)),
                        "intercept": float(pattern.get("start_price", 0.0))  # Use start price as intercept approximation
                    },
                    "lower_line": {
                        "slope": float(pattern.get("slope_lows", 0.0)),
                        "intercept": float(pattern.get("start_price", 0.0))  # Use start price as intercept approximation
                    },
                    "target": float(pattern.get("target", 0.0)),
                    "completion": float(pattern.get("completion_status", 0.0) == "completed"),
                    "quality_score": float(pattern.get("quality_score", 0.0)),
                    "confidence": float(pattern.get("quality_score", 0.0) / 100.0)  # Convert quality score to confidence
                })
            
            # Channel patterns (if implemented)
            advanced_patterns["channel_patterns"] = []
            
            # Add triangles
            triangle_indices = PatternRecognition.detect_triangle(data['close'])
            triangles = []
            for tri in triangle_indices:
                vertices = []
                for idx in tri:
                    date = str(data.index[idx])
                    price = float(data['close'].iloc[idx])
                    vertices.append({"date": date, "price": price, "index": idx})
                
                # Calculate pattern quality based on vertex count
                quality = min(1.0, len(vertices) / 3)
                
                triangles.append({
                    "vertices": vertices,
                    "type": "ascending" if vertices[0]["price"] < vertices[-1]["price"] else "descending",
                    "quality": quality
                })
            
            # Double Tops
            double_top_indices = PatternRecognition.detect_double_top(data['close'])
            double_tops = []
            for dt in double_top_indices:
                if isinstance(dt, tuple) and len(dt) == 2:
                    peak1_idx, peak2_idx = dt
                    # Find the trough between the peaks
                    trough_idx = None
                    if peak1_idx < peak2_idx:
                        trough_data = data['close'].iloc[peak1_idx:peak2_idx+1]
                        if not trough_data.empty:
                            trough_ts = trough_data.idxmin()
                            trough_idx = int(data.index.get_loc(trough_ts))
                    
                    double_tops.append({
                        "peak1": {
                            "date": str(data.index[peak1_idx]),
                            "price": float(data['close'].iloc[peak1_idx]),
                            "index": peak1_idx
                        },
                        "peak2": {
                            "date": str(data.index[peak2_idx]),
                            "price": float(data['close'].iloc[peak2_idx]),
                            "index": peak2_idx
                        },
                        "trough": {
                            "date": str(data.index[trough_idx]) if trough_idx is not None else None,
                            "price": float(data['close'].iloc[trough_idx]) if trough_idx is not None else None,
                            "index": trough_idx if trough_idx is not None else None
                        },
                        "neckline": float(data['close'].iloc[trough_idx]) if trough_idx is not None else 0.0,
                        "target": 0.0,  # Placeholder
                        "confidence": 0.7
                    })
            
            # Double Bottoms
            double_bottom_indices = PatternRecognition.detect_double_bottom(data['close'])
            double_bottoms = []
            for db in double_bottom_indices:
                if isinstance(db, tuple) and len(db) == 2:
                    trough1_idx, trough2_idx = db
                    # Find the peak between the troughs
                    peak_idx = None
                    if trough1_idx < trough2_idx:
                        peak_data = data['close'].iloc[trough1_idx:trough2_idx+1]
                        if not peak_data.empty:
                            peak_ts = peak_data.idxmax()
                            peak_idx = int(data.index.get_loc(peak_ts))
                    
                    double_bottoms.append({
                        "trough1": {
                            "date": str(data.index[trough1_idx]),
                            "price": float(data['close'].iloc[trough1_idx]),
                            "index": trough1_idx
                        },
                        "trough2": {
                            "date": str(data.index[trough2_idx]),
                            "price": float(data['close'].iloc[trough2_idx]),
                            "index": trough2_idx
                        },
                        "peak": {
                            "date": str(data.index[peak_idx]) if peak_idx is not None else None,
                            "price": float(data['close'].iloc[peak_idx]) if peak_idx is not None else None,
                            "index": peak_idx if peak_idx is not None else None
                        },
                        "neckline": float(data['close'].iloc[peak_idx]) if peak_idx is not None else 0.0,
                        "target": 0.0,  # Placeholder
                        "confidence": 0.7
                    })
            
            # Assemble the final patterns dictionary
            patterns["triangles"] = triangles
            patterns["double_tops"] = double_tops
            patterns["double_bottoms"] = double_bottoms
            patterns["advanced_patterns"] = advanced_patterns
            patterns["flags"] = []  # Placeholder for flag patterns
            patterns["support_resistance"] = {
                "support": [],
                "resistance": []
            }
            patterns["divergences"] = []
            patterns["volume_anomalies"] = []
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Pattern detection completed in {elapsed_time:.2f} seconds")
            
            # Count detected patterns
            total_patterns = (
                len(patterns["triangles"]) + 
                len(patterns["double_tops"]) + 
                len(patterns["double_bottoms"]) + 
                len(advanced_patterns["head_and_shoulders"]) +
                len(advanced_patterns["inverse_head_and_shoulders"]) +
                len(advanced_patterns["cup_and_handle"]) +
                len(advanced_patterns["triple_tops"]) +
                len(advanced_patterns["triple_bottoms"]) +
                len(advanced_patterns["wedge_patterns"]) +
                len(advanced_patterns["channel_patterns"])
            )
            
            logger.debug(f"Total patterns detected: {total_patterns}")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
            # Return empty pattern structure on error
            return {
                "triangles": [],
                "flags": [],
                "double_tops": [],
                "double_bottoms": [],
                "divergences": [],
                "volume_anomalies": [],
                "support_resistance": {"support": [], "resistance": []},
                "advanced_patterns": {
                    "head_and_shoulders": [],
                    "inverse_head_and_shoulders": [],
                    "cup_and_handle": [],
                    "triple_tops": [],
                    "triple_bottoms": [],
                    "wedge_patterns": [],
                    "channel_patterns": []
                }
            }

# Create a singleton instance
parallel_pattern_detection = ParallelPatternDetection()
