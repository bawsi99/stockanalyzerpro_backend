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
                if pattern.get("head_idx") is not None:
                    advanced_patterns["head_and_shoulders"].append({
                        "pattern_type": "head_and_shoulders",
                        "start_index": pattern["start_idx"],
                        "end_index": pattern["end_idx"],
                        "head_index": pattern["head_idx"],
                        "left_shoulder_index": pattern["left_shoulder_idx"],
                        "right_shoulder_index": pattern["right_shoulder_idx"],
                        "neckline": {
                            "level": float(pattern["neckline_value"])
                        },
                        "target": float(pattern.get("target", 0.0)),
                        "completion": float(pattern.get("completion", 0.0)),
                        "quality_score": float(pattern.get("quality_score", 0.0)),
                        "confidence": float(pattern.get("confidence", 0.0))
                    })
            
            # Inverse Head and Shoulders patterns
            ihs_patterns = PatternRecognition.detect_inverse_head_and_shoulders(data['close'])
            logger.debug(f"Inverse Head and Shoulders patterns detected: {len(ihs_patterns)}")
            advanced_patterns["inverse_head_and_shoulders"] = []
            for pattern in ihs_patterns:
                if pattern.get("head_idx") is not None:
                    advanced_patterns["inverse_head_and_shoulders"].append({
                        "pattern_type": "inverse_head_and_shoulders",
                        "start_index": pattern["start_idx"],
                        "end_index": pattern["end_idx"],
                        "head_index": pattern["head_idx"],
                        "left_shoulder_index": pattern["left_shoulder_idx"],
                        "right_shoulder_index": pattern["right_shoulder_idx"],
                        "neckline": {
                            "level": float(pattern["neckline_value"])
                        },
                        "target": float(pattern.get("target", 0.0)),
                        "completion": float(pattern.get("completion", 0.0)),
                        "quality_score": float(pattern.get("quality_score", 0.0)),
                        "confidence": float(pattern.get("confidence", 0.0))
                    })
            
            # Cup and Handle patterns
            ch_patterns = PatternRecognition.detect_cup_and_handle(data['close'])
            logger.debug(f"Cup and Handle patterns detected: {len(ch_patterns)}")
            advanced_patterns["cup_and_handle"] = []
            for pattern in ch_patterns:
                advanced_patterns["cup_and_handle"].append({
                    "pattern_type": "cup_and_handle",
                    "start_index": pattern["start_idx"],
                    "end_index": pattern["end_idx"],
                    "left_rim_index": pattern.get("left_rim_idx", 0),
                    "right_rim_index": pattern.get("right_rim_idx", 0),
                    "cup_bottom_index": pattern.get("bottom_idx", 0),
                    "handle_bottom_index": pattern.get("handle_bottom_idx", 0),
                    "rim_level": float(pattern.get("rim_level", 0.0)),
                    "target": float(pattern.get("target", 0.0)),
                    "completion": float(pattern.get("completion", 0.0)),
                    "quality_score": float(pattern.get("quality_score", 0.0)),
                    "confidence": float(pattern.get("confidence", 0.0))
                })
            
            # Triple Tops patterns
            tt_patterns = PatternRecognition.detect_triple_top(data['close'])
            logger.debug(f"Triple Tops patterns detected: {len(tt_patterns)}")
            advanced_patterns["triple_tops"] = []
            for pattern in tt_patterns:
                if pattern.get("peaks") is not None:
                    advanced_patterns["triple_tops"].append({
                        "pattern_type": "triple_top",
                        "start_index": pattern["start_idx"],
                        "end_index": pattern["end_idx"],
                        "peaks": [{"index": idx, "price": float(price)} for idx, price in pattern["peaks"]],
                        "support_level": float(pattern.get("support_level", 0.0)),
                        "target": float(pattern.get("target", 0.0)),
                        "completion": float(pattern.get("completion", 0.0)),
                        "quality_score": float(pattern.get("quality_score", 0.0)),
                        "confidence": float(pattern.get("confidence", 0.0))
                    })
            
            # Triple Bottoms patterns
            tb_patterns = PatternRecognition.detect_triple_bottom(data['close'])
            logger.debug(f"Triple Bottoms patterns detected: {len(tb_patterns)}")
            advanced_patterns["triple_bottoms"] = []
            for pattern in tb_patterns:
                if pattern.get("lows") is not None:
                    advanced_patterns["triple_bottoms"].append({
                        "pattern_type": "triple_bottom",
                        "start_index": pattern["start_idx"],
                        "end_index": pattern["end_idx"],
                        "lows": [{"index": idx, "price": float(price)} for idx, price in pattern["lows"]],
                        "resistance_level": float(pattern.get("resistance_level", 0.0)),
                        "target": float(pattern.get("target", 0.0)),
                        "completion": float(pattern.get("completion", 0.0)),
                        "quality_score": float(pattern.get("quality_score", 0.0)),
                        "confidence": float(pattern.get("confidence", 0.0))
                    })
            
            # Wedge patterns
            wedge_patterns = PatternRecognition.detect_wedge(data)
            logger.debug(f"Wedge patterns detected: {len(wedge_patterns)}")
            advanced_patterns["wedge_patterns"] = []
            for pattern in wedge_patterns:
                advanced_patterns["wedge_patterns"].append({
                    "pattern_type": "wedge",
                    "start_index": pattern["start_idx"],
                    "end_index": pattern["end_idx"],
                    "type": pattern.get("type", "rising"),
                    "upper_line": {
                        "slope": float(pattern.get("upper_slope", 0.0)),
                        "intercept": float(pattern.get("upper_intercept", 0.0))
                    },
                    "lower_line": {
                        "slope": float(pattern.get("lower_slope", 0.0)),
                        "intercept": float(pattern.get("lower_intercept", 0.0))
                    },
                    "target": float(pattern.get("target", 0.0)),
                    "completion": float(pattern.get("completion", 0.0)),
                    "quality_score": float(pattern.get("quality_score", 0.0)),
                    "confidence": float(pattern.get("confidence", 0.0))
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
                if "peak1_idx" in dt and "peak2_idx" in dt:
                    double_tops.append({
                        "peak1": {
                            "date": str(data.index[dt["peak1_idx"]]),
                            "price": float(data['close'].iloc[dt["peak1_idx"]]),
                            "index": dt["peak1_idx"]
                        },
                        "peak2": {
                            "date": str(data.index[dt["peak2_idx"]]),
                            "price": float(data['close'].iloc[dt["peak2_idx"]]),
                            "index": dt["peak2_idx"]
                        },
                        "trough": {
                            "date": str(data.index[dt["trough_idx"]]) if "trough_idx" in dt else None,
                            "price": float(data['close'].iloc[dt["trough_idx"]]) if "trough_idx" in dt else None,
                            "index": dt["trough_idx"] if "trough_idx" in dt else None
                        },
                        "neckline": float(dt.get("neckline_value", 0.0)),
                        "target": float(dt.get("target", 0.0)),
                        "confidence": float(dt.get("confidence", 0.7))
                    })
            
            # Double Bottoms
            double_bottom_indices = PatternRecognition.detect_double_bottom(data['close'])
            double_bottoms = []
            for db in double_bottom_indices:
                if "trough1_idx" in db and "trough2_idx" in db:
                    double_bottoms.append({
                        "trough1": {
                            "date": str(data.index[db["trough1_idx"]]),
                            "price": float(data['close'].iloc[db["trough1_idx"]]),
                            "index": db["trough1_idx"]
                        },
                        "trough2": {
                            "date": str(data.index[db["trough2_idx"]]),
                            "price": float(data['close'].iloc[db["trough2_idx"]]),
                            "index": db["trough2_idx"]
                        },
                        "peak": {
                            "date": str(data.index[db["peak_idx"]]) if "peak_idx" in db else None,
                            "price": float(data['close'].iloc[db["peak_idx"]]) if "peak_idx" in db else None,
                            "index": db["peak_idx"] if "peak_idx" in db else None
                        },
                        "neckline": float(db.get("neckline_value", 0.0)),
                        "target": float(db.get("target", 0.0)),
                        "confidence": float(db.get("confidence", 0.7))
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
