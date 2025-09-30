#!/usr/bin/env python3
"""
MTF Analysis Comprehensive Prompt Generator

Location: backend/agents/mtf_analysis/prompt_testing/mtf_comprehensive/multi_stock_test.py

What it does:
- For each symbol provided, it:
  1) Retrieves stock data across all MTF timeframes (1min → 1day)
  2) Runs the MTF agents integration manager to perform comprehensive multi-timeframe analysis
  3) Extracts the actual prompt/context that would be sent to the LLM
  4) Captures any images/charts that would be sent with the prompt
  5) Writes the prompt, context, and images to disk in a well-formatted structure
  6) Optionally calls the LLM and saves the response

It provides complete transparency into what the MTF analysis system sends to the LLM.

Usage:
  python backend/agents/mtf_analysis/prompt_testing/mtf_comprehensive/multi_stock_test.py \
    --symbols RELIANCE,TCS,INFY --exchange NSE --call-llm

Notes:
- This captures the ACTUAL prompts sent in production MTF analysis pipeline
- All computed values, indicators, and timeframe analyses are real
- Any charts/images generated are saved alongside prompts
- Use --call-llm to also capture LLM responses
"""

import os
import sys
import argparse
import re
import json
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Ensure backend/ is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Backend imports
from agents.mtf_analysis import mtf_agent_integration_manager
from ml.indicators.technical_indicators import TechnicalIndicators
from gemini.gemini_client import GeminiClient
from gemini.context_engineer import ContextEngineer, AnalysisType
from gemini.prompt_manager import PromptManager
from zerodha.client import ZerodhaDataClient

# For capturing images if charts are generated
try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available. Image saving will be limited.")


def normalize_prompt(text: str) -> str:
    """Remove dynamic values like timestamps and collapse whitespace for comparison."""
    if not isinstance(text, str):
        return ""
    s = text
    # Remove ISO datetimes
    s = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?", "<DATETIME>", s)
    # Remove epoch timestamps
    s = re.sub(r"\b\d{10}(?:\d{3})?\b", "<EPOCH>", s)
    # Normalize whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


async def fetch_stock_data(symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
    """
    Fetch stock data for all MTF timeframes.
    Returns dict of timeframe -> DataFrame
    """
    z = ZerodhaDataClient()
    
    # Timeframe configurations matching CoreMTFProcessor
    timeframe_configs = {
        '1min': {'interval': 'minute', 'period': 30},
        '5min': {'interval': '5minute', 'period': 60},
        '15min': {'interval': '15minute', 'period': 90},
        '30min': {'interval': '30minute', 'period': 120},
        '1hour': {'interval': '60minute', 'period': 180},
        '1day': {'interval': 'day', 'period': 365}
    }
    
    data = {}
    for tf, config in timeframe_configs.items():
        try:
            if hasattr(z, 'get_historical_data_async'):
                df = await z.get_historical_data_async(
                    symbol=symbol,
                    exchange=exchange,
                    interval=config['interval'],
                    period=config['period']
                )
            else:
                import asyncio
                loop = asyncio.get_event_loop()
                df = await loop.run_in_executor(
                    None, 
                    z.get_historical_data,
                    symbol, exchange, config['interval'], None, None, config['period']
                )
            
            if df is not None and not df.empty:
                data[tf] = df
                print(f"  ✓ Fetched {len(df)} records for {tf}")
        except Exception as e:
            print(f"  ✗ Failed to fetch {tf}: {e}")
    
    return data


def extract_mtf_context_for_llm(mtf_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the context that would be passed to the LLM from MTF analysis results.
    This mimics what ContextEngineer would structure.
    """
    context = {
        'analysis_type': 'multi_timeframe_analysis',
        'timestamp': datetime.now().isoformat(),
        'success': mtf_analysis.get('success', False),
        'symbol': mtf_analysis.get('symbol', ''),
        'exchange': mtf_analysis.get('exchange', ''),
    }
    
    # Timeframe analyses
    timeframe_analyses = mtf_analysis.get('timeframe_analyses', {})
    context['timeframe_summary'] = {}
    
    for tf, analysis in timeframe_analyses.items():
        if isinstance(analysis, dict):
            context['timeframe_summary'][tf] = {
                'trend': analysis.get('trend', 'unknown'),
                'confidence': analysis.get('confidence', 0.0),
                'data_points': analysis.get('data_points', 0),
                'key_indicators': analysis.get('key_indicators', {}),
                'signals': analysis.get('signals', {}),
                'support_levels': analysis.get('support_levels', []),
                'resistance_levels': analysis.get('resistance_levels', [])
            }
    
    # Cross-timeframe validation
    validation = mtf_analysis.get('cross_timeframe_validation', {})
    context['cross_timeframe_validation'] = {
        'consensus_trend': validation.get('consensus_trend', 'neutral'),
        'signal_strength': validation.get('signal_strength', 0.0),
        'confidence_score': validation.get('confidence_score', 0.0),
        'supporting_timeframes': validation.get('supporting_timeframes', []),
        'conflicting_timeframes': validation.get('conflicting_timeframes', []),
        'divergence_detected': validation.get('divergence_detected', False),
        'divergence_type': validation.get('divergence_type'),
        'key_conflicts': validation.get('key_conflicts', [])
    }
    
    # Summary
    summary = mtf_analysis.get('summary', {})
    context['mtf_summary'] = {
        'overall_signal': summary.get('overall_signal', 'neutral'),
        'confidence': summary.get('confidence', 0.0),
        'signal_alignment': summary.get('signal_alignment', 'unknown'),
        'risk_level': summary.get('risk_level', 'unknown'),
        'recommended_action': summary.get('recommended_action', 'hold')
    }
    
    # Agent insights (from distributed agents)
    agent_insights = mtf_analysis.get('agent_insights', {})
    context['agent_insights'] = {
        'total_agents_run': agent_insights.get('total_agents_run', 0),
        'successful_agents': agent_insights.get('successful_agents', 0),
        'failed_agents': agent_insights.get('failed_agents', 0),
        'confidence_score': agent_insights.get('confidence_score', 0.0),
        'fallback_used': agent_insights.get('fallback_used', False)
    }
    
    # Unified analysis (consensus across agents)
    unified = mtf_analysis.get('unified_analysis', {})
    if unified:
        context['unified_analysis'] = unified
    
    # Trading recommendations
    recommendations = mtf_analysis.get('trading_recommendations', [])
    if recommendations:
        context['trading_recommendations'] = recommendations
    
    return context


def build_production_knowledge_context(mtf_analysis: Dict[str, Any]) -> str:
    """
    Build the MTF context exactly as it appears in production within knowledge_context.
    
    In production (orchestrator.py), the MTF context is added to knowledge_context as:
    supplemental_blocks.append("MultiTimeframeContext:\n" + json.dumps(clean_for_json(mtf_context)))
    
    This is then passed to the LLM as part of the overall knowledge_context parameter.
    """
    from core.utils import clean_for_json
    
    # Remove the chart bytes before JSON serialization (charts are sent separately to LLM)
    mtf_for_json = mtf_analysis.copy()
    if 'chart' in mtf_for_json:
        # Save chart info but remove the actual bytes
        chart_info = mtf_for_json['chart']
        if isinstance(chart_info, dict):
            mtf_for_json['chart'] = {
                'type': chart_info.get('type'),
                'format': chart_info.get('format'),
                'size_bytes': chart_info.get('size_bytes'),
                'chart_type': chart_info.get('chart_type'),
                'symbol': chart_info.get('symbol'),
                'exchange': chart_info.get('exchange'),
                'note': 'Chart image sent separately to LLM'
            }
        else:
            # Legacy format - remove entirely
            del mtf_for_json['chart']
    
    # This is the EXACT format used in production (without chart bytes)
    mtf_json_block = "MultiTimeframeContext:\n" + json.dumps(clean_for_json(mtf_for_json), indent=2)
    
    return mtf_json_block


def build_mtf_prompt(symbol: str, exchange: str, mtf_context: Dict[str, Any]) -> str:
    """
    Build the actual prompt that would be sent to the LLM.
    
    IMPORTANT: In production, MTF analysis is NOT sent as a separate prompt.
    Instead, it's passed as a JSON block ("MultiTimeframeContext:") within the
    knowledge_context string of the main decision-making LLM call.
    
    This function shows how the MTF data would be formatted as part of that JSON block.
    """
    prompt_mgr = PromptManager()
    ctx_engineer = ContextEngineer()
    
    # Format the MTF context into a structured prompt context
    timeframe_summary = mtf_context.get('timeframe_summary', {})
    validation = mtf_context.get('cross_timeframe_validation', {})
    summary = mtf_context.get('mtf_summary', {})
    agent_insights = mtf_context.get('agent_insights', {})
    
    # Build structured context string
    context_str = f"""
MULTI-TIMEFRAME ANALYSIS FOR {symbol}

=== OVERALL SUMMARY ===
Signal: {summary.get('overall_signal', 'unknown').upper()}
Confidence: {summary.get('confidence', 0) * 100:.1f}%
Signal Alignment: {summary.get('signal_alignment', 'unknown')}
Risk Level: {summary.get('risk_level', 'unknown')}
Recommended Action: {summary.get('recommended_action', 'hold')}

=== CROSS-TIMEFRAME VALIDATION ===
Consensus Trend: {validation.get('consensus_trend', 'neutral')}
Signal Strength: {validation.get('signal_strength', 0) * 100:.1f}%
Confidence Score: {validation.get('confidence_score', 0) * 100:.1f}%
Supporting Timeframes: {', '.join(validation.get('supporting_timeframes', [])) or 'None'}
Conflicting Timeframes: {', '.join(validation.get('conflicting_timeframes', [])) or 'None'}
Divergence Detected: {'Yes' if validation.get('divergence_detected') else 'No'}
"""
    
    if validation.get('divergence_detected'):
        context_str += f"Divergence Type: {validation.get('divergence_type', 'unknown')}\n"
    
    if validation.get('key_conflicts'):
        context_str += f"Key Conflicts:\n"
        for conflict in validation.get('key_conflicts', []):
            context_str += f"  - {conflict}\n"
    
    context_str += "\n=== TIMEFRAME-BY-TIMEFRAME ANALYSIS ===\n\n"
    
    # Sort timeframes for consistent output
    sorted_timeframes = sorted(timeframe_summary.keys(), 
                               key=lambda x: ['1min', '5min', '15min', '30min', '1hour', '1day'].index(x) 
                               if x in ['1min', '5min', '15min', '30min', '1hour', '1day'] else 999)
    
    for tf in sorted_timeframes:
        analysis = timeframe_summary[tf]
        confidence = analysis.get('confidence', 0) * 100
        
        # Assign importance indicator
        if confidence >= 80:
            importance = "🔥 HIGH CONFIDENCE"
        elif confidence >= 60:
            importance = "⚡ MEDIUM-HIGH CONFIDENCE"
        elif confidence >= 40:
            importance = "📊 MEDIUM CONFIDENCE"
        else:
            importance = "⚠️ LOW CONFIDENCE"
        
        context_str += f"""
--- {tf} Timeframe ({importance}) ---
Trend: {analysis.get('trend', 'unknown')}
Confidence: {confidence:.1f}%
Data Points: {analysis.get('data_points', 0)}

Key Indicators:
"""
        key_indicators = analysis.get('key_indicators', {})
        if key_indicators:
            if 'rsi' in key_indicators:
                context_str += f"  • RSI: {key_indicators['rsi']}\n"
            if 'macd_signal' in key_indicators:
                context_str += f"  • MACD Signal: {key_indicators['macd_signal']}\n"
            if 'volume_status' in key_indicators:
                context_str += f"  • Volume Status: {key_indicators['volume_status']}\n"
            
            # Support/resistance levels are inside key_indicators
            support_levels = key_indicators.get('support_levels', [])
            resistance_levels = key_indicators.get('resistance_levels', [])
            
            if support_levels:
                context_str += f"  • Support Levels: {', '.join(f'{s:.2f}' for s in support_levels[:3])}\n"
            if resistance_levels:
                context_str += f"  • Resistance Levels: {', '.join(f'{r:.2f}' for r in resistance_levels[:3])}\n"
        
        signals = analysis.get('signals', {})
        if signals:
            context_str += f"  • Signals: {signals}\n"
        
        context_str += "\n"
    
    # Removed unified analysis section - it's redundant with OVERALL SUMMARY above
    # (contained duplicate trend, confidence, risk_level info plus empty fields)
    
    # Add trading recommendations if available
    recommendations = mtf_context.get('trading_recommendations', [])
    if recommendations:
        context_str += "\n=== TRADING RECOMMENDATIONS ===\n"
        for i, rec in enumerate(recommendations, 1):
            context_str += f"{i}. {rec}\n"
    
    # Try to use the optimized_mtf_comparison template if available
    try:
        # The prompt template should ask the LLM to synthesize the MTF analysis
        prompt = prompt_mgr.format_prompt("optimized_mtf_comparison", context=context_str)
    except Exception:
        # Fallback to a basic prompt if template not found
        prompt = f"""
{context_str}

TASK:
Analyze this comprehensive multi-timeframe analysis and provide:
1. Overall trend assessment across all timeframes
2. Key support and resistance levels to watch
3. Trading recommendations based on timeframe alignment
4. Risk assessment and conflict resolution
5. Entry/exit suggestions if signals are clear

Focus on synthesizing insights from all timeframes while prioritizing higher-confidence signals.
"""
    
    return prompt


def save_images_if_present(mtf_analysis: Dict[str, Any], output_dir: str, symbol: str, timestamp: str) -> Dict[str, str]:
    """
    Save any images/charts that are part of the MTF analysis.
    Returns dict of image_name -> saved_path
    """
    saved_images = {}
    
    if not HAS_PIL:
        return saved_images
    
    # Check for single chart (new MTF visualization format)
    # The integration manager stores chart as: mtf_analysis['chart'] = {'type': 'image_bytes', 'data': chart_bytes, ...}
    chart_data = mtf_analysis.get('chart')
    if chart_data:
        try:
            # Handle dict format with 'data' key (new format from integration_manager)
            if isinstance(chart_data, dict) and 'data' in chart_data:
                img_bytes = chart_data['data']
                if isinstance(img_bytes, bytes):
                    img = Image.open(io.BytesIO(img_bytes))
                    chart_type = chart_data.get('chart_type', 'mtf_comparison')
                    img_path = os.path.join(output_dir, f"chart_{chart_type}_{symbol}_{timestamp}.png")
                    img.save(img_path)
                    saved_images[chart_type] = img_path
                    size_kb = len(img_bytes) / 1024
                    print(f"  ✓ Saved MTF chart: {chart_type} ({size_kb:.1f} KB)")
            # Handle raw bytes format (legacy)
            elif isinstance(chart_data, bytes):
                img = Image.open(io.BytesIO(chart_data))
                img_path = os.path.join(output_dir, f"chart_mtf_comparison_{symbol}_{timestamp}.png")
                img.save(img_path)
                saved_images['mtf_comparison'] = img_path
                size_kb = len(chart_data) / 1024
                print(f"  ✓ Saved MTF chart ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"  ✗ Failed to save MTF chart: {e}")
            import traceback
            traceback.print_exc()
    
    # Also check for multiple charts in 'charts' key (for backward compatibility)
    charts = mtf_analysis.get('charts', {})
    for chart_name, chart_data in charts.items():
        try:
            # If chart_data is bytes
            if isinstance(chart_data, bytes):
                img = Image.open(io.BytesIO(chart_data))
                img_path = os.path.join(output_dir, f"chart_{chart_name}_{symbol}_{timestamp}.png")
                img.save(img_path)
                saved_images[chart_name] = img_path
                size_kb = len(chart_data) / 1024
                print(f"  ✓ Saved chart: {chart_name} ({size_kb:.1f} KB)")
            
            # If chart_data is a dict with 'data' key
            elif isinstance(chart_data, dict) and 'data' in chart_data:
                img_bytes = chart_data['data']
                if isinstance(img_bytes, bytes):
                    img = Image.open(io.BytesIO(img_bytes))
                    img_path = os.path.join(output_dir, f"chart_{chart_name}_{symbol}_{timestamp}.png")
                    img.save(img_path)
                    saved_images[chart_name] = img_path
                    size_kb = len(img_bytes) / 1024
                    print(f"  ✓ Saved chart: {chart_name} ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"  ✗ Failed to save chart {chart_name}: {e}")
    
    return saved_images


def write_results(
    out_dir: str,
    symbol: str,
    exchange: str,
    mtf_analysis: Dict[str, Any],
    mtf_context: Dict[str, Any],
    prompt: str,
    saved_images: Dict[str, str],
    llm_response: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Write all results to disk in a well-formatted structure.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    sym_safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", symbol)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Write the prompt file
    prompt_path = os.path.join(out_dir, f"prompt_{sym_safe}_{ts}.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("MTF ANALYSIS PROMPT FOR LLM\n")
        f.write("="*80 + "\n\n")
        f.write(f"Stock Symbol: {symbol}\n")
        f.write(f"Exchange: {exchange}\n")
        
        if metadata:
            f.write(f"Company: {metadata.get('company', 'N/A')}\n")
            f.write(f"Sector: {metadata.get('sector', 'N/A')}\n")
        
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Prompt Length: {len(prompt)} characters\n")
        
        # Analysis quality metrics
        agent_insights = mtf_analysis.get('agent_insights', {})
        f.write(f"Agents Run: {agent_insights.get('total_agents_run', 0)}\n")
        f.write(f"Agents Successful: {agent_insights.get('successful_agents', 0)}\n")
        f.write(f"Overall Confidence: {agent_insights.get('confidence_score', 0) * 100:.1f}%\n")
        
        if saved_images:
            f.write(f"Charts Generated: {len(saved_images)}\n")
            for chart_name, path in saved_images.items():
                f.write(f"  - {chart_name}: {os.path.basename(path)}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("PROMPT SENT TO LLM:\n")
        f.write("="*80 + "\n\n")
        f.write(prompt)
    
    print(f"  ✓ Saved prompt: {prompt_path}")
    
    # 2. Write the raw MTF analysis JSON
    analysis_path = os.path.join(out_dir, f"analysis_{sym_safe}_{ts}.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        # Remove chart bytes before serialization
        mtf_for_json = mtf_analysis.copy()
        if 'chart' in mtf_for_json:
            chart_info = mtf_for_json['chart']
            if isinstance(chart_info, dict):
                mtf_for_json['chart'] = {
                    'type': chart_info.get('type'),
                    'format': chart_info.get('format'),
                    'size_bytes': chart_info.get('size_bytes'),
                    'chart_type': chart_info.get('chart_type'),
                    'symbol': chart_info.get('symbol'),
                    'exchange': chart_info.get('exchange'),
                    'note': 'Chart bytes removed for JSON serialization - see saved PNG file'
                }
            else:
                del mtf_for_json['chart']
        
        # Remove any remaining non-serializable objects
        clean_analysis = json.loads(json.dumps(mtf_for_json, default=str))
        json.dump(clean_analysis, f, indent=2, default=str)
    
    print(f"  ✓ Saved analysis: {analysis_path}")
    
    # 3. Write the structured context
    context_path = os.path.join(out_dir, f"context_{sym_safe}_{ts}.json")
    with open(context_path, "w", encoding="utf-8") as f:
        json.dump(mtf_context, f, indent=2, default=str)
    
    print(f"  ✓ Saved context: {context_path}")
    
    # 3b. Write the PRODUCTION format MTF block (as used in knowledge_context)
    production_block_path = os.path.join(out_dir, f"production_mtf_block_{sym_safe}_{ts}.txt")
    with open(production_block_path, "w", encoding="utf-8") as f:
        from core.utils import clean_for_json
        
        # Remove chart bytes before JSON serialization (same as in build_production_knowledge_context)
        mtf_for_production = mtf_analysis.copy()
        if 'chart' in mtf_for_production:
            chart_info = mtf_for_production['chart']
            if isinstance(chart_info, dict):
                mtf_for_production['chart'] = {
                    'type': chart_info.get('type'),
                    'format': chart_info.get('format'),
                    'size_bytes': chart_info.get('size_bytes'),
                    'chart_type': chart_info.get('chart_type'),
                    'symbol': chart_info.get('symbol'),
                    'exchange': chart_info.get('exchange'),
                    'note': 'Chart image sent separately to LLM'
                }
            else:
                del mtf_for_production['chart']
        
        production_block = "MultiTimeframeContext:\n" + json.dumps(clean_for_json(mtf_for_production), indent=2)
        f.write("="*80 + "\n")
        f.write("PRODUCTION MTF BLOCK FORMAT\n")
        f.write("="*80 + "\n\n")
        f.write("This is the EXACT format used in production.\n")
        f.write("In orchestrator.py, this is added to knowledge_context as:\n")
        f.write('  supplemental_blocks.append("MultiTimeframeContext:\\n" + json.dumps(clean_for_json(mtf_context)))\n\n')
        f.write("Then passed to gemini_client.analyze_stock_with_enhanced_calculations()\n")
        f.write("where it's available in the knowledge_context parameter.\n\n")
        f.write("="*80 + "\n")
        f.write("PRODUCTION MTF BLOCK:\n")
        f.write("="*80 + "\n\n")
        f.write(production_block)
    
    print(f"  ✓ Saved production MTF block: {production_block_path}")
    
    # 4. Write LLM response if available
    if llm_response:
        response_path = os.path.join(out_dir, f"response_{sym_safe}_{ts}.txt")
        with open(response_path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("LLM RESPONSE TO MTF ANALYSIS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Stock Symbol: {symbol}\n")
            f.write(f"Exchange: {exchange}\n")
            f.write(f"Response Time: {datetime.now().isoformat()}\n")
            f.write(f"Response Length: {len(llm_response)} characters\n\n")
            f.write("="*80 + "\n")
            f.write("COMPLETE LLM RESPONSE:\n")
            f.write("="*80 + "\n\n")
            f.write(llm_response)
        
        print(f"  ✓ Saved LLM response: {response_path}")
    
    # 5. Write a summary report
    summary_path = os.path.join(out_dir, f"summary_{sym_safe}_{ts}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("MTF ANALYSIS SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Exchange: {exchange}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write("--- ANALYSIS RESULTS ---\n\n")
        
        summary = mtf_analysis.get('summary', {})
        f.write(f"Overall Signal: {summary.get('overall_signal', 'unknown')}\n")
        f.write(f"Confidence: {summary.get('confidence', 0) * 100:.1f}%\n")
        f.write(f"Signal Alignment: {summary.get('signal_alignment', 'unknown')}\n")
        f.write(f"Risk Level: {summary.get('risk_level', 'unknown')}\n\n")
        
        f.write("--- TIMEFRAME COVERAGE ---\n\n")
        timeframe_analyses = mtf_analysis.get('timeframe_analyses', {})
        for tf in ['1min', '5min', '15min', '30min', '1hour', '1day']:
            if tf in timeframe_analyses:
                tf_data = timeframe_analyses[tf]
                if isinstance(tf_data, dict):
                    f.write(f"{tf}: ✓ ({tf_data.get('data_points', 0)} data points, "
                           f"{tf_data.get('confidence', 0) * 100:.0f}% confidence)\n")
            else:
                f.write(f"{tf}: ✗ (not available)\n")
        
        f.write("\n--- CROSS-TIMEFRAME VALIDATION ---\n\n")
        validation = mtf_analysis.get('cross_timeframe_validation', {})
        f.write(f"Consensus: {validation.get('consensus_trend', 'unknown')}\n")
        f.write(f"Signal Strength: {validation.get('signal_strength', 0) * 100:.1f}%\n")
        f.write(f"Supporting TFs: {', '.join(validation.get('supporting_timeframes', []))}\n")
        f.write(f"Conflicting TFs: {', '.join(validation.get('conflicting_timeframes', []))}\n")
        
        f.write("\n--- AGENT SYSTEM PERFORMANCE ---\n\n")
        agent_insights = mtf_analysis.get('agent_insights', {})
        f.write(f"Total Agents: {agent_insights.get('total_agents_run', 0)}\n")
        f.write(f"Successful: {agent_insights.get('successful_agents', 0)}\n")
        f.write(f"Failed: {agent_insights.get('failed_agents', 0)}\n")
        f.write(f"Overall Confidence: {agent_insights.get('confidence_score', 0) * 100:.1f}%\n")
        
        if saved_images:
            f.write("\n--- GENERATED CHARTS ---\n\n")
            for chart_name, path in saved_images.items():
                f.write(f"  • {chart_name}: {os.path.basename(path)}\n")
    
    print(f"  ✓ Saved summary: {summary_path}")


async def process_symbol(
    symbol: str,
    exchange: str,
    out_dir: str,
    call_llm: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Process a single symbol: fetch data, run MTF analysis, extract prompts, optionally call LLM.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {symbol}")
    print(f"{'='*60}")
    
    try:
        # Step 1: Run MTF analysis through the integration manager
        print(f"\n[1/5] Running MTF analysis...")
        success, mtf_analysis = await mtf_agent_integration_manager.get_comprehensive_mtf_analysis(
            symbol=symbol,
            exchange=exchange
        )
        
        if not success:
            print(f"  ✗ MTF analysis failed: {mtf_analysis.get('error', 'Unknown error')}")
            return False
        
        print(f"  ✓ MTF analysis completed successfully")
        
        # Step 2: Extract context that would be sent to LLM
        print(f"\n[2/5] Extracting LLM context...")
        mtf_context = extract_mtf_context_for_llm(mtf_analysis)
        print(f"  ✓ Context extracted")
        
        # Step 3: Build the actual production format (MTF as JSON block in knowledge_context)
        print(f"\n[3/5] Building production knowledge context...")
        production_mtf_block = build_production_knowledge_context(mtf_analysis)
        print(f"  ✓ Production MTF block built ({len(production_mtf_block)} characters)")
        
        # Also build the formatted prompt for visualization
        prompt = build_mtf_prompt(symbol, exchange, mtf_context)
        print(f"  ✓ Formatted prompt built ({len(prompt)} characters)")
        
        # Step 4: Save any images/charts
        print(f"\n[4/5] Checking for charts/images...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_images = save_images_if_present(mtf_analysis, out_dir, symbol, timestamp)
        if saved_images:
            print(f"  ✓ Saved {len(saved_images)} chart(s)")
        else:
            print(f"  ℹ No charts to save")
        
        # Step 5: Optionally call LLM
        llm_response = None
        if call_llm:
            print(f"\n[5/5] Calling LLM for analysis...")
            try:
                from gemini.gemini_client import GeminiClient
                gc = GeminiClient()
                text, code_results, exec_results = await gc.core.call_llm_with_code_execution(prompt)
                llm_response = text or ""
                print(f"  ✓ LLM response received ({len(llm_response)} characters)")
                
                if code_results:
                    print(f"  ℹ Mathematical calculations: {len(code_results)} code snippets executed")
                if exec_results:
                    print(f"  ℹ Calculation results: {len(exec_results)} outputs")
                    
            except Exception as e:
                print(f"  ✗ LLM call failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n[5/5] Skipping LLM call (use --call-llm to enable)")
        
        # Write all results
        print(f"\nWriting results to disk...")
        write_results(
            out_dir=out_dir,
            symbol=symbol,
            exchange=exchange,
            mtf_analysis=mtf_analysis,
            mtf_context=mtf_context,
            prompt=prompt,
            saved_images=saved_images,
            llm_response=llm_response,
            metadata=metadata
        )
        
        print(f"\n✓ Successfully processed {symbol}")
        return True
        
    except Exception as e:
        print(f"\n✗ Error processing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    ap = argparse.ArgumentParser(
        description="MTF Analysis Comprehensive Prompt Generator - Captures actual prompts/responses sent to LLM"
    )
    ap.add_argument("--symbols", required=True, help="Comma-separated list of symbols", default="RELIANCE,TCS,INFY")
    ap.add_argument("--exchange", default="NSE", help="Exchange name (default: NSE)")
    ap.add_argument("--out-dir", default=CURRENT_DIR, help="Output directory for results")
    ap.add_argument("--call-llm", action="store_true", help="Actually call LLM and save response")
    
    args = ap.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    
    print("="*80)
    print("MTF ANALYSIS PROMPT TESTING")
    print("="*80)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Exchange: {args.exchange}")
    print(f"Output Directory: {args.out_dir}")
    print(f"Call LLM: {'Yes' if args.call_llm else 'No'}")
    print("="*80)
    
    # Known stock metadata
    METADATA = {
        "RELIANCE": {"company": "Reliance Industries", "sector": "Energy/Petrochemicals"},
        "TCS": {"company": "Tata Consultancy Services", "sector": "IT Services"},
        "HDFCBANK": {"company": "HDFC Bank", "sector": "Banking"},
        "ICICIBANK": {"company": "ICICI Bank", "sector": "Banking"},
        "ITC": {"company": "ITC Limited", "sector": "FMCG/Tobacco"},
        "INFY": {"company": "Infosys", "sector": "IT Services"},
    }
    
    import asyncio
    
    results = []
    for symbol in symbols:
        metadata = METADATA.get(symbol.upper(), {})
        success = asyncio.run(
            process_symbol(
                symbol=symbol,
                exchange=args.exchange,
                out_dir=args.out_dir,
                call_llm=args.call_llm,
                metadata=metadata
            )
        )
        results.append((symbol, success))
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for symbol, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{symbol}: {status}")
    
    successful = sum(1 for _, s in results if s)
    print(f"\nTotal: {successful}/{len(results)} successful")
    print("="*80)


if __name__ == "__main__":
    main()