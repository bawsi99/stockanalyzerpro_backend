"""
analysis_service.py

Analysis Service - Handles all analysis, AI processing, chart generation, and risk analysis.
This service is responsible for:
- Stock analysis and AI processing
- Technical indicator calculations
- Chart generation and visualization
- Sector analysis orchestration (delegates to sector agent)
- Volume analysis (multi-agent)
- Risk analysis (quantitative + LLM synthesis)
- Pattern recognition
- Real-time analysis callbacks

ARCHITECTURE OVERVIEW - SECTOR ENDPOINTS:

1. THIN CLIENT ENDPOINTS (Delegate to Sector Agent):
   - /sector/benchmark -> /agents/sector/analyze-all
   - /sector/{sector_name}/performance -> /agents/sector/performance/{sector_name}
   - /sector/compare -> /agents/sector/compare
   These endpoints are lightweight wrappers that delegate heavy data processing
   to the sector agent service for caching and optimization.

2. METADATA ENDPOINTS (Direct Access):
   - /sector/list
   - /sector/{sector_name}/stocks
   - /stock/{symbol}/sector
   These endpoints return lightweight metadata directly from SectorClassifier
   without heavy data fetching.

3. SECTOR AGENT ENDPOINTS (Heavy Processing):
   - /agents/sector/analyze-all (PRIMARY - Use this for all sector analysis)
   - /agents/sector/performance/{sector_name}
   - /agents/sector/compare
   These are the actual sector agent implementation endpoints that perform
   all data fetching, caching, computation, and analysis autonomously.

4. VOLUME AGENT ENDPOINTS (Heavy Processing):
   - /agents/volume/analyze-all (runs all volume agents and aggregates)
   - /agents/volume/anomaly
   - /agents/volume/institutional
   - /agents/volume/confirmation
   - /agents/volume/support-resistance
   - /agents/volume/momentum
   These endpoints perform single- or multi-agent volume analysis with
   in-process prefetch reuse using correlation_id where available.

5. RISK AGENT ENDPOINTS (Heavy Processing):
   - /agents/risk/analyze-all (comprehensive risk analysis)
   Performs quantitative metrics (VaR, ES, Sharpe), stress tests, scenarios,
   and LLM-synthesized risk bullets across short/medium/long horizons.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import asyncio
import traceback
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import httpx # New import for HTTP requests
import math # New import for checking NaN values
from services.database_service import make_json_serializable # Importing the serialization utility

# Lightweight in-process cache to pass prefetched data between endpoints within the same service
# Keyed by correlation_id; values contain {'stock_data': pd.DataFrame, 'indicators': Dict[str, Any], 'created_at': datetime}
# OPTIMIZATION: Used by volume agents, sector agents, AND pattern agents to avoid redundant data fetching
# - Volume agents: Reuse stock_data and indicators from enhanced_analyze
# - Sector agents: Reuse stock_data from enhanced_analyze (saves 200-500ms per request)
# - Pattern agents: Reuse stock_data and indicators from enhanced_analyze (saves 300-600ms per request)
VOLUME_PREFETCH_CACHE: dict[str, dict] = {}

# Cache cleanup mechanism - shared by all agent types
CACHE_MAX_AGE_MINUTES = 30  # Pattern analysis can take time, so extend cache lifetime
CACHE_MAX_SIZE = 200  # Support more concurrent analyses

def cleanup_prefetch_cache():
    """Clean up expired cache entries to prevent memory bloat."""
    try:
        current_time = datetime.now()
        expired_keys = [
            key for key, value in VOLUME_PREFETCH_CACHE.items()
            if isinstance(value, dict) and 
               value.get('created_at') and 
               (current_time - value['created_at']).total_seconds() > (CACHE_MAX_AGE_MINUTES * 60)
        ]
        
        for key in expired_keys:
            del VOLUME_PREFETCH_CACHE[key]
        
        # Limit cache size
        if len(VOLUME_PREFETCH_CACHE) > CACHE_MAX_SIZE:
            # Remove oldest entries
            sorted_items = sorted(
                VOLUME_PREFETCH_CACHE.items(),
                key=lambda x: x[1].get('created_at', datetime.min) if isinstance(x[1], dict) else datetime.min
            )
            keys_to_remove = [item[0] for item in sorted_items[:len(VOLUME_PREFETCH_CACHE) - CACHE_MAX_SIZE]]
            for key in keys_to_remove:
                del VOLUME_PREFETCH_CACHE[key]
                
        if expired_keys or len(VOLUME_PREFETCH_CACHE) > CACHE_MAX_SIZE:
            print(f"[CACHE_CLEANUP] Removed {len(expired_keys)} expired entries, cache size: {len(VOLUME_PREFETCH_CACHE)}")
            
    except Exception as e:
        print(f"[CACHE_CLEANUP] Error during cleanup: {e}")
        
def _extract_pattern_insights_for_decision(pattern_results: dict) -> str:
    """
    Extract pattern insights in a format suitable for the final decision agent.
    
    This function converts the structured pattern analysis results into a narrative
    format that the final decision agent can use for its decision making process.
    """
    try:
        if not pattern_results or not isinstance(pattern_results, dict):
            return ""
            
        if not pattern_results.get('success', False):
            return f"Pattern analysis failed: {pattern_results.get('error', 'Unknown error')}"
            
        insights = []
        
        # Overall confidence and summary
        overall_confidence = pattern_results.get('overall_confidence', 0.0)
        insights.append(f"Pattern Analysis Confidence: {overall_confidence:.1%}")
        
        # Consensus signals
        consensus_signals = pattern_results.get('consensus_signals', {})
        if consensus_signals:
            signal_direction = consensus_signals.get('signal_direction', 'neutral')
            signal_strength = consensus_signals.get('signal_strength', 'weak')
            insights.append(f"Consensus Signal: {signal_direction.upper()} ({signal_strength} strength)")
            
            # Detected patterns
            detected_patterns = consensus_signals.get('detected_patterns', [])
            if detected_patterns:
                patterns_text = ", ".join(detected_patterns[:5])  # Limit to top 5 patterns
                insights.append(f"Key Patterns Detected: {patterns_text}")
        
        # Market structure analysis highlights
        ms_analysis = pattern_results.get('market_structure_analysis', {})
        if ms_analysis and ms_analysis.get('success', False):
            ms_confidence = ms_analysis.get('confidence_score', 0.0)
            insights.append(f"Market Structure Confidence: {ms_confidence:.1%}")
            
            # Add key market structure insights
            ms_technical = ms_analysis.get('technical_analysis', {})
            if ms_technical:
                bos_events = ms_technical.get('bos_events', [])
                if bos_events:
                    recent_bos = bos_events[-1] if bos_events else {}
                    if recent_bos:
                        insights.append(f"Recent Break of Structure: {recent_bos.get('direction', 'unknown')} at {recent_bos.get('price', 'N/A')}")
        
        # Cross-validation analysis highlights
        cv_analysis = pattern_results.get('cross_validation_analysis', {})
        if cv_analysis and cv_analysis.get('success', False):
            # Pattern detection results
            pattern_detection = cv_analysis.get('pattern_detection', {})
            if pattern_detection:
                detected_count = len(pattern_detection.get('detected_patterns', []))
                if detected_count > 0:
                    insights.append(f"Cross-Validation: {detected_count} patterns confirmed")
        
        # Pattern conflicts (important for decision making)
        pattern_conflicts = pattern_results.get('pattern_conflicts', [])
        if pattern_conflicts:
            conflict_count = len(pattern_conflicts)
            insights.append(f"Pattern Conflicts Detected: {conflict_count} (requires careful consideration)")
        
        # Unified analysis summary
        unified_analysis = pattern_results.get('unified_analysis', {})
        if unified_analysis:
            recommendation = unified_analysis.get('recommendation', '')
            if recommendation:
                insights.append(f"Pattern Recommendation: {recommendation}")
            
            risk_level = unified_analysis.get('risk_assessment', '')
            if risk_level:
                insights.append(f"Pattern-based Risk: {risk_level}")
        
        # Join all insights into a coherent narrative
        if insights:
            return "\n".join([f"• {insight}" for insight in insights])
        else:
            return "Pattern analysis completed but no significant insights generated."
            
    except Exception as e:
        print(f"[PATTERN_INSIGHTS_EXTRACTION] Error: {e}")
        return f"Error extracting pattern insights: {str(e)}"

# Try to import optional dependencies
try:
    import dotenv
    dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', '.env'))
except ImportError:
    pass

try:
    import pandas as pd
    import numpy as np
    from pandas import Timestamp
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# FastAPI imports
from fastapi import FastAPI, HTTPException, status
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Local imports (top-level, since backend/ is added to sys.path by start scripts)
from core.orchestrator import StockAnalysisOrchestrator
from agents.sector import SectorClassifier, SectorBenchmarkingProvider, enhanced_sector_classifier, SectorCacheManager
from patterns.recognition import PatternRecognition
from analysis.technical_indicators import TechnicalIndicators
from api.responses import FrontendResponseBuilder
from core.chart_manager import get_chart_manager, initialize_chart_manager
from config.deployment_config import DeploymentConfig
from config.storage_config import StorageConfig

# Import token counter for LLM usage tracking
from llm.token_counter import (
    get_token_counter, reset_token_counter, print_token_usage_summary, 
    get_token_usage_summary, get_model_usage_summary, get_agent_model_combinations,
    get_agent_timing_breakdown
)

# Volume agents integration (we'll expose service endpoints that use the existing orchestrator-based implementation)
from agents.volume import VolumeAgentIntegrationManager
from agents.volume import VolumeAgentsOrchestrator
from agents.final_decision.processor import FinalDecisionProcessor

app = FastAPI(title="Stock Analysis Service", version="1.0.0")
logger = logging.getLogger(__name__)

# Database service URL - Updated for distributed services architecture
DATABASE_SERVICE_URL = os.getenv("DATABASE_SERVICE_URL", "http://localhost:8003")
print(f"🔗 Database Service URL: {DATABASE_SERVICE_URL}")

# Sector Agent service URL - for delegating sector-related operations
# Note: Sector agent endpoints are part of this service but treated as a separate logical agent
SECTOR_AGENT_SERVICE_URL = os.getenv("ANALYSIS_SERVICE_URL", "http://localhost:8002")
print(f"🔗 Sector Agent Service URL: {SECTOR_AGENT_SERVICE_URL}")

async def _make_database_request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    json_data: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
    initial_delay: float = 1.0 # seconds
) -> httpx.Response:
    """Makes an HTTP request to the database service with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            if method == "POST":
                response = await client.post(url, json=json_data, timeout=10.0)
            elif method == "GET":
                response = await client.get(url, timeout=10.0)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            response.raise_for_status() # Raise an exception for bad status codes
            return response
        except httpx.HTTPStatusError as e:
            # For HTTP 5xx errors, retry. For 4xx, don't retry as it's likely a client error.
            if 500 <= e.response.status_code < 600 and attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"[DB_RETRY] Attempt {attempt + 1}/{max_retries}: HTTP error {e.response.status_code}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
            else:
                print(f"[DB_RETRY] HTTP error {e.response.status_code}. Not retrying or max retries reached.")
                raise
        except httpx.RequestError as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"[DB_RETRY] Attempt {attempt + 1}/{max_retries}: Request error {e}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
            else:
                print(f"[DB_RETRY] Request error {e}. Max retries reached.")
                raise
    raise Exception("Max retries reached for database request.")

async def _make_sector_agent_request(
    method: str,
    endpoint: str,
    json_data: Optional[Dict[str, Any]] = None,
    timeout: float = 180.0,
    max_retries: int = 3,
    initial_delay: float = 1.0
) -> httpx.Response:
    """
    Makes an HTTP request to the sector agent service with retry logic.
    
    Args:
        method: HTTP method ("POST" or "GET")
    endpoint: Endpoint path (e.g., "/agents/sector/analyze-all")
        json_data: Optional JSON payload for POST requests
        timeout: Request timeout in seconds (default: 180s for sector analysis)
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (exponential backoff)
    
    Returns:
        httpx.Response object
    """
    url = f"{SECTOR_AGENT_SERVICE_URL}{endpoint}"
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method == "POST":
                    response = await client.post(url, json=json_data)
                elif method == "GET":
                    response = await client.get(url)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response
                
        except httpx.HTTPStatusError as e:
            # For HTTP 5xx errors, retry. For 4xx, don't retry as it's likely a client error.
            if 500 <= e.response.status_code < 600 and attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"[SECTOR_RETRY] Attempt {attempt + 1}/{max_retries}: HTTP error {e.response.status_code}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
            else:
                print(f"[SECTOR_RETRY] HTTP error {e.response.status_code}. Not retrying or max retries reached.")
                raise
                
        except httpx.RequestError as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"[SECTOR_RETRY] Attempt {attempt + 1}/{max_retries}: Request error {e}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
            else:
                print(f"[SECTOR_RETRY] Request error {e}. Max retries reached.")
                raise
                
    raise Exception(f"Max retries reached for sector agent request: {endpoint}")

# Load CORS origins from environment variable
# CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:8080,http://127.0.0.1:5173,https://www.stockanalyzerpro.com,https://stock-analyzer-pro.vercel.app,https://stock-analyzer-pro-git-prototype-aaryan-manawats-projects.vercel.app,https://stock-analyzer-cl9o3tivx-aaryan-manawats-projects.vercel.app,https://stockanalyzer-pro.vercel.app").split(",")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = CORS_ORIGINS.split(",") if CORS_ORIGINS else []
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]
print(f"🔧 CORS_ORIGINS: {CORS_ORIGINS}")


# --- Token Analytics Endpoints ---
@app.get("/analytics/tokens")
async def get_token_analytics():
    """Get comprehensive token usage analytics."""
    try:
        token_summary = get_token_usage_summary()
        model_usage = get_model_usage_summary() 
        agent_model_combos = get_agent_model_combinations()
        
        return {
            "success": True,
            "analytics": {
                "summary": token_summary,
                "model_breakdown": model_usage,
                "agent_model_combinations": agent_model_combos
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"/analytics/tokens failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/tokens/models")
async def get_model_analytics():
    """Get model-specific token usage analytics."""
    try:
        model_usage = get_model_usage_summary()
        return {
            "success": True,
            "model_usage": model_usage,
            "models_count": len(model_usage),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"/analytics/tokens/models failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/tokens/agent-table")
async def get_agent_details_table_api():
    """Get per-agent table with image inclusion and size info."""
    try:
        from llm.token_counter import get_agent_details_table
        table = get_agent_details_table()
        return {
            "success": True,
            "table": table,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"/analytics/tokens/agent-table failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/tokens/compare")
async def compare_models(models: Dict[str, List[str]]):
    """Compare efficiency between two models."""
    try:
        model_list = models.get('models', [])
        if len(model_list) != 2:
            raise HTTPException(status_code=400, detail="Exactly 2 models required for comparison")
            
        from llm.token_counter import compare_model_efficiency
        comparison = compare_model_efficiency(model_list[0], model_list[1])
        
        return {
            "success": True,
            "comparison": comparison,
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/analytics/tokens/compare failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/tokens/reset")
async def reset_token_analytics():
    """Reset token usage analytics."""
    try:
        reset_token_counter()
        return {
            "success": True,
            "message": "Token usage analytics reset",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"/analytics/tokens/reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Only allow specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for initialization
MAIN_EVENT_LOOP = None
SECTOR_CACHE = None  # Sector cache manager for caching sector analysis
SECTOR_BENCHMARKING_PROVIDER = None  # Singleton instance for sector benchmarking with caching

@app.on_event("startup")
async def startup_event():
    """Initialize the analysis service on startup."""
    global MAIN_EVENT_LOOP
    MAIN_EVENT_LOOP = asyncio.get_running_loop()
    
    print("🚀 Starting Analysis Service...")
    
    # Initialize Zerodha data client for historical data (no WebSocket needed)
    try:
        from zerodha.client import ZerodhaDataClient
        
        # Test Zerodha credentials
        api_key = os.getenv("ZERODHA_API_KEY")
        access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
        
        if api_key and access_token and api_key != "your_api_key":
            print("🔗 Zerodha credentials configured - historical data available")
        else:
            print("ℹ️  Zerodha credentials not configured - historical data limited")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize Zerodha data client: {e}")
    
    # Initialize other components
    try:
        # Initialize chart manager for deployment
        print("📊 Initializing chart manager...")
        chart_config = DeploymentConfig.get_chart_config()
        chart_manager = initialize_chart_manager(**chart_config)
        print(f"✅ Chart manager initialized: max_age={chart_manager.max_age_hours}h, max_size={chart_manager.max_total_size_mb}MB")
        
        # Redis image manager removed - charts are now generated in-memory
        # Note: Redis is still used for data caching, just not for image storage
        print("📊 Charts are generated in-memory - Redis not used for image storage")
        print("ℹ️  Using file-based chart storage and Redis for data caching")
        
        # Initialize Redis cache manager (still needed for data caching, just not image storage)
        print("💾 Initializing Redis cache manager...")
        try:
            redis_cache_config = DeploymentConfig.get_redis_cache_config()
            from core.redis_unified_cache_manager import initialize_unified_redis_cache_manager
            redis_cache_manager = initialize_unified_redis_cache_manager(**redis_cache_config)
            print(f"✅ Redis cache manager initialized: compression={redis_cache_manager.enable_compression}")
        except Exception as cache_e:
            print(f"⚠️  Warning: Could not initialize Redis cache manager: {cache_e}")
            print("ℹ️  Falling back to local caching")
        
        # Initialize storage configuration
        print("📁 Initializing storage configuration...")
        StorageConfig.ensure_directories_exist()
        storage_info = StorageConfig.get_storage_info()
        print(f"✅ Storage initialized: {storage_info['storage_type']} storage in {storage_info['environment']} environment")
        
        # Initialize sector classifiers (lazy loading - data loaded on first use)
        print("🏭 Sector classifiers ready (lazy loading enabled)")
        # Note: SectorClassifier and enhanced_sector_classifier are already instantiated on import
        # No need to call get_all_sectors() here - it will load on first actual use
        # This saves ~2-3 seconds and several MB of memory on startup
        
        # Initialize sector cache manager
        print("💾 Initializing sector cache manager...")
        global SECTOR_CACHE
        SECTOR_CACHE = SectorCacheManager()
        print(f"✅ Sector cache manager initialized - cache_enabled={SECTOR_CACHE.cache_enabled}, refresh_days={SECTOR_CACHE.refresh_days}")
        
        # Initialize global SectorBenchmarkingProvider for persistent caching across requests
        print("🏭 Initializing global sector benchmarking provider...")
        global SECTOR_BENCHMARKING_PROVIDER
        SECTOR_BENCHMARKING_PROVIDER = SectorBenchmarkingProvider()
        print(f"✅ Global sector benchmarking provider initialized - cache will persist across requests")
    except Exception as e:
        print(f"⚠️  Warning during initialization: {e}")

    # Legacy signals system removed - timeframe weighting now handled by individual agents

    # Calibration system disabled - scripts not implemented
    print("ℹ️  Calibration system disabled (scripts not implemented)")

    # Weekly scheduler disabled - calibration scripts not implemented
    # async def _weekly_scheduler():
    #     ...
    # Scheduler creation disabled
    print("ℹ️  Weekly scheduler disabled - calibration scripts not implemented")


def scheduled_calibration_task() -> None:
    """Weekly calibration job: generate fixtures, calibrate, and backup weights."""
    import subprocess
    import glob
    import time
    
    try:
        # Avoid any work if not explicitly enabled
        if os.environ.get("ENABLE_SCHEDULED_CALIBRATION") != "1":
            return
            
        print("[CALIBRATION] Starting scheduled calibration task")
        
        symbols = os.environ.get(
            "CALIB_SYMBOLS",
            "NIFTY_50,NIFTY_BANK,NIFTY_IT,NIFTY_PHARMA,NIFTY_AUTO,"
            "NIFTY_FMCG,NIFTY_OIL_AND_GAS,NIFTY_METAL,NIFTY_REALTY,NIFTY_MEDIA,"
            "NIFTY_CONSUMER_DURABLES,NIFTY_HEALTHCARE,NIFTY_CHEMICALS,"
            "NIFTY_FINANCIAL_SERVICES,NIFTY_PRIVATE_BANK,NIFTY_PSU_BANK"
        )
        
        # Write outside the watched tree to avoid reload loops
        fixtures_out = os.environ.get(
            "CALIB_FIXTURES_DIR",
            os.path.join(os.path.expanduser('~'), '.traderpro', 'fixtures', 'auto')
        )
        os.makedirs(fixtures_out, exist_ok=True)
        
        # Clean up any old temporary files before starting
        try:
            old_temp_files = glob.glob(os.path.join(fixtures_out, "*.tmp"))
            for temp_file in old_temp_files:
                try:
                    os.remove(temp_file)
                    print(f"[CALIBRATION] Removed old temp file: {os.path.basename(temp_file)}")
                except Exception:
                    pass
        except Exception:
            pass
            
        # Generate fixtures with timeout and memory management
        print("[CALIBRATION] Generating fixtures...")
        gen_cmd = [
            'python', os.path.join(os.path.dirname(__file__), 'scripts', 'generate_fixtures.py'),
            '--symbols', symbols,
            '--period', '365',
            '--interval', 'day',
            '--horizon', '10',
            '--bull', '0.02',
            '--bear', '-0.02',
            '--stride', '10',
            '--out', fixtures_out
        ]
        
        try:
            # Run with timeout to prevent hanging processes
            process = subprocess.run(gen_cmd, check=False, timeout=600)  # 10 minute timeout
            print(f"[CALIBRATION] Fixtures generation completed with return code: {process.returncode}")
        except subprocess.TimeoutExpired:
            print("[CALIBRATION] ⚠️ Fixtures generation timed out after 10 minutes")
        
        # Clear memory before next subprocess
        time.sleep(1)  # Brief pause to allow OS to reclaim resources
        
        # Run calibration with timeout
        print("[CALIBRATION] Running calibration...")
        calib_cmd = [
            'python', os.path.join(os.path.dirname(__file__), 'scripts', 'calibrate_all.py'),
            fixtures_out,
            '--weights', os.path.join(os.path.dirname(__file__), 'signals', 'weights_config.json'),
            '--backup_dir', os.path.join(os.path.dirname(__file__), 'signals', 'weights_history')
        ]
        
        try:
            # Run with timeout to prevent hanging processes
            process = subprocess.run(calib_cmd, check=False, timeout=600)  # 10 minute timeout
            print(f"[CALIBRATION] Calibration completed with return code: {process.returncode}")
        except subprocess.TimeoutExpired:
            print("[CALIBRATION] ⚠️ Calibration timed out after 10 minutes")
        
        # Clean up any temporary files after calibration
        try:
            temp_files = glob.glob(os.path.join(fixtures_out, "*.tmp"))
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
        except Exception:
            pass
            
        print("✅ Scheduled calibration run completed")
    except Exception as e:
        print('[CALIBRATION] Scheduled calibration failed:', e)
    

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("🛑 Shutting down Analysis Service...")
    
    # Cleanup any data clients or resources
    try:
        # No specific cleanup needed for data clients
        print("✅ Data clients cleaned up")
    except Exception as e:
        print(f"⚠️  Warning: Could not cleanup data clients: {e}")
    
    # Cleanup any background tasks or resources
    try:
        # Cancel any running tasks
        tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        if tasks:
            print(f"🔄 Cancelling {len(tasks)} background tasks...")
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            print("✅ Background tasks cleaned up")
    except Exception as e:
        print(f"⚠️  Warning: Could not cleanup background tasks: {e}")
    
    print("✅ Analysis Service shutdown completed")

def convert_charts_to_base64(charts_dict: dict) -> dict:
    """Convert chart file paths to base64 encoded images with improved memory management."""
    import base64
    import gc
    converted_charts = {}
    
    # Process charts one by one to minimize memory usage
    for chart_name, chart_path in charts_dict.items():
        # Handle file paths (charts are now generated in-memory)
        if isinstance(chart_path, str) and os.path.exists(chart_path):
            try:
                # Use a separate function to scope the memory usage
                def process_file_chart():
                    with open(chart_path, 'rb') as f:
                        img_data = f.read()
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        return {
                            'data': f"data:image/png;base64,{img_base64}",
                            'filename': os.path.basename(chart_path),
                            'type': 'image/png'
                        }
                
                # Process the chart and immediately assign the result
                converted_charts[chart_name] = process_file_chart()
                
                # Explicitly suggest garbage collection
                gc.collect()
                
            except Exception as e:
                print(f"Error converting chart {chart_name}: {e}")
                converted_charts[chart_name] = {
                    'error': f"Failed to load chart: {str(e)}",
                    'filename': os.path.basename(chart_path) if isinstance(chart_path, str) else 'unknown'
                }
                
        elif isinstance(chart_path, dict) and chart_path.get('type') == 'image_bytes' and 'data' in chart_path:
            try:
                # Use a separate function to scope the memory usage
                def process_memory_chart():
                    img_data = chart_path['data']
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    return {
                        'data': f"data:image/png;base64,{img_base64}",
                        'format': chart_path.get('format', 'png'),
                        'type': 'image/png'
                    }
                
                # Process the chart and immediately assign the result
                converted_charts[chart_name] = process_memory_chart()
                
                # Explicitly suggest garbage collection
                gc.collect()
                
            except Exception as e:
                print(f"Error converting in-memory chart {chart_name}: {e}")
                converted_charts[chart_name] = {
                    'error': f"Failed to convert in-memory chart: {str(e)}"
                }
                
        else:
            # Handle invalid chart paths
            if isinstance(chart_path, str):
                converted_charts[chart_name] = {
                    'error': 'Chart file not found',
                    'path': chart_path
                }
            else:
                # Don't print the actual chart_path object which might contain binary data
                converted_charts[chart_name] = {
                    'error': 'Invalid chart path format',
                    'path_type': str(type(chart_path))
                }
    
    # Clear the input dictionary to help with garbage collection
    if charts_dict:
        charts_dict.clear()
        
    # Final garbage collection
    gc.collect()
    
    return converted_charts

def cleanup_chart_files(chart_paths: dict) -> dict:
    """Clean up chart files or Redis keys referenced in chart_paths with improved error handling.

    Returns basic stats about cleanup operations performed.
    """
    import gc
    stats = {"files_removed": 0, "redis_keys_cleaned": 0, "errors": 0}
    
    # Guard against None input
    if chart_paths is None:
        return stats
        
    try:
        # Make a copy of the keys to avoid modification during iteration
        chart_keys = list(chart_paths.keys())
        
        for chart_name in chart_keys:
            try:
                chart_path = chart_paths.get(chart_name)
                
                # Skip if the path is None or already removed
                if chart_path is None:
                    continue
                    
                if isinstance(chart_path, str):
                    if chart_path.startswith('chart:'):
                        # Redis is still used for data caching but no longer for image storage
                        # We'll count these for backward compatibility but no action needed
                        stats["redis_keys_cleaned"] += 1
                    elif os.path.exists(chart_path):
                        # Remove only the file; do not remove directories
                        os.remove(chart_path)
                        print(f"✅ Removed chart file: {os.path.basename(chart_path)}")
                        stats["files_removed"] += 1
                        
                # Remove the reference from the dictionary
                chart_paths[chart_name] = None
                
            except Exception as e:
                print(f"⚠️ Error cleaning up chart {chart_name}: {str(e)}")
                stats["errors"] += 1
                # Continue with other charts even if one fails
                continue
        
        # Clear the chart_paths dictionary reference after cleanup
        chart_paths.clear()
        
        # Suggest garbage collection
        gc.collect()
            
        return stats
    except Exception as e:
        print(f"⚠️ Error in cleanup_chart_files: {str(e)}")
        stats["errors"] += 1
        
        # Try to clear the dictionary even in case of error
        try:
            if chart_paths:
                chart_paths.clear()
        except:
            pass
            
        # Suggest garbage collection
        gc.collect()
        
        return stats

def validate_analysis_results(results: dict) -> dict:
    """Validate and ensure all required fields are present in analysis results."""
    required_fields = {
        'ai_analysis': {},
        'indicators': {},
        'overlays': {},
        'indicator_summary': '',
        'chart_insights': '',
        'summary': {},
        'trading_guidance': {},
        'sector_benchmarking': {},
        'metadata': {}
    }
    
    validated_results = {}
    
    for field, default_value in required_fields.items():
        if field in results and results[field] is not None:
            validated_results[field] = results[field]
        else:
            validated_results[field] = default_value
            print(f"Warning: Missing or null field '{field}' in analysis results")
    
    return validated_results

# The resolve_user_id function is moved to database_service.py
# def resolve_user_id(user_id: Optional[str] = None, email: Optional[str] = None) -> str:
#     ... (removed code)

# --- Pydantic Models ---
class AnalysisRequest(BaseModel):
    stock: str = Field(..., description="Stock symbol to analyze")
    exchange: str = Field(default="NSE", description="Stock exchange")
    period: int = Field(default=365, description="Analysis period in days")
    interval: str = Field(default="day", description="Data interval")
    output: Optional[str] = Field(default=None, description="Output directory")
    sector: Optional[str] = Field(default=None, description="Optional sector override")
    user_id: Optional[str] = Field(default=None, description="User ID (UUID)")
    email: Optional[str] = Field(default=None, description="User email for ID mapping")

class EnhancedAnalysisRequest(BaseModel):
    stock: str = Field(..., description="Stock symbol to analyze")
    exchange: str = Field(default="NSE", description="Stock exchange")
    period: int = Field(default=365, description="Analysis period in days")
    interval: str = Field(default="day", description="Data interval")
    output: Optional[str] = Field(default=None, description="Output directory")
    sector: Optional[str] = Field(default=None, description="Optional sector override")
    enable_code_execution: bool = Field(default=True, description="Enable mathematical validation with code execution")
    user_id: Optional[str] = Field(default=None, description="User ID (UUID)")
    email: Optional[str] = Field(default=None, description="User email for ID mapping")

class SectorAnalysisRequest(BaseModel):
    sector: str = Field(..., description="Sector to analyze")
    period: int = Field(default=365, description="Analysis period in days")

class SectorComparisonRequest(BaseModel):
    sectors: List[str] = Field(..., description="List of sectors to compare")
    period: int = Field(default=365, description="Analysis period in days")

class IndicatorsRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    exchange: str = Field(default="NSE", description="Stock exchange")
    interval: str = Field(default="1d", description="Data interval")
    indicators: str = Field(default="rsi,macd,sma,ema,bollinger", description="Comma-separated list of indicators")

# --- Volume Agents API Models ---
class VolumeAgentRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    exchange: str = Field(default="NSE", description="Stock exchange")
    interval: str = Field(default="day", description="Data interval (internal mapping)")
    period: int = Field(default=365, description="Analysis period in days")
    correlation_id: Optional[str] = Field(default=None, description="Optional correlation ID for tracing")
    return_prompt: Optional[bool] = Field(default=False, description="Return per-agent prompt where applicable")

# --- Risk Analysis Agent API Models ---
class RiskAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    exchange: str = Field(default="NSE", description="Stock exchange")
    interval: str = Field(default="day", description="Data interval (internal mapping)")
    period: int = Field(default=365, description="Analysis period in days")
    correlation_id: Optional[str] = Field(default=None, description="Optional correlation ID for tracing")
    return_prompt: Optional[bool] = Field(default=False, description="Return enhanced prompt where applicable")
    timeframes: Optional[List[str]] = Field(default=["short", "medium", "long"], description="Risk analysis timeframes")

# --- Market Structure Agent API Models ---
class MarketStructureRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    exchange: str = Field(default="NSE", description="Stock exchange")
    interval: str = Field(default="day", description="Data interval (internal mapping)")
    period: int = Field(default=365, description="Analysis period in days")
    correlation_id: Optional[str] = Field(default=None, description="Optional correlation ID for tracing")
    return_prompt: Optional[bool] = Field(default=False, description="Return enhanced prompt where applicable")
    context: Optional[str] = Field(default="", description="Additional context for market structure analysis")
    include_charts: Optional[bool] = Field(default=True, description="Whether to generate charts")
    include_llm_analysis: Optional[bool] = Field(default=True, description="Whether to include LLM analysis")

# --- Pattern Analysis Agent API Models ---
class PatternAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    exchange: str = Field(default="NSE", description="Stock exchange")
    interval: str = Field(default="day", description="Data interval (internal mapping)")
    period: int = Field(default=365, description="Analysis period in days")
    correlation_id: Optional[str] = Field(default=None, description="Optional correlation ID for tracing")
    return_prompt: Optional[bool] = Field(default=False, description="Return enhanced prompt where applicable")
    context: Optional[str] = Field(default="", description="Additional context for pattern analysis")

# --- REST API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed service status."""
    try:
        # Check Zerodha data client status
        zerodha_status = "unknown"
        try:
            api_key = os.getenv("ZERODHA_API_KEY")
            access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
            if api_key and access_token and api_key != "your_api_key":
                zerodha_status = "configured"
            else:
                zerodha_status = "not_configured"
        except Exception:
            zerodha_status = "error"
        
        # Check LLM API keys (new backend/llm system)
        llm_status = "unknown"
        try:
            # Check for any configured LLM providers
            from llm.config.config import LLMConfig
            llm_config = LLMConfig()
            available_providers = llm_config.get_available_providers()
            if available_providers:
                llm_status = "configured"
            else:
                llm_status = "not_configured"
        except Exception:
            llm_status = "error"
        
        # Check sector classifiers (use existing global instances)
        sector_status = "unknown"
        try:
            # Use the global enhanced_sector_classifier instead of creating new instance
            from agents.sector import enhanced_sector_classifier
            # Don't call get_all_sectors() unless necessary - just check if available
            sector_status = "available"
        except Exception:
            sector_status = "error"

        # Check database service status
        database_service_status = "unknown"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{DATABASE_SERVICE_URL}/health")
                response.raise_for_status()
                database_service_status = response.json().get("status", "unhealthy")
        except Exception:
            database_service_status = "unreachable"
        
        return {
            "status": "healthy",
            "service": "Stock Analysis Service",
            "timestamp": pd.Timestamp.now().isoformat(),
            "components": {
                "zerodha_data_client": zerodha_status,
                "llm_system": llm_status,
                "sector_classifiers": sector_status,
                "database_service": database_service_status,
                "main_event_loop": "running" if MAIN_EVENT_LOOP and not MAIN_EVENT_LOOP.is_closed() else "not_running"
            },
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Stock Analysis Service",
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    """
    Standard analysis endpoint - now redirects to enhanced analysis.
    This endpoint is maintained for backward compatibility but uses enhanced analysis.
    """
    # Convert AnalysisRequest to EnhancedAnalysisRequest
    enhanced_request = EnhancedAnalysisRequest(
        stock=request.stock,
        exchange=request.exchange,
        period=request.period,
        interval=request.interval,
        output=request.output,
        sector=request.sector,
        enable_code_execution=True,  # Default to enhanced analysis
        user_id=request.user_id,
        email=request.email
    )
    
    # Call the enhanced analysis endpoint
    return await enhanced_analyze(enhanced_request)

@app.post("/analyze/enhanced")
async def enhanced_analyze(request: EnhancedAnalysisRequest):
    """
    Enhanced stock analysis with mathematical validation using code execution.
    Now orchestrates independent analyses in parallel (volume agents via service endpoint,
    MTF, sector, advanced), and only blocks on the final decision LLM.
    """
    start_ts = time.monotonic()
    try:
        print(f"[ENHANCED ANALYSIS] Starting enhanced analysis for {request.stock}")
        
        # Reset token counter for this analysis to get clean metrics
        reset_token_counter()
        print(f"🔄 Token counter reset for analysis of {request.stock}")
        
        serialized_frontend_response = None

        # Resolve user ID
        resolved_user_id = "default_user_id"
        if request.user_id:
            resolved_user_id = request.user_id
        elif request.email:
            try:
                async with httpx.AsyncClient() as client:
                    user_id_response = await _make_database_request_with_retry(
                        client, "POST", f"{DATABASE_SERVICE_URL}/users/resolve-id", json_data={"email": request.email}
                    )
                    resolved_user_id = user_id_response.json().get("user_id")
                    print(f"✅ Resolved user ID from email {request.email}: {resolved_user_id}")
            except Exception as e:
                print(f"❌ Error resolving user ID for email {request.email}: {e}. Using default user ID.")
                resolved_user_id = str(uuid.uuid4())
        else:
            resolved_user_id = str(uuid.uuid4())
            print(f"⚠️ No user ID or email provided. Generated new user ID: {resolved_user_id}")

        # Orchestrator for data and final decision LLM
        orchestrator = StockAnalysisOrchestrator()

        # 1) Retrieve stock data
        try:
            stock_data = await orchestrator.retrieve_stock_data(
                request.stock, request.exchange, request.interval, request.period
            )
        except ValueError as e:
            error_msg = f"Data retrieval failed for {request.stock}: {str(e)}"
            print(f"[ENHANCED ANALYSIS ERROR] {error_msg}")
            elapsed = time.monotonic() - start_ts
            print(f"[ANALYSIS-TIMER] {request.stock} failed early (data retrieval) in {elapsed:.2f}s")
            return JSONResponse(
                content={
                    "success": False,
                    "error": error_msg,
                    "stock_symbol": request.stock,
                    "suggestion": "Please check if the stock symbol is correct and try again."
                },
                status_code=400
            )
        except Exception as e:
            error_msg = f"Unexpected error retrieving data for {request.stock}: {str(e)}"
            print(f"[ENHANCED ANALYSIS ERROR] {error_msg}")
            elapsed = time.monotonic() - start_ts
            print(f"[ANALYSIS-TIMER] {request.stock} failed early (unexpected data retrieval error) in {elapsed:.2f}s")
            return JSONResponse(
                content={
                    "success": False,
                    "error": error_msg,
                    "stock_symbol": request.stock
                },
                status_code=500
            )

        # 2) Calculate indicators
        try:
            indicators = TechnicalIndicators.calculate_all_indicators_optimized(stock_data, request.stock)
        except Exception as e:
            error_msg = f"Technical indicator calculation failed for {request.stock}: {str(e)}"
            print(f"[ENHANCED ANALYSIS ERROR] {error_msg}")
            elapsed = time.monotonic() - start_ts
            print(f"[ANALYSIS-TIMER] {request.stock} failed early (indicator calc) in {elapsed:.2f}s")
            return JSONResponse(
                content={
                    "success": False,
                    "error": error_msg,
                    "stock_symbol": request.stock,
                    "suggestion": "Unable to calculate technical indicators. Please try again later."
                },
                status_code=500
            )

# 3) Launch independent tasks in parallel
        print(f"[ENHANCED ANALYSIS] Launching parallel tasks for {request.stock}")
        # Prepare correlation ID and cache prefetched data to avoid duplicate fetch in volume analyze-all
        correlation_id = str(uuid.uuid4())
        try:
            # Debug: Check stock_data before storing
            print(f"[PREFETCH_DEBUG] Storing stock_data - type: {type(stock_data)}, shape: {getattr(stock_data, 'shape', 'N/A')}")
            if hasattr(stock_data, 'empty'):
                print(f"[PREFETCH_DEBUG] stock_data.empty: {stock_data.empty}")
            if hasattr(stock_data, 'columns'):
                print(f"[PREFETCH_DEBUG] stock_data.columns: {list(stock_data.columns)}")
            if hasattr(stock_data, '__len__'):
                print(f"[PREFETCH_DEBUG] stock_data length: {len(stock_data)}")
            
            # Store prefetched stock_data and indicators in in-process cache for reuse by analyze-all
            VOLUME_PREFETCH_CACHE[correlation_id] = {
                'stock_data': stock_data,
                'indicators': indicators,
                'created_at': datetime.now()
            }
            print(f"[PREFETCH_DEBUG] ✅ Successfully stored in cache with correlation_id: {correlation_id}")
            
            # Cleanup expired cache entries to manage memory
            cleanup_prefetch_cache()
        except Exception as cache_e:
            print(f"[VOLUME_PREFETCH_CACHE] Failed to store prefetched data: {cache_e}")

        # Volume agents via service endpoint (loopback)
        volume_agents_url = os.getenv("ANALYSIS_SERVICE_URL", "http://localhost:8002") + "/agents/volume/analyze-all"
        async def _call_volume_agents():
            try:
                print(f"🔑 [VOLUME_AGENTS] Calling analyze-all with 5 distributed API keys (KEY1-5 for agents 0-4)")
                async with httpx.AsyncClient(timeout=200.0) as client:
                    resp = await client.post(
                        volume_agents_url,
                        json={
                            "symbol": request.stock,
                            "exchange": request.exchange,
                            "interval": request.interval,
                            "period": request.period,
                            "correlation_id": correlation_id
                        }
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    print(f"✅ [VOLUME_AGENTS] All 5 agents completed (each with dedicated API key)")
                    return result
            except Exception as e:
                print(f"[VOLUME_AGENTS] Error calling analyze-all endpoint: {e}")
                return {}

        # Helper: logging wrapper with timeout
        start_base = time.monotonic()
        durations: dict[str, float] = {}
        statuses: dict[str, bool] = {}
        async def _with_logging(name: str, coro, timeout: float | None = None):
            t0 = time.monotonic()
            print(f"[TASK-START] {name} t={t0 - start_base:+.3f}s")
            try:
                if timeout is not None:
                    result = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    result = await coro
                dt = time.monotonic() - t0
                durations[name] = dt
                statuses[name] = True
                print(f"[TASK-END] {name} ok=True dt={dt:.2f}s")
                return result
            except Exception as e:
                dt = time.monotonic() - t0
                durations[name] = dt
                statuses[name] = False
                print(f"[TASK-END] {name} ok=False dt={dt:.2f}s err={type(e).__name__}: {e!r}")
                raise

        # Create tasks with logging + timeouts
        volume_task = asyncio.create_task(_with_logging("volume_agents", _call_volume_agents(), timeout=180.0))

        # MTF analysis with LLM agent
        async def _mtf():
            try:
                print(f"[MTF_DEBUG] Starting MTF analysis for {request.stock}...")
                
                # Step 1: Get technical MTF analysis
                from agents.mtf_analysis import mtf_agent_integration_manager
                success, mtf_technical = await mtf_agent_integration_manager.get_comprehensive_mtf_analysis(
                    symbol=request.stock, exchange=request.exchange
                )
                print(f"[MTF_DEBUG] MTF technical analysis complete for {request.stock}. Success: {success}")
                
                if not success or not isinstance(mtf_technical, dict):
                    print(f"[MTF_DEBUG] MTF technical analysis failed, returning empty")
                    return {}
                
                # Step 2: Send MTF results to LLM agent for natural language analysis
                print(f"[MTF_DEBUG] Calling MTF LLM agent for {request.stock}...")
                from agents.mtf_analysis.mtf_llm_agent import mtf_llm_agent
                llm_success, mtf_llm_analysis = await mtf_llm_agent.analyze_mtf_with_llm(
                    symbol=request.stock,
                    exchange=request.exchange,
                    mtf_analysis=mtf_technical,
                    context=""
                )
                print(f"[MTF_DEBUG] MTF LLM analysis complete for {request.stock}. Success: {llm_success}")
                
                # Step 3: Combine technical and LLM analysis
                combined_mtf = mtf_technical.copy()
                if llm_success and isinstance(mtf_llm_analysis, dict):
                    combined_mtf['llm_insights'] = mtf_llm_analysis
                    print(f"[MTF_DEBUG] MTF LLM insights added to result for {request.stock}")
                else:
                    print(f"[MTF_DEBUG] MTF LLM analysis failed, using technical only for {request.stock}")
                
                print(f"[MTF_DEBUG] Finished complete MTF analysis for {request.stock}")
                return combined_mtf
                
            except Exception as e:
                print(f"[MTF] Error: {e}")
                import traceback
                traceback.print_exc()
                return {}

        mtf_task = asyncio.create_task(_with_logging("mtf", _mtf(), timeout=120.0))
        
        # Risk analysis agent via service endpoint (loopback)
        risk_agents_url = os.getenv("ANALYSIS_SERVICE_URL", "http://localhost:8002") + "/agents/risk/analyze-all"
        async def _risk():
            try:
                print(f"[RISK_DEBUG] Starting risk analysis for {request.stock}...")
                
                async with httpx.AsyncClient(timeout=180.0) as client:
                    resp = await client.post(
                        risk_agents_url,
                        json={
                            "symbol": request.stock,
                            "exchange": request.exchange,
                            "interval": request.interval,
                            "period": request.period,
                            "correlation_id": correlation_id,
                            "return_prompt": False,  # Don't return prompt in parallel execution
                            "timeframes": ["short", "medium", "long"]
                        }
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    
                    print(f"[RISK_DEBUG] Risk analysis completed for {request.stock}. Success: {result.get('success')}")
                    
                    if result.get('success'):
                        print(f"✅ [RISK_AGENT] Analysis completed for {request.stock}")
                        print(f"   Risk Level: {result.get('risk_summary', {}).get('overall_risk_level', 'Unknown')}")
                        print(f"   Risk Score: {result.get('risk_summary', {}).get('overall_risk_score', 0)}")
                        return result
                    else:
                        print(f"⚠️ [RISK_AGENT] Analysis failed: {result.get('error', 'Unknown error')}")
                        return {}
                        
            except Exception as e:
                print(f"[RISK] Error calling risk analysis endpoint: {e}")
                return {}

        risk_task = asyncio.create_task(_with_logging("risk", _risk(), timeout=200.0))

        # Advanced analysis
        async def _advanced():
            try:
                from analysis.advanced_analysis import advanced_analysis_provider
                return await advanced_analysis_provider.generate_advanced_analysis(stock_data, request.stock, indicators)
            except Exception as e:
                print(f"[ADVANCED] Error: {e}")
                return {}

        advanced_task = asyncio.create_task(_with_logging("advanced", _advanced(), timeout=90.0))

        # Sector agent via service endpoint (loopback) - runs in parallel with other agents
        sector_agents_url = os.getenv("ANALYSIS_SERVICE_URL", "http://localhost:8002") + "/agents/sector/analyze-all"
        async def _sector():
            sector_start_time = time.monotonic()
            try:
                print(f"[SECTOR_HTTP_DEBUG] Starting sector HTTP call for {request.stock}")
                print(f"[SECTOR_HTTP_DEBUG] URL: {sector_agents_url}")
                print(f"[SECTOR_HTTP_DEBUG] Correlation ID: {correlation_id}")
                
                # Use longer HTTP timeout than task timeout to avoid race conditions
                http_timeout = 200.0  # 200 seconds for HTTP client
                print(f"[SECTOR_HTTP_DEBUG] HTTP timeout: {http_timeout}s")
                
                request_payload = {
                    "symbol": request.stock,
                    "exchange": request.exchange,
                    "interval": request.interval,
                    "period": request.period,
                    "sector": request.sector,  # Optional - auto-detected if not provided
                    "correlation_id": correlation_id  # Pass correlation_id for prefetch cache lookup
                }
                print(f"[SECTOR_HTTP_DEBUG] Request payload: {request_payload}")
                
                async with httpx.AsyncClient(timeout=http_timeout) as client:
                    print(f"[SECTOR_HTTP_DEBUG] Making POST request...")
                    resp = await client.post(sector_agents_url, json=request_payload)
                    
                    print(f"[SECTOR_HTTP_DEBUG] Response status: {resp.status_code}")
                    print(f"[SECTOR_HTTP_DEBUG] Response headers: {dict(resp.headers)}")
                    
                    resp.raise_for_status()
                    result = resp.json()
                    
                    elapsed = time.monotonic() - sector_start_time
                    print(f"[SECTOR_HTTP_DEBUG] HTTP call completed in {elapsed:.2f}s")
                    print(f"[SECTOR_HTTP_DEBUG] Response success: {result.get('success')}")
                    
                    if result.get('success'):
                        print(f"✅ [SECTOR_AGENT] Analysis completed for {request.stock} in sector {result.get('sector')}")
                        
                        # DEBUG: Check synthesis data
                        synthesis = result.get('synthesis', {})
                        agent_name = synthesis.get('agent_name', 'unknown')
                        bullets = synthesis.get('bullets', '')
                        print(f"[SECTOR_HTTP_DEBUG] Synthesis agent: {agent_name}")
                        print(f"[SECTOR_HTTP_DEBUG] Synthesis bullets length: {len(bullets) if bullets else 0}")
                        if bullets:
                            first_line = bullets.split('\n')[0][:100] if bullets else ''
                            print(f"[SECTOR_HTTP_DEBUG] First bullet line: {first_line}")
                        
                        # Transform response to match expected format
                        transformed_result = {
                            'sector': result.get('sector'),
                            'sector_display_name': result.get('sector_display_name'),
                            'sector_benchmarking': result.get('sector_benchmarking', {}),
                            'sector_rotation': result.get('sector_rotation', {}),
                            'sector_correlation': result.get('sector_correlation', {}),
                            'optimization_metrics': result.get('optimization_metrics', {}),
                            'synthesis_bullets': result.get('synthesis', {}).get('bullets', ''),
                            'synthesis_metadata': {
                                'agent_name': result.get('synthesis', {}).get('agent_name', 'sector_synthesis'),
                                'timestamp': result.get('synthesis', {}).get('timestamp', ''),
                                'used_structured_metrics': result.get('synthesis', {}).get('used_structured_metrics', False)
                            }
                        }
                        
                        print(f"[SECTOR_HTTP_DEBUG] Transformed result has synthesis_bullets: {bool(transformed_result['synthesis_bullets'])}")
                        return transformed_result
                    else:
                        print(f"⚠️ [SECTOR_AGENT] Analysis failed: {result.get('error', 'Unknown error')}")
                        print(f"[SECTOR_HTTP_DEBUG] Full error response: {result}")
                        return {}
                        
            except httpx.TimeoutException as e:
                elapsed = time.monotonic() - sector_start_time
                print(f"[SECTOR_HTTP_DEBUG] ❌ HTTP timeout after {elapsed:.2f}s: {e}")
                return {}
            except httpx.HTTPStatusError as e:
                elapsed = time.monotonic() - sector_start_time
                print(f"[SECTOR_HTTP_DEBUG] ❌ HTTP status error after {elapsed:.2f}s: {e.response.status_code}")
                print(f"[SECTOR_HTTP_DEBUG] Error response: {e.response.text}")
                return {}
            except Exception as e:
                elapsed = time.monotonic() - sector_start_time
                print(f"[SECTOR_HTTP_DEBUG] ❌ General error after {elapsed:.2f}s: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return {}

        sector_task = asyncio.create_task(_with_logging("sector", _sector(), timeout=220.0))  # Increased from 120s to 220s to be longer than HTTP timeout (200s)

        # Indicator summary LLM (runs in parallel with volume/MTF/sector/advanced) - NEW SYSTEM
        async def _indicator_summary():
            try:
                print(f"🚀 [INDICATOR_SUMMARY] Using NEW backend/llm system with indicator-specific logic")
                
                # Use the new enhanced indicators summary with LLM integration
                success, md, ind_json, dbg = await orchestrator.indicator_agents_manager.get_enhanced_indicators_summary(
                    symbol=request.stock,
                    stock_data=stock_data,
                    indicators=indicators,
                    period=request.period,
                    interval=request.interval,
                    context="",
                    return_debug=True
                )
                
                if success:
                    print(f"✅ [INDICATOR_SUMMARY] Completed with NEW system - enhanced conflict detection")
                    # Extract json_blob from debug info for backward compatibility
                    json_blob = dbg.get('json_blob', '') if dbg else ''
                    return md, ind_json, json_blob
                else:
                    print(f"⚠️ [INDICATOR_SUMMARY] NEW system failed, using fallback: {md}")
                    # Fallback response
                    return "Analysis completed with fallback data", {
                        "trend_analysis": {"direction": "neutral", "strength": "weak", "confidence": 50},
                        "momentum": {"rsi_signal": "neutral", "macd_signal": "neutral"},
                        "confidence_score": 50,
                        "fallback_used": True
                    }, ''
                    
            except Exception as e:
                print(f"[INDICATOR_SUMMARY] NEW system error: {e}")
                import traceback
                traceback.print_exc()
                # Fallback empty
                return "Analysis failed", {"fallback_used": True, "error": str(e)}, ''

        indicator_task = asyncio.create_task(_with_logging("indicator_summary", _indicator_summary(), timeout=120.0))

        # Pattern analysis agent via service endpoint (loopback)
        patterns_agents_url = os.getenv("ANALYSIS_SERVICE_URL", "http://localhost:8002") + "/agents/patterns/analyze-all"
        async def _patterns():
            try:
                print(f"[PATTERN_DEBUG] Starting pattern analysis for {request.stock}...")
                
                async with httpx.AsyncClient(timeout=180.0) as client:
                    resp = await client.post(
                        patterns_agents_url,
                        json={
                            "symbol": request.stock,
                            "exchange": request.exchange,
                            "interval": request.interval,
                            "period": request.period,
                            "correlation_id": correlation_id,
                            "return_prompt": False,  # Don't return prompt in parallel execution
                            "context": f"Enhanced pattern analysis for {request.stock}"
                        }
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    
                    print(f"[PATTERN_DEBUG] Pattern analysis completed for {request.stock}. Success: {result.get('success')}")
                    
                    if result.get('success'):
                        print(f"✅ [PATTERN_AGENT] Analysis completed for {request.stock}")
                        print(f"   Confidence: {result.get('pattern_summary', {}).get('overall_confidence', 0):.2%}")
                        print(f"   Patterns Detected: {result.get('pattern_summary', {}).get('patterns_detected', 0)}")
                        return result
                    else:
                        print(f"⚠️ [PATTERN_AGENT] Analysis failed: {result.get('error', 'Unknown error')}")
                        return {}
                        
            except Exception as e:
                print(f"[PATTERN] Error calling pattern analysis endpoint: {e}")
                return {}

        patterns_task = asyncio.create_task(_with_logging("patterns", _patterns(), timeout=200.0))

        # Await all independent tasks with return_exceptions=True to avoid cancellation
        parallel_t0 = time.monotonic()
        results = await asyncio.gather(volume_task, mtf_task, advanced_task, sector_task, indicator_task, risk_task, patterns_task, return_exceptions=True)
        parallel_dt = time.monotonic() - parallel_t0
        print(f"[PARALLEL] Independent tasks completed in {parallel_dt:.2f}s")
        # One-line summary for quick glance
        names = ["volume_agents", "mtf", "advanced", "sector", "indicator_summary", "risk", "patterns"]
        summary = ", ".join(
            f"{n}={durations.get(n, float('nan')):.2f}s/{'OK' if statuses.get(n) else 'ERR'}" for n in names
        )
        print(f"[PARALLEL] Summary: {summary}")

        volume_agents_result = results[0]
        mtf_context = results[1]
        advanced_analysis = results[2]
        sector_context = results[3]
        indicator_summary_result = results[4]
        risk_context = results[5]
        patterns_context = results[6]

        # Normalize exceptions to empty fallbacks
        if isinstance(volume_agents_result, Exception):
            print(f"[PARALLEL] volume_agents failed: {type(volume_agents_result).__name__}: {volume_agents_result!r}")
            volume_agents_result = {}
        if isinstance(mtf_context, Exception):
            print(f"[PARALLEL] mtf failed: {type(mtf_context).__name__}: {mtf_context!r}")
            mtf_context = {}
        if isinstance(advanced_analysis, Exception):
            print(f"[PARALLEL] advanced failed: {type(advanced_analysis).__name__}: {advanced_analysis!r}")
            advanced_analysis = {}
        if isinstance(sector_context, Exception):
            print(f"[PARALLEL] sector failed: {type(sector_context).__name__}: {sector_context!r}")
            sector_context = {}
        else:
            # DEBUG: Log what we actually got from sector task
            print(f"[PARALLEL] sector result type: {type(sector_context)}, has_data: {bool(sector_context)}")
            if isinstance(sector_context, dict) and sector_context:
                print(f"[PARALLEL] sector result keys: {list(sector_context.keys())}")
                if 'synthesis_bullets' in sector_context:
                    bullets_len = len(sector_context['synthesis_bullets']) if sector_context['synthesis_bullets'] else 0
                    print(f"[PARALLEL] sector synthesis_bullets length: {bullets_len}")
            else:
                print(f"[PARALLEL] sector result is empty or not dict")
        if isinstance(indicator_summary_result, Exception):
            print(f"[PARALLEL] indicator_summary failed: {type(indicator_summary_result).__name__}: {indicator_summary_result!r}")
            indicator_summary_result = ("", {})
        if isinstance(risk_context, Exception):
            print(f"[PARALLEL] risk failed: {type(risk_context).__name__}: {risk_context!r}")
            risk_context = {}
        else:
            # DEBUG: Log what we actually got from risk task
            print(f"[PARALLEL] risk result type: {type(risk_context)}, has_data: {bool(risk_context)}")
            if isinstance(risk_context, dict) and risk_context:
                print(f"[PARALLEL] risk result keys: {list(risk_context.keys())}")
                if 'risk_bullets_for_decision' in risk_context:
                    bullets_len = len(risk_context['risk_bullets_for_decision']) if risk_context['risk_bullets_for_decision'] else 0
                    print(f"[PARALLEL] risk bullets_for_decision length: {bullets_len}")
            else:
                print(f"[PARALLEL] risk result is empty or not dict")
        if isinstance(patterns_context, Exception):
            print(f"[PARALLEL] patterns failed: {type(patterns_context).__name__}: {patterns_context!r}")
            patterns_context = {}
        else:
            # DEBUG: Log what we actually got from patterns task
            print(f"[PARALLEL] patterns result type: {type(patterns_context)}, has_data: {bool(patterns_context)}")
            if isinstance(patterns_context, dict) and patterns_context:
                print(f"[PARALLEL] patterns result keys: {list(patterns_context.keys())}")
                if 'pattern_insights_for_decision' in patterns_context:
                    insights_len = len(patterns_context['pattern_insights_for_decision']) if patterns_context['pattern_insights_for_decision'] else 0
                    print(f"[PARALLEL] pattern insights_for_decision length: {insights_len}")
            else:
                print(f"[PARALLEL] patterns result is empty or not dict")

        # Normalize contexts
        if not isinstance(mtf_context, dict):
            mtf_context = {}
        if not isinstance(advanced_analysis, dict):
            advanced_analysis = {}
        if not isinstance(sector_context, dict):
            sector_context = {}
        if not isinstance(volume_agents_result, dict):
            volume_agents_result = {}
        if not isinstance(risk_context, dict):
            risk_context = {}
        if not isinstance(patterns_context, dict):
            patterns_context = {}

        indicator_summary_md, indicator_json, indicator_json_blob = ("", {}, "")
        if isinstance(indicator_summary_result, tuple):
            if len(indicator_summary_result) == 3:
                indicator_summary_md, indicator_json, indicator_json_blob = indicator_summary_result
            elif len(indicator_summary_result) == 2:
                indicator_summary_md, indicator_json = indicator_summary_result
        elif isinstance(indicator_summary_result, dict):
            # In case a dict accidentally gets returned
            indicator_summary_md, indicator_json = "", indicator_summary_result

        # 4) Final decision (centralized agent) — depends on prior results; no charts here.
        chart_insights_md = ""

        # Extract risk bullets for final decision agent
        risk_bullets = None
        try:
            if risk_context and isinstance(risk_context, dict):
                risk_bullets = risk_context.get('risk_bullets_for_decision') or None
        except Exception as e:
            print(f"[RISK_INTEGRATION] Error extracting risk bullets: {e}")
            risk_bullets = None

        # Prepare minimal payloads for the final decision agent
        # DEBUG: Print sector_context to see what we actually get
        print(f"\n[SECTOR_DEBUG] sector_context type: {type(sector_context)}")
        print(f"[SECTOR_DEBUG] sector_context has_data: {bool(sector_context)}")
        if sector_context and isinstance(sector_context, dict):
            print(f"[SECTOR_DEBUG] sector_context keys: {list(sector_context.keys())}")
            if 'synthesis_bullets' in sector_context:
                bullets_preview = sector_context['synthesis_bullets'][:100] if sector_context['synthesis_bullets'] else "(empty)"
                print(f"[SECTOR_DEBUG] synthesis_bullets found: {bullets_preview}...")
            else:
                print(f"[SECTOR_DEBUG] synthesis_bullets NOT found in keys")
        else:
            print(f"[SECTOR_DEBUG] sector_context is empty or not dict")
        
        sector_bullets = None
        try:
            if sector_context:
                sector_bullets = sector_context.get('synthesis_bullets') or None
        except Exception as e:
            print(f"[SECTOR_DEBUG] Error extracting sector_bullets: {e}")
            sector_bullets = None

        # Pass the entire MTF context directly to final decision agent
        # This ensures the raw LLM response from MTF analysis is fully integrated
        mtf_payload = None
        try:
            if isinstance(mtf_context, dict) and mtf_context:
                # Check if we have LLM insights from MTF LLM agent
                mtf_llm_insights = mtf_context.get('llm_insights', {})
                if mtf_llm_insights and mtf_llm_insights.get('success'):
                    # Pass the complete LLM insights including the raw llm_analysis
                    mtf_payload = mtf_llm_insights
                    print(f"[MTF_INTEGRATION] Using MTF LLM insights with raw analysis ({len(mtf_llm_insights.get('llm_analysis', ''))} chars)")
                else:
                    # Fallback to technical MTF analysis only
                    mtf_payload = {
                        'summary': mtf_context.get('summary', {}),
                        'cross_timeframe_validation': mtf_context.get('cross_timeframe_validation', {})
                    }
                    print(f"[MTF_INTEGRATION] Using technical MTF analysis only (no LLM insights)")
        except Exception as e:
            print(f"[MTF_INTEGRATION] Error preparing MTF payload for final decision: {e}")
            mtf_payload = None

        # Extract pattern insights for final decision agent
        pattern_insights = None
        try:
            if patterns_context and isinstance(patterns_context, dict):
                pattern_insights = patterns_context.get('pattern_insights_for_decision') or None
        except Exception as e:
            print(f"[PATTERN_INTEGRATION] Error extracting pattern insights: {e}")
            pattern_insights = None

        # Final decision with START/END logging
        fd_t0 = time.monotonic()
        print(f"[TASK-START] final_decision t={fd_t0 - start_base:+.3f}s")

        # Use the new LLM system (no direct API key access needed)
        try:
            # The new system handles API key management internally
            fd_api_key = None  # Let FinalDecisionProcessor use its own LLM client
        except Exception:
            fd_api_key = None
        api_key_hint = "backend/llm_system"
        print(f"🔑 [FINAL_DECISION] Using new backend/llm system for processing")

        from agents.final_decision.processor import FinalDecisionProcessor
        fd_processor = FinalDecisionProcessor(api_key=fd_api_key)
        
        # DEBUG: Print all inputs to final decision agent for debugging
        print(f"\n{'='*80}")
        print(f"[FINAL_DECISION_DEBUG] Final Decision Agent Inputs for {request.stock}")
        print(f"{'='*80}")
        print(f"Symbol: {request.stock}")
        print(f"Exchange: {request.exchange}")
        print(f"Period: {request.period}")
        print(f"Interval: {request.interval}")
        
        # Debug: Indicator JSON
        ind_json_input = indicator_json_blob or indicator_json or {}
        print(f"\n[FD_INPUT] Indicator JSON (type: {type(ind_json_input)}, length: {len(str(ind_json_input)) if ind_json_input else 0}):")
        if isinstance(ind_json_input, dict) and ind_json_input:
            for key, value in list(ind_json_input.items())[:3]:  # Show first 3 keys only
                print(f"  {key}: {str(value)[:100]}..." if len(str(value)) > 100 else f"  {key}: {value}")
        elif isinstance(ind_json_input, str) and ind_json_input:
            print(f"  {ind_json_input[:200]}..." if len(ind_json_input) > 200 else f"  {ind_json_input}")
        else:
            print(f"  (Empty or None)")
        
        # Debug: MTF Context
        print(f"\n[FD_INPUT] MTF Context (type: {type(mtf_payload)}, has_data: {bool(mtf_payload)}):")
        if mtf_payload and isinstance(mtf_payload, dict):
            # Show if this contains raw LLM analysis from MTF agent
            has_raw_llm = 'llm_analysis' in mtf_payload and isinstance(mtf_payload.get('llm_analysis'), str)
            print(f"  Contains raw LLM analysis: {has_raw_llm}")
            
            for key, value in mtf_payload.items():
                if key == 'llm_analysis' and isinstance(value, str):
                    print(f"  {key}: {value[:150]}..." if len(value) > 150 else f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value).__name__} ({len(str(value)) if value else 0} chars)")
        else:
            print(f"  (Empty or None)")
        
        # Debug: Sector Bullets (THIS IS KEY FOR OUR FALLBACK TESTING)
        print(f"\n[FD_INPUT] Sector Bullets (type: {type(sector_bullets)}, has_data: {bool(sector_bullets)}):")
        if sector_bullets:
            if isinstance(sector_bullets, str):
                # Show first few lines of sector bullets
                lines = sector_bullets.split('\n')[:5]  # First 5 lines
                print(f"  Content (first 5 lines):")
                for i, line in enumerate(lines, 1):
                    print(f"    {i}. {line}")
                total_lines = len(sector_bullets.split('\n'))
                if total_lines > 5:
                    remaining_lines = total_lines - 5
                    print(f"    ... ({remaining_lines} more lines)")
            else:
                print(f"  {str(sector_bullets)[:200]}..." if len(str(sector_bullets)) > 200 else f"  {sector_bullets}")
        else:
            print(f"  (Empty or None)")
        
        # Debug: Advanced Analysis
        print(f"\n[FD_INPUT] Advanced Analysis (type: {type(advanced_analysis)}, has_data: {bool(advanced_analysis)}):")
        if advanced_analysis and isinstance(advanced_analysis, dict):
            print(f"  Keys: {list(advanced_analysis.keys())[:5]}")
            total_chars = sum(len(str(v)) for v in advanced_analysis.values())
            print(f"  Total content size: {total_chars} characters")
        else:
            print(f"  (Empty or None)")
        
        # Debug: Volume Analysis  
        print(f"\n[FD_INPUT] Volume Analysis (type: {type(volume_agents_result)}, has_data: {bool(volume_agents_result)}):")
        if volume_agents_result and isinstance(volume_agents_result, dict):
            print(f"  Success: {volume_agents_result.get('success', False)}")
            if 'consensus_analysis' in volume_agents_result:
                consensus = volume_agents_result['consensus_analysis']
                print(f"  Successful Agents: {consensus.get('successful_agents', 0)}")
                print(f"  Failed Agents: {consensus.get('failed_agents', 0)}")
                print(f"  Overall Confidence: {consensus.get('overall_confidence', 0.0)}")
                print(f"  LLM Responses Available: {consensus.get('llm_responses_available', 0)}")
            if 'individual_agents' in volume_agents_result:
                agent_count = len(volume_agents_result['individual_agents'])
                successful_agents = [name for name, result in volume_agents_result['individual_agents'].items() if result.get('success', False)]
                agents_with_llm = [name for name, result in volume_agents_result['individual_agents'].items() if result.get('has_llm_response', False)]
                print(f"  Individual Agents: {agent_count} total, {len(successful_agents)} successful, {len(agents_with_llm)} with LLM responses")
                print(f"  Successful Agent Names: {successful_agents}")
                print(f"  Agents with LLM: {agents_with_llm}")
            
            # Show combined LLM analysis preview
            combined_llm = volume_agents_result.get('combined_llm_analysis', '')
            if combined_llm:
                preview_length = min(200, len(combined_llm))
                preview = combined_llm[:preview_length].replace('\n', ' ')
                print(f"  Combined LLM Analysis: {len(combined_llm)} chars - \"{preview}...\"")
            else:
                print(f"  Combined LLM Analysis: Not available")
            
            volume_size = len(str(volume_agents_result))
            print(f"  Total data size: {volume_size} characters")
        else:
            print(f"  (Empty or None)")
        
        # Debug: Risk Bullets (NEW - ENHANCED MULTI-TIMEFRAME RISK ANALYSIS)
        print(f"\n[FD_INPUT] Risk Bullets (type: {type(risk_bullets)}, has_data: {bool(risk_bullets)}):")
        if risk_bullets:
            if isinstance(risk_bullets, str):
                # Show first few lines of risk bullets
                lines = risk_bullets.split('\n')[:5]  # First 5 lines
                print(f"  Content (first 5 lines):")
                for i, line in enumerate(lines, 1):
                    print(f"    {i}. {line}")
                total_lines = len(risk_bullets.split('\n'))
                if total_lines > 5:
                    remaining_lines = total_lines - 5
                    print(f"    ... ({remaining_lines} more lines)")
                print(f"  Total length: {len(risk_bullets)} characters")
            else:
                print(f"  {str(risk_bullets)[:200]}..." if len(str(risk_bullets)) > 200 else f"  {risk_bullets}")
        else:
            print(f"  (Empty or None)")
        
        # Debug: Chart Insights
        print(f"\n[FD_INPUT] Chart Insights (length: {len(chart_insights_md) if chart_insights_md else 0}):")
        if chart_insights_md:
            print(f"  {chart_insights_md[:200]}..." if len(chart_insights_md) > 200 else f"  {chart_insights_md}")
        else:
            print(f"  (Empty or None)")
        
        # Debug: Pattern Insights (NEW - PATTERN ANALYSIS LLM AGENT)
        print(f"\n[FD_INPUT] Pattern Insights (type: {type(pattern_insights)}, has_data: {bool(pattern_insights)}):")
        if pattern_insights:
            if isinstance(pattern_insights, str):
                # Show first few lines of pattern insights
                lines = pattern_insights.split('\n')[:5]  # First 5 lines
                print(f"  Content (first 5 lines):")
                for i, line in enumerate(lines, 1):
                    print(f"    {i}. {line}")
                total_lines = len(pattern_insights.split('\n'))
                if total_lines > 5:
                    remaining_lines = total_lines - 5
                    print(f"    ... ({remaining_lines} more lines)")
                print(f"  Total length: {len(pattern_insights)} characters")
            else:
                print(f"  {str(pattern_insights)[:200]}..." if len(str(pattern_insights)) > 200 else f"  {pattern_insights}")
        else:
            print(f"  (Empty or None)")
        
        print(f"\n[FD_INPUT] Knowledge Context: (Empty string)")
        print(f"{'='*80}\n")

        try:
            fd_result = await fd_processor.analyze_async(
                symbol=request.stock,
                ind_json=(indicator_json_blob or indicator_json or {}),
                mtf_context=mtf_payload,
                sector_bullets=sector_bullets,
                advanced_digest=advanced_analysis or {},
                risk_bullets=risk_bullets,  # NEW: Pass enhanced multi-timeframe risk bullets
                pattern_insights=pattern_insights,  # NEW: Pass pattern analysis LLM insights
                chart_insights=chart_insights_md,
                knowledge_context="",
                volume_analysis=volume_agents_result or {}
            )
            ai_analysis = fd_result.get('result', {})
            fd_dt = time.monotonic() - fd_t0
            print(f"✅ [FINAL_DECISION] Completed with API key ...{api_key_hint}")
            print(f"[TASK-END] final_decision ok=True dt={fd_dt:.2f}s")
        except Exception as e:
            fd_dt = time.monotonic() - fd_t0
            print(f"[TASK-END] final_decision ok=False dt={fd_dt:.2f}s err={type(e).__name__}: {e!r}")
            raise

        # For frontend compatibility
        indicator_summary = indicator_summary_md
        chart_insights = chart_insights_md

        # 5) ML predictions removed
        ml_predictions = {}

        # 6) Build frontend response
        chart_paths = {}  # No charts generated in enhanced path
        frontend_response = FrontendResponseBuilder.build_frontend_response(
            symbol=request.stock,
            exchange=request.exchange,
            data=stock_data,
            indicators=indicators,
            ai_analysis=ai_analysis or {},
            indicator_summary=indicator_summary or '',
            chart_insights=chart_insights or '',
            chart_paths=chart_paths,
            sector_context=sector_context,
            mtf_context=mtf_context,
            advanced_analysis=advanced_analysis,
            ml_predictions=ml_predictions,
            period=request.period,
            interval=request.interval
        )

        # 7) Store in DB service
        try:
            serialized_frontend_response = make_json_serializable(frontend_response)
            async with httpx.AsyncClient() as client:
                store_response = await _make_database_request_with_retry(
                    client, "POST", f"{DATABASE_SERVICE_URL}/analyses/store",
                    json_data={
                        "analysis": serialized_frontend_response,
                        "user_id": resolved_user_id,
                        "symbol": request.stock,
                        "exchange": request.exchange,
                        "period": request.period,
                        "interval": request.interval
                    }
                )
                analysis_id = store_response.json().get("analysis_id")
            if analysis_id:
                print(f"✅ Stored enhanced analysis for {request.stock} with ID: {analysis_id}")
            else:
                print(f"⚠️ Warning: Failed to store enhanced analysis for {request.stock}")
        except Exception as e:
            print(f"❌ Storage error (non-fatal): {e}")

        # 8) Cleanup prefetch cache to prevent memory leaks
        # IMPORTANT: Delay cleanup to ensure all agents have finished using the cache
        async def delayed_cache_cleanup(correlation_id, delay=350.0):
            """Clean up prefetch cache after a delay to ensure all agents are done."""
            await asyncio.sleep(delay)
            try:
                if correlation_id in VOLUME_PREFETCH_CACHE:
                    del VOLUME_PREFETCH_CACHE[correlation_id]
                    print(f"🗑️  [PREFETCH_CLEANUP] Removed correlation_id {correlation_id} from cache after {delay}s delay")
            except Exception as cleanup_e:
                print(f"⚠️  [PREFETCH_CLEANUP] Error cleaning up cache: {cleanup_e}")
        
        # Schedule cleanup as a background task (non-blocking)
        # Extended to 350 seconds to allow for 300s cache usage + buffer
        asyncio.create_task(delayed_cache_cleanup(correlation_id, delay=350.0))
        
        # 9) Token usage summary and analysis completion
        elapsed = time.monotonic() - start_ts
        
        # Get token usage summary for this analysis
        token_summary = get_token_usage_summary()
        model_usage = get_model_usage_summary()
        agent_model_combos = get_agent_model_combinations()
        
        print(f"\n{'='*100}")
        print(f"📊 TOKEN USAGE SUMMARY for {request.stock}")
        print(f"{'='*100}")
        print(f"Total Analysis Time: {elapsed:.2f}s")
        print(f"Total LLM Calls: {token_summary['total_usage']['total_calls']}")
        print(f"Total Input Tokens: {token_summary['total_usage']['total_input_tokens']:,}")
        print(f"Total Output Tokens: {token_summary['total_usage']['total_output_tokens']:,}")
        print(f"Total Tokens: {token_summary['total_usage']['total_tokens']:,}")
        
        # Show per-agent breakdown in proper table format
        if agent_model_combos:
            print(f"\n🤖 AGENT DETAILS:")
            print(f"{'='*100}")
            
            # New table with image columns sourced from token_counter
            try:
                from llm.token_counter import get_agent_details_table
                table = get_agent_details_table()
                rows = table.get('rows', [])
                totals_row = table.get('totals', {})
                # Header
                print(f"{'Agent':25} | {'Model':10} | {'Input':>8} | {'Output':>8} | {'Total':>8} | {'Time':>8} | {'Image?':>7} | {'Img Size':>10} | {'Img Tokens':>11}")
                print(f"{'-'*25} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*7} | {'-'*10} | {'-'*11}")
                # Rows
                for r in rows:
                    model_display = r['model'].replace('gemini-2.5-', '').upper() if r['model'] != '-' else '-'
                    img_tokens_disp = (str(r['image_tokens']) if isinstance(r['image_tokens'], int) else '-')
                    print(
                        f"{r['agent'][:25]:25} | {model_display:10} | "
                        f"{r['input']:>8,} | {r['output']:>8,} | {r['total']:>8,} | "
                        f"{r['time_s']:>7.2f}s | {r['image_included']:>7} | {r['image_size']:>10} | {img_tokens_disp:>11}"
                    )
                # Footer
                print(f"{'-'*25} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*7} | {'-'*10} | {'-'*11}")
                print(
                    f"{'TOTAL':25} | {'':10} | {totals_row.get('input_tokens',0):>8,} | "
                    f"{totals_row.get('output_tokens',0):>8,} | {totals_row.get('total_tokens',0):>8,} | "
                    f"{totals_row.get('total_time_s',0.0):>7.2f}s | {'':>7} | {'':>10} | {'':>11}"
                )
            except Exception as _e:
                print(f"[ANALYTICS] Failed to print enhanced agent table: {_e}")
            
            # Add per-model breakdown (unchanged)
            print(f"\n📱 MODEL BREAKDOWN:")
            print(f"{'='*70}")
            print(f"{'Model':20} | {'Input':>12} | {'Output':>12} | {'Total':>12} | {'Calls':>6}")
            print(f"{'-'*20} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*6}")
            
            # Use model_usage summary from token_counter for breakdown
            model_total_input = 0
            model_total_output = 0
            model_total_tokens = 0
            model_total_calls = 0
            
            # Sort models (FLASH before PRO) based on name
            sorted_models = sorted(model_usage.items(), key=lambda x: ('pro' in x[0], x[0]))
            for model, stats in sorted_models:
                model_display = model.replace('gemini-2.5-', '').upper()
                inp = stats.get('input_tokens', 0)
                out = stats.get('output_tokens', 0)
                tot = stats.get('total_tokens', 0)
                calls = stats.get('calls', 0)
                model_total_input += inp
                model_total_output += out
                model_total_tokens += tot
                model_total_calls += calls
                print(f"{model_display:20} | {inp:>12,} | {out:>12,} | {tot:>12,} | {calls:>6}")
            
            # Model breakdown footer
            print(f"{'-'*20} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*6}")
            print(f"{'TOTAL':20} | {model_total_input:>12,} | {model_total_output:>12,} | {model_total_tokens:>12,} | {model_total_calls:>6}")
        
        print(f"{'='*100}")
        print(f"[ANALYSIS-TIMER] {request.stock} completed in {elapsed:.2f}s")
        
        # Add token usage to response metadata
        if isinstance(serialized_frontend_response or frontend_response, dict):
            response_data = serialized_frontend_response or frontend_response
            if 'metadata' not in response_data:
                response_data['metadata'] = {}
            
            response_data['metadata']['token_usage'] = {
                'total_tokens': token_summary['total_usage']['total_tokens'],
                'total_input_tokens': token_summary['total_usage']['total_input_tokens'], 
                'total_output_tokens': token_summary['total_usage']['total_output_tokens'],
                'total_calls': token_summary['total_usage']['total_calls'],
                'analysis_duration_seconds': elapsed,
                'usage_by_model': model_usage,
                'usage_by_agent': token_summary['usage_by_agent']
            }
            
        return JSONResponse(content=serialized_frontend_response or frontend_response, status_code=200)

    except Exception as e:
        error_msg = f"Enhanced analysis failed for {request.stock}: {str(e)}"
        print(f"[ENHANCED ANALYSIS ERROR] {error_msg}")
        print(f"[ENHANCED ANALYSIS ERROR] Traceback: {traceback.format_exc()}")
        elapsed = time.monotonic() - start_ts
        print(f"[ANALYSIS-TIMER] {request.stock} failed in {elapsed:.2f}s")
        return JSONResponse(
            content={
                "success": False,
                "error": error_msg,
                "stock_symbol": request.stock,
                "exchange": request.exchange,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

@app.post("/analyze/async")
async def analyze_async(request: AnalysisRequest):
    """Perform comprehensive stock analysis with async index data fetching for better performance."""
    output_dir = request.output or f"./output/{request.stock}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        orchestrator = StockAnalysisOrchestrator()
        # Authenticate
        auth_success = orchestrator.authenticate()
        if not auth_success:
            raise HTTPException(status_code=401, detail="Failed to authenticate with Zerodha API")

        # Analyze stock with async index data fetching
        results, success_message, error_message = await orchestrator.analyze_stock_with_async_index_data(
            symbol=request.stock,
            exchange=request.exchange,
            period=request.period,
            interval=request.interval,
            output_dir=output_dir,
            sector=request.sector  # Pass sector information
        )
        
        if error_message:
            raise HTTPException(status_code=500, detail=error_message)

        # Make all data JSON serializable
        serialized_results = make_json_serializable(results)
        
        # Best-effort cleanup of any chart files referenced in results
        try:
            if isinstance(results, dict) and isinstance(results.get('charts'), dict):
                _ = cleanup_chart_files(results['charts'])
        except Exception:
            pass

        return {
            "success": True,
            "message": success_message,
            "data": serialized_results,
            "analysis_type": "async_index_data",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in async analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/enhanced-mtf")
async def enhanced_mtf_analyze(request: AnalysisRequest):
    """
    Enhanced multi-timeframe analysis with comprehensive signal validation.
    This endpoint performs analysis across all available timeframes (1min, 5min, 15min, 30min, 1hour, 1day)
    and provides cross-timeframe validation, divergence detection, and confidence-weighted recommendations.
    """
    try:
        print(f"[ENHANCED MTF] Starting enhanced multi-timeframe analysis for {request.stock}")
        
        # Use the new MTF agents integration manager
        from agents.mtf_analysis import mtf_agent_integration_manager
        
        # Perform comprehensive multi-timeframe analysis
        success, mtf_results = await mtf_agent_integration_manager.get_comprehensive_mtf_analysis(
            symbol=request.stock,
            exchange=request.exchange
        )
        
        if not success or not mtf_results.get('success', False):
            error_msg = f"Enhanced multi-timeframe analysis failed: {mtf_results.get('error', 'Unknown error') if mtf_results else 'MTF agents failed'}"
            print(f"[ENHANCED MTF ERROR] {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Store analysis in database service
        try:
            # Make the response JSON serializable to handle NaN values before sending
            serialized_mtf_results = make_json_serializable(mtf_results)

            async with httpx.AsyncClient() as client:
                store_response = await _make_database_request_with_retry(
                    client, "POST", f"{DATABASE_SERVICE_URL}/analyses/store",
                    json_data={
                        "analysis": serialized_mtf_results,
                        "user_id": request.user_id,
                        "symbol": request.stock,
                        "exchange": request.exchange,
                        "period": request.period,
                        "interval": request.interval
                    }
                )
                analysis_id = store_response.json().get("analysis_id")
            
            if not analysis_id:
                print(f"⚠️ Warning: Failed to store enhanced MTF analysis for {request.stock} via database service")
            else:
                print(f"✅ Successfully stored enhanced MTF analysis for {request.stock} with ID: {analysis_id}")
                
        except httpx.HTTPStatusError as http_error:
            print(f"❌ HTTP error storing enhanced MTF analysis: {http_error}")
            print(f"⚠️ Enhanced MTF analysis completed but not stored due to database service HTTP error")
        except httpx.RequestError as req_error:
            print(f"❌ Request error storing enhanced MTF analysis: {req_error}")
            print(f"⚠️ Enhanced MTF analysis completed but not stored due to database service request error")
        except Exception as e:
            print(f"❌ Error storing enhanced MTF analysis: {e}")
            print(f"⚠️ Enhanced MTF analysis completed but not stored due to storage error")
        
        print(f"[ENHANCED MTF] Completed enhanced multi-timeframe analysis for {request.stock}")
        
        # Add metadata to response
        mtf_results['request_metadata'] = {
            'stock_symbol': request.stock,
            'exchange': request.exchange,
            'period_days': request.period,
            'interval': request.interval,
            'analysis_type': 'enhanced_multi_timeframe',
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(content=mtf_results, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Enhanced multi-timeframe analysis failed for {request.stock}: {str(e)}"
        print(f"[ENHANCED MTF ERROR] {error_msg}")
        print(f"[ENHANCED MTF ERROR] Traceback: {traceback.format_exc()}")
        
        # Attempt to clean up any resources that might have been created
        try:
            # Clean up any module references
            if 'enhanced_mtf_analyzer' in locals():
                del locals()['enhanced_mtf_analyzer']
                
            # Clean up any results that might have been created
            if 'mtf_results' in locals() and isinstance(locals()['mtf_results'], dict):
                locals()['mtf_results'].clear()
                
        except Exception as cleanup_e:
            print(f"⚠️ Non-fatal error during MTF error cleanup: {str(cleanup_e)}")
        
        return JSONResponse(
            content={
                "error": error_msg,
                "analysis_type": "enhanced_multi_timeframe",
                "status": "failed",
                "timestamp": time.time()
            },
            status_code=500
        )

@app.post("/sector/benchmark")
async def sector_benchmark(request: AnalysisRequest):
    """
    Get sector benchmarking for a specific stock.
    
    ARCHITECTURE NOTE:
    This endpoint now delegates to /agents/sector/analyze-all for complete sector analysis.
    The sector agent handles all data fetching, caching, computation, and analysis autonomously.
    This is a thin client wrapper for backward compatibility.
    """
    start_ts = time.monotonic()
    
    try:
        print(f"[SECTOR_BENCHMARK] Delegating request for {request.stock} to sector agent (analyze-all)")
        
        # Delegate to sector agent service (analyze-all)
        response = await _make_sector_agent_request(
            method="POST",
            endpoint="/agents/sector/analyze-all",
            json_data={
                "symbol": request.stock,
                "exchange": request.exchange,
                "interval": request.interval,
                "period": request.period,
                "sector": request.sector
            },
            timeout=180.0
        )
        
        result = response.json()
        
        # Transform response format for backward compatibility
        if result.get("success"):
            # Build backward-compatible response from analyze-all
            response_data = {
                "success": True,
                "stock_symbol": request.stock,
                "sector": result.get("sector"),
                "results": result.get("sector_benchmarking", {}),
                "synthesis": result.get("synthesis", {}),
                "timestamp": result.get("timestamp", pd.Timestamp.now().isoformat()),
                "delegated_to": "sector_agent_analyze_all",
                "agent_response_time": round(time.monotonic() - start_ts, 2)
            }
            
            elapsed = time.monotonic() - start_ts
            print(f"[SECTOR_BENCHMARK] ✅ Completed (delegated to analyze-all) in {elapsed:.2f}s")
            
            return JSONResponse(content=response_data)
        else:
            # Agent returned error
            error_msg = result.get("error", "Unknown error from sector agent")
            elapsed = time.monotonic() - start_ts
            print(f"[SECTOR_BENCHMARK] ❌ Sector agent error after {elapsed:.2f}s: {error_msg}")
            
            raise HTTPException(
                status_code=500,
                detail={
                    "success": False,
                    "error": error_msg,
                    "stock_symbol": request.stock,
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            )
        
    except HTTPException:
        elapsed = time.monotonic() - start_ts
        print(f"[SECTOR_BENCHMARK] ❌ HTTP exception after {elapsed:.2f}s")
        print(f"{'='*80}\n")
        raise
    except Exception as e:
        elapsed = time.monotonic() - start_ts
        print(f"[SECTOR_BENCHMARK] ❌ Error after {elapsed:.2f}s: {type(e).__name__}: {e!r}")
        traceback.print_exc()
        print(f"{'='*80}\n")
        
        error_response = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "stock_symbol": request.stock,
            "sector": sector if 'sector' in locals() else None,
            "performance_metrics": {
                "total_time_seconds": round(elapsed, 2),
                "failed": True
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/sector/list")
async def get_sectors():
    """
    Get list of all available sectors.
    
    ARCHITECTURE NOTE:
    This is a lightweight metadata endpoint that returns sector information directly
    from SectorClassifier. No heavy data fetching or computation required.
    """
    try:
        sector_classifier = SectorClassifier(sector_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sector_category'))
        sectors = sector_classifier.get_all_sectors()
        
        response = {
            "success": True,
            "sectors": sectors,
            "total_sectors": len(sectors),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/sector/{sector_name}/stocks")
async def get_sector_stocks(sector_name: str):
    """
    Get all stocks in a specific sector.
    
    ARCHITECTURE NOTE:
    This is a lightweight metadata endpoint that returns stock lists directly
    from SectorClassifier. No heavy data fetching or computation required.
    """
    try:
        sector_classifier = SectorClassifier(sector_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sector_category'))
        stocks = sector_classifier.get_sector_stocks(sector_name)
        sector_info = {
            "sector": sector_name,
            "display_name": sector_classifier.get_sector_display_name(sector_name),
            "primary_index": sector_classifier.get_primary_sector_index(sector_name),
            "stock_count": len(stocks)
        }
        
        response = {
            "success": True,
            "sector_info": sector_info,
            "stocks": stocks,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "sector": sector_name,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/sector/{sector_name}/performance")
async def get_sector_performance(sector_name: str, period: int = 365):
    """
    Get sector performance data.
    
    ARCHITECTURE NOTE:
    This endpoint delegates to the sector agent service at /agents/sector/performance/{sector_name}.
    The sector agent handles all data fetching, caching, and computation.
    This is a thin client wrapper for backward compatibility.
    """
    try:
        print(f"[SECTOR_PERFORMANCE] Delegating request for {sector_name} to sector agent")
        
        # Delegate to sector agent service
        response = await _make_sector_agent_request(
            method="GET",
            endpoint=f"/agents/sector/performance/{sector_name}?period={period}",
            timeout=60.0
        )
        
        result = response.json()
        
        # Return the response directly (already in correct format)
        print(f"[SECTOR_PERFORMANCE] ✅ Completed (delegated) for {sector_name}")
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SECTOR_PERFORMANCE] ❌ Error: {e}")
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "sector": sector_name,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.post("/sector/compare")
async def compare_sectors(request: SectorComparisonRequest):
    """
    Compare multiple sectors.
    
    ARCHITECTURE NOTE:
    This endpoint delegates to the sector agent service at /agents/sector/compare.
    The sector agent handles all data fetching, caching, and computation.
    This is a thin client wrapper for backward compatibility.
    """
    try:
        print(f"[SECTOR_COMPARE] Delegating request for {len(request.sectors)} sectors to sector agent")
        
        # Delegate to sector agent service
        response = await _make_sector_agent_request(
            method="POST",
            endpoint="/agents/sector/compare",
            json_data={"sectors": request.sectors, "period": request.period},
            timeout=120.0
        )
        
        result = response.json()
        
        # Return the response directly (already in correct format)
        print(f"[SECTOR_COMPARE] ✅ Completed (delegated) for {len(request.sectors)} sectors")
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SECTOR_COMPARE] ❌ Error: {e}")
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "sectors": request.sectors,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/stock/{symbol}/sector")
async def get_stock_sector(symbol: str):
    """
    Get sector information for a specific stock.
    
    ARCHITECTURE NOTE:
    This is a lightweight metadata endpoint that returns sector classification directly
    from SectorClassifier. No heavy data fetching or computation required.
    """
    try:
        sector_classifier = SectorClassifier(sector_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sector_category'))
        sector = sector_classifier.get_stock_sector(symbol)
        
        if sector:
            sector_info = {
                "stock_symbol": symbol,
                "sector": sector,
                "sector_name": sector_classifier.get_sector_display_name(sector),
                "sector_index": sector_classifier.get_primary_sector_index(sector),
                "sector_stocks": sector_classifier.get_sector_stocks(sector),
                "sector_stock_count": len(sector_classifier.get_sector_stocks(sector))
            }
        else:
            sector_info = {
                "stock_symbol": symbol,
                "sector": None,
                "sector_name": None,
                "sector_index": None,
                "sector_stocks": [],
                "sector_stock_count": 0,
                "note": "Stock not classified in any sector"
            }
        
        response = {
            "success": True,
            "sector_info": sector_info,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "stock_symbol": symbol,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/stock/{symbol}/indicators")
async def get_stock_indicators(
    symbol: str,
    interval: str = "1day",
    exchange: str = "NSE",
    indicators: str = "rsi,macd,sma,ema,bollinger"
):
    """
    Get technical indicators for a stock symbol.
    This endpoint calculates and returns technical indicators for chart overlays.
    """
    try:
        # Validate interval
        valid_intervals = ['1min', '3min', '5min', '10min', '15min', '30min', '60min', '1day']
        if interval not in valid_intervals:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid interval. Must be one of: {valid_intervals}"
            )
        
        # Map frontend intervals to backend intervals
        interval_mapping = {
            '1min': 'minute',
            '3min': '3minute', 
            '5min': '5minute',
            '10min': '10minute',
            '15min': '15minute',
            '30min': '30minute',
            '60min': '60minute',
            '1day': 'day'
        }
        
        backend_interval = interval_mapping.get(interval, 'day')
        
        # Parse requested indicators
        requested_indicators = [ind.strip() for ind in indicators.split(',')]
        
        # Get orchestrator and authenticate
        orchestrator = StockAnalysisOrchestrator()
        if not orchestrator.authenticate():
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        # Retrieve stock data with appropriate period based on interval
        period_mapping = {
            'minute': 60,      # 60 days for 1min
            '3minute': 100,    # 100 days for 3min
            '5minute': 100,    # 100 days for 5min
            '10minute': 150,   # 150 days for 10min
            '15minute': 200,   # 200 days for 15min
            '30minute': 300,   # 300 days for 30min
            '60minute': 365,   # 365 days for 60min
            'day': 365         # 1 year for daily
        }
        
        period = period_mapping.get(backend_interval, 365)
        try:
            df = await orchestrator.retrieve_stock_data(
                symbol=symbol,
                exchange=exchange,
                interval=backend_interval,
                period=period
            )
        except ValueError as e:
            error_msg = f"Data retrieval failed for {symbol}: {str(e)}"
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Calculate indicators
        indicators_data = {}
        
        if 'rsi' in requested_indicators:
            rsi_values = TechnicalIndicators.calculate_rsi(df, column='close', window=14)
            indicators_data['rsi'] = [float(val) if not pd.isna(val) else None for val in rsi_values]
        
        if 'macd' in requested_indicators:
            macd_result = TechnicalIndicators.calculate_macd(df, column='close')
            indicators_data['macd'] = {
                'macd': [float(val) if not pd.isna(val) else None for val in macd_result[0]],
                'signal': [float(val) if not pd.isna(val) else None for val in macd_result[1]],
                'histogram': [float(val) if not pd.isna(val) else None for val in macd_result[2]]
            }
        
        if 'sma' in requested_indicators:
            sma_20 = TechnicalIndicators.calculate_sma(df, column='close', window=20)
            sma_50 = TechnicalIndicators.calculate_sma(df, column='close', window=50)
            sma_200 = TechnicalIndicators.calculate_sma(df, column='close', window=200)
            indicators_data['sma'] = {
                'sma_20': [float(val) if not pd.isna(val) else None for val in sma_20],
                'sma_50': [float(val) if not pd.isna(val) else None for val in sma_50],
                'sma_200': [float(val) if not pd.isna(val) else None for val in sma_200]
            }
        
        if 'ema' in requested_indicators:
            ema_12 = TechnicalIndicators.calculate_ema(df, column='close', window=12)
            ema_26 = TechnicalIndicators.calculate_ema(df, column='close', window=26)
            ema_50 = TechnicalIndicators.calculate_ema(df, column='close', window=50)
            indicators_data['ema'] = {
                'ema_12': [float(val) if not pd.isna(val) else None for val in ema_12],
                'ema_26': [float(val) if not pd.isna(val) else None for val in ema_26],
                'ema_50': [float(val) if not pd.isna(val) else None for val in ema_50]
            }
        
        if 'bollinger' in requested_indicators:
            bb_result = TechnicalIndicators.calculate_bollinger_bands(df, column='close', window=20, num_std=2.0)
            indicators_data['bollinger_bands'] = {
                'upper_band': [float(val) if not pd.isna(val) else None for val in bb_result[0]],
                'middle_band': [float(val) if not pd.isna(val) else None for val in bb_result[1]],
                'lower_band': [float(val) if not pd.isna(val) else None for val in bb_result[2]]
            }
        
        # Ensure proper datetime index for timestamp conversion
        if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            print(f"[AnalysisService] Setting 'date' column as index for indicators endpoint")
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Get timestamps for alignment
        timestamps = []
        for index in df.index:
            if hasattr(index, 'timestamp'):
                timestamps.append(int(index.timestamp()))
            else:
                print(f"[AnalysisService] Warning: Index {index} has no timestamp method, using fallback")
                timestamps.append(int(pd.Timestamp.now().timestamp()))
        
        response = {
            "success": True,
            "symbol": symbol,
            "interval": interval,
            "indicators": indicators_data,
            "timestamps": timestamps,
            "data_points": len(df),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "symbol": symbol,
            "interval": interval,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/patterns/{symbol}")
async def get_patterns(
    symbol: str,
    interval: str = "1day",
    exchange: str = "NSE",
    pattern_types: str = "all"
):
    """
    Get pattern recognition results for a stock symbol.
    """
    try:
        # Validate interval
        valid_intervals = ['1min', '3min', '5min', '10min', '15min', '30min', '60min', '1day']
        if interval not in valid_intervals:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid interval. Must be one of: {valid_intervals}"
            )
        
        # Map frontend intervals to backend intervals
        interval_mapping = {
            '1min': 'minute',
            '3min': '3minute', 
            '5min': '5minute',
            '10min': '10minute',
            '15min': '15minute',
            '30min': '30minute',
            '60min': '60minute',
            '1day': 'day'
        }
        
        backend_interval = interval_mapping.get(interval, 'day')
        
        # Get orchestrator and authenticate
        orchestrator = StockAnalysisOrchestrator()
        if not orchestrator.authenticate():
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        # Retrieve stock data
        try:
            df = await orchestrator.retrieve_stock_data(
                symbol=symbol,
                exchange=exchange,
                interval=backend_interval,
                period=365
            )
        except ValueError as e:
            error_msg = f"Data retrieval failed for {symbol}: {str(e)}"
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Parse requested pattern types
        requested_patterns = [p.strip() for p in pattern_types.split(',')]
        
        # Calculate patterns
        patterns_data = {}
        
        if 'all' in requested_patterns or 'candlestick' in requested_patterns:
            candlestick_patterns = PatternRecognition.detect_candlestick_patterns(df)
            patterns_data['candlestick_patterns'] = candlestick_patterns[-5:]  # Last 5 patterns
        
        if 'all' in requested_patterns or 'double_top' in requested_patterns:
            double_tops = PatternRecognition.detect_double_top(df['close'])
            patterns_data['double_tops'] = double_tops
        
        if 'all' in requested_patterns or 'double_bottom' in requested_patterns:
            double_bottoms = PatternRecognition.detect_double_bottom(df['close'])
            patterns_data['double_bottoms'] = double_bottoms
        
        if 'all' in requested_patterns or 'head_shoulders' in requested_patterns:
            head_shoulders = PatternRecognition.detect_head_and_shoulders(df['close'])
            patterns_data['head_shoulders'] = head_shoulders
        
        if 'all' in requested_patterns or 'triangles' in requested_patterns:
            triangles = PatternRecognition.detect_triangles(df['close'])
            patterns_data['triangles'] = triangles
        
        # Ensure proper datetime index for timestamp conversion
        if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            print(f"[AnalysisService] Setting 'date' column as index for patterns endpoint")
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Get timestamps for alignment
        timestamps = []
        for index in df.index:
            if hasattr(index, 'timestamp'):
                timestamps.append(int(index.timestamp()))
            else:
                print(f"[AnalysisService] Warning: Index {index} has no timestamp method, using fallback")
                timestamps.append(int(pd.Timestamp.now().timestamp()))
        
        response = {
            "success": True,
            "symbol": symbol,
            "interval": interval,
            "patterns": patterns_data,
            "timestamps": timestamps,
            "data_points": len(df),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "symbol": symbol,
            "interval": interval,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/charts/{symbol}")
async def get_charts(
    symbol: str,
    interval: str = "1day",
    exchange: str = "NSE",
    chart_types: str = "all"
):
    """
    Generate and return charts for a stock symbol.
    """
    try:
        print(f"🔍 Generating charts for {symbol} with interval {interval}")
        
        # Validate interval
        valid_intervals = ['1min', '3min', '5min', '10min', '15min', '30min', '60min', '1day']
        if interval not in valid_intervals:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid interval. Must be one of: {valid_intervals}"
            )
        
        # Map frontend intervals to backend intervals
        interval_mapping = {
            '1min': 'minute',
            '3min': '3minute', 
            '5min': '5minute',
            '10min': '10minute',
            '15min': '15minute',
            '30min': '30minute',
            '60min': '60minute',
            '1day': 'day'
        }
        
        backend_interval = interval_mapping.get(interval, 'day')
        
        # Get orchestrator and authenticate
        orchestrator = StockAnalysisOrchestrator()
        if not orchestrator.authenticate():
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        # Retrieve stock data
        try:
            df = await orchestrator.retrieve_stock_data(
                symbol=symbol,
                exchange=exchange,
                interval=backend_interval,
                period=365
            )
        except ValueError as e:
            error_msg = f"Data retrieval failed for {symbol}: {str(e)}"
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Calculate indicators for charts
        print(f"📊 Calculating indicators for {symbol}")
        indicators = orchestrator.calculate_indicators(df, symbol)
        
        # Use chart manager for directory management
        chart_manager = get_chart_manager()
        chart_dir = chart_manager.create_chart_directory(symbol, interval)
        output_dir = str(chart_dir)
        
        # Generate charts
        print(f"🎨 Generating chart visualizations for {symbol}")
        chart_paths = orchestrator.create_visualizations(df, indicators, symbol, output_dir, backend_interval)
        
        # Convert charts to base64
        print(f"🔄 Converting charts to base64 for {symbol}")
        # Print only chart metadata, not the binary data
        chart_metadata = {name: (path if isinstance(path, str) else "image_data_object") for name, path in chart_paths.items()}
        print(f"Chart paths before conversion: {chart_metadata}")
        
        charts_base64 = convert_charts_to_base64(chart_paths)
        
        # Clear original chart paths dictionary after conversion
        chart_paths.clear()
        
        # Clear indicators data that's no longer needed
        if isinstance(indicators, dict):
            indicators.clear()
        
        response = {
            "success": True,
            "symbol": symbol,
            "interval": interval,
            "charts": charts_base64,
            "chart_count": len(charts_base64),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Clear dataframe to free memory
        del df
        
        print(f"✅ Chart generation completed for {symbol}")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "symbol": symbol,
            "interval": interval,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

# --- Sector Agent Endpoints ---

class SectorAgentRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    exchange: str = Field(default="NSE", description="Stock exchange")
    interval: str = Field(default="day", description="Data interval (internal mapping)")
    period: int = Field(default=365, description="Analysis period in days")
    sector: Optional[str] = Field(default=None, description="Optional sector override")
    return_full_analysis: Optional[bool] = Field(default=True, description="Return comprehensive analysis")
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID for prefetch cache lookup")


@app.post("/agents/sector/analyze-all")
async def agents_sector_analyze_all(req: SectorAgentRequest):
    """Get comprehensive sector analysis including synthesis, benchmarking, and all metrics."""
    try:
        orchestrator = StockAnalysisOrchestrator()
        
        # Check prefetch cache first to avoid redundant data fetching
        stock_data = None
        if req.correlation_id and req.correlation_id in VOLUME_PREFETCH_CACHE:
            cached_entry = VOLUME_PREFETCH_CACHE[req.correlation_id]
            cache_age = (datetime.now() - cached_entry['created_at']).total_seconds()
            
            # Only use cache if it's fresh (less than 300 seconds old to handle long analysis times)
            if cache_age < 300.0:
                stock_data = cached_entry['stock_data']
                logging.info(f"✅ [SECTOR_PREFETCH] Using prefetched stock data for {req.symbol} (age: {cache_age:.1f}s, correlation_id: {req.correlation_id})")
                # Debug: Check what we got from cache
                logging.info(f"[SECTOR_DEBUG] Cached stock_data type: {type(stock_data)}, shape: {stock_data.shape if hasattr(stock_data, 'shape') else 'N/A'}")
                if hasattr(stock_data, 'empty'):
                    logging.info(f"[SECTOR_DEBUG] Cached stock_data empty: {stock_data.empty}")
                if hasattr(stock_data, 'columns'):
                    logging.info(f"[SECTOR_DEBUG] Cached stock_data columns: {list(stock_data.columns)}")
                    
                # Validate cached data
                if (not hasattr(stock_data, 'empty') or stock_data.empty or 
                    not hasattr(stock_data, 'columns') or 'close' not in stock_data.columns):
                    logging.warning(f"⚠️ [SECTOR_PREFETCH] Cached data invalid, will fetch fresh data")
                    stock_data = None
            else:
                logging.info(f"⚠️ [SECTOR_PREFETCH] Cache too old ({cache_age:.1f}s), fetching fresh data for {req.symbol}")
        # Fetch fresh data if not in cache or cache was invalid
        if stock_data is None:
            if req.correlation_id:
                logging.info(f"🔄 [SECTOR_PREFETCH] Cache miss/invalid for correlation_id {req.correlation_id}, fetching fresh data for {req.symbol}")
            else:
                logging.info(f"🔄 [SECTOR_PREFETCH] No correlation_id provided, fetching fresh data for {req.symbol}")
            
            try:
                stock_data = await orchestrator.retrieve_stock_data(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    interval=req.interval,
                    period=req.period
                )
                # Debug: Check what we fetched
                logging.info(f"[SECTOR_DEBUG] Fetched stock_data type: {type(stock_data)}, shape: {stock_data.shape if hasattr(stock_data, 'shape') else 'N/A'}")
                if hasattr(stock_data, 'empty'):
                    logging.info(f"[SECTOR_DEBUG] Fetched stock_data empty: {stock_data.empty}")
                if hasattr(stock_data, 'columns'):
                    logging.info(f"[SECTOR_DEBUG] Fetched stock_data columns: {list(stock_data.columns)}")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Data retrieval failed: {str(e)}")
        
        # Determine sector
        if req.sector:
            sector = req.sector
            logging.info(f"Using user-provided sector '{req.sector}' for {req.symbol}")
        else:
            sector_classifier = SectorClassifier()
            sector = sector_classifier.get_stock_sector(req.symbol)
            logging.info(f"Using auto-detected sector '{sector}' for {req.symbol}")
        
        if not sector or sector == 'UNKNOWN':
            return JSONResponse(
                content={
                    "success": False,
                    "error": "Sector not found or unknown for this stock",
                    "symbol": req.symbol,
                    "timestamp": datetime.now().isoformat()
                },
                status_code=400
            )
        
        # Check file-based cache first (persistent across restarts)
        global SECTOR_CACHE, SECTOR_BENCHMARKING_PROVIDER
        
        # Try to get cached SECTOR-AGNOSTIC analysis from file cache
        # CRITICAL FIX: Cache only contains sector rotation and correlation (not stock-specific benchmarking)
        cached_sector_data = SECTOR_CACHE.get_cached_analysis(sector) if SECTOR_CACHE else None
        
        # Initialize variables for sector-agnostic and stock-specific data
        sector_rotation = None
        sector_correlation = None
        
        if cached_sector_data:
            # Extract cached sector-agnostic data (reusable for all stocks in sector)
            sector_rotation = cached_sector_data.get('sector_rotation')
            sector_correlation = cached_sector_data.get('sector_correlation')
            cache_age = cached_sector_data.get('cache_metadata', {}).get('age_hours', 'N/A')
            logging.info(f"✅ Using cached sector-agnostic data for {sector} (age: {cache_age}h) - rotation & correlation")
        else:
            logging.info(f"🔄 Cache miss for {sector} - will fetch sector-agnostic data")
        
        # ALWAYS calculate stock-specific benchmarking (never cached)
        logging.info(f"🔄 Calculating fresh stock-specific benchmarking for {req.symbol} vs {sector}")
        
        # Get comprehensive sector analysis
        # If we have cached sector data, we'll reuse rotation/correlation but recalculate benchmarking
        comprehensive = await SECTOR_BENCHMARKING_PROVIDER.get_optimized_comprehensive_sector_analysis(
            req.symbol, 
            stock_data, 
            sector, 
            requested_period=req.period,
            cached_rotation=sector_rotation,      # Pass cached rotation if available
            cached_correlation=sector_correlation  # Pass cached correlation if available
        )
        
        if not comprehensive:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "Failed to generate comprehensive sector analysis",
                    "symbol": req.symbol,
                    "sector": sector,
                    "timestamp": datetime.now().isoformat()
                },
                status_code=500
            )
        
        # Extract key metrics for synthesis generation
        sector_benchmarking = comprehensive.get('sector_benchmarking', {})
        sector_rotation = comprehensive.get('sector_rotation', {})
        
        # Check if sector data quality is sufficient for LLM synthesis
        def _is_sector_data_sufficient_for_synthesis(comprehensive_data: dict) -> bool:
            """
            Determine if sector data quality is sufficient for LLM synthesis.
            Returns False if data is unreliable and we should skip LLM calls.
            """
            try:
                sector_benchmarking = comprehensive_data.get('sector_benchmarking', {})
                
                # Check multiple indicators of insufficient data
                data_quality = sector_benchmarking.get('data_quality', {})
                market_benchmarking = sector_benchmarking.get('market_benchmarking', {})
                sector_rotation = comprehensive_data.get('sector_rotation', {})
                
                # Primary indicators of insufficient data - FOCUS ON CORE SECTOR BENCHMARKING QUALITY
                # Only skip LLM synthesis if the core sector benchmarking data is unreliable
                insufficient_indicators = [
                    # CRITICAL: Core data quality flags (if these fail, skip synthesis)
                    data_quality.get('sufficient_data') == False,
                    data_quality.get('analysis_mode') == 'fallback', 
                    data_quality.get('reliability') == 'none',
                    data_quality.get('data_points', 0) == 0,
                    
                    # CRITICAL: Market benchmarking fallback indicators (core functionality)
                    market_benchmarking.get('note') == 'Default values - insufficient data',
                    market_benchmarking.get('data_points', 0) == 0,
                    
                    # CRITICAL: Core sector benchmarking errors
                    'error' in sector_benchmarking,
                    
                    # CRITICAL: System-level fallback (entire analysis failed)
                    comprehensive_data.get('optimization_metrics', {}).get('fallback_reason') == 'Insufficient market/sector data for optimized analysis'
                    
                    # REMOVED: sector_rotation.get('error') - auxiliary feature failures shouldn't skip LLM synthesis
                    # The sector rotation is supplementary data - if core benchmarking works, proceed with synthesis
                ]
                
                # If any major indicator shows insufficient data, skip synthesis
                if any(insufficient_indicators):
                    logging.info(f"[SECTOR_SYNTHESIS] Insufficient data detected - skipping LLM synthesis")
                    logging.info(f"[SECTOR_SYNTHESIS] Insufficient indicators: {[i for i, x in enumerate(insufficient_indicators) if x]}")
                    return False
                
                logging.info(f"[SECTOR_SYNTHESIS] Data quality sufficient - proceeding with LLM synthesis")
                return True
                
            except Exception as e:
                logging.error(f"[SECTOR_SYNTHESIS] Error checking data quality: {e}")
                # Default to skipping synthesis if we can't determine quality
                return False
        
        # ALWAYS generate fresh synthesis (it contains stock-specific data and should never be cached)
        # Generate synthesis using SectorSynthesisProcessor
        from agents.sector import SectorSynthesisProcessor
        sector_synthesis = SectorSynthesisProcessor()
        
        # Extract market and sector benchmarking metrics
        market_benchmarking = comprehensive.get('market_benchmarking', {}) or {}

        # Derive rotation stage/momentum from sector_rotation data for this specific sector
        rotation_stage = None
        rotation_momentum = None
        try:
            perf = (sector_rotation.get('sector_performance') or {}) if isinstance(sector_rotation, dict) else {}
            perf_entry = perf.get(sector)
            if isinstance(perf_entry, dict):
                rs = perf_entry.get('relative_strength')
                mom = perf_entry.get('momentum')
                if rs is not None and mom is not None:
                    rs_val = float(rs)
                    mom_val = float(mom)
                    if rs_val >= 0 and mom_val >= 0:
                        rotation_stage = 'Leading'
                    elif rs_val < 0 and mom_val >= 0:
                        rotation_stage = 'Improving'
                    elif rs_val >= 0 and mom_val < 0:
                        rotation_stage = 'Weakening'
                    else:
                        rotation_stage = 'Lagging'
                    rotation_momentum = mom_val
        except Exception:
            rotation_stage = None
            rotation_momentum = None

        # Build sector_data with correct keys and units expected by SectorSynthesisProcessor
        sector_data = {
            'sector_name': sector,
            # Convert decimals to percentages where applicable
            'sector_outperformance_pct': (sector_benchmarking.get('sector_excess_return') * 100) if (isinstance(sector_benchmarking, dict) and sector_benchmarking.get('sector_excess_return') is not None) else None,
            'market_outperformance_pct': (market_benchmarking.get('excess_return') * 100) if (market_benchmarking.get('excess_return') is not None) else None,
            'sector_beta': sector_benchmarking.get('sector_beta') if isinstance(sector_benchmarking, dict) else None,
            'market_beta': market_benchmarking.get('beta'),
            'rotation_stage': rotation_stage,
            'rotation_momentum': rotation_momentum,
            # Enhanced metrics
            'sector_correlation': (sector_benchmarking.get('sector_correlation') * 100) if (isinstance(sector_benchmarking, dict) and sector_benchmarking.get('sector_correlation') is not None) else None,
            'market_correlation': (market_benchmarking.get('correlation') * 100) if (market_benchmarking.get('correlation') is not None) else None,
            'sector_sharpe': sector_benchmarking.get('sector_sharpe_ratio') if isinstance(sector_benchmarking, dict) else None,
            'market_sharpe': market_benchmarking.get('stock_sharpe'),
            'sector_volatility': (sector_benchmarking.get('sector_volatility') * 100) if (isinstance(sector_benchmarking, dict) and sector_benchmarking.get('sector_volatility') is not None) else None,
            'market_volatility': (market_benchmarking.get('volatility') * 100) if (market_benchmarking.get('volatility') is not None) else None,
            'sector_return': (sector_benchmarking.get('sector_cumulative_return') * 100) if (isinstance(sector_benchmarking, dict) and sector_benchmarking.get('sector_cumulative_return') is not None) else None,
            'market_return': (market_benchmarking.get('cumulative_return') * 100) if (market_benchmarking.get('cumulative_return') is not None) else None
        }
        
        # Check if data quality is sufficient for LLM synthesis
        if _is_sector_data_sufficient_for_synthesis(comprehensive):
            print(f"[SECTOR_AGENT] Generating fresh synthesis for {req.symbol} vs {sector} (includes stock-specific data)...")
            synthesis_result = await sector_synthesis.analyze_async(
                symbol=req.symbol,
                sector_data=sector_data,
                knowledge_context=""
            )
            print(f"[SECTOR_AGENT] Synthesis generation completed for {req.symbol}")
        else:
            # Skip LLM synthesis due to insufficient data quality
            print(f"[SECTOR_AGENT] ⚠️  Skipping LLM synthesis for {req.symbol} due to insufficient data quality")
            synthesis_result = {
                "bullets": "• Sector data unavailable - insufficient historical data for reliable analysis\n• Using fallback sector classification only\n• Recommend checking data availability and trying again later",
                "agent_name": "sector_synthesis_fallback",
                "analysis_timestamp": datetime.now().isoformat(),
                "used_structured_metrics": False,
                "data_quality_check": "insufficient",
                "fallback_reason": "Data quality markers indicate unreliable sector analysis results"
            }
        
        # Add synthesis to comprehensive dict
        comprehensive['sector_synthesis'] = synthesis_result
        
        # CRITICAL: Update cache with ONLY sector-agnostic data (rotation & correlation)
        # DO NOT cache synthesis as it contains stock-specific benchmarking data
        if SECTOR_CACHE:
            try:
                # Build sector-agnostic cache data (reusable for all stocks in this sector)
                sector_agnostic_cache = {
                    'sector_rotation': comprehensive.get('sector_rotation', {}),
                    'sector_correlation': comprehensive.get('sector_correlation', {}),
                    # NOTE: synthesis is NOT cached as it includes stock-specific metrics
                    'optimization_metrics': comprehensive.get('optimization_metrics', {}),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Get current sector index price for cache metadata
                sector_index_name = comprehensive.get('sector_benchmarking', {}).get('sector_info', {}).get('sector_index')
                current_price = 0
                if sector_index_name:
                    sector_index_data = comprehensive.get('sector_benchmarking', {}).get('sector_benchmarking', {})
                    if isinstance(sector_index_data, dict) and 'sector_cumulative_return' in sector_index_data:
                        # Approximate current price from sector data if available
                        current_price = sector_index_data.get('sector_cumulative_return', 0) * 10000  # Placeholder
                
                SECTOR_CACHE.save_analysis(sector, sector_agnostic_cache, current_price)
                logging.info(f"💾 Saved SECTOR-AGNOSTIC data to cache for {sector} (rotation, correlation only)")
                logging.info(f"⚠️  Synthesis NOT cached (contains stock-specific data, will regenerate per stock)")
                logging.info(f"⚠️  Stock-specific benchmarking NOT cached (will be calculated fresh per stock)")
            except Exception as cache_error:
                logging.warning(f"Failed to save sector-agnostic data to cache: {cache_error}")
        
        # Build comprehensive response
        response = {
            "success": True,
            "symbol": req.symbol,
            "sector": sector,
            "sector_display_name": SECTOR_BENCHMARKING_PROVIDER.sector_classifier.get_sector_display_name(sector),
            "synthesis": {
                "bullets": synthesis_result.get('bullets', ''),
                "agent_name": synthesis_result.get('agent_name', 'sector_synthesis'),
                "timestamp": synthesis_result.get('analysis_timestamp', ''),
                "used_structured_metrics": synthesis_result.get('used_structured_metrics', False)
            },
            "sector_benchmarking": sector_benchmarking,
            "sector_rotation": sector_rotation,
            "sector_correlation": comprehensive.get('sector_correlation', {}),
            "optimization_metrics": comprehensive.get('optimization_metrics', {}),
            "comprehensive_analysis": comprehensive,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=make_json_serializable(response), status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SECTOR_AGENT] analyze-all endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "agent_group": "sector",
                "endpoint": "/agents/sector/analyze-all",
                "symbol": req.symbol,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

@app.get("/agents/sector/performance/{sector_name}")
async def agents_sector_performance(sector_name: str, period: int = 365):
    """
    Get sector performance data with caching.
    This endpoint handles all sector data fetching and computation.
    """
    try:
        print(f"[SECTOR_AGENT] Performance request for {sector_name}, period={period}")
        
        # Get sector classifier
        sector_classifier = SectorClassifier(
            sector_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sector_category')
        )
        
        # Get sector index
        sector_index = sector_classifier.get_primary_sector_index(sector_name)
        if not sector_index:
            return JSONResponse(
                content={
                    "success": False,
                    "error": f"No index found for sector: {sector_name}",
                    "sector": sector_name,
                    "timestamp": pd.Timestamp.now().isoformat()
                },
                status_code=404
            )
        
        # Get sector data
        orchestrator = StockAnalysisOrchestrator()
        if not orchestrator.authenticate():
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        try:
            sector_data = await orchestrator.retrieve_stock_data(
                symbol=sector_index,
                exchange="NSE",
                period=period,
                interval="day"
            )
        except ValueError as e:
            return JSONResponse(
                content={
                    "success": False,
                    "error": f"Data retrieval failed for sector {sector_name}: {str(e)}",
                    "sector": sector_name,
                    "timestamp": pd.Timestamp.now().isoformat()
                },
                status_code=400
            )
        
        # Calculate sector performance metrics
        sector_returns = sector_data['close'].pct_change().dropna()
        cumulative_return = (1 + sector_returns).prod() - 1
        volatility = sector_returns.std() * np.sqrt(252)
        
        # Get sector stocks
        sector_stocks = sector_classifier.get_sector_stocks(sector_name)
        
        performance_data = {
            "sector": sector_name,
            "sector_index": sector_index,
            "display_name": sector_classifier.get_sector_display_name(sector_name),
            "period_days": period,
            "cumulative_return": float(cumulative_return),
            "annualized_volatility": float(volatility),
            "stock_count": len(sector_stocks),
            "data_points": len(sector_data),
            "last_price": float(sector_data['close'].iloc[-1]),
            "last_date": sector_data.index[-1].isoformat()
        }
        
        response = {
            "success": True,
            "sector_performance": performance_data,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        print(f"[SECTOR_AGENT] Performance data generated for {sector_name}")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SECTOR_AGENT] performance endpoint error: {e}")
        traceback.print_exc()
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "sector": sector_name,
                "timestamp": pd.Timestamp.now().isoformat()
            },
            status_code=500
        )

@app.post("/agents/sector/compare")
async def agents_sector_compare(request: SectorComparisonRequest):
    """
    Compare multiple sectors with caching.
    This endpoint handles all sector comparison data fetching and computation.
    """
    try:
        print(f"[SECTOR_AGENT] Compare request for {len(request.sectors)} sectors, period={request.period}")
        
        comparison_data = {}
        sector_classifier = SectorClassifier(
            sector_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sector_category')
        )
        
        for sector in request.sectors:
            try:
                # Get sector performance
                sector_index = sector_classifier.get_primary_sector_index(sector)
                if sector_index:
                    orchestrator = StockAnalysisOrchestrator()
                    if not orchestrator.authenticate():
                        raise HTTPException(status_code=401, detail="Authentication failed")
                    
                    try:
                        sector_data = await orchestrator.retrieve_stock_data(
                            symbol=sector_index,
                            exchange="NSE",
                            period=request.period,
                            interval="day"
                        )
                    except ValueError as e:
                        # Skip this sector if data retrieval fails
                        comparison_data[sector] = {
                            "sector": sector,
                            "display_name": sector_classifier.get_sector_display_name(sector),
                            "sector_index": sector_index,
                            "error": f"Data retrieval failed: {str(e)}",
                            "cumulative_return": None,
                            "annualized_volatility": None,
                            "stock_count": len(sector_classifier.get_sector_stocks(sector)),
                            "last_price": None
                        }
                        continue
                    
                    if sector_data is not None and not sector_data.empty:
                        sector_returns = sector_data['close'].pct_change().dropna()
                        cumulative_return = (1 + sector_returns).prod() - 1
                        volatility = sector_returns.std() * np.sqrt(252)
                        
                        comparison_data[sector] = {
                            "sector": sector,
                            "display_name": sector_classifier.get_sector_display_name(sector),
                            "sector_index": sector_index,
                            "cumulative_return": float(cumulative_return),
                            "annualized_volatility": float(volatility),
                            "stock_count": len(sector_classifier.get_sector_stocks(sector)),
                            "last_price": float(sector_data['close'].iloc[-1])
                        }
                    else:
                        comparison_data[sector] = {
                            "sector": sector,
                            "error": "Data not available"
                        }
                else:
                    comparison_data[sector] = {
                        "sector": sector,
                        "error": "Index not found"
                    }
                    
            except Exception as e:
                comparison_data[sector] = {
                    "sector": sector,
                    "error": str(e)
                }
        
        response = {
            "success": True,
            "sector_comparison": comparison_data,
            "period_days": request.period,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        print(f"[SECTOR_AGENT] Comparison data generated for {len(request.sectors)} sectors")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SECTOR_AGENT] compare endpoint error: {e}")
        traceback.print_exc()
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "sectors": request.sectors,
                "timestamp": pd.Timestamp.now().isoformat()
            },
            status_code=500
        )

# --- Volume Agents Endpoints ---

@app.post("/agents/volume/anomaly")
async def agents_volume_anomaly(req: VolumeAgentRequest):
    try:
        orchestrator = StockAnalysisOrchestrator()
        # Retrieve stock data
        try:
            stock_data = await orchestrator.retrieve_stock_data(
                symbol=req.symbol,
                exchange=req.exchange,
                interval=req.interval,
                period=req.period
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Data retrieval failed: {str(e)}")

        # Indicators
        try:
            indicators = TechnicalIndicators.calculate_all_indicators_optimized(stock_data, req.symbol)
        except Exception:
            indicators = {}

        # Run single agent using new distributed architecture
        from agents.volume.volume_anomaly.agent import VolumeAnomalyAgent
        agent = VolumeAnomalyAgent()
        result_data = await agent.analyze_complete(stock_data, req.symbol)
        
        # Convert to compatible format
        result = type('Result', (), {
            'success': result_data.get('success', False),
            'processing_time': result_data.get('processing_time', 0.0),
            'confidence_score': result_data.get('confidence_score', 0),
            'analysis_data': result_data.get('technical_analysis', {}),
            'error_message': result_data.get('error'),
            'prompt_text': None  # Not exposed in new architecture
        })()
        response = {
            "success": result.success,
            "processing_time": result.processing_time,
            "confidence": (result.confidence_score or 0.0),
            "key_data": (result.analysis_data or {}),
            "error_message": result.error_message,
            "metadata": {
                "agent_name": "volume_anomaly",
                "symbol": req.symbol,
                "exchange": req.exchange,
                "interval": req.interval,
                "period": req.period
            }
        }
        if req.return_prompt and result.prompt_text:
            response["prompt"] = result.prompt_text
        return JSONResponse(content=make_json_serializable(response), status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[VOLUME_AGENTS] anomaly endpoint error: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "agent": "volume_anomaly",
                "symbol": req.symbol,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

@app.post("/agents/volume/institutional")
async def agents_volume_institutional(req: VolumeAgentRequest):
    try:
        orchestrator = StockAnalysisOrchestrator()
        # Retrieve stock data
        try:
            stock_data = await orchestrator.retrieve_stock_data(
                symbol=req.symbol,
                exchange=req.exchange,
                interval=req.interval,
                period=req.period
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Data retrieval failed: {str(e)}")

        # Indicators
        try:
            indicators = TechnicalIndicators.calculate_all_indicators_optimized(stock_data, req.symbol)
        except Exception:
            indicators = {}

        # Run single agent using new distributed architecture
        from agents.volume.institutional_activity.agent import InstitutionalActivityAgent
        agent = InstitutionalActivityAgent()
        result_data = await agent.analyze_complete(stock_data, req.symbol)
        
        # Convert to compatible format
        result = type('Result', (), {
            'success': result_data.get('success', False),
            'processing_time': result_data.get('processing_time', 0.0),
            'confidence_score': result_data.get('confidence_score', 0),
            'analysis_data': result_data.get('technical_analysis', {}),
            'error_message': result_data.get('error'),
            'prompt_text': None  # Not exposed in new architecture
        })()
        response = {
            "success": result.success,
            "processing_time": result.processing_time,
            "confidence": (result.confidence_score or 0.0),
            "key_data": (result.analysis_data or {}),
            "error_message": result.error_message,
            "metadata": {
                "agent_name": "institutional_activity",
                "symbol": req.symbol,
                "exchange": req.exchange,
                "interval": req.interval,
                "period": req.period
            }
        }
        if req.return_prompt and result.prompt_text:
            response["prompt"] = result.prompt_text
        return JSONResponse(content=make_json_serializable(response), status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[VOLUME_AGENTS] institutional endpoint error: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "agent": "institutional_activity",
                "symbol": req.symbol,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

@app.post("/agents/volume/confirmation")
async def agents_volume_confirmation(req: VolumeAgentRequest):
    try:
        orchestrator = StockAnalysisOrchestrator()
        # Retrieve stock data
        try:
            stock_data = await orchestrator.retrieve_stock_data(
                symbol=req.symbol,
                exchange=req.exchange,
                interval=req.interval,
                period=req.period
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Data retrieval failed: {str(e)}")

        # Indicators
        try:
            indicators = TechnicalIndicators.calculate_all_indicators_optimized(stock_data, req.symbol)
        except Exception:
            indicators = {}

        # Run single agent using new distributed architecture
        from agents.volume.volume_confirmation.llm_agent import create_volume_confirmation_llm_agent
        agent = create_volume_confirmation_llm_agent()
        result_data = await agent.analyze_complete(stock_data, req.symbol)
        
        # Convert to compatible format
        result = type('Result', (), {
            'success': result_data.get('success', False),
            'processing_time': result_data.get('processing_time', 0.0),
            'confidence_score': result_data.get('confidence_score', 0),
            'analysis_data': result_data.get('technical_analysis', {}),
            'error_message': result_data.get('error'),
            'prompt_text': None  # Not exposed in new architecture
        })()
        response = {
            "success": result.success,
            "processing_time": result.processing_time,
            "confidence": (result.confidence_score or 0.0),
            "key_data": (result.analysis_data or {}),
            "error_message": result.error_message,
            "metadata": {
                "agent_name": "volume_confirmation",
                "symbol": req.symbol,
                "exchange": req.exchange,
                "interval": req.interval,
                "period": req.period
            }
        }
        if req.return_prompt and result.prompt_text:
            response["prompt"] = result.prompt_text
        return JSONResponse(content=make_json_serializable(response), status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[VOLUME_AGENTS] confirmation endpoint error: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "agent": "volume_confirmation",
                "symbol": req.symbol,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

@app.post("/agents/volume/support-resistance")
async def agents_volume_support_resistance(req: VolumeAgentRequest):
    try:
        orchestrator = StockAnalysisOrchestrator()
        # Retrieve stock data
        try:
            stock_data = await orchestrator.retrieve_stock_data(
                symbol=req.symbol,
                exchange=req.exchange,
                interval=req.interval,
                period=req.period
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Data retrieval failed: {str(e)}")

        # Indicators
        try:
            indicators = TechnicalIndicators.calculate_all_indicators_optimized(stock_data, req.symbol)
        except Exception:
            indicators = {}

        # Run single agent using new distributed architecture
        from agents.volume.support_resistance.llm_agent import SupportResistanceLLMAgent
        agent = SupportResistanceLLMAgent()
        result_data = await agent.analyze_complete(stock_data, req.symbol)
        
        # Convert to compatible format
        result = type('Result', (), {
            'success': result_data.get('success', False),
            'processing_time': result_data.get('processing_time', 0.0),
            'confidence_score': result_data.get('confidence_score', 0),
            'analysis_data': result_data.get('technical_analysis', {}),
            'error_message': result_data.get('error'),
            'prompt_text': None  # Not exposed in new architecture
        })()
        response = {
            "success": result.success,
            "processing_time": result.processing_time,
            "confidence": (result.confidence_score or 0.0),
            "key_data": (result.analysis_data or {}),
            "error_message": result.error_message,
            "metadata": {
                "agent_name": "support_resistance",
                "symbol": req.symbol,
                "exchange": req.exchange,
                "interval": req.interval,
                "period": req.period
            }
        }
        if req.return_prompt and result.prompt_text:
            response["prompt"] = result.prompt_text
        return JSONResponse(content=make_json_serializable(response), status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[VOLUME_AGENTS] support-resistance endpoint error: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "agent": "support_resistance",
                "symbol": req.symbol,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

@app.post("/agents/volume/momentum")
async def agents_volume_momentum(req: VolumeAgentRequest):
    try:
        orchestrator = StockAnalysisOrchestrator()
        # Retrieve stock data
        try:
            stock_data = await orchestrator.retrieve_stock_data(
                symbol=req.symbol,
                exchange=req.exchange,
                interval=req.interval,
                period=req.period
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Data retrieval failed: {str(e)}")

        # Indicators
        try:
            indicators = TechnicalIndicators.calculate_all_indicators_optimized(stock_data, req.symbol)
        except Exception:
            indicators = {}

        # Run single agent using new distributed architecture
        from agents.volume.volume_momentum.agent import VolumeMomentumAgent
        agent = VolumeMomentumAgent()
        result_data = await agent.analyze_complete(stock_data, req.symbol)
        
        # Convert to compatible format
        result = type('Result', (), {
            'success': result_data.get('success', False),
            'processing_time': result_data.get('processing_time', 0.0),
            'confidence_score': result_data.get('confidence_score', 0),
            'analysis_data': result_data.get('technical_analysis', {}),
            'error_message': result_data.get('error'),
            'prompt_text': None  # Not exposed in new architecture
        })()
        response = {
            "success": result.success,
            "processing_time": result.processing_time,
            "confidence": (result.confidence_score or 0.0),
            "key_data": (result.analysis_data or {}),
            "error_message": result.error_message,
            "metadata": {
                "agent_name": "volume_momentum",
                "symbol": req.symbol,
                "exchange": req.exchange,
                "interval": req.interval,
                "period": req.period
            }
        }
        if req.return_prompt and result.prompt_text:
            response["prompt"] = result.prompt_text
        return JSONResponse(content=make_json_serializable(response), status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[VOLUME_AGENTS] momentum endpoint error: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "agent": "volume_momentum",
                "symbol": req.symbol,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

@app.post("/agents/volume/analyze-all")
async def agents_volume_analyze_all(req: VolumeAgentRequest):
    try:
        # Prepare orchestrator and data
        orchestrator = StockAnalysisOrchestrator()

        # Attempt to reuse prefetched data if provided via correlation_id
        stock_data = None
        indicators = None
        if req.correlation_id:
            try:
                cached = VOLUME_PREFETCH_CACHE.get(req.correlation_id, None)  # Use .get() instead of .pop()
                if cached and isinstance(cached, dict):
                    stock_data = cached.get('stock_data')
                    indicators = cached.get('indicators')
                    print(f"[VOLUME_PREFETCH_CACHE] Using prefetched data for correlation_id={req.correlation_id}")
            except Exception as cache_e:
                print(f"[VOLUME_PREFETCH_CACHE] Error retrieving prefetched data: {cache_e}")

        # Retrieve stock data only if not provided
        if stock_data is None:
            try:
                stock_data = await orchestrator.retrieve_stock_data(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    interval=req.interval,
                    period=req.period
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Data retrieval failed: {str(e)}")

        # Calculate indicators only if not provided
        if indicators is None:
            try:
                indicators = TechnicalIndicators.calculate_all_indicators_optimized(stock_data, req.symbol)
            except Exception as ind_e:
                # Non-fatal: proceed with minimal indicators
                indicators = {}
                print(f"[VOLUME_AGENTS] Warning: indicator calculation failed: {ind_e}")

        # Run integration manager (runs all 5 agents concurrently and aggregates)
        # Pass None to enable distributed API keys (each agent gets its own key)
        integ = VolumeAgentIntegrationManager(None)
        result = await integ.get_comprehensive_volume_analysis(stock_data, req.symbol, indicators)

        # Print agent details table (with image info) to service logs
        try:
            from llm.token_counter import print_agent_details_table
            print_agent_details_table()
        except Exception as _e:
            print(f"[ANALYTICS] Failed to print agent details table: {_e}")

        # Ensure JSON serializable
        serializable = make_json_serializable(result)
        return JSONResponse(content=serializable, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[VOLUME_AGENTS] analyze-all endpoint error: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "agent_group": "volume",
                "endpoint": "/agents/volume/analyze-all",
                "symbol": req.symbol,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

# ===== RISK ANALYSIS AGENT ENDPOINTS =====

@app.post("/agents/risk/analyze-all")
async def agents_risk_analyze_all(req: RiskAnalysisRequest):
    """
    Comprehensive Risk Analysis Agent endpoint.
    
    This endpoint performs multi-timeframe quantitative risk analysis using:
    - Advanced risk metrics (VaR, Expected Shortfall, Sharpe ratios, etc.)
    - Comprehensive stress testing (Historical, Monte Carlo, Sector-specific, Market crash scenarios)
    - Scenario analysis (Bull/Bear/Sideways/Volatility spike scenarios)
    - Enhanced LLM analysis for 15 structured risk bullets across timeframes
    
    The analysis provides actionable risk insights for short-term (1-3 months),
    medium-term (3-12 months), and long-term (1+ years) trading decisions.
    """
    start_time = time.monotonic()
    print(f"[RISK_AGENT] Starting comprehensive risk analysis for {req.symbol}")
    
    try:
        # Import risk analysis components
        from agents.risk_analysis.quantitative_risk.processor import QuantitativeRiskProcessor
        from agents.risk_analysis.risk_llm_agent import get_risk_llm_agent
        
        # Attempt to reuse prefetched data if provided via correlation_id
        stock_data = None
        indicators = None
        if req.correlation_id:
            try:
                cached = VOLUME_PREFETCH_CACHE.get(req.correlation_id, None)
                if cached and isinstance(cached, dict):
                    stock_data = cached.get('stock_data')
                    indicators = cached.get('indicators')
                    print(f"[RISK_AGENT] Using prefetched data for correlation_id={req.correlation_id}")
            except Exception as cache_e:
                print(f"[RISK_AGENT] Error retrieving prefetched data: {cache_e}")
        
        # Retrieve stock data only if not provided
        if stock_data is None:
            try:
                orchestrator = StockAnalysisOrchestrator()
                stock_data = await orchestrator.retrieve_stock_data(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    interval=req.interval,
                    period=req.period
                )
                print(f"[RISK_AGENT] Retrieved {len(stock_data)} days of data for {req.symbol}")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Data retrieval failed: {str(e)}")
        
        # Calculate indicators only if not provided
        if indicators is None:
            try:
                indicators = TechnicalIndicators.calculate_all_indicators_optimized(stock_data, req.symbol)
                print(f"[RISK_AGENT] Calculated {len(indicators)} indicators for {req.symbol}")
            except Exception as ind_e:
                # Non-fatal: proceed with minimal indicators
                indicators = {}
                print(f"[RISK_AGENT] Warning: indicator calculation failed: {ind_e}")
        
        # Step 1: Run Quantitative Risk Processor
        processor_start = time.monotonic()
        processor = QuantitativeRiskProcessor()
        
        context = f"Enhanced multi-timeframe risk analysis for {req.symbol}"
        risk_analysis_result = await processor.analyze_async(
            stock_data=stock_data,
            indicators=indicators,
            context=context
        )
        
        processor_time = time.monotonic() - processor_start
        print(f"[RISK_AGENT] Quantitative analysis completed in {processor_time:.2f}s")
        
        # Add metadata to quantitative result
        risk_analysis_result['symbol'] = req.symbol
        risk_analysis_result['company'] = f"{req.symbol} Company"  # Could be enhanced with actual company lookup
        risk_analysis_result['sector'] = "Unknown"  # Could be enhanced with sector lookup
        risk_analysis_result['timestamp'] = datetime.now().isoformat()
        
        # Step 2: Run Risk LLM Agent for enhanced analysis
        llm_start = time.monotonic()
        risk_agent = get_risk_llm_agent()  # Get the migrated risk agent instance
        llm_success, risk_llm_analysis = await risk_agent.analyze_risk_with_llm(
            symbol=req.symbol,
            risk_analysis_result=risk_analysis_result,
            context=context
        )
        
        llm_time = time.monotonic() - llm_start
        print(f"[RISK_AGENT] LLM analysis completed in {llm_time:.2f}s. Success: {llm_success}")
        
        # Step 3: Build comprehensive response
        total_time = time.monotonic() - start_time
        
        # Extract key metrics for summary
        advanced_metrics = risk_analysis_result.get('advanced_risk_metrics', {})
        overall_assessment = risk_analysis_result.get('overall_risk_assessment', {})
        
        comprehensive_response = {
            "success": True,
            "agent": "risk_analysis",
            "symbol": req.symbol,
            "exchange": req.exchange,
            "timestamp": datetime.now().isoformat(),
            "processing_time": total_time,
            
            # Quantitative Risk Analysis Results
            "quantitative_analysis": {
                "success": "error" not in risk_analysis_result,
                "processing_time": processor_time,
                "advanced_metrics": advanced_metrics,
                "stress_testing": risk_analysis_result.get('stress_testing', {}),
                "scenario_analysis": risk_analysis_result.get('scenario_analysis', {}),
                "overall_assessment": overall_assessment,
                "error": risk_analysis_result.get('error', None)
            },
            
            # LLM Analysis Results (Enhanced 15-bullet format)
            "llm_analysis": {
                "success": llm_success,
                "processing_time": llm_time,
                "enhanced_risk_bullets": risk_llm_analysis.get('risk_bullets', '') if llm_success else '',
                "prompt_length": risk_llm_analysis.get('prompt_length', 0) if llm_success else 0,
                "response_length": risk_llm_analysis.get('response_length', 0) if llm_success else 0,
                "confidence": risk_llm_analysis.get('confidence', 0.0) if llm_success else 0.0,
                "error": risk_llm_analysis.get('error', None) if not llm_success else None
            },
            
            # Summary metrics for final decision agent integration
            "risk_summary": {
                "overall_risk_score": advanced_metrics.get('risk_score', 0),
                "overall_risk_level": advanced_metrics.get('risk_level', 'Medium'),
                "combined_risk_score": overall_assessment.get('combined_risk_score', 0),
                "key_risk_factors": overall_assessment.get('key_risk_factors', []),
                "sharpe_ratio": advanced_metrics.get('sharpe_ratio', 0),
                "max_drawdown": advanced_metrics.get('max_drawdown', 0),
                "var_95": advanced_metrics.get('var_95', 0),
                "stress_level": risk_analysis_result.get('stress_testing', {}).get('stress_level', 'Medium')
            },
            
            # For final decision agent integration
            "risk_bullets_for_decision": risk_llm_analysis.get('risk_bullets', '') if llm_success else None,
            
            # Optional: Return prompt if requested
            "prompt_details": {
                "returned": req.return_prompt and llm_success,
                "prompt_content": risk_llm_analysis.get('generated_prompt', '') if req.return_prompt and llm_success else None
            } if req.return_prompt else None,
            
            # Metadata
            "request_metadata": {
                "timeframes": req.timeframes,
                "correlation_id": req.correlation_id,
                "used_prefetched_data": bool(req.correlation_id and cached)
            }
        }
        
        # Ensure JSON serializable
        serializable_response = make_json_serializable(comprehensive_response)
        
        print(f"[RISK_AGENT] ✅ Comprehensive risk analysis completed for {req.symbol} in {total_time:.2f}s")
        print(f"[RISK_AGENT] - Quantitative: {processor_time:.2f}s, LLM: {llm_time:.2f}s")
        print(f"[RISK_AGENT] - Risk Level: {advanced_metrics.get('risk_level', 'Unknown')}, Score: {advanced_metrics.get('risk_score', 0)}")
        
        return JSONResponse(content=serializable_response, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.monotonic() - start_time
        error_msg = f"Risk analysis failed for {req.symbol}: {str(e)}"
        print(f"[RISK_AGENT] ❌ {error_msg} (after {total_time:.2f}s)")
        
        return JSONResponse(
            content={
                "success": False,
                "error": error_msg,
                "agent": "risk_analysis",
                "symbol": req.symbol,
                "processing_time": total_time,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

# ===== MARKET STRUCTURE AGENT ENDPOINT =====

@app.post("/agents/market-structure/analyze")
async def agents_market_structure_analyze(req: MarketStructureRequest):
    """
    Standalone Market Structure Analysis Agent endpoint.
    
    Provides comprehensive market structure analysis including:
    - Swing points detection and analysis
    - BOS (Break of Structure) and CHOCH (Change of Character) events
    - Trend structure analysis
    - Support and resistance levels from structure
    - Fractal analysis
    - LLM-enhanced structural insights
    - Multi-modal chart analysis
    
    This is the primary endpoint for market structure analysis, similar to
    the cross-validation agent pattern.
    """
    start_time = time.monotonic()
    print(f"[MARKET_STRUCTURE_AGENT] Starting standalone analysis for {req.symbol}")
    
    try:
        # Attempt to reuse prefetched data if provided via correlation_id
        stock_data = None
        if req.correlation_id:
            try:
                cached = VOLUME_PREFETCH_CACHE.get(req.correlation_id, None)
                if cached and isinstance(cached, dict):
                    stock_data = cached.get('stock_data')
                    print(f"[MARKET_STRUCTURE_AGENT] Using prefetched data for correlation_id={req.correlation_id}")
            except Exception as cache_e:
                print(f"[MARKET_STRUCTURE_AGENT] Error retrieving prefetched data: {cache_e}")
        
        # Retrieve stock data only if not provided
        if stock_data is None:
            try:
                orchestrator = StockAnalysisOrchestrator()
                stock_data = await orchestrator.retrieve_stock_data(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    interval=req.interval,
                    period=req.period
                )
                print(f"[MARKET_STRUCTURE_AGENT] Retrieved {len(stock_data)} days of data for {req.symbol}")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Data retrieval failed: {str(e)}")
        
        # Run market structure agent with configurable options
        from agents.patterns.market_structure_agent.agent import MarketStructureAgent
        agent = MarketStructureAgent()
        
        # Use analyze_complete for full analysis
        result_data = await agent.analyze_complete(
            stock_data=stock_data, 
            symbol=req.symbol, 
            context=req.context or ""
        )
        
        # Build comprehensive response
        total_time = time.monotonic() - start_time
        
        # Extract key insights for summary
        key_insights = agent.get_key_insights(result_data) if result_data.get('success') else []
        
        comprehensive_response = {
            "success": result_data.get('success', False),
            "agent": "market_structure_analysis",
            "symbol": req.symbol,
            "exchange": req.exchange,
            "timestamp": datetime.now().isoformat(),
            "processing_time": total_time,
            "confidence_score": result_data.get('confidence_score', 0.0),
            
            # Technical Analysis Results
            "technical_analysis": result_data.get('technical_analysis', {}),
            
            # LLM Analysis Results
            "llm_analysis": {
                "success": result_data.get('has_llm_analysis', False),
                "analysis": result_data.get('llm_analysis'),
                "enhanced_insights": bool(result_data.get('llm_analysis'))
            },
            
            # Chart Information
            "chart_info": {
                "has_chart": result_data.get('chart_image') is not None,
                "chart_generated": bool(result_data.get('chart_image')),
                "chart_size_bytes": len(result_data.get('chart_image', b'')) if result_data.get('chart_image') else 0
            },
            
            # Key Insights Summary
            "key_insights": key_insights,
            "insights_count": len(key_insights),
            
            # Agent Information
            "agent_info": result_data.get('agent_info', {}),
            
            # For final decision agent integration (if needed)
            "market_structure_insights_for_decision": {
                "structural_bias": result_data.get('technical_analysis', {}).get('bos_choch_analysis', {}).get('structural_bias', 'unknown'),
                "trend_direction": result_data.get('technical_analysis', {}).get('trend_analysis', {}).get('trend_direction', 'unknown'),
                "trend_strength": result_data.get('technical_analysis', {}).get('trend_analysis', {}).get('trend_strength', 'unknown'),
                "structure_quality": result_data.get('technical_analysis', {}).get('structure_quality', {}),
                "key_levels_count": len(result_data.get('technical_analysis', {}).get('key_levels', {}).get('support_levels', [])) + len(result_data.get('technical_analysis', {}).get('key_levels', {}).get('resistance_levels', [])),
                "confidence_score": result_data.get('confidence_score', 0.0)
            },
            
            # Error handling
            "error": result_data.get('error') if not result_data.get('success') else None,
            
            # Optional: Return prompt if requested
            "prompt_details": {
                "returned": req.return_prompt and result_data.get('has_llm_analysis', False),
                "note": "LLM prompts available in technical analysis results"
            } if req.return_prompt else None,
            
            # Metadata
            "request_metadata": {
                "context": req.context,
                "correlation_id": req.correlation_id,
                "used_prefetched_data": bool(req.correlation_id and cached),
                "include_charts": req.include_charts,
                "include_llm_analysis": req.include_llm_analysis
            }
        }
        
        # Handle errors
        if not result_data.get('success', False):
            comprehensive_response['error'] = result_data.get('error', 'Market structure analysis failed')
        
        # Ensure JSON serializable
        serializable_response = make_json_serializable(comprehensive_response)
        
        success_status = "✅" if result_data.get('success', False) else "❌"
        print(f"[MARKET_STRUCTURE_AGENT] {success_status} Standalone analysis completed for {req.symbol} in {total_time:.2f}s")
        print(f"[MARKET_STRUCTURE_AGENT] - Confidence: {result_data.get('confidence_score', 0)}")
        print(f"[MARKET_STRUCTURE_AGENT] - LLM Analysis: {result_data.get('has_llm_analysis', False)}")
        print(f"[MARKET_STRUCTURE_AGENT] - Chart Generated: {bool(result_data.get('chart_image'))}")
        
        return JSONResponse(content=serializable_response, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.monotonic() - start_time
        error_msg = f"Market structure analysis failed for {req.symbol}: {str(e)}"
        print(f"[MARKET_STRUCTURE_AGENT] ❌ {error_msg} (after {total_time:.2f}s)")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            content={
                "success": False,
                "error": error_msg,
                "agent": "market_structure_analysis",
                "symbol": req.symbol,
                "processing_time": total_time,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

# ===== PATTERN ANALYSIS AGENT ENDPOINTS =====

# ===== PATTERN ANALYSIS AGENT ENDPOINTS =====

@app.post("/agents/patterns/market-structure")
async def agents_patterns_market_structure(req: PatternAnalysisRequest):
    """
    Market Structure Analysis Agent endpoint.
    
    Analyzes market structure including:
    - Swing points detection and analysis
    - BOS (Break of Structure) and CHOCH (Change of Character) events
    - Trend structure analysis
    - Support and resistance levels from structure
    - Fractal analysis
    """
    start_time = time.monotonic()
    print(f"[MARKET_STRUCTURE_AGENT] Starting analysis for {req.symbol}")
    
    try:
        # Attempt to reuse prefetched data if provided via correlation_id
        stock_data = None
        if req.correlation_id:
            try:
                cached = VOLUME_PREFETCH_CACHE.get(req.correlation_id, None)
                if cached and isinstance(cached, dict):
                    stock_data = cached.get('stock_data')
                    print(f"[MARKET_STRUCTURE_AGENT] Using prefetched data for correlation_id={req.correlation_id}")
            except Exception as cache_e:
                print(f"[MARKET_STRUCTURE_AGENT] Error retrieving prefetched data: {cache_e}")
        
        # Retrieve stock data only if not provided
        if stock_data is None:
            try:
                orchestrator = StockAnalysisOrchestrator()
                stock_data = await orchestrator.retrieve_stock_data(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    interval=req.interval,
                    period=req.period
                )
                print(f"[MARKET_STRUCTURE_AGENT] Retrieved {len(stock_data)} days of data for {req.symbol}")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Data retrieval failed: {str(e)}")
        
        # Run market structure agent
        from agents.patterns.market_structure_agent.agent import MarketStructureAgent
        agent = MarketStructureAgent()
        result_data = await agent.analyze_complete(stock_data, req.symbol, req.context)
        
        # Build response
        total_time = time.monotonic() - start_time
        response = {
            "success": result_data.get('success', False),
            "agent": "market_structure",
            "symbol": req.symbol,
            "exchange": req.exchange,
            "timestamp": datetime.now().isoformat(),
            "processing_time": total_time,
            "confidence_score": result_data.get('confidence_score', 0.0),
            "technical_analysis": result_data.get('technical_analysis', {}),
            "llm_analysis": result_data.get('llm_analysis', {}),
            "has_chart": result_data.get('chart_image') is not None,
            "error": result_data.get('error') if not result_data.get('success') else None,
            "metadata": {
                "context": req.context,
                "correlation_id": req.correlation_id,
                "used_prefetched_data": bool(req.correlation_id and stock_data)
            }
        }
        
        print(f"✅ [MARKET_STRUCTURE_AGENT] Analysis completed for {req.symbol} in {total_time:.2f}s")
        return JSONResponse(content=make_json_serializable(response), status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.monotonic() - start_time
        print(f"❌ [MARKET_STRUCTURE_AGENT] Analysis failed for {req.symbol}: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "agent": "market_structure",
                "symbol": req.symbol,
                "processing_time": total_time,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

@app.post("/agents/patterns/cross-validation")
async def agents_patterns_cross_validation(req: PatternAnalysisRequest):
    """
    Cross-Validation Pattern Analysis Agent endpoint.
    
    Performs:
    - Pattern detection (triangles, flags, channels, head & shoulders, double patterns)
    - Multi-method pattern validation
    - Confidence assessment
    - LLM-powered validation insights
    """
    start_time = time.monotonic()
    print(f"[CROSS_VALIDATION_AGENT] Starting analysis for {req.symbol}")
    
    try:
        # Attempt to reuse prefetched data if provided via correlation_id
        stock_data = None
        if req.correlation_id:
            try:
                cached = VOLUME_PREFETCH_CACHE.get(req.correlation_id, None)
                if cached and isinstance(cached, dict):
                    stock_data = cached.get('stock_data')
                    print(f"[CROSS_VALIDATION_AGENT] Using prefetched data for correlation_id={req.correlation_id}")
            except Exception as cache_e:
                print(f"[CROSS_VALIDATION_AGENT] Error retrieving prefetched data: {cache_e}")
        
        # Retrieve stock data only if not provided
        if stock_data is None:
            try:
                orchestrator = StockAnalysisOrchestrator()
                stock_data = await orchestrator.retrieve_stock_data(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    interval=req.interval,
                    period=req.period
                )
                print(f"[CROSS_VALIDATION_AGENT] Retrieved {len(stock_data)} days of data for {req.symbol}")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Data retrieval failed: {str(e)}")
        
        # Run cross-validation agent (includes pattern detection)
        from agents.patterns.cross_validation_agent.agent import CrossValidationAgent
        agent = CrossValidationAgent()
        result_data = await agent.analyze_and_validate_patterns(
            stock_data=stock_data,
            symbol=req.symbol,
            include_charts=True,
            include_llm_analysis=True
        )
        
        # Build response
        total_time = time.monotonic() - start_time
        response = {
            "success": result_data.get('success', False),
            "agent": "cross_validation",
            "symbol": req.symbol,
            "exchange": req.exchange,
            "timestamp": datetime.now().isoformat(),
            "processing_time": total_time,
            "pattern_detection": result_data.get('pattern_detection', {}),
            "cross_validation": result_data.get('cross_validation', {}),
            "llm_analysis": result_data.get('llm_analysis', {}),
            "components_executed": result_data.get('components_executed', []),
            "error": result_data.get('error') if not result_data.get('success') else None,
            "metadata": {
                "context": req.context,
                "correlation_id": req.correlation_id,
                "used_prefetched_data": bool(req.correlation_id and stock_data)
            }
        }
        
        print(f"✅ [CROSS_VALIDATION_AGENT] Analysis completed for {req.symbol} in {total_time:.2f}s")
        return JSONResponse(content=make_json_serializable(response), status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.monotonic() - start_time
        print(f"❌ [CROSS_VALIDATION_AGENT] Analysis failed for {req.symbol}: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "agent": "cross_validation",
                "symbol": req.symbol,
                "processing_time": total_time,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

@app.post("/agents/patterns/analyze-all")
async def agents_patterns_analyze_all(req: PatternAnalysisRequest):
    """
    Comprehensive Pattern Analysis endpoint - runs all pattern agents.
    
    This endpoint runs both pattern agents concurrently:
    - Market Structure Agent: swing points, BOS/CHOCH, trend structure
    - Cross-Validation Agent: pattern detection and validation
    
    Provides aggregated results with consensus signals and conflict detection.
    """
    start_time = time.monotonic()
    print(f"[PATTERN_AGENTS] Starting comprehensive pattern analysis for {req.symbol}")
    
    try:
        # Attempt to reuse prefetched data if provided via correlation_id
        stock_data = None
        indicators = None
        if req.correlation_id:
            try:
                cached = VOLUME_PREFETCH_CACHE.get(req.correlation_id, None)
                if cached and isinstance(cached, dict):
                    stock_data = cached.get('stock_data')
                    indicators = cached.get('indicators')
                    print(f"[PATTERN_AGENTS] Using prefetched data for correlation_id={req.correlation_id}")
            except Exception as cache_e:
                print(f"[PATTERN_AGENTS] Error retrieving prefetched data: {cache_e}")
        
        # Retrieve stock data only if not provided
        if stock_data is None:
            try:
                orchestrator = StockAnalysisOrchestrator()
                stock_data = await orchestrator.retrieve_stock_data(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    interval=req.interval,
                    period=req.period
                )
                print(f"[PATTERN_AGENTS] Retrieved {len(stock_data)} days of data for {req.symbol}")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Data retrieval failed: {str(e)}")
        
        # Calculate indicators only if not provided
        if indicators is None:
            try:
                indicators = TechnicalIndicators.calculate_all_indicators_optimized(stock_data, req.symbol)
                print(f"[PATTERN_AGENTS] Calculated {len(indicators)} indicators for {req.symbol}")
            except Exception as ind_e:
                indicators = {}
                print(f"[PATTERN_AGENTS] Warning: indicator calculation failed: {ind_e}")
        
        # Run comprehensive pattern analysis using the integration manager
        from agents.patterns.pattern_agents import PatternAgentIntegrationManager
        manager = PatternAgentIntegrationManager()
        
        pattern_results = await manager.get_comprehensive_pattern_analysis(
            stock_data=stock_data,
            symbol=req.symbol,
            context=req.context,
            include_charts=True,
            include_llm_analysis=True
        )
        
        # Build comprehensive response
        total_time = time.monotonic() - start_time
        
        comprehensive_response = {
            "success": pattern_results.get('success', False),
            "agent": "pattern_analysis_all",
            "symbol": req.symbol,
            "exchange": req.exchange,
            "timestamp": datetime.now().isoformat(),
            "processing_time": total_time,
            "overall_confidence": pattern_results.get('overall_confidence', 0.0),
            
            # Individual agent results
            "market_structure_analysis": pattern_results.get('market_structure_analysis', {}),
            "cross_validation_analysis": pattern_results.get('cross_validation_analysis', {}),
            
            # Aggregated insights
            "consensus_signals": pattern_results.get('consensus_signals', {}),
            "pattern_conflicts": pattern_results.get('pattern_conflicts', []),
            "unified_analysis": pattern_results.get('unified_analysis', {}),
            "agents_summary": pattern_results.get('agents_summary', {}),
            
        # For final decision agent integration
        "pattern_insights_for_decision": _extract_pattern_insights_for_decision(pattern_results),
            
            # Metadata
            "request_metadata": {
                "context": req.context,
                "correlation_id": req.correlation_id,
                "used_prefetched_data": bool(req.correlation_id and cached)
            },
            
            # Optional: Return prompt details if requested
            "prompt_details": {
                "returned": req.return_prompt,
                "note": "Individual agent prompts available in agent-specific results"
            } if req.return_prompt else None
        }
        
        # Handle errors
        if not pattern_results.get('success', False):
            comprehensive_response['error'] = pattern_results.get('error', 'Pattern analysis failed')
        
        # Ensure JSON serializable
        serializable_response = make_json_serializable(comprehensive_response)
        
        success_status = "✅" if pattern_results.get('success', False) else "❌"
        print(f"[PATTERN_AGENTS] {success_status} Comprehensive analysis completed for {req.symbol} in {total_time:.2f}s")
        print(f"[PATTERN_AGENTS] - Overall Confidence: {pattern_results.get('overall_confidence', 0):.2%}")
        print(f"[PATTERN_AGENTS] - Agents Success: {pattern_results.get('agents_summary', {}).get('success_rate', 0):.1%}")
        
        return JSONResponse(content=serializable_response, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.monotonic() - start_time
        error_msg = f"Comprehensive pattern analysis failed for {req.symbol}: {str(e)}"
        print(f"[PATTERN_AGENTS] ❌ {error_msg} (after {total_time:.2f}s)")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            content={
                "success": False,
                "error": error_msg,
                "agent": "pattern_analysis_all",
                "symbol": req.symbol,
                "processing_time": total_time,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

# ===== USER ANALYSIS ENDPOINTS - REMOVED, NOW IN DATABASE SERVICE =====
# All endpoints below are commented out as they are moved to database_service.py

# @app.get("/analyses/user/{user_id}")
# async def get_user_analyses(user_id: str, limit: int = 50):
#     """Get analysis history for a user."""
#     try:
#         ...
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to fetch user analyses: {str(e)}")

# @app.get("/analyses/{analysis_id}")
# async def get_analysis_by_id(analysis_id: str):
#     """Get a specific analysis by ID."""
#     try:
#         ...
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to fetch analysis: {str(e)}")

# @app.get("/analyses/signal/{signal}")
# async def get_analyses_by_signal(signal: str, user_id: Optional[str] = None, limit: int = 20):
#     """Get analyses filtered by signal type."""
#     try:
#         ...
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to fetch analyses by signal: {str(e)}")

# @app.get("/analyses/sector/{sector}")
# async def get_analyses_by_sector(sector: str, user_id: Optional[str] = None, limit: int = 20):
#     """Get analyses filtered by sector."""
#     try:
#         ...
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to fetch analyses by sector: {str(e)}")

# @app.get("/analyses/confidence/{min_confidence}")
# async def get_high_confidence_analyses(min_confidence: float = 80.0, user_id: Optional[str] = None, limit: int = 20):
#     """Get analyses with confidence above threshold."""
#     try:
#         ...
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to fetch high confidence analyses: {str(e)}")

# @app.get("/analyses/summary/user/{user_id}")
# async def get_user_analysis_summary(user_id: str):
#     """Get analysis summary for a user."""
#     try:
#         ...
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to fetch user analysis summary: {str(e)}")

# Chart Management Endpoints
@app.get("/charts/storage/stats")
async def get_chart_storage_stats():
    """Get chart storage statistics."""
    try:
        chart_manager = get_chart_manager()
        stats = chart_manager.get_storage_stats()
        
        # Redis image stats removed - charts are now generated in-memory
        return {
            "success": True,
            "file_storage_stats": stats,
            "redis_storage_stats": {"status": "removed", "reason": "Charts generated in-memory"}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chart storage stats: {str(e)}")

@app.post("/charts/cleanup")
async def cleanup_charts():
    """Manually trigger chart cleanup."""
    try:
        chart_manager = get_chart_manager()
        file_stats = chart_manager.cleanup_old_charts()
        
        # Redis image cleanup removed - charts are now generated in-memory
        return {
            "success": True,
            "message": "Chart cleanup completed",
            "file_cleanup_stats": file_stats,
            "redis_cleanup_stats": {"status": "removed", "reason": "Charts generated in-memory"}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup charts: {str(e)}")

@app.delete("/charts/{symbol}/{interval}")
async def cleanup_specific_charts(symbol: str, interval: str):
    """Clean up charts for a specific symbol and interval."""
    try:
        chart_manager = get_chart_manager()
        file_success = chart_manager.cleanup_specific_charts(symbol, interval)
        
        # Redis image cleanup removed - charts are now generated in-memory
        if file_success:
            return {
                "success": True,
                "message": f"Cleaned up charts for {symbol}_{interval}",
                "file_cleanup": file_success,
                "redis_cleanup": {"status": "removed", "reason": "Charts generated in-memory"}
            }
        else:
            return {
                "success": False,
                "message": f"No charts found for {symbol}_{interval}"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup charts for {symbol}_{interval}: {str(e)}")

@app.delete("/charts/all")
async def cleanup_all_charts():
    """Clean up all charts."""
    try:
        chart_manager = get_chart_manager()
        stats = chart_manager.cleanup_all_charts()
        return {
            "success": True,
            "message": "All charts cleaned up",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup all charts: {str(e)}")

# Redis Image Management Endpoints removed - charts are now generated in-memory
# Only Redis image storage functionality has been removed, Redis is still used for data caching
# Redis cache is important for sector analysis, stock data caching, and other performance optimizations

# Redis Cache Management Endpoints
# Note: Redis is still used for data caching, but not for image storage

@app.get("/redis/cache/stats")
async def get_redis_cache_stats():
    """Get Redis cache statistics."""
    try:
        from core.redis_unified_cache_manager import get_unified_redis_cache_manager
        cache_manager = get_unified_redis_cache_manager()
        stats = cache_manager.get_stats()
        return {
            "success": True,
            "stats": stats,
            "note": "Redis is used for data caching, but not for image storage"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Redis cache stats: {str(e)}")

@app.post("/redis/cache/clear")
async def clear_redis_cache(data_type: str = None):
    """Clear Redis cache entries."""
    try:
        from core.redis_unified_cache_manager import get_unified_redis_cache_manager
        cache_manager = get_unified_redis_cache_manager()
        deleted_counts = cache_manager.clear(data_type)
        return {
            "success": True,
            "message": f"Redis cache cleared for {data_type if data_type else 'all data types'}",
            "deleted_counts": deleted_counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear Redis cache: {str(e)}")

@app.delete("/redis/cache/stock/{symbol}")
async def clear_stock_cache(symbol: str, exchange: str = "NSE"):
    """Clear cache for a specific stock."""
    try:
        from core.redis_unified_cache_manager import clear_stock_cache
        stats = clear_stock_cache(symbol, exchange)
        return {
            "success": True,
            "message": f"Cache cleared for {symbol}",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear stock cache: {str(e)}")

@app.get("/redis/cache/stock/{symbol}")
async def get_cached_stock_data(symbol: str, exchange: str = "NSE", interval: str = "day", period: int = 365):
    """Get cached stock data."""
    try:
        from core.redis_unified_cache_manager import get_cached_stock_data
        data = get_cached_stock_data(symbol, exchange, interval, period)
        if data is not None:
            return {
                "success": True,
                "data": data.to_dict('records') if hasattr(data, 'to_dict') else data,
                "cached": True
            }
        else:
            return {
                "success": False,
                "message": "No cached data found",
                "cached": False
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cached stock data: {str(e)}")

# Storage Management Endpoints
@app.get("/storage/info")
async def get_storage_info():
    """Get comprehensive storage information."""
    try:
        storage_info = StorageConfig.get_storage_info()
        return {
            "success": True,
            "storage_info": storage_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get storage info: {str(e)}")

@app.get("/storage/recommendations")
async def get_storage_recommendations():
    """Get storage recommendations for current environment."""
    try:
        from config.storage_config import get_storage_recommendations
        recommendations = get_storage_recommendations()
        return {
            "success": True,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get storage recommendations: {str(e)}")

# Root route
@app.get("/")
async def root():
    """Root endpoint for the Analysis Service."""
    return {
        "service": "Stock Analysis Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "sector_list": "/sector/list",
            "stock_sector": "/stock/{symbol}/sector",
            "analyses_user": f"{DATABASE_SERVICE_URL}/analyses/user/{{user_id}}" # Points to unified backend
        },
        "timestamp": datetime.now().isoformat()
    }


# General OPTIONS handler to ensure CORS preflight succeeds for all analysis routes
@app.options("/{path:path}")
async def options_any(path: str):
    """Handle OPTIONS preflight for any endpoint under the analysis service."""
    return JSONResponse(
        status_code=200,
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Load environment variables
    # Use PORT env var (provided by Render) if available, otherwise fall back to SERVICE_PORT
    port = int(os.getenv("ANALYSIS_PORT", 8002))
    host = os.getenv("SERVICE_HOST", "0.0.0.0")
    
    print(f"🚀 Starting {os.getenv('SERVICE_NAME', 'Analysis Service')} on {host}:{port}")
    print(f"🔍 Analysis endpoints available at /analyze/*")
    print(f"📊 Technical indicators available at /stock/*/indicators")
    print(f"🏭 Sector analysis available at /sector/*")
    print(f"📈 Pattern recognition available at /patterns/*")
    
    uvicorn.run(app, host=host, port=port)