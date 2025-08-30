"""
analysis_service.py

Analysis Service - Handles all analysis, AI processing, and chart generation.
This service is responsible for:
- Stock analysis and AI processing
- Technical indicator calculations
- Chart generation and visualization
- Sector analysis and benchmarking
- Pattern recognition
- Real-time analysis callbacks
"""

import os
import time
import json
import asyncio
import traceback
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# Try to import optional dependencies
try:
    import dotenv
    dotenv.load_dotenv()
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
from agent_capabilities import StockAnalysisOrchestrator
from analysis_storage import store_analysis_in_supabase
from sector_benchmarking import sector_benchmarking_provider
from sector_classifier import sector_classifier
from enhanced_sector_classifier import enhanced_sector_classifier
from patterns.recognition import PatternRecognition
from technical_indicators import TechnicalIndicators
from database_manager import DatabaseManager
from frontend_response_builder import FrontendResponseBuilder
from chart_manager import get_chart_manager, initialize_chart_manager
from deployment_config import DeploymentConfig
from storage_config import StorageConfig

app = FastAPI(title="Stock Analysis Service", version="1.0.0")
logger = logging.getLogger(__name__)

# Initialize database manager
simple_db_manager = DatabaseManager()

# Load CORS origins from environment variable
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:8080,http://127.0.0.1:5173,https://stock-analyzer-pro.vercel.app,https://stock-analyzer-pro-git-prototype-aaryan-manawats-projects.vercel.app,https://stock-analyzer-cl9o3tivx-aaryan-manawats-projects.vercel.app,https://stockanalyzer-pro.vercel.app").split(",")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]

# --- ML endpoints for CatBoost training and prediction ---
from pydantic import BaseModel

class MLTrainRequest(BaseModel):
    days: int | None = None
    patterns: list[str] | None = None

class MLPredictRequest(BaseModel):
    features: dict
    pattern_type: str | None = None

@app.post("/ml/train")
async def ml_train(req: MLTrainRequest):
    try:
        # Lazy import to avoid heavy import-time overhead
        from ml.dataset import build_pooled_dataset
        from ml.model import train_global_model
        ds = build_pooled_dataset()
        rep = train_global_model(ds)
        if rep is None:
            raise HTTPException(status_code=500, detail="Training failed or no data")
        return {
            "model_path": rep.model_path,
            "trained_at": rep.trained_at,
            "metrics": rep.metrics,
            "feature_schema": rep.feature_schema,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/ml/train failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/model")
async def ml_model():
    try:
        from ml.model import load_registry
        reg = load_registry() or {}
        if not reg:
            return {"status": "unavailable"}
        return reg
    except Exception as e:
        logger.error(f"/ml/model failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/predict")
async def ml_predict(req: MLPredictRequest):
    try:
        from ml.inference import predict_probability, get_model_version, get_pattern_prediction_breakdown
        p = predict_probability(req.features or {}, req.pattern_type)
        return {"probability": p, "model_version": get_model_version()}
    except Exception as e:
        logger.error(f"/ml/predict failed: {e}")
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

@app.on_event("startup")
async def startup_event():
    """Initialize the analysis service on startup."""
    global MAIN_EVENT_LOOP
    MAIN_EVENT_LOOP = asyncio.get_running_loop()
    
    print("🚀 Starting Analysis Service...")
    
    # Initialize Zerodha data client for historical data (no WebSocket needed)
    try:
        from zerodha_client import ZerodhaDataClient
        
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
        print("📊 Charts are generated in-memory - no Redis image storage needed")
        print("ℹ️  Falling back to file-based chart storage")
        
        # Initialize Redis cache manager
        print("💾 Initializing Redis cache manager...")
        try:
            redis_cache_config = DeploymentConfig.get_redis_cache_config()
            from redis_cache_manager import initialize_redis_cache_manager
            redis_cache_manager = initialize_redis_cache_manager(**redis_cache_config)
            print(f"✅ Redis cache manager initialized: compression={redis_cache_manager.enable_compression}")
        except Exception as cache_e:
            print(f"⚠️  Warning: Could not initialize Redis cache manager: {cache_e}")
            print("ℹ️  Falling back to local caching")
        
        # Initialize storage configuration
        print("📁 Initializing storage configuration...")
        StorageConfig.ensure_directories_exist()
        storage_info = StorageConfig.get_storage_info()
        print(f"✅ Storage initialized: {storage_info['storage_type']} storage in {storage_info['environment']} environment")
        
        # Initialize sector classifiers
        print("🏭 Initializing sector classifiers...")
        sector_classifier.get_all_sectors()
        enhanced_sector_classifier.get_all_sectors()
        print("✅ Sector classifiers initialized")
    except Exception as e:
        print(f"⚠️  Warning during initialization: {e}")

    # Log currently active signals weighting profiles (for transparency)
    try:
        from signals.config import load_timeframe_weights
        default_weights = load_timeframe_weights()
        trending_weights = load_timeframe_weights(regime="trending")
        ranging_weights = load_timeframe_weights(regime="ranging")
        print(f"⚖️  Signals timeframe weights (default): {default_weights}")
        if trending_weights:
            print(f"⚖️  Signals timeframe weights (trending profile): {trending_weights}")
        if ranging_weights:
            print(f"⚖️  Signals timeframe weights (ranging profile): {ranging_weights}")
    except Exception as e:
        print(f"⚠️  Warning: Could not load signals weighting profiles: {e}")

    # Do not kick calibration immediately to avoid duplicate runs with the scheduler
    if os.environ.get("ENABLE_SCHEDULED_CALIBRATION") == "1":
        print("⏲️  Weekly calibration scheduler enabled")
    else:
        print("ℹ️  Scheduled calibration disabled (set ENABLE_SCHEDULED_CALIBRATION=1 to enable)")

    # Start background weekly scheduler
    async def _weekly_scheduler():
        try:
            await asyncio.sleep(5)
            week_seconds = 7 * 24 * 60 * 60
            while True:
                try:
                    if os.environ.get("ENABLE_SCHEDULED_CALIBRATION") == "1":
                        result = scheduled_calibration_task()
                        if asyncio.iscoroutine(result):
                            await result
                    else:
                        print("ℹ️  Scheduled calibration disabled (set ENABLE_SCHEDULED_CALIBRATION=1 to enable)")
                except Exception as inner_e:
                    print("[CALIBRATION] Background weekly scheduler error:", inner_e)
                await asyncio.sleep(week_seconds)
        except asyncio.CancelledError:
            return

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_weekly_scheduler())
        print("🔁 Background weekly calibration scheduler started")
    except Exception as e:
        print("⚠️  Warning: Failed to start background weekly scheduler:", e)


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
            "NIFTY_FMCG,NIFTY_ENERGY,NIFTY_METAL,NIFTY_REALTY,NIFTY_MEDIA,"
            "NIFTY_CONSUMER_DURABLES,NIFTY_HEALTHCARE,NIFTY_INFRA,NIFTY_OIL_GAS,"
            "NIFTY_SERV_SECTOR"
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
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize sector classifiers: {e}")
    

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

def make_json_serializable(obj):
    """Recursively convert objects to JSON serializable format."""
    if isinstance(obj, (str, int, type(None))):
        return obj
    elif isinstance(obj, bool):
        return bool(obj)  # Ensure it's a Python bool
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isinf(obj) or np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)  # Convert NumPy boolean to Python boolean
    elif isinstance(obj, Timestamp):
        return obj.isoformat()
    else:
        return str(obj)

def convert_charts_to_base64(charts_dict: dict) -> dict:
    """Convert chart file paths to base64 encoded images."""
    import base64
    converted_charts = {}
    
    for chart_name, chart_path in charts_dict.items():
        # Handle file paths (charts are now generated in-memory)
        if isinstance(chart_path, str) and os.path.exists(chart_path):
            try:
                # Read the image file and convert to base64
                with open(chart_path, 'rb') as f:
                    img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    converted_charts[chart_name] = {
                        'data': f"data:image/png;base64,{img_base64}",
                        'filename': os.path.basename(chart_path),
                        'type': 'image/png'
                    }
                    # Clear large variables immediately after use
                    del img_data
                    del img_base64
            except Exception as e:
                print(f"Error converting chart {chart_name}: {e}")
                converted_charts[chart_name] = {
                    'error': f"Failed to load chart: {str(e)}",
                    'filename': os.path.basename(chart_path) if isinstance(chart_path, str) else 'unknown'
                }
        elif isinstance(chart_path, dict) and chart_path.get('type') == 'image_bytes' and 'data' in chart_path:
            try:
                # Handle in-memory image data
                img_data = chart_path['data']
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                converted_charts[chart_name] = {
                    'data': f"data:image/png;base64,{img_base64}",
                    'format': chart_path.get('format', 'png'),
                    'type': 'image/png'
                }
                # Clear large variables immediately after use
                del img_data
                del img_base64
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
    
    return converted_charts

def cleanup_chart_files(chart_paths: dict) -> dict:
    """Clean up chart files or Redis keys referenced in chart_paths.

    Returns basic stats about cleanup operations performed.
    """
    stats = {"files_removed": 0, "redis_keys_cleaned": 0, "errors": 0}
    try:
        for chart_name, chart_path in (chart_paths or {}).items():
            try:
                if isinstance(chart_path, str):
                    if chart_path.startswith('chart:'):
                        # This is a Redis key - no cleanup needed as Redis handles expiration
                        stats["redis_keys_cleaned"] += 1
                    elif os.path.exists(chart_path):
                        # Remove only the file; do not remove directories
                        os.remove(chart_path)
                        print(f"✅ Removed chart file: {os.path.basename(chart_path)}")
                        stats["files_removed"] += 1
            except Exception as e:
                print(f"⚠️ Error cleaning up chart {chart_name}: {str(e)}")
                stats["errors"] += 1
        
        # Clear the chart_paths dictionary reference after cleanup
        if chart_paths:
            chart_paths.clear()
            
        return stats
    except Exception as e:
        print(f"⚠️ Error in cleanup_chart_files: {str(e)}")
        stats["errors"] += 1
        return stats

def validate_analysis_results(results: dict) -> dict:
    """Validate and ensure all required fields are present in analysis results."""
    required_fields = {
        'ai_analysis': {},
        'indicators': {},
        'overlays': {},
        'indicator_summary_md': '',
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

def resolve_user_id(user_id: Optional[str] = None, email: Optional[str] = None) -> str:
    """
    Resolve user ID from provided user_id or email.
    Email mapping is the primary method for user identification.
    
    Args:
        user_id: Optional user ID (UUID)
        email: Optional user email for ID mapping
        
    Returns:
        str: Valid user ID (UUID)
        
    Raises:
        ValueError: If no valid user ID can be resolved
    """
    try:
        # If user_id is provided and valid, use it
        if user_id:
            try:
                uuid.UUID(user_id)
                # Ensure user exists
                simple_db_manager.ensure_user_exists(user_id)
                print(f"✅ Using provided user ID: {user_id}")
                return user_id
            except (ValueError, TypeError):
                print(f"⚠️ Invalid user_id format: {user_id}")
        
        # If email is provided, try to get user ID from email
        if email:
            resolved_user_id = simple_db_manager.get_user_id_by_email(email)
            if resolved_user_id:
                print(f"✅ Resolved user ID from email: {email} -> {resolved_user_id}")
                return resolved_user_id
            else:
                print(f"❌ User not found for email: {email}")
                raise ValueError(f"User not found for email: {email}")
        
        # No user_id or email provided
        raise ValueError("No user_id or email provided for analysis request")
        
    except Exception as e:
        print(f"❌ Error resolving user ID: {e}")
        raise ValueError(f"Failed to resolve user ID: {e}")

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
        
        # Check Gemini API key
        gemini_status = "unknown"
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")
            gemini_status = "configured" if gemini_api_key else "not_configured"
        except Exception:
            gemini_status = "error"
        
        # Check sector classifiers
        sector_status = "unknown"
        try:
            sectors = sector_classifier.get_all_sectors()
            sector_status = f"loaded_{len(sectors)}_sectors"
        except Exception:
            sector_status = "error"
        
        return {
            "status": "healthy",
            "service": "Stock Analysis Service",
            "timestamp": pd.Timestamp.now().isoformat(),
            "components": {
                "zerodha_data_client": zerodha_status,
                "gemini_ai": gemini_status,
                "sector_classifiers": sector_status,
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
    This endpoint provides more accurate analysis by performing actual calculations
    instead of relying on LLM estimation.
    """
    try:
        print(f"[ENHANCED ANALYSIS] Starting enhanced analysis for {request.stock}")
        
        # Create orchestrator instance
        orchestrator = StockAnalysisOrchestrator()
        
        # Perform enhanced analysis with code execution
        result = await orchestrator.enhanced_analyze_stock(
            symbol=request.stock,
            exchange=request.exchange,
            period=request.period,
            interval=request.interval,
            output_dir=request.output,
            sector=request.sector
        )
        
        # Extract components from the result
        analysis_results, success_message, error_message = result
        
        if error_message:
            raise HTTPException(status_code=500, detail=error_message)
        
        # Extract data and indicators from the analysis results
        # The enhanced_analyze_stock method already retrieves data and calculates indicators
        try:
            # Get the data again for building frontend response
            stock_data = await orchestrator.retrieve_stock_data(
                request.stock, request.exchange, request.interval, request.period
            )
            
            # Extract indicators from analysis_results if available, otherwise calculate them
            if analysis_results and 'technical_indicators' in analysis_results:
                indicators = analysis_results['technical_indicators']
            else:
                indicators = TechnicalIndicators.calculate_all_indicators_optimized(stock_data, request.stock)
                
        except ValueError as e:
            error_msg = f"Data retrieval failed for {request.stock}: {str(e)}"
            print(f"[ENHANCED ANALYSIS ERROR] {error_msg}")
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
            error_msg = f"Technical indicator calculation failed for {request.stock}: {str(e)}"
            print(f"[ENHANCED ANALYSIS ERROR] {error_msg}")
            return JSONResponse(
                content={
                    "success": False,
                    "error": error_msg,
                    "stock_symbol": request.stock,
                    "suggestion": "Unable to calculate technical indicators. Please try again later."
                },
                status_code=500
            )
        
        # Create visualizations (store in Redis)
        chart_paths = orchestrator.create_visualizations(stock_data, indicators, request.stock, request.output, request.interval)
        
        # Get sector context (auto-detect sector if not provided)
        sector_context = None
        try:
            # Determine sector to use
            detected_sector = None
            try:
                if not request.sector:
                    from sector_classifier import sector_classifier as _sc
                    detected_sector = _sc.get_stock_sector(request.stock)
                    # Clear module reference when done
                    del _sc
            except Exception:
                detected_sector = None

            # Use sector benchmarking provider directly
            from sector_benchmarking import sector_benchmarking_provider

            # Launch sector tasks in parallel for reduced latency
            print(f"🔄 Starting parallel sector analysis tasks for {request.stock}")
            benchmarking_task = sector_benchmarking_provider.get_comprehensive_benchmarking_async(
                request.stock,
                stock_data,
                user_sector=request.sector
            )
            rotation_task = sector_benchmarking_provider.analyze_sector_rotation_async("1M")
            correlation_task = sector_benchmarking_provider.generate_sector_correlation_matrix_async("3M")

            sector_benchmarking, sector_rotation, sector_correlation = await asyncio.gather(
                benchmarking_task,
                rotation_task,
                correlation_task,
                return_exceptions=True,
            )

            # Robust fallback handling
            if isinstance(sector_benchmarking, Exception):
                print(f"Warning: sector_benchmarking failed: {sector_benchmarking}")
                sector_benchmarking = {}
            if isinstance(sector_rotation, Exception):
                print(f"Warning: sector_rotation failed: {sector_rotation}")
                sector_rotation = None
            if isinstance(sector_correlation, Exception):
                print(f"Warning: sector_correlation failed: {sector_correlation}")
                sector_correlation = None

            # Prefer explicit request.sector, then detected, then fallback from benchmarking payload
            sector_value = request.sector or detected_sector or (
                (sector_benchmarking or {}).get('sector_info', {}).get('sector') if isinstance(sector_benchmarking, dict) else None
            ) or ''

            # Create sector context with only necessary data
            # Extract only the essential information from sector_benchmarking to reduce memory usage
            essential_benchmarking = {}
            if isinstance(sector_benchmarking, dict):
                # Extract only key metrics and summary data
                if 'sector_info' in sector_benchmarking:
                    essential_benchmarking['sector_info'] = sector_benchmarking['sector_info']
                if 'performance_metrics' in sector_benchmarking:
                    essential_benchmarking['performance_metrics'] = sector_benchmarking['performance_metrics']
                if 'summary' in sector_benchmarking:
                    essential_benchmarking['summary'] = sector_benchmarking['summary']
                
                # Clear original large data structure
                sector_benchmarking.clear()
                
            sector_context = {
                'sector_benchmarking': essential_benchmarking,
                'sector_rotation': sector_rotation,
                'sector_correlation': sector_correlation,
                'sector': sector_value
            }

            print(f"✅ Sector context generated successfully for {request.stock}")

        except Exception as e:
            print(f"Warning: Could not get sector context: {e}")
            sector_context = {}
        
        # Get Enhanced MTF context
        mtf_context = None
        try:
            from enhanced_mtf_analysis import enhanced_mtf_analyzer

            # Perform comprehensive multi-timeframe analysis in parallel with sector tasks already done
            mtf_results = await enhanced_mtf_analyzer.comprehensive_mtf_analysis(
                symbol=request.stock,
                exchange=request.exchange
            )

            if mtf_results.get('success', False):
                mtf_context = mtf_results
                print(f"✅ Enhanced MTF analysis generated successfully for {request.stock}")
            else:
                print(f"Warning: Enhanced MTF analysis failed: {mtf_results.get('error', 'Unknown error')}")
                mtf_context = {}

        except Exception as e:
            print(f"Warning: Could not get Enhanced MTF context: {e}")
            mtf_context = {}
        
        # Get Advanced Analysis for Advanced Tab
        advanced_analysis = None
        try:
            from advanced_analysis import advanced_analysis_provider
            advanced_analysis = await advanced_analysis_provider.generate_advanced_analysis(
                stock_data, request.stock, indicators
            )
            print(f"✅ Advanced analysis generated successfully for {request.stock}")
        except Exception as e:
            print(f"Warning: Could not get advanced analysis: {e}")
            advanced_analysis = {}
        
        # Generate ML predictions (price direction, magnitude, volatility, regime)
        ml_predictions = {}
        try:
            from ml.quant_system.ml.unified_manager import unified_ml_manager
            # Train engines that require stock data (raw_data_ml, hybrid) before prediction
            try:
                print(f"🧠 Training ML engines for {request.stock}")
                _ = unified_ml_manager.train_all_engines(stock_data, None)
            except Exception as e:
                print(f"⚠️ Warning: ML engine training failed: {str(e)}")
                pass
                
            # Get predictions
            print(f"🔮 Generating ML predictions for {request.stock}")
            ml_predictions = unified_ml_manager.get_comprehensive_prediction(stock_data)
            print(f"✅ ML predictions generated successfully for {request.stock}")
            
            # Clear any cached training data if possible
            try:
                if hasattr(unified_ml_manager, 'clear_cache') and callable(unified_ml_manager.clear_cache):
                    unified_ml_manager.clear_cache()
                    print(f"🧹 Cleared ML training cache for {request.stock}")
                # If no explicit clear_cache method, try to clean up the most memory-intensive components
                elif hasattr(unified_ml_manager, '_trained_models') and isinstance(unified_ml_manager._trained_models, dict):
                    unified_ml_manager._trained_models.clear()
                    print(f"🧹 Cleared ML model cache for {request.stock}")
            except Exception as cache_e:
                print(f"⚠️ Non-fatal error during ML cache cleanup: {str(cache_e)}")
                
            # Clear module reference when done
            del unified_ml_manager
            
        except Exception as e:
            print(f"Warning: Could not generate ML predictions: {e}")
            ml_predictions = {}
        
        # Build frontend-expected response structure
        frontend_response = FrontendResponseBuilder.build_frontend_response(
            symbol=request.stock,
            exchange=request.exchange,
            data=stock_data,
            indicators=indicators,
            ai_analysis=analysis_results.get('ai_analysis', {}),
            indicator_summary=analysis_results.get('indicator_summary', ''),
            chart_insights=analysis_results.get('chart_insights', ''),
            chart_paths=chart_paths,
            sector_context=sector_context,
            mtf_context=mtf_context,
            advanced_analysis=advanced_analysis,
            ml_predictions=ml_predictions,
            period=request.period,
            interval=request.interval
        )
        
        # Clear large data structures that are no longer needed after building response
        print(f"🧹 Cleaning up memory for {request.stock} analysis...")
        # These structures have been incorporated into frontend_response and are no longer needed
        del sector_context
        del mtf_context
        del advanced_analysis
        del ml_predictions
        
        # Resolve user ID from request
        try:
            resolved_user_id = resolve_user_id(
                user_id=request.user_id,
                email=request.email
            )

            # Store analysis in Supabase using simple database manager (JSON-safe)
            analysis_id = simple_db_manager.store_analysis(
                analysis=make_json_serializable(frontend_response),
                user_id=resolved_user_id,
                symbol=request.stock,
                exchange=request.exchange,
                period=request.period,
                interval=request.interval
            )
            
            if not analysis_id:
                print(f"⚠️ Warning: Failed to store enhanced analysis for {request.stock}")
            else:
                print(f"✅ Successfully stored enhanced analysis for {request.stock} with ID: {analysis_id}")
                
        except ValueError as e:
            print(f"❌ User ID resolution failed: {e}")
            # Continue with analysis but don't store it
            print(f"⚠️ Enhanced analysis completed but not stored due to user ID resolution failure")
        except Exception as e:
            print(f"❌ Error storing enhanced analysis: {e}")
            # Continue with analysis but don't store it
            print(f"⚠️ Enhanced analysis completed but not stored due to storage error")
        
        print(f"[ENHANCED ANALYSIS] Completed enhanced analysis for {request.stock}")
        # Make the response JSON serializable to handle NaN values
        serialized_response = make_json_serializable(frontend_response)
        
        # Clear original response after serialization
        del frontend_response

        # Cleanup generated chart image files after the final response content is prepared
        try:
            # Cleanup charts created in this endpoint response
            if 'results' in serialized_response and isinstance(serialized_response['results'].get('charts'), dict):
                _ = cleanup_chart_files(serialized_response['results']['charts'])
            # Also cleanup any charts created by the earlier enhanced_analyze_stock result
            if isinstance(analysis_results, dict) and isinstance(analysis_results.get('charts'), dict):
                _ = cleanup_chart_files(analysis_results['charts'])
            # Clear analysis results after cleanup
            del analysis_results
        except Exception as e:
            # Non-fatal: cleanup best-effort
            print(f"⚠️ Non-fatal error during chart cleanup: {str(e)}")
            pass

        return JSONResponse(content=serialized_response, status_code=200)
        
    except Exception as e:
        error_msg = f"Enhanced analysis failed for {request.stock}: {str(e)}"
        print(f"[ENHANCED ANALYSIS ERROR] {error_msg}")
        print(f"[ENHANCED ANALYSIS ERROR] Traceback: {traceback.format_exc()}")
        
        # Attempt to clean up any resources that might have been created
        try:
            # Clean up any local variables that might be holding large data
            locals_to_clean = ['stock_data', 'indicators', 'analysis_results', 
                              'sector_context', 'mtf_context', 'advanced_analysis', 
                              'ml_predictions', 'frontend_response']
            
            for var_name in locals_to_clean:
                if var_name in locals():
                    var_value = locals()[var_name]
                    if isinstance(var_value, dict):
                        var_value.clear()
                    elif var_name in locals():
                        del locals()[var_name]
            
            # Clean up any chart files that might have been created
            chart_manager = get_chart_manager()
            chart_dir = chart_manager.get_chart_directory(request.stock, request.interval)
            if os.path.exists(chart_dir):
                print(f"🧹 Cleaning up chart directory for {request.stock}")
                chart_manager.cleanup_specific_charts(request.stock, request.interval)
                
        except Exception as cleanup_e:
            print(f"⚠️ Non-fatal error during error cleanup: {str(cleanup_e)}")
        
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
        
        # Import the enhanced analyzer
        from enhanced_mtf_analysis import enhanced_mtf_analyzer
        
        # Perform comprehensive multi-timeframe analysis
        mtf_results = await enhanced_mtf_analyzer.comprehensive_mtf_analysis(
            symbol=request.stock,
            exchange=request.exchange
        )
        
        if not mtf_results.get('success', False):
            error_msg = f"Enhanced multi-timeframe analysis failed: {mtf_results.get('error', 'Unknown error')}"
            print(f"[ENHANCED MTF ERROR] {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Resolve user ID from request
        try:
            resolved_user_id = resolve_user_id(
                user_id=request.user_id,
                email=request.email
            )

            # Store analysis in Supabase using simple database manager
            analysis_id = simple_db_manager.store_analysis(
                analysis=mtf_results,
                user_id=resolved_user_id,
                symbol=request.stock,
                exchange=request.exchange,
                period=request.period,
                interval=request.interval
            )
            
            if not analysis_id:
                print(f"⚠️ Warning: Failed to store enhanced MTF analysis for {request.stock}")
            else:
                print(f"✅ Successfully stored enhanced MTF analysis for {request.stock} with ID: {analysis_id}")
                
        except ValueError as e:
            print(f"❌ User ID resolution failed: {e}")
            # Continue with analysis but don't store it
            print(f"⚠️ Enhanced MTF analysis completed but not stored due to user ID resolution failure")
        except Exception as e:
            print(f"❌ Error storing enhanced MTF analysis: {e}")
            # Continue with analysis but don't store it
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
    """Get sector benchmarking for a specific stock."""
    try:
        # Get stock data
        orchestrator = StockAnalysisOrchestrator()
        if not orchestrator.authenticate():
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        try:
            data = await orchestrator.retrieve_stock_data(
                symbol=request.stock,
                exchange=request.exchange,
                period=request.period,
                interval=request.interval
            )
        except ValueError as e:
            error_msg = f"Data retrieval failed for {request.stock}: {str(e)}"
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Get comprehensive sector benchmarking
        benchmarking = await sector_benchmarking_provider.get_comprehensive_benchmarking_async(request.stock, data, user_sector=request.sector)
        
        # Make JSON serializable
        serialized_benchmarking = make_json_serializable(benchmarking)
        
        response = {
            "success": True,
            "stock_symbol": request.stock,
            "sector_benchmarking": serialized_benchmarking,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "stock_symbol": request.stock,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.post("/sector/benchmark/async")
async def sector_benchmark_async(request: AnalysisRequest):
    """Get sector benchmarking with async index data fetching."""
    try:
        from sector_benchmarking import SectorBenchmarkingProvider
        
        # Create sector benchmarking provider
        provider = SectorBenchmarkingProvider()
        
        # Get stock data first
        orchestrator = StockAnalysisOrchestrator()
        auth_success = orchestrator.authenticate()
        if not auth_success:
            raise HTTPException(status_code=401, detail="Failed to authenticate with Zerodha API")
        
        try:
            stock_data = await orchestrator.retrieve_stock_data(
                symbol=request.stock,
                exchange=request.exchange,
                interval=request.interval,
                period=request.period
            )
        except ValueError as e:
            error_msg = f"Data retrieval failed for {request.stock}: {str(e)}"
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Get comprehensive benchmarking with async index data
        benchmarking_results = await provider.get_comprehensive_benchmarking_async(
            request.stock, 
            stock_data,
            user_sector=request.sector  # Pass user-provided sector
        )
        
        # Make data JSON serializable
        serialized_results = make_json_serializable(benchmarking_results)
        
        return {
            "success": True,
            "data": serialized_results,
            "analysis_type": "async_sector_benchmarking",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in async sector benchmarking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sector benchmarking failed: {str(e)}")

@app.get("/sector/list")
async def get_sectors():
    """Get list of all available sectors."""
    try:
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
    """Get all stocks in a specific sector."""
    try:
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
    """Get sector performance data."""
    try:
        # Get sector index data
        sector_index = sector_classifier.get_primary_sector_index(sector_name)
        if not sector_index:
            raise HTTPException(status_code=404, detail=f"No index found for sector: {sector_name}")
        
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
            error_msg = f"Data retrieval failed for sector {sector_name}: {str(e)}"
            raise HTTPException(status_code=400, detail=error_msg)
        
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

@app.post("/sector/compare")
async def compare_sectors(request: SectorComparisonRequest):
    """Compare multiple sectors."""
    try:
        comparison_data = {}
        
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
        
        return JSONResponse(content=response)
        
    except Exception as e:
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
    """Get sector information for a specific stock."""
    try:
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

# ===== USER ANALYSIS ENDPOINTS =====

@app.get("/analyses/user/{user_id}")
async def get_user_analyses(user_id: str, limit: int = 50):
    """Get analysis history for a user."""
    try:

        
        # Validate that user_id is not empty
        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id cannot be empty.")
        
        user_id = user_id.strip()
        
        # Check if user_id is a valid UUID
        try:
            uuid.UUID(user_id)
            # If it's a valid UUID, use it directly
            actual_user_id = user_id
        except (ValueError, TypeError):
            # If it's not a UUID, assume it's an email and look up the UUID
            actual_user_id = simple_db_manager.get_user_id_by_email(user_id)
            if not actual_user_id:
                raise HTTPException(status_code=404, detail=f"User not found: {user_id}")
        
        analyses = simple_db_manager.get_user_analyses(actual_user_id, limit)
        return {
            "success": True,
            "analyses": analyses,
            "count": len(analyses)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user analyses: {str(e)}")

@app.get("/analyses/{analysis_id}")
async def get_analysis_by_id(analysis_id: str):
    """Get a specific analysis by ID."""
    try:
        
        # Validate that analysis_id is not empty
        if not analysis_id or not analysis_id.strip():
            raise HTTPException(status_code=400, detail="analysis_id cannot be empty.")
        
        # For now, accept any non-empty string as analysis_id
        analysis_id = analysis_id.strip()
        
        analysis = simple_db_manager.get_analysis_by_id(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")
        
        return {
            "success": True,
            "analysis": analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch analysis: {str(e)}")

@app.get("/analyses/signal/{signal}")
async def get_analyses_by_signal(signal: str, user_id: Optional[str] = None, limit: int = 20):
    """Get analyses filtered by signal type."""
    try:
        
        actual_user_id = None
        if user_id:
            # Validate that user_id is not empty
            if not user_id.strip():
                raise HTTPException(status_code=400, detail="user_id cannot be empty.")
            
            user_id = user_id.strip()
            
            # Check if user_id is a valid UUID
            try:
                uuid.UUID(user_id)
                # If it's a valid UUID, use it directly
                actual_user_id = user_id
            except (ValueError, TypeError):
                # If it's not a UUID, assume it's an email and look up the UUID
                actual_user_id = simple_db_manager.get_user_id_by_email(user_id)
                if not actual_user_id:
                    raise HTTPException(status_code=404, detail=f"User not found: {user_id}")
        
        analyses = simple_db_manager.get_analyses_by_signal(signal, actual_user_id, limit)
        return {
            "success": True,
            "analyses": analyses,
            "count": len(analyses)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch analyses by signal: {str(e)}")

@app.get("/analyses/sector/{sector}")
async def get_analyses_by_sector(sector: str, user_id: Optional[str] = None, limit: int = 20):
    """Get analyses filtered by sector."""
    try:
        
        actual_user_id = None
        if user_id:
            # Validate that user_id is not empty
            if not user_id.strip():
                raise HTTPException(status_code=400, detail="user_id cannot be empty.")
            
            user_id = user_id.strip()
            
            # Check if user_id is a valid UUID
            try:
                uuid.UUID(user_id)
                # If it's a valid UUID, use it directly
                actual_user_id = user_id
            except (ValueError, TypeError):
                # If it's not a UUID, assume it's an email and look up the UUID
                actual_user_id = simple_db_manager.get_user_id_by_email(user_id)
                if not actual_user_id:
                    raise HTTPException(status_code=404, detail=f"User not found: {user_id}")
        
        analyses = simple_db_manager.get_analyses_by_sector(sector, actual_user_id, limit)
        return {
            "success": True,
            "analyses": analyses,
            "count": len(analyses)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch analyses by sector: {str(e)}")

@app.get("/analyses/confidence/{min_confidence}")
async def get_high_confidence_analyses(min_confidence: float = 80.0, user_id: Optional[str] = None, limit: int = 20):
    """Get analyses with confidence above threshold."""
    try:
        
        actual_user_id = None
        if user_id:
            # Validate that user_id is not empty
            if not user_id.strip():
                raise HTTPException(status_code=400, detail="user_id cannot be empty.")
            
            user_id = user_id.strip()
            
            # Check if user_id is a valid UUID
            try:
                uuid.UUID(user_id)
                # If it's a valid UUID, use it directly
                actual_user_id = user_id
            except (ValueError, TypeError):
                # If it's not a UUID, assume it's an email and look up the UUID
                actual_user_id = simple_db_manager.get_user_id_by_email(user_id)
                if not actual_user_id:
                    raise HTTPException(status_code=404, detail=f"User not found: {user_id}")
        
        analyses = simple_db_manager.get_high_confidence_analyses(min_confidence, actual_user_id, limit)
        return {
            "success": True,
            "analyses": analyses,
            "count": len(analyses)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch high confidence analyses: {str(e)}")

@app.get("/analyses/summary/user/{user_id}")
async def get_user_analysis_summary(user_id: str):
    """Get analysis summary for a user."""
    try:
        
        # Validate that user_id is not empty
        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id cannot be empty.")
        
        user_id = user_id.strip()
        
        # Check if user_id is a valid UUID
        try:
            uuid.UUID(user_id)
            # If it's a valid UUID, use it directly
            actual_user_id = user_id
        except (ValueError, TypeError):
            # If it's not a UUID, assume it's an email and look up the UUID
            actual_user_id = simple_db_manager.get_user_id_by_email(user_id)
            if not actual_user_id:
                raise HTTPException(status_code=404, detail=f"User not found: {user_id}")
        
        analyses = simple_db_manager.get_user_analyses(actual_user_id, 50)
        
        # Create summary
        summary = {
            "total_analyses": len(analyses),
            "unique_stocks": len(set(analysis.get("stock_symbol", "") for analysis in analyses)),
            "recent_analyses": analyses[:5] if analyses else [],
            "sectors_analyzed": list(set(analysis.get("sector", "") for analysis in analyses if analysis.get("sector")))
        }
        
        return {
            "success": True,
            "summary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user analysis summary: {str(e)}")

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
# All Redis image storage functionality has been removed as charts are generated on-demand

# Redis Cache Management Endpoints
@app.get("/redis/cache/stats")
async def get_redis_cache_stats():
    """Get Redis cache statistics."""
    try:
        from redis_cache_manager import get_redis_cache_manager
        cache_manager = get_redis_cache_manager()
        stats = cache_manager.get_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Redis cache stats: {str(e)}")

@app.post("/redis/cache/clear")
async def clear_redis_cache(data_type: str = None):
    """Clear Redis cache entries."""
    try:
        from redis_cache_manager import get_redis_cache_manager
        cache_manager = get_redis_cache_manager()
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
        from redis_cache_manager import clear_stock_cache
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
        from redis_cache_manager import get_cached_stock_data
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
        from storage_config import get_storage_recommendations
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
            "analyses_user": "/analyses/user/{user_id}"
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