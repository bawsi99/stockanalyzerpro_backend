# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os
from agent_capabilities import StockAnalysisOrchestrator
import traceback
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import numpy as np
import io
import base64
from pandas import Timestamp
import math
from fastapi.responses import JSONResponse

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isinf(obj) or np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Timestamp):
            return obj.isoformat()
        return super().default(obj)

def make_json_serializable(obj):
    """Recursively convert objects to JSON serializable format."""
    if isinstance(obj, (str, int, bool, type(None))):
        return obj
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
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
    elif isinstance(obj, Timestamp):
        return obj.isoformat()
    else:
        return str(obj)

def convert_charts_to_base64(charts_dict: dict) -> dict:
    """Convert chart file paths to base64 encoded images."""
    converted_charts = {}
    
    for chart_name, chart_path in charts_dict.items():
        if isinstance(chart_path, str) and os.path.exists(chart_path):
            try:
                with open(chart_path, 'rb') as f:
                    img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    converted_charts[chart_name] = {
                        'data': f"data:image/png;base64,{img_base64}",
                        'filename': os.path.basename(chart_path),
                        'type': 'image/png'
                    }
            except Exception as e:
                print(f"Error converting chart {chart_name}: {e}")
                converted_charts[chart_name] = {
                    'error': f"Failed to load chart: {str(e)}",
                    'filename': os.path.basename(chart_path) if isinstance(chart_path, str) else 'unknown'
                }
        else:
            converted_charts[chart_name] = {
                'error': 'Chart file not found',
                'path': chart_path
            }
    
    return converted_charts

def validate_analysis_results(results: dict) -> dict:
    """Validate and ensure all required fields are present in analysis results."""
    required_fields = {
        'consensus': {},
        'indicators': {},
        'charts': {},
        'ai_analysis': {},
        'indicator_summary_md': '',
        'chart_insights': '',
        'summary': {}
    }
    
    validated_results = {}
    
    for field, default_value in required_fields.items():
        if field in results and results[field] is not None:
            validated_results[field] = results[field]
        else:
            validated_results[field] = default_value
            print(f"Warning: Missing or null field '{field}' in analysis results")
    
    return validated_results

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class AnalysisRequest(BaseModel):
    stock: str
    exchange: str = "NSE"
    period: int = 365
    interval: str = Field("day", pattern="^(minute|3minute|5minute|10minute|15minute|30minute|60minute|day)$")
    output: Optional[str] = None

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    output_dir = request.output or f"./output/{request.stock}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        orchestrator = StockAnalysisOrchestrator()
        # Authenticate
        auth_success = orchestrator.authenticate()
        if not auth_success:
            raise HTTPException(status_code=401, detail="Failed to authenticate with Zerodha API")

        # Analyze stock
        results, data = await orchestrator.analyze_stock(
            symbol=request.stock,
            exchange=request.exchange,
            period=request.period,
            interval=request.interval,
            output_dir=output_dir
        )

        # Convert data to JSON-serializable format
        data_dict = data.reset_index().to_dict(orient="records") if data is not None else []

        # Make all data JSON serializable
        serialized_results = make_json_serializable(results)
        serialized_data = make_json_serializable(data_dict)

        # Clean, efficient response
        response = {
            "success": True,
            "stock_symbol": request.stock,
            "exchange": request.exchange,
            "analysis_period": f"{request.period} days",
            "interval": request.interval,
            "timestamp": pd.Timestamp.now().isoformat(),
            "results": serialized_results,
            "data": serialized_data
        }

        return JSONResponse(content=response)

    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "stock_symbol": request.stock,
            "exchange": request.exchange,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()}

@app.get("/stock/{symbol}/info")
async def get_stock_info(symbol: str, exchange: str = "NSE"):
    """Get basic stock information."""
    try:
        orchestrator = StockAnalysisOrchestrator()
        
        if not orchestrator.authenticate():
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        # Get current quote
        quote = orchestrator.data_client.get_quote(symbol, exchange)
        
        if quote:
            return {
                "success": True,
                "symbol": symbol,
                "exchange": exchange,
                "quote": quote,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        else:
            return {
                "success": False,
                "symbol": symbol,
                "exchange": exchange,
                "error": "Quote not available",
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
    except Exception as e:
        return {
            "success": False,
            "symbol": symbol,
            "exchange": exchange,
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }