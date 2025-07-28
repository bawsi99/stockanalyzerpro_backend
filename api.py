# api.py
"""
DEPRECATED: This file is deprecated and should not be used.

The backend has been split into two separate services:
- data_service.py (Port 8000) - Handles data fetching, WebSocket connections, real-time streaming
- analysis_service.py (Port 8001) - Handles analysis, AI processing, chart generation

Please use the new split services instead:
- Start Data Service: python start_data_service.py
- Start Analysis Service: python start_analysis_service.py
- Or use convenience script: python run_services.py

This file will be removed in a future version.
"""

import os
import time
import json
import math
import asyncio
import traceback
import datetime
from typing import Dict, List, Optional, Any

# Try to import optional dependencies
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
    from pandas import Timestamp
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# FastAPI imports
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Local imports
from agent_capabilities import StockAnalysisOrchestrator
from zerodha_ws_client import get_zerodha_ws_client, candle_aggregator
from analysis_storage import store_analysis_in_supabase
# Market hours manager removed - continuous data flow enabled
from enhanced_data_service import enhanced_data_service, DataRequest

# Import sector components
from sector_benchmarking import sector_benchmarking_provider
from sector_classifier import sector_classifier
from enhanced_sector_classifier import enhanced_sector_classifier

# Security configuration
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')
API_KEY_HEADER = 'X-API-Key'
API_KEYS = os.getenv('API_KEYS', '').split(',')  # Comma-separated API keys
REQUIRE_AUTH = os.getenv('REQUIRE_AUTH', 'true').lower() == 'true'

# Simple in-memory token store (use Redis in production)
active_tokens = set()

def create_jwt_token(user_id: str, expires_delta: int = 3600) -> str:
    """Create a JWT token for a user."""
    if not JWT_AVAILABLE:
        raise ImportError("PyJWT library is required but not installed")
        
    payload = {
        'user_id': user_id,
        'exp': time.time() + expires_delta,
        'iat': time.time()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_jwt_token(token: str) -> Optional[Dict]:
    """Verify a JWT token and return payload if valid."""
    if not JWT_AVAILABLE:
        return None
        
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        if payload['exp'] < time.time():
            return None
        return payload
    except jwt.InvalidTokenError:
        return None

def verify_api_key(api_key: str) -> bool:
    """Verify if the API key is valid."""
    return api_key in API_KEYS

async def authenticate_websocket(websocket: WebSocket) -> Optional[Dict]:
    """Authenticate WebSocket connection using JWT token or API key and validate origin."""
    # First, validate the origin
    origin = websocket.headers.get('origin')
    if origin and origin not in CORS_ORIGINS:
        print(f"❌ WebSocket connection rejected from unauthorized origin: {origin}")
        print(f"   Allowed origins: {CORS_ORIGINS}")
        return None
    
    if not REQUIRE_AUTH:
        return {'user_id': str(uuid.uuid4()), 'auth_type': 'none'}
    
    # Try to get token from query parameters or headers
    token = websocket.query_params.get('token')
    api_key = websocket.headers.get(API_KEY_HEADER)
    
    if token:
        # JWT authentication
        payload = verify_jwt_token(token)
        if payload:
            return {'user_id': payload['user_id'], 'auth_type': 'jwt'}
    elif api_key:
        # API key authentication
        if verify_api_key(api_key):
            return {'user_id': f'api_user_{api_key[:8]}', 'auth_type': 'api_key'}
    
    return None

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if PANDAS_AVAILABLE:
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
    if isinstance(obj, (str, int, type(None))):
        return obj
    elif isinstance(obj, bool):
        return bool(obj)  # Ensure it's a Python bool
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
    elif isinstance(obj, np.bool_):
        return bool(obj)  # Convert NumPy boolean to Python boolean
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

app = FastAPI()

# Load CORS origins from environment variable
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:8080,http://127.0.0.1:5173").split(",")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Only allow specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Live Data Pub/Sub System ---
class LiveDataPubSub:
    def __init__(self):
        self.clients = {}  # queue -> filter dict
        self.lock = asyncio.Lock()
        self.client_throttle = {}  # queue -> last_sent_time
        self.client_batch = {}     # queue -> list of batched messages
        self.client_format = {}    # queue -> 'json' or 'msgpack'
        self.client_health = {}    # queue -> last_heartbeat_time
        self.client_connected = {} # queue -> connection status
        self.client_auth = {}      # queue -> auth info
        
        # NEW: Centralized subscription management
        self.token_subscribers = {}  # token -> set of queues
        self.global_subscribed_tokens = set()  # All tokens currently subscribed to Zerodha

    async def subscribe(self, auth_info: Optional[Dict] = None):
        queue = asyncio.Queue()
        # Default filter: subscribe to everything
        filter_ = {
            'tokens': set(),
            'timeframes': set(),
            'all': True,
            'throttle_ms': 0,
            'batch': False,
            'batch_size': 1,
            'format': 'json'  # Default to JSON
        }
        async with self.lock:
            self.clients[queue] = filter_
            self.client_throttle[queue] = 0
            self.client_batch[queue] = []
            self.client_format[queue] = 'json'
            self.client_health[queue] = time.time()
            self.client_connected[queue] = True
            self.client_auth[queue] = auth_info or {'user_id': str(uuid.uuid4()), 'auth_type': 'none'}
        return queue, filter_

    async def unsubscribe(self, queue):
        async with self.lock:
            # Remove from token subscribers
            tokens_to_remove = []
            for token, subscribers in list(self.token_subscribers.items()):
                if queue in subscribers:
                    subscribers.remove(queue)
                    # If no more subscribers for this token, remove it from global subscription
                    if not subscribers:
                        tokens_to_remove.append(token)
            
            # Remove tokens that have no subscribers
            for token in tokens_to_remove:
                await self._unsubscribe_from_zerodha([token])
                del self.token_subscribers[token]
            
            if queue in self.clients:
                del self.clients[queue]
            if queue in self.client_throttle:
                del self.client_throttle[queue]
            if queue in self.client_batch:
                del self.client_batch[queue]
            if queue in self.client_format:
                del self.client_format[queue]
            if queue in self.client_health:
                del self.client_health[queue]
            if queue in self.client_connected:
                del self.client_connected[queue]
            if queue in self.client_auth:
                del self.client_auth[queue]

    async def update_filter(self, queue, **kwargs):
        async with self.lock:
            if queue not in self.clients:
                return
                
            filter_ = self.clients[queue]
            old_tokens = filter_.get('tokens', set()).copy()
            
            for key, value in kwargs.items():
                if key == 'tokens' and isinstance(value, list):
                    filter_['tokens'] = set(value)
                elif key == 'timeframes' and isinstance(value, list):
                    filter_['timeframes'] = set(value)
                elif key in ['all', 'throttle_ms', 'batch', 'batch_size', 'format']:
                    filter_[key] = value
                    if key == 'format':
                        self.client_format[queue] = value
            
            # Update token subscriptions
            new_tokens = filter_.get('tokens', set())
            
            # Remove old token subscriptions
            tokens_to_unsubscribe = old_tokens - new_tokens
            for token in tokens_to_unsubscribe:
                if token in self.token_subscribers:
                    self.token_subscribers[token].discard(queue)
                    # If no more subscribers for this token, remove it from global subscription
                    if not self.token_subscribers[token]:
                        await self._unsubscribe_from_zerodha([token])
                        del self.token_subscribers[token]
            
            # Add new token subscriptions
            tokens_to_subscribe = new_tokens - old_tokens
            for token in tokens_to_subscribe:
                if token not in self.token_subscribers:
                    self.token_subscribers[token] = set()
                self.token_subscribers[token].add(queue)
                # If this is a new token globally, subscribe to it
                if token not in self.global_subscribed_tokens:
                    await self._subscribe_to_zerodha([token])

    async def update_heartbeat(self, queue):
        """Update the last heartbeat time for a client."""
        async with self.lock:
            if queue in self.client_health:
                self.client_health[queue] = time.time()

    async def mark_disconnected(self, queue):
        """Mark a client as disconnected."""
        async with self.lock:
            if queue in self.client_connected:
                self.client_connected[queue] = False

    async def _subscribe_to_zerodha(self, tokens):
        """Subscribe to tokens in Zerodha WebSocket if not already subscribed."""
        new_tokens = [token for token in tokens if token not in self.global_subscribed_tokens]
        if new_tokens:
            try:
                # Convert string tokens to integers for Zerodha
                int_tokens = [int(token) for token in new_tokens]
                ws_client = get_zerodha_ws_client()
                ws_client.subscribe(int_tokens)
                ws_client.set_mode('quote', int_tokens)
                self.global_subscribed_tokens.update(new_tokens)
                print(f"🔗 Subscribed to new Zerodha tokens: {new_tokens}")
            except Exception as e:
                print(f"❌ Error subscribing to Zerodha tokens {new_tokens}: {e}")

    async def _unsubscribe_from_zerodha(self, tokens):
        """Unsubscribe from tokens in Zerodha WebSocket if no clients are subscribed."""
        tokens_to_unsubscribe = [token for token in tokens if token in self.global_subscribed_tokens]
        if tokens_to_unsubscribe:
            try:
                # Convert string tokens to integers for Zerodha
                int_tokens = [int(token) for token in tokens_to_unsubscribe]
                ws_client = get_zerodha_ws_client()
                ws_client.unsubscribe(int_tokens)
                self.global_subscribed_tokens.difference_update(tokens_to_unsubscribe)
                print(f"🔗 Unsubscribed from Zerodha tokens: {tokens_to_unsubscribe}")
            except Exception as e:
                print(f"❌ Error unsubscribing from Zerodha tokens {tokens_to_unsubscribe}: {e}")

    async def get_connection_stats(self):
        """Get connection statistics."""
        async with self.lock:
            total_clients = len(self.clients)
            connected_clients = sum(1 for connected in self.client_connected.values() if connected)
            total_tokens = len(self.global_subscribed_tokens)
            
            return {
                'total_clients': total_clients,
                'connected_clients': connected_clients,
                'total_tokens': total_tokens,
                'subscribed_tokens': list(self.global_subscribed_tokens),
                'token_subscribers': {token: len(subscribers) for token, subscribers in self.token_subscribers.items()}
            }

    async def publish(self, data):
        current_time = time.time() * 1000
        token = data.get('token')
        
        # NEW: Only send to clients subscribed to this specific token
        target_queues = set()
        if token and token in self.token_subscribers:
            target_queues = self.token_subscribers[token].copy()
        elif data.get('type') == 'candle':  # For candle data, check timeframe subscribers
            timeframe = data.get('timeframe')
            if timeframe:
                # Find clients subscribed to this timeframe
                for queue, filter_ in self.clients.items():
                    if (filter_.get('all', True) or 
                        (filter_.get('timeframes') and timeframe in filter_.get('timeframes', set()))):
                        target_queues.add(queue)
        else:
            # For other data types, send to all clients (fallback)
            target_queues = set(self.clients.keys())
        
        async with self.lock:
            for queue in target_queues:
                if queue not in self.clients:
                    continue
                    
                filter_ = self.clients[queue]
                
                # Skip disconnected clients
                if not self.client_connected.get(queue, False):
                    continue
                    
                # Check if client wants this data (additional filtering)
                if not filter_['all']:
                    timeframe = data.get('timeframe')
                    if token and filter_['tokens'] and token not in filter_['tokens']:
                        continue
                    if timeframe and filter_['timeframes'] and timeframe not in filter_['timeframes']:
                        continue

                # Check throttle
                throttle_ms = filter_['throttle_ms']
                if throttle_ms > 0:
                    last_sent = self.client_throttle.get(queue, 0)
                    if current_time - last_sent < throttle_ms:
                        continue

                # Handle batching
                if filter_['batch']:
                    self.client_batch[queue].append(data)
                    if len(self.client_batch[queue]) >= filter_['batch_size']:
                        batch_data = {
                            'type': 'batch',
                            'messages': self.client_batch[queue].copy()
                        }
                        await queue.put(batch_data)
                        self.client_batch[queue].clear()
                        self.client_throttle[queue] = current_time
                else:
                    await queue.put(data)
                    self.client_throttle[queue] = current_time

live_pubsub = LiveDataPubSub()

# --- Alert Management ---
import uuid

class AlertManager:
    def __init__(self):
        # queue -> list of alerts
        self.client_alerts = {}
        self.lock = asyncio.Lock()

    async def register_alert(self, queue, alert_def):
        async with self.lock:
            if queue not in self.client_alerts:
                self.client_alerts[queue] = []
            alert_id = str(uuid.uuid4())
            alert_def = dict(alert_def)
            alert_def['id'] = alert_id
            self.client_alerts[queue].append(alert_def)
            return alert_id

    async def remove_alert(self, queue, alert_id):
        async with self.lock:
            if queue in self.client_alerts:
                self.client_alerts[queue] = [a for a in self.client_alerts[queue] if a['id'] != alert_id]

    async def get_alerts(self, queue):
        async with self.lock:
            return list(self.client_alerts.get(queue, []))

    async def get_all_alerts(self):
        async with self.lock:
            return dict(self.client_alerts)

alert_manager = AlertManager()

# --- Alert Evaluation Logic ---
def evaluate_alert(alert, data):
    # alert: dict, data: tick or candle dict
    alert_type = alert.get('type')
    params = alert.get('params', {})
    if alert_type == 'price_cross':
        price = data.get('last_price') or data.get('close')
        threshold = params.get('threshold')
        direction = params.get('direction', 'above')
        if price is not None and threshold is not None:
            if direction == 'above' and price > threshold:
                return True
            if direction == 'below' and price < threshold:
                return True
    elif alert_type == 'volume_spike':
        volume = data.get('volume') or data.get('volume_traded')
        threshold = params.get('threshold')
        if volume is not None and threshold is not None:
            if volume > threshold:
                return True
    return False

# --- Patch tick/candle hooks to evaluate alerts ---
async def alert_publish_hook(data):
    all_alerts = await alert_manager.get_all_alerts()
    for queue, alerts in all_alerts.items():
        for alert in alerts:
            # Match token/timeframe if specified
            token_match = (not alert.get('token')) or (alert.get('token') == data.get('token'))
            tf_match = (not alert.get('timeframe')) or (alert.get('timeframe') == data.get('timeframe'))
            if token_match and tf_match and evaluate_alert(alert, data):
                alert_event = {
                    'type': 'alert',
                    'alert_id': alert['id'],
                    'alert_type': alert['type'],
                    'token': data.get('token'),
                    'timeframe': data.get('timeframe'),
                    'data': data,
                    'params': alert.get('params', {})
                }
                # Use client's preferred format
                format_type = live_pubsub.client_format.get(queue, 'json')
                if format_type == 'msgpack':
                    await queue.put(msgpack.packb(alert_event, use_bin_type=True))
                else:
                    await queue.put(alert_event)

# Patch hooks

def tick_forward_hook(token, tick):
    data = {
        'type': 'tick',
        'token': int(token),
        'price': float(tick.get('last_price', 0)),
        'timestamp': float(tick.get('timestamp', time.time())),
        'volume_traded': float(tick.get('volume_traded', 0)),
    }
    asyncio.create_task(live_pubsub.publish(data))
    asyncio.create_task(alert_publish_hook(data))

def candle_forward_hook(token, timeframe, candle):
    """Forward completed candles to WebSocket subscribers with proper data transformation."""
    transformed_candle = {
        'type': 'candle',
        'token': str(token),
        'timeframe': timeframe,
        'data': {
            'open': float(candle['open']),
            'high': float(candle['high']),
            'low': float(candle['low']),
            'close': float(candle['close']),
            'volume': float(candle['volume']),
            'start': float(candle['start']),
            'end': float(candle['end'])
        },
        'timestamp': float(time.time())
    }
    print(f"[LIVE] Forwarding candle: {transformed_candle}")
    try:
        loop = asyncio.get_running_loop()
        asyncio.create_task(live_pubsub.publish(transformed_candle))
        if timeframe in ['1m', '5m', '15m', '1h', '1d']:
            asyncio.create_task(enhanced_realtime_analysis_callback(token, timeframe, candle))
    except RuntimeError:
        pass

def legacy_candle_forward_hook(token, tf, candle):
    data = {
        'type': 'candle',
        'token': str(token),
        'timeframe': tf,
        'data': {
            'open': float(candle['open']),
            'high': float(candle['high']),
            'low': float(candle['low']),
            'close': float(candle['close']),
            'volume': float(candle['volume']),
            'start': float(candle['start']),
            'end': float(candle['end'])
        },
        'timestamp': float(time.time())
    }
    try:
        loop = asyncio.get_running_loop()
        asyncio.create_task(live_pubsub.publish(data))
        asyncio.create_task(alert_publish_hook(data))
    except RuntimeError:
        pass

# --- Volume Confirmation Helper ---
def check_volume_confirmation(df, threshold_moderate=1.2, threshold_strong=1.5, window=20):
    """
    Check if the latest candle's volume is above average.
    Returns: 'strong', 'moderate', 'weak', or 'none'
    """
    if len(df) < window + 1 or 'volume' not in df.columns:
        return 'none'
    recent = df.iloc[-(window+1):-1]['volume']
    avg_vol = recent.mean()
    curr_vol = df.iloc[-1]['volume']
    if avg_vol == 0:
        return 'none'
    ratio = curr_vol / avg_vol
    if ratio >= threshold_strong:
        return 'strong'
    elif ratio >= threshold_moderate:
        return 'moderate'
    elif ratio > 1.0:
        return 'weak'
    else:
        return 'none'

# --- Sector/Index Relative Strength Helpers ---
def get_sector_and_index_for_stock(symbol):
    from sector_classifier import sector_classifier
    sector = sector_classifier.get_stock_sector(symbol)
    sector_index = sector_classifier.get_primary_sector_index(sector) if sector else None
    # Assume NIFTY 50 as main index
    main_index = 'NIFTY50'
    return sector, sector_index, main_index

def get_candles_for_symbol(symbol, interval, N=50):
    from agent_capabilities import StockAnalysisOrchestrator
    orchestrator = StockAnalysisOrchestrator()
    df = orchestrator.retrieve_stock_data(symbol=symbol, interval=interval, period=N)
    if df is not None and not df.empty:
        return df.tail(N)
    return None

def calculate_relative_strength(stock_df, ref_df, window=20):
    if stock_df is None or ref_df is None or len(stock_df) < window or len(ref_df) < window:
        return None
    stock_returns = stock_df['close'].pct_change().dropna().tail(window)
    ref_returns = ref_df['close'].pct_change().dropna().tail(window)
    # Align by date if possible
    if hasattr(stock_returns, 'index') and hasattr(ref_returns, 'index'):
        aligned = stock_returns.align(ref_returns, join='inner')
        stock_returns, ref_returns = aligned
    rs = (1 + stock_returns).prod() / (1 + ref_returns).prod() if (1 + ref_returns).prod() != 0 else None
    return rs

def get_trend_from_indicators(indicators):
    return indicators.get('trend_data', {}).get('direction', 'unknown')

# --- Dynamic Risk Management Helper ---
def calculate_dynamic_stop_and_position(df, entry_price, risk_pct=0.01, capital=100000):
    """
    Calculate ATR-based stop-loss and position size.
    Args:
        df: DataFrame with OHLCV data
        entry_price: float, entry price for the trade
        risk_pct: float, risk per trade as a fraction of capital
        capital: float, total capital
    Returns:
        dict with 'atr_stop', 'atr_value', 'percent_stop', 'position_size_atr', 'position_size_percent'
    """
    if len(df) < 20 or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
        return None
    # ATR calculation (14-period)
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    # ATR-based stop (2x ATR below entry for long, above for short)
    atr_stop = entry_price - 2 * atr
    percent_stop = entry_price * (1 - 2 * risk_pct)
    # Position sizing (risk per trade / stop distance)
    risk_amount = capital * risk_pct
    position_size_atr = risk_amount / (entry_price - atr_stop) if (entry_price - atr_stop) > 0 else 0
    position_size_percent = risk_amount / (entry_price - percent_stop) if (entry_price - percent_stop) > 0 else 0
    return {
        'atr_stop': atr_stop,
        'atr_value': atr,
        'percent_stop': percent_stop,
        'position_size_atr': position_size_atr,
        'position_size_percent': position_size_percent
    }

# --- Real-Time Analysis Storage and Management ---
class RealTimeAnalysisManager:
    """Manages real-time analysis results, storage, and user notifications."""
    
    def __init__(self):
        self.analysis_results = {}  # token:tf -> latest analysis
        self.user_subscriptions = {}  # user_id -> list of (token, tf) tuples
        self.analysis_queue = asyncio.Queue()
        self.notification_queue = asyncio.Queue()
        self.chart_cache = {}  # token:tf -> chart paths
        self.last_analysis_time = {}  # token:tf -> timestamp
        
    async def store_analysis_result(self, token: str, tf: str, analysis_data: dict):
        """Store analysis result with timestamp."""
        key = f"{token}:{tf}"
        self.analysis_results[key] = {
            **analysis_data,
            'timestamp': time.time(),
            'analysis_id': f"{key}_{int(time.time())}"
        }
        self.last_analysis_time[key] = time.time()
        
        # Trigger notifications for subscribed users
        await self.notify_subscribers(token, tf, analysis_data)
        
    async def get_latest_analysis(self, token: str, tf: str) -> dict:
        """Get the latest analysis result for a token/timeframe."""
        key = f"{token}:{tf}"
        return self.analysis_results.get(key, {})
        
    async def subscribe_user(self, user_id: str, token: str, tf: str):
        """Subscribe a user to real-time analysis updates."""
        if user_id not in self.user_subscriptions:
            self.user_subscriptions[user_id] = []
        self.user_subscriptions[user_id].append((token, tf))
        
    async def unsubscribe_user(self, user_id: str, token: str = None, tf: str = None):
        """Unsubscribe a user from updates."""
        if user_id in self.user_subscriptions:
            if token is None and tf is None:
                del self.user_subscriptions[user_id]
            else:
                self.user_subscriptions[user_id] = [
                    (t, timeframe) for t, timeframe in self.user_subscriptions[user_id]
                    if not (t == token and timeframe == tf)
                ]
                
    async def notify_subscribers(self, token: str, tf: str, analysis_data: dict):
        """Notify all subscribed users of new analysis."""
        notification = {
            'type': 'analysis_update',
            'token': token,
            'timeframe': tf,
            'analysis': analysis_data,
            'timestamp': time.time()
        }
        
        # Find users subscribed to this token/timeframe
        for user_id, subscriptions in self.user_subscriptions.items():
            for sub_token, sub_tf in subscriptions:
                if sub_token == token and sub_tf == tf:
                    await self.notification_queue.put({
                        'user_id': user_id,
                        'notification': notification
                    })
                    break

# Global real-time analysis manager
realtime_analysis_manager = RealTimeAnalysisManager()

# --- Enhanced Real-Time Analysis Callback ---
async def enhanced_realtime_analysis_callback(token, tf, candle):
    """
    Enhanced real-time analysis callback with full AI analysis, chart generation, 
    result storage, and user notifications.
    """
    import pandas as pd
    from agent_capabilities import StockAnalysisOrchestrator
    from patterns.recognition import PatternRecognition
    import logging
    import asyncio
    from datetime import datetime
    
    logger = logging.getLogger("enhanced_realtime_analysis")
    
    try:
        # Check if we should run analysis (rate limiting)
        key = f"{token}:{tf}"
        last_analysis = realtime_analysis_manager.last_analysis_time.get(key, 0)
        current_time = time.time()
        
        # Rate limiting: minimum 30 seconds between analyses for same token:tf
        if current_time - last_analysis < 30:
            logger.debug(f"Rate limited: Skipping analysis for {key}")
            return
            
        logger.info(f"Starting enhanced real-time analysis for {key}")
        
        # Fetch last N candles for this token/timeframe
        N = 50
        M = 500  # For backtesting window
        candles = candle_aggregator.candles[token][tf]
        sorted_buckets = sorted(candles.keys())
        recent_candles = [candles[b] for b in sorted_buckets[-N:]]
        
        if len(recent_candles) < 20:  # Minimum required candles
            logger.warning(f"Insufficient candles for {key}: {len(recent_candles)}")
            return
            
        df = pd.DataFrame(recent_candles)
        df['datetime'] = pd.to_datetime(df['start'], unit='s')
        df = df.set_index('datetime')
        
        # Get stock symbol from token
        orchestrator = StockAnalysisOrchestrator()
        symbol = orchestrator.data_client.get_symbol_from_token(token)
        if not symbol:
            logger.warning(f"Could not find symbol for token {token}")
            return
            
        # 1. Calculate technical indicators
        indicators = orchestrator.calculate_indicators(df, symbol)
        candlestick_patterns = PatternRecognition.detect_candlestick_patterns(df)
        
        # 2. Volume confirmation
        volume_confirmation = check_volume_confirmation(df)
        
        # 3. Multi-timeframe confirmation logic
        confirmations = {}
        for lower_tf in ['1h', '4h']:
            try:
                lower_candles = candle_aggregator.candles[token][lower_tf]
                if lower_candles:
                    lower_sorted = sorted(lower_candles.keys())
                    lower_recent = [lower_candles[b] for b in lower_sorted[-N:]]
                    lower_df = pd.DataFrame(lower_recent)
                    lower_df['datetime'] = pd.to_datetime(lower_df['start'], unit='s')
                    lower_df = df.set_index('datetime')
                    lower_indicators = orchestrator.calculate_indicators(lower_df)
                    lower_patterns = PatternRecognition.detect_candlestick_patterns(lower_df)
                    confirmations[lower_tf] = {
                        'indicators': lower_indicators,
                        'candlestick_patterns': lower_patterns[-3:],
                        'volume_confirmation': check_volume_confirmation(lower_df)
                    }
                else:
                    confirmations[lower_tf] = None
            except Exception as e:
                confirmations[lower_tf] = None
                logger.warning(f"[RealTime] Could not analyze {lower_tf} for token {token}: {e}")
        
        # 4. Generate real-time charts
        chart_paths = {}
        try:
            output_dir = f"./output/realtime/{symbol}_{tf}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Create essential charts for real-time analysis
            chart_paths = orchestrator.create_visualizations(df, indicators, symbol, output_dir)
            
            # Cache chart paths
            realtime_analysis_manager.chart_cache[key] = chart_paths
            
        except Exception as e:
            logger.error(f"Error generating charts for {key}: {e}")
        
        # 5. Full AI Analysis (LLM Call)
        ai_analysis = None
        try:
            # Get sector context
            sector = orchestrator.sector_benchmarking_provider.sector_classifier.get_stock_sector(symbol)
            sector_context = None
            if sector:
                try:
                    sector_benchmarking = orchestrator.sector_benchmarking_provider.get_comprehensive_benchmarking(symbol, df)
                    sector_rotation = orchestrator.sector_benchmarking_provider.analyze_sector_rotation("3M")
                    sector_correlation = orchestrator.sector_benchmarking_provider.generate_sector_correlation_matrix("6M")
                    sector_context = orchestrator._build_enhanced_sector_context(
                        sector, sector_benchmarking, sector_rotation, sector_correlation
                    )
                except Exception as e:
                    logger.warning(f"Could not get sector context for {symbol}: {e}")
            
            # Call AI analysis
            ai_analysis, ind_summary_md, chart_insights_md = await orchestrator.analyze_with_ai(
                symbol, indicators, chart_paths, 30, tf, "", sector_context
            )
            
        except Exception as e:
            logger.error(f"Error in AI analysis for {key}: {e}")
            ai_analysis = {
                'trend': 'unknown',
                'confidence_pct': 0,
                'error': str(e)
            }
        
        # 6. Additional analysis components
        # Confirmation check for daily signals
        confirmation_status = 'not_checked'
        if tf == '1d' and confirmations['1h'] and confirmations['4h']:
            daily_trend = indicators.get('trend_data', {}).get('direction')
            h1_trend = confirmations['1h']['indicators'].get('trend_data', {}).get('direction')
            h4_trend = confirmations['4h']['indicators'].get('trend_data', {}).get('direction')
            if daily_trend and h1_trend and h4_trend:
                if daily_trend == h1_trend == h4_trend:
                    confirmation_status = f'confirmed_{daily_trend}'
                else:
                    confirmation_status = 'not_confirmed'
        
        # Sector/Index Relative Strength
        sector, sector_index, main_index = get_sector_and_index_for_stock(token)
        sector_df = get_candles_for_symbol(sector_index, tf, N) if sector_index else None
        index_df = get_candles_for_symbol(main_index, tf, N) if main_index else None
        rs_sector = calculate_relative_strength(df, sector_df)
        rs_index = calculate_relative_strength(df, index_df)
        sector_trend = get_trend_from_indicators(orchestrator.calculate_indicators(sector_df)) if sector_df is not None else 'unknown'
        index_trend = get_trend_from_indicators(orchestrator.calculate_indicators(index_df)) if index_df is not None else 'unknown'
        rs_confirmation = rs_sector is not None and rs_sector > 1 and sector_trend == get_trend_from_indicators(indicators)
        
        # Dynamic risk management
        entry_price = candle['close'] if 'close' in candle else None
        dynamic_risk = calculate_dynamic_stop_and_position(df, entry_price) if entry_price else None
        
        # Pattern backtesting stats
        if len(sorted_buckets) >= M:
            backtest_df = pd.DataFrame([candles[b] for b in sorted_buckets[-M:]])
            backtest_df['datetime'] = pd.to_datetime(backtest_df['start'], unit='s')
            backtest_df = df.set_index('datetime')
            pattern_stats = PatternRecognition.backtest_pattern(
                backtest_df,
                PatternRecognition.detect_double_top,
                window=100,
                hold_period=10
            )
        else:
            pattern_stats = None
            
        pattern_confidence = (
            pattern_stats is not None and pattern_stats['win_rate'] > 0.5 and pattern_stats['n_trades'] > 10
        )
        
        # Pattern failure/invalidation triggers
        invalidation_trigger = None
        double_tops = PatternRecognition.detect_double_top(df['close'])
        if double_tops:
            last_top = double_tops[-1]
            if isinstance(last_top, (tuple, list)) and len(last_top) > 1:
                pattern_high = df['close'].iloc[last_top[0]:last_top[1]+1].max()
                invalidation_trigger = f"Close above {pattern_high:.2f} invalidates double top short setup."
        
        # Time-based stop
        time_stop = None
        time_stop_period = 10
        if entry_price:
            time_stop = f"Exit or re-evaluate if target not reached within {time_stop_period} {tf} candles."
        
        # Signal confluence logic
        confluence_signals = {
            'trend': indicators.get('trend_data', {}).get('direction'),
            'volume': volume_confirmation,
            'multi_tf': confirmation_status,
            'rs': 'confirmed' if rs_confirmation else 'not_confirmed',
            'dynamic_risk': 'ok' if dynamic_risk else 'none',
            'pattern_stats': 'strong' if pattern_confidence else 'weak',
            'invalidation': 'set' if invalidation_trigger else 'none',
            'time_stop': 'set' if time_stop else 'none'
        }
        
        confirming = 0
        total = len(confluence_signals)
        rationale = {}
        for k, v in confluence_signals.items():
            if v in ['bullish', 'confirmed_bullish', 'strong', 'ok', 'confirmed', 'moderate', 'set']:
                confirming += 1
                rationale[k] = f"confirming: {v}"
            else:
                rationale[k] = f"not confirming: {v}"
                
        confluence_score = confirming / total if total > 0 else 0
        high_confidence = confluence_score >= 0.7
        
        # 7. Compile comprehensive analysis result
        analysis_result = {
            'symbol': symbol,
            'token': token,
            'timeframe': tf,
            'last_candle': candle,
            'indicators': indicators,
            'candlestick_patterns': candlestick_patterns[-3:],
            'volume_confirmation': volume_confirmation,
            'multi_timeframe_confirmations': confirmations,
            'confirmation_status': confirmation_status,
            'sector_analysis': {
                'sector': sector,
                'sector_index': sector_index,
                'main_index': main_index,
                'rs_sector': rs_sector,
                'rs_index': rs_index,
                'sector_trend': sector_trend,
                'index_trend': index_trend,
                'rs_confirmation': rs_confirmation
            },
            'risk_management': {
                'dynamic_risk': dynamic_risk,
                'pattern_stats': pattern_stats,
                'pattern_confidence': pattern_confidence,
                'invalidation_trigger': invalidation_trigger,
                'time_stop': time_stop
            },
            'signal_confluence': {
                'signals': confluence_signals,
                'score': confluence_score,
                'rationale': rationale,
                'high_confidence': high_confidence
            },
            'ai_analysis': ai_analysis,
            'chart_paths': chart_paths,
            'analysis_metadata': {
                'candles_analyzed': len(recent_candles),
                'analysis_timestamp': datetime.now().isoformat(),
                'data_freshness': 'real_time'
            }
        }
        
        # 8. Store analysis result
        await realtime_analysis_manager.store_analysis_result(token, tf, analysis_result)
        
        # 9. Log results
        logger.info(f"[Enhanced RealTime] Completed analysis for {key}")
        logger.info(f"[Enhanced RealTime] AI Trend: {ai_analysis.get('trend', 'unknown')}, Confidence: {ai_analysis.get('confidence_pct', 0)}%")
        logger.info(f"[Enhanced RealTime] Confluence Score: {confluence_score:.2f}, High Confidence: {high_confidence}")
        
        if high_confidence:
            logger.info(f"[Enhanced RealTime] HIGH-CONFIDENCE SIGNAL: {key} | AI: {ai_analysis.get('trend', 'unknown')} | Confluence: {confluence_score:.2f}")
            
    except Exception as e:
        logger.error(f"Error in enhanced realtime analysis for {token}:{tf}: {e}")
        import traceback
        traceback.print_exc()

# Replace the old callback with the enhanced version
def realtime_analysis_callback(token, tf, candle):
    """Wrapper to run enhanced analysis asynchronously."""
    asyncio.create_task(enhanced_realtime_analysis_callback(token, tf, candle))

# Register hooks
candle_aggregator.register_callback(candle_forward_hook)

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    global MAIN_EVENT_LOOP
    MAIN_EVENT_LOOP = asyncio.get_running_loop()
    
    # Ensure publish callback and main event loop are set before starting WebSocket client
    print("Setting up publish callback and main event loop...")
    from zerodha_ws_client import set_publish_callback, set_main_event_loop
    set_publish_callback(live_pubsub.publish)
    set_main_event_loop(MAIN_EVENT_LOOP)
    
    # Start Zerodha WebSocket client in a background thread
    print("Starting Zerodha WebSocket client...")
    # Get fresh instance to ensure latest credentials
    from zerodha_ws_client import get_zerodha_ws_client
    ws_client = get_zerodha_ws_client()
    ws_client.connect()
    
    # Wait a moment for the connection to establish
    await asyncio.sleep(3)
    
    # Subscribe to default tokens for basic coverage
    default_tokens = [738561, 11536, 1330, 1594, 4963]  # Updated token, TCS, HDFC, INFY, ICICIBANK
    try:
        print(f"Attempting to subscribe to default tokens: {default_tokens}")
        ws_client.subscribe(default_tokens)
        
        # Set mode to 'quote' for OHLCV data (44 bytes per packet)
        # This provides open, high, low, close, volume data needed for charts
        ws_client.set_mode('quote', default_tokens)
        
        print(f"Successfully subscribed to default Zerodha tokens: {default_tokens}")
        print("WebSocket mode set to 'quote' for OHLCV data")
    except Exception as e:
        print(f"Error subscribing to default tokens: {e}")
        
        # Check if Zerodha credentials are available
        api_key = os.getenv("ZERODHA_API_KEY")
        access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
        
        if not api_key or not access_token:
            print("⚠️  This is expected if Zerodha credentials are not configured")
            print("To enable live data, please set ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN environment variables")
            print("Make sure your .env file is in the backend directory and contains valid credentials")
        else:
            print("⚠️  Zerodha credentials are configured but subscription failed")
            print("This might be due to network issues or invalid tokens")
    
    # No need to start a background task for pubsub; handled by hooks

# --- WebSocket Endpoint for Frontend ---
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    # Authenticate the connection
    auth_info = await authenticate_websocket(websocket)
    if not auth_info:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required")
        return
    
    await websocket.accept()
    queue, filter_ = await live_pubsub.subscribe(auth_info)
    
    # Start heartbeat task
    heartbeat_task = asyncio.create_task(heartbeat_loop(websocket, queue))
    
    try:
        while True:
            done, pending = await asyncio.wait(
                [asyncio.create_task(websocket.receive_text()), asyncio.create_task(queue.get())],
                return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                result = task.result()
                if isinstance(result, str):
                    try:
                        msg = json.loads(result)
                        if msg.get('action') == 'subscribe':
                            tokens = msg.get('tokens', [])
                            timeframes = msg.get('timeframes', [])
                            throttle_ms = msg.get('throttle_ms', 0)
                            batch = msg.get('batch', False)
                            batch_size = msg.get('batch_size', 1)
                            format_type = msg.get('format', 'json')
                            
                            # Token subscription is now handled centrally by live_pubsub
                            # No need to manage individual subscriptions here
                            
                            await live_pubsub.update_filter(
                                queue, 
                                tokens=tokens, 
                                timeframes=timeframes, 
                                throttle_ms=throttle_ms,
                                batch=batch,
                                batch_size=batch_size,
                                format=format_type
                            )
                            
                            response = {
                                'type': 'subscribed',
                                'tokens': tokens,
                                'timeframes': timeframes,
                                'throttle_ms': throttle_ms,
                                'batch': batch,
                                'batch_size': batch_size,
                                'format': format_type
                            }
                            
                            # Send response in client's preferred format
                            if format_type == 'msgpack':
                                await websocket.send_bytes(msgpack.packb(response, use_bin_type=True))
                            else:
                                await websocket.send_json(response)
                                
                        elif msg.get('action') == 'unsubscribe':
                            tokens = msg.get('tokens', [])
                            timeframes = msg.get('timeframes', [])
                            
                            # Token unsubscription is now handled centrally by live_pubsub
                            # Just update the filter to remove tokens/timeframes
                            await live_pubsub.update_filter(
                                queue,
                                tokens=[],  # Clear all tokens
                                timeframes=[]  # Clear all timeframes
                            )
                            
                            response = {
                                'type': 'unsubscribed',
                                'tokens': tokens,
                                'timeframes': timeframes
                            }
                            
                            format_type = live_pubsub.client_format.get(queue, 'json')
                            if format_type == 'msgpack':
                                await websocket.send_bytes(msgpack.packb(response, use_bin_type=True))
                            else:
                                await websocket.send_json(response)
                                
                        elif msg.get('action') == 'set_all':
                            all_flag = msg.get('all', True)
                            await live_pubsub.update_filter(queue, all=all_flag)
                            response = {'type': 'set_all', 'all': all_flag}
                            
                            format_type = live_pubsub.client_format.get(queue, 'json')
                            if format_type == 'msgpack':
                                await websocket.send_bytes(msgpack.packb(response, use_bin_type=True))
                            else:
                                await websocket.send_json(response)
                                
                        elif msg.get('action') == 'ping':
                            # Handle heartbeat ping
                            await live_pubsub.update_heartbeat(queue)
                            response = {'type': 'pong', 'timestamp': time.time()}
                            
                            format_type = live_pubsub.client_format.get(queue, 'json')
                            if format_type == 'msgpack':
                                await websocket.send_bytes(msgpack.packb(response, use_bin_type=True))
                            else:
                                await websocket.send_json(response)
                                
                        elif msg.get('action') == 'history':
                            # Handle historical replay request
                            token = msg.get('token')
                            timeframe = msg.get('timeframe')
                            count = msg.get('count', 50)
                            # Defensive: ensure count is reasonable
                            try:
                                count = int(count)
                                if count < 1 or count > 500:
                                    count = 50
                            except Exception:
                                count = 50
                            candles = []
                            if token and timeframe:
                                try:
                                    # Access in-memory candles
                                    candles_dict = candle_aggregator.candles.get(token, {}).get(timeframe, {})
                                    sorted_buckets = sorted(candles_dict.keys())
                                    candles = [candles_dict[b] for b in sorted_buckets[-count:]]
                                except Exception as e:
                                    candles = []
                            response = {
                                'type': 'history',
                                'token': token,
                                'timeframe': timeframe,
                                'count': len(candles),
                                'candles': candles
                            }
                            format_type = live_pubsub.client_format.get(queue, 'json')
                            if format_type == 'msgpack':
                                await websocket.send_bytes(msgpack.packb(response, use_bin_type=True))
                            else:
                                await websocket.send_json(response)
                                
                        elif msg.get('action') == 'register_alert':
                            # Register a new alert for this client
                            alert_def = msg.get('alert')
                            if not alert_def:
                                response = {'type': 'error', 'message': 'Missing alert definition'}
                            else:
                                alert_id = await alert_manager.register_alert(queue, alert_def)
                                response = {'type': 'alert_registered', 'alert_id': alert_id}
                            format_type = live_pubsub.client_format.get(queue, 'json')
                            if format_type == 'msgpack':
                                await websocket.send_bytes(msgpack.packb(response, use_bin_type=True))
                            else:
                                await websocket.send_json(response)
                        elif msg.get('action') == 'remove_alert':
                            alert_id = msg.get('alert_id')
                            if not alert_id:
                                response = {'type': 'error', 'message': 'Missing alert_id'}
                            else:
                                await alert_manager.remove_alert(queue, alert_id)
                                response = {'type': 'alert_removed', 'alert_id': alert_id}
                            format_type = live_pubsub.client_format.get(queue, 'json')
                            if format_type == 'msgpack':
                                await websocket.send_bytes(msgpack.packb(response, use_bin_type=True))
                            else:
                                await websocket.send_json(response)
                                
                    except json.JSONDecodeError:
                        await websocket.send_json({'error': 'Invalid JSON'})
                else:
                    # Data from queue - send to client
                    format_type = live_pubsub.client_format.get(queue, 'json')
                    if format_type == 'msgpack':
                        await websocket.send_bytes(msgpack.packb(result, use_bin_type=True))
                    else:
                        await websocket.send_json(result)
                        
            for task in pending:
                task.cancel()
    except WebSocketDisconnect:
        await live_pubsub.mark_disconnected(queue)
        await live_pubsub.unsubscribe(queue)
    except Exception as e:
        import traceback
        print(f"WebSocket error: {e}")
        traceback.print_exc()
        await live_pubsub.mark_disconnected(queue)
        await live_pubsub.unsubscribe(queue)
    finally:
        heartbeat_task.cancel()

async def heartbeat_loop(websocket: WebSocket, queue):
    """Send periodic heartbeat messages to keep connection alive."""
    try:
        while True:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            try:
                # Send heartbeat
                heartbeat_msg = {
                    'type': 'heartbeat',
                    'timestamp': time.time()
                }
                format_type = live_pubsub.client_format.get(queue, 'json')
                if format_type == 'msgpack':
                    await websocket.send_bytes(msgpack.packb(heartbeat_msg, use_bin_type=True))
                else:
                    await websocket.send_json(heartbeat_msg)
            except Exception as e:
                print(f"Heartbeat error: {e}")
                break
    except asyncio.CancelledError:
        pass

@app.get("/ws/health")
async def websocket_health():
    """Get WebSocket connection health statistics."""
    stats = await live_pubsub.get_connection_stats()
    return {
        'status': 'healthy',
        'websocket_stats': stats,
        'timestamp': time.time()
    }

@app.get("/ws/test")
async def test_websocket():
    """Test endpoint to check WebSocket client status."""
    return {
        'zerodha_ws_running': get_zerodha_ws_client().running,
        'subscribed_tokens': list(get_zerodha_ws_client().subscribed_tokens),
        'api_key_configured': bool(get_zerodha_ws_client().api_key and get_zerodha_ws_client().api_key != 'your_api_key'),
        'access_token_configured': bool(get_zerodha_ws_client().access_token and get_zerodha_ws_client().access_token != 'your_access_token'),
        'candle_aggregator_timeframes': list(candle_aggregator.timeframes),
        'registered_callbacks': len(candle_aggregator.callbacks),
        'timestamp': time.time()
    }

@app.get("/ws/connections")
async def get_connections():
    """Get detailed connection information."""
    async with live_pubsub.lock:
        connections = []
        current_time = time.time()
        for queue, filter_ in live_pubsub.clients.items():
            last_heartbeat = live_pubsub.client_health.get(queue, 0)
            connected = live_pubsub.client_connected.get(queue, False)
            connections.append({
                'connected': connected,
                'last_heartbeat': last_heartbeat,
                'heartbeat_age': current_time - last_heartbeat,
                'filter': {
                    'tokens': list(filter_.get('tokens', [])),
                    'timeframes': list(filter_.get('timeframes', [])),
                    'all': filter_.get('all', True),
                    'throttle_ms': filter_.get('throttle_ms', 0),
                    'batch': filter_.get('batch', False),
                    'batch_size': filter_.get('batch_size', 1),
                    'format': filter_.get('format', 'json')
                }
            })
        return {
            'connections': connections,
            'total': len(connections),
            'timestamp': current_time
        }

class AnalysisRequest(BaseModel):
    stock: str = Field(..., description="Stock symbol to analyze")
    exchange: str = Field(default="NSE", description="Stock exchange")
    period: int = Field(default=365, description="Analysis period in days")
    interval: str = Field(default="day", description="Data interval")
    output: Optional[str] = Field(default=None, description="Output directory")
    sector: Optional[str] = Field(default=None, description="Optional sector override")

class SectorAnalysisRequest(BaseModel):
    sector: str = Field(..., description="Sector to analyze")
    period: int = Field(default=365, description="Analysis period in days")

class SectorComparisonRequest(BaseModel):
    sectors: List[str] = Field(..., description="List of sectors to compare")
    period: int = Field(default=365, description="Analysis period in days")

class TriggerAnalysisRequest(BaseModel):
    stock: str
    exchange: str = "NSE"
    period: int = 365
    interval: str = "day"
    sector: str = None
    output: str = None  # Optional output directory

class RealTimeSubscriptionRequest(BaseModel):
    user_id: str
    token: str
    timeframe: str

class RealTimeAnalysisRequest(BaseModel):
    token: str
    timeframe: str

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

        # Analyze stock with sector awareness
        results, success_message, error_message = await orchestrator.analyze_stock(
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

        # Store analysis in Supabase
        # Generate a proper UUID for anonymous/system users
        user_id = getattr(request, 'user_id', None)
        if not user_id:
            # Generate a UUID for anonymous/system users
            user_id = str(uuid.uuid4())
        store_analysis_in_supabase(
            serialized_results, 
            user_id, 
            request.stock,
            exchange=request.exchange,
            period=request.period,
            interval=request.interval
        )

        # Clean, efficient response
        response = {
            "success": True,
            "stock_symbol": request.stock,
            "exchange": request.exchange,
            "analysis_period": f"{request.period} days",
            "interval": request.interval,
            "timestamp": pd.Timestamp.now().isoformat(),
            "message": success_message,
            "results": serialized_results
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

@app.post("/sector/benchmark")
async def sector_benchmark(request: AnalysisRequest):
    """Get sector benchmarking for a specific stock."""
    try:
        # Get stock data
        orchestrator = StockAnalysisOrchestrator()
        if not orchestrator.authenticate():
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        data = orchestrator.retrieve_stock_data(
            symbol=request.stock,
            exchange=request.exchange,
            period=request.period,
            interval=request.interval
        )
        
        if data is None:
            raise HTTPException(status_code=404, detail="Stock data not found")
        
        # Get comprehensive sector benchmarking
        benchmarking = sector_benchmarking_provider.get_comprehensive_benchmarking(request.stock, data)
        
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
        
        sector_data = orchestrator.retrieve_stock_data(
            symbol=sector_index,
            exchange="NSE",
            period=period,
            interval="day"
        )
        
        if sector_data is None:
            raise HTTPException(status_code=404, detail="Sector data not found")
        
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
                    
                    sector_data = orchestrator.retrieve_stock_data(
                        symbol=sector_index,
                        exchange="NSE",
                        period=request.period,
                        interval="day"
                    )
                    
                    if sector_data is not None:
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
                "stock_symbol": symbol,
                "exchange": exchange,
                "quote": quote,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Stock quote not found")
            
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "stock_symbol": symbol,
            "exchange": exchange,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.post("/auth/token")
async def create_token(user_id: str):
    """Create a JWT token for WebSocket authentication."""
    if REQUIRE_AUTH:
        token = create_jwt_token(user_id)
        return {"token": token, "user_id": user_id}
    else:
        return {"message": "Authentication disabled"}

@app.get("/auth/verify")
async def verify_token(token: str):
    """Verify a JWT token."""
    payload = verify_jwt_token(token)
    if payload:
        return {"valid": True, "user_id": payload['user_id']}
    else:
        return {"valid": False}

@app.post("/trigger-analysis")
async def trigger_analysis(request: TriggerAnalysisRequest):
    output_dir = request.output or f"./output/{request.stock}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        orchestrator = StockAnalysisOrchestrator()
        # Authenticate
        auth_success = orchestrator.authenticate()
        if not auth_success:
            raise HTTPException(status_code=401, detail="Failed to authenticate with Zerodha API")

        # Analyze stock with sector awareness, using real-time data if available
        results, success_message, error_message = await orchestrator.analyze_stock(
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

        response = {
            "success": True,
            "stock_symbol": request.stock,
            "exchange": request.exchange,
            "analysis_period": f"{request.period} days",
            "interval": request.interval,
            "timestamp": pd.Timestamp.now().isoformat(),
            "message": success_message,
            "results": serialized_results
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

@app.get("/realtime/analysis/{token}/{timeframe}")
async def get_realtime_analysis(token: str, timeframe: str):
    """Get the latest real-time analysis result for a token/timeframe."""
    try:
        analysis = await realtime_analysis_manager.get_latest_analysis(token, timeframe)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="No analysis found for this token/timeframe")
        
        return {
            "success": True,
            "token": token,
            "timeframe": timeframe,
            "analysis": make_json_serializable(analysis),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "token": token,
            "timeframe": timeframe,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.post("/realtime/subscribe")
async def subscribe_to_realtime_analysis(request: RealTimeSubscriptionRequest):
    """Subscribe a user to real-time analysis updates for a specific token/timeframe."""
    try:
        await realtime_analysis_manager.subscribe_user(
            request.user_id, 
            request.token, 
            request.timeframe
        )
        
        return {
            "success": True,
            "message": f"Subscribed to real-time analysis for token {request.token}, timeframe {request.timeframe}",
            "user_id": request.user_id,
            "token": request.token,
            "timeframe": request.timeframe,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "user_id": request.user_id,
            "token": request.token,
            "timeframe": request.timeframe,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.post("/realtime/unsubscribe")
async def unsubscribe_from_realtime_analysis(request: RealTimeSubscriptionRequest):
    """Unsubscribe a user from real-time analysis updates for a specific token/timeframe."""
    try:
        await realtime_analysis_manager.unsubscribe_user(
            request.user_id, 
            request.token, 
            request.timeframe
        )
        
        return {
            "success": True,
            "message": f"Unsubscribed from real-time analysis for token {request.token}, timeframe {request.timeframe}",
            "user_id": request.user_id,
            "token": request.token,
            "timeframe": request.timeframe,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "user_id": request.user_id,
            "token": request.token,
            "timeframe": request.timeframe,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/realtime/subscriptions/{user_id}")
async def get_user_subscriptions(user_id: str):
    """Get all real-time analysis subscriptions for a user."""
    try:
        subscriptions = realtime_analysis_manager.user_subscriptions.get(user_id, [])
        
        return {
            "success": True,
            "user_id": user_id,
            "subscriptions": [
                {"token": token, "timeframe": tf} 
                for token, tf in subscriptions
            ],
            "count": len(subscriptions),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "user_id": user_id,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/realtime/analysis-history/{token}/{timeframe}")
async def get_analysis_history(token: str, timeframe: str, limit: int = 10):
    """Get analysis history for a token/timeframe (limited to recent analyses)."""
    try:
        # For now, we only store the latest analysis
        # In a production system, you'd want to store historical analyses in a database
        analysis = await realtime_analysis_manager.get_latest_analysis(token, timeframe)
        
        history = [analysis] if analysis else []
        
        return {
            "success": True,
            "token": token,
            "timeframe": timeframe,
            "history": make_json_serializable(history),
            "count": len(history),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "token": token,
            "timeframe": timeframe,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/realtime/status")
async def get_realtime_status():
    """Get the status of the real-time analysis system."""
    try:
        # Get some statistics
        total_analyses = len(realtime_analysis_manager.analysis_results)
        total_subscriptions = sum(len(subs) for subs in realtime_analysis_manager.user_subscriptions.values())
        total_users = len(realtime_analysis_manager.user_subscriptions)
        
        # Get recent analysis keys
        recent_analyses = list(realtime_analysis_manager.analysis_results.keys())[-10:]
        
        return {
            "success": True,
            "status": "active",
            "statistics": {
                "total_analyses": total_analyses,
                "total_subscriptions": total_subscriptions,
                "total_users": total_users,
                "chart_cache_size": len(realtime_analysis_manager.chart_cache)
            },
            "recent_analyses": recent_analyses,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.websocket("/ws/realtime-analysis")
async def ws_realtime_analysis(websocket: WebSocket):
    """WebSocket endpoint for real-time analysis notifications."""
    # Authenticate the connection
    auth_info = await authenticate_websocket(websocket)
    if not auth_info:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required")
        return
    
    await websocket.accept()
    user_id = auth_info.get('user_id', 'anonymous')
    
    # Create a queue for this user's notifications
    user_queue = asyncio.Queue()
    
    # Start notification monitoring task
    notification_task = asyncio.create_task(monitor_user_notifications(user_id, user_queue))
    
    try:
        while True:
            done, pending = await asyncio.wait(
                [asyncio.create_task(websocket.receive_text()), asyncio.create_task(user_queue.get())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                result = task.result()
                if isinstance(result, str):
                    try:
                        msg = json.loads(result)
                        if msg.get('action') == 'subscribe':
                            token = msg.get('token')
                            timeframe = msg.get('timeframe')
                            
                            if token and timeframe:
                                await realtime_analysis_manager.subscribe_user(user_id, token, timeframe)
                                response = {
                                    'type': 'subscribed',
                                    'token': token,
                                    'timeframe': timeframe,
                                    'message': f'Subscribed to real-time analysis for {token}:{timeframe}'
                                }
                                await websocket.send_json(response)
                            else:
                                await websocket.send_json({
                                    'type': 'error',
                                    'message': 'Missing token or timeframe'
                                })
                                
                        elif msg.get('action') == 'unsubscribe':
                            token = msg.get('token')
                            timeframe = msg.get('timeframe')
                            
                            if token and timeframe:
                                await realtime_analysis_manager.unsubscribe_user(user_id, token, timeframe)
                                response = {
                                    'type': 'unsubscribed',
                                    'token': token,
                                    'timeframe': timeframe,
                                    'message': f'Unsubscribed from real-time analysis for {token}:{timeframe}'
                                }
                                await websocket.send_json(response)
                            else:
                                await websocket.send_json({
                                    'type': 'error',
                                    'message': 'Missing token or timeframe'
                                })
                                
                        elif msg.get('action') == 'get_subscriptions':
                            subscriptions = realtime_analysis_manager.user_subscriptions.get(user_id, [])
                            response = {
                                'type': 'subscriptions',
                                'subscriptions': [
                                    {'token': token, 'timeframe': tf} 
                                    for token, tf in subscriptions
                                ],
                                'count': len(subscriptions)
                            }
                            await websocket.send_json(response)
                            
                        elif msg.get('action') == 'get_latest_analysis':
                            token = msg.get('token')
                            timeframe = msg.get('timeframe')
                            
                            if token and timeframe:
                                analysis = await realtime_analysis_manager.get_latest_analysis(token, timeframe)
                                if analysis:
                                    response = {
                                        'type': 'latest_analysis',
                                        'token': token,
                                        'timeframe': timeframe,
                                        'analysis': make_json_serializable(analysis)
                                    }
                                else:
                                    response = {
                                        'type': 'error',
                                        'message': f'No analysis found for {token}:{timeframe}'
                                    }
                                await websocket.send_json(response)
                            else:
                                await websocket.send_json({
                                    'type': 'error',
                                    'message': 'Missing token or timeframe'
                                })
                                
                        elif msg.get('action') == 'ping':
                            response = {'type': 'pong', 'timestamp': time.time()}
                            await websocket.send_json(response)
                            
                    except json.JSONDecodeError:
                        await websocket.send_json({'type': 'error', 'message': 'Invalid JSON'})
                else:
                    # Notification from queue - send to client
                    await websocket.send_json(result)
                    
            for task in pending:
                task.cancel()
                
    except WebSocketDisconnect:
        # Clean up subscriptions when user disconnects
        await realtime_analysis_manager.unsubscribe_user(user_id)
        notification_task.cancel()
    except Exception as e:
        print(f"WebSocket real-time analysis error: {e}")
        await realtime_analysis_manager.unsubscribe_user(user_id)
        notification_task.cancel()
    finally:
        notification_task.cancel()

async def monitor_user_notifications(user_id: str, user_queue: asyncio.Queue):
    """Monitor notifications for a specific user and send them to their queue."""
    try:
        while True:
            # Check notification queue for this user
            try:
                notification_data = await asyncio.wait_for(
                    realtime_analysis_manager.notification_queue.get(), 
                    timeout=1.0
                )
                
                if notification_data.get('user_id') == user_id:
                    await user_queue.put(notification_data['notification'])
                    
            except asyncio.TimeoutError:
                # No notifications, continue monitoring
                continue
            except Exception as e:
                print(f"Error monitoring notifications for user {user_id}: {e}")
                continue
                
    except asyncio.CancelledError:
        # Task was cancelled, exit gracefully
        pass
    except Exception as e:
        print(f"Error in notification monitoring for user {user_id}: {e}")

class EnhancedAnalysisRequest(BaseModel):
    stock: str = Field(..., description="Stock symbol to analyze")
    exchange: str = Field(default="NSE", description="Stock exchange")
    period: int = Field(default=365, description="Analysis period in days")
    interval: str = Field(default="day", description="Data interval")
    output: Optional[str] = Field(default=None, description="Output directory")
    sector: Optional[str] = Field(default=None, description="Optional sector override")
    enable_code_execution: bool = Field(default=True, description="Enable mathematical validation with code execution")

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
        if request.enable_code_execution:
            result = await orchestrator.enhanced_analyze_stock(
                symbol=request.stock,
                exchange=request.exchange,
                period=request.period,
                interval=request.interval,
                output_dir=request.output,
                sector_override=request.sector
            )
        else:
            # Fallback to regular analysis
            result = await orchestrator.analyze_stock(
                symbol=request.stock,
                exchange=request.exchange,
                period=request.period,
                interval=request.interval,
                output_dir=request.output,
                sector_override=request.sector
            )
        
        # Validate and enhance the result
        validated_result = validate_analysis_results(result)
        
        # Add enhanced analysis metadata
        if 'analysis_metadata' not in validated_result:
            validated_result['analysis_metadata'] = {}
        
        validated_result['analysis_metadata'].update({
            'analysis_type': 'enhanced_with_code_execution' if request.enable_code_execution else 'standard',
            'mathematical_validation': request.enable_code_execution,
            'calculation_method': 'code_execution' if request.enable_code_execution else 'llm_estimation',
            'accuracy_improvement': 'high' if request.enable_code_execution else 'standard',
            'enhanced_timestamp': time.time()
        })
        
        # Convert charts to base64 if present
        if 'charts' in validated_result:
            validated_result['charts'] = convert_charts_to_base64(validated_result['charts'])
        
        print(f"[ENHANCED ANALYSIS] Completed enhanced analysis for {request.stock}")
        return JSONResponse(content=validated_result, status_code=200)
        
    except Exception as e:
        error_msg = f"Enhanced analysis failed for {request.stock}: {str(e)}"
        print(f"[ENHANCED ANALYSIS ERROR] {error_msg}")
        print(f"[ENHANCED ANALYSIS ERROR] Traceback: {traceback.format_exc()}")
        
        return JSONResponse(
            content={
                "error": error_msg,
                "analysis_type": "enhanced_with_code_execution" if request.enable_code_execution else "standard",
                "status": "failed",
                "timestamp": time.time()
            },
            status_code=500
        )

class HistoricalDataRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    exchange: str = Field(default="NSE", description="Stock exchange")
    interval: str = Field(..., description="Data interval (1min, 3min, 5min, 10min, 15min, 30min, 60min, 1day)")
    limit: Optional[int] = Field(default=1000, description="Maximum number of candles to return")

@app.get("/stock/{symbol}/history")
async def get_stock_history(
    symbol: str, 
    interval: str = "1day",
    exchange: str = "NSE",
    limit: int = 1000
):
    """
    Get historical OHLCV data for a stock symbol.
    This endpoint provides full historical data for charting, not limited by analysis period.
    """
    try:
        # Validate interval (include frontend intervals)
        valid_intervals = ['1min', '3min', '5min', '10min', '15min', '30min', '60min', '1h', '1day']
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
            '1h': '60minute',
            '1day': 'day'
        }
        
        backend_interval = interval_mapping.get(interval, 'day')
        
        print(f"[API] Fetching historical data for {symbol} with interval: {interval} -> {backend_interval}")
        
        # Get orchestrator and authenticate
        orchestrator = StockAnalysisOrchestrator()
        if not orchestrator.authenticate():
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        # Get token for the symbol
        token = orchestrator.data_client.get_instrument_token(symbol, exchange)
        if not token:
            raise HTTPException(status_code=404, detail=f"Token not found for {symbol}")
        
        # Retrieve historical data
        # Use appropriate periods based on interval to avoid exceeding Zerodha limits
        period_mapping = {
            'minute': 60,      # 60 days for 1min
            '3minute': 100,    # 100 days for 3min
            '5minute': 100,    # 100 days for 5min
            '10minute': 150,   # 150 days for 10min
            '15minute': 200,   # 200 days for 15min
            '30minute': 300,   # 300 days for 30min
            '60minute': 365,   # 365 days for 60min
            'day': 365 * 5     # 5 years for daily
        }
        
        max_period = period_mapping.get(backend_interval, 365)
        df = orchestrator.retrieve_stock_data(
            symbol=symbol,
            exchange=exchange,
            interval=backend_interval,
            period=max_period
        )
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
        
        # Debug: Print DataFrame info
        print(f"[API] DataFrame info for {symbol}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Index type: {type(df.index)}")
        print(f"  Index values: {df.index.tolist()[:5]}")
        print(f"  Sample data:")
        print(df.head(3))
        
        # Ensure DataFrame has proper datetime index
        if 'date' in df.columns:
            print(f"[API] Found 'date' column in DataFrame for {symbol}")
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"[API] Setting 'date' column as index for {symbol}")
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                print(f"[API] After setting index - Index type: {type(df.index)}")
                print(f"[API] Index values: {df.index.tolist()[:5]}")
            else:
                print(f"[API] DataFrame already has datetime index for {symbol}")
        else:
            print(f"[API] WARNING: No 'date' column found for {symbol}")
            print(f"[API] DataFrame columns: {df.columns.tolist()}")
            print(f"[API] DataFrame index: {df.index}")
        
        # Limit the data points if requested
        if limit > 0:
            df = df.tail(limit)
            # Ensure index is preserved after tail operation
            if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                print(f"[API] Re-setting 'date' column as index after tail operation for {symbol}")
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                print(f"[API] After tail and re-setting index - Index type: {type(df.index)}")
                print(f"[API] Index values: {df.index.tolist()[:5]}")
        
        # Convert to JSON serializable format
        candles = []
        for index, row in df.iterrows():
            # Debug: Print row info
            print(f"[API] Processing row - Index: {index} (type: {type(index)})")
            
            # Handle timestamp conversion properly
            if hasattr(index, 'timestamp'):
                # If index is a datetime object
                timestamp = int(index.timestamp())
                print(f"[API] Using index timestamp: {timestamp}")
            elif 'date' in row:
                # If date is in the row data (shouldn't happen if index is set properly)
                if isinstance(row['date'], str):
                    timestamp = int(pd.to_datetime(row['date']).timestamp())
                    print(f"[API] Using date string: {row['date']} -> {timestamp}")
                else:
                    timestamp = int(row['date'].timestamp())
                    print(f"[API] Using date object: {row['date']} -> {timestamp}")
            else:
                # Fallback: use current time
                timestamp = int(pd.Timestamp.now().timestamp())
                print(f"[API] Using fallback timestamp: {timestamp}")
            
            candle = {
                'time': timestamp,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            candles.append(candle)
            print(f"[API] Created candle: {candle}")
        
        # Sort by time (oldest first)
        candles.sort(key=lambda x: x['time'])
        
        print(f"[API] Returning {len(candles)} candles for {symbol} with interval {interval}")
        
        response = {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "token": str(token),
            "candles": candles,
            "count": len(candles),
            "first_candle": candles[0] if candles else None,
            "last_candle": candles[-1] if candles else None,
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
        df = orchestrator.retrieve_stock_data(
            symbol=symbol,
            exchange=exchange,
            interval=backend_interval,
            period=period
        )
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Calculate indicators
        indicators_data = {}
        
        if 'rsi' in requested_indicators:
            from technical_indicators import calculate_rsi
            rsi_values = calculate_rsi(df['close'], period=14)
            indicators_data['rsi'] = [float(val) if not pd.isna(val) else None for val in rsi_values]
        
        if 'macd' in requested_indicators:
            from technical_indicators import calculate_macd
            macd_result = calculate_macd(df['close'])
            indicators_data['macd'] = {
                'macd': [float(val) if not pd.isna(val) else None for val in macd_result['macd']],
                'signal': [float(val) if not pd.isna(val) else None for val in macd_result['signal']],
                'histogram': [float(val) if not pd.isna(val) else None for val in macd_result['histogram']]
            }
        
        if 'sma' in requested_indicators:
            from technical_indicators import calculate_sma
            sma_20 = calculate_sma(df['close'], period=20)
            sma_50 = calculate_sma(df['close'], period=50)
            sma_200 = calculate_sma(df['close'], period=200)
            indicators_data['sma'] = {
                'sma_20': [float(val) if not pd.isna(val) else None for val in sma_20],
                'sma_50': [float(val) if not pd.isna(val) else None for val in sma_50],
                'sma_200': [float(val) if not pd.isna(val) else None for val in sma_200]
            }
        
        if 'ema' in requested_indicators:
            from technical_indicators import calculate_ema
            ema_12 = calculate_ema(df['close'], period=12)
            ema_26 = calculate_ema(df['close'], period=26)
            ema_50 = calculate_ema(df['close'], period=50)
            indicators_data['ema'] = {
                'ema_12': [float(val) if not pd.isna(val) else None for val in ema_12],
                'ema_26': [float(val) if not pd.isna(val) else None for val in ema_26],
                'ema_50': [float(val) if not pd.isna(val) else None for val in ema_50]
            }
        
        if 'bollinger' in requested_indicators:
            from technical_indicators import calculate_bollinger_bands
            bb_result = calculate_bollinger_bands(df['close'], period=20, std_dev=2)
            indicators_data['bollinger_bands'] = {
                'upper': [float(val) if not pd.isna(val) else None for val in bb_result['upper']],
                'middle': [float(val) if not pd.isna(val) else None for val in bb_result['middle']],
                'lower': [float(val) if not pd.isna(val) else None for val in bb_result['lower']]
            }
        
        # Ensure proper datetime index for timestamp conversion
        if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            print(f"[API] Setting 'date' column as index for indicators endpoint")
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Get timestamps for alignment
        timestamps = []
        for index in df.index:
            if hasattr(index, 'timestamp'):
                timestamps.append(int(index.timestamp()))
            else:
                print(f"[API] Warning: Index {index} has no timestamp method, using fallback")
                timestamps.append(int(pd.Timestamp.now().timestamp()))
        
        response = {
            "success": True,
            "symbol": symbol,
            "interval": interval,
            "indicators": indicators_data,
            "timestamps": timestamps,
            "count": len(timestamps),
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

# --- Token/Symbol Mapping Endpoints ---
@app.get("/mapping/token-to-symbol")
async def token_to_symbol(token: int, exchange: str = "NSE"):
    """
    Map an instrument token to its trading symbol.
    Args:
        token: Instrument token (int)
        exchange: Exchange code (default: NSE)
    Returns:
        JSON with symbol or error
    """
    try:
        data_client = ZerodhaDataClient()
        symbol = data_client.get_symbol_from_token(token, exchange)
        if not symbol:
            raise HTTPException(status_code=404, detail=f"Symbol not found for token {token} on {exchange}")
        return {"token": token, "exchange": exchange, "symbol": symbol}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mapping/symbol-to-token")
async def symbol_to_token(symbol: str, exchange: str = "NSE"):
    """
    Map a trading symbol to its instrument token.
    Args:
        symbol: Trading symbol (str)
        exchange: Exchange code (default: NSE)
    Returns:
        JSON with token or error
    """
    try:
        data_client = ZerodhaDataClient()
        token = data_client.get_instrument_token(symbol, exchange)
        if not token:
            raise HTTPException(status_code=404, detail=f"Token not found for symbol {symbol} on {exchange}")
        return {"symbol": symbol, "exchange": exchange, "token": token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Market Hours and Optimization Endpoints

@app.get("/market/status")
async def get_market_status():
    """Get current market status - always returns open for continuous data flow."""
    try:
        return {
            "current_time": datetime.datetime.now().isoformat(),
            "timezone": "Asia/Kolkata",
            "market_status": "open",
            "is_weekend": False,
            "is_holiday": False,
            "continuous_flow_enabled": True,
            "market_always_open": True,
            "regular_session": {
                "start": "09:15:00",
                "end": "15:30:00",
                "name": "Continuous Trading"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting market status: {str(e)}")

@app.get("/market/optimization/strategy")
async def get_optimization_strategy(symbol: str, interval: str = "1d"):
    """Get optimal data fetching strategy - always recommends live data for continuous flow."""
    try:
        strategy = {
            "market_status": "open",
            "current_time": datetime.datetime.now().isoformat(),
            "recommended_approach": "live",
            "reason": "Continuous data flow enabled - always use live data",
            "cost_efficiency": "continuous_flow",
            "data_freshness": "real_time",
            "websocket_recommended": True,
            "cache_duration": 60,  # 1 minute
            "next_update": None,
            "continuous_flow": True,
            "market_always_open": True
        }
        return {
            "symbol": symbol,
            "interval": interval,
            "strategy": strategy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting optimization strategy: {str(e)}")

@app.get("/market/optimization/stats")
async def get_optimization_stats():
    """Get optimization statistics and cost analysis."""
    try:
        return {
            "data_service_stats": enhanced_data_service.get_optimization_stats(),
            "websocket_stats": get_zerodha_ws_client().get_optimization_stats(),
            "cost_analysis": enhanced_data_service.get_cost_analysis()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting optimization stats: {str(e)}")

@app.post("/market/optimization/clear-cache")
async def clear_optimization_cache():
    """Clear all optimization caches."""
    try:
        enhanced_data_service.clear_cache()
        return {"message": "Optimization cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@app.post("/market/optimization/clear-interval-cache")
async def clear_interval_cache(symbol: str, interval: str):
    """Clear cache for a specific symbol and interval."""
    try:
        enhanced_data_service.clear_interval_cache(symbol, interval)
        return {"message": f"Cache cleared for {symbol} with interval {interval}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing interval cache: {str(e)}")

class OptimizedDataRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    exchange: str = Field(default="NSE", description="Stock exchange")
    interval: str = Field(default="1d", description="Data interval")
    period: int = Field(default=365, description="Analysis period in days")
    force_live: bool = Field(default=False, description="Force live data even if market is closed")

@app.post("/market/optimization/data")
async def get_optimized_data(request: OptimizedDataRequest):
    """Get optimized data based on market status and cost efficiency."""
    try:
        data_request = DataRequest(
            symbol=request.symbol,
            exchange=request.exchange,
            interval=request.interval,
            period=request.period,
            force_live=request.force_live
        )
        
        response = enhanced_data_service.get_optimal_data(data_request)
        
        # Convert DataFrame to JSON-serializable format
        if not response.data.empty:
            data_dict = response.data.reset_index().to_dict('records')
        else:
            data_dict = []
        
        return {
            "symbol": request.symbol,
            "data": data_dict,
            "metadata": {
                "data_freshness": response.data_freshness,
                "market_status": response.market_status,
                "source": response.source,
                "cache_until": response.cache_until.isoformat() if response.cache_until else None,
                "cost_estimate": response.cost_estimate,
                "optimization_applied": response.optimization_applied
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting optimized data: {str(e)}")

@app.get("/market/optimization/recommendations")
async def get_optimization_recommendations():
    """Get cost optimization recommendations based on current market status."""
    try:
        return {
            "recommendations": enhanced_data_service._get_cost_recommendations(),
            "market_status": "open",
            "current_time": datetime.datetime.now().isoformat(),
            "continuous_flow_enabled": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")