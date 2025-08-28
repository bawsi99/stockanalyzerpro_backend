"""
data_service.py

Data Service - Handles all data fetching, WebSocket connections, and real-time data streaming.
This service is responsible for:
- Historical data retrieval
- Real-time data streaming via WebSocket
- Market data caching and optimization
- Token/symbol mapping
- Market status monitoring
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
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Local imports
from zerodha_ws_client import zerodha_ws_client, candle_aggregator
# Market hours manager removed - continuous data flow enabled
from enhanced_data_service import enhanced_data_service, DataRequest
from zerodha_client import ZerodhaDataClient

# Security configuration
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')
API_KEY_HEADER = 'X-API-Key'
API_KEYS = os.getenv('API_KEYS', '').split(',')  # Comma-separated API keys
REQUIRE_AUTH = os.getenv('REQUIRE_AUTH', 'false').lower() == 'true'

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
    print(f"🔐 WebSocket authentication attempt - REQUIRE_AUTH: {REQUIRE_AUTH}")
    
    # First, validate the origin
    origin = websocket.headers.get('origin')
    print(f"🔐 Origin: {origin}")
    print(f"🔐 Allowed origins: {CORS_ORIGINS}")
    
    if origin and origin not in CORS_ORIGINS:
        print(f"❌ WebSocket connection rejected from unauthorized origin: {origin}")
        print(f"   Allowed origins: {CORS_ORIGINS}")
        return None
    
    if not REQUIRE_AUTH:
        print(f"✅ Authentication disabled, allowing connection")
        return {'user_id': str(uuid.uuid4()), 'auth_type': 'none'}
    
    # Try to get token from query parameters or headers
    token = websocket.query_params.get('token')
    api_key = websocket.headers.get(API_KEY_HEADER)
    
    print(f"🔐 Token from query params: {token[:20] if token else 'None'}...")
    print(f"🔐 API key from headers: {api_key[:10] if api_key else 'None'}...")
    
    if token and token != "undefined":
        # JWT authentication
        payload = verify_jwt_token(token)
        if payload and token in active_tokens:
            print(f"✅ JWT authentication successful for user: {payload['user_id']}")
            return {'user_id': payload['user_id'], 'auth_type': 'jwt'}
        else:
            print(f"❌ JWT authentication failed - token invalid or not in active tokens")
    elif api_key:
        # API key authentication
        if verify_api_key(api_key):
            print(f"✅ API key authentication successful")
            return {'user_id': f'api_user_{api_key[:8]}', 'auth_type': 'api_key'}
        else:
            print(f"❌ API key authentication failed")
    
    print(f"❌ No valid authentication found")
    return None

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

app = FastAPI(title="Stock Data Service", version="1.0.0")

# Load CORS origins from environment variable
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:8080,http://127.0.0.1:5173,https://stock-analyzer-pro.vercel.app,https://stock-analyzer-pro-git-prototype-aaryan-manawats-projects.vercel.app,https://stock-analyzer-cl9o3tivx-aaryan-manawats-projects.vercel.app,https://stockanalyzer-pro.vercel.app").split(",")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Only allow specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Root route
@app.get("/")
async def root():
    """Root endpoint for the Data Service."""
    return {
        "service": "Stock Data Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "stock_data": "/stock/{symbol}/history",
            "stock_info": "/stock/{symbol}/info",
            "websocket": "/ws/stream",
            "market_status": "/market/status"
        },
        "timestamp": datetime.now().isoformat()
    }

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

    async def _subscribe_to_zerodha(self, tokens):
        """Subscribe to tokens in Zerodha WebSocket if not already subscribed."""
        new_tokens = [token for token in tokens if token not in self.global_subscribed_tokens]
        if new_tokens:
            try:
                # Check if Zerodha WebSocket client is running
                if not hasattr(zerodha_ws_client, 'running') or not zerodha_ws_client.running:
                    print(f"⚠️  Zerodha WebSocket client not running. Cannot subscribe to tokens: {new_tokens}")
                    print("📊 Historical data is still available via REST API endpoints")
                    # Add to global subscribed tokens anyway - they will be picked up when client reconnects
                    self.global_subscribed_tokens.update(new_tokens)
                    return
                
                # Convert string tokens to integers for Zerodha
                int_tokens = [int(token) for token in new_tokens]
                zerodha_ws_client.subscribe(int_tokens)
                zerodha_ws_client.set_mode('quote', int_tokens)
                self.global_subscribed_tokens.update(new_tokens)
                print(f"🔗 Subscribed to new Zerodha tokens: {new_tokens}")
            except Exception as e:
                print(f"❌ Error subscribing to Zerodha tokens {new_tokens}: {e}")
                print("📊 Historical data is still available via REST API endpoints")

    async def _unsubscribe_from_zerodha(self, tokens):
        """Unsubscribe from tokens in Zerodha WebSocket if no clients are subscribed."""
        tokens_to_unsubscribe = [token for token in tokens if token in self.global_subscribed_tokens]
        if tokens_to_unsubscribe:
            try:
                # Convert string tokens to integers for Zerodha
                int_tokens = [int(token) for token in tokens_to_unsubscribe]
                zerodha_ws_client.unsubscribe(int_tokens)
                self.global_subscribed_tokens.difference_update(tokens_to_unsubscribe)
                print(f"🔗 Unsubscribed from Zerodha tokens: {tokens_to_unsubscribe}")
            except Exception as e:
                print(f"❌ Error unsubscribing from Zerodha tokens {tokens_to_unsubscribe}: {e}")

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
    # Import global variables from zerodha_ws_client
    from zerodha_ws_client import publish_callback, system_ready, MAIN_EVENT_LOOP
    
    data = {
        'type': 'tick',
        'token': int(token),
        'price': float(tick.get('last_price', 0)),
        'timestamp': float(tick.get('timestamp', time.time())),
        'volume_traded': float(tick.get('volume_traded', 0)),
    }
    # Use the existing publish mechanism that handles async context properly
    if MAIN_EVENT_LOOP is not None and publish_callback is not None and system_ready:
        MAIN_EVENT_LOOP.call_soon_threadsafe(asyncio.create_task, publish_callback(data))
    else:
        print(f"⚠️  Cannot publish tick data - system not ready")

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
        asyncio.create_task(alert_publish_hook(transformed_candle))
    except RuntimeError:
        pass

# Register hooks
candle_aggregator.register_callback(candle_forward_hook)
zerodha_ws_client.register_tick_hook(tick_forward_hook)

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
    
    # Check if Zerodha credentials are available before starting WebSocket client
    api_key = os.getenv("ZERODHA_API_KEY")
    access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
    
    if not api_key or not access_token or api_key == 'your_api_key' or access_token == 'your_access_token':
        print("⚠️  Zerodha credentials not configured or invalid")
        print("📊 Historical data will be available via REST API")
        print("🔴 Live data streaming will be disabled")
        print("💡 To enable live data, set ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN in .env file")
    else:
        print("✅ Zerodha credentials found")
        print("🚀 Starting Zerodha WebSocket client...")
        try:
            zerodha_ws_client.connect()
            await asyncio.sleep(3)
            
            if zerodha_ws_client.running:
                print("✅ Zerodha WebSocket client started successfully")
                print("✅ Live data streaming is enabled")
            else:
                print("⚠️  Zerodha WebSocket client failed to start")
                print("📊 Historical data will still be available")
        except Exception as e:
            print(f"❌ Error starting Zerodha WebSocket client: {e}")
            print("📊 Historical data will still be available")
    
    print("🔧 WebSocket mode will be set to 'quote' for OHLCV data when symbols are subscribed")
    print("🎯 Data service is ready to handle requests")

# --- Pydantic Models ---
class HistoricalDataRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    exchange: str = Field(default="NSE", description="Stock exchange")
    interval: str = Field(..., description="Data interval (1min, 3min, 5min, 10min, 15min, 30min, 60min, 1day)")
    limit: Optional[int] = Field(default=1000, description="Maximum number of candles to return")

class OptimizedDataRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    exchange: str = Field(default="NSE", description="Stock exchange")
    interval: str = Field(default="1d", description="Data interval")
    period: int = Field(default=365, description="Analysis period in days")
    force_live: bool = Field(default=False, description="Force live data even if market is closed")

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
    
    # Subscription tracking is now handled centrally by live_pubsub
    
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
                            symbols = msg.get('symbols', [])  # Frontend sends symbols
                            tokens = msg.get('tokens', [])     # Also support direct tokens
                            timeframes = msg.get('timeframes', [])
                            throttle_ms = msg.get('throttle_ms', 0)
                            batch = msg.get('batch', False)
                            batch_size = msg.get('batch_size', 1)
                            format_type = msg.get('format', 'json')
                            
                            # Convert symbols to tokens if provided
                            all_tokens = []
                            if symbols:
                                try:
                                    # Read directly from CSV file to avoid authentication issues
                                    df = pd.read_csv('zerodha_instruments.csv')
                                    for symbol in symbols:
                                        instrument = df[(df['tradingsymbol'] == symbol) & (df['exchange'] == 'NSE')]
                                        if len(instrument) > 0:
                                            token = int(instrument.iloc[0]['instrument_token'])
                                            all_tokens.append(token)
                                            print(f"Converted symbol {symbol} to token {token}")
                                        else:
                                            print(f"Warning: Could not find token for symbol {symbol}")
                                except Exception as e:
                                    print(f"Error converting symbols to tokens: {e}")
                            
                            # Add direct tokens if provided
                            if tokens:
                                all_tokens.extend([int(token) for token in tokens])
                            
                            # Token subscription is now handled centrally by live_pubsub
                            # No need to manage individual subscriptions here
                            
                            # IMPORTANT: keep token type consistent with publisher (use integers)
                            await live_pubsub.update_filter(
                                queue,
                                tokens=all_tokens,
                                timeframes=timeframes,
                                throttle_ms=throttle_ms,
                                batch=batch,
                                batch_size=batch_size,
                                format=format_type
                            )
                            
                            response = {
                                'type': 'subscribed',
                                'symbols': symbols,
                                'tokens': [str(token) for token in all_tokens],
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
                            symbols = msg.get('symbols', [])
                            tokens = msg.get('tokens', [])
                            timeframes = msg.get('timeframes', [])
                            
                            # Convert symbols to tokens if provided
                            all_tokens = []
                            if symbols:
                                try:
                                    # Read directly from CSV file to avoid authentication issues
                                    df = pd.read_csv('zerodha_instruments.csv')
                                    for symbol in symbols:
                                        instrument = df[(df['tradingsymbol'] == symbol) & (df['exchange'] == 'NSE')]
                                        if len(instrument) > 0:
                                            token = int(instrument.iloc[0]['instrument_token'])
                                            all_tokens.append(token)
                                except Exception as e:
                                    print(f"Error converting symbols to tokens for unsubscribe: {e}")
                            
                            # Add direct tokens if provided
                            if tokens:
                                all_tokens.extend([int(token) for token in tokens])
                            
                            # Token unsubscription is now handled centrally by live_pubsub
                            # Just update the filter to remove tokens/timeframes
                            # Remove only the specified tokens/timeframes instead of clearing everything
                            # If none provided, clear all as a fallback
                            if not symbols and not tokens and not timeframes:
                                await live_pubsub.update_filter(queue, tokens=[], timeframes=[])
                            else:
                                # Compute remaining set by subtracting provided tokens
                                current = live_pubsub.clients.get(queue, {}).get('tokens', set())
                                to_remove = set(all_tokens)
                                remaining = [t for t in current if t not in to_remove]
                                await live_pubsub.update_filter(queue, tokens=remaining)
                            
                            response = {
                                'type': 'unsubscribed',
                                'symbols': symbols,
                                'tokens': [str(token) for token in all_tokens],
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

# --- REST API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "data_service", "timestamp": pd.Timestamp.now().isoformat()}

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
        'zerodha_ws_running': zerodha_ws_client.running,
        'subscribed_tokens': list(zerodha_ws_client.subscribed_tokens),
        'api_key_configured': bool(zerodha_ws_client.api_key and zerodha_ws_client.api_key != 'your_api_key'),
        'access_token_configured': bool(zerodha_ws_client.access_token and zerodha_ws_client.access_token != 'your_access_token'),
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
        
        # print(f"[DataService] Fetching historical data for {symbol} with interval: {interval} -> {backend_interval}")
        
        # Get orchestrator and authenticate
        zerodha_client = ZerodhaDataClient()
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        # Get token for the symbol
        token = zerodha_client.get_instrument_token(symbol, exchange)
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
        df = await zerodha_client.get_historical_data_async(
            symbol=symbol,
            exchange=exchange,
            interval=backend_interval,
            period=max_period
        )
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
        
        # Ensure proper datetime index for timestamp conversion
        if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            # print(f"[DataService] Setting 'date' column as index for {symbol}")
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            # print(f"[DataService] Index type after setting: {type(df.index)}")
            # print(f"[DataService] Index values: {df.index.tolist()[:5]}")
        
        # Limit the data points if requested
        if limit > 0:
            df = df.tail(limit)
            # Ensure index is preserved after tail operation
            if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                # print(f"[DataService] Re-setting 'date' column as index after tail operation for {symbol}")
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                # print(f"[DataService] After tail and re-setting index - Index type: {type(df.index)}")
                # print(f"[DataService] Index values: {df.index.tolist()[:5]}")
        
        # Convert to JSON serializable format
        candles = []
        for index, row in df.iterrows():
            # Handle timestamp conversion properly
            if hasattr(index, 'timestamp'):
                # If index is a datetime object
                timestamp = int(index.timestamp())
            elif 'date' in row:
                # If date is in the row data (shouldn't happen if index is set properly)
                if isinstance(row['date'], str):
                    timestamp = int(pd.to_datetime(row['date']).timestamp())
                else:
                    timestamp = int(row['date'].timestamp())
            else:
                # Fallback: use current time
                timestamp = int(pd.Timestamp.now().timestamp())
            
            candle = {
                'time': timestamp,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            candles.append(candle)
        
        # Sort by time (oldest first)
        candles.sort(key=lambda x: x['time'])
        
        # print(f"[DataService] Returning {len(candles)} candles for {symbol} with interval {interval}")
        
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

@app.post("/data/optimized")
async def get_optimized_data(request: OptimizedDataRequest):
    """
    Get optimized data based on market status and cost efficiency.
    This endpoint uses the enhanced data service for intelligent data fetching.
    """
    try:
        # Create data request
        data_request = DataRequest(
            symbol=request.symbol,
            exchange=request.exchange,
            interval=request.interval,
            period=request.period,
            force_live=request.force_live
        )
        
        # Get optimized data
        response = await enhanced_data_service.get_optimal_data(data_request)
        
        if response.data is None or response.data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Ensure proper datetime index for timestamp conversion
        if 'date' in response.data.columns and not isinstance(response.data.index, pd.DatetimeIndex):
            # print(f"[DataService] Setting 'date' column as index for optimized data")
            response.data['date'] = pd.to_datetime(response.data['date'])
            response.data.set_index('date', inplace=True)
        
        # Convert to JSON serializable format
        candles = []
        for index, row in response.data.iterrows():
            # Handle timestamp conversion properly
            if hasattr(index, 'timestamp'):
                # If index is a datetime object
                timestamp = int(index.timestamp())
            elif 'date' in row:
                # If date is in the row data (shouldn't happen if index is set properly)
                if isinstance(row['date'], str):
                    timestamp = int(pd.to_datetime(row['date']).timestamp())
                else:
                    timestamp = int(row['date'].timestamp())
            else:
                # Fallback: use current time
                timestamp = int(pd.Timestamp.now().timestamp())
            
            candle = {
                'time': timestamp,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            candles.append(candle)
        
        # Sort by time (oldest first)
        candles.sort(key=lambda x: x['time'])
        
        response_data = {
            "success": True,
            "symbol": request.symbol,
            "exchange": request.exchange,
            "interval": request.interval,
            "candles": candles,
            "count": len(candles),
            "data_freshness": response.data_freshness,
            "market_status": response.market_status,
            "source": response.source,
            "optimization_applied": response.optimization_applied,
            "cost_estimate": response.cost_estimate,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "symbol": request.symbol,
            "interval": request.interval,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/stock/{symbol}/info")
async def get_stock_info(symbol: str, exchange: str = "NSE"):
    """Get basic stock information."""
    try:
        zerodha_client = ZerodhaDataClient()
        
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        # Get current quote
        quote = zerodha_client.get_quote(symbol, exchange)
        
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

@app.get("/mapping/token-to-symbol")
async def token_to_symbol(token: int, exchange: str = "NSE"):
    """Get symbol for a given token."""
    try:
        zerodha_client = ZerodhaDataClient()
        symbol = zerodha_client.get_symbol_from_token(token)
        
        if symbol:
            return {
                "success": True,
                "token": token,
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Symbol not found for token {token}")
            
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "token": token,
            "exchange": exchange,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/mapping/symbol-to-token")
async def symbol_to_token(symbol: str, exchange: str = "NSE"):
    """Get token for a given symbol."""
    try:
        # Read directly from CSV file to avoid authentication issues
        df = pd.read_csv('zerodha_instruments.csv')
        instrument = df[(df['tradingsymbol'] == symbol) & (df['exchange'] == exchange)]
        
        if len(instrument) > 0:
            token = instrument.iloc[0]['instrument_token']
            return {
                "success": True,
                "symbol": symbol,
                "token": int(token),
                "exchange": exchange,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Token not found for symbol {symbol}")
            
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/market/status")
async def get_market_status():
    """Get current market status - always returns open for continuous data flow."""
    try:
        return {
            "success": True,
            "market_status": "open",
            "continuous_flow_enabled": True,
            "market_always_open": True,
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

@app.get("/market/optimization/stats")
async def get_optimization_stats():
    """Get data optimization statistics."""
    try:
        stats = enhanced_data_service.get_optimization_stats()
        return {
            "success": True,
            "optimization_stats": stats,
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

@app.post("/market/optimization/clear-cache")
async def clear_optimization_cache():
    """Clear the data cache."""
    try:
        enhanced_data_service.clear_cache()
        return {
            "success": True,
            "message": "Cache cleared successfully",
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

@app.post("/market/optimization/clear-interval-cache")
async def clear_interval_cache(symbol: str, interval: str):
    """Clear cache for a specific symbol and interval."""
    try:
        enhanced_data_service.clear_interval_cache(symbol, interval)
        return {
            "success": True,
            "message": f"Cache cleared for {symbol} with interval {interval}",
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

@app.post("/auth/token")
async def create_token(user_id: str):
    """Create a JWT token for WebSocket authentication."""
    try:
        if not JWT_AVAILABLE:
            raise HTTPException(status_code=500, detail="JWT library not available")
        
        # Always create a token when this endpoint is called
        # The REQUIRE_AUTH setting controls whether WebSocket connections require authentication
        token = create_jwt_token(user_id)
        
        # Store the token in active tokens set
        active_tokens.add(token)
        
        print(f"🔐 Created JWT token for user: {user_id}")
        return {"token": token, "user_id": user_id}
        
    except Exception as e:
        print(f"❌ Error creating JWT token: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create token: {str(e)}")

@app.get("/auth/verify")
async def verify_token(token: str):
    """Verify a JWT token."""
    try:
        if not token or token == "undefined":
            return {"valid": False, "error": "No token provided"}
        
        if not JWT_AVAILABLE:
            return {"valid": False, "error": "JWT library not available"}
        
        payload = verify_jwt_token(token)
        if payload:
            # Check if token is in active tokens set
            if token in active_tokens:
                return {"valid": True, "user_id": payload['user_id']}
            else:
                return {"valid": False, "error": "Token not found in active tokens"}
        else:
            return {"valid": False, "error": "Invalid or expired token"}
            
    except Exception as e:
        print(f"❌ Error verifying token: {e}")
        return {"valid": False, "error": f"Verification failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 