"""
zerodha_ws_client.py

WebSocket client for Zerodha real-time data with market hours optimization.
WARNING: Requires MAIN_EVENT_LOOP to be set by FastAPI startup before any tick processing or WebSocket publishing. Do not use tick publishing before FastAPI startup event.
"""
import os
import threading
import time
import logging
import datetime
import struct
import random
from collections import defaultdict
from typing import Dict, List, Optional, Any, Callable
import asyncio

# Market status enum (simplified without market hours manager)
from enum import Enum

class MarketStatus(Enum):
    """Market status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    POST_MARKET = "post_market"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"

# Try to import optional dependencies
try:
    import redis
    REDIS_AVAILABLE = True
    redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
except ImportError:
    REDIS_AVAILABLE = False
    redis_client = None

try:
    from kiteconnect import KiteTicker
    KITECONNECT_AVAILABLE = True
except ImportError:
    KITECONNECT_AVAILABLE = False
    KiteTicker = None

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

# Configurations (should be set via environment variables or config file)
API_KEY = os.getenv('ZERODHA_API_KEY') or os.getenv('KITE_API_KEY', 'your_api_key')
ACCESS_TOKEN = os.getenv('ZERODHA_ACCESS_TOKEN') or os.getenv('KITE_ACCESS_TOKEN', 'your_access_token')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global data stores
tick_store: Dict[int, Dict[str, Any]] = {}
tick_store_lock = threading.Lock()

# Timeframe definitions (in seconds)
TIMEFRAMES = {
    '1m': 60,
    '5m': 300,
    '15m': 900,
    '1h': 3600,
    '4h': 14400,  # 4 hours
    '1d': 86400
}

def floor_time(ts: float, tf: str) -> float:
    """Floor a timestamp to the nearest timeframe bucket."""
    if tf == '1d':
        return datetime.datetime.fromtimestamp(ts).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    return ts - (ts % TIMEFRAMES[tf])

class CandleAggregator:
    """Aggregates ticks into candles for multiple timeframes and triggers callbacks on new candles."""
    
    def __init__(self, timeframes: List[str]):
        self.timeframes = timeframes
        self.candles = defaultdict(lambda: defaultdict(dict))
        self.callbacks: List[Callable] = []

    def register_callback(self, fn: Callable) -> None:
        """Register a callback to be called when a new candle is formed."""
        self.callbacks.append(fn)

    def process_tick(self, tick: Dict[str, Any]) -> None:
        """Process a tick and update or create candles for all timeframes."""
        token = tick['instrument_token']
        ts = tick.get('timestamp')
        
        if not ts:
            ts = int(time.time())
        elif isinstance(ts, str):
            ts = int(datetime.datetime.fromisoformat(ts).timestamp())
        
        for tf in self.timeframes:
            bucket = floor_time(ts, tf)
            cdict = self.candles[token][tf]
            candle = cdict.get(bucket)
            price = tick['last_price']
            
            if not candle:
                candle = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': tick.get('volume_traded', 0),
                    'start': bucket,
                    'end': bucket + TIMEFRAMES[tf] - 1
                }
                cdict[bucket] = candle
                prev_bucket = bucket - TIMEFRAMES[tf]
                if prev_bucket in cdict:
                    prev_candle = cdict[prev_bucket]
                    for cb in self.callbacks:
                        cb(token, tf, prev_candle)
            else:
                candle['high'] = max(candle['high'], price)
                candle['low'] = min(candle['low'], price)
                candle['close'] = price
                candle['volume'] += tick.get('volume_traded', 0)

    def get_latest_candle(self, token: int, tf: str) -> Optional[Dict[str, Any]]:
        """Get the latest candle for a token and timeframe."""
        cdict = self.candles[token][tf]
        if not cdict:
            return None
        latest_bucket = max(cdict.keys())
        return cdict[latest_bucket]

# Singleton aggregator
candle_aggregator = CandleAggregator(list(TIMEFRAMES.keys()))

# Global variables that will be set by api.py
MAIN_EVENT_LOOP = None

def set_main_event_loop(loop):
    """Set the main event loop from api.py"""
    global MAIN_EVENT_LOOP
    MAIN_EVENT_LOOP = loop
    logger.info(f"[SYSTEM] Main event loop set: {loop}")

# Global publish callback - will be set by api.py after initialization
publish_callback = None
system_ready = False

def set_publish_callback(callback):
    """Set the publish callback function from api.py"""
    global publish_callback, system_ready
    publish_callback = callback
    system_ready = True
    logger.info("[SYSTEM] Publish callback set and system marked as ready")

class ZerodhaWSClient:
    """WebSocket client for Zerodha real-time data, with extensible event hooks and thread safety."""
    
    def __init__(self, api_key: str, access_token: str):
        if not KITECONNECT_AVAILABLE:
            raise ImportError("kiteconnect library is required but not installed")
            
        self.api_key = api_key
        self.access_token = access_token
        self.kws = KiteTicker(api_key, access_token)
        self.subscribed_tokens: set = set()
        self._ws_thread: Optional[threading.Thread] = None
        self.running: bool = False
        self.tick_hooks: List[Callable] = []
        
        # Always treat market as open for continuous data flow
        self.market_status = MarketStatus.OPEN
        self.last_tick_time = {}  # Track last tick time per token
        self.duplicate_tick_threshold = 30  # Seconds to consider tick as duplicate
        
        self._setup_callbacks()

    def parse_binary_message(self, data: bytes) -> List[Dict[str, Any]]:
        """Parse Zerodha's binary message structure according to their documentation."""
        try:
            if len(data) < 4:
                return []
            
            # First 2 bytes: number of packets (SHORT/int16)
            num_packets = struct.unpack('!H', data[0:2])[0]
            # Next 2 bytes: length of first packet (SHORT/int16)
            first_packet_length = struct.unpack('!H', data[2:4])[0]
            
            packets = []
            offset = 4
            
            for i in range(num_packets):
                if offset + first_packet_length > len(data):
                    break
                
                # Extract packet data
                packet_data = data[offset:offset + first_packet_length]
                tick = self.parse_quote_packet(packet_data)
                if tick:
                    packets.append(tick)
                
                offset += first_packet_length
                
                # Get next packet length if not the last packet
                if i < num_packets - 1 and offset + 2 <= len(data):
                    first_packet_length = struct.unpack('!H', data[offset:offset + 2])[0]
                    offset += 2
            
            return packets
        except Exception as e:
            logger.error(f"Error parsing binary message: {e}")
            return []

    def parse_quote_packet(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Parse individual quote packet according to Zerodha's documentation."""
        try:
            if len(data) < 44:  # Minimum size for quote mode
                return None
            
            # Parse according to Zerodha's quote packet structure
            # All prices are in paise, need to divide by 100 for rupees
            unpacked = struct.unpack('!IIIIIIIIIIII', data[:48])
            
            tick = {
                'instrument_token': unpacked[0],
                'last_price': unpacked[1] / 100.0,  # Convert from paise to rupees
                'last_quantity': unpacked[2],
                'average_price': unpacked[3] / 100.0,
                'volume_traded': unpacked[4],
                'total_buy_quantity': unpacked[5],
                'total_sell_quantity': unpacked[6],
                'open': unpacked[7] / 100.0,
                'high': unpacked[8] / 100.0,
                'low': unpacked[9] / 100.0,
                'close': unpacked[10] / 100.0,
                'timestamp': int(time.time())
            }
            
            return tick
        except Exception as e:
            logger.error(f"Error parsing quote packet: {e}")
            return None

    def register_tick_hook(self, fn: Callable) -> None:
        """Register a function to be called with each batch of ticks."""
        self.tick_hooks.append(fn)

    def unregister_tick_hook(self, fn: Callable) -> None:
        """Unregister a previously registered tick hook function."""
        if fn in self.tick_hooks:
            self.tick_hooks.remove(fn)

    def _setup_callbacks(self) -> None:
        """Set up WebSocket event callbacks."""
        self.kws.on_ticks = self.on_ticks
        self.kws.on_connect = self.on_connect
        self.kws.on_close = self.on_close
        self.kws.on_error = self.on_error
        self.kws.on_reconnect = self.on_reconnect
        self.kws.on_noreconnect = self.on_noreconnect
        self.kws.on_order_update = self.on_order_update
    
    def _get_market_status(self) -> MarketStatus:
        """Get current market status - always returns OPEN for continuous data flow."""
        return self.market_status
    
    def _is_duplicate_tick(self, token: int, tick_data: Dict[str, Any]) -> bool:
        """Check if tick is duplicate based on price and time."""
        # Always process all ticks for continuous data flow
        return False
            
        current_time = time.time()
        last_time = self.last_tick_time.get(token, 0)
        
        # If it's been too long since last tick, it's not a duplicate
        if current_time - last_time > self.duplicate_tick_threshold:
            return False
        
        # Check if price hasn't changed (common when market is closed)
        with tick_store_lock:
            if token in tick_store:
                last_tick = tick_store[token]
                if (last_tick.get('last_price') == tick_data.get('last_price') and
                    last_tick.get('volume_traded') == tick_data.get('volume_traded')):
                    logger.info(f"[DUPLICATE] Tick is duplicate for token {token} - same price and volume")
                    return True
        
        return False
    
    def _should_process_tick(self, token: int, tick_data: Dict[str, Any]) -> bool:
        """Determine if tick should be processed - always process for continuous data flow."""
        # Always process all ticks for continuous data flow
        # logger.info(f"[CONTINUOUS FLOW] Processing tick for token {token}")
        return True

    def connect(self) -> None:
        """Start the WebSocket client in a background thread."""
        if not self.running:
            try:
                logger.info(f"Attempting to connect to Zerodha WebSocket with API key: {self.api_key[:10]}...")
                if self.api_key == 'your_api_key' or self.access_token == 'your_access_token':
                    logger.warning("Zerodha credentials not properly configured. Cannot connect to real data.")
                    logger.warning("Please set ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN environment variables for real data.")
                    logger.info("Historical data will still be available, but live streaming will be disabled.")
                    return
                
                # Validate credentials before attempting connection
                if not self.api_key or not self.access_token:
                    logger.warning("Missing Zerodha credentials. Live data streaming disabled.")
                    logger.info("Historical data will still be available, but live streaming will be disabled.")
                    return
                    
                self._ws_thread = threading.Thread(target=self.kws.connect, kwargs={"threaded": True})
                self._ws_thread.daemon = True
                self._ws_thread.start()
                self.running = True
                logger.info("Kite WebSocket client started successfully.")
            except Exception as e:
                logger.error(f"Failed to start Kite WebSocket client: {e}")
                logger.error("Cannot connect to real data. Please check your Zerodha credentials and network connection.")
                logger.info("Historical data will still be available, but live streaming will be disabled.")
                self.running = False

    def subscribe(self, tokens: List[int]) -> None:
        """Subscribe to a list of instrument tokens."""
        if not isinstance(tokens, list):
            tokens = [tokens]
        
        try:
            if not self.running:
                logger.warning("WebSocket client not running. Cannot subscribe to tokens.")
                logger.info("This is likely due to missing or invalid Zerodha credentials.")
                logger.info("Historical data is still available via REST API endpoints.")
                return
                
            logger.info(f"Subscribing to tokens: {tokens}")
            self.kws.subscribe(tokens)
            self.subscribed_tokens.update(tokens)
            logger.info(f"Successfully subscribed to tokens: {tokens}")
        except Exception as e:
            logger.error(f"Failed to subscribe to tokens {tokens}: {e}")
            logger.info("Historical data is still available via REST API endpoints.")
            # Don't raise the exception to prevent service disruption

    def unsubscribe(self, tokens: List[int]) -> None:
        """Unsubscribe from a list of instrument tokens."""
        if not isinstance(tokens, list):
            tokens = [tokens]
        self.kws.unsubscribe(tokens)
        self.subscribed_tokens.difference_update(tokens)
        logger.info(f"Unsubscribed from tokens: {tokens}")

    def set_mode(self, mode: str, tokens: List[int]) -> None:
        """Set the data mode for a list of tokens."""
        if not isinstance(tokens, list):
            tokens = [tokens]
        self.kws.set_mode(mode, tokens)
        logger.info(f"Set mode {mode} for tokens: {tokens}")

    def on_ticks(self, ws, ticks) -> None:
        """Handle incoming tick data with market hours optimization."""
        global publish_callback, system_ready, MAIN_EVENT_LOOP
        market_status = self._get_market_status()
        # logger.info(f"[ON_TICKS] Received {len(ticks)} ticks (Market: {market_status.value}, Continuous Flow: Enabled)")
        # logger.info(f"[ON_TICKS] Tick data: {ticks}")
        
        processed_ticks = []
        skipped_ticks = 0
        
        for tick in ticks:
            # Handle binary data from Zerodha WebSocket
            if isinstance(tick, bytes):
                # Parse binary message structure
                parsed_ticks = self.parse_binary_message(tick)
                processed_ticks.extend(parsed_ticks)
            else:
                # Handle JSON/dict format (fallback)
                processed_ticks.append(tick)
        
        # Process all ticks with optimization
        for tick in processed_ticks:
            if not tick or 'instrument_token' not in tick:
                logger.debug(f"Skipping invalid tick: {tick}")
                continue
                
            token = tick['instrument_token']
            
            # Check if we should process this tick
            should_process = self._should_process_tick(token, tick)
            if not should_process:
                skipped_ticks += 1
                logger.info(f"Skipping tick for token {token} (optimization rule) - price: {tick.get('last_price')}")
                continue
            # else:
            #     logger.info(f"Processing tick for token {token} - price: {tick.get('last_price')}")
            
            # Update last tick time
            self.last_tick_time[token] = time.time()
            
            # Store tick data
            if redis_client:
                redis_client.set(f"tick:{token}", str(tick))
            else:
                with tick_store_lock:
                    tick_store[token] = tick
            
            # Process for candle aggregation
            candle_aggregator.process_tick(tick)
            
            # Log based on market status
            if market_status == MarketStatus.OPEN:
                # logger.info(f"[LIVE] Processed tick for token: {token}")
                pass
            else:
                # logger.debug(f"[CONTINUOUS] Processed tick for token: {token} (market: {market_status.value})")
                pass
            
            # --- Forward tick to WebSocket clients ---
            tick_message = {
                'type': 'tick',
                'token': str(token),
                'price': tick.get('last_price'),
                'timestamp': tick.get('timestamp', time.time()),
                'volume_traded': tick.get('volume_traded'),
                'market_status': market_status.value,
                'data_freshness': 'real_time' if market_status == MarketStatus.OPEN else 'last_close',
                'continuous_flow': True,
                'market_always_open': True
            }
            
            # Publish tick with retry logic
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    if MAIN_EVENT_LOOP is not None and publish_callback is not None and system_ready:
                        MAIN_EVENT_LOOP.call_soon_threadsafe(asyncio.create_task, publish_callback(tick_message))
                        logger.debug(f"[TICK] Forwarded tick for token {token}: {tick_message['price']}")
                        break
                    else:
                        if MAIN_EVENT_LOOP is None:
                            logger.error("[ERROR] MAIN_EVENT_LOOP is not set! Tick publishing is disabled until FastAPI startup event.")
                        if publish_callback is None:
                            logger.error("[ERROR] publish_callback is not available! Tick publishing is disabled.")
                        if not system_ready:
                            logger.warning("[WARNING] System not ready for tick publishing. Skipping publish attempt.")
                        return
                except Exception as e:
                    logger.error(f"[ERROR] Failed to schedule tick publish (attempt {attempt}): {e}")
                    if attempt < max_retries:
                        backoff = 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
                        logger.info(f"[RETRY] Waiting {backoff:.2f}s before retrying tick publish...")
                        time.sleep(backoff)
                    else:
                        logger.error(f"[CRITICAL] All attempts to publish tick failed for token: {token}. Surfacing backend error to frontend.")
                        error_event = {
                            'type': 'backend_error',
                            'error': str(e),
                            'context': 'tick_publish',
                            'tick_message': tick_message,
                            'timestamp': time.time()
                        }
                        try:
                            if MAIN_EVENT_LOOP is not None and publish_callback is not None and system_ready:
                                MAIN_EVENT_LOOP.call_soon_threadsafe(asyncio.create_task, publish_callback(error_event))
                        except Exception as puberr:
                            logger.error(f"[CRITICAL] Failed to surface backend error to frontend: {puberr}")
        
        # Log optimization summary
        processed_count = len(processed_ticks) - skipped_ticks
        logger.info(f"[TICK SUMMARY] Processed {processed_count} ticks, skipped {skipped_ticks} ticks (market: {market_status.value}, continuous_flow: enabled)")
        
        # Call registered hooks with individual ticks
        for hook in self.tick_hooks:
            try:
                for tick in processed_ticks:
                    token = tick.get('instrument_token')
                    if token:
                        hook(token, tick)
            except Exception as e:
                logger.error(f"Tick hook error: {e}")

    def on_connect(self, ws, response) -> None:
        """Handle WebSocket connection event."""
        logger.info("WebSocket connected.")
        if self.subscribed_tokens:
            self.kws.subscribe(list(self.subscribed_tokens))

    def on_close(self, ws, code, reason) -> None:
        """Handle WebSocket close event."""
        logger.warning(f"WebSocket closed: {code}, {reason}")
        self.running = False

    def on_error(self, ws, code, reason) -> None:
        """Handle WebSocket error event."""
        logger.error(f"WebSocket error: {code}, {reason}")
        
        # If it's an authentication error (403), don't retry
        if code == 403:
            logger.error("Authentication failed. Please check your Zerodha credentials.")
            logger.error("Make sure ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN are properly set in .env file")
            logger.error("You may need to regenerate your access token if it has expired.")
            logger.info("Historical data is still available via REST API endpoints.")
            self.running = False
            return
        
        # For other errors, attempt to reconnect
        if self.running:
            logger.info("Attempting to reconnect after error...")
            time.sleep(5)  # Wait before reconnecting
            self.connect()

    def on_reconnect(self, ws, attempts_count) -> None:
        """Handle WebSocket reconnect event."""
        logger.info(f"WebSocket reconnecting, attempt {attempts_count}")
        
        # Resubscribe to tokens after reconnection
        if self.subscribed_tokens:
            try:
                self.kws.subscribe(list(self.subscribed_tokens))
                logger.info(f"Resubscribed to {len(self.subscribed_tokens)} tokens after reconnection")
            except Exception as e:
                logger.error(f"Error resubscribing after reconnection: {e}")

    def on_noreconnect(self, ws) -> None:
        """Handle event when WebSocket will not reconnect."""
        logger.error("WebSocket will NOT reconnect.")
        self.running = False

    def on_order_update(self, ws, data) -> None:
        """Handle order update event."""
        logger.info(f"Order update: {data}")

    def get_latest_tick(self, token: int) -> Optional[Dict[str, Any]]:
        """Get the latest tick for a token."""
        if redis_client:
            tick = redis_client.get(f"tick:{token}")
            return eval(tick) if tick else None
        with tick_store_lock:
            return tick_store.get(token)

    def get_latest_candle(self, token: int, tf: str) -> Optional[Dict[str, Any]]:
        """Get the latest candle for a token and timeframe."""
        return candle_aggregator.get_latest_candle(token, tf)
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status and optimization information."""
        return {
            "current_time": datetime.datetime.now().isoformat(),
            "timezone": "Asia/Kolkata",
            "market_status": self.market_status.value,
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
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        current_time = time.time()
        active_tokens = len(self.last_tick_time)
        
        # Calculate tokens with recent activity
        recent_threshold = 300  # 5 minutes
        recent_tokens = sum(1 for last_time in self.last_tick_time.values() 
                          if current_time - last_time < recent_threshold)
        
        return {
            "active_tokens": active_tokens,
            "recent_tokens": recent_tokens,
            "market_status": self._get_market_status().value,
            "optimization_enabled": True,
            "duplicate_threshold_seconds": self.duplicate_tick_threshold
        }

    def disconnectWebSocket(self) -> None:
        """Disconnect the WebSocket client."""
        if self.running:
            self.running = False
            if self._ws_thread and self._ws_thread.is_alive():
                self.kws.close()
                self._ws_thread.join(timeout=5)

# Singleton instance for use in backend
zerodha_ws_client = ZerodhaWSClient(API_KEY, ACCESS_TOKEN) 