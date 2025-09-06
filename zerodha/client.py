import os
import pandas as pd
from datetime import datetime, timedelta, time as dt_time
import logging
from typing import Dict, Optional, Union, Tuple
from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException, NetworkException
import webbrowser
import dotenv
import json
import hashlib
from pathlib import Path
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import tempfile
from threading import RLock


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ZerodhaClient')


# Load environment variables from .env file (for initial load)
dotenv.load_dotenv("../config/.env")

# Utility function to always read the latest value from .env
def get_env_value(key: str, env_path: str = "../config/.env") -> str:
    """Read a value from the .env file directly, with fallback to system environment."""
    # First try to read from .env file if it exists
    if os.path.exists(env_path):
        try:
            with open(env_path, "r") as f:
                for line in f:
                    if line.strip().startswith(f"{key}="):
                        return line.strip().split("=", 1)[1]
        except Exception as e:
            logger.warning(f"Error reading from .env file: {e}")
    
    # Fallback to system environment variables (for production deployments)
    system_value = os.getenv(key, "")
    if system_value:
        logger.info(f"Using system environment variable for {key}")
    return system_value

class CacheManager:
    """Manages caching of stock data with expiration and invalidation policies."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._lock = RLock()
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to file."""
        try:
            with self._lock:
                # Take a snapshot to avoid 'dictionary changed size during iteration'
                metadata_copy = dict(self.metadata)
                # Write atomically via temp file and replace
                with tempfile.NamedTemporaryFile('w', delete=False, dir=str(self.cache_dir)) as tmp_file:
                    json.dump(metadata_copy, tmp_file)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())
                    temp_path = tmp_file.name
                os.replace(temp_path, str(self.metadata_file))
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def _generate_cache_key(self, symbol: str, exchange: str, interval: str, 
                          from_date: datetime, to_date: datetime) -> str:
        """Generate a unique cache key for the data request."""
        key_str = f"{symbol}:{exchange}:{interval}:{from_date.strftime('%Y%m%d')}:{to_date.strftime('%Y%m%d')}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cached_data(self, symbol: str, exchange: str, interval: str,
                       from_date: datetime, to_date: datetime) -> Optional[pd.DataFrame]:
        """Get cached data if available and valid."""
        if not self._is_market_closed():
            return None
            
        cache_key = self._generate_cache_key(symbol, exchange, interval, from_date, to_date)
        cache_file = self.cache_dir / f"{cache_key}.csv"
        
        with self._lock:
            key_present = cache_key in self.metadata
        if key_present and cache_file.exists():
            try:
                data = pd.read_csv(cache_file, parse_dates=['date'])
                data.set_index('date', inplace=True)
                return data
            except Exception as e:
                logger.error(f"Error reading cached data: {e}")
                return None
        return None
    
    def cache_data(self, data: pd.DataFrame, symbol: str, exchange: str, interval: str,
                  from_date: datetime, to_date: datetime):
        """Cache the data with metadata."""
        if not self._is_market_closed():
            return
            
        cache_key = self._generate_cache_key(symbol, exchange, interval, from_date, to_date)
        cache_file = self.cache_dir / f"{cache_key}.csv"
        
        try:
            data.to_csv(cache_file)
            with self._lock:
                self.metadata[cache_key] = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'interval': interval,
                    'from_date': from_date.isoformat(),
                    'to_date': to_date.isoformat(),
                    'cached_at': datetime.now().isoformat()
                }
                self._save_metadata()
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    def _is_market_closed(self) -> bool:
        """Check if the market is currently closed (after 3:30 PM IST or before 9:15 AM IST)."""
        now = datetime.now()
        ist_time = now + timedelta(hours=5, minutes=30)  # Convert to IST
        market_open = dt_time(9, 15)  # 9:15 AM IST
        market_close = dt_time(15, 30)  # 3:30 PM IST
        
        return ist_time.time() < market_open or ist_time.time() > market_close

def auto_refresh_token(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except TokenException as e:
            logger.warning(f"Token expired or invalid in {func.__name__}: {e}. Attempting to re-authenticate...")
            if self.authenticate():
                try:
                    return func(self, *args, **kwargs)
                except Exception as e2:
                    logger.error(f"Failed after re-authentication in {func.__name__}: {e2}")
                    return None
            else:
                logger.error(f"Re-authentication failed in {func.__name__}.")
                return None
    return wrapper

class ZerodhaDataClient:
    """
    Client for fetching data from Zerodha API with async support.
    Implements singleton pattern to ensure only one instance exists.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern to prevent multiple sessions."""
        if cls._instance is None:
            cls._instance = super(ZerodhaDataClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, api_key: str = None, api_secret: str = None, access_token: str = None):
        """
        Initialize the Zerodha client with authentication credentials.
        Always read the latest access and request tokens from .env.
        """
        # Skip initialization if already initialized (singleton pattern)
        if hasattr(self, '_initialized') and self._initialized:
            logger.info("ZerodhaDataClient already initialized - reusing instance")
            return
            
        self.api_key = api_key or get_env_value("ZERODHA_API_KEY")
        self.api_secret = api_secret or get_env_value("ZERODHA_API_SECRET")
        self.access_token = access_token or get_env_value("ZERODHA_ACCESS_TOKEN")
        self.request_token = get_env_value("ZERODHA_REQUEST_TOKEN")
        
        # Mark as initialized
        self._initialized = True
        logger.info("ZerodhaDataClient initialized with new session")

        # Initialize KiteConnect client
        self.kite = KiteConnect(api_key=self.api_key)

        # Set access token if provided
        if self.access_token:
            self.kite.set_access_token(self.access_token)
            logger.info("Session initialized with existing access token")

        # Initialize cache manager
        self.cache_manager = CacheManager()

        # Rate limiting
        self.last_request_time = datetime.now()
        self.min_request_interval = timedelta(seconds=0)  # Minimum time between requests

        # Cache for instruments and data
        self.instruments_cache = {}
        self.data_cache = {}
        self.all_instruments = None
        
        self._executor = ThreadPoolExecutor(max_workers=10)  # For running sync methods in async context

        # In-memory LRU cache for historical data
        from collections import OrderedDict
        from threading import RLock
        self._historical_lru: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        self._historical_lru_capacity: int = int(os.environ.get("ZERODHA_LRU_CAPACITY", "128"))
        self._historical_lru_lock = RLock()

    def _normalize_history_key(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> str:
        # Normalize dates to YYYYMMDD to maximize hits irrespective of time components
        fd = from_date.strftime("%Y%m%d")
        td = to_date.strftime("%Y%m%d")
        return f"{exchange}:{symbol}:{interval}:{fd}:{td}"

    def _lru_get(self, key: str) -> Optional[pd.DataFrame]:
        with self._historical_lru_lock:
            df = self._historical_lru.get(key)
            if df is not None:
                # Move to recent
                self._historical_lru.move_to_end(key)
            return df

    def _lru_put(self, key: str, df: pd.DataFrame) -> None:
        with self._historical_lru_lock:
            self._historical_lru[key] = df
            self._historical_lru.move_to_end(key)
            # Evict oldest if over capacity
            while len(self._historical_lru) > self._historical_lru_capacity:
                self._historical_lru.popitem(last=False)
        
    def _save_access_token(self, access_token: str):
        """Save the access token to the .env file, replacing the old value if present."""
        env_path = "../config/.env"
        lines = []
        found = False
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.strip().startswith("ZERODHA_ACCESS_TOKEN="):
                        lines.append(f"ZERODHA_ACCESS_TOKEN={access_token}\n")
                        found = True
                    else:
                        lines.append(line)
        if not found:
            lines.append(f"ZERODHA_ACCESS_TOKEN={access_token}\n")
        with open(env_path, "w") as f:
            f.writelines(lines)
        logger.info("Access token saved to .env successfully.")

    def _save_request_token(self, request_token: str) -> None:
        """
        Save the request token to .env file immediately after user input.
        
        Args:
            request_token: The request token obtained from the user
        """
        env_path = "../config/.env"
        try:
            # Create .env file if it doesn't exist
            if not os.path.exists(env_path):
                with open(env_path, "w") as f:
                    f.write(f"ZERODHA_REQUEST_TOKEN={request_token}\n")
                logger.info("Created new .env file with request token")
                return

            # Read existing content
            with open(env_path, "r") as f:
                lines = f.readlines()

            # Check if request token already exists
            request_token_exists = False
            for i, line in enumerate(lines):
                if line.strip().startswith("ZERODHA_REQUEST_TOKEN="):
                    lines[i] = f"ZERODHA_REQUEST_TOKEN={request_token}\n"
                    request_token_exists = True
                    break

            # If request token doesn't exist, append it
            if not request_token_exists:
                lines.append(f"ZERODHA_REQUEST_TOKEN={request_token}\n")

            # Write back to file
            with open(env_path, "w") as f:
                f.writelines(lines)

            logger.info("Request token saved successfully")
            print("Request token saved successfully")

        except PermissionError:
            logger.error("Permission denied: Cannot write to .env file")
            print("Error: Cannot write to .env file. Please check file permissions.")
        except Exception as e:
            logger.error(f"Error saving request token: {str(e)}")
            print(f"Warning: Could not save request token: {str(e)}")

    def authenticate(self) -> bool:
        """
        Authenticate with Zerodha API and obtain access token.
        Returns True if authentication successful, False otherwise.
        """
        if not self.api_key or not self.api_secret:
            logger.error("API key or secret not provided")
            print("Error: API key and secret are required for authentication")
            print("Please set ZERODHA_API_KEY and ZERODHA_API_SECRET environment variables or provide them during initialization")
            return False

        # Always read the latest access token from .env before authenticating
        self.access_token = get_env_value("ZERODHA_ACCESS_TOKEN")
        if self.access_token:
            try:
                self.kite.set_access_token(self.access_token)
                self.kite.profile()  # simple call to check validity
                print("Access token is still valid.")
                logger.info("Authentication successful with existing access token")
                return True
            except TokenException as e:
                # Only treat explicit token errors as invalidation events
                print("Stored access token is invalid or expired. Will attempt re-authentication.")
                logger.warning("Stored access token invalid: " + str(e))
                # Remove invalid access_token from .env
                env_path = "../config/.env"
                try:
                    if os.path.exists(env_path):
                        with open(env_path, "r") as f:
                            lines = f.readlines()
                        with open(env_path, "w") as f:
                            for line in lines:
                                if not line.strip().startswith("ZERODHA_ACCESS_TOKEN="):
                                    f.write(line)
                except Exception as env_err:
                    logger.error("Failed to remove invalid token from .env: " + str(env_err))
                    print("Error while updating .env")
                self.access_token = None
            except NetworkException as e:
                # Network blips should not invalidate a perfectly good token
                logger.warning(
                    "Network error while validating existing access token; assuming token remains valid: "
                    + str(e)
                )
                return True
            except Exception as e:
                # Any other non-token error should not nuke the token; continue optimistically
                logger.warning(
                    "Non-token error during access token validation; keeping token and proceeding: "
                    + str(e)
                )
                return True

        # Always read the latest request token from .env
        self.request_token = get_env_value("ZERODHA_REQUEST_TOKEN")
        if self.request_token:
            try:
                data = self.kite.generate_session(self.request_token, api_secret=self.api_secret)
                self.access_token = data["access_token"]
                self.kite.set_access_token(self.access_token)
                self._save_access_token(self.access_token)
                print("Authentication successful!")
                logger.info("Authentication successful, new access token obtained")
                return True
            except TokenException as e:
                logger.error(f"Stored request token invalid or expired: {str(e)}")
                print("Stored request token is invalid. Will need to re-authenticate.")
            except NetworkException as e:
                # Do not treat network errors as invalid request tokens
                logger.warning(
                    "Network error while exchanging stored request token; not invalidating it and will retry later: "
                    + str(e)
                )
                return False
            except Exception as e:
                # Unknown errors - do not proceed with forced re-auth here
                logger.warning(
                    "Unexpected error while exchanging stored request token; not invalidating it: "
                    + str(e)
                )
                return False

        # If no valid tokens, guide user to obtain new request token
        login_url = self.kite.login_url()
        print("\nOpening Zerodha login URL in your browser...")
        webbrowser.open(login_url)
        print("-----------------------------------")
        print("1. Complete login in your browser.")
        print("2. You will be redirected to your registered redirect URL.")
        print("3. Copy the `request_token` from the redirected URL.")
        print("4. Paste it below.")
        print("-----------------------------------\n")
        request_token = input("Paste the request_token here: ").strip()

        # Save request token immediately after user input
        self._save_request_token(request_token)

        # Try generating access token from request token
        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            self._save_access_token(self.access_token)
            print("Authentication successful!")
            logger.info("Authentication successful, new access token obtained")
            return True

        except TokenException as e:
            logger.error(f"Token error during authentication: {str(e)}")
            print(f"Token error during authentication: {str(e)}")
            return False
        except NetworkException as e:
            logger.error(f"Network error during authentication: {str(e)}")
            print(f"Network error during authentication: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error during authentication: {str(e)}")
            print(f"Error during authentication: {str(e)}")
            return False

    def _load_all_instruments(self, exchange: str = None, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Load all instruments into memory, using CSV cache if valid. 
        Always fetch new data if not fetched after today's 8:30 AM.
        """
        import pandas as pd
        from datetime import datetime, timedelta, time as dt_time
        import os
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', '.env'))
        now = datetime.now()
        today_830am = datetime.combine(now.date(), dt_time(8, 30))
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', '.env')
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'zerodha_instruments.csv')

        # Read last fetch timestamp from .env
        last_fetch_str = os.getenv("TIME_LAST_FETCH_INSTRUMENT")
        last_fetch_time = None
        if last_fetch_str:
            try:
                last_fetch_time = datetime.fromisoformat(last_fetch_str)
            except ValueError:
                logger.warning("Invalid TIME_LAST_FETCH_INSTRUMENT format in .env")

        use_cache = False
        if last_fetch_time and not force_refresh:
            # Case 1: Before 8:30 AM today and < 24 hours
            if now < today_830am and now - last_fetch_time < timedelta(hours=24):
                use_cache = True
            # Case 2: After 8:30 AM today but last fetch was after 8:30 AM
            elif now >= today_830am and last_fetch_time >= today_830am:
                use_cache = True

        if use_cache:
            try:
                df = pd.read_csv(csv_path)
                if exchange:
                    return df[df['exchange'] == exchange]
                return df
            except Exception as e:
                logger.warning(f"Failed to load cached instruments CSV: {e}. Falling back to API.")

        try:
            # Fetch from Zerodha API
            instruments = self.kite.instruments(exchange=exchange)
            df = pd.DataFrame(instruments)

            # Save new CSV
            if os.path.exists(csv_path):
                os.remove(csv_path)
            df.to_csv(csv_path, index=False)

            # Update .env with new fetch time
            new_env_lines = []
            found = False
            if os.path.exists(env_path):
                with open(env_path, "r") as f:
                    for line in f:
                        if line.startswith("TIME_LAST_FETCH_INSTRUMENT="):
                            new_env_lines.append(f"TIME_LAST_FETCH_INSTRUMENT={now.isoformat()}\n")
                            found = True
                        else:
                            new_env_lines.append(line)
            if not found:
                new_env_lines.append(f"\nTIME_LAST_FETCH_INSTRUMENT={now.isoformat()}\n")

            with open(env_path, "w") as f:
                f.writelines(new_env_lines)

            return df[df['exchange'] == exchange] if exchange else df

        except TokenException as e:
            logger.error(f"Token error getting instruments: {str(e)}")
            print("Your authentication token may have expired. Please re-authenticate.")
            return None
        except NetworkException as e:
            logger.error(f"Network error getting instruments: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error getting instruments: {str(e)}")
            return None
    
    @auto_refresh_token
    def get_instruments(self, exchange: str = None) -> Optional[pd.DataFrame]:
        """
        Get all instruments available for trading.
        
        Args:
            exchange: Optional filter by exchange (e.g., "NSE", "BSE")
            
        Returns:
            pd.DataFrame: DataFrame with instruments or None if error
        """
        return self._load_all_instruments(exchange)
    
    @auto_refresh_token
    def get_instrument_token(self, symbol: str, exchange: str = "NSE") -> Optional[int]:
        """
        Get the instrument token for a given symbol.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            exchange: Exchange code (default: "NSE")
            
        Returns:
            int: Instrument token if found, None otherwise
        """
        try:
            # Load all instruments if not already loaded
            instruments = self._load_all_instruments(exchange)

            if instruments is None:
                logger.error("Failed to get instruments")
                return None
            
            # Find the instrument with matching symbol and exchange
            instrument = instruments[(instruments['tradingsymbol'] == symbol) & 
                                    (instruments['exchange'] == exchange)]
            
            if len(instrument) == 0:
                logger.warning(f"Instrument not found: {exchange}:{symbol}")
                return None
            
            # Get the instrument token
            token = instrument.iloc[0]['instrument_token']
            return token
            
        except Exception as e:
            logger.error(f"Error getting instrument token: {str(e)}")
            return None

    async def get_instrument_token_async(self, symbol: str, exchange: str = "NSE") -> Optional[int]:
        """
        Async version of get_instrument_token.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.get_instrument_token,
            symbol,
            exchange
        )
    
    @auto_refresh_token
    def get_symbol_from_token(self, token: int, exchange: str = "NSE") -> Optional[str]:
        """
        Get the symbol for a given instrument token.
        
        Args:
            token: Instrument token
            exchange: Exchange code (default: "NSE")
            
        Returns:
            str: Trading symbol if found, None otherwise
        """
        try:
            # Load all instruments if not already loaded
            instruments = self._load_all_instruments(exchange)

            if instruments is None:
                logger.error("Failed to get instruments")
                return None
            
            # Find the instrument with matching token and exchange
            instrument = instruments[(instruments['instrument_token'] == token) & 
                                    (instruments['exchange'] == exchange)]
            
            if len(instrument) == 0:
                logger.warning(f"Instrument not found for token: {token} on {exchange}")
                return None
            
            # Get the trading symbol
            symbol = instrument.iloc[0]['tradingsymbol']
            return symbol
            
        except Exception as e:
            logger.error(f"Error getting symbol from token: {str(e)}")
            return None
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting between API calls."""
        now = datetime.now()
        time_since_last_request = now - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = (self.min_request_interval - time_since_last_request).total_seconds()
            time.sleep(sleep_time)
        self.last_request_time = datetime.now()

    @auto_refresh_token
    def get_historical_data(
        self, 
        symbol: str, 
        exchange: str = "NSE",
        interval: str = "day", 
        from_date: Union[str, datetime] = None,
        to_date: Union[str, datetime] = None,
        period: int = 365,
        continuous: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data for a given symbol from Zerodha.
        Supports 'week' and 'month' intervals by aggregating daily data.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            exchange: Exchange code (default: "NSE")
            interval: Candle interval (minute, day, 3minute, 5minute, 10minute, 15minute, 30minute, 60minute, week, month)
            from_date: Start date (default: 1 year ago)
            to_date: End date (default: today)
            continuous: Whether to adjust for splits and bonuses
            
        Returns:
            pd.DataFrame: DataFrame with historical data or None if error
        """
        # Set default dates if not provided
        if to_date is None:
            to_date = datetime.now()
        elif isinstance(to_date, str):
            to_date = datetime.strptime(to_date, "%Y-%m-%d")

        if from_date is None:
            from_date = to_date - timedelta(days=period)
        elif isinstance(from_date, str):
            from_date = datetime.strptime(from_date, "%Y-%m-%d")

        # Support for 'week' and 'month' intervals by aggregating daily data
        aggregate_week = interval.lower() in ["week", "weekly"]
        aggregate_month = interval.lower() in ["month", "monthly"]
        fetch_interval = "day" if (aggregate_week or aggregate_month) else interval

        # Get instrument token
        instrument_token = self.get_instrument_token(symbol, exchange)
        if not instrument_token:
            logger.error(f"Instrument token not found for {exchange}:{symbol}")
            return None

        try:
            # Build normalized key for caches (keyed by requested interval, not fetch_interval)
            cache_key = self._normalize_history_key(symbol, exchange, interval, from_date, to_date)

            # 1) In-memory LRU cache
            cached_df = self._lru_get(cache_key)
            if cached_df is not None:
                logger.info(f"Cache hit (LRU) for {exchange}:{symbol} {interval} {from_date.date()}->{to_date.date()}")
                return cached_df

            # 2) Disk cache when market is closed
            disk_cached_df = self.cache_manager.get_cached_data(symbol, exchange, interval, from_date, to_date)
            if disk_cached_df is not None:
                logger.info(f"Cache hit (disk) for {exchange}:{symbol} {interval} {from_date.date()}->{to_date.date()}")
                # Place into LRU for faster subsequent access
                self._lru_put(cache_key, disk_cached_df)
                return disk_cached_df

            logger.info(f"Fetching historical data for {symbol} from {from_date} to {to_date} (interval: {interval})")

            # Implement rate limiting
            self._wait_for_rate_limit()

            # Fetch historical data (daily if aggregating)
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=fetch_interval,
                continuous=False
            )

            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # If not aggregating, return as is
            if not (aggregate_week or aggregate_month):
                logger.info(f"Retrieved {len(df)} records for {symbol}")
                # Save to caches
                self._lru_put(cache_key, df)
                # Persist to disk cache only when market is closed
                self.cache_manager.cache_data(df, symbol, exchange, interval, from_date, to_date)
                return df

            # --- Aggregation for week/month ---
            # Ensure 'date' is datetime and set as index
            if 'date' not in df.columns:
                logger.error("No 'date' column in returned data for aggregation.")
                return None
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            rule = 'W' if aggregate_week else 'M'
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            # Only aggregate columns that exist
            agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
            resampled = df.resample(rule).agg(agg_dict).dropna()
            resampled.reset_index(inplace=True)
            logger.info(f"Aggregated {len(resampled)} {interval} records for {symbol}")
            # Save to caches
            self._lru_put(cache_key, resampled)
            self.cache_manager.cache_data(resampled, symbol, exchange, interval, from_date, to_date)
            return resampled

        except TokenException as e:
            logger.error(f"Token error getting historical data: {str(e)}")
            return None
        except NetworkException as e:
            logger.error(f"Network error getting historical data: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None
    
    async def get_historical_data_async(
        self, 
        symbol: str, 
        exchange: str = "NSE",
        interval: str = "day", 
        from_date: Union[str, datetime] = None,
        to_date: Union[str, datetime] = None,
        period: int = 365,
        continuous: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Async version of get_historical_data for index data fetching.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.get_historical_data,
            symbol,
            exchange,
            interval,
            from_date,
            to_date,
            period,
            continuous
        )

    @auto_refresh_token
    def get_quote(self, symbol: str, exchange: str = "NSE") -> Optional[Dict]:
        """
        Get current market quote for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            exchange: Exchange code (default: "NSE")
            
        Returns:
            dict: Quote data or None if error
        """
        try:
            logger.info(f"Fetching current quote for {symbol}")
            print(f"Fetching current quote for {symbol}")
            
            # Format the instrument for Kite
            instrument = f"{exchange}:{symbol}"
            
            # Get quote
            quotes = self.kite.quote([instrument])
            
            if instrument in quotes:
                return quotes[instrument]
            
            logger.error(f"Quote not found for {instrument}")
            return None
            
        except TokenException as e:
            logger.error(f"Token error getting quote: {str(e)}")
            print(f"Token error getting quote: {str(e)}")
            print("Your authentication token may have expired. Please re-authenticate.")
            return None
        except NetworkException as e:
            logger.error(f"Network error getting quote: {str(e)}")
            print(f"Network error getting quote: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error getting quote: {str(e)}")
            print(f"Error getting quote: {str(e)}")
            return None
    
    async def get_quote_async(self, symbol: str, exchange: str = "NSE") -> Optional[Dict]:
        """
        Async version of get_quote.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.get_quote,
            symbol,
            exchange
        )

    @auto_refresh_token
    def get_market_status(self) -> Optional[Dict]:
        """
        Infer current market status based on time and quotes.
        
        Returns:
            dict: Market status information or None if error
        """
        try:
            logger.info("Inferring market status")
            
            # Get current time in IST (UTC+5:30)
            now = datetime.now() + timedelta(hours=5, minutes=30)
            
            # Check if it's a weekday
            is_weekday = now.weekday() < 5  # 0-4 are Monday to Friday
            
            # Check if it's market hours (9:15 AM to 3:30 PM IST)
            is_market_hour = False
            if (9 < now.hour < 15) or (now.hour == 9 and now.minute >= 15) or (now.hour == 15 and now.minute <= 30):
                is_market_hour = True
            
            # Basic status based on time
            status = {
                "timestamp": datetime.now().isoformat(),
                "inferred_status": "closed",
                "exchanges": {
                    "NSE": {
                        "status": "closed",
                        "segment_status": {
                            "equity": "closed",
                            "futures_&_options": "closed"
                        }
                    },
                    "BSE": {
                        "status": "closed",
                        "segment_status": {
                            "equity": "closed"
                        }
                    }
                },
                "note": "Status inferred from time. For accurate status, check official Zerodha/exchange sources."
            }
            
            # Update status based on time
            if is_weekday and is_market_hour:
                status["inferred_status"] = "open"
                status["exchanges"]["NSE"]["status"] = "open"
                status["exchanges"]["NSE"]["segment_status"]["equity"] = "open"
                status["exchanges"]["NSE"]["segment_status"]["futures_&_options"] = "open"
                status["exchanges"]["BSE"]["status"] = "open"
                status["exchanges"]["BSE"]["segment_status"]["equity"] = "open"
            
            # Try to get a quote for NIFTY 50 to verify if market is actually open
            try:
                nifty_quote = self.kite.quote(["NSE:NIFTY 50"])
                if "NSE:NIFTY 50" in nifty_quote:
                    last_trade_time = nifty_quote["NSE:NIFTY 50"].get("timestamp")
                    if last_trade_time:
                        # Convert to datetime if it's a string
                        if isinstance(last_trade_time, str):
                            last_trade_time = datetime.fromisoformat(last_trade_time.replace('Z', '+00:00'))
                        
                        # If last trade is within the last 5 minutes, market is likely open
                        time_diff = datetime.now() - last_trade_time
                        if time_diff.total_seconds() < 300:  # 5 minutes
                            status["verified_by_quote"] = True
                            status["last_trade_time"] = last_trade_time.isoformat()
                        else:
                            status["verified_by_quote"] = False
                            status["last_trade_time"] = last_trade_time.isoformat()
                            # If we're in market hours but no recent trades, market might be halted
                            if is_weekday and is_market_hour:
                                status["inferred_status"] = "halted"
                                status["exchanges"]["NSE"]["status"] = "halted"
                                status["exchanges"]["BSE"]["status"] = "halted"
            except Exception as e:
                logger.warning(f"Could not verify market status with quote: {str(e)}")
                status["verified_by_quote"] = False
            
            return status
            
        except Exception as e:
            logger.error(f"Error inferring market status: {str(e)}")
            print(f"Error inferring market status: {str(e)}")
            return None
    
    async def get_market_status_async(self) -> Optional[Dict]:
        """
        Async version of get_market_status.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.get_market_status
        )


# Example usage
if __name__ == "__main__":
    # Create client
    client = ZerodhaDataClient()

    # #Authenticate

    if client.authenticate():
        # Get historical data for a stock
        data = client.get_historical_data("RELIANCE")
        if data is not None:
            print(f"Retrieved {len(data)} days of data for RELIANCE")
            print(data.head())
        
        # Get current quote
        quote = client.get_quote("RELIANCE")
        if quote is not None:
            print(f"Current price of RELIANCE: {quote['last_price']}")
        
        # Get market status
        status = client.get_market_status()
        if status is not None:
            print(f"Market status: {status['inferred_status']}")
        
    else:
        print("Authentication failed or skipped. Please try again with valid credentials.")
