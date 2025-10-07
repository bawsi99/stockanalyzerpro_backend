"""
API Key Manager for Load Distribution
Manages multiple Gemini API keys and distributes load across them.
"""
import os
import threading
from typing import Optional

# Load environment variables from .env file
try:
    import dotenv
    # Load from backend/config/.env
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', '.env')
    if os.path.exists(env_path):
        dotenv.load_dotenv(dotenv_path=env_path)
        print(f"📁 Loaded .env from: {env_path}")
except ImportError:
    print("⚠️ python-dotenv not installed, reading from system environment only")

class APIKeyManager:
    """
    Manages multiple Gemini API keys and provides key rotation for load distribution.
    Thread-safe singleton implementation.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._keys = []
        self._current_index = 0
        self._index_lock = threading.Lock()
        
        # Load API keys from environment
        self._load_keys()
    
    def _load_keys(self):
        """Load API keys from environment variables."""
        # Try to load all 5 keys
        for i in range(1, 6):
            key = os.environ.get(f"GEMINI_API_KEY{i}")
            if key:
                self._keys.append(key)
                print(f"✅ Loaded GEMINI_API_KEY{i}")
        
        # Fallback to single GEMINI_API_KEY if numbered keys not found
        if not self._keys:
            fallback_key = os.environ.get("GEMINI_API_KEY")
            if fallback_key:
                self._keys.append(fallback_key)
                print("⚠️ Using fallback GEMINI_API_KEY (no numbered keys found)")
            else:
                raise ValueError(
                    "No Gemini API keys found. Please set GEMINI_API_KEY1-5 "
                    "or GEMINI_API_KEY environment variable."
                )
        
        print(f"🔑 API Key Manager initialized with {len(self._keys)} key(s)")
    
    def get_key(self, agent_name: Optional[str] = None) -> str:
        """
        Get the next API key in rotation.
        
        Args:
            agent_name: Optional agent name for logging purposes
            
        Returns:
            API key string
        """
        if not self._keys:
            raise ValueError("No API keys available")
        
        # If only one key, return it
        if len(self._keys) == 1:
            key = self._keys[0]
            key_hint = key[-8:] if key else "unknown"
            if agent_name:
                print(f"🔑 [{agent_name}] Using single API key ...{key_hint}")
            return key
        
        # Round-robin rotation
        with self._index_lock:
            key = self._keys[self._current_index]
            key_num = self._current_index + 1
            key_hint = key[-8:] if key else "unknown"
            self._current_index = (self._current_index + 1) % len(self._keys)
            
            if agent_name:
                print(f"🔑 [{agent_name}] Using API key #{key_num} (GEMINI_API_KEY{key_num}) ending in ...{key_hint}")
            
            return key
    
    def get_key_for_agent(self, agent_index: int) -> str:
        """
        Get a specific API key by index (for parallel agents).
        
        Args:
            agent_index: Index of the agent (0-4)
            
        Returns:
            API key string
        """
        if not self._keys:
            raise ValueError("No API keys available")
        
        # If we have fewer keys than agents, use modulo
        key_index = agent_index % len(self._keys)
        key = self._keys[key_index]
        key_hint = key[-8:] if key else "unknown"
        
        # Map agent names for better logging
        agent_names = [
            "volume_anomaly",
            "institutional_activity", 
            "volume_confirmation",
            "support_resistance",
            "volume_momentum"
        ]
        agent_name = agent_names[agent_index] if agent_index < len(agent_names) else f"agent_{agent_index}"
        
        print(f"🔑 [{agent_name}] Using API key #{key_index + 1} (GEMINI_API_KEY{key_index + 1}) ending in ...{key_hint}")
        return key
    
    def get_all_keys(self):
        """Get all available keys (for testing)."""
        return self._keys.copy()
    
    def get_key_count(self) -> int:
        """Get the number of available keys."""
        return len(self._keys)


# Global singleton instance
_api_key_manager = None

def get_api_key_manager() -> APIKeyManager:
    """Get the global API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager