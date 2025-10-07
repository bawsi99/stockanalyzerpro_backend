"""
API Key Manager for LLM Providers

This module provides a universal API key management system that works with any LLM provider.
Supports multiple key assignment strategies and thread-safe rotation.

Ported and enhanced from backend/gemini.archive/api_key_manager.py
"""

import os
import threading
from typing import Dict, List, Optional, Tuple
from enum import Enum


class KeyStrategy(Enum):
    """API key assignment strategies."""
    ROUND_ROBIN = "round_robin"     # Rotate through keys globally
    AGENT_SPECIFIC = "agent_specific"  # Each agent gets specific key
    SINGLE = "single"               # All agents use same key


class APIKeyManager:
    """
    Universal API key manager for all LLM providers.
    
    Features:
    - Multiple key assignment strategies
    - Thread-safe rotation
    - Provider-agnostic design
    - Detailed logging with key masking
    - Load from environment variables
    """
    
    _instances: Dict[str, 'APIKeyManager'] = {}
    _lock = threading.Lock()
    
    def __new__(cls, provider: str):
        """Singleton per provider implementation."""
        if provider not in cls._instances:
            with cls._lock:
                if provider not in cls._instances:
                    cls._instances[provider] = super().__new__(cls)
        return cls._instances[provider]
    
    def __init__(self, provider: str):
        """
        Initialize API key manager for a specific provider.
        
        Args:
            provider: Provider name (e.g., "gemini", "openai", "claude")
        """
        # Only initialize once per provider
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.provider = provider.lower()
        self._keys: List[str] = []
        self._current_index = 0
        self._index_lock = threading.Lock()
        
        # Load API keys for this provider
        self._load_keys()
        
        print(f"🔑 APIKeyManager initialized for {self.provider}: {len(self._keys)} key(s) available")
    
    def _load_keys(self):
        """Load API keys from environment variables based on provider."""
        self._keys = []
        
        # Provider-specific environment variable patterns
        if self.provider == "gemini":
            key_patterns = ["GEMINI_API_KEY", "GOOGLE_GEMINI_API_KEY"]
            numbered_pattern = "GEMINI_API_KEY"
        elif self.provider == "openai":
            key_patterns = ["OPENAI_API_KEY"]
            numbered_pattern = "OPENAI_API_KEY"
        elif self.provider == "claude":
            key_patterns = ["CLAUDE_API_KEY", "ANTHROPIC_API_KEY"]
            numbered_pattern = "CLAUDE_API_KEY"
        else:
            # Generic pattern for unknown providers
            key_patterns = [f"{self.provider.upper()}_API_KEY"]
            numbered_pattern = f"{self.provider.upper()}_API_KEY"
        
        # Try to load numbered keys first (for rotation)
        for i in range(1, 6):  # Support up to 5 keys per provider
            key = os.environ.get(f"{numbered_pattern}{i}")
            if key and key.strip():
                self._keys.append(key.strip())
                print(f"✅ Loaded {numbered_pattern}{i} for {self.provider}")
        
        # If no numbered keys found, try fallback patterns
        if not self._keys:
            for pattern in key_patterns:
                key = os.environ.get(pattern)
                if key and key.strip():
                    self._keys.append(key.strip())
                    print(f"⚠️  Using fallback {pattern} for {self.provider} (no numbered keys found)")
                    break
        
        if not self._keys:
            raise ValueError(
                f"No API keys found for {self.provider}. Please set {numbered_pattern}1-5 "
                f"or one of {key_patterns} environment variables."
            )
    
    def get_key(self, 
                agent_name: str, 
                strategy: KeyStrategy = KeyStrategy.ROUND_ROBIN,
                key_index: Optional[int] = None) -> Tuple[str, str]:
        """
        Get API key for an agent based on strategy.
        
        Args:
            agent_name: Name of the requesting agent
            strategy: Key assignment strategy
            key_index: Specific key index (only used with AGENT_SPECIFIC strategy)
            
        Returns:
            Tuple of (api_key, debug_info)
        """
        if not self._keys:
            raise ValueError(f"No API keys available for {self.provider}")
        
        if strategy == KeyStrategy.SINGLE:
            return self._get_single_key(agent_name)
        elif strategy == KeyStrategy.AGENT_SPECIFIC:
            return self._get_agent_specific_key(agent_name, key_index)
        elif strategy == KeyStrategy.ROUND_ROBIN:
            return self._get_round_robin_key(agent_name)
        else:
            raise ValueError(f"Unknown key strategy: {strategy}")
    
    def _get_single_key(self, agent_name: str) -> Tuple[str, str]:
        """Get single key (always the first one)."""
        key = self._keys[0]
        key_hint = self._mask_key(key)
        debug_info = f"[{agent_name}] Using SINGLE key for {self.provider}: ...{key_hint}"
        
        print(f"🔑 {debug_info}")
        return key, debug_info
    
    def _get_agent_specific_key(self, agent_name: str, key_index: Optional[int]) -> Tuple[str, str]:
        """Get agent-specific key by index."""
        if key_index is None:
            # Fall back to hash-based assignment if no index specified
            key_index = hash(agent_name) % len(self._keys)
        
        # Ensure key_index is within bounds
        key_index = key_index % len(self._keys)
        
        key = self._keys[key_index]
        key_hint = self._mask_key(key)
        key_num = key_index + 1
        
        debug_info = f"[{agent_name}] Using AGENT_SPECIFIC key #{key_num} for {self.provider}: ...{key_hint}"
        
        print(f"🔑 {debug_info}")
        return key, debug_info
    
    def _get_round_robin_key(self, agent_name: str) -> Tuple[str, str]:
        """Get key using round-robin rotation."""
        if len(self._keys) == 1:
            return self._get_single_key(agent_name)
        
        with self._index_lock:
            key = self._keys[self._current_index]
            key_index = self._current_index
            key_num = self._current_index + 1
            key_hint = self._mask_key(key)
            
            # Advance to next key
            self._current_index = (self._current_index + 1) % len(self._keys)
            
            debug_info = f"[{agent_name}] Using ROUND_ROBIN key #{key_num}/{len(self._keys)} for {self.provider}: ...{key_hint}"
            
            print(f"🔑 {debug_info}")
            return key, debug_info
    
    def _mask_key(self, key: str) -> str:
        """Mask API key for logging (show only last 8 characters)."""
        if len(key) <= 8:
            return "*" * len(key)
        return key[-8:]
    
    def get_key_count(self) -> int:
        """Get number of available keys."""
        return len(self._keys)
    
    def get_all_keys_masked(self) -> List[str]:
        """Get all keys with masking for debugging."""
        return [f"...{self._mask_key(key)}" for key in self._keys]
    
    def validate_keys(self) -> bool:
        """Validate that all keys are non-empty."""
        for i, key in enumerate(self._keys):
            if not key or not key.strip():
                print(f"❌ Invalid key at index {i} for {self.provider}")
                return False
        return True


class MultiProviderKeyManager:
    """
    Manager for multiple provider key managers.
    
    This provides a single interface to manage keys across all providers.
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
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self._managers: Dict[str, APIKeyManager] = {}
        
        print("🔐 MultiProviderKeyManager initialized")
    
    def get_manager(self, provider: str) -> APIKeyManager:
        """Get or create key manager for a provider."""
        provider = provider.lower()
        
        if provider not in self._managers:
            self._managers[provider] = APIKeyManager(provider)
        
        return self._managers[provider]
    
    def get_key_for_agent(self,
                         provider: str,
                         agent_name: str,
                         strategy: KeyStrategy = KeyStrategy.ROUND_ROBIN,
                         key_index: Optional[int] = None) -> Tuple[str, str]:
        """
        Get API key for an agent across any provider.
        
        Args:
            provider: Provider name
            agent_name: Agent name
            strategy: Key assignment strategy
            key_index: Specific key index (for agent_specific strategy)
            
        Returns:
            Tuple of (api_key, debug_info)
        """
        manager = self.get_manager(provider)
        return manager.get_key(agent_name, strategy, key_index)
    
    def print_status(self):
        """Print status of all provider key managers."""
        print("\n🔐 Multi-Provider Key Manager Status")
        print("=" * 50)
        
        if not self._managers:
            print("No providers initialized yet.")
            return
        
        for provider, manager in self._managers.items():
            print(f"\n{provider.upper()}:")
            print(f"  Keys available: {manager.get_key_count()}")
            print(f"  Masked keys: {', '.join(manager.get_all_keys_masked())}")


# Global singleton instance
def get_key_manager() -> MultiProviderKeyManager:
    """Get the global multi-provider key manager instance."""
    return MultiProviderKeyManager()


# Convenience functions
def get_key_for_agent(provider: str,
                     agent_name: str,
                     strategy: KeyStrategy = KeyStrategy.ROUND_ROBIN,
                     key_index: Optional[int] = None) -> Tuple[str, str]:
    """
    Convenience function to get API key for an agent.
    
    Args:
        provider: Provider name ("gemini", "openai", "claude")
        agent_name: Agent name for logging
        strategy: Key assignment strategy
        key_index: Specific key index (for agent_specific)
        
    Returns:
        Tuple of (api_key, debug_info)
    """
    manager = get_key_manager()
    return manager.get_key_for_agent(provider, agent_name, strategy, key_index)


if __name__ == "__main__":
    # Test the key manager
    print("🧪 Testing API Key Manager")
    print("=" * 40)
    
    try:
        # Test Gemini provider
        manager = get_key_manager()
        
        # Test different strategies
        print("\n1. Testing Round Robin:")
        for i in range(3):
            key, info = get_key_for_agent("gemini", f"test_agent_{i}", KeyStrategy.ROUND_ROBIN)
            
        print("\n2. Testing Agent Specific:")
        key, info = get_key_for_agent("gemini", "volume_agent", KeyStrategy.AGENT_SPECIFIC, 0)
        key, info = get_key_for_agent("gemini", "indicator_agent", KeyStrategy.AGENT_SPECIFIC, 1)
        
        print("\n3. Testing Single:")
        key, info = get_key_for_agent("gemini", "final_decision", KeyStrategy.SINGLE)
        
        # Print status
        manager.print_status()
        
        print("\n✅ Key manager tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()