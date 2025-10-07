"""
Configuration module for Gemini debugging.
This allows easy enabling/disabling of debug logging.
"""

import os
from typing import Optional

class GeminiDebugConfig:
    """Configuration class for Gemini debugging"""
    
    def __init__(self):
        # Changed defaults to reduce debug output but show time and tokens
        self._enable_debug = self._get_env_bool("GEMINI_DEBUG", True)  # Changed from False to True
        self._log_to_file = self._get_env_bool("GEMINI_LOG_TO_FILE", False)  # Changed from True to False
        self._log_level = os.environ.get("GEMINI_LOG_LEVEL", "DEBUG").upper()  # Changed from WARNING to DEBUG
        self._max_prompt_length = int(os.environ.get("GEMINI_MAX_PROMPT_LOG", "500"))  # Reduced from 1000
        self._max_response_length = int(os.environ.get("GEMINI_MAX_RESPONSE_LOG", "1000"))  # Reduced from 2000
    
    def _get_env_bool(self, key: str, default: bool) -> bool:
        """Get boolean value from environment variable"""
        value = os.environ.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    @property
    def enable_debug(self) -> bool:
        """Whether to enable debug logging"""
        return self._enable_debug
    
    @property
    def log_to_file(self) -> bool:
        """Whether to log to file"""
        return self._log_to_file
    
    @property
    def log_level(self) -> str:
        """Log level (DEBUG, INFO, WARNING, ERROR)"""
        return self._log_level
    
    @property
    def max_prompt_length(self) -> int:
        """Maximum prompt length to log"""
        return self._max_prompt_length
    
    @property
    def max_response_length(self) -> int:
        """Maximum response length to log"""
        return self._max_response_length
    
    def enable(self):
        """Enable debugging"""
        self._enable_debug = True
        os.environ["GEMINI_DEBUG"] = "true"
        print("✅ Gemini debugging enabled")
    
    def disable(self):
        """Disable debugging"""
        self._enable_debug = False
        os.environ["GEMINI_DEBUG"] = "false"
        print("❌ Gemini debugging disabled")
    
    def set_log_level(self, level: str):
        """Set log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        
        self._log_level = level.upper()
        os.environ["GEMINI_LOG_LEVEL"] = level.upper()
        print(f"📊 Gemini log level set to: {level.upper()}")
    
    def print_status(self):
        """Print current debug configuration"""
        print("🔧 Gemini Debug Configuration:")
        print(f"   Debug Enabled: {self.enable_debug}")
        print(f"   Log to File: {self.log_to_file}")
        print(f"   Log Level: {self.log_level}")
        print(f"   Max Prompt Log Length: {self.max_prompt_length}")
        print(f"   Max Response Log Length: {self.max_response_length}")

# Global configuration instance
config = GeminiDebugConfig()

def enable_gemini_debug():
    """Enable Gemini debugging"""
    config.enable()

def disable_gemini_debug():
    """Disable Gemini debugging"""
    config.disable()

def set_gemini_log_level(level: str):
    """Set Gemini log level"""
    config.set_log_level(level)

def show_gemini_debug_status():
    """Show current Gemini debug status"""
    config.print_status()

# Environment variable documentation
ENV_VARS = {
    "GEMINI_DEBUG": "Enable/disable Gemini debugging (true/false, default: true)",  # Updated default
    "GEMINI_LOG_TO_FILE": "Enable/disable file logging (true/false, default: false)",  # Updated default
    "GEMINI_LOG_LEVEL": "Log level (DEBUG/INFO/WARNING/ERROR, default: DEBUG)",  # Updated default
    "GEMINI_MAX_PROMPT_LOG": "Maximum prompt length to log (default: 500)",  # Updated default
    "GEMINI_MAX_RESPONSE_LOG": "Maximum response length to log (default: 1000)"  # Updated default
}

def print_env_help():
    """Print environment variable help"""
    print("🔧 Gemini Debug Environment Variables:")
    for var, description in ENV_VARS.items():
        print(f"   {var}: {description}") 