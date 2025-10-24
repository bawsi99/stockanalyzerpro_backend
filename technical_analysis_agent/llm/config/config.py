"""
LLM Configuration Loader and Validator

This module handles loading and validating LLM configuration from YAML files
and environment variables.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class LLMConfig:
    """
    LLM Configuration loader and manager.
    
    Loads configuration from:
    1. YAML file (llm_assignments.yaml)
    2. Environment variables (overrides)
    3. Runtime overrides
    """
    
    def __init__(self, config_file: Optional[str] = None, environment: str = None):
        """
        Initialize LLM configuration.
        
        Args:
            config_file: Path to YAML config file (optional)
            environment: Environment name (development, staging, production)
        """
        self.config_file = config_file or self._get_default_config_path()
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.config = self._load_configuration()
        
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        current_dir = Path(__file__).parent
        return str(current_dir / "llm_assignments.yaml")
        
    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
                
            # Apply environment-specific overrides
            if self.environment in config.get('environments', {}):
                env_config = config['environments'][self.environment]
                self._merge_config(config['default'], env_config.get('default', {}))
                
            return config
            
        except FileNotFoundError:
            print(f"⚠️  Config file not found: {self.config_file}")
            return self._get_fallback_config()
        except yaml.YAMLError as e:
            print(f"⚠️  Error parsing YAML config: {e}")
            return self._get_fallback_config()
            
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Get fallback configuration if YAML loading fails."""
        return {
            'default': {
                'provider': 'gemini',
                'model': 'gemini-2.5-flash',
                'enable_code_execution': True,
                'max_retries': 3,
                'timeout': 60
            },
            'agents': {}
        }
        
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Merge override configuration into base configuration."""
        for key, value in override.items():
            if isinstance(value, dict) and key in base:
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent (e.g., "indicator_agent", "volume_agent")
            
        Returns:
            Configuration dictionary for the agent
        """
        # Start with default config
        agent_config = self.config['default'].copy()
        
        # Override with agent-specific config if exists
        agents_config = self.config.get('agents', {})
        if agent_name in agents_config:
            self._merge_config(agent_config, agents_config[agent_name])
            
        # Override with environment variables
        self._apply_env_overrides(agent_config, agent_name)
        
        # Ensure API key strategy defaults are set
        if 'api_key_strategy' not in agent_config:
            agent_config['api_key_strategy'] = 'round_robin'
        if 'api_key_index' not in agent_config:
            agent_config['api_key_index'] = None
        
        return agent_config
    
    def _apply_env_overrides(self, config: Dict[str, Any], agent_name: str) -> None:
        """Apply environment variable overrides to configuration."""
        # General overrides
        env_mappings = {
            'LLM_PROVIDER': 'provider',
            'LLM_MODEL': 'model', 
            'LLM_MAX_RETRIES': 'max_retries',
            'LLM_TIMEOUT': 'timeout',
            'LLM_CODE_EXECUTION': 'enable_code_execution'
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if config_key in ['max_retries', 'timeout']:
                    try:
                        config[config_key] = int(value)
                    except ValueError:
                        pass
                elif config_key == 'enable_code_execution':
                    config[config_key] = value.lower() in ('true', '1', 'yes')
                else:
                    config[config_key] = value
        
        # Agent-specific overrides
        agent_prefix = f"LLM_{agent_name.upper()}_"
        for env_var in os.environ:
            if env_var.startswith(agent_prefix):
                config_key = env_var[len(agent_prefix):].lower()
                value = os.getenv(env_var)
                
                if config_key in ['max_retries', 'timeout']:
                    try:
                        config[config_key] = int(value)
                    except ValueError:
                        pass
                elif config_key == 'enable_code_execution':
                    config[config_key] = value.lower() in ('true', '1', 'yes')
                else:
                    config[config_key] = value
                    
    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """
        Get provider-specific configuration.
        
        Args:
            provider_name: Name of the provider ("gemini", "openai", "claude")
            
        Returns:
            Provider configuration dictionary
        """
        provider_configs = {
            'gemini': {
                'api_key_env_vars': ['GEMINI_API_KEY', 'GEMINI_API_KEY1', 'GEMINI_API_KEY2', 
                                   'GEMINI_API_KEY3', 'GEMINI_API_KEY4', 'GEMINI_API_KEY5'],
                'available_models': ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-1.5-pro'],
                'supports_code_execution': True,
                'supports_images': True
            },
            'openai': {
                'api_key_env_vars': ['OPENAI_API_KEY'],
                'available_models': ['gpt-4o-mini', 'gpt-4o', 'o1-preview'],
                'supports_code_execution': False,  # OpenAI doesn't have built-in code execution
                'supports_images': True
            },
            'claude': {
                'api_key_env_vars': ['CLAUDE_API_KEY', 'ANTHROPIC_API_KEY'],
                'available_models': ['claude-3-5-haiku', 'claude-3-5-sonnet'],
                'supports_code_execution': False,
                'supports_images': True
            }
        }
        
        return provider_configs.get(provider_name, {})
        
    def validate_configuration(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check if default configuration exists
            if 'default' not in self.config:
                print("❌ No default configuration found")
                return False
                
            # Validate default config has required fields
            required_fields = ['provider', 'model']
            for field in required_fields:
                if field not in self.config['default']:
                    print(f"❌ Missing required field in default config: {field}")
                    return False
            
            # Validate each agent configuration
            agents = self.config.get('agents', {})
            for agent_name, agent_config in agents.items():
                if not self._validate_agent_config(agent_name, agent_config):
                    return False
                    
            print("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def _validate_agent_config(self, agent_name: str, config: Dict[str, Any]) -> bool:
        """Validate a single agent's configuration."""
        # Check provider is supported
        provider = config.get('provider', self.config['default']['provider'])
        supported_providers = ['gemini', 'openai', 'claude']
        
        if provider not in supported_providers:
            print(f"❌ Unsupported provider for {agent_name}: {provider}")
            return False
            
        # Check model is available for provider
        model = config.get('model', self.config['default']['model'])
        provider_config = self.get_provider_config(provider)
        available_models = provider_config.get('available_models', [])
        
        if available_models and model not in available_models:
            print(f"⚠️  Model {model} not in known models for {provider}. This may still work.")
            
        return True
    
    def list_agents(self) -> list:
        """Get list of configured agents."""
        return list(self.config.get('agents', {}).keys())
    
    def print_configuration(self) -> None:
        """Print current configuration for debugging."""
        print("\n🔧 LLM Configuration Summary")
        print("=" * 40)
        print(f"Environment: {self.environment}")
        print(f"Config file: {self.config_file}")
        
        print(f"\nDefault config:")
        for key, value in self.config['default'].items():
            print(f"  {key}: {value}")
            
        agents = self.config.get('agents', {})
        if agents:
            print(f"\nAgent-specific configs:")
            for agent_name in agents:
                agent_config = self.get_agent_config(agent_name)
                print(f"  {agent_name}:")
                print(f"    Provider: {agent_config['provider']}")
                print(f"    Model: {agent_config['model']}")
                print(f"    Code execution: {agent_config.get('enable_code_execution', 'N/A')}")
                print(f"    Timeout: {agent_config.get('timeout', 'N/A')}s")


# Global configuration instance
_config_instance = None

def get_llm_config(config_file: Optional[str] = None, 
                   environment: Optional[str] = None) -> LLMConfig:
    """
    Get the global LLM configuration instance.
    
    Args:
        config_file: Path to config file (only used on first call)
        environment: Environment name (only used on first call)
        
    Returns:
        LLMConfig instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = LLMConfig(config_file, environment)
        
    return _config_instance


if __name__ == "__main__":
    # Test configuration loading
    config = LLMConfig()
    config.print_configuration()
    config.validate_configuration()