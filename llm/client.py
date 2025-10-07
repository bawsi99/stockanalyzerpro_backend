"""
LLM Client - Universal interface to all LLM providers

This is the main client that agents will use. It handles:
1. Provider selection based on configuration
2. Model routing per agent
3. Simple, clean API for agents
4. Error handling and retries
"""

import asyncio
from typing import Optional, List, Any, Dict
from .providers.base import BaseLLMProvider
from .providers.gemini import GeminiProvider
from .config.config import LLMConfig, get_llm_config
from .utils import SimpleTimer, truncate_for_logging, debug_print


class LLMClient:
    """
    Universal LLM client that routes requests to appropriate providers.
    
    This is the main interface that agents use to communicate with LLMs.
    It abstracts away the complexities of different providers and models.
    """
    
    def __init__(self, 
                 agent_name: Optional[str] = None,
                 provider: Optional[str] = None, 
                 model: Optional[str] = None,
                 config: Optional[LLMConfig] = None,
                 **kwargs):
        """
        Initialize LLM client.
        
        Args:
            agent_name: Name of the agent using this client (for config lookup)
            provider: Override provider ("gemini", "openai", "claude")
            model: Override model name
            config: Custom configuration object
            **kwargs: Additional provider-specific parameters
        """
        self.agent_name = agent_name or "default"
        self.config = config or get_llm_config()
        self.custom_kwargs = kwargs
        
        # Get configuration for this agent
        self.agent_config = self.config.get_agent_config(self.agent_name)
        
        # Override with explicit parameters
        if provider:
            self.agent_config['provider'] = provider
        if model:
            self.agent_config['model'] = model
            
        # Initialize provider
        self.provider = self._create_provider()
        
        debug_print(f"LLMClient initialized for {self.agent_name}: "
                   f"{self.agent_config['provider']}:{self.agent_config['model']}")
    
    def _create_provider(self) -> BaseLLMProvider:
        """Create the appropriate provider instance based on configuration."""
        provider_name = self.agent_config['provider']
        model = self.agent_config['model']
        
        # Provider factory
        if provider_name == 'gemini':
            return GeminiProvider(
                model=model,
                agent_name=self.agent_name,
                **self.custom_kwargs
            )
        elif provider_name == 'openai':
            # Will be implemented in Phase 7
            raise NotImplementedError(f"OpenAI provider not yet implemented")
        elif provider_name == 'claude':
            # Will be implemented in Phase 7
            raise NotImplementedError(f"Claude provider not yet implemented")
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
    
    async def generate(self, 
                      prompt: str, 
                      images: Optional[List[Any]] = None,
                      **kwargs) -> str:
        """
        Generate response from prompt (and optionally images).
        
        This is the main method agents use for LLM calls.
        
        Args:
            prompt: Text prompt
            images: Optional list of images (PIL, bytes, etc.)
            **kwargs: Additional parameters (timeout, retries, etc.)
            
        Returns:
            Generated text response
        """
        # Merge config with kwargs
        call_config = self.agent_config.copy()
        call_config.update(kwargs)
        
        # Extract parameters
        enable_code_execution = call_config.get('enable_code_execution', True)
        max_retries = call_config.get('max_retries', 3)
        timeout = call_config.get('timeout', 60)
        
        debug_print(f"Generating for {self.agent_name}: "
                   f"prompt={truncate_for_logging(prompt, 50)}, "
                   f"images={len(images) if images else 0}, "
                   f"code_exec={enable_code_execution}")
        
        # Start timing
        with SimpleTimer() as timer:
            try:
                # Choose appropriate generation method
                if images:
                    response = await asyncio.wait_for(
                        self.provider.generate_with_images(
                            prompt=prompt,
                            images=images,
                            enable_code_execution=enable_code_execution,
                            max_retries=max_retries
                        ),
                        timeout=timeout
                    )
                else:
                    response = await asyncio.wait_for(
                        self.provider.generate_text(
                            prompt=prompt,
                            enable_code_execution=enable_code_execution,
                            max_retries=max_retries
                        ),
                        timeout=timeout
                    )
                
                debug_print(f"Generation completed for {self.agent_name}: "
                           f"time={timer.elapsed()}s, "
                           f"response={truncate_for_logging(response, 100)}")
                
                return response
                
            except asyncio.TimeoutError:
                error_msg = f"LLM request timed out after {timeout}s for {self.agent_name}"
                debug_print(error_msg)
                raise TimeoutError(error_msg)
                
            except Exception as e:
                error_msg = f"LLM request failed for {self.agent_name}: {str(e)}"
                debug_print(error_msg)
                raise
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text-only response (convenience method).
        
        Args:
            prompt: Text prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        return await self.generate(prompt=prompt, images=None, **kwargs)
    
    async def generate_with_images(self, prompt: str, images: List[Any], **kwargs) -> str:
        """
        Generate response with images (convenience method).
        
        Args:
            prompt: Text prompt
            images: List of images
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        return await self.generate(prompt=prompt, images=images, **kwargs)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration for this client."""
        return self.agent_config.copy()
    
    def get_provider_info(self) -> str:
        """Get provider and model information."""
        return f"{self.agent_config['provider']}:{self.agent_config['model']}"


class LLMClientFactory:
    """
    Factory for creating LLM clients with different configurations.
    
    This provides convenient methods for creating clients for different agents.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize factory.
        
        Args:
            config: Custom configuration (uses default if None)
        """
        self.config = config or get_llm_config()
    
    def create_for_agent(self, agent_name: str, **kwargs) -> LLMClient:
        """
        Create client for specific agent.
        
        Args:
            agent_name: Agent name (looks up configuration)
            **kwargs: Override parameters
            
        Returns:
            LLMClient configured for the agent
        """
        return LLMClient(agent_name=agent_name, config=self.config, **kwargs)
    
    def create_custom(self, provider: str, model: str, **kwargs) -> LLMClient:
        """
        Create client with custom provider/model.
        
        Args:
            provider: Provider name
            model: Model name
            **kwargs: Additional parameters
            
        Returns:
            LLMClient with custom configuration
        """
        return LLMClient(provider=provider, model=model, config=self.config, **kwargs)
    
    def list_configured_agents(self) -> List[str]:
        """Get list of agents with specific configurations."""
        return self.config.list_agents()


# Global factory instance for convenience
_factory = None

def get_llm_client(agent_name: str = None, **kwargs) -> LLMClient:
    """
    Convenience function to get an LLM client.
    
    Args:
        agent_name: Agent name for configuration lookup
        **kwargs: Override parameters
        
    Returns:
        LLMClient instance
    """
    global _factory
    
    if _factory is None:
        _factory = LLMClientFactory()
    
    if agent_name:
        return _factory.create_for_agent(agent_name, **kwargs)
    else:
        return LLMClient(**kwargs)


# Simple usage examples and testing
async def test_llm_client():
    """Test the LLM client with different configurations."""
    print("🧪 Testing LLM Client")
    print("=" * 40)
    
    try:
        # Test default client
        print("\n1. Testing default client...")
        default_client = LLMClient()
        print(f"   Provider: {default_client.get_provider_info()}")
        
        # Test agent-specific client
        print("\n2. Testing agent-specific client...")
        indicator_client = get_llm_client("indicator_agent")
        print(f"   Provider: {indicator_client.get_provider_info()}")
        print(f"   Config: {indicator_client.get_config()}")
        
        # Test custom client
        print("\n3. Testing custom client...")
        custom_client = LLMClient(provider="gemini", model="gemini-2.5-pro")
        print(f"   Provider: {custom_client.get_provider_info()}")
        
        # Test factory
        print("\n4. Testing factory...")
        factory = LLMClientFactory()
        agents = factory.list_configured_agents()
        print(f"   Configured agents: {agents}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_llm_client())