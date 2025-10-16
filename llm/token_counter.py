"""
Token Counter - Universal token usage tracking system for LLM requests

This module provides a flexible token counting system that works with different
LLM providers and response structures. It's designed to integrate seamlessly
with the existing backend/llm system.

Key Features:
- Provider-agnostic token extraction
- Flexible response structure parsing  
- Thread-safe usage tracking
- Detailed breakdowns by agent/provider
- Integration with analysis service
"""

import time
import threading
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json


@dataclass
class TokenUsageData:
    """Represents token usage for a single LLM call."""
    call_id: str
    agent_name: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: float
    duration_ms: float
    success: bool = True
    error_message: Optional[str] = None
    call_metadata: Dict[str, Any] = field(default_factory=dict)


class TokenExtractor(ABC):
    """Abstract base class for extracting token usage from provider-specific responses."""
    
    @abstractmethod
    def extract_tokens(self, response: Any) -> Tuple[int, int, int]:
        """
        Extract token counts from response.
        
        Args:
            response: Provider-specific response object
            
        Returns:
            Tuple of (input_tokens, output_tokens, total_tokens)
        """
        pass
    
    @abstractmethod
    def can_handle(self, response: Any) -> bool:
        """
        Check if this extractor can handle the given response.
        
        Args:
            response: Response object to check
            
        Returns:
            True if this extractor can handle the response
        """
        pass


class GeminiTokenExtractor(TokenExtractor):
    """Token extractor for Gemini API responses."""
    
    def extract_tokens(self, response: Any) -> Tuple[int, int, int]:
        """Extract token counts from Gemini response."""
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        
        try:
            # Handle different Gemini response formats
            if hasattr(response, 'usage_metadata'):
                # Direct response object with usage_metadata attribute
                usage = response.usage_metadata
                input_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                total_tokens = getattr(usage, 'total_token_count', 0) or 0
                
            elif isinstance(response, dict):
                # Dictionary response structure
                if 'usageMetadata' in response:
                    # New format with usageMetadata
                    usage = response['usageMetadata']
                    input_tokens = usage.get('promptTokenCount', 0) or 0
                    output_tokens = usage.get('candidatesTokenCount', 0) or 0
                    total_tokens = usage.get('totalTokenCount', 0) or 0
                    
                elif 'usage_metadata' in response:
                    # Legacy format with usage_metadata
                    usage = response['usage_metadata']
                    if isinstance(usage, dict):
                        input_tokens = usage.get('prompt_token_count', 0) or 0
                        output_tokens = usage.get('candidates_token_count', 0) or 0
                        total_tokens = usage.get('total_token_count', 0) or 0
                    else:
                        # Handle object-style usage metadata
                        input_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                        output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                        total_tokens = getattr(usage, 'total_token_count', 0) or 0
                        
                elif 'promptTokenCount' in response:
                    # Direct token counts in response root
                    input_tokens = response.get('promptTokenCount', 0) or 0
                    output_tokens = response.get('candidatesTokenCount', 0) or 0
                    total_tokens = response.get('totalTokenCount', 0) or 0
                    
        except Exception as e:
            print(f"⚠️ Error extracting tokens from Gemini response: {e}")
            # Return zeros if extraction fails
            return 0, 0, 0
        
        # Validate token counts
        if total_tokens == 0 and (input_tokens > 0 or output_tokens > 0):
            total_tokens = input_tokens + output_tokens
            
        return input_tokens, output_tokens, total_tokens
    
    def can_handle(self, response: Any) -> bool:
        """Check if this is a Gemini response."""
        if response is None:
            return False
            
        # Check for Gemini-specific fields
        if hasattr(response, 'usage_metadata'):
            return True
            
        if isinstance(response, dict):
            gemini_fields = ['usageMetadata', 'usage_metadata', 'promptTokenCount', 'candidates']
            return any(field in response for field in gemini_fields)
            
        return False


class OpenAITokenExtractor(TokenExtractor):
    """Token extractor for OpenAI API responses (future implementation)."""
    
    def extract_tokens(self, response: Any) -> Tuple[int, int, int]:
        """Extract token counts from OpenAI response."""
        # TODO: Implement when OpenAI provider is added
        if isinstance(response, dict) and 'usage' in response:
            usage = response['usage']
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            return input_tokens, output_tokens, total_tokens
        return 0, 0, 0
    
    def can_handle(self, response: Any) -> bool:
        """Check if this is an OpenAI response."""
        if isinstance(response, dict):
            return 'usage' in response and 'model' in response
        return False


class ClaudeTokenExtractor(TokenExtractor):
    """Token extractor for Claude API responses (future implementation)."""
    
    def extract_tokens(self, response: Any) -> Tuple[int, int, int]:
        """Extract token counts from Claude response."""
        # TODO: Implement when Claude provider is added
        if isinstance(response, dict) and 'usage' in response:
            usage = response['usage']
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            total_tokens = input_tokens + output_tokens
            return input_tokens, output_tokens, total_tokens
        return 0, 0, 0
    
    def can_handle(self, response: Any) -> bool:
        """Check if this is a Claude response."""
        if isinstance(response, dict):
            return 'usage' in response and 'input_tokens' in response.get('usage', {})
        return False


class TokenCounter:
    """
    Main token counter class for tracking LLM usage across agents and requests.
    
    This class is thread-safe and can be used across multiple concurrent requests.
    """
    
    def __init__(self):
        self.extractors: List[TokenExtractor] = [
            GeminiTokenExtractor(),
            OpenAITokenExtractor(),
            ClaudeTokenExtractor()
        ]
        self._usage_data: List[TokenUsageData] = []
        self._lock = threading.Lock()
        self._call_counter = 0
    
    def track_usage(self, 
                   response: Any,
                   agent_name: str = "unknown",
                   provider: str = "unknown", 
                   model: str = "unknown",
                   duration_ms: float = 0.0,
                   success: bool = True,
                   error_message: Optional[str] = None,
                   call_metadata: Optional[Dict[str, Any]] = None) -> Optional[TokenUsageData]:
        """
        Track token usage from an LLM response.
        
        Args:
            response: Raw response from LLM provider
            agent_name: Name of the agent making the call
            provider: LLM provider name (gemini, openai, claude)
            model: Model name used
            duration_ms: Request duration in milliseconds
            success: Whether the request was successful
            error_message: Error message if request failed
            call_metadata: Additional metadata about the call
            
        Returns:
            TokenUsageData if successful, None otherwise
        """
        if response is None and success:
            print("⚠️ Warning: Received None response for successful call")
            return None
        
        # Find appropriate extractor
        extractor = None
        for ext in self.extractors:
            if ext.can_handle(response):
                extractor = ext
                break
        
        if extractor is None:
            print(f"⚠️ No extractor found for response type: {type(response)}")
            # Create zero usage data for failed extractions
            input_tokens, output_tokens, total_tokens = 0, 0, 0
        else:
            try:
                input_tokens, output_tokens, total_tokens = extractor.extract_tokens(response)
            except Exception as e:
                print(f"⚠️ Token extraction failed: {e}")
                input_tokens, output_tokens, total_tokens = 0, 0, 0
        
        # Create usage data
        with self._lock:
            self._call_counter += 1
            call_id = f"{agent_name}_{self._call_counter}_{int(time.time() * 1000)}"
        
        usage_data = TokenUsageData(
            call_id=call_id,
            agent_name=agent_name,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            timestamp=time.time(),
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            call_metadata=call_metadata or {}
        )
        
        # Store usage data
        with self._lock:
            self._usage_data.append(usage_data)
        
        # Log the usage
        if success and total_tokens > 0:
            print(f"📊 {agent_name} token usage: {input_tokens} input + {output_tokens} output = {total_tokens} total")
        
        return usage_data
    
    def get_total_usage(self) -> Dict[str, int]:
        """Get total token usage across all tracked calls."""
        with self._lock:
            total_input = sum(data.input_tokens for data in self._usage_data)
            total_output = sum(data.output_tokens for data in self._usage_data)
            total_tokens = sum(data.total_tokens for data in self._usage_data)
            successful_calls = sum(1 for data in self._usage_data if data.success)
            failed_calls = len(self._usage_data) - successful_calls
            
        return {
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_tokens': total_tokens,
            'total_calls': len(self._usage_data),
            'successful_calls': successful_calls,
            'failed_calls': failed_calls
        }
    
    def get_usage_by_agent(self) -> Dict[str, Dict[str, int]]:
        """Get token usage breakdown by agent."""
        agent_usage = {}
        
        with self._lock:
            for data in self._usage_data:
                agent = data.agent_name
                if agent not in agent_usage:
                    agent_usage[agent] = {
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'total_tokens': 0,
                        'calls': 0,
                        'successful_calls': 0,
                        'failed_calls': 0
                    }
                
                agent_usage[agent]['input_tokens'] += data.input_tokens
                agent_usage[agent]['output_tokens'] += data.output_tokens
                agent_usage[agent]['total_tokens'] += data.total_tokens
                agent_usage[agent]['calls'] += 1
                
                if data.success:
                    agent_usage[agent]['successful_calls'] += 1
                else:
                    agent_usage[agent]['failed_calls'] += 1
        
        return agent_usage
    
    def get_usage_by_provider(self) -> Dict[str, Dict[str, int]]:
        """Get token usage breakdown by provider."""
        provider_usage = {}
        
        with self._lock:
            for data in self._usage_data:
                provider = data.provider
                if provider not in provider_usage:
                    provider_usage[provider] = {
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'total_tokens': 0,
                        'calls': 0,
                        'successful_calls': 0,
                        'failed_calls': 0
                    }
                
                provider_usage[provider]['input_tokens'] += data.input_tokens
                provider_usage[provider]['output_tokens'] += data.output_tokens
                provider_usage[provider]['total_tokens'] += data.total_tokens
                provider_usage[provider]['calls'] += 1
                
                if data.success:
                    provider_usage[provider]['successful_calls'] += 1
                else:
                    provider_usage[provider]['failed_calls'] += 1
        
        return provider_usage
    
    def get_usage_by_model(self) -> Dict[str, Dict[str, int]]:
        """Get token usage breakdown by model (e.g., gemini-2.5-pro vs gemini-2.5-flash)."""
        model_usage = {}
        
        with self._lock:
            for data in self._usage_data:
                model = data.model
                if model not in model_usage:
                    model_usage[model] = {
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'total_tokens': 0,
                        'calls': 0,
                        'successful_calls': 0,
                        'failed_calls': 0,
                        'avg_input_per_call': 0.0,
                        'avg_output_per_call': 0.0,
                        'agents_using_model': set()
                    }
                
                model_usage[model]['input_tokens'] += data.input_tokens
                model_usage[model]['output_tokens'] += data.output_tokens
                model_usage[model]['total_tokens'] += data.total_tokens
                model_usage[model]['calls'] += 1
                model_usage[model]['agents_using_model'].add(data.agent_name)
                
                if data.success:
                    model_usage[model]['successful_calls'] += 1
                else:
                    model_usage[model]['failed_calls'] += 1
        
        # Calculate averages and convert sets to lists for JSON serialization
        for model, usage in model_usage.items():
            if usage['calls'] > 0:
                usage['avg_input_per_call'] = usage['input_tokens'] / usage['calls']
                usage['avg_output_per_call'] = usage['output_tokens'] / usage['calls']
            usage['agents_using_model'] = list(usage['agents_using_model'])
        
        return model_usage
    
    def get_usage_by_agent_and_model(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Get token usage breakdown by agent and model combination."""
        agent_model_usage = {}
        
        with self._lock:
            for data in self._usage_data:
                agent = data.agent_name
                model = data.model
                
                if agent not in agent_model_usage:
                    agent_model_usage[agent] = {}
                    
                if model not in agent_model_usage[agent]:
                    agent_model_usage[agent][model] = {
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'total_tokens': 0,
                        'calls': 0,
                        'successful_calls': 0,
                        'failed_calls': 0
                    }
                
                agent_model_usage[agent][model]['input_tokens'] += data.input_tokens
                agent_model_usage[agent][model]['output_tokens'] += data.output_tokens
                agent_model_usage[agent][model]['total_tokens'] += data.total_tokens
                agent_model_usage[agent][model]['calls'] += 1
                
                if data.success:
                    agent_model_usage[agent][model]['successful_calls'] += 1
                else:
                    agent_model_usage[agent][model]['failed_calls'] += 1
        
        return agent_model_usage
    
    def get_detailed_usage(self) -> List[Dict[str, Any]]:
        """Get detailed usage data for all calls."""
        with self._lock:
            return [
                {
                    'call_id': data.call_id,
                    'agent_name': data.agent_name,
                    'provider': data.provider,
                    'model': data.model,
                    'input_tokens': data.input_tokens,
                    'output_tokens': data.output_tokens,
                    'total_tokens': data.total_tokens,
                    'timestamp': data.timestamp,
                    'duration_ms': data.duration_ms,
                    'success': data.success,
                    'error_message': data.error_message,
                    'call_metadata': data.call_metadata
                }
                for data in self._usage_data.copy()
            ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive token usage summary."""
        total_usage = self.get_total_usage()
        agent_usage = self.get_usage_by_agent()
        provider_usage = self.get_usage_by_provider()
        model_usage = self.get_usage_by_model()
        agent_model_usage = self.get_usage_by_agent_and_model()
        
        # Calculate timing statistics
        with self._lock:
            durations = [data.duration_ms for data in self._usage_data if data.duration_ms > 0]
        
        timing_stats = {}
        if durations:
            timing_stats = {
                'avg_duration_ms': sum(durations) / len(durations),
                'min_duration_ms': min(durations),
                'max_duration_ms': max(durations),
                'total_duration_ms': sum(durations)
            }
        
        return {
            'total_usage': total_usage,
            'usage_by_agent': agent_usage,
            'usage_by_provider': provider_usage,
            'usage_by_model': model_usage,
            'usage_by_agent_and_model': agent_model_usage,
            'timing_stats': timing_stats,
            'summary_timestamp': time.time()
        }
    
    def print_summary(self):
        """Print a human-readable token usage summary."""
        summary = self.get_summary()
        total = summary['total_usage']
        
        print("\n" + "=" * 60)
        print("📊 TOKEN USAGE SUMMARY")
        print("=" * 60)
        print(f"Total Calls: {total['total_calls']}")
        print(f"Successful: {total['successful_calls']}, Failed: {total['failed_calls']}")
        print(f"\nTotal Input Tokens: {total['total_input_tokens']:,}")
        print(f"Total Output Tokens: {total['total_output_tokens']:,}")
        print(f"Total Tokens: {total['total_tokens']:,}")
        
        # Agent breakdown
        print(f"\n{'USAGE BY AGENT':-^60}")
        for agent, usage in summary['usage_by_agent'].items():
            print(f"{agent:20} | {usage['total_tokens']:>8,} tokens | {usage['calls']:>3} calls")
        
        # Provider breakdown  
        print(f"\n{'USAGE BY PROVIDER':-^60}")
        for provider, usage in summary['usage_by_provider'].items():
            print(f"{provider:20} | {usage['total_tokens']:>8,} tokens | {usage['calls']:>3} calls")
        
        # Model breakdown (KEY FEATURE - shows different models used)
        print(f"\n{'USAGE BY MODEL':-^60}")
        for model, usage in summary['usage_by_model'].items():
            avg_tokens = usage['total_tokens'] / usage['calls'] if usage['calls'] > 0 else 0
            agents = ', '.join(usage['agents_using_model'][:3])  # Show first 3 agents
            if len(usage['agents_using_model']) > 3:
                agents += f" +{len(usage['agents_using_model']) - 3} more"
            print(f"{model:20} | {usage['total_tokens']:>8,} tokens | {usage['calls']:>3} calls | {avg_tokens:>6.0f} avg")
            print(f"{' ':20} | agents: {agents}")
        
        # Agent-Model combination breakdown (shows which agents use which models)
        print(f"\n{'AGENT-MODEL COMBINATIONS':-^60}")
        for agent, models in summary['usage_by_agent_and_model'].items():
            print(f"\n{agent}:")
            for model, usage in models.items():
                print(f"  {model:18} | {usage['total_tokens']:>6,} tokens | {usage['calls']:>2} calls")
        
        # Timing stats
        if summary['timing_stats']:
            timing = summary['timing_stats']
            print(f"\n{'TIMING STATISTICS':-^60}")
            print(f"Average Duration: {timing['avg_duration_ms']:.2f}ms")
            print(f"Total Duration: {timing['total_duration_ms']:.2f}ms")
            print(f"Min/Max: {timing['min_duration_ms']:.2f}ms / {timing['max_duration_ms']:.2f}ms")
        
        print("=" * 60)
    
    def clear(self):
        """Clear all stored usage data."""
        with self._lock:
            self._usage_data.clear()
            self._call_counter = 0
        print("🗑️ Token usage data cleared")
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export all data to a dictionary for serialization."""
        return {
            'usage_data': self.get_detailed_usage(),
            'summary': self.get_summary()
        }
    
    def export_to_json(self, filename: Optional[str] = None) -> str:
        """Export usage data to JSON format."""
        data = self.export_to_dict()
        json_str = json.dumps(data, indent=2, default=str)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(json_str)
            print(f"✅ Token usage data exported to {filename}")
        
        return json_str


# Global token counter instance for the analysis service
_global_counter: Optional[TokenCounter] = None
_counter_lock = threading.Lock()


def get_token_counter() -> TokenCounter:
    """Get the global token counter instance (thread-safe singleton)."""
    global _global_counter
    
    if _global_counter is None:
        with _counter_lock:
            if _global_counter is None:
                _global_counter = TokenCounter()
                
    return _global_counter


def reset_token_counter():
    """Reset the global token counter."""
    global _global_counter
    
    with _counter_lock:
        if _global_counter is not None:
            _global_counter.clear()
        else:
            _global_counter = TokenCounter()


def track_llm_usage(response: Any, 
                   agent_name: str = "unknown",
                   provider: str = "unknown",
                   model: str = "unknown", 
                   duration_ms: float = 0.0,
                   success: bool = True,
                   error_message: Optional[str] = None,
                   call_metadata: Optional[Dict[str, Any]] = None) -> Optional[TokenUsageData]:
    """
    Convenience function to track LLM usage with the global counter.
    
    Args:
        response: Raw response from LLM provider
        agent_name: Name of the agent making the call  
        provider: LLM provider name
        model: Model name used
        duration_ms: Request duration in milliseconds
        success: Whether the request was successful
        error_message: Error message if request failed
        call_metadata: Additional metadata about the call
        
    Returns:
        TokenUsageData if successful, None otherwise
    """
    counter = get_token_counter()
    return counter.track_usage(
        response=response,
        agent_name=agent_name,
        provider=provider,
        model=model,
        duration_ms=duration_ms,
        success=success,
        error_message=error_message,
        call_metadata=call_metadata
    )


def get_token_usage_summary() -> Dict[str, Any]:
    """Get comprehensive token usage summary from global counter."""
    counter = get_token_counter()
    return counter.get_summary()


def print_token_usage_summary():
    """Print token usage summary from global counter."""
    counter = get_token_counter()
    counter.print_summary()


def export_token_usage(filename: Optional[str] = None) -> str:
    """Export token usage data from global counter."""
    counter = get_token_counter()
    return counter.export_to_json(filename)


def get_model_usage_summary() -> Dict[str, Any]:
    """Get model-specific usage summary from global counter."""
    counter = get_token_counter()
    return counter.get_usage_by_model()


def get_agent_model_combinations() -> Dict[str, Any]:
    """Get agent-model combination usage from global counter."""
    counter = get_token_counter()
    return counter.get_usage_by_agent_and_model()


def get_agent_timing_breakdown() -> Dict[str, float]:
    """Get total duration (in seconds) for each agent from global counter."""
    counter = get_token_counter()
    agent_timings = {}
    
    with counter._lock:
        for usage_data in counter._usage_data:
            agent = usage_data.agent_name
            if agent not in agent_timings:
                agent_timings[agent] = 0.0
            agent_timings[agent] += usage_data.duration_ms / 1000.0  # Convert to seconds
    
    return agent_timings


def get_agent_details_table() -> Dict[str, Any]:
    """Build a per-agent details table including image usage and size.

    Returns a dict with 'rows' (list of row dicts) and 'totals'.
    """
    counter = get_token_counter()
    rows = []
    totals = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'total_time_s': 0.0
    }
    # Aggregate per agent
    with counter._lock:
        # Map: agent -> aggregation
        agg: Dict[str, Dict[str, Any]] = {}
        for u in counter._usage_data:
            a = u.agent_name
            if a not in agg:
                agg[a] = {
                    'input': 0,
                    'output': 0,
                    'total': 0,
                    'time_ms': 0.0,
                    'models': {},  # model -> total tokens
                    'image_used': False,
                    'image_sizes': []  # list of (w,h)
                }
            ag = agg[a]
            ag['input'] += u.input_tokens
            ag['output'] += u.output_tokens
            ag['total'] += u.total_tokens
            ag['time_ms'] += (u.duration_ms or 0.0)
            # model usage
            ag['models'][u.model] = ag['models'].get(u.model, 0) + u.total_tokens
            # image metrics
            cm = u.call_metadata or {}
            if cm.get('with_images'):
                ag['image_used'] = True
                for m in (cm.get('image_metrics') or []):
                    w, h = m.get('width'), m.get('height')
                    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                        ag['image_sizes'].append((w, h))
        
        # Build rows
        for agent, ag in agg.items():
            # choose model by max tokens
            model = '-'
            if ag['models']:
                model = max(ag['models'].items(), key=lambda kv: kv[1])[0]
            # choose image size: most common
            img_size_str = '-'
            img_tokens = '-'
            if ag['image_used'] and ag['image_sizes']:
                from collections import Counter
                import math
                common = Counter(ag['image_sizes']).most_common(1)
                if common:
                    w, h = common[0][0]
                    img_size_str = f"{w}x{h}"
                    # Gemini 2.0 image token rule
                    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                        if w <= 384 and h <= 384:
                            img_tokens = 258
                        else:
                            tiles = math.ceil(w/768) * math.ceil(h/768)
                            img_tokens = tiles * 258
            row = {
                'agent': agent,
                'model': model,
                'input': ag['input'],
                'output': ag['output'],
                'total': ag['total'],
                'time_s': round(ag['time_ms'] / 1000.0, 2),
                'image_included': 'yes' if ag['image_used'] else 'no',
                'image_size': img_size_str,
                'image_tokens': img_tokens
            }
            rows.append(row)
            totals['input_tokens'] += ag['input']
            totals['output_tokens'] += ag['output']
            totals['total_tokens'] += ag['total']
            totals['total_time_s'] += ag['time_ms'] / 1000.0
    # Sort rows by total tokens desc
    rows.sort(key=lambda r: r['total'], reverse=True)
    return {'rows': rows, 'totals': totals}


def print_agent_details_table():
    """Print a formatted agent details table with image columns."""
    data = get_agent_details_table()
    rows = data['rows']
    totals = data['totals']
    # Header
    print("\n🤖 AGENT DETAILS:")
    print("=" * 100)
    header = (
        f"{'Agent':<25} | {'Model':<10} | {'Input':>8} | {'Output':>8} | {'Total':>8} | {'Time':>8} | {'Image?':>7} | {'Img Size':>10} | {'Img Tokens':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        img_tokens_disp = (str(r['image_tokens']) if isinstance(r['image_tokens'], int) else '-')
        print(
            f"{r['agent']:<25} | "
            f"{(r['model'].split(':')[-1].split('-')[-1].upper() if r['model']!='-' else '-'): <10} | "
            f"{r['input']:>8,} | {r['output']:>8,} | {r['total']:>8,} | "
            f"{r['time_s']:>7.2f}s | {r['image_included']:>7} | {r['image_size']:>10} | {img_tokens_disp:>11}"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':<25} | {'':<10} | {totals['input_tokens']:>8,} | {totals['output_tokens']:>8,} | "
        f"{totals['total_tokens']:>8,} | {totals['total_time_s']:>7.2f}s | {'':>7} | {'':>10} | {'':>11}"
    )


def compare_model_efficiency(model1: str, model2: str) -> Dict[str, Any]:
    """Compare efficiency between two models."""
    counter = get_token_counter()
    model_usage = counter.get_usage_by_model()
    
    if model1 not in model_usage or model2 not in model_usage:
        return {
            'error': f'One or both models not found. Available models: {list(model_usage.keys())}'
        }
    
    usage1 = model_usage[model1]
    usage2 = model_usage[model2]
    
    return {
        'model1': model1,
        'model2': model2,
        'comparison': {
            'total_tokens': {
                model1: usage1['total_tokens'],
                model2: usage2['total_tokens'],
                'difference': usage1['total_tokens'] - usage2['total_tokens']
            },
            'calls': {
                model1: usage1['calls'],
                model2: usage2['calls'],
                'difference': usage1['calls'] - usage2['calls']
            },
            'avg_tokens_per_call': {
                model1: usage1['total_tokens'] / usage1['calls'] if usage1['calls'] > 0 else 0,
                model2: usage2['total_tokens'] / usage2['calls'] if usage2['calls'] > 0 else 0
            },
            'agents_using_model': {
                model1: usage1['agents_using_model'],
                model2: usage2['agents_using_model'],
                'common_agents': list(set(usage1['agents_using_model']) & set(usage2['agents_using_model'])),
                'model1_only': list(set(usage1['agents_using_model']) - set(usage2['agents_using_model'])),
                'model2_only': list(set(usage2['agents_using_model']) - set(usage1['agents_using_model']))
            }
        }
    }


# Example usage and testing
if __name__ == "__main__":
    # Test the token counter with mock responses demonstrating model-based tracking
    counter = TokenCounter()
    
    print("🧪 Testing Model-Based Token Tracking")
    print("=" * 50)
    
    # Test different Gemini models with different agents
    models_and_agents = [
        ("gemini-2.5-flash", "indicator_agent", 150, 75),
        ("gemini-2.5-pro", "final_decision_agent", 300, 120),
        ("gemini-2.5-flash", "volume_agent", 120, 60),
        ("gemini-2.5-pro", "sector_agent", 280, 140),
        ("gemini-2.5-flash", "risk_agent", 180, 90),
        ("gemini-2.5-pro", "indicator_agent", 250, 100)  # Same agent, different model
    ]
    
    for model, agent, input_tokens, output_tokens in models_and_agents:
        mock_response = {
            'usageMetadata': {
                'promptTokenCount': input_tokens,
                'candidatesTokenCount': output_tokens,
                'totalTokenCount': input_tokens + output_tokens
            }
        }
        
        usage = counter.track_usage(
            response=mock_response,
            agent_name=agent,
            provider="gemini",
            model=model,
            duration_ms=1250.5 + (input_tokens * 2)  # Simulate different durations
        )
        
        print(f"✅ Tracked: {agent} using {model} - {input_tokens + output_tokens} tokens")
    
    print("\n" + "=" * 50)
    print("📊 COMPREHENSIVE TOKEN USAGE SUMMARY")
    counter.print_summary()
    
    # Demonstrate model comparison using the counter's methods directly
    print("\n" + "=" * 50)
    print("🔍 MODEL EFFICIENCY COMPARISON")
    model_usage = counter.get_usage_by_model()
    
    if len(model_usage) >= 2:
        models = list(model_usage.keys())
        model1, model2 = models[0], models[1]
        usage1, usage2 = model_usage[model1], model_usage[model2]
        
        print(f"\n{model1} vs {model2}:")
        print(f"  Total tokens: {usage1['total_tokens']:,} vs {usage2['total_tokens']:,}")
        print(f"  Calls: {usage1['calls']} vs {usage2['calls']}")
        print(f"  Avg per call: {usage1['avg_input_per_call'] + usage1['avg_output_per_call']:.0f} vs {usage2['avg_input_per_call'] + usage2['avg_output_per_call']:.0f}")
        print(f"  {model1} agents: {usage1['agents_using_model']}")
        print(f"  {model2} agents: {usage2['agents_using_model']}")
        common_agents = list(set(usage1['agents_using_model']) & set(usage2['agents_using_model']))
        if common_agents:
            print(f"  Common agents: {common_agents}")
    else:
        print(f"Need at least 2 models for comparison. Found: {list(model_usage.keys())}")
