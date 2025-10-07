"""
Token Tracker - Manages token usage tracking for LLM calls within a single analysis.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class TokenUsage:
    """Represents token usage for a single LLM call."""
    call_id: str
    call_type: str  # e.g., 'indicator_summary', 'chart_analysis', 'final_decision'
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    timestamp: float
    model: str
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class AnalysisTokenTracker:
    """Tracks token usage across all LLM calls within a single analysis."""
    analysis_id: str
    symbol: str
    start_time: float = field(default_factory=time.time)
    token_usage: List[TokenUsage] = field(default_factory=list)
    
    def add_token_usage(self, call_type: str, response, model: str = "gemini-2.5-flash", 
                       success: bool = True, error_message: Optional[str] = None) -> str:
        """
        Add token usage from an LLM response.
        
        Args:
            call_type: Type of LLM call (e.g., 'indicator_summary', 'chart_analysis')
            response: The response object from Gemini API
            model: The model used for the call
            success: Whether the call was successful
            error_message: Error message if call failed
            
        Returns:
            str: Call ID for reference
        """
        call_id = f"{call_type}_{len(self.token_usage)}_{int(time.time())}"
        
        # Extract token usage from response
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        if response is not None:
            # Handle different response formats
            if hasattr(response, 'usage_metadata'):
                # Real Gemini API response object
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                completion_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                total_tokens = getattr(usage, 'total_token_count', 0) or 0
                
            elif isinstance(response, dict) and 'usage_metadata' in response:
                # Dictionary-style response (for testing)
                usage = response['usage_metadata']
                if isinstance(usage, dict):
                    prompt_tokens = usage.get('prompt_token_count', 0) or 0
                    completion_tokens = usage.get('candidates_token_count', 0) or 0
                    total_tokens = usage.get('total_token_count', 0) or 0
                else:
                    # Handle case where usage_metadata is an object
                    prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                    completion_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                    total_tokens = getattr(usage, 'total_token_count', 0) or 0
            
            elif isinstance(response, dict) and 'prompt_token_count' in response:
                # Direct token counts in response
                prompt_tokens = response.get('prompt_token_count', 0) or 0
                completion_tokens = response.get('candidates_token_count', 0) or 0
                total_tokens = response.get('total_token_count', 0) or 0
        
        # Validate token counts
        if total_tokens > 0 and prompt_tokens + completion_tokens != total_tokens:
            # Log the mismatch but use the reported total
            print(f"⚠️ Token count mismatch in {call_type}: prompt({prompt_tokens}) + completion({completion_tokens}) != total({total_tokens})")
            print(f"   Using reported total: {total_tokens}")
        
        token_usage = TokenUsage(
            call_id=call_id,
            call_type=call_type,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            timestamp=time.time(),
            model=model,
            success=success,
            error_message=error_message
        )
        
        self.token_usage.append(token_usage)
        return call_id
    
    def get_total_usage(self) -> Dict[str, int]:
        """Get total token usage across all calls."""
        total_prompt = sum(usage.prompt_tokens for usage in self.token_usage)
        total_completion = sum(usage.completion_tokens for usage in self.token_usage)
        total_tokens = sum(usage.total_tokens for usage in self.token_usage)
        
        # Calculate expected total for validation
        calculated_total = total_prompt + total_completion
        
        return {
            'total_input_tokens': total_prompt,
            'total_output_tokens': total_completion,
            'total_tokens': total_tokens,
            'calculated_total': calculated_total,
            'token_mismatch': total_tokens - calculated_total if total_tokens > 0 else 0,
            'llm_calls_count': len(self.token_usage)
        }
    
    def get_usage_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of token usage by call type."""
        breakdown = {}
        
        for usage in self.token_usage:
            call_type = usage.call_type
            if call_type not in breakdown:
                breakdown[call_type] = {
                    'calls': 0,
                    'total_input_tokens': 0,
                    'total_output_tokens': 0,
                    'total_tokens': 0,
                    'successful_calls': 0,
                    'failed_calls': 0,
                    'call_details': []
                }
            
            breakdown[call_type]['calls'] += 1
            breakdown[call_type]['total_input_tokens'] += usage.prompt_tokens
            breakdown[call_type]['total_output_tokens'] += usage.completion_tokens
            breakdown[call_type]['total_tokens'] += usage.total_tokens
            
            if usage.success:
                breakdown[call_type]['successful_calls'] += 1
            else:
                breakdown[call_type]['failed_calls'] += 1
            
            breakdown[call_type]['call_details'].append({
                'call_id': usage.call_id,
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens,
                'timestamp': usage.timestamp,
                'model': usage.model,
                'success': usage.success,
                'error_message': usage.error_message
            })
        
        return breakdown
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis token usage."""
        total_usage = self.get_total_usage()
        breakdown = self.get_usage_breakdown()
        
        return {
            'analysis_id': self.analysis_id,
            'symbol': self.symbol,
            'start_time': self.start_time,
            'end_time': time.time(),
            'duration_seconds': time.time() - self.start_time,
            **total_usage,
            'usage_breakdown': breakdown,
            'call_types': list(breakdown.keys()),
            'total_calls': len(self.token_usage),
            'successful_calls': sum(1 for usage in self.token_usage if usage.success),
            'failed_calls': sum(1 for usage in self.token_usage if not usage.success)
        }
    
    def print_summary(self):
        """Print a human-readable summary of token usage."""
        summary = self.get_summary()
        
        print(f"\n📊 Token Usage Summary for {summary['symbol']}")
        print("=" * 50)
        print(f"Analysis ID: {summary['analysis_id']}")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"Total LLM Calls: {summary['total_calls']}")
        print(f"Successful Calls: {summary['successful_calls']}")
        print(f"Failed Calls: {summary['failed_calls']}")
        print(f"\nToken Usage:")
        print(f"  Input Tokens: {summary['total_input_tokens']:,}")
        print(f"  Output Tokens: {summary['total_output_tokens']:,}")
        print(f"  Total Tokens: {summary['total_tokens']:,}")
        
        # Show token validation
        if summary['total_tokens'] > 0:
            calculated_total = summary['total_input_tokens'] + summary['total_output_tokens']
            mismatch = summary['total_tokens'] - calculated_total
            print(f"  Calculated Total: {calculated_total:,}")
            if mismatch != 0:
                print(f"  ⚠️ Token Mismatch: {mismatch:,} tokens")
                print(f"     (This may include system tokens or overhead)")
        
        print(f"\nBreakdown by Call Type:")
        for call_type, details in summary['usage_breakdown'].items():
            print(f"  {call_type}:")
            print(f"    Calls: {details['calls']}")
            print(f"    Input Tokens: {details['total_input_tokens']:,}")
            print(f"    Output Tokens: {details['total_output_tokens']:,}")
            print(f"    Total Tokens: {details['total_tokens']:,}")
            print(f"    Success Rate: {details['successful_calls']}/{details['calls']}")


# Global tracker registry
_analysis_trackers: Dict[str, AnalysisTokenTracker] = {}


def get_or_create_tracker(analysis_id: str, symbol: str) -> AnalysisTokenTracker:
    """Get or create a token tracker for an analysis."""
    if analysis_id not in _analysis_trackers:
        _analysis_trackers[analysis_id] = AnalysisTokenTracker(analysis_id, symbol)
    return _analysis_trackers[analysis_id]


def get_tracker(analysis_id: str) -> Optional[AnalysisTokenTracker]:
    """Get a token tracker by analysis ID."""
    return _analysis_trackers.get(analysis_id)


def remove_tracker(analysis_id: str):
    """Remove a token tracker from the registry."""
    if analysis_id in _analysis_trackers:
        del _analysis_trackers[analysis_id]


def get_all_trackers() -> Dict[str, AnalysisTokenTracker]:
    """Get all active token trackers."""
    return _analysis_trackers.copy() 