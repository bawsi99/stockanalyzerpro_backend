# Token Tracking Implementation

## Overview

This document describes the implementation of token usage tracking for LLM calls within the stock analysis system. The system tracks total input tokens, output tokens, and total tokens for each analysis, providing detailed breakdowns by call type and storing this data in the database.

## Architecture

### Core Components

1. **TokenTracker** (`backend/gemini/token_tracker.py`)
   - Manages token usage tracking per analysis
   - Provides detailed breakdowns and summaries
   - Handles multiple concurrent analyses

2. **Enhanced GeminiCore** (`backend/gemini/gemini_core.py`)
   - Modified to capture token usage from API responses
   - Supports token tracking for all LLM call types

3. **Enhanced GeminiClient** (`backend/gemini/gemini_client.py`)
   - Integrates token tracking with analysis workflow
   - Passes token tracker to all LLM calls

4. **Database Integration** (`backend/database_manager.py`)
   - Stores token usage data in `stock_analyses` table
   - Tracks total tokens, input/output breakdown, and call counts

## Database Schema

The `stock_analyses` table has been enhanced with token tracking fields:

```sql
-- Token tracking fields added to stock_analyses table
total_input_tokens INTEGER DEFAULT 0,
total_output_tokens INTEGER DEFAULT 0,
total_tokens INTEGER DEFAULT 0,
llm_calls_count INTEGER DEFAULT 0,
token_usage_breakdown JSONB DEFAULT '{}',
```

## Token Tracking Flow

### 1. Analysis Initialization
```python
# Generate unique analysis ID
analysis_id = str(uuid.uuid4())

# Create token tracker for this analysis
token_tracker = get_or_create_tracker(analysis_id, symbol)
```

### 2. LLM Call Tracking
Each LLM call is tracked with:
- Call type (e.g., "indicator_summary", "chart_analysis", "final_decision")
- Input tokens (prompt tokens)
- Output tokens (completion tokens)
- Total tokens
- Success/failure status
- Timestamp

### 3. Token Usage Accumulation
```python
# Extract token usage from Gemini API response
if hasattr(response, 'usage_metadata'):
    usage = response.usage_metadata
    prompt_tokens = getattr(usage, 'prompt_token_count', 0)
    completion_tokens = getattr(usage, 'candidates_token_count', 0)
    total_tokens = getattr(usage, 'total_token_count', 0)
```

### 4. Analysis Completion
```python
# Get comprehensive token usage summary
token_summary = token_tracker.get_summary()

# Add to analysis result
result['analysis_metadata']['token_usage'] = token_summary

# Store in database
analysis_data = {
    "total_input_tokens": token_usage.get("total_input_tokens", 0),
    "total_output_tokens": token_usage.get("total_output_tokens", 0),
    "total_tokens": token_usage.get("total_tokens", 0),
    "llm_calls_count": token_usage.get("llm_calls_count", 0),
    "token_usage_breakdown": token_usage.get("usage_breakdown", {})
}
```

## Supported LLM Call Types

The system tracks tokens for all LLM calls within an analysis:

1. **indicator_summary** - Technical indicators analysis
2. **comprehensive_overview** - Chart overview analysis
3. **volume_analysis** - Volume pattern analysis
4. **reversal_patterns** - Reversal pattern detection
5. **continuation_levels** - Continuation patterns and levels
6. **final_decision** - Final analysis and recommendations

## Token Usage Data Structure

### Individual Call Data
```python
@dataclass
class TokenUsage:
    call_id: str
    call_type: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    timestamp: float
    model: str
    success: bool
    error_message: Optional[str]
```

### Analysis Summary
```python
{
    "analysis_id": "uuid",
    "symbol": "RELIANCE",
    "start_time": 1234567890.123,
    "end_time": 1234567895.456,
    "duration_seconds": 5.333,
    "total_input_tokens": 5300,
    "total_output_tokens": 3000,
    "total_tokens": 8300,
    "llm_calls_count": 6,
    "usage_breakdown": {
        "indicator_summary": {
            "calls": 1,
            "total_input_tokens": 1500,
            "total_output_tokens": 800,
            "total_tokens": 2300,
            "successful_calls": 1,
            "failed_calls": 0,
            "call_details": [...]
        },
        # ... other call types
    },
    "call_types": ["indicator_summary", "chart_analysis", "final_decision"],
    "total_calls": 6,
    "successful_calls": 6,
    "failed_calls": 0
}
```

## Usage Examples

### Basic Token Tracking
```python
from gemini.token_tracker import get_or_create_tracker

# Create tracker for analysis
tracker = get_or_create_tracker("analysis_123", "RELIANCE")

# Track LLM call
call_id = tracker.add_token_usage("indicator_summary", response, "gemini-2.5-flash")

# Get summary
summary = tracker.get_summary()
print(f"Total tokens: {summary['total_tokens']:,}")
```

### Multiple Concurrent Analyses
```python
# Each analysis gets its own tracker
tracker1 = get_or_create_tracker("analysis_1", "RELIANCE")
tracker2 = get_or_create_tracker("analysis_2", "TCS")

# Token usage is tracked separately
tracker1.add_token_usage("indicator_summary", response1, "gemini-2.5-flash")
tracker2.add_token_usage("indicator_summary", response2, "gemini-2.5-flash")

# Each tracker maintains its own totals
print(f"RELIANCE tokens: {tracker1.get_total_usage()['total_tokens']}")
print(f"TCS tokens: {tracker2.get_total_usage()['total_tokens']}")
```

### Human-Readable Summary
```python
# Print detailed summary
tracker.print_summary()

# Output:
# 📊 Token Usage Summary for RELIANCE
# ==================================================
# Analysis ID: analysis_123
# Duration: 5.33 seconds
# Total LLM Calls: 6
# Successful Calls: 6
# Failed Calls: 0
# 
# Token Usage:
#   Input Tokens: 5,300
#   Output Tokens: 3,000
#   Total Tokens: 8,300
# 
# Breakdown by Call Type:
#   indicator_summary:
#     Calls: 1
#     Input Tokens: 1,500
#     Output Tokens: 800
#     Total Tokens: 2,300
#     Success Rate: 1/1
```

## Integration Points

### 1. Gemini Core Integration
All LLM call methods in `GeminiCore` now support token tracking:
- `call_llm()`
- `call_llm_with_code_execution()`
- `call_llm_with_image()`
- `call_llm_with_images()`

### 2. Gemini Client Integration
The `GeminiClient.analyze_stock()` method:
- Creates a token tracker for each analysis
- Passes the tracker to all LLM calls
- Includes token usage in the final result
- Cleans up the tracker after completion

### 3. Database Integration
The `DatabaseManager.store_analysis()` method:
- Extracts token usage from analysis metadata
- Stores token data in dedicated database fields
- Maintains backward compatibility

## Error Handling

### Failed LLM Calls
```python
# Track failed calls with error information
tracker.add_token_usage(
    "indicator_summary", 
    None, 
    "gemini-2.5-flash", 
    success=False, 
    error_message="API timeout"
)
```

### Token Extraction Failures
```python
# Graceful handling when token data is unavailable
if hasattr(response, 'usage_metadata'):
    # Extract tokens
else:
    # Use default values (0 tokens)
```

## Testing

### Unit Tests
Run the token tracking tests:
```bash
cd backend
python test_token_tracking_simple.py
```

### Test Coverage
- Token tracker creation and management
- Token usage accumulation
- Multiple concurrent analyses
- Success/failure tracking
- Cleanup and memory management

## Performance Considerations

### Memory Management
- Token trackers are automatically cleaned up after analysis completion
- Global registry prevents memory leaks
- Each analysis gets a unique tracker instance

### Concurrent Access
- Thread-safe tracker registry
- Each analysis has isolated token tracking
- No cross-contamination between analyses

## Monitoring and Analytics

### Token Usage Metrics
- Total tokens per analysis
- Tokens per call type
- Success/failure rates
- Analysis duration vs token usage

### Cost Analysis
- Input tokens (prompt cost)
- Output tokens (completion cost)
- Total cost per analysis
- Cost breakdown by analysis type

## Future Enhancements

### Potential Improvements
1. **Real-time Monitoring** - Live token usage dashboard
2. **Cost Optimization** - Token usage optimization suggestions
3. **Rate Limiting** - Token-based rate limiting
4. **Budget Management** - Per-user token budgets
5. **Analytics Dashboard** - Token usage analytics and reporting

### API Extensions
1. **Token Usage API** - Endpoint to retrieve token usage data
2. **Cost Estimation** - Pre-analysis token cost estimation
3. **Usage Alerts** - Notifications for high token usage

## Conclusion

The token tracking system provides comprehensive monitoring of LLM usage across all analyses. It enables:

- **Cost Management** - Track and optimize token usage
- **Performance Monitoring** - Monitor analysis efficiency
- **Debugging** - Identify high-token usage patterns
- **Analytics** - Understand analysis complexity and costs

The implementation is production-ready and provides a solid foundation for token usage management and optimization. 