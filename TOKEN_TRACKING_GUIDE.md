# Model-Based Token Tracking System

## Overview

This document describes the comprehensive token tracking system implemented for the StockAnalyzer Pro backend. The system provides **model-based token tracking** that can separately track usage for different LLM models (e.g., `gemini-2.5-flash` vs `gemini-2.5-pro`) across different agents.

## Key Features

✅ **Model-Based Tracking**: Separate tracking for different models (flash vs pro)  
✅ **Agent-Specific Analytics**: Track token usage per agent  
✅ **Provider-Agnostic Design**: Works with Gemini, OpenAI, Claude (extensible)  
✅ **Thread-Safe**: Safe for concurrent requests  
✅ **Cost Analysis**: Built-in cost analysis capabilities  
✅ **Efficiency Comparisons**: Compare model efficiency  
✅ **Multiple Response Formats**: Handles different API response structures  
✅ **Real-time Integration**: Integrated into analysis service for live tracking  

## Architecture

```
backend/llm/
├── token_counter.py           # Main token counting system
├── providers/
│   ├── gemini.py             # Updated with token tracking
│   └── base.py               # Updated interface
├── client.py                 # Updated LLM client
└── key_manager.py            # Existing key management

backend/services/
└── analysis_service.py       # Integrated token tracking
```

## Example Output

When you run an analysis, you'll see output like this:

```
================================================================================
📊 TOKEN USAGE SUMMARY for AAPL
================================================================================
Total Analysis Time: 45.23s
Total LLM Calls: 8
Total Input Tokens: 2,650
Total Output Tokens: 1,325
Total Tokens: 3,975

📱 Model Usage:
  gemini-2.5-flash     | 1,275 tokens |  5 calls | agents: indicator_agent, volume_agent, mtf_agent
  gemini-2.5-pro       | 2,700 tokens |  3 calls | agents: final_decision_agent, sector_agent, risk_agent

🤖 Agent Usage:
  final_decision_agent |    750 tokens |  1 calls
  volume_agent         |    600 tokens |  2 calls
  indicator_agent      |    450 tokens |  1 calls
  sector_agent         |    525 tokens |  1 calls
  risk_agent           |    450 tokens |  1 calls
================================================================================
```

## Usage Examples

### 1. Basic Token Tracking

```python
from llm.token_counter import track_llm_usage

# Track token usage from any LLM response
token_data = track_llm_usage(
    response=llm_response,
    agent_name="indicator_agent", 
    provider="gemini",
    model="gemini-2.5-flash",
    duration_ms=1250.0,
    success=True
)

print(f"Used {token_data.total_tokens} tokens")
```

### 2. Get Analytics

```python
from llm.token_counter import get_token_usage_summary, get_model_usage_summary

# Get comprehensive summary
summary = get_token_usage_summary()
print(f"Total tokens: {summary['total_usage']['total_tokens']:,}")

# Get model-specific breakdown
models = get_model_usage_summary()
for model, usage in models.items():
    print(f"{model}: {usage['total_tokens']:,} tokens, {usage['calls']} calls")
```

### 3. Compare Models

```python
from llm.token_counter import compare_model_efficiency

comparison = compare_model_efficiency("gemini-2.5-flash", "gemini-2.5-pro")
if 'error' not in comparison:
    comp = comparison['comparison']
    print(f"Flash avg: {comp['avg_tokens_per_call']['gemini-2.5-flash']:.0f}")
    print(f"Pro avg: {comp['avg_tokens_per_call']['gemini-2.5-pro']:.0f}")
```

### 4. LLM Client Integration

```python
from llm.client import LLMClient

client = LLMClient(agent_name="indicator_agent", model="gemini-2.5-flash")

# Get response with token tracking
text_response, token_usage = await client.generate(
    prompt="Analyze this technical indicator...",
    return_token_usage=True
)

print(f"Response: {text_response}")
print(f"Tokens used: {token_usage.total_tokens if token_usage else 'Unknown'}")
```

## API Endpoints

The system includes REST API endpoints for token analytics:

### GET `/analytics/tokens`
Get comprehensive token usage analytics
```json
{
  "success": true,
  "analytics": {
    "summary": {...},
    "model_breakdown": {...},
    "agent_model_combinations": {...}
  }
}
```

### GET `/analytics/tokens/models`
Get model-specific analytics
```json
{
  "success": true,
  "model_usage": {
    "gemini-2.5-flash": {
      "total_tokens": 1275,
      "calls": 5,
      "agents_using_model": ["indicator_agent", "volume_agent"]
    }
  }
}
```

### POST `/analytics/tokens/compare`
Compare two models
```json
{
  "models": ["gemini-2.5-flash", "gemini-2.5-pro"]
}
```

### POST `/analytics/tokens/reset`
Reset token usage analytics

## Model-Based Tracking Details

The system automatically tracks:

1. **Different Models**: `gemini-2.5-flash`, `gemini-2.5-pro`, etc.
2. **Agent Usage**: Which agents use which models
3. **Token Efficiency**: Average tokens per call by model
4. **Cost Analysis**: Estimated costs based on token usage
5. **Timing**: Request duration tracking
6. **Success Rates**: Failed vs successful calls

### Example Scenario

Consider this realistic usage pattern:

- **indicator_agent**: Uses `gemini-2.5-flash` for fast analysis (avg 200 tokens)
- **final_decision_agent**: Uses `gemini-2.5-pro` for complex synthesis (avg 600 tokens)  
- **volume_agent**: Uses `gemini-2.5-flash` for pattern detection (avg 180 tokens)
- **sector_agent**: Uses `gemini-2.5-pro` for sector analysis (avg 450 tokens)

The system will track each separately and show:
- Total flash usage: 380 tokens (2 calls)
- Total pro usage: 1,050 tokens (2 calls) 
- Pro uses 176% more tokens per call than flash
- Estimated cost difference

## Response Format Support

The system handles multiple Gemini API response formats:

### New Format (usageMetadata)
```json
{
  "usageMetadata": {
    "promptTokenCount": 150,
    "candidatesTokenCount": 75, 
    "totalTokenCount": 225
  }
}
```

### Legacy Format (usage_metadata)
```json
{
  "usage_metadata": {
    "prompt_token_count": 150,
    "candidates_token_count": 75,
    "total_token_count": 225
  }
}
```

### Direct Fields
```json
{
  "promptTokenCount": 150,
  "candidatesTokenCount": 75,
  "totalTokenCount": 225
}
```

## Integration with Analysis Service

The token tracking is automatically integrated into the analysis service. When you run an enhanced analysis (`/analyze/enhanced`), the system will:

1. **Reset counter** at the start of each analysis
2. **Track all LLM calls** made by various agents
3. **Display summary** at the end with model breakdown
4. **Include in response** metadata with token usage info

## Advanced Features

### 1. Agent-Model Combinations
See which agents use which models:
```python
from llm.token_counter import get_agent_model_combinations

combos = get_agent_model_combinations()
for agent, models in combos.items():
    print(f"{agent}:")
    for model, usage in models.items():
        print(f"  {model}: {usage['total_tokens']} tokens")
```

### 2. Timing Analysis
Track request durations:
```python
summary = get_token_usage_summary()
timing = summary['timing_stats']
print(f"Average duration: {timing['avg_duration_ms']:.2f}ms")
```

### 3. Cost Estimation
Built-in cost analysis with hypothetical rates:
```python
# Flash: $0.075 per 1K tokens
# Pro: $0.30 per 1K tokens
# Shows total estimated costs per model
```

## Error Handling

The system gracefully handles:
- ✅ Invalid/malformed responses
- ✅ Missing token data 
- ✅ Zero token responses
- ✅ API failures
- ✅ Concurrent access
- ✅ Different response formats

## Testing

Run the comprehensive test suite:

```bash
cd backend
python test_token_counter_standalone.py
```

This tests:
- Response format parsing
- Model-based tracking scenarios
- Analytics generation
- Edge cases and error handling

## Benefits

1. **Cost Monitoring**: Track actual token usage and costs per model
2. **Performance Optimization**: Identify which agents/models are most efficient
3. **Resource Planning**: Plan API usage and costs based on historical data
4. **Model Comparison**: Compare efficiency between flash and pro models
5. **Debugging**: Track which agents are using the most tokens
6. **Scalability**: Monitor token usage as you scale up analyses

## Configuration

The system requires no additional configuration and works with your existing:
- LLM API keys (managed by `key_manager.py`)
- Agent configurations (in `llm/config/llm_assignments.yaml`)
- Model assignments per agent

## Future Extensions

The architecture supports easy extension for:
- OpenAI token tracking (GPT-4, GPT-3.5, etc.)
- Claude token tracking (Claude-3, etc.)
- Custom token extractors for new providers
- Database persistence of token usage
- Advanced reporting and visualization

## Conclusion

This model-based token tracking system provides comprehensive visibility into your LLM usage patterns, enabling you to optimize costs, performance, and resource allocation across different models and agents. It automatically tracks `gemini-2.5-flash` vs `gemini-2.5-pro` usage separately and provides detailed analytics to help you make informed decisions about model usage.