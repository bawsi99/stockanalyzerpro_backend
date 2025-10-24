# MTF LLM Agent Integration

## Overview

The MTF LLM Agent is now fully integrated into the analysis pipeline, providing natural language insights from multi-timeframe technical analysis. Similar to how indicator agents work, this agent sends MTF data to Gemini for interpretation.

## Architecture

```
Enhanced Analysis Request
    │
    ├── Fetch Stock Data
    ├── Calculate Indicators
    │
    └── Launch Parallel Tasks:
        │
        ├── Volume Agents ────────────┐
        │                              │
        ├── MTF Analysis               │
        │   ├── 1. Technical MTF      │
        │   │   (CoreMTFProcessor)    │
        │   │                          │
        │   └── 2. MTF LLM Agent ─────┤
        │       (Gemini Analysis)      │
        │                              │
        ├── Advanced Analysis ─────────┤
        ├── Sector Analysis ───────────┤
        └── Indicator Summary ─────────┤
                                       │
                                       ▼
                            Final Decision LLM
                                       │
                                       ▼
                                  Response
```

## Flow

### 1. Technical MTF Analysis
- **CoreMTFProcessor** analyzes data across 6 timeframes:
  - 1min, 5min, 15min (intraday)
  - 30min, 1hour (swing)
  - 1day (position)
- Calculates technical indicators per timeframe
- Performs cross-timeframe validation
- Detects divergences and conflicts
- Returns structured technical data

### 2. MTF LLM Agent
- **MTFLLMAgent** receives technical MTF analysis
- Formats data into comprehensive prompt
- Sends to Gemini with code execution capability
- Returns natural language insights including:
  - Trend synthesis across timeframes
  - Key support/resistance levels
  - Conflict resolution strategies
  - Trading recommendations per style (intraday/swing/position)
  - Risk assessment
  - Entry/exit suggestions with specific price levels

### 3. Combined Result
- Technical MTF data + LLM insights are merged
- Passed to final decision LLM as part of knowledge context
- Final LLM uses both for comprehensive analysis

## Files

### New Files
- **`agents/mtf_analysis/mtf_llm_agent.py`**: Main MTF LLM agent implementation
  - `MTFLLMAgent` class
  - `analyze_mtf_with_llm()` method
  - Prompt building and response structuring
  - Global instance: `mtf_llm_agent`

### Modified Files
- **`services/analysis_service.py`**: 
  - Updated `_mtf()` function to call both technical and LLM agents
  - Added debug logging for MTF LLM calls
  
- **`agents/mtf_analysis/__init__.py`**:
  - Exported `MTFLLMAgent` and `mtf_llm_agent`

- **`agents/mtf_analysis/mtf_agents.py`**:
  - Disabled placeholder agents (intraday, swing, position)
  - Only core processor is used

## Debug Logging

The integration includes comprehensive debug logging:

```python
[MTF_DEBUG] Starting MTF analysis for {symbol}...
[MTF_DEBUG] MTF technical analysis complete for {symbol}. Success: True
[MTF_DEBUG] Calling MTF LLM agent for {symbol}...
[MTF_LLM_AGENT_DEBUG] Starting MTF LLM analysis for {symbol}...
[MTF_LLM_AGENT_DEBUG] Sending request to Gemini API for {symbol}...
[MTF_LLM_AGENT_DEBUG] Received response from Gemini API for {symbol} in 3.45s
[MTF_LLM_AGENT_DEBUG] Finished MTF LLM analysis for {symbol}. Success: True
[MTF_DEBUG] MTF LLM insights added to result for {symbol}
[MTF_DEBUG] Finished complete MTF analysis for {symbol}
```

## Response Structure

The combined MTF response includes:

```json
{
  "success": true,
  "symbol": "RELIANCE",
  "timeframe_analyses": { ... },
  "cross_timeframe_validation": { ... },
  "summary": { ... },
  "llm_insights": {
    "success": true,
    "agent": "mtf_llm_agent",
    "llm_analysis": "Natural language analysis from Gemini...",
    "processing_time": 4.23,
    "llm_processing_time": 3.45,
    "confidence": 0.85,
    "code_executions": 0,
    "mtf_summary": { ... },
    "cross_timeframe_validation": { ... }
  }
}
```

## Usage Example

The MTF LLM agent is automatically called during enhanced analysis:

```python
POST /analyze/enhanced
{
  "stock": "RELIANCE",
  "exchange": "NSE",
  "period": 365,
  "interval": "day"
}
```

The response will include MTF analysis with both technical data and LLM insights.

## Error Handling

- If technical MTF fails → entire MTF returns empty
- If LLM agent fails → technical MTF still returned (without LLM insights)
- Graceful degradation ensures analysis continues even if MTF LLM fails
- All errors logged with full stack traces

## Performance

Typical timing:
- Technical MTF: 2-3 seconds
- LLM Analysis: 3-5 seconds
- **Total MTF: 5-8 seconds** (runs in parallel with other agents)

## Benefits

1. **Natural Language Insights**: Technical MTF data is now interpreted by LLM
2. **Conflict Resolution**: LLM explains how to handle divergences across timeframes
3. **Actionable Recommendations**: Specific entry/exit levels, not just signals
4. **Trading Style Specificity**: Separate recommendations for intraday/swing/position traders
5. **Enhanced Final Decision**: Final LLM has both technical data and expert interpretation

## Testing

To test the MTF LLM agent separately, you can use the existing test tool:

```bash
python backend/agents/mtf_analysis/prompt_testing/mtf_comprehensive/multi_stock_test.py \
  --symbols RELIANCE,TCS --exchange NSE --call-llm
```

This will show exactly what prompt is sent to the LLM and what response is received.

## Future Enhancements

- [ ] Add caching for frequently analyzed symbols
- [ ] Enable/disable MTF LLM via configuration flag
- [ ] Add MTF LLM health monitoring
- [ ] Optimize prompt length for faster response
- [ ] Add specialized prompts for different market conditions