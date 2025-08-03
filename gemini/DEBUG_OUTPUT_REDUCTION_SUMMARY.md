# Gemini Debug Output Reduction Summary

## Overview
This document summarizes the changes made to reduce verbose debug output from the Gemini module while showing essential time and token information.

## Changes Made

### 1. Debug Configuration (`debug_config.py`)
- **Default Debug Enabled**: Changed to `True` (to show time and tokens)
- **Default Log to File**: Changed from `True` to `False`
- **Default Log Level**: Changed to `DEBUG` (to show time and tokens)
- **Max Prompt Log Length**: Reduced from `1000` to `500` characters
- **Max Response Log Length**: Reduced from `2000` to `1000` characters

### 2. Debug Logger (`debug_logger.py`)

#### `log_api_request()` Method
- **Commented out verbose logging**:
  - Banner lines (`=` * 80)
  - Timestamp, method, model, code execution details
  - Image information and details
  - Prompt length and preview
  - Full prompt logging
- **Added time and token logging**: Shows timestamp, method, model, and estimated tokens at DEBUG level

#### `log_api_response()` Method
- **Commented out verbose logging**:
  - Response banners and separators
  - Response time, method details
  - Response structure information
  - Candidate and part details
  - Text response length and preview
  - Full response logging
  - Code execution results
- **Added time and token logging**: Shows timestamp, method, response time, and actual token counts at DEBUG level

#### `log_error()` Method
- **Commented out verbose logging**:
  - Error banners and separators
  - Detailed error context and type
  - Prompt that caused error
  - Full traceback
- **Added minimal logging**: Only logs error type and message

#### `log_processing_step()` Method
- **Commented out verbose logging**:
  - Step details with emojis
  - Detailed information
- **Added minimal logging**: Only logs step name at DEBUG level

#### `log_json_parsing()` Method
- **Commented out verbose logging**:
  - Success/failure banners
  - Parsed data keys
  - JSON string content
  - Detailed error information
- **Added minimal logging**: Only logs success/failure status

## Current Debug Output Format

### API Request Logging
```
[HH:MM:SS] Gemini API call: method_name (model_name) - ~estimated_tokens tokens
```

### API Response Logging
```
[HH:MM:SS] Gemini API response: method_name (response_time) - prompt_tokens+completion_tokens=total_tokens tokens
```

## Benefits

1. **Essential Information**: Shows time and token usage for monitoring API costs
2. **Clean Output**: No verbose banners or detailed content logging
3. **Performance Monitoring**: Easy to track response times
4. **Cost Tracking**: Token usage is clearly displayed
5. **Configurable**: Can still be disabled when not needed

## How to Disable Debug (When Not Needed)

### Method 1: Environment Variables
```bash
export GEMINI_DEBUG=false
export GEMINI_LOG_LEVEL=WARNING
```

### Method 2: Python Code
```python
from gemini.debug_config import disable_gemini_debug, set_gemini_log_level

disable_gemini_debug()
set_gemini_log_level("WARNING")
```

### Method 3: Direct Configuration
```python
from gemini.debug_config import config

config.disable()
config.set_log_level("WARNING")
```

## Current Default Behavior

- **Debug Enabled**: Shows time and token information by default
- **Log Level**: DEBUG (shows time and token details)
- **File Logging**: Disabled
- **Console Output**: Clean, focused on time and tokens

## Testing

The changes have been tested to ensure:
- Debug is enabled by default
- Log level is set to DEBUG
- Time stamps are displayed in HH:MM:SS format
- Token information is shown for both requests and responses
- Verbose logging is commented out but preserved for future use

## Example Output

```
23:46:49 - DEBUG - [23:46:49] Gemini API call: call_llm (gemini-2.5-flash) - ~150 tokens
23:46:52 - DEBUG - [23:46:52] Gemini API response: call_llm (2.500s) - 150+75=225 tokens
```

## Files Modified

1. `backend/gemini/debug_config.py` - Updated default configuration
2. `backend/gemini/debug_logger.py` - Modified to show time and tokens
3. `backend/gemini/DEBUG_OUTPUT_REDUCTION_SUMMARY.md` - This summary document 