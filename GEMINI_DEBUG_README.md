# Gemini API Debugging System

This system provides comprehensive debugging and logging for all Gemini API calls in the trading application. It logs detailed information about requests, responses, and execution flow to help you understand and troubleshoot API interactions.

## Features

- **Detailed API Request Logging**: Logs prompts, model parameters, and request configuration
- **Comprehensive Response Analysis**: Extracts and logs text responses, code execution results, and execution outputs
- **Error Tracking**: Detailed error logging with context and stack traces
- **JSON Parsing Debugging**: Tracks JSON parsing attempts and fixes
- **Performance Monitoring**: Response time tracking for all API calls
- **Flexible Configuration**: Environment variable-based configuration
- **File and Console Logging**: Logs to both console and timestamped files
- **Image Analysis Support**: Logs multi-modal requests with image information

## Quick Start

### 1. Enable Debugging

```bash
# Enable debugging (default)
python gemini_debug_cli.py enable

# Or set environment variable
export GEMINI_DEBUG=true
```

### 2. Check Status

```bash
python gemini_debug_cli.py status
```

### 3. Run a Test

```bash
python gemini_debug_cli.py test
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_DEBUG` | Enable/disable debugging | `true` |
| `GEMINI_LOG_TO_FILE` | Enable/disable file logging | `true` |
| `GEMINI_LOG_LEVEL` | Log level (DEBUG/INFO/WARNING/ERROR) | `INFO` |
| `GEMINI_MAX_PROMPT_LOG` | Maximum prompt length to log | `1000` |
| `GEMINI_MAX_RESPONSE_LOG` | Maximum response length to log | `2000` |

### CLI Commands

```bash
# Enable debugging
python gemini_debug_cli.py enable

# Disable debugging
python gemini_debug_cli.py disable

# Set log level
python gemini_debug_cli.py level DEBUG

# Show current status
python gemini_debug_cli.py status

# Show environment variables
python gemini_debug_cli.py env

# Run test
python gemini_debug_cli.py test
```

## What Gets Logged

### API Requests
- Timestamp and method name
- Model being used (e.g., gemini-2.5-flash)
- Code execution settings
- Prompt length and preview
- Image information (for multi-modal requests)
- Full prompt (if within length limits)

### API Responses
- Response time
- Response structure analysis
- Text response length and preview
- Code execution results
- Execution outputs
- Full response (if within length limits)

### Error Information
- Error type and message
- Context where error occurred
- Prompt that caused the error
- Full stack trace

### JSON Parsing
- JSON extraction attempts
- Parsing success/failure
- JSON fixing attempts
- Fallback JSON creation

## Log Output Format

### Console Output
```
14:30:15 - INFO - ================================================================================
14:30:15 - INFO - 🚀 GEMINI API REQUEST
14:30:15 - INFO - ================================================================================
14:30:15 - INFO - 📅 Timestamp: 2024-01-15T14:30:15.123456
14:30:15 - INFO - 🔧 Method: call_llm_with_code_execution
14:30:15 - INFO - 🤖 Model: gemini-2.5-flash
14:30:15 - INFO - ⚙️ Code Execution: True
14:30:15 - INFO - 📝 Prompt Length: 1250 characters
14:30:15 - INFO - 📝 Prompt Preview: Analyze the following stock data...
14:30:15 - INFO - --------------------------------------------------------------------------------
14:30:18 - INFO - 📥 GEMINI API RESPONSE
14:30:18 - INFO - --------------------------------------------------------------------------------
14:30:18 - INFO - ⏱️ Response Time: 3.245 seconds
14:30:18 - INFO - 🔧 Method: call_llm_with_code_execution
14:30:18 - INFO - 📊 Response Type: GenerateContentResponse
14:30:18 - INFO - 📋 Candidates: 1
14:30:18 - INFO - 📄 Parts: 3
14:30:18 - INFO - 📝 Text Response Length: 850 characters
14:30:18 - INFO - 💻 Code Results: 2 snippet(s)
14:30:18 - INFO - ⚡ Execution Results: 2 output(s)
14:30:18 - INFO - ================================================================================
```

### File Logging
Logs are saved to `backend/logs/gemini_debug_YYYYMMDD_HHMMSS.log` with full timestamps and detailed information.

## Integration Points

The debugging system is integrated into:

1. **GeminiCore** (`gemini_core.py`)
   - `call_llm()` - Basic text requests
   - `call_llm_with_code_execution()` - Code execution requests
   - `call_llm_with_image()` - Single image requests
   - `call_llm_with_images()` - Multiple image requests

2. **GeminiClient** (`gemini_client.py`)
   - `build_indicators_summary()` - Indicator analysis
   - `analyze_stock()` - Stock analysis
   - `extract_markdown_and_json()` - JSON parsing
   - All image analysis methods

## Example Usage

### Basic Text Request
```python
from gemini.gemini_client import GeminiClient

client = GeminiClient(api_key)
response = await client.build_indicators_summary(
    symbol="AAPL",
    indicators={"rsi": 65, "price": 150.00},
    period=14,
    interval="1d"
)
```

### Code Execution Request
```python
response = await client.analyze_stock_with_enhanced_calculations(
    symbol="AAPL",
    indicators={"prices": [100, 102, 98, 105, 103]},
    chart_paths={},
    period=14,
    interval="1d"
)
```

### Image Analysis Request
```python
with open("chart.png", "rb") as f:
    image_data = f.read()
    
response = await client.analyze_comprehensive_overview(image_data)
```

## Troubleshooting

### No Logs Appearing
1. Check if debugging is enabled: `python gemini_debug_cli.py status`
2. Verify log level: `python gemini_debug_cli.py level DEBUG`
3. Check file permissions for log directory

### Missing API Key
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Performance Issues
- Reduce log level to INFO or WARNING
- Disable file logging: `export GEMINI_LOG_TO_FILE=false`
- Reduce log lengths: `export GEMINI_MAX_PROMPT_LOG=500`

### Log File Management
Log files are automatically created with timestamps. Old logs can be safely deleted:
```bash
# List log files
ls -la backend/logs/

# Clean old logs (older than 7 days)
find backend/logs/ -name "gemini_debug_*.log" -mtime +7 -delete
```

## Advanced Configuration

### Custom Log Format
Modify `debug_logger.py` to change log format or add custom fields.

### Integration with External Logging
The debug logger can be integrated with external logging systems by modifying the `_setup_logger()` method.

### Performance Monitoring
Response times are automatically logged. For detailed performance analysis, check the log files for patterns in response times.

## Security Notes

- Logs may contain sensitive information (API keys, prompts, responses)
- Ensure log files are properly secured
- Consider disabling file logging in production
- Review logs before sharing or committing to version control

## Support

For issues or questions about the debugging system:
1. Check the log files for detailed error information
2. Verify configuration with `python gemini_debug_cli.py status`
3. Run the test: `python gemini_debug_cli.py test`
4. Review this README for configuration options 