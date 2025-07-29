# Gemini API Debugging Implementation Summary

## Overview

A comprehensive debugging system has been implemented for all Gemini API calls in the trading application. This system provides detailed logging of requests, responses, and execution flow to help you understand and troubleshoot API interactions.

## What Was Implemented

### 1. Core Debugging Module (`gemini/debug_logger.py`)

**Features:**
- Comprehensive API request logging (prompts, model parameters, configuration)
- Detailed API response analysis (text, code execution, execution outputs)
- Error tracking with context and stack traces
- JSON parsing debugging and fix attempts
- Performance monitoring with response time tracking
- File and console logging with timestamped files
- Image analysis support for multi-modal requests

**Key Methods:**
- `log_api_request()` - Logs detailed request information
- `log_api_response()` - Logs response structure and content
- `log_error()` - Logs errors with full context
- `log_processing_step()` - Logs processing steps
- `log_json_parsing()` - Logs JSON parsing attempts

### 2. Configuration System (`gemini/debug_config.py`)

**Features:**
- Environment variable-based configuration
- Easy enable/disable functionality
- Log level control (DEBUG, INFO, WARNING, ERROR)
- Configurable log length limits
- Runtime configuration changes

**Environment Variables:**
- `GEMINI_DEBUG` - Enable/disable debugging (default: true)
- `GEMINI_LOG_TO_FILE` - Enable/disable file logging (default: true)
- `GEMINI_LOG_LEVEL` - Log level (default: INFO)
- `GEMINI_MAX_PROMPT_LOG` - Max prompt length to log (default: 1000)
- `GEMINI_MAX_RESPONSE_LOG` - Max response length to log (default: 2000)

### 3. Command-Line Interface (`gemini_debug_cli.py`)

**Commands:**
- `enable` - Enable debugging
- `disable` - Disable debugging
- `status` - Show current configuration
- `level <level>` - Set log level
- `env` - Show environment variables
- `test` - Run test to verify functionality
- `help` - Show usage information

### 4. Integration Points

**Enhanced Files:**
- `gemini/gemini_core.py` - All API call methods now include debugging
- `gemini/gemini_client.py` - JSON parsing and client methods include debugging

**Methods with Debugging:**
- `call_llm()` - Basic text requests
- `call_llm_with_code_execution()` - Code execution requests
- `call_llm_with_image()` - Single image requests
- `call_llm_with_images()` - Multiple image requests
- `build_indicators_summary()` - Indicator analysis
- `analyze_stock()` - Stock analysis
- `extract_markdown_and_json()` - JSON parsing
- All image analysis methods

### 5. Test and Demo Scripts

**Test Script (`test_gemini_debug.py`):**
- Simple test to verify debugging functionality
- Tests both text and code execution requests
- Shows log output format

**Demo Script (`demo_gemini_debug.py`):**
- Comprehensive demonstration of debugging features
- Shows real API calls with detailed logging
- Educational tool for understanding the system

## How to Use

### Quick Start

1. **Check Status:**
   ```bash
   cd backend
   python gemini_debug_cli.py status
   ```

2. **Enable Debugging (if needed):**
   ```bash
   python gemini_debug_cli.py enable
   ```

3. **Run Demo:**
   ```bash
   python demo_gemini_debug.py
   ```

4. **View Logs:**
   - Console output shows real-time logs
   - File logs saved to `backend/logs/gemini_debug_YYYYMMDD_HHMMSS.log`

### Configuration

**Set Log Level:**
```bash
python gemini_debug_cli.py level DEBUG
```

**Disable File Logging:**
```bash
export GEMINI_LOG_TO_FILE=false
```

**Reduce Log Lengths:**
```bash
export GEMINI_MAX_PROMPT_LOG=500
export GEMINI_MAX_RESPONSE_LOG=1000
```

## What You'll See in the Terminal

### API Request Log
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
```

### API Response Log
```
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

### Error Log
```
14:30:20 - ERROR - ❌ GEMINI API ERROR
14:30:20 - ERROR - --------------------------------------------------------------------------------
14:30:20 - ERROR - 🔧 Context: call_llm
14:30:20 - ERROR - 💥 Error Type: RateLimitError
14:30:20 - ERROR - 💥 Error Message: Rate limit exceeded
14:30:20 - ERROR - 📝 Prompt that caused error: Analyze the following...
14:30:20 - ERROR - 📋 Traceback: [full stack trace]
14:30:20 - ERROR - ================================================================================
```

## Benefits

1. **Complete Visibility:** See exactly what's sent to and received from Gemini API
2. **Error Diagnosis:** Detailed error information with context and stack traces
3. **Performance Monitoring:** Track response times and identify bottlenecks
4. **JSON Debugging:** Monitor JSON parsing and fixing attempts
5. **Code Execution Tracking:** See code execution results and outputs
6. **Flexible Configuration:** Easy to enable/disable and configure
7. **File Logging:** Persistent logs for later analysis
8. **Non-Intrusive:** Doesn't affect normal operation when disabled

## Security Considerations

- Logs may contain sensitive information (prompts, responses)
- Log files are automatically created with timestamps
- Consider disabling file logging in production
- Review logs before sharing or committing to version control

## Next Steps

1. **Test the System:** Run `python demo_gemini_debug.py` to see it in action
2. **Configure as Needed:** Use the CLI to adjust settings
3. **Monitor Logs:** Check both console and file logs for insights
4. **Troubleshoot Issues:** Use detailed logs to diagnose problems

## Files Created/Modified

**New Files:**
- `gemini/debug_logger.py` - Core debugging functionality
- `gemini/debug_config.py` - Configuration management
- `gemini_debug_cli.py` - Command-line interface
- `test_gemini_debug.py` - Test script
- `demo_gemini_debug.py` - Demonstration script
- `GEMINI_DEBUG_README.md` - Comprehensive documentation
- `GEMINI_DEBUG_IMPLEMENTATION_SUMMARY.md` - This summary

**Modified Files:**
- `gemini/gemini_core.py` - Added debugging to all API methods
- `gemini/gemini_client.py` - Added debugging to client methods

The debugging system is now fully integrated and ready to use. You can see detailed information about all Gemini API calls in your terminal and log files. 