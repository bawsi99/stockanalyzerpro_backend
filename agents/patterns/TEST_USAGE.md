# Pattern Analysis Test Suite Usage Guide

## Overview

The Pattern Analysis Test Suite (`test_pattern_analysis.py`) provides comprehensive testing and debugging capabilities for the Pattern LLM Agent. It allows you to:

- Test pattern analysis with real stock data
- Inspect the full LLM prompt before it's sent
- Analyze LLM responses in detail
- Save all outputs for debugging
- Run in no-LLM mode for prompt-only testing

## Quick Start

### Basic Usage
```bash
# Test RELIANCE with default settings
cd backend/agents/patterns
python test_pattern_analysis.py --symbol RELIANCE

# Test with custom period
python test_pattern_analysis.py --symbol TATAMOTORS --period 180

# Test US stock
python test_pattern_analysis.py --symbol AAPL --exchange NASDAQ
```

### Advanced Usage
```bash
# No-LLM mode (build prompt but don't send to LLM)
python test_pattern_analysis.py --symbol INFY --no-llm

# Save all files for detailed analysis
python test_pattern_analysis.py --symbol RELIANCE --save-files

# Full test with custom context
python test_pattern_analysis.py --symbol HDFC --context "Bank stock analysis for swing trading" --save-files

# Custom output directory
python test_pattern_analysis.py --symbol TCS --save-files --output-dir ~/pattern_tests
```

## Command Line Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--symbol, -s` | Stock symbol (required) | - | `RELIANCE`, `TATAMOTORS`, `AAPL` |
| `--exchange, -e` | Stock exchange | `NSE` | `NSE`, `NASDAQ`, `NYSE` |
| `--period, -p` | Analysis period in days | `365` | `180`, `90`, `500` |
| `--interval, -i` | Data interval | `day` | `day`, `hour`, `5minute` |
| `--context, -c` | Additional analysis context | `""` | `"Swing trading analysis"` |
| `--no-llm` | Skip LLM call, build prompt only | `False` | Use for prompt debugging |
| `--save-files` | Save prompt, response, and data | `False` | Saves to `test_outputs/` |
| `--output-dir, -o` | Custom output directory | `test_outputs/` | `~/my_tests/` |

## What It Tests

### 1. Data Retrieval 📊
- Fetches real stock data from your data provider
- Validates data quality and completeness
- Shows current price and date range

### 2. Technical Indicators 🔧
- Calculates all technical indicators
- Displays key indicator values (SMA, RSI, MACD, Bollinger Bands)
- Validates indicator calculations

### 3. Pattern Context Building 🧠
- Runs pattern orchestrator for technical analysis
- Builds comprehensive LLM context
- Shows confidence scores and pattern detection

### 4. LLM Prompt Construction 🤖
- Builds the complete LLM prompt
- Shows prompt statistics (length, lines)
- Previews prompt content

### 5. LLM Analysis (Optional) 💬
- Sends prompt to LLM for analysis
- Captures and displays response
- Shows response statistics

### 6. File Saving 💾
When `--save-files` is used, saves:
- `{symbol}_{exchange}_{timestamp}_prompt.txt` - Full LLM prompt
- `{symbol}_{exchange}_{timestamp}_response.txt` - LLM response (if available)
- `{symbol}_{exchange}_{timestamp}_context.txt` - Pattern analysis context
- `{symbol}_{exchange}_{timestamp}_technical.json` - Technical analysis data
- `{symbol}_{exchange}_{timestamp}_results.json` - Complete test results

## Example Output

```
🚀 Starting Pattern Analysis Test Suite
======================================================================
Symbol: RELIANCE
Exchange: NSE
Period: 365 days
Interval: day
Context: Default analysis context
No-LLM Mode: False
Save Files: True
======================================================================

📊 Step 1: Retrieving Stock Data
----------------------------------------
✅ Retrieved 365 days of stock data
   Date range: 2023-10-10 to 2024-10-09
   Current price: $2,456.75

🔧 Step 2: Calculating Technical Indicators
----------------------------------------
✅ Calculated 45 technical indicators
   SMA_20: 2,398.50
   RSI: 58.42
   MACD: 12.34
   BOLLINGER_UPPER: 2,587.23
   BOLLINGER_LOWER: 2,209.77

🧠 Step 3: Building Pattern Context
----------------------------------------
✅ Pattern technical analysis completed
   Overall confidence: 68%
✅ Built LLM context (2,847 characters)

🤖 Step 4: Constructing LLM Prompt
----------------------------------------
✅ Built LLM prompt (3,456 characters)
   Prompt has 89 lines
   First 3 lines:
     1. You are an expert technical analyst. Analyze the pattern data for RE...
     2. 
     3. PATTERN ANALYSIS DATA:

🤖 Step 5: LLM Analysis
----------------------------------------
✅ LLM analysis completed
   Response length: 1,247 characters
   Response has 23 lines
   First 3 lines:
     1. Based on the comprehensive pattern analysis for RELIANCE, I can ident...
     2. 
     3. ## Key Patterns Identified

💾 Step 6: Saving Test Files
----------------------------------------
✅ Saved LLM prompt: RELIANCE_NSE_20241010_143052_prompt.txt
✅ Saved LLM response: RELIANCE_NSE_20241010_143052_response.txt
✅ Saved pattern context: RELIANCE_NSE_20241010_143052_context.txt
✅ Saved technical analysis: RELIANCE_NSE_20241010_143052_technical.json
✅ Saved test results: RELIANCE_NSE_20241010_143052_results.json
📁 All files saved to: /path/to/test_outputs

🎯 Final Results
======================================================================
Overall Success: ✅ PASS
Total Processing Time: 12.34 seconds
  Data Retrieval: ✅ PASS
  Indicators: ✅ PASS
  Context Building: ✅ PASS
  Prompt Construction: ✅ PASS
  Llm Analysis: ✅ PASS
```

## Debugging Tips

### 1. Use No-LLM Mode First
```bash
python test_pattern_analysis.py --symbol RELIANCE --no-llm --save-files
```
This builds the complete prompt without making API calls, perfect for:
- Debugging prompt construction
- Checking pattern context formatting
- Validating technical analysis data

### 2. Save Files for Analysis
```bash
python test_pattern_analysis.py --symbol RELIANCE --save-files
```
The saved files help you:
- Inspect the exact prompt sent to LLM
- Analyze LLM responses in detail
- Debug pattern context building
- Review technical analysis data

### 3. Test Different Time Periods
```bash
# Short-term patterns
python test_pattern_analysis.py --symbol RELIANCE --period 90

# Long-term patterns  
python test_pattern_analysis.py --symbol RELIANCE --period 500
```

### 4. Add Context for Specific Analysis
```bash
python test_pattern_analysis.py --symbol RELIANCE --context "Day trading setup analysis with focus on breakout patterns"
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're running from the `backend/agents/patterns` directory
2. **Data Provider Error**: Check your data provider authentication
3. **LLM Error**: Verify your LLM API keys are configured
4. **Permission Error**: Check write permissions for output directory

### Exit Codes
- `0`: All tests passed
- `1`: One or more tests failed or error occurred

### Getting Help
```bash
python test_pattern_analysis.py --help
```

## Integration with Your Workflow

This test suite is perfect for:
- **Development**: Testing pattern analysis changes
- **Debugging**: Inspecting LLM prompts and responses  
- **Optimization**: Fine-tuning pattern detection algorithms
- **Validation**: Ensuring consistent analysis results
- **Documentation**: Understanding how the pattern agent works

## Next Steps

After running tests:
1. Review saved prompt files to understand LLM inputs
2. Analyze response files to validate LLM outputs
3. Use technical analysis JSON for debugging pattern detection
4. Adjust patterns or prompts based on results
5. Re-test with different stocks and time periods