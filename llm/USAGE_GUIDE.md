# How to Use backend/llm for LLM Requests

This guide shows you exactly how to send requests to Gemini (and other LLMs) using the new `backend/llm` system.

## 🚀 Quick Start

### 1. Basic Import and Setup

```python
from backend.llm import get_llm_client
import asyncio

# Create a client (uses configuration from YAML)
client = get_llm_client("indicator_agent")  # Uses agent-specific config
# OR
client = get_llm_client(provider="gemini", model="gemini-2.5-flash")  # Direct config
```

### 2. Simple Text Request

```python
async def simple_request():
    client = get_llm_client("indicator_agent")
    
    prompt = "Analyze this stock data: AAPL price $150, RSI 65, MACD bullish. What's your analysis?"
    
    response = await client.generate_text(prompt)
    print(response)

# Run it
asyncio.run(simple_request())
```

### 3. Request with Code Execution

```python
async def request_with_code():
    client = get_llm_client("indicator_agent")
    
    prompt = """
    Calculate the RSI for this price data: [145, 147, 149, 146, 150, 152, 148]
    Use Python to calculate and explain the result.
    """
    
    response = await client.generate(
        prompt=prompt,
        enable_code_execution=True  # Enables Python code execution
    )
    print(response)

asyncio.run(request_with_code())
```

## 📋 Complete Usage Patterns

### Pattern 1: Agent-Based Configuration (Recommended)

This is the **recommended approach** - uses predefined agent configurations from `llm_assignments.yaml`.

```python
from backend.llm import get_llm_client
import asyncio

async def agent_based_request():
    # These agents are pre-configured in llm_assignments.yaml:
    
    # Indicator Agent (gemini-2.5-flash, 45s timeout, code execution enabled)
    indicator_client = get_llm_client("indicator_agent")
    
    # Risk Agent (gemini-2.5-pro, 60s timeout, code execution enabled) 
    risk_client = get_llm_client("risk_agent")
    
    # Volume Agent (gemini-2.5-flash, 90s timeout, code execution enabled)
    volume_client = get_llm_client("volume_agent")
    
    # Send requests
    indicator_response = await indicator_client.generate_text(
        "Analyze RSI and MACD signals for bullish momentum"
    )
    
    risk_response = await risk_client.generate_text(
        "Assess the risk level for this trading setup"
    )
    
    print("Indicator Analysis:", indicator_response)
    print("Risk Assessment:", risk_response)

asyncio.run(agent_based_request())
```

### Pattern 2: Direct Provider Configuration

For custom configurations not in YAML:

```python
from backend.llm import get_llm_client
import asyncio

async def direct_config_request():
    # Create client with specific provider and model
    client = get_llm_client(
        provider="gemini",
        model="gemini-2.5-pro",  # Use Pro model for better analysis
        timeout=120,             # 2 minute timeout
        max_retries=5           # More retries for important requests
    )
    
    response = await client.generate_text(
        "Provide detailed technical analysis of TSLA stock patterns"
    )
    
    print(response)

asyncio.run(direct_config_request())
```

### Pattern 3: Image Analysis (Multi-Modal)

For analyzing charts and images:

```python
from backend.llm import get_llm_client
from PIL import Image
import asyncio

async def image_analysis_request():
    client = get_llm_client("pattern_agent")  # Pre-configured for image analysis
    
    # Load your chart image
    chart_image = Image.open("path/to/your/chart.png")
    
    prompt = """
    Analyze this stock chart and identify:
    1. Key support and resistance levels
    2. Chart patterns (triangles, flags, etc.)
    3. Volume confirmation
    4. Overall trend direction
    """
    
    response = await client.generate_with_images(
        prompt=prompt,
        images=[chart_image]
    )
    
    print("Chart Analysis:", response)

asyncio.run(image_analysis_request())
```

### Pattern 4: Advanced Request with All Options

```python
from backend.llm import get_llm_client
import asyncio

async def advanced_request():
    client = get_llm_client("final_decision_agent")  # Uses gemini-2.5-pro
    
    prompt = """
    Based on this technical analysis data:
    - RSI: 72 (overbought)
    - MACD: Bullish crossover 
    - Volume: 150% above average
    - Price: Above all moving averages
    
    Provide a trading recommendation with risk assessment.
    """
    
    response = await client.generate(
        prompt=prompt,
        enable_code_execution=True,  # Allow calculations
        timeout=90,                  # 90 second timeout
        max_retries=3               # Retry on failures
    )
    
    print("Trading Decision:", response)
    print("Provider used:", client.get_provider_info())

asyncio.run(advanced_request())
```

## 🛠 Practical Integration Examples

### Example 1: Replace Old Gemini Usage

**OLD WAY (backend/gemini):**
```python
from backend.gemini.gemini_core import GeminiCore
from backend.gemini.prompt_manager import PromptManager

gemini = GeminiCore()
prompt_manager = PromptManager()

# Complex setup...
prompt = prompt_manager.format_prompt("template_name", context=data)
response = gemini.call_llm(prompt, model="gemini-2.5-flash")
```

**NEW WAY (backend/llm):**
```python
from backend.llm import get_llm_client

client = get_llm_client("indicator_agent")
response = await client.generate_text(prompt)
```

### Example 2: In Your Agent Files

```python
# In your agent files (e.g., backend/agents/indicators/some_agent.py)

from backend.llm import get_llm_client
import asyncio

class IndicatorAgent:
    def __init__(self):
        # Create client for this agent
        self.llm_client = get_llm_client("indicator_agent")
    
    async def analyze_indicators(self, stock_data):
        prompt = f"""
        Analyze these technical indicators:
        {stock_data}
        
        Provide analysis of trend, momentum, and key signals.
        """
        
        analysis = await self.llm_client.generate_text(prompt)
        return analysis
    
    async def analyze_with_calculation(self, price_data):
        prompt = f"""
        Calculate technical indicators for this price data: {price_data}
        Use Python to calculate RSI, MACD, and moving averages.
        """
        
        result = await self.llm_client.generate(
            prompt=prompt,
            enable_code_execution=True
        )
        return result
```

### Example 3: Error Handling

```python
from backend.llm import get_llm_client
import asyncio

async def robust_request():
    try:
        client = get_llm_client("indicator_agent")
        
        response = await client.generate_text(
            prompt="Analyze market sentiment",
            timeout=30,
            max_retries=3
        )
        
        return response
        
    except TimeoutError:
        print("Request timed out")
        return None
        
    except Exception as e:
        print(f"LLM request failed: {e}")
        return None

result = asyncio.run(robust_request())
```

## ⚙️ Configuration Options

### Available Agents (Pre-configured)

From `llm_assignments.yaml`:

- `indicator_agent` - Fast analysis (gemini-2.5-flash, 45s timeout)
- `volume_agent` - Volume analysis (gemini-2.5-flash, 90s timeout)  
- `risk_agent` - Risk assessment (gemini-2.5-pro, 60s timeout)
- `final_decision_agent` - Final decisions (gemini-2.5-pro, 90s timeout)
- `pattern_agent` - Image/chart analysis (gemini-2.5-flash, 60s timeout)
- `sector_agent` - Sector analysis (gemini-2.5-flash, 30s timeout)

### Runtime Parameter Override

```python
# Override any parameter at runtime
client = get_llm_client(
    "indicator_agent",           # Base agent config
    model="gemini-2.5-pro",     # Override to use Pro model
    timeout=120,                # Override timeout
    max_retries=5               # Override retry count
)
```

### Environment Variable Overrides

Set these to override configuration:
```bash
export LLM_PROVIDER=gemini
export LLM_MODEL=gemini-2.5-pro
export LLM_TIMEOUT=90
export LLM_MAX_RETRIES=5

# Agent-specific overrides
export LLM_INDICATOR_AGENT_MODEL=gemini-2.5-pro
export LLM_RISK_AGENT_TIMEOUT=120
```

## 🔧 Best Practices

1. **Use agent-based configuration** when possible for consistency
2. **Enable code execution** for calculations and data processing
3. **Handle errors gracefully** with try/catch blocks
4. **Set appropriate timeouts** based on request complexity
5. **Use async/await** pattern for all requests
6. **Check provider info** if you need to know which model was used

## 🚨 Common Mistakes to Avoid

1. **Don't forget async/await** - all requests are asynchronous
2. **Don't block the event loop** - use proper async patterns
3. **Don't ignore timeout errors** - handle them appropriately
4. **Don't hardcode models** - use agent configurations when possible

## 🎯 Summary

**Simple request:**
```python
from backend.llm import get_llm_client
client = get_llm_client("indicator_agent")
response = await client.generate_text("Your prompt here")
```

**That's it!** The system handles everything else - API keys, retries, model selection, configuration, etc.