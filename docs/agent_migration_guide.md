# Agent Migration Guide: backend/gemini → backend/llm

This guide explains how to migrate agents from the old `backend/gemini` system to the new `backend/llm` system.

## Overview

The new `backend/llm` system provides:
- ✅ **Provider-agnostic** design (Gemini, OpenAI, Claude, etc.)
- ✅ **Model-specific configuration** per agent
- ✅ **Simple, clean API** calls without over-engineering
- ✅ **Easy to test** and maintain

## Migration Pattern (Risk Agent Example)

The risk analysis agent was successfully migrated as a proof of concept. Here's the step-by-step pattern:

### Step 1: Add Agent Configuration

Add your agent to `backend/llm/config/llm_assignments.yaml`:

```yaml
agents:
  # Your Agent Name
  your_agent:
    provider: "gemini"
    model: "gemini-2.5-flash"  # or gemini-2.5-pro for complex reasoning
    enable_code_execution: true  # if needed for calculations
    max_retries: 3
    timeout: 90  # adjust based on expected processing time
```

### Step 2: Update Agent Implementation

**Before (using GeminiClient):**
```python
class YourAgent:
    def __init__(self):
        from gemini.gemini_client import GeminiClient
        self.gemini_client = GeminiClient(api_key=api_key, agent_name="your_agent")
    
    async def analyze(self, data):
        # Uses GeminiClient's context_engineer and prompt_manager
        context = self.gemini_client.context_engineer.structure_context(...)
        prompt = self.gemini_client.prompt_manager.format_prompt("template", context=context)
        response = await self.gemini_client.core.call_llm_with_code_execution(prompt)
```

**After (using backend/llm):**
```python
class YourAgent:
    def __init__(self):
        # Import with proper path handling
        import sys
        import os
        parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if parent_path not in sys.path:
            sys.path.insert(0, parent_path)
        from backend.llm import get_llm_client
        
        self.llm_client = get_llm_client("your_agent")  # Uses config from YAML
        # Keep existing prompt and context logic
        self.prompt_template = self._load_prompt_template()
    
    async def analyze(self, data):
        # Build context using your own logic (no context_engineer dependency)
        context = self._build_context(data)
        # Build prompt using your own logic (no prompt_manager dependency)  
        prompt = self._build_prompt(context)
        # Call LLM using provider-agnostic interface
        response = await self.llm_client.generate(prompt, enable_code_execution=True)
```

### Step 3: Handle Module Import Issues

**Problem**: Agents that create global instances may fail with `ModuleNotFoundError: No module named 'backend'`

**Solution**: Use lazy initialization pattern:

```python
# At bottom of agent file - OLD WAY (causes import errors)
# your_agent = YourAgent()  # ❌ Don't do this

# NEW WAY - Lazy initialization
_your_agent_instance = None

def get_your_agent():
    """Get or create the global agent instance."""
    global _your_agent_instance
    if _your_agent_instance is None:
        _your_agent_instance = YourAgent()
    return _your_agent_instance

# For backwards compatibility
your_agent = None  # Will be set on first access
```

Update `__init__.py`:
```python
from .your_agent import YourAgent, get_your_agent

def _get_your_agent():
    return get_your_agent()

your_agent = _get_your_agent  # Function to get agent

__all__ = ['YourAgent', 'get_your_agent', 'your_agent']
```

### Step 4: Update Service References

Update any services that import the agent:

**Before:**
```python
from agents.your_package.your_agent import your_agent
result = await your_agent.analyze(data)
```

**After:**
```python
from agents.your_package.your_agent import get_your_agent
agent = get_your_agent()
result = await agent.analyze(data)
```

### Step 5: Test the Migration

Create a test script (see `backend/agents/risk_analysis/test_migration.py` for example):

```python
async def test_your_agent_migration():
    # Test 1: Import the migrated agent
    from agents.your_package.your_agent import YourAgent
    
    # Test 2: Initialize the agent  
    agent = YourAgent()
    print(f"Provider: {agent.llm_client.get_provider_info()}")
    print(f"Config: {agent.llm_client.get_config()}")
    
    # Test 3: Test prompt building (without LLM call)
    prompt = agent._build_prompt(mock_data)
    assert prompt, "Prompt building failed"
    
    print("✅ Migration successful!")
```

## Agent Types & Migration Complexity

### Easy Migration (Like Risk Agent)
**Characteristics:**
- ❌ Does NOT use `context_engineer.structure_context()`
- ❌ Does NOT use `prompt_manager.format_prompt()`  
- ✅ Has own prompt loading and context building logic
- ✅ Only uses `GeminiClient.core` for LLM calls

**Examples:** Risk Agent, Volume Agents

**Migration:** Simple - just replace `GeminiClient.core` with `backend/llm`

### Medium Migration (Like Final Decision Agent)
**Characteristics:**
- ✅ Uses GeminiClient methods for context building
- ✅ Uses `prompt_manager.format_prompt()`
- ⚠️ Moderately coupled to GeminiClient helper methods

**Examples:** Final Decision Agent

**Migration:** Move GeminiClient helper methods to agent-specific code

### Complex Migration (Like Indicators Agent)
**Characteristics:**
- ✅ Heavily uses `context_engineer.structure_context()`
- ✅ Uses `AnalysisType` enums and complex structuring logic  
- ✅ Tightly coupled to GeminiClient context engineering

**Examples:** Indicators Agent, Some Volume Agents

**Migration:** Move entire context engineering logic to agent-specific code

## Benefits After Migration

### For Agents:
- 🎯 **Agent-specific configuration**: Each agent can use different models/providers
- 🧪 **Easy testing**: No complex GeminiClient dependencies  
- 🔧 **Simple maintenance**: Clear separation of concerns
- 📝 **Clean code**: Agent owns its prompts and context logic

### For System:
- 🔄 **Provider flexibility**: Switch between Gemini/OpenAI/Claude per agent
- ⚡ **Better performance**: Reduced overhead and complexity
- 🛠️ **Easier debugging**: Clear request/response flow
- 📊 **Better monitoring**: Provider-specific metrics and logging

## Next Steps

1. **Start with easy agents** (like Risk Agent)
2. **Test thoroughly** with actual API calls
3. **Apply same pattern** to medium complexity agents
4. **Handle complex agents last** (may need more planning)
5. **Update documentation** as you learn from each migration

## Migration Checklist

- [ ] Add agent configuration to `llm_assignments.yaml`
- [ ] Update agent implementation to use `backend/llm`
- [ ] Fix module import paths  
- [ ] Handle global instance creation (lazy initialization)
- [ ] Update service references
- [ ] Create migration test script
- [ ] Test with actual API keys
- [ ] Update documentation
- [ ] Monitor for regressions in production

## Troubleshooting

### Import Errors
**Error:** `ModuleNotFoundError: No module named 'backend'`

**Solution:** Add proper path handling in agent `__init__`:
```python
import sys
import os
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
```

### Configuration Errors  
**Error:** `ValueError: Unsupported provider: xyz`

**Solution:** Check `llm_assignments.yaml` has correct provider name (`gemini`, `openai`, `claude`)

### API Key Errors
**Error:** `ValueError: Gemini API key is required`

**Solution:** Ensure environment variables are set (`GEMINI_API_KEY1-5` or `GEMINI_API_KEY`)

---

## Success Story: Risk Agent Migration

The risk analysis agent migration was completed successfully with:

**Results:**
- ✅ **Agent import: SUCCESS**  
- ✅ **Agent initialization: SUCCESS**
- ✅ **LLM client setup: SUCCESS**
- ✅ **Prompt template loading: SUCCESS**  
- ✅ **Prompt building: SUCCESS**

**Configuration Used:**
```yaml
risk_agent:
  provider: "gemini"
  model: "gemini-2.5-flash"
  enable_code_execution: true
  max_retries: 3
  timeout: 90
```

**Key Changes:**
- Replaced `GeminiClient()` → `get_llm_client("risk_agent")`
- Replaced `self.gemini_client.core.call_llm_with_code_execution()` → `self.llm_client.generate()`
- Added proper path handling for imports
- Used lazy initialization for global instance
- Updated service references to use getter function

This migration pattern can now be applied to other agents with similar characteristics.