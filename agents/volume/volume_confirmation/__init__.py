"""
Volume Confirmation Analysis Agent

This agent specializes in confirming price movements through volume analysis.
Focuses on volume confirmation signals, divergences, and price-volume relationship validation.

Migration Status:
- ✅ LLM Agent: Migrated to backend/llm (self-contained prompt processing)
- 📊 Processor: Uses existing data processing logic
- 🎨 Charts: Uses existing chart generation
- 📝 Context: Legacy context formatter (may be deprecated)
"""

from .processor import VolumeConfirmationProcessor
from .charts import VolumeConfirmationChartGenerator as VolumeConfirmationCharts
from .context import VolumeConfirmationContext
from .llm_agent import VolumeConfirmationLLMAgent, create_volume_confirmation_llm_agent

__all__ = [
    'VolumeConfirmationProcessor',
    'VolumeConfirmationCharts', 
    'VolumeConfirmationContext',
    'VolumeConfirmationLLMAgent',
    'create_volume_confirmation_llm_agent'
]
