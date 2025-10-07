#!/usr/bin/env python3
"""
Indicator Prompt Manager

Handles prompt template loading and formatting specifically for indicator agents.
Extracted from backend/gemini/prompt_manager.py but focused only on indicator needs.
"""

import os
from typing import Dict, Any, Optional


class IndicatorPromptManager:
    """
    Prompt manager specifically for indicator analysis agents.
    
    Handles:
    - Loading indicator summary prompt template
    - Safe formatting with context injection
    - JSON data handling in prompts
    """
    
    # Solving line appended to prompts
    SOLVING_LINE = "\n\nLet me solve this by .."
    
    def __init__(self):
        self.template_dir = os.path.dirname(os.path.abspath(__file__))
        self.template_cache = {}
        
    def load_template(self, template_name: str) -> str:
        """
        Load a prompt template from the indicators directory.
        
        Args:
            template_name: Name of template file (without .txt extension)
            
        Returns:
            Template content as string
        """
        if template_name in self.template_cache:
            return self.template_cache[template_name]
            
        template_path = os.path.join(self.template_dir, f"{template_name}.txt")
        
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                content = f.read()
                self.template_cache[template_name] = content
                return content
        else:
            raise FileNotFoundError(f"Indicator template '{template_name}.txt' not found in {self.template_dir}")
    
    def format_indicator_summary_prompt(self, context: str) -> str:
        """
        Format the indicator summary prompt with context.
        
        Args:
            context: Formatted context string to inject into template
            
        Returns:
            Complete prompt ready for LLM
        """
        try:
            template = self.load_template("indicator_summary_prompt")
            
            # Safe formatting to handle JSON data in context
            formatted_prompt = self._format_prompt_safely(template, {"context": context})
            
            # Add solving line
            formatted_prompt += self.SOLVING_LINE
            
            return formatted_prompt
            
        except Exception as e:
            raise RuntimeError(f"Failed to format indicator summary prompt: {e}")
    
    def _format_prompt_safely(self, template: str, kwargs: Dict[str, Any]) -> str:
        """
        Safely format prompt when context contains problematic characters.
        
        Handles JSON data with curly braces that could interfere with string formatting.
        """
        try:
            # Try standard formatting first
            return template.format(**kwargs)
        except (KeyError, ValueError) as e:
            # Fallback to manual replacement for context
            if 'context' in kwargs:
                context = kwargs['context']
                # Escape curly braces in context to prevent conflicts
                safe_context = self._escape_context_braces(context)
                
                # Replace {context} placeholder manually
                formatted = template.replace('{context}', safe_context)
                
                # Format remaining kwargs normally
                remaining_kwargs = {k: v for k, v in kwargs.items() if k != 'context'}
                if remaining_kwargs:
                    try:
                        formatted = formatted.format(**remaining_kwargs)
                    except Exception:
                        # If still failing, just do manual replacement
                        for key, value in remaining_kwargs.items():
                            formatted = formatted.replace(f'{{{key}}}', str(value))
                
                return formatted
            else:
                # No context, try manual replacement
                result = template
                for key, value in kwargs.items():
                    result = result.replace(f'{{{key}}}', str(value))
                return result
    
    def _escape_context_braces(self, context: str) -> str:
        """
        Escape curly braces in context to prevent conflicts with template formatting.
        
        This handles JSON data in context that contains { and }.
        """
        if not context:
            return context
        
        # Check if context contains JSON-like structures
        if '{' in context and '}' in context:
            # Escape braces so they don't interfere with template formatting
            escaped_context = context.replace('{', '{{').replace('}', '}}')
            return escaped_context
        
        return context
    
    def get_available_templates(self) -> list:
        """Get list of available template files."""
        template_files = []
        for filename in os.listdir(self.template_dir):
            if filename.endswith('.txt') and 'prompt' in filename:
                template_files.append(filename[:-4])  # Remove .txt extension
        return template_files


# Global instance for indicators
indicator_prompt_manager = IndicatorPromptManager()