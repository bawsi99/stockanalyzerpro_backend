import os

class PromptManager:
    # Class variable for the solving line
    SOLVING_LINE = "\n\nLet me solve this by .."
    
    def __init__(self, prompt_dir=None):
        if prompt_dir is None:
            # Set to ../prompts relative to this file
            prompt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'prompts'))
        self.prompt_dir = prompt_dir

    def load_template(self, template_name: str) -> str:
        prompt_path = os.path.join(self.prompt_dir, f"{template_name}.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r') as f:
                return f.read()
        return None

    def format_prompt(self, template_name: str, **kwargs) -> str:
        # print(f"[DEBUG] format_prompt called with template_name: {template_name}")
        template = self.load_template(template_name)
        # print(f"[DEBUG] Loaded template for {template_name}:\n{template}")
        # print(f"[DEBUG] format_prompt kwargs: {kwargs}")
        if template:
            try:
                # Check if template contains {context} but context is not provided
                if '{context}' in template and 'context' not in kwargs:
                    print(f"[WARNING] Template {template_name} requires context but none provided. Using default context.")
                    kwargs['context'] = "## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and technical indicators."
                
                # Use safe string formatting to handle context with JSON
                return template.format(**kwargs)
            except (KeyError, ValueError) as e:
                # If formatting fails, try a safer approach
                # print(f"[DEBUG] Prompt formatting failed: {e}, using fallback")
                return self._format_prompt_safely(template, kwargs)
        raise FileNotFoundError(f"Prompt template '{template_name}.txt' not found in {self.prompt_dir}")
    
    def _format_prompt_safely(self, template: str, kwargs: dict) -> str:
        """Safely format prompt when context contains problematic characters."""
        try:
            # Replace {context} manually to avoid formatting issues
            if 'context' in kwargs:
                context = kwargs['context']
                # First, escape any curly braces in the context that might conflict with template
                # This is the key fix: we need to handle JSON data in context that contains { and }
                safe_context = self._escape_context_braces(context)
                # Replace the {context} placeholder manually
                formatted = template.replace('{context}', safe_context)
                # Format the rest of the template normally
                remaining_kwargs = {k: v for k, v in kwargs.items() if k != 'context'}
                if remaining_kwargs:
                    formatted = formatted.format(**remaining_kwargs)
                return formatted
            else:
                # No context, format normally
                return template.format(**kwargs)
        except Exception as e:
            # print(f"[DEBUG] Safe formatting also failed: {e}")
            # Last resort: return template with context as plain text
            if 'context' in kwargs:
                # Replace {context} with the actual context value
                return template.replace('{context}', str(kwargs['context']))
            return template
    
    def _escape_context_braces(self, context: str) -> str:
        """
        Escape curly braces in context to prevent conflicts with template formatting.
        This is specifically designed to handle JSON data in context that contains { and }.
        """
        if not context:
            return context
        
        # The issue is that the template uses {{ and }} to escape braces for JSON schema
        # But the context contains JSON data with { and } that conflicts with this
        # We need to escape the context braces properly
        
        # First, let's check if the context contains JSON-like structures
        if '{' in context and '}' in context:
            # This is likely JSON data that needs special handling
            # We need to escape the braces in the context so they don't interfere with template formatting
            escaped_context = context.replace('{', '{{').replace('}', '}}')
            return escaped_context
        
        return context 