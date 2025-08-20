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

    def create_ml_enhanced_prompt(self, base_template: str, ml_context: dict, **kwargs) -> str:
        """
        Create an enhanced prompt that incorporates ML validation context.
        This method helps the LLM utilize ML system insights for better analysis.
        """
        try:
            # Load the base template
            template = self.load_template(base_template)
            if not template:
                raise FileNotFoundError(f"Base template '{base_template}.txt' not found")
            
            # Create ML context instructions
            ml_instructions = self._create_ml_instructions(ml_context)
            
            # Add ML instructions to the context
            if 'context' in kwargs:
                kwargs['context'] += f"\n\n{ml_instructions}"
            else:
                kwargs['context'] = ml_instructions
            
            # Format the enhanced prompt
            return self.format_prompt(base_template, **kwargs)
            
        except Exception as e:
            # logger.error(f"Failed to create ML-enhanced prompt: {e}") # Assuming logger is defined elsewhere
            # Fallback to base template
            return self.format_prompt(base_template, **kwargs)
    
    def _create_ml_instructions(self, ml_context: dict) -> str:
        """
        Create specific instructions for the LLM to utilize ML validation context.
        """
        if not ml_context or not isinstance(ml_context, dict):
            return ""
        
        instructions = "\n## ML System Validation Context (Use This to Enhance Your Analysis):\n"
        
        # Add pattern validation insights
        if ml_context.get('pattern_validation'):
            instructions += "\n### Pattern Success Probabilities (ML-Validated):\n"
            for pattern, data in ml_context['pattern_validation'].items():
                if isinstance(data, dict):
                    prob = data.get('success_probability', 0)
                    confidence = data.get('confidence', 'medium')
                    risk = data.get('risk_level', 'medium')
                    instructions += f"- {pattern.replace('_', ' ').title()}: {prob:.1%} success rate, {confidence} confidence, {risk} risk\n"
        
        # Add risk assessment
        if ml_context.get('risk_assessment'):
            risk_data = ml_context['risk_assessment']
            instructions += f"\n### Overall Risk Assessment (ML-Generated):\n"
            instructions += f"- Overall Pattern Success Rate: {risk_data.get('overall_pattern_success_rate', 0):.1%}\n"
            instructions += f"- High Confidence Patterns: {risk_data.get('high_confidence_patterns', 0)}\n"
            instructions += f"- Low Risk Patterns: {risk_data.get('low_risk_patterns', 0)}\n"
            
            if risk_data.get('risk_distribution'):
                dist = risk_data['risk_distribution']
                instructions += f"- Risk Distribution: Low: {dist.get('low', 0)}, Medium: {dist.get('medium', 0)}, High: {dist.get('high', 0)}\n"
        
        # Add confidence metrics
        if ml_context.get('confidence_metrics'):
            conf_data = ml_context['confidence_metrics']
            instructions += f"\n### Confidence Metrics (ML-Validated):\n"
            instructions += f"- Average Confidence: {conf_data.get('average_confidence', 0):.1%}\n"
            
            if conf_data.get('confidence_distribution'):
                conf_dist = conf_data['confidence_distribution']
                instructions += f"- Confidence Distribution: Very High: {conf_dist.get('very_high', 0)}, High: {conf_dist.get('high', 0)}, Medium: {conf_dist.get('medium', 0)}, Low: {conf_dist.get('low', 0)}\n"
        
        # Add specific instructions for the LLM
        instructions += "\n### Instructions for Enhanced Analysis:\n"
        instructions += "1. **Use ML probabilities** to validate your pattern identification\n"
        instructions += "2. **Adjust confidence levels** based on ML validation results\n"
        instructions += "3. **Incorporate risk assessments** from ML system into your recommendations\n"
        instructions += "4. **Highlight patterns** with high ML confidence scores\n"
        instructions += "5. **Provide risk-adjusted recommendations** using ML risk levels\n"
        instructions += "6. **Mention ML validation** when discussing pattern reliability\n"
        
        return instructions 