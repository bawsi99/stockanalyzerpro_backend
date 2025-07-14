import os

class PromptManager:
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
            return template.format(**kwargs)
        raise FileNotFoundError(f"Prompt template '{template_name}.txt' not found in {self.prompt_dir}") 