import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import os
from .debug_config import config

class GeminiDebugLogger:
    """
    Comprehensive debug logger for Gemini API calls.
    Logs detailed information about requests, responses, and execution flow.
    """
    
    def __init__(self, enable_debug: bool = None, log_to_file: bool = None):
        # Use configuration values if not explicitly provided
        self.enable_debug = enable_debug if enable_debug is not None else config.enable_debug
        self.log_to_file = log_to_file if log_to_file is not None else config.log_to_file
        self.logger = None
        
        if self.enable_debug:
            self._setup_logger()
    
    def _setup_logger(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('GeminiDebug')
        
        # Set log level from configuration
        log_level = getattr(logging, config.log_level)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with colored output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for detailed logs
        if self.log_to_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_handler = logging.FileHandler(
                os.path.join(logs_dir, f'gemini_debug_{timestamp}.log'),
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def log_api_request(self, method: str, model: str, prompt: str, enable_code_execution: bool = False, images: Optional[list] = None):
        """Log detailed API request information"""
        if not self.enable_debug or not self.logger:
            return
        
        # Commented out verbose logging to reduce debug output
        # self.logger.info("=" * 80)
        # self.logger.info("🚀 GEMINI API REQUEST")
        # self.logger.info("=" * 80)
        # self.logger.info(f"📅 Timestamp: {datetime.now().isoformat()}")
        # self.logger.info(f"🔧 Method: {method}")
        # self.logger.info(f"🤖 Model: {model}")
        # self.logger.info(f"⚙️ Code Execution: {enable_code_execution}")
        
        # if images:
        #     self.logger.info(f"🖼️ Images: {len(images)} image(s)")
        #     for i, img in enumerate(images):
        #         if hasattr(img, 'shape'):
        #             self.logger.info(f"   Image {i+1}: Shape {img.shape}")
        #         elif isinstance(img, bytes):
        #             self.logger.info(f"   Image {i+1}: {len(img)} bytes")
        
        # self.logger.info(f"📝 Prompt Length: {len(prompt)} characters")
        # self.logger.info(f"📝 Prompt Preview: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        
        # Log full prompt if it's not too long
        # if len(prompt) <= config.max_prompt_length:
        #     self.logger.debug(f"📝 Full Prompt:\n{prompt}")
        # else:
        #     self.logger.debug(f"📝 Full Prompt (truncated):\n{prompt[:config.max_prompt_length]}...")
        
        # self.logger.info("-" * 80)
    
    def log_api_response(self, response: Any, response_time: float, method: str):
        """Log detailed API response information"""
        if not self.enable_debug or not self.logger:
            return
        
        # Commented out verbose logging to reduce debug output
        # self.logger.info("📥 GEMINI API RESPONSE")
        # self.logger.info("-" * 80)
        # self.logger.info(f"⏱️ Response Time: {response_time:.3f} seconds")
        # self.logger.info(f"🔧 Method: {method}")
        
        if response is None:
            self.logger.error("❌ Response is None")
            return
        
        # Log response structure
        # self.logger.info(f"📊 Response Type: {type(response).__name__}")
        
        # Extract and log text response
        text_response = ""
        code_results = []
        execution_results = []
        
        # Extract token information
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        try:
            if hasattr(response, 'candidates') and response.candidates:
                # self.logger.info(f"📋 Candidates: {len(response.candidates)}")
                candidate = response.candidates[0]
                
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        # self.logger.info(f"📄 Parts: {len(candidate.content.parts)}")
                        
                        for i, part in enumerate(candidate.content.parts):
                            if hasattr(part, 'text') and part.text:
                                text_response += part.text
                                # self.logger.debug(f"   Part {i+1} (text): {len(part.text)} chars")
                            
                            if hasattr(part, 'executable_code') and part.executable_code:
                                code_results.append(part.executable_code.code)
                                # self.logger.info(f"   Part {i+1} (code): {len(part.executable_code.code)} chars")
                            
                            if hasattr(part, 'code_execution_result') and part.code_execution_result:
                                execution_results.append(part.code_execution_result.output)
                                # self.logger.info(f"   Part {i+1} (execution): {len(part.code_execution_result.output)} chars")
                    else:
                        if hasattr(candidate.content, 'text'):
                            text_response = candidate.content.text
                            # self.logger.info(f"📄 Direct text: {len(text_response)} chars")
                else:
                    self.logger.warning("⚠️ No content in candidate")
            else:
                # Fallback: try to get text directly
                if hasattr(response, 'text'):
                    text_response = response.text
                    # self.logger.info(f"📄 Direct response text: {len(text_response)} chars")
                else:
                    self.logger.warning("⚠️ No candidates or direct text found")
            
            # Extract token usage from response
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                completion_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                total_tokens = getattr(usage, 'total_token_count', 0) or 0
            elif isinstance(response, dict) and 'usage_metadata' in response:
                usage = response['usage_metadata']
                if isinstance(usage, dict):
                    prompt_tokens = usage.get('prompt_token_count', 0) or 0
                    completion_tokens = usage.get('candidates_token_count', 0) or 0
                    total_tokens = usage.get('total_token_count', 0) or 0
                else:
                    prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                    completion_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                    total_tokens = getattr(usage, 'total_token_count', 0) or 0
            
            # Log text response details
            if text_response:
                # self.logger.info(f"📝 Text Response Length: {len(text_response)} characters")
                # self.logger.info(f"📝 Text Response Preview: {text_response[:200]}{'...' if len(text_response) > 200 else ''}")
                
                # Log full response if not too long
                # if len(text_response) <= config.max_response_length:
                #     self.logger.debug(f"📝 Full Text Response:\n{text_response}")
                # else:
                #     self.logger.debug(f"📝 Full Text Response (truncated):\n{text_response[:config.max_response_length]}...")
                pass
            else:
                self.logger.warning("⚠️ No text response extracted")
            
            # Log code execution results
            # if code_results:
            #     self.logger.info(f"💻 Code Results: {len(code_results)} snippet(s)")
            #     for i, code in enumerate(code_results):
            #         self.logger.debug(f"   Code {i+1}:\n{code}")
            
            # if execution_results:
            #     self.logger.info(f"⚡ Execution Results: {len(execution_results)} output(s)")
            #     for i, result in enumerate(execution_results):
            #         self.logger.debug(f"   Execution {i+1}:\n{result}")
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing response: {e}")
            import traceback
            self.logger.error(f"❌ Traceback:\n{traceback.format_exc()}")
        
        # self.logger.info("=" * 80)
        
        return text_response, code_results, execution_results
    
    def log_error(self, error: Exception, context: str = "", prompt: str = ""):
        """Log error information"""
        if not self.enable_debug or not self.logger:
            return
        
        # Commented out verbose logging to reduce debug output
        # self.logger.error("❌ GEMINI API ERROR")
        # self.logger.error("-" * 80)
        # self.logger.error(f"🔧 Context: {context}")
        # self.logger.error(f"💥 Error Type: {type(error).__name__}")
        # self.logger.error(f"💥 Error Message: {str(error)}")
        
        # if prompt:
        #     self.logger.error(f"📝 Prompt that caused error: {prompt[:500]}{'...' if len(prompt) > 500 else ''}")
        
        # import traceback
        # self.logger.error(f"📋 Traceback:\n{traceback.format_exc()}")
        # self.logger.error("=" * 80)
        
        # Minimal error logging
        self.logger.error(f"Gemini API error in {context}: {type(error).__name__}: {str(error)}")
    
    def log_processing_step(self, step: str, details: str = ""):
        """Log processing step information"""
        if not self.enable_debug or not self.logger:
            return
        
        # Commented out verbose logging to reduce debug output
        # self.logger.info(f"🔄 {step}")
        # if details:
        #     self.logger.debug(f"   Details: {details}")
        
        # Minimal logging - only log at debug level
        self.logger.debug(f"Processing: {step}")
    
    def log_json_parsing(self, json_str: str, success: bool, parsed_data: Any = None, error: Exception = None):
        """Log JSON parsing attempts"""
        if not self.enable_debug or not self.logger:
            return
        
        # Commented out verbose logging to reduce debug output
        # if success:
        #     self.logger.info("✅ JSON parsing successful")
        #     if parsed_data:
        #         self.logger.debug(f"📊 Parsed data keys: {list(parsed_data.keys()) if isinstance(parsed_data, dict) else 'Not a dict'}")
        # else:
        #     self.logger.error("❌ JSON parsing failed")
        #     self.logger.error(f"📝 JSON string: {json_str[:500]}{'...' if len(json_str) > 500 else ''}")
        #     if error:
        #         self.logger.error(f"💥 Error: {str(error)}")
        
        # Minimal logging - only log errors and at debug level
        if success:
            self.logger.debug("JSON parsing successful")
        else:
            self.logger.error(f"JSON parsing failed: {str(error) if error else 'Unknown error'}")

# Global debug logger instance
debug_logger = GeminiDebugLogger() 