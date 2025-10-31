"""
OpenRouter API client for accessing Qwen2.5 models.
Provides interface for generation with rate limiting and error handling.
"""

import os
import time
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.utils.logging_config import get_logger
from src.utils.exceptions import GenerationError

logger = get_logger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    model: str = "qwen/qwen-2.5-1.5b-instruct"
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    stop: Optional[List[str]] = None


class OpenRouterClient:
    """Client for OpenRouter API to access Qwen2.5 models."""
    
    # Available models on OpenRouter
    # Note: Model IDs may vary, check https://openrouter.ai/models for current list
    AVAILABLE_MODELS = {
        "qwen3-coder": "qwen/qwen3-coder:free",
        "deepseek-chat": "deepseek/deepseek-chat-v3.1:free",
        "deepseek-r1": "deepseek/deepseek-r1-0528:free",
        "llama-3.3-8b": "meta-llama/llama-3.3-8b-instruct:free",
        "mistral-small": "mistralai/mistral-small-3.2-24b-instruct:free",
        "gemma-3n": "google/gemma-3n-e4b-it:free"
    }
    
    def __init__(self, api_key: Optional[str] = None, 
                 base_url: str = "https://openrouter.ai/api/v1",
                 default_model: str = "qwen/qwen-2.5-1.5b-instruct",
                 max_retries: int = 5,
                 retry_delay: float = 2.0):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (if None, reads from env)
            base_url: Base URL for OpenRouter API
            default_model: Default model to use
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable.")
        
        self.base_url = base_url
        self.default_model = default_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Request tracking for rate limiting
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum 100ms between requests
        
        logger.info(f"OpenRouter client initialized with model: {default_model}")
    
    def _wait_for_rate_limit(self) -> None:
        """Implement simple rate limiting."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request to OpenRouter API with retry logic.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            
        Returns:
            Response JSON
            
        Raises:
            GenerationError: If request fails after retries
        """
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",  # Optional: for rankings
            "X-Title": "RAG System"  # Optional: for rankings
        }
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                self._wait_for_rate_limit()
                
                # Make request
                response = requests.post(url, json=payload, headers=headers, timeout=60)
                
                # Track request
                self.request_count += 1
                
                # Check response
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limit exceeded
                    retry_after = int(response.headers.get('Retry-After', self.retry_delay * (attempt + 1)))
                    logger.warning(f"Rate limit exceeded. Retrying after {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                elif response.status_code >= 500:
                    # Server error, retry
                    logger.warning(f"Server error {response.status_code}. Retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    # Client error, don't retry
                    error_msg = response.text
                    raise GenerationError(f"API request failed: {response.status_code} - {error_msg}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout. Attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise GenerationError("Request timed out after maximum retries")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request exception: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise GenerationError(f"Request failed: {e}")
        
        raise GenerationError(f"Failed after {self.max_retries} attempts")
    
    def generate(self, prompt: str, 
                 config: Optional[GenerationConfig] = None,
                 model: Optional[str] = None) -> str:
        """
        Generate text using OpenRouter API.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            model: Model to use (overrides default)
            
        Returns:
            Generated text
            
        Raises:
            GenerationError: If generation fails
        """
        try:
            # Use provided config or create default
            if config is None:
                config = GenerationConfig(model=model or self.default_model)
            elif model:
                config.model = model
            
            # Prepare payload
            payload = {
                "model": config.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repetition_penalty": config.repetition_penalty
            }
            
            if config.stop:
                payload["stop"] = config.stop
            
            logger.debug(f"Generating with model: {config.model}")
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # Make API request
            response = self._make_request("chat/completions", payload)
            
            # Extract generated text
            if "choices" in response and len(response["choices"]) > 0:
                generated_text = response["choices"][0]["message"]["content"]
                
                # Log usage if available
                if "usage" in response:
                    usage = response["usage"]
                    logger.debug(f"Token usage - Prompt: {usage.get('prompt_tokens', 0)}, "
                               f"Completion: {usage.get('completion_tokens', 0)}, "
                               f"Total: {usage.get('total_tokens', 0)}")
                
                return generated_text.strip()
            else:
                raise GenerationError("No response generated from API")
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise GenerationError(f"Failed to generate text: {e}")
    
    def generate_batch(self, prompts: List[str], 
                      config: Optional[GenerationConfig] = None,
                      model: Optional[str] = None) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            config: Generation configuration
            model: Model to use
            
        Returns:
            List of generated texts
        """
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating {i+1}/{len(prompts)}")
            try:
                result = self.generate(prompt, config, model)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate for prompt {i+1}: {e}")
                results.append("")  # Empty string for failed generations
        
        return results
    
    def get_available_models(self) -> Dict[str, str]:
        """
        Get available Qwen2.5 models.
        
        Returns:
            Dictionary of model aliases to full model names
        """
        return self.AVAILABLE_MODELS.copy()
    
    def set_default_model(self, model: str) -> None:
        """
        Set default model for generation.
        
        Args:
            model: Model name or alias
        """
        if model in self.AVAILABLE_MODELS:
            model = self.AVAILABLE_MODELS[model]
        
        self.default_model = model
        logger.info(f"Default model set to: {model}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "request_count": self.request_count,
            "default_model": self.default_model,
            "base_url": self.base_url
        }
