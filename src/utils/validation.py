"""
Input validation utilities for security and data integrity.
"""

import re
import os
from typing import Any, Optional
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


def sanitize_query(query: str, max_length: int = 2000) -> str:
    """
    Sanitize user query to prevent injection and ensure valid input.
    
    Args:
        query: User query string
        max_length: Maximum allowed query length
        
    Returns:
        Sanitized query string
        
    Raises:
        ValidationError: If query is invalid
    """
    if not query:
        raise ValidationError("Query cannot be empty")
    
    # Remove leading/trailing whitespace
    query = query.strip()
    
    if not query:
        raise ValidationError("Query cannot be empty after stripping whitespace")
    
    # Check length
    if len(query) > max_length:
        logger.warning(f"Query truncated from {len(query)} to {max_length} characters")
        query = query[:max_length]
    
    # Remove null bytes and other control characters
    query = query.replace('\x00', '')
    query = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', query)
    
    # Check for minimum length
    if len(query) < 3:
        raise ValidationError("Query too short (minimum 3 characters)")
    
    return query


def sanitize_file_path(file_path: str, base_dir: Optional[str] = None) -> str:
    """
    Sanitize file path to prevent directory traversal attacks.
    
    Args:
        file_path: File path to sanitize
        base_dir: Base directory to restrict access to
        
    Returns:
        Sanitized absolute file path
        
    Raises:
        ValidationError: If path is invalid or outside base_dir
    """
    if not file_path:
        raise ValidationError("File path cannot be empty")
    
    # Convert to Path object and resolve
    try:
        path = Path(file_path).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid file path: {e}")
    
    # If base_dir specified, ensure path is within it
    if base_dir:
        try:
            base = Path(base_dir).resolve()
            path.relative_to(base)
        except ValueError:
            raise ValidationError(f"Path {file_path} is outside base directory {base_dir}")
    
    return str(path)


def validate_document_id(doc_id: str) -> str:
    """
    Validate document ID format.
    
    Args:
        doc_id: Document ID to validate
        
    Returns:
        Validated document ID
        
    Raises:
        ValidationError: If ID is invalid
    """
    if not doc_id:
        raise ValidationError("Document ID cannot be empty")
    
    # Remove whitespace
    doc_id = doc_id.strip()
    
    # Check length
    if len(doc_id) > 255:
        raise ValidationError("Document ID too long (max 255 characters)")
    
    # Check for invalid characters (allow alphanumeric, underscore, hyphen, dot)
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', doc_id):
        raise ValidationError("Document ID contains invalid characters")
    
    return doc_id


def validate_api_key(api_key: str) -> str:
    """
    Validate API key format (basic check).
    
    Args:
        api_key: API key to validate
        
    Returns:
        Validated API key
        
    Raises:
        ValidationError: If API key is invalid
    """
    if not api_key:
        raise ValidationError("API key cannot be empty")
    
    api_key = api_key.strip()
    
    # Check minimum length
    if len(api_key) < 10:
        raise ValidationError("API key too short")
    
    # Check for whitespace
    if ' ' in api_key or '\t' in api_key or '\n' in api_key:
        raise ValidationError("API key contains whitespace")
    
    return api_key


def validate_model_name(model_name: str, allowed_prefixes: Optional[list] = None) -> str:
    """
    Validate model name format.
    
    Args:
        model_name: Model name to validate
        allowed_prefixes: List of allowed model name prefixes (e.g., ['qwen/', 'deepseek/'])
        
    Returns:
        Validated model name
        
    Raises:
        ValidationError: If model name is invalid
    """
    if not model_name:
        raise ValidationError("Model name cannot be empty")
    
    model_name = model_name.strip()
    
    # Check format (should be provider/model-name)
    if '/' not in model_name:
        raise ValidationError("Model name must be in format 'provider/model-name'")
    
    # Check allowed prefixes if specified
    if allowed_prefixes:
        if not any(model_name.startswith(prefix) for prefix in allowed_prefixes):
            raise ValidationError(f"Model name must start with one of: {allowed_prefixes}")
    
    return model_name


def validate_k_value(k: int, min_k: int = 1, max_k: int = 100) -> int:
    """
    Validate k value for retrieval.
    
    Args:
        k: Number of documents to retrieve
        min_k: Minimum allowed k value
        max_k: Maximum allowed k value
        
    Returns:
        Validated k value
        
    Raises:
        ValidationError: If k is invalid
    """
    if not isinstance(k, int):
        raise ValidationError("k must be an integer")
    
    if k < min_k:
        raise ValidationError(f"k must be at least {min_k}")
    
    if k > max_k:
        raise ValidationError(f"k must be at most {max_k}")
    
    return k


def validate_temperature(temperature: float, min_temp: float = 0.0, max_temp: float = 2.0) -> float:
    """
    Validate temperature value for generation.
    
    Args:
        temperature: Temperature value
        min_temp: Minimum allowed temperature
        max_temp: Maximum allowed temperature
        
    Returns:
        Validated temperature
        
    Raises:
        ValidationError: If temperature is invalid
    """
    if not isinstance(temperature, (int, float)):
        raise ValidationError("Temperature must be a number")
    
    temperature = float(temperature)
    
    if temperature < min_temp:
        raise ValidationError(f"Temperature must be at least {min_temp}")
    
    if temperature > max_temp:
        raise ValidationError(f"Temperature must be at most {max_temp}")
    
    return temperature


def sanitize_error_message(error_message: str, hide_sensitive: bool = True) -> str:
    """
    Sanitize error message to prevent information leakage.
    
    Args:
        error_message: Original error message
        hide_sensitive: Whether to hide potentially sensitive information
        
    Returns:
        Sanitized error message
    """
    if not hide_sensitive:
        return error_message
    
    # Remove API keys (pattern: sk-... or similar)
    error_message = re.sub(r'sk-[a-zA-Z0-9]{20,}', '[API_KEY_HIDDEN]', error_message)
    error_message = re.sub(r'key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}', 'key=[HIDDEN]', error_message, flags=re.IGNORECASE)
    
    # Remove file paths
    error_message = re.sub(r'[A-Za-z]:\\[^"\s]+', '[PATH_HIDDEN]', error_message)
    error_message = re.sub(r'/[a-zA-Z0-9_\-/\.]+', '[PATH_HIDDEN]', error_message)
    
    # Remove IP addresses
    error_message = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_HIDDEN]', error_message)
    
    return error_message
