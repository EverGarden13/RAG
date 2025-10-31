"""
Error handling utilities and recovery mechanisms.
"""

import logging
import traceback
import functools
from typing import Callable, Any, Optional, Type, Union
from src.utils.exceptions import RAGSystemError


logger = logging.getLogger(__name__)


def safe_execute(
    func: Callable,
    fallback_value: Any = None,
    fallback_func: Optional[Callable] = None,
    max_retries: int = 3,
    exceptions: tuple = (Exception,),
    log_errors: bool = True
) -> Any:
    """
    Safely execute a function with error handling and retry logic.
    
    Args:
        func: Function to execute
        fallback_value: Value to return if function fails
        fallback_func: Alternative function to call if main function fails
        max_retries: Maximum number of retry attempts
        exceptions: Tuple of exceptions to catch
        log_errors: Whether to log errors
    
    Returns:
        Function result or fallback value
    """
    
    for attempt in range(max_retries):
        try:
            return func()
        except exceptions as e:
            if log_errors:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            
            if attempt == max_retries - 1:
                if fallback_func:
                    try:
                        return fallback_func()
                    except Exception as fallback_error:
                        if log_errors:
                            logger.error(f"Fallback function also failed: {fallback_error}")
                
                if log_errors:
                    logger.error(f"All attempts failed. Returning fallback value: {fallback_value}")
                return fallback_value
    
    return fallback_value


def retry_on_failure(
    max_retries: int = 3,
    exceptions: tuple = (Exception,),
    delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """
    Decorator for automatic retry on function failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        exceptions: Tuple of exceptions to catch and retry on
        delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay on each retry
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}")
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {current_delay:.1f} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
            
            return None
        
        return wrapper
    return decorator


def handle_errors(
    fallback_value: Any = None,
    log_level: str = "ERROR",
    reraise: bool = False
):
    """
    Decorator for comprehensive error handling.
    
    Args:
        fallback_value: Value to return on error
        log_level: Logging level for errors
        reraise: Whether to reraise the exception after logging
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"
                
                # Log with appropriate level
                log_func = getattr(logger, log_level.lower(), logger.error)
                log_func(error_msg)
                
                # Log full traceback at debug level
                logger.debug(f"Full traceback for {func.__name__}:\n{traceback.format_exc()}")
                
                if reraise:
                    raise
                
                return fallback_value
        
        return wrapper
    return decorator


class ErrorRecoveryManager:
    """Manager for error recovery strategies."""
    
    def __init__(self):
        self.recovery_strategies = {}
        self.error_counts = {}
    
    def register_recovery(self, error_type: Type[Exception], recovery_func: Callable):
        """Register a recovery function for a specific error type."""
        self.recovery_strategies[error_type] = recovery_func
    
    def handle_error(self, error: Exception, context: dict = None) -> Any:
        """Handle an error using registered recovery strategies."""
        error_type = type(error)
        
        # Track error frequency
        error_name = error_type.__name__
        self.error_counts[error_name] = self.error_counts.get(error_name, 0) + 1
        
        logger.error(f"Handling error ({self.error_counts[error_name]} occurrences): {error}")
        
        # Try to find a recovery strategy
        for registered_type, recovery_func in self.recovery_strategies.items():
            if isinstance(error, registered_type):
                try:
                    logger.info(f"Attempting recovery using {recovery_func.__name__}")
                    return recovery_func(error, context or {})
                except Exception as recovery_error:
                    logger.error(f"Recovery failed: {recovery_error}")
        
        # No recovery strategy found
        logger.warning(f"No recovery strategy found for {error_type.__name__}")
        return None
    
    def get_error_stats(self) -> dict:
        """Get error occurrence statistics."""
        return self.error_counts.copy()


# Global error recovery manager instance
error_recovery = ErrorRecoveryManager()


def safe_retrieval_fallback(error: Exception, context: dict) -> list:
    """Fallback function for retrieval errors."""
    logger.warning("Using empty retrieval results as fallback")
    return []


def safe_generation_fallback(error: Exception, context: dict) -> str:
    """Fallback function for generation errors."""
    logger.warning("Using default answer as fallback")
    return "I'm sorry, I couldn't generate an answer due to a technical issue."


# Register default recovery strategies
error_recovery.register_recovery(RetrievalError, safe_retrieval_fallback)
error_recovery.register_recovery(GenerationError, safe_generation_fallback)