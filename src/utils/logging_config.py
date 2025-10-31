"""
Comprehensive logging system for debugging and monitoring.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "./logs",
    enable_console: bool = True,
    enable_file: bool = True
) -> logging.Logger:
    """
    Set up comprehensive logging system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional specific log file name
        log_dir: Directory for log files
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
    
    Returns:
        Configured logger instance
    """
    
    # Create logs directory
    if enable_file:
        os.makedirs(log_dir, exist_ok=True)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if enable_file:
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"rag_system_{timestamp}.log"
        
        log_path = os.path.join(log_dir, log_file)
        
        # Rotating file handler to prevent large log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Log the setup
    logger.info(f"Logging initialized - Level: {log_level}")
    if enable_file:
        logger.info(f"Log file: {log_path}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(name)


class PerformanceLogger:
    """Logger for performance monitoring and metrics collection."""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
        self.metrics = {}
    
    def log_retrieval_time(self, method: str, query_id: str, time_ms: float):
        """Log retrieval performance metrics."""
        self.logger.info(f"Retrieval - Method: {method}, Query: {query_id}, Time: {time_ms:.2f}ms")
        
        if method not in self.metrics:
            self.metrics[method] = []
        self.metrics[method].append(time_ms)
    
    def log_generation_time(self, model: str, query_id: str, time_ms: float, tokens: int = 0):
        """Log generation performance metrics."""
        self.logger.info(f"Generation - Model: {model}, Query: {query_id}, Time: {time_ms:.2f}ms, Tokens: {tokens}")
    
    def log_memory_usage(self, component: str, memory_mb: float):
        """Log memory usage metrics."""
        self.logger.info(f"Memory - Component: {component}, Usage: {memory_mb:.2f}MB")
    
    def get_average_time(self, method: str) -> float:
        """Get average time for a specific method."""
        if method in self.metrics and self.metrics[method]:
            return sum(self.metrics[method]) / len(self.metrics[method])
        return 0.0
    
    def log_summary(self):
        """Log performance summary."""
        self.logger.info("=== Performance Summary ===")
        for method, times in self.metrics.items():
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                self.logger.info(f"{method}: Avg={avg_time:.2f}ms, Min={min_time:.2f}ms, Max={max_time:.2f}ms, Count={len(times)}")