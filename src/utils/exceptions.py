"""
Custom exception classes and error handling utilities.
"""

from typing import Optional, Any


class RAGSystemError(Exception):
    """Base exception for RAG system errors."""
    
    def __init__(self, message: str, component: str = "unknown", details: Optional[Any] = None):
        self.message = message
        self.component = component
        self.details = details
        super().__init__(self.message)
    
    def __str__(self):
        return f"[{self.component}] {self.message}"


class DataLoadingError(RAGSystemError):
    """Exception raised when data loading fails."""
    
    def __init__(self, message: str, dataset_name: str = "", details: Optional[Any] = None):
        super().__init__(message, "DataLoader", details)
        self.dataset_name = dataset_name


class RetrievalError(RAGSystemError):
    """Exception raised during retrieval operations."""
    
    def __init__(self, message: str, method: str = "", query_id: str = "", details: Optional[Any] = None):
        super().__init__(message, "Retrieval", details)
        self.method = method
        self.query_id = query_id


class EmbeddingError(RetrievalError):
    """Exception raised during embedding operations."""
    
    def __init__(self, message: str, model_name: str = "", details: Optional[Any] = None):
        super().__init__(message, "Embedding", details)
        self.model_name = model_name


class IndexError(RetrievalError):
    """Exception raised during indexing operations."""
    
    def __init__(self, message: str, index_type: str = "", details: Optional[Any] = None):
        super().__init__(message, "Indexing", details)
        self.index_type = index_type


class GenerationError(RAGSystemError):
    """Exception raised during answer generation."""
    
    def __init__(self, message: str, model_name: str = "", query_id: str = "", details: Optional[Any] = None):
        super().__init__(message, "Generation", details)
        self.model_name = model_name
        self.query_id = query_id


class ModelLoadError(GenerationError):
    """Exception raised when model loading fails."""
    
    def __init__(self, message: str, model_name: str = "", details: Optional[Any] = None):
        super().__init__(message, "ModelLoader", details)
        self.model_name = model_name


class APIError(GenerationError):
    """Exception raised during API calls."""
    
    def __init__(self, message: str, api_name: str = "", status_code: int = 0, details: Optional[Any] = None):
        super().__init__(message, "API", details)
        self.api_name = api_name
        self.status_code = status_code


class ConfigurationError(RAGSystemError):
    """Exception raised for configuration issues."""
    
    def __init__(self, message: str, config_key: str = "", details: Optional[Any] = None):
        super().__init__(message, "Configuration", details)
        self.config_key = config_key


class EvaluationError(RAGSystemError):
    """Exception raised during evaluation."""
    
    def __init__(self, message: str, metric: str = "", details: Optional[Any] = None):
        super().__init__(message, "Evaluation", details)
        self.metric = metric


class UIError(RAGSystemError):
    """Exception raised in user interface components."""
    
    def __init__(self, message: str, interface_type: str = "", details: Optional[Any] = None):
        super().__init__(message, "UI", details)
        self.interface_type = interface_type