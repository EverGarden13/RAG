"""
Factory for creating dense retrieval instances.
Provides easy instantiation of E5, BGE, and GTE retrievers.
"""

from typing import Optional, Dict, Any
from src.retrieval.dense import E5Retriever, BGERetriever, GTERetriever
from src.interfaces.base import BaseRetriever
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DenseRetrieverFactory:
    """Factory class for creating dense retrieval instances."""
    
    # Available models for each retriever type
    AVAILABLE_MODELS = {
        'e5': {
            'base': 'intfloat/e5-base-v2',
            'large': 'intfloat/e5-large-v2'
        },
        'bge': {
            'base': 'BAAI/bge-base-en-v1.5',
            'large': 'BAAI/bge-large-en-v1.5'
        },
        'gte': {
            'base': 'thenlper/gte-base'
        }
    }
    
    @classmethod
    def create_retriever(cls, 
                        retriever_type: str, 
                        model_size: str = 'base',
                        index_path: Optional[str] = None,
                        batch_size: int = 32,
                        **kwargs) -> BaseRetriever:
        """
        Create a dense retriever instance.
        
        Args:
            retriever_type: Type of retriever ('e5', 'bge', 'gte')
            model_size: Size of the model ('base', 'large')
            index_path: Path to save/load index files
            batch_size: Batch size for embedding generation
            **kwargs: Additional arguments for the retriever
            
        Returns:
            Configured retriever instance
            
        Raises:
            ValueError: If retriever_type or model_size is not supported
        """
        retriever_type = retriever_type.lower()
        model_size = model_size.lower()
        
        # Validate retriever type
        if retriever_type not in cls.AVAILABLE_MODELS:
            available_types = list(cls.AVAILABLE_MODELS.keys())
            raise ValueError(f"Unsupported retriever type: {retriever_type}. "
                           f"Available types: {available_types}")
        
        # Validate model size
        if model_size not in cls.AVAILABLE_MODELS[retriever_type]:
            available_sizes = list(cls.AVAILABLE_MODELS[retriever_type].keys())
            raise ValueError(f"Unsupported model size '{model_size}' for {retriever_type}. "
                           f"Available sizes: {available_sizes}")
        
        # Get model name
        model_name = cls.AVAILABLE_MODELS[retriever_type][model_size]
        
        # Create retriever instance
        if retriever_type == 'e5':
            retriever = E5Retriever(
                model_name=model_name,
                index_path=index_path,
                batch_size=batch_size,
                **kwargs
            )
        elif retriever_type == 'bge':
            retriever = BGERetriever(
                model_name=model_name,
                index_path=index_path,
                batch_size=batch_size,
                **kwargs
            )
        elif retriever_type == 'gte':
            retriever = GTERetriever(
                model_name=model_name,
                index_path=index_path,
                batch_size=batch_size,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")
        
        logger.info(f"Created {retriever_type}-{model_size} retriever with model: {model_name}")
        return retriever
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, str]]:
        """
        Get dictionary of available models for each retriever type.
        
        Returns:
            Dictionary mapping retriever types to their available models
        """
        return cls.AVAILABLE_MODELS.copy()
    
    @classmethod
    def create_e5_base(cls, index_path: Optional[str] = None, **kwargs) -> E5Retriever:
        """Create E5-base retriever."""
        return cls.create_retriever('e5', 'base', index_path, **kwargs)
    
    @classmethod
    def create_e5_large(cls, index_path: Optional[str] = None, **kwargs) -> E5Retriever:
        """Create E5-large retriever."""
        return cls.create_retriever('e5', 'large', index_path, **kwargs)
    
    @classmethod
    def create_bge_base(cls, index_path: Optional[str] = None, **kwargs) -> BGERetriever:
        """Create BGE-base retriever."""
        return cls.create_retriever('bge', 'base', index_path, **kwargs)
    
    @classmethod
    def create_bge_large(cls, index_path: Optional[str] = None, **kwargs) -> BGERetriever:
        """Create BGE-large retriever."""
        return cls.create_retriever('bge', 'large', index_path, **kwargs)
    
    @classmethod
    def create_gte_base(cls, index_path: Optional[str] = None, **kwargs) -> GTERetriever:
        """Create GTE-base retriever."""
        return cls.create_retriever('gte', 'base', index_path, **kwargs)


# Convenience functions for quick access
def create_e5_retriever(model_size: str = 'base', 
                       index_path: Optional[str] = None, 
                       **kwargs) -> E5Retriever:
    """Create E5 retriever with specified model size."""
    return DenseRetrieverFactory.create_retriever('e5', model_size, index_path, **kwargs)


def create_bge_retriever(model_size: str = 'base', 
                        index_path: Optional[str] = None, 
                        **kwargs) -> BGERetriever:
    """Create BGE retriever with specified model size."""
    return DenseRetrieverFactory.create_retriever('bge', model_size, index_path, **kwargs)


def create_gte_retriever(index_path: Optional[str] = None, 
                        **kwargs) -> GTERetriever:
    """Create GTE retriever."""
    return DenseRetrieverFactory.create_retriever('gte', 'base', index_path, **kwargs)