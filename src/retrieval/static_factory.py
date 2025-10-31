"""
Factory for creating static embedding retrievers.
"""

from typing import Dict, Any, Optional
from src.retrieval.static_embedding import Word2VecRetriever, GloVeRetriever, Model2VecRetriever
from src.interfaces.base import BaseRetriever
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class StaticEmbeddingFactory:
    """Factory class for creating static embedding retrievers."""
    
    AVAILABLE_MODELS = {
        'word2vec': {
            'class': Word2VecRetriever,
            'default_model': 'word2vec-google-news-300',
            'description': 'Word2Vec embeddings from Google News corpus'
        },
        'glove': {
            'class': GloVeRetriever,
            'default_model': 'glove-wiki-gigaword-300',
            'description': 'GloVe embeddings from Wikipedia and Gigaword'
        },
        'model2vec': {
            'class': Model2VecRetriever,
            'default_model': 'minishlab/potion-base-8M',
            'description': 'Model2Vec efficient static embeddings'
        }
    }
    
    @classmethod
    def create_retriever(cls, method: str, model_name: Optional[str] = None, 
                        index_path: Optional[str] = None, **kwargs) -> BaseRetriever:
        """
        Create a static embedding retriever.
        
        Args:
            method: Type of static embedding ('word2vec', 'glove', 'model2vec')
            model_name: Name of the pre-trained model (optional, uses default if None)
            index_path: Path to save/load embeddings (optional)
            **kwargs: Additional arguments for the retriever
            
        Returns:
            Configured static embedding retriever
            
        Raises:
            ValueError: If method is not supported
        """
        if method not in cls.AVAILABLE_MODELS:
            available = list(cls.AVAILABLE_MODELS.keys())
            raise ValueError(f"Unsupported method '{method}'. Available: {available}")
        
        model_info = cls.AVAILABLE_MODELS[method]
        retriever_class = model_info['class']
        
        # Use default model if none specified
        if model_name is None:
            model_name = model_info['default_model']
        
        logger.info(f"Creating {method} retriever with model: {model_name}")
        
        return retriever_class(
            model_name=model_name,
            index_path=index_path,
            **kwargs
        )
    
    @classmethod
    def get_available_methods(cls) -> Dict[str, str]:
        """
        Get available static embedding methods and their descriptions.
        
        Returns:
            Dictionary mapping method names to descriptions
        """
        return {
            method: info['description'] 
            for method, info in cls.AVAILABLE_MODELS.items()
        }
    
    @classmethod
    def create_all_retrievers(cls, index_dir: Optional[str] = None) -> Dict[str, BaseRetriever]:
        """
        Create all available static embedding retrievers.
        
        Args:
            index_dir: Directory to store indices (optional)
            
        Returns:
            Dictionary mapping method names to retriever instances
        """
        retrievers = {}
        
        for method in cls.AVAILABLE_MODELS.keys():
            try:
                index_path = None
                if index_dir:
                    index_path = f"{index_dir}/{method}_embeddings.pkl"
                
                retriever = cls.create_retriever(method, index_path=index_path)
                retrievers[method] = retriever
                logger.info(f"Created {method} retriever successfully")
                
            except Exception as e:
                logger.error(f"Failed to create {method} retriever: {e}")
                # Continue with other retrievers even if one fails
                continue
        
        return retrievers


# Convenience functions for quick access
def create_word2vec_retriever(model_name: str = "word2vec-google-news-300", 
                             index_path: Optional[str] = None) -> Word2VecRetriever:
    """Create a Word2Vec retriever with default settings."""
    return Word2VecRetriever(model_name=model_name, index_path=index_path)


def create_glove_retriever(model_name: str = "glove-wiki-gigaword-300", 
                          index_path: Optional[str] = None) -> GloVeRetriever:
    """Create a GloVe retriever with default settings."""
    return GloVeRetriever(model_name=model_name, index_path=index_path)


def create_model2vec_retriever(model_name: str = "minishlab/potion-base-8M", 
                              index_path: Optional[str] = None) -> Model2VecRetriever:
    """Create a Model2Vec retriever with default settings."""
    return Model2VecRetriever(model_name=model_name, index_path=index_path)