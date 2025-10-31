"""
Factory for creating multi-vector retrieval instances.
"""

from typing import Optional
from src.retrieval.multi_vector import ColBERTRetriever, GTEColBERTRetriever
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_multi_vector_retriever(
    method: str,
    index_path: Optional[str] = None,
    batch_size: int = 16,
    max_length: int = 512,
    doc_max_length: int = 180,
    query_max_length: int = 32,
    **kwargs
):
    """
    Factory function to create multi-vector retriever instances.
    
    Args:
        method: Retrieval method name ('colbert', 'gte-colbert')
        index_path: Path to save/load index
        batch_size: Batch size for embedding generation
        max_length: Maximum sequence length
        doc_max_length: Maximum document length
        query_max_length: Maximum query length
        **kwargs: Additional arguments for specific retrievers
        
    Returns:
        Multi-vector retriever instance
        
    Raises:
        ValueError: If method is not supported
    """
    method = method.lower()
    
    if method == "colbert":
        logger.info("Creating ColBERT retriever")
        return ColBERTRetriever(
            model_name="colbert-ir/colbertv2.0",
            index_path=index_path,
            batch_size=batch_size,
            max_length=max_length,
            doc_max_length=doc_max_length,
            query_max_length=query_max_length
        )
    
    elif method == "gte-colbert" or method == "gte_colbert":
        logger.info("Creating GTE-ColBERT retriever")
        return GTEColBERTRetriever(
            model_name="lightonai/GTE-ModernColBERT-v",
            index_path=index_path,
            batch_size=batch_size,
            max_length=max_length,
            doc_max_length=doc_max_length,
            query_max_length=query_max_length
        )
    
    else:
        raise ValueError(
            f"Unsupported multi-vector retrieval method: {method}. "
            f"Supported methods: colbert, gte-colbert"
        )


def get_available_multi_vector_methods():
    """
    Get list of available multi-vector retrieval methods.
    
    Returns:
        List of method names
    """
    return [
        "colbert",
        "gte-colbert"
    ]
