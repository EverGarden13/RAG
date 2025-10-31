"""
Factory for creating instruction-based dense retrieval instances.
"""

from typing import Optional
from src.retrieval.instruction_dense import E5MistralRetriever, QwenEmbeddingRetriever
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_instruction_retriever(
    method: str,
    index_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_length: int = 512,
    **kwargs
):
    """
    Factory function to create instruction-based retriever instances.
    
    Args:
        method: Retrieval method name ('e5-mistral', 'qwen-embedding')
        index_path: Path to save/load index
        batch_size: Batch size for embedding generation
        max_length: Maximum sequence length
        **kwargs: Additional arguments for specific retrievers
        
    Returns:
        Instruction-based retriever instance
        
    Raises:
        ValueError: If method is not supported
    """
    method = method.lower()
    
    if method == "e5-mistral":
        logger.info("Creating E5-Mistral retriever")
        return E5MistralRetriever(
            model_name="intfloat/e5-mistral-7b-instruct",
            index_path=index_path,
            batch_size=batch_size or 8,
            max_length=max_length
        )
    
    elif method == "qwen-embedding" or method == "qwen":
        logger.info("Creating Qwen embedding retriever")
        # Try different Qwen embedding models
        model_name = kwargs.get('model_name', 'Qwen/Qwen2-Embedder-0.5B-Instruct')
        return QwenEmbeddingRetriever(
            model_name=model_name,
            index_path=index_path,
            batch_size=batch_size or 16,
            max_length=max_length
        )
    
    else:
        raise ValueError(
            f"Unsupported instruction-based retrieval method: {method}. "
            f"Supported methods: e5-mistral, qwen-embedding"
        )


def get_available_instruction_methods():
    """
    Get list of available instruction-based retrieval methods.
    
    Returns:
        List of method names
    """
    return [
        "e5-mistral",
        "qwen-embedding"
    ]
