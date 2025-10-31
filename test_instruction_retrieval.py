"""
Test script for instruction-based dense retrieval methods.
Tests E5-Mistral and Qwen embedding retrievers.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.data_models import Query, Document
from src.retrieval.instruction_factory import create_instruction_retriever, get_available_instruction_methods
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def test_instruction_retrieval():
    """Test instruction-based retrieval with sample documents."""
    
    logger.info("=" * 80)
    logger.info("Testing Instruction-Based Dense Retrieval")
    logger.info("=" * 80)
    
    # Create sample documents
    documents = [
        Document(id="doc1", text="Python is a high-level programming language known for its simplicity and readability."),
        Document(id="doc2", text="Machine learning is a subset of artificial intelligence that enables systems to learn from data."),
        Document(id="doc3", text="The Eiffel Tower is a wrought-iron lattice tower located in Paris, France."),
        Document(id="doc4", text="Quantum computing uses quantum-mechanical phenomena to perform operations on data."),
        Document(id="doc5", text="The Great Wall of China is an ancient series of walls and fortifications."),
    ]
    
    # Create test queries
    queries = [
        Query(id="q1", text="What is Python programming language?"),
        Query(id="q2", text="Tell me about machine learning"),
        Query(id="q3", text="Where is the Eiffel Tower located?"),
    ]
    
    # Get available methods
    available_methods = get_available_instruction_methods()
    logger.info(f"Available instruction-based methods: {available_methods}")
    
    # Test each method
    for method in available_methods:
        logger.info("\n" + "=" * 80)
        logger.info(f"Testing method: {method.upper()}")
        logger.info("=" * 80)
        
        try:
            # Create retriever
            logger.info(f"Creating {method} retriever...")
            retriever = create_instruction_retriever(
                method=method,
                index_path=None,  # Don't save for testing
                batch_size=2  # Small batch for testing
            )
            
            # Index documents
            logger.info(f"Indexing {len(documents)} documents...")
            retriever.index_documents(documents)
            logger.info("Indexing completed successfully!")
            
            # Test retrieval for each query
            for query in queries:
                logger.info(f"\nQuery: {query.text}")
                results = retriever.retrieve(query, k=3)
                
                logger.info(f"Retrieved {len(results)} documents:")
                for result in results:
                    doc = documents[int(result.document_id.replace('doc', '')) - 1]
                    logger.info(f"  - Rank {result.rank}: {result.document_id} (score: {result.score:.4f})")
                    logger.info(f"    Text: {doc.text[:80]}...")
            
            logger.info(f"\n✓ {method.upper()} test completed successfully!")
            
        except Exception as e:
            logger.error(f"✗ Error testing {method}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info("\n" + "=" * 80)
    logger.info("Instruction-based retrieval testing completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_instruction_retrieval()
