"""Test complete RAG pipeline end-to-end."""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.models.data_models import Query, Document
from src.retrieval.sparse import BM25Retriever
from src.generation.openrouter_client import OpenRouterClient
from src.generation.rag_pipeline import BasicRAGPipeline
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def test_full_pipeline():
    """Test complete RAG pipeline."""
    print("\n" + "="*80)
    print("TESTING COMPLETE RAG PIPELINE")
    print("="*80)
    
    # Step 1: Create sample documents
    print("\n1. Creating sample documents...")
    documents = [
        Document(id="doc1", text="Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum."),
        Document(id="doc2", text="Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming."),
        Document(id="doc3", text="The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. It was completed in 1889."),
        Document(id="doc4", text="Quantum computing uses quantum-mechanical phenomena like superposition and entanglement to perform operations on data."),
        Document(id="doc5", text="The Great Wall of China is an ancient series of walls and fortifications built to protect Chinese states from invasions."),
    ]
    print(f"✓ Created {len(documents)} documents")
    
    # Step 2: Initialize retriever
    print("\n2. Initializing BM25 retriever...")
    retriever = BM25Retriever(index_path=None)
    retriever.index_documents(documents)
    print("✓ Retriever indexed documents")
    
    # Step 3: Initialize generator
    print("\n3. Initializing OpenRouter client...")
    generator = OpenRouterClient()
    print(f"✓ Generator initialized with model: {generator.default_model}")
    
    # Step 4: Create RAG pipeline
    print("\n4. Creating RAG pipeline...")
    pipeline = BasicRAGPipeline(
        retriever=retriever,
        generator=generator,
        k_retrieve=3
    )
    print("✓ Pipeline created")
    
    # Step 5: Test queries
    test_queries = [
        Query(id="q1", text="Who created Python programming language?"),
        Query(id="q2", text="Where is the Eiffel Tower located?"),
        Query(id="q3", text="What is machine learning?"),
    ]
    
    print("\n5. Processing test queries...")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query.text}")
        print("-"*80)
        
        try:
            result = pipeline.process_query(query)
            
            print(f"Answer: {result.answer}")
            print(f"\nRetrieved {len(result.retrieved_docs)} documents:")
            for j, (doc_id, score) in enumerate(result.retrieved_docs[:3], 1):
                print(f"  {j}. {doc_id} (score: {score:.4f})")
            
            print("✓ Query processed successfully")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("PIPELINE TEST COMPLETE")
    print("="*80)
    
    # Show statistics
    stats = generator.get_stats()
    print(f"\nAPI Requests: {stats['request_count']}")
    print(f"Model Used: {stats['default_model']}")

if __name__ == "__main__":
    test_full_pipeline()
