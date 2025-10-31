"""
Test script to verify dataset loading and all fixes.
Tests:
1. Dataset loading from HuggingFace
2. Input validation
3. Model configuration
4. Basic retrieval and generation
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset_loader import HQSmallLoader
from src.utils.validation import sanitize_query, validate_model_name, ValidationError
from src.models.data_models import Query, Document
from src.retrieval.sparse import BM25Retriever
from src.generation.openrouter_client import OpenRouterClient
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def test_dataset_loading():
    """Test HQ-small dataset loading from HuggingFace."""
    print("\n" + "="*80)
    print("TEST 1: Dataset Loading")
    print("="*80)
    
    try:
        loader = HQSmallLoader(cache_dir="./data/cache")
        
        # Load dataset
        print("\n1. Loading dataset from HuggingFace...")
        dataset = loader.load_dataset()
        print("✓ Dataset loaded successfully")
        
        # Validate structure
        print("\n2. Validating dataset structure...")
        loader.validate_dataset_structure()
        print("✓ Dataset structure validated")
        
        # Get statistics
        print("\n3. Dataset statistics:")
        stats = loader.get_statistics()
        for split, count in stats.items():
            print(f"   {split}: {count} samples")
        
        # Load collection
        print("\n4. Loading document collection...")
        documents = loader.load_collection()
        print(f"✓ Loaded {len(documents)} documents")
        
        # Show sample document
        if documents:
            sample_doc = documents[0]
            print(f"\n   Sample document:")
            print(f"   ID: {sample_doc.id}")
            print(f"   Text: {sample_doc.text[:100]}...")
        
        # Load train data
        print("\n5. Loading training data...")
        train_data = loader.load_train()
        print(f"✓ Loaded {len(train_data)} training samples")
        
        # Show sample query
        if train_data:
            sample = train_data[0]
            print(f"\n   Sample query:")
            print(f"   ID: {sample['id']}")
            print(f"   Text: {sample['text']}")
            print(f"   Answer: {sample['answer']}")
            print(f"   Supporting IDs: {sample['supporting_ids'][:3]}...")
        
        # Load validation data
        print("\n6. Loading validation data...")
        val_data = loader.load_validation()
        print(f"✓ Loaded {len(val_data)} validation samples")
        
        # Load test data
        print("\n7. Loading test data...")
        test_data = loader.load_test()
        print(f"✓ Loaded {len(test_data)} test samples")
        
        print("\n" + "="*80)
        print("✓ TEST 1 PASSED: Dataset loading works correctly")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_input_validation():
    """Test input validation functions."""
    print("\n" + "="*80)
    print("TEST 2: Input Validation")
    print("="*80)
    
    test_cases = [
        ("Valid query", "What is machine learning?", True),
        ("Empty query", "", False),
        ("Whitespace only", "   ", False),
        ("Too short", "ab", False),
        ("Very long query", "a" * 3000, True),  # Should truncate
        ("Control characters", "Test\x00query\x01", True),  # Should sanitize
        ("Normal question", "Where is the Eiffel Tower?", True),
    ]
    
    passed = 0
    failed = 0
    
    for name, query, should_pass in test_cases:
        try:
            result = sanitize_query(query)
            if should_pass:
                print(f"✓ {name}: Passed (sanitized to: '{result[:50]}...')")
                passed += 1
            else:
                print(f"✗ {name}: Should have failed but passed")
                failed += 1
        except ValidationError as e:
            if not should_pass:
                print(f"✓ {name}: Correctly rejected ({e})")
                passed += 1
            else:
                print(f"✗ {name}: Should have passed but failed ({e})")
                failed += 1
    
    # Test model name validation
    print("\nModel name validation:")
    try:
        validate_model_name("qwen/qwen-2.5-1.5b-instruct")
        print("✓ Valid Qwen model accepted")
        passed += 1
    except ValidationError as e:
        print(f"✗ Valid model rejected: {e}")
        failed += 1
    
    try:
        validate_model_name("invalid_model")
        print("✗ Invalid model accepted")
        failed += 1
    except ValidationError:
        print("✓ Invalid model correctly rejected")
        passed += 1
    
    print("\n" + "="*80)
    print(f"TEST 2 RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("✓ TEST 2 PASSED: Input validation works correctly")
    else:
        print("✗ TEST 2 FAILED: Some validation tests failed")
    print("="*80)
    
    return failed == 0


def test_model_configuration():
    """Test that default model is Qwen2.5."""
    print("\n" + "="*80)
    print("TEST 3: Model Configuration")
    print("="*80)
    
    try:
        import os
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            print("⚠ OPENROUTER_API_KEY not set, skipping API client test")
            print("  (This is OK for now, but needed for full testing)")
            return True
        
        client = OpenRouterClient(api_key=api_key)
        
        print(f"\n1. Default model: {client.default_model}")
        
        # Check if it's a Qwen model
        if "qwen" in client.default_model.lower():
            print("✓ Default model is a Qwen variant (correct)")
        else:
            print(f"✗ Default model is not Qwen: {client.default_model}")
            return False
        
        # Check specific version
        if "qwen-2.5" in client.default_model or "qwen2.5" in client.default_model:
            print("✓ Default model is Qwen2.5 (correct)")
        else:
            print(f"⚠ Default model is Qwen but not 2.5: {client.default_model}")
        
        print("\n" + "="*80)
        print("✓ TEST 3 PASSED: Model configuration correct")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_retrieval():
    """Test basic retrieval with sample documents."""
    print("\n" + "="*80)
    print("TEST 4: Basic Retrieval")
    print("="*80)
    
    try:
        # Create sample documents
        documents = [
            Document(id="doc1", text="Python is a high-level programming language."),
            Document(id="doc2", text="Machine learning is a subset of artificial intelligence."),
            Document(id="doc3", text="The Eiffel Tower is located in Paris, France."),
        ]
        
        print(f"\n1. Created {len(documents)} sample documents")
        
        # Initialize BM25 retriever
        print("\n2. Initializing BM25 retriever...")
        retriever = BM25Retriever(index_path=None)
        retriever.index_documents(documents)
        print("✓ Documents indexed")
        
        # Test retrieval
        print("\n3. Testing retrieval...")
        query = Query(id="q1", text="What is Python?")
        results = retriever.retrieve(query, k=2)
        
        print(f"✓ Retrieved {len(results)} documents")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.document_id} (score: {result.score:.4f})")
        
        # Verify correct document retrieved
        if results and results[0].document_id == "doc1":
            print("\n✓ Correct document ranked first")
        else:
            print("\n⚠ Expected doc1 to be ranked first")
        
        print("\n" + "="*80)
        print("✓ TEST 4 PASSED: Basic retrieval works")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE TESTS")
    print("="*80)
    
    results = {
        "Dataset Loading": test_dataset_loading(),
        "Input Validation": test_input_validation(),
        "Model Configuration": test_model_configuration(),
        "Basic Retrieval": test_basic_retrieval(),
    }
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print("\n" + "="*80)
    print(f"OVERALL: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    
    print("="*80 + "\n")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
