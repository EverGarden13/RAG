"""
Main entry point for RAG system.
Provides CLI for running the complete RAG pipeline.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset_loader import HQSmallLoader
from src.models.data_models import Query
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense_factory import create_dense_retriever
from src.retrieval.hybrid import HybridRetriever
from src.generation.openrouter_client import OpenRouterClient
from src.generation.rag_pipeline import BasicRAGPipeline, MultiHopRAGPipeline
from src.generation.agentic_workflow import AgenticRAGPipeline
from src.ui.terminal_interface import TerminalInterface
from src.evaluation.metrics import RAGEvaluator
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def setup_retriever(args):
    """Set up retriever based on arguments."""
    logger.info(f"Setting up retriever: {args.retrieval_method}")
    
    # Load documents
    loader = HQSmallLoader(cache_dir=args.data_dir)
    documents = loader.load_collection()
    logger.info(f"Loaded {len(documents)} documents")
    
    # Create retriever
    if args.retrieval_method == "bm25":
        retriever = BM25Retriever(index_path=os.path.join(args.index_dir, "bm25_index.pkl"))
    elif args.retrieval_method in ["e5-base", "e5-large", "bge-base", "bge-large", "gte-base"]:
        retriever = create_dense_retriever(
            method=args.retrieval_method,
            index_path=os.path.join(args.index_dir, f"{args.retrieval_method}_index.pkl")
        )
    elif args.retrieval_method == "hybrid":
        # Create hybrid retriever with BM25 + E5
        bm25 = BM25Retriever(index_path=os.path.join(args.index_dir, "bm25_index.pkl"))
        e5 = create_dense_retriever(
            method="e5-base",
            index_path=os.path.join(args.index_dir, "e5-base_index.pkl")
        )
        
        # Index documents if needed
        if not bm25.load_index():
            logger.info("Indexing documents with BM25...")
            bm25.index_documents(documents)
        
        if not e5.load_index():
            logger.info("Indexing documents with E5...")
            e5.index_documents(documents)
        
        retriever = HybridRetriever(
            retrievers=[bm25, e5],
            weights=[0.5, 0.5],
            fusion_method="rrf"
        )
        return retriever
    else:
        raise ValueError(f"Unknown retrieval method: {args.retrieval_method}")
    
    # Load or create index
    if not retriever.load_index():
        logger.info("Index not found. Creating new index...")
        retriever.index_documents(documents)
    else:
        logger.info("Loaded existing index")
    
    return retriever


def setup_generator(args):
    """Set up generator based on arguments."""
    logger.info("Setting up OpenRouter client...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment")
    
    generator = OpenRouterClient(
        api_key=api_key,
        default_model=args.model
    )
    
    return generator


def setup_pipeline(retriever, generator, args):
    """Set up RAG pipeline based on arguments."""
    logger.info(f"Setting up pipeline: {args.pipeline_type}")
    
    if args.pipeline_type == "basic":
        pipeline = BasicRAGPipeline(
            retriever=retriever,
            generator=generator,
            k_retrieve=args.k_retrieve
        )
    elif args.pipeline_type == "multi_hop":
        pipeline = MultiHopRAGPipeline(
            retriever=retriever,
            generator=generator,
            k_retrieve=args.k_retrieve
        )
    elif args.pipeline_type == "agentic":
        pipeline = AgenticRAGPipeline(
            retriever=retriever,
            generator=generator,
            enable_decomposition=True,
            enable_self_check=True,
            enable_cot=True,
            k_retrieve=args.k_retrieve
        )
    else:
        raise ValueError(f"Unknown pipeline type: {args.pipeline_type}")
    
    return pipeline


def run_interactive(args):
    """Run interactive terminal interface."""
    logger.info("Starting interactive mode...")
    
    # Setup components
    retriever = setup_retriever(args)
    generator = setup_generator(args)
    pipeline = setup_pipeline(retriever, generator, args)
    
    # Create terminal interface
    interface = TerminalInterface(
        rag_pipeline=pipeline,
        enable_multi_turn=args.multi_turn
    )
    
    # Run interface
    interface.run()


def run_batch(args):
    """Run batch processing on test set."""
    logger.info("Starting batch processing...")
    
    # Setup components
    retriever = setup_retriever(args)
    generator = setup_generator(args)
    pipeline = setup_pipeline(retriever, generator, args)
    
    # Load queries
    loader = HQSmallLoader(cache_dir=args.data_dir)
    
    if args.split == "validation":
        queries_data = loader.load_validation()
    elif args.split == "test":
        queries_data = loader.load_test()
    else:
        raise ValueError(f"Unknown split: {args.split}")
    
    logger.info(f"Loaded {len(queries_data)} queries from {args.split} split")
    
    # Convert to Query objects
    queries = [Query(id=q['id'], text=q['text']) for q in queries_data]
    
    # Process queries
    logger.info("Processing queries...")
    results = pipeline.process_batch(queries)
    
    # Save results
    output_file = args.output or f"{args.split}_predictions.jsonl"
    pipeline.save_results(results, output_file)
    logger.info(f"Results saved to {output_file}")
    
    # Evaluate if references available
    if args.split == "validation":
        logger.info("Evaluating results...")
        evaluator = RAGEvaluator(k=10)
        
        # Prepare references
        references = [
            {
                'id': q['id'],
                'answer': q.get('answer', ''),
                'supporting_ids': q.get('supporting_ids', [])
            }
            for q in queries_data
        ]
        
        # Prepare predictions
        predictions = [
            {
                'id': r.id,
                'answer': r.answer,
                'retrieved_docs': r.retrieved_docs
            }
            for r in results
        ]
        
        eval_results = evaluator.evaluate(predictions, references)
        
        # Save evaluation results
        import json
        eval_file = output_file.replace('.jsonl', '_eval.json')
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Evaluation results saved to {eval_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG System CLI")
    
    # Mode
    parser.add_argument('--mode', choices=['interactive', 'batch'], default='interactive',
                       help='Run mode: interactive or batch')
    
    # Retrieval
    parser.add_argument('--retrieval-method', default='bm25',
                       choices=['bm25', 'e5-base', 'e5-large', 'bge-base', 'bge-large', 
                               'gte-base', 'hybrid'],
                       help='Retrieval method to use')
    parser.add_argument('--k-retrieve', type=int, default=10,
                       help='Number of documents to retrieve')
    
    # Generation
    parser.add_argument('--model', default='qwen/qwen-2.5-1.5b-instruct',
                       help='OpenRouter model to use (Qwen2.5 variants required)')
    parser.add_argument('--pipeline-type', default='basic',
                       choices=['basic', 'multi_hop', 'agentic'],
                       help='Pipeline type')
    
    # Multi-turn
    parser.add_argument('--multi-turn', action='store_true',
                       help='Enable multi-turn conversation')
    
    # Batch processing
    parser.add_argument('--split', default='validation',
                       choices=['validation', 'test'],
                       help='Dataset split for batch processing')
    parser.add_argument('--output', help='Output file for batch processing')
    
    # Paths
    parser.add_argument('--data-dir', default='./data/cache',
                       help='Data directory')
    parser.add_argument('--index-dir', default='./data/indices',
                       help='Index directory')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.index_dir, exist_ok=True)
    
    try:
        if args.mode == 'interactive':
            run_interactive(args)
        elif args.mode == 'batch':
            run_batch(args)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
