"""
Script to run performance optimization for RAG system.
Performs hyperparameter tuning and ablation studies.
"""

import os
import sys
import argparse
from pathlib import Path

from src.optimization.optimization_pipeline import (
    OptimizationPipeline,
    OptimizationConfig,
    create_quick_optimization_config,
    create_comprehensive_optimization_config
)
from src.data.dataset_loader import HotpotQALoader
from src.models.data_models import Query
from src.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def create_rag_system_factory():
    """
    Create factory function for RAG system with configurable parameters.
    
    Returns:
        Factory function that takes params and returns RAG system
    """
    from src.retrieval.sparse import BM25Retriever
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.hybrid import HybridRetriever
    from src.generation.openrouter_client import OpenRouterClient, GenerationConfig
    from src.generation.rag_pipeline import BasicRAGPipeline
    from src.generation.prompt_templates import PromptManager
    
    # Load documents once
    logger.info("Loading dataset...")
    loader = HotpotQALoader()
    train_data, val_data, test_data, documents = loader.load_dataset()
    logger.info(f"Loaded {len(documents)} documents")
    
    # Initialize OpenRouter client
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)
    
    def factory(params):
        """Create RAG system with given parameters."""
        # Extract parameters
        k_retrieve = params.get('k_retrieve', 100)
        k_final = params.get('k_final', 10)
        fusion_method = params.get('fusion_method', 'rrf')
        bm25_weight = params.get('bm25_weight', 0.5)
        dense_weight = params.get('dense_weight', 0.5)
        
        temperature = params.get('temperature', 0.1)
        max_tokens = params.get('max_tokens', 512)
        top_p = params.get('top_p', 0.9)
        max_context_length = params.get('max_context_length', 2000)
        
        # Create retrievers
        logger.info(f"Creating retrievers with k={k_retrieve}, fusion={fusion_method}")
        
        bm25_retriever = BM25Retriever()
        bm25_retriever.index_documents(documents)
        
        dense_retriever = DenseRetriever(model_name="intfloat/e5-base-v2")
        dense_retriever.index_documents(documents)
        
        # Create hybrid retriever
        retriever = HybridRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[bm25_weight, dense_weight],
            fusion_method=fusion_method
        )
        
        # Create generator
        gen_config = GenerationConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        generator = OpenRouterClient(api_key=api_key)
        
        # Create RAG pipeline
        prompt_manager = PromptManager()
        rag_pipeline = BasicRAGPipeline(
            retriever=retriever,
            generator=generator,
            prompt_manager=prompt_manager,
            k_retrieve=k_final,
            max_context_length=max_context_length
        )
        
        # Wrap to match expected interface
        class RAGSystemWrapper:
            def __init__(self, pipeline):
                self.pipeline = pipeline
            
            def process_query(self, item):
                query = Query(
                    id=item['id'],
                    text=item['question']
                )
                return self.pipeline.process_query(query)
        
        return RAGSystemWrapper(rag_pipeline)
    
    return factory, val_data


def main():
    """Main optimization script."""
    parser = argparse.ArgumentParser(description="Run RAG system optimization")
    parser.add_argument(
        '--mode',
        choices=['quick', 'comprehensive'],
        default='quick',
        help='Optimization mode (quick for testing, comprehensive for production)'
    )
    parser.add_argument(
        '--output-dir',
        default='./optimization_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of validation samples to use (None = all)'
    )
    parser.add_argument(
        '--n-iterations',
        type=int,
        default=None,
        help='Number of tuning iterations'
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("RAG SYSTEM PERFORMANCE OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create configuration
    if args.mode == 'quick':
        config = create_quick_optimization_config()
    else:
        config = create_comprehensive_optimization_config()
    
    # Override with command line args
    config.output_dir = args.output_dir
    if args.sample_size:
        config.validation_sample_size = args.sample_size
    if args.n_iterations:
        config.n_tuning_iterations = args.n_iterations
    
    logger.info(f"Configuration: {config}")
    
    # Create RAG system factory and load validation data
    logger.info("\nInitializing RAG system factory...")
    factory, validation_data = create_rag_system_factory()
    
    logger.info(f"Validation data: {len(validation_data)} samples")
    
    # Create optimization pipeline
    logger.info("\nCreating optimization pipeline...")
    pipeline = OptimizationPipeline(
        rag_system_factory=factory,
        validation_data=validation_data,
        config=config
    )
    
    # Run optimization
    logger.info("\nStarting optimization...")
    results = pipeline.run_full_optimization()
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nBaseline Score: {results['baseline']['combined']:.4f}")
    logger.info(f"Final Score: {results['final_results']['combined']:.4f}")
    improvement = results['final_results']['combined'] - results['baseline']['combined']
    logger.info(f"Improvement: {improvement:+.4f}")
    logger.info(f"\nBest Parameters:")
    for param, value in results['best_params'].items():
        logger.info(f"  {param}: {value}")
    logger.info(f"\nResults saved to: {args.output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
