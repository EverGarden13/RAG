"""
Example script demonstrating how to use the optimization system.
Shows hyperparameter tuning and ablation studies.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.optimization.hyperparameter_tuner import (
    HyperparameterTuner,
    HyperparameterSpace,
    TuningResult
)
from src.optimization.ablation_study import AblationStudyFramework
from src.utils.logging_config import setup_logging, get_logger

setup_logging(log_level="INFO")
logger = get_logger(__name__)


def example_hyperparameter_tuning():
    """Example of hyperparameter tuning."""
    logger.info("="*60)
    logger.info("EXAMPLE: Hyperparameter Tuning")
    logger.info("="*60)
    
    # Mock validation data
    validation_data = [
        {'id': f'q{i}', 'question': f'Question {i}', 'answer': f'Answer {i}'}
        for i in range(10)
    ]
    
    # Mock evaluation function
    def mock_evaluation_fn(params, data):
        """Simulate evaluation with some randomness."""
        import random
        
        # Simulate that certain params are better
        base_score = 0.5
        
        # k_retrieve: higher is slightly better
        if 'k_retrieve' in params:
            base_score += (params['k_retrieve'] - 50) / 500
        
        # temperature: lower is better for factual tasks
        if 'temperature' in params:
            base_score += (0.3 - params['temperature']) / 2
        
        # fusion_method: rrf is best
        if params.get('fusion_method') == 'rrf':
            base_score += 0.05
        
        # Add some noise
        noise = random.uniform(-0.05, 0.05)
        combined = max(0.0, min(1.0, base_score + noise))
        
        # Split into EM and nDCG
        em = combined * random.uniform(0.9, 1.1)
        ndcg = combined * random.uniform(0.9, 1.1)
        
        return em, ndcg, combined
    
    # Define parameter spaces
    param_spaces = {
        'k_retrieve': HyperparameterSpace(
            name='k_retrieve',
            values=[50, 100, 150],
            param_type='categorical'
        ),
        'temperature': HyperparameterSpace(
            name='temperature',
            values=[0.0, 0.1, 0.2, 0.3],
            param_type='categorical'
        ),
        'fusion_method': HyperparameterSpace(
            name='fusion_method',
            values=['rrf', 'combsum', 'weighted'],
            param_type='categorical'
        )
    }
    
    # Create tuner
    tuner = HyperparameterTuner(
        evaluation_fn=mock_evaluation_fn,
        validation_data=validation_data,
        output_dir="./examples/tuning_results"
    )
    
    # Run grid search
    logger.info("\nRunning grid search...")
    best_result = tuner.grid_search(param_spaces, max_combinations=10)
    
    logger.info(f"\nBest parameters found:")
    logger.info(f"  {best_result.params}")
    logger.info(f"  Combined score: {best_result.combined_score:.4f}")
    
    # Get top 3 results
    top_results = tuner.get_top_k_results(3)
    logger.info(f"\nTop 3 configurations:")
    for i, result in enumerate(top_results, 1):
        logger.info(f"  {i}. {result.params} -> {result.combined_score:.4f}")


def example_ablation_study():
    """Example of ablation study."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE: Ablation Study")
    logger.info("="*60)
    
    # Mock validation data
    validation_data = [
        {'id': f'q{i}', 'question': f'Question {i}', 'answer': f'Answer {i}'}
        for i in range(10)
    ]
    
    # Mock baseline function
    def mock_baseline_fn(params):
        """Simulate baseline system."""
        return 0.65, 0.70, 0.675  # EM, nDCG, Combined
    
    # Create framework
    framework = AblationStudyFramework(
        baseline_fn=mock_baseline_fn,
        validation_data=validation_data,
        output_dir="./examples/ablation_results"
    )
    
    # Run baseline
    logger.info("\nRunning baseline...")
    baseline = framework.run_baseline()
    
    # Ablate components
    logger.info("\nAblating components...")
    
    # Remove BM25
    def without_bm25(data):
        return 0.55, 0.68, 0.615  # Worse without BM25
    
    framework.ablate_component(
        component_name="bm25_retrieval",
        evaluation_fn=without_bm25,
        description="System without BM25 sparse retrieval"
    )
    
    # Remove dense retrieval
    def without_dense(data):
        return 0.60, 0.65, 0.625  # Worse without dense
    
    framework.ablate_component(
        component_name="dense_retrieval",
        evaluation_fn=without_dense,
        description="System without dense retrieval"
    )
    
    # Remove multi-turn
    def without_multiturn(data):
        return 0.64, 0.69, 0.665  # Slightly worse
    
    framework.ablate_component(
        component_name="multi_turn",
        evaluation_fn=without_multiturn,
        description="System without multi-turn capability"
    )
    
    # Analyze importance
    logger.info("\nAnalyzing component importance...")
    importance = framework.analyze_component_importance()
    
    # Generate report
    logger.info("\nGenerating report...")
    report_path = framework.generate_report()
    logger.info(f"Report saved to: {report_path}")


def example_parameter_comparison():
    """Example of comparing different parameter configurations."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE: Parameter Configuration Comparison")
    logger.info("="*60)
    
    validation_data = [
        {'id': f'q{i}', 'question': f'Question {i}', 'answer': f'Answer {i}'}
        for i in range(10)
    ]
    
    def mock_baseline_fn(params):
        return 0.65, 0.70, 0.675
    
    framework = AblationStudyFramework(
        baseline_fn=mock_baseline_fn,
        validation_data=validation_data,
        output_dir="./examples/comparison_results"
    )
    
    # Define variants to compare
    variants = {
        'conservative': {
            'temperature': 0.0,
            'k_retrieve': 50,
            'fusion': 'rrf'
        },
        'balanced': {
            'temperature': 0.1,
            'k_retrieve': 100,
            'fusion': 'rrf'
        },
        'aggressive': {
            'temperature': 0.3,
            'k_retrieve': 200,
            'fusion': 'combsum'
        }
    }
    
    def eval_variant(config, data):
        """Simulate evaluation of variant."""
        import random
        base = 0.65
        
        # Temperature effect
        if config.get('temperature', 0.1) < 0.15:
            base += 0.02
        
        # k_retrieve effect
        if config.get('k_retrieve', 100) >= 100:
            base += 0.01
        
        noise = random.uniform(-0.02, 0.02)
        combined = base + noise
        
        return combined * 0.95, combined * 1.05, combined
    
    logger.info("\nComparing variants...")
    results = framework.compare_variants(variants, eval_variant)
    
    logger.info(f"\nComparison complete. {len(results)} variants evaluated.")


if __name__ == "__main__":
    logger.info("OPTIMIZATION SYSTEM EXAMPLES")
    logger.info("="*60 + "\n")
    
    # Run examples
    example_hyperparameter_tuning()
    example_ablation_study()
    example_parameter_comparison()
    
    logger.info("\n" + "="*60)
    logger.info("ALL EXAMPLES COMPLETE")
    logger.info("="*60)
    logger.info("\nCheck the ./examples/ directory for results:")
    logger.info("  - tuning_results/")
    logger.info("  - ablation_results/")
    logger.info("  - comparison_results/")
