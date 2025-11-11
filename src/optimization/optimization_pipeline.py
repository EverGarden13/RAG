"""
Complete optimization pipeline for RAG system.
Orchestrates hyperparameter tuning and ablation studies for top-tier performance.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import time

from src.optimization.hyperparameter_tuner import (
    HyperparameterTuner,
    HyperparameterSpace,
    create_retrieval_param_spaces,
    create_generation_param_spaces
)
from src.optimization.ablation_study import AblationStudyFramework
from src.evaluation.metrics import RAGEvaluator
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization pipeline."""
    # Tuning settings
    enable_retrieval_tuning: bool = True
    enable_generation_tuning: bool = True
    tuning_method: str = "bayesian"  # grid, random, bayesian
    n_tuning_iterations: int = 20
    
    # Ablation settings
    enable_ablation_study: bool = True
    ablation_components: List[str] = None
    
    # Data settings
    validation_sample_size: Optional[int] = None  # None = use all
    
    # Output settings
    output_dir: str = "./optimization_results"
    save_intermediate: bool = True


class OptimizationPipeline:
    """Complete optimization pipeline for RAG system."""
    
    def __init__(self, rag_system_factory,
                 validation_data: List[Dict[str, Any]],
                 config: Optional[OptimizationConfig] = None):
        """
        Initialize optimization pipeline.
        
        Args:
            rag_system_factory: Factory function to create RAG system with params
            validation_data: Validation dataset
            config: Optimization configuration
        """
        self.rag_system_factory = rag_system_factory
        self.validation_data = validation_data
        self.config = config or OptimizationConfig()
        
        # Sample validation data if needed
        if self.config.validation_sample_size:
            import random
            self.validation_data = random.sample(
                validation_data,
                min(self.config.validation_sample_size, len(validation_data))
            )
        
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluator = RAGEvaluator(k=10)
        
        self.best_params: Dict[str, Any] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        
        logger.info(f"Optimization pipeline initialized")
        logger.info(f"Validation samples: {len(self.validation_data)}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_full_optimization(self) -> Dict[str, Any]:
        """
        Run complete optimization pipeline.
        
        Returns:
            Dictionary with best parameters and results
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING FULL OPTIMIZATION PIPELINE")
        logger.info("="*80 + "\n")
        
        start_time = time.time()
        
        # Phase 1: Baseline evaluation
        logger.info("Phase 1: Baseline Evaluation")
        baseline_results = self._evaluate_baseline()
        
        # Phase 2: Retrieval optimization
        if self.config.enable_retrieval_tuning:
            logger.info("\nPhase 2: Retrieval Optimization")
            retrieval_results = self._optimize_retrieval()
        else:
            retrieval_results = None
        
        # Phase 3: Generation optimization
        if self.config.enable_generation_tuning:
            logger.info("\nPhase 3: Generation Optimization")
            generation_results = self._optimize_generation()
        else:
            generation_results = None
        
        # Phase 4: Joint optimization
        logger.info("\nPhase 4: Joint Optimization")
        joint_results = self._optimize_joint()
        
        # Phase 5: Ablation study
        if self.config.enable_ablation_study:
            logger.info("\nPhase 5: Ablation Study")
            ablation_results = self._run_ablation_study()
        else:
            ablation_results = None
        
        # Phase 6: Final evaluation
        logger.info("\nPhase 6: Final Evaluation")
        final_results = self._evaluate_best_config()
        
        elapsed_time = time.time() - start_time
        
        # Compile results
        optimization_results = {
            'baseline': baseline_results,
            'retrieval_optimization': retrieval_results,
            'generation_optimization': generation_results,
            'joint_optimization': joint_results,
            'ablation_study': ablation_results,
            'final_results': final_results,
            'best_params': self.best_params,
            'elapsed_time': elapsed_time,
            'validation_samples': len(self.validation_data)
        }
        
        # Save results
        self._save_optimization_results(optimization_results)
        
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION PIPELINE COMPLETE")
        logger.info(f"Total time: {elapsed_time/60:.2f} minutes")
        logger.info(f"Best combined score: {final_results['combined']:.4f}")
        logger.info("="*80 + "\n")
        
        return optimization_results
    
    def _evaluate_baseline(self) -> Dict[str, float]:
        """Evaluate baseline system."""
        logger.info("Evaluating baseline configuration...")
        
        # Default parameters
        baseline_params = {
            'k_retrieve': 100,
            'k_final': 10,
            'fusion_method': 'rrf',
            'temperature': 0.1,
            'max_tokens': 512,
            'max_context_length': 2000
        }
        
        em, ndcg, combined = self._evaluate_params(baseline_params)
        
        results = {
            'em': em,
            'ndcg': ndcg,
            'combined': combined,
            'params': baseline_params
        }
        
        logger.info(f"Baseline: EM={em:.4f}, nDCG={ndcg:.4f}, Combined={combined:.4f}")
        
        return results
    
    def _optimize_retrieval(self) -> Dict[str, Any]:
        """Optimize retrieval parameters."""
        logger.info("Optimizing retrieval parameters...")
        
        param_spaces = create_retrieval_param_spaces()
        
        def eval_fn(params, data):
            # Merge with default generation params
            full_params = {
                'temperature': 0.1,
                'max_tokens': 512,
                'max_context_length': 2000,
                **params
            }
            return self._evaluate_params(full_params)
        
        tuner = HyperparameterTuner(
            evaluation_fn=eval_fn,
            validation_data=self.validation_data,
            output_dir=str(self.output_dir / "retrieval_tuning")
        )
        
        if self.config.tuning_method == "grid":
            best_result = tuner.grid_search(param_spaces, max_combinations=30)
        elif self.config.tuning_method == "random":
            best_result = tuner.random_search(param_spaces, n_iterations=self.config.n_tuning_iterations)
        else:  # bayesian
            best_result = tuner.bayesian_optimization(param_spaces, n_iterations=self.config.n_tuning_iterations)
        
        # Update best params
        self.best_params.update(best_result.params)
        
        return {
            'best_params': best_result.params,
            'best_score': best_result.combined_score,
            'em': best_result.em_score,
            'ndcg': best_result.ndcg_score
        }
    
    def _optimize_generation(self) -> Dict[str, Any]:
        """Optimize generation parameters."""
        logger.info("Optimizing generation parameters...")
        
        param_spaces = create_generation_param_spaces()
        
        def eval_fn(params, data):
            # Merge with best retrieval params
            full_params = {**self.best_params, **params}
            return self._evaluate_params(full_params)
        
        tuner = HyperparameterTuner(
            evaluation_fn=eval_fn,
            validation_data=self.validation_data,
            output_dir=str(self.output_dir / "generation_tuning")
        )
        
        if self.config.tuning_method == "grid":
            best_result = tuner.grid_search(param_spaces, max_combinations=30)
        elif self.config.tuning_method == "random":
            best_result = tuner.random_search(param_spaces, n_iterations=self.config.n_tuning_iterations)
        else:  # bayesian
            best_result = tuner.bayesian_optimization(param_spaces, n_iterations=self.config.n_tuning_iterations)
        
        # Update best params
        self.best_params.update(best_result.params)
        
        return {
            'best_params': best_result.params,
            'best_score': best_result.combined_score,
            'em': best_result.em_score,
            'ndcg': best_result.ndcg_score
        }
    
    def _optimize_joint(self) -> Dict[str, Any]:
        """Optimize all parameters jointly."""
        logger.info("Performing joint optimization...")
        
        # Combine parameter spaces
        param_spaces = {
            **create_retrieval_param_spaces(),
            **create_generation_param_spaces()
        }
        
        # Start from best known params
        initial_params = self.best_params.copy()
        
        def eval_fn(params, data):
            full_params = {**initial_params, **params}
            return self._evaluate_params(full_params)
        
        tuner = HyperparameterTuner(
            evaluation_fn=eval_fn,
            validation_data=self.validation_data,
            output_dir=str(self.output_dir / "joint_tuning")
        )
        
        # Use Bayesian optimization for joint tuning
        best_result = tuner.bayesian_optimization(
            param_spaces,
            n_iterations=min(self.config.n_tuning_iterations, 15),
            n_initial=5
        )
        
        # Update best params
        self.best_params.update(best_result.params)
        
        return {
            'best_params': best_result.params,
            'best_score': best_result.combined_score,
            'em': best_result.em_score,
            'ndcg': best_result.ndcg_score
        }
    
    def _run_ablation_study(self) -> Dict[str, Any]:
        """Run ablation study on system components."""
        logger.info("Running ablation study...")
        
        def baseline_fn(params):
            return self._evaluate_params(self.best_params)
        
        framework = AblationStudyFramework(
            baseline_fn=baseline_fn,
            validation_data=self.validation_data,
            output_dir=str(self.output_dir / "ablation_study")
        )
        
        # Run baseline
        baseline_result = framework.run_baseline()
        
        # Test different retrieval methods
        retrieval_variants = {
            'bm25_only': {'methods': ['bm25'], 'weights': [1.0]},
            'dense_only': {'methods': ['e5-base'], 'weights': [1.0]},
            'hybrid_rrf': {'methods': ['bm25', 'e5-base'], 'weights': [0.5, 0.5], 'fusion_method': 'rrf'},
            'hybrid_combsum': {'methods': ['bm25', 'e5-base'], 'weights': [0.5, 0.5], 'fusion_method': 'combsum'}
        }
        
        def eval_variant(config, data):
            params = {**self.best_params, **config}
            return self._evaluate_params(params)
        
        framework.compare_variants(retrieval_variants, eval_variant)
        
        # Analyze importance
        importance = framework.analyze_component_importance()
        
        # Generate report
        report_path = framework.generate_report()
        
        return {
            'baseline_score': baseline_result.combined_score,
            'component_importance': importance,
            'report_path': report_path
        }
    
    def _evaluate_best_config(self) -> Dict[str, float]:
        """Evaluate final best configuration."""
        logger.info("Evaluating best configuration...")
        logger.info(f"Best parameters: {self.best_params}")
        
        em, ndcg, combined = self._evaluate_params(self.best_params)
        
        logger.info(f"Final Results: EM={em:.4f}, nDCG={ndcg:.4f}, Combined={combined:.4f}")
        
        return {
            'em': em,
            'ndcg': ndcg,
            'combined': combined,
            'params': self.best_params
        }
    
    def _evaluate_params(self, params: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Evaluate system with given parameters.
        
        Args:
            params: System parameters
            
        Returns:
            Tuple of (em, ndcg, combined)
        """
        try:
            # Create RAG system with params
            rag_system = self.rag_system_factory(params)
            
            # Generate predictions
            predictions = []
            for item in self.validation_data:
                try:
                    result = rag_system.process_query(item)
                    predictions.append({
                        'id': result.id,
                        'answer': result.answer,
                        'retrieved_docs': [[doc[0], doc[1]] for doc in result.retrieved_docs]
                    })
                except Exception as e:
                    logger.error(f"Error processing query {item.get('id', 'unknown')}: {e}")
                    # Add empty prediction
                    predictions.append({
                        'id': item.get('id', 'unknown'),
                        'answer': '',
                        'retrieved_docs': []
                    })
            
            # Evaluate
            results = self.evaluator.evaluate(predictions, self.validation_data)
            
            em = results['em']
            ndcg = results['ndcg@10']
            combined = results['combined']
            
            # Track history
            self.optimization_history.append({
                'params': params,
                'em': em,
                'ndcg': ndcg,
                'combined': combined
            })
            
            return em, ndcg, combined
            
        except Exception as e:
            logger.error(f"Error evaluating params: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.0, 0.0, 0.0
    
    def _save_optimization_results(self, results: Dict[str, Any]) -> None:
        """Save optimization results to file."""
        output_file = self.output_dir / "optimization_results.json"
        
        # Convert to serializable format
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Optimization results saved to {output_file}")
        
        # Save summary
        summary_file = self.output_dir / "optimization_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("RAG SYSTEM OPTIMIZATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("BASELINE RESULTS\n")
            f.write("-" * 80 + "\n")
            baseline = results['baseline']
            f.write(f"EM: {baseline['em']:.4f}\n")
            f.write(f"nDCG: {baseline['ndcg']:.4f}\n")
            f.write(f"Combined: {baseline['combined']:.4f}\n\n")
            
            f.write("FINAL OPTIMIZED RESULTS\n")
            f.write("-" * 80 + "\n")
            final = results['final_results']
            f.write(f"EM: {final['em']:.4f}\n")
            f.write(f"nDCG: {final['ndcg']:.4f}\n")
            f.write(f"Combined: {final['combined']:.4f}\n\n")
            
            improvement = final['combined'] - baseline['combined']
            f.write(f"IMPROVEMENT: {improvement:+.4f} ({improvement/baseline['combined']*100:+.2f}%)\n\n")
            
            f.write("BEST PARAMETERS\n")
            f.write("-" * 80 + "\n")
            for param, value in results['best_params'].items():
                f.write(f"{param}: {value}\n")
            
            f.write(f"\nTotal optimization time: {results['elapsed_time']/60:.2f} minutes\n")
        
        logger.info(f"Optimization summary saved to {summary_file}")


def create_quick_optimization_config() -> OptimizationConfig:
    """Create config for quick optimization (testing)."""
    return OptimizationConfig(
        enable_retrieval_tuning=True,
        enable_generation_tuning=True,
        tuning_method="random",
        n_tuning_iterations=10,
        enable_ablation_study=False,
        validation_sample_size=100
    )


def create_comprehensive_optimization_config() -> OptimizationConfig:
    """Create config for comprehensive optimization (production)."""
    return OptimizationConfig(
        enable_retrieval_tuning=True,
        enable_generation_tuning=True,
        tuning_method="bayesian",
        n_tuning_iterations=30,
        enable_ablation_study=True,
        validation_sample_size=None  # Use all data
    )
