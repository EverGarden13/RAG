"""
Hyperparameter tuning for retrieval and generation components.
Implements grid search and random search strategies.
"""

import json
import itertools
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class HyperparameterSpace:
    """Defines the search space for hyperparameters."""
    name: str
    values: List[Any]
    param_type: str = "categorical"  # categorical, continuous, integer
    
    def sample(self, n: int = 1) -> List[Any]:
        """Sample n values from the space."""
        if self.param_type == "categorical":
            return list(np.random.choice(self.values, size=n, replace=True))
        elif self.param_type == "continuous":
            return list(np.random.uniform(self.values[0], self.values[1], size=n))
        elif self.param_type == "integer":
            return list(np.random.randint(self.values[0], self.values[1] + 1, size=n))
        return self.values[:n]


@dataclass
class TuningResult:
    """Result from hyperparameter tuning."""
    params: Dict[str, Any]
    em_score: float
    ndcg_score: float
    combined_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare by combined score."""
        return self.combined_score < other.combined_score


class HyperparameterTuner:
    """Hyperparameter tuner for RAG system components."""
    
    def __init__(self, evaluation_fn: Callable,
                 validation_data: List[Dict[str, Any]],
                 output_dir: str = "./optimization_results"):
        """
        Initialize hyperparameter tuner.
        
        Args:
            evaluation_fn: Function that takes params and returns (em, ndcg, combined)
            validation_data: Validation dataset for tuning
            output_dir: Directory to save results
        """
        self.evaluation_fn = evaluation_fn
        self.validation_data = validation_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[TuningResult] = []
        
        logger.info(f"Hyperparameter tuner initialized with {len(validation_data)} validation samples")
    
    def grid_search(self, param_spaces: Dict[str, HyperparameterSpace],
                   max_combinations: Optional[int] = None) -> TuningResult:
        """
        Perform grid search over hyperparameter space.
        
        Args:
            param_spaces: Dictionary of parameter spaces
            max_combinations: Maximum number of combinations to try
            
        Returns:
            Best tuning result
        """
        logger.info("Starting grid search...")
        logger.info(f"Parameter spaces: {list(param_spaces.keys())}")
        
        # Generate all combinations
        param_names = list(param_spaces.keys())
        param_values = [space.values for space in param_spaces.values()]
        
        all_combinations = list(itertools.product(*param_values))
        
        if max_combinations and len(all_combinations) > max_combinations:
            logger.info(f"Limiting to {max_combinations} random combinations out of {len(all_combinations)}")
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            all_combinations = [all_combinations[i] for i in indices]
        
        logger.info(f"Testing {len(all_combinations)} parameter combinations")
        
        # Evaluate each combination
        for i, combination in enumerate(all_combinations, 1):
            params = dict(zip(param_names, combination))
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Combination {i}/{len(all_combinations)}")
            logger.info(f"Parameters: {params}")
            logger.info(f"{'='*60}")
            
            try:
                em, ndcg, combined = self.evaluation_fn(params, self.validation_data)
                
                result = TuningResult(
                    params=params,
                    em_score=em,
                    ndcg_score=ndcg,
                    combined_score=combined,
                    metadata={'iteration': i, 'method': 'grid_search'}
                )
                
                self.results.append(result)
                
                logger.info(f"Results: EM={em:.4f}, nDCG={ndcg:.4f}, Combined={combined:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating combination {i}: {e}")
                continue
        
        # Find best result
        best_result = max(self.results, key=lambda x: x.combined_score)
        
        logger.info(f"\n{'='*60}")
        logger.info("GRID SEARCH COMPLETE")
        logger.info(f"Best parameters: {best_result.params}")
        logger.info(f"Best combined score: {best_result.combined_score:.4f}")
        logger.info(f"{'='*60}\n")
        
        # Save results
        self._save_results("grid_search")
        
        return best_result
    
    def random_search(self, param_spaces: Dict[str, HyperparameterSpace],
                     n_iterations: int = 20) -> TuningResult:
        """
        Perform random search over hyperparameter space.
        
        Args:
            param_spaces: Dictionary of parameter spaces
            n_iterations: Number of random samples to try
            
        Returns:
            Best tuning result
        """
        logger.info("Starting random search...")
        logger.info(f"Parameter spaces: {list(param_spaces.keys())}")
        logger.info(f"Number of iterations: {n_iterations}")
        
        for i in range(n_iterations):
            # Sample random parameters
            params = {}
            for name, space in param_spaces.items():
                params[name] = space.sample(1)[0]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {i+1}/{n_iterations}")
            logger.info(f"Parameters: {params}")
            logger.info(f"{'='*60}")
            
            try:
                em, ndcg, combined = self.evaluation_fn(params, self.validation_data)
                
                result = TuningResult(
                    params=params,
                    em_score=em,
                    ndcg_score=ndcg,
                    combined_score=combined,
                    metadata={'iteration': i+1, 'method': 'random_search'}
                )
                
                self.results.append(result)
                
                logger.info(f"Results: EM={em:.4f}, nDCG={ndcg:.4f}, Combined={combined:.4f}")
                
            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {e}")
                continue
        
        # Find best result
        best_result = max(self.results, key=lambda x: x.combined_score)
        
        logger.info(f"\n{'='*60}")
        logger.info("RANDOM SEARCH COMPLETE")
        logger.info(f"Best parameters: {best_result.params}")
        logger.info(f"Best combined score: {best_result.combined_score:.4f}")
        logger.info(f"{'='*60}\n")
        
        # Save results
        self._save_results("random_search")
        
        return best_result
    
    def bayesian_optimization(self, param_spaces: Dict[str, HyperparameterSpace],
                            n_iterations: int = 20,
                            n_initial: int = 5) -> TuningResult:
        """
        Perform Bayesian optimization (simplified version).
        Uses best results to guide search.
        
        Args:
            param_spaces: Dictionary of parameter spaces
            n_iterations: Total number of iterations
            n_initial: Number of random initial samples
            
        Returns:
            Best tuning result
        """
        logger.info("Starting Bayesian optimization...")
        logger.info(f"Initial random samples: {n_initial}")
        logger.info(f"Total iterations: {n_iterations}")
        
        # Phase 1: Random initialization
        logger.info("\nPhase 1: Random initialization")
        for i in range(n_initial):
            params = {name: space.sample(1)[0] for name, space in param_spaces.items()}
            
            logger.info(f"Initial sample {i+1}/{n_initial}: {params}")
            
            try:
                em, ndcg, combined = self.evaluation_fn(params, self.validation_data)
                result = TuningResult(
                    params=params,
                    em_score=em,
                    ndcg_score=ndcg,
                    combined_score=combined,
                    metadata={'iteration': i+1, 'phase': 'initialization'}
                )
                self.results.append(result)
                logger.info(f"Score: {combined:.4f}")
            except Exception as e:
                logger.error(f"Error in initial sample {i+1}: {e}")
        
        # Phase 2: Guided search
        logger.info("\nPhase 2: Guided search")
        for i in range(n_initial, n_iterations):
            # Get top 3 results
            top_results = sorted(self.results, key=lambda x: x.combined_score, reverse=True)[:3]
            
            # Sample near best parameters with some exploration
            base_params = top_results[0].params
            params = {}
            
            for name, space in param_spaces.items():
                if np.random.random() < 0.7:  # 70% exploitation
                    # Use value from top results
                    params[name] = np.random.choice([r.params[name] for r in top_results])
                else:  # 30% exploration
                    params[name] = space.sample(1)[0]
            
            logger.info(f"Iteration {i+1}/{n_iterations}: {params}")
            
            try:
                em, ndcg, combined = self.evaluation_fn(params, self.validation_data)
                result = TuningResult(
                    params=params,
                    em_score=em,
                    ndcg_score=ndcg,
                    combined_score=combined,
                    metadata={'iteration': i+1, 'phase': 'guided'}
                )
                self.results.append(result)
                logger.info(f"Score: {combined:.4f}")
            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {e}")
        
        best_result = max(self.results, key=lambda x: x.combined_score)
        
        logger.info(f"\n{'='*60}")
        logger.info("BAYESIAN OPTIMIZATION COMPLETE")
        logger.info(f"Best parameters: {best_result.params}")
        logger.info(f"Best combined score: {best_result.combined_score:.4f}")
        logger.info(f"{'='*60}\n")
        
        self._save_results("bayesian_optimization")
        
        return best_result
    
    def get_top_k_results(self, k: int = 5) -> List[TuningResult]:
        """Get top k results by combined score."""
        return sorted(self.results, key=lambda x: x.combined_score, reverse=True)[:k]
    
    def _save_results(self, method_name: str) -> None:
        """Save tuning results to file."""
        output_file = self.output_dir / f"{method_name}_results.json"
        
        results_data = []
        for result in self.results:
            results_data.append({
                'params': result.params,
                'em_score': result.em_score,
                'ndcg_score': result.ndcg_score,
                'combined_score': result.combined_score,
                'metadata': result.metadata
            })
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Also save summary
        summary_file = self.output_dir / f"{method_name}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Hyperparameter Tuning Results - {method_name}\n")
            f.write("=" * 60 + "\n\n")
            
            top_results = self.get_top_k_results(5)
            for i, result in enumerate(top_results, 1):
                f.write(f"Rank {i}:\n")
                f.write(f"  Parameters: {result.params}\n")
                f.write(f"  EM: {result.em_score:.4f}\n")
                f.write(f"  nDCG: {result.ndcg_score:.4f}\n")
                f.write(f"  Combined: {result.combined_score:.4f}\n\n")
        
        logger.info(f"Summary saved to {summary_file}")


def create_retrieval_param_spaces() -> Dict[str, HyperparameterSpace]:
    """Create parameter spaces for retrieval tuning."""
    return {
        'k_retrieve': HyperparameterSpace(
            name='k_retrieve',
            values=[50, 100, 150, 200],
            param_type='categorical'
        ),
        'k_final': HyperparameterSpace(
            name='k_final',
            values=[10],  # Fixed by requirements
            param_type='categorical'
        ),
        'fusion_method': HyperparameterSpace(
            name='fusion_method',
            values=['rrf', 'combsum', 'weighted'],
            param_type='categorical'
        ),
        'bm25_weight': HyperparameterSpace(
            name='bm25_weight',
            values=[0.2, 0.3, 0.4, 0.5, 0.6],
            param_type='categorical'
        ),
        'dense_weight': HyperparameterSpace(
            name='dense_weight',
            values=[0.4, 0.5, 0.6, 0.7, 0.8],
            param_type='categorical'
        )
    }


def create_generation_param_spaces() -> Dict[str, HyperparameterSpace]:
    """Create parameter spaces for generation tuning."""
    return {
        'temperature': HyperparameterSpace(
            name='temperature',
            values=[0.0, 0.1, 0.2, 0.3],
            param_type='categorical'
        ),
        'max_tokens': HyperparameterSpace(
            name='max_tokens',
            values=[256, 384, 512, 768],
            param_type='categorical'
        ),
        'top_p': HyperparameterSpace(
            name='top_p',
            values=[0.8, 0.9, 0.95, 1.0],
            param_type='categorical'
        ),
        'max_context_length': HyperparameterSpace(
            name='max_context_length',
            values=[1500, 2000, 2500, 3000],
            param_type='categorical'
        )
    }
